#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GREENTRACK TOOLS
A toolbox to analyze multiple-pixel biomass indicators, generate interpolated annual 
indicator curves and related statistics.


Created on Mon Nov 13 16:38:50 2023

@author: Fabio Oriani, Agroscope, fabio.oriani@agroscope.admin.ch

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
from scipy.optimize import curve_fit
from scipy.interpolate import interp2d
import geopandas as gpd
import os
import re
import rasterio
import shapefile
#from osgeo.gdal import GetDriverByName, GetDataTypeByName
from osgeo import osr
#import geopandas
import json
import requests
from rasterio.warp import reproject, calculate_default_transform
from rasterio.io import MemoryFile
from rasterio.merge import merge
from rasterio.enums import Resampling
from rasterio.transform import from_origin
#from shapefile import Reader
#import matplotlib.path as mplp
#import time
import concurrent.futures
import threading
import zarr
import pandas as pd
from tqdm import tqdm
import multiprocessing
from functools import partial
from multiprocessing import shared_memory


print(
    """
    Greentrack  Copyright (C) 2024 Fabio Oriani, Agroscope, Swiss Confederation
    This program comes with ABSOLUTELY NO WARRANTY under GNU General public 
    Licence v.3 or later.
    
    Please cite the related publication:
        F. Oriani, H. Aasen, M. K. Schneider
        Different growth response of mountain rangeland habitats to annual 
        weather fluctuations, submitted to Alpine Botany.
    """
    )

def purge(target_dir, target_pattern): 
    """
    PURGE remove all pattern-matching files

    Parameters
    ----------
    target_dir : str
        directory where the function search RECURSIVELY for files 
        matching target_pattern
    
    target_pattern : str
        string searched in filenames, all filenames containing it will be 
        removed

    Returns
    -------
    None.

    """
    
    for f in os.listdir(target_dir):
        if re.search(target_pattern, f):
            os.remove(os.path.join(target_dir, f))



def make_grid_vec(shp_path,res,target_crs=None):
    """
    MAKE_GRID_VEC
    Generate x, y grid vectors for a given polygon shapefile and wanted resolution.
    The grid is contained the polygon bounding box and defined in the shapefile crs.

    Parameters
    ----------
    shp_fname : str
        path to the shapefile (.shp or .gpkg) of the input polygon
    res : int
        grid resolution compatible with the shapefile crs
    target_crs: int
        epsg code for the target crs for the output coordinate vectors tx and ty.
        Deafult is None. If None the original crs of shp_fname is used. 

    Returns
    -------
    tx : 1d array
        x coordinate vector of the grid 
    ty : 1d array
        y coordiante vector of the grid

    """
    
    
    if target_crs!=None:
        shp = gpd.read_file(shp_path).to_crs("EPSG:" + str(target_crs))
    else:
        shp = gpd.read_file(shp_path)
    
    # define pixel-center coordinate contained in the bounding box (+-res/2)
    lef = np.min(shp.bounds.minx) + res/2
    rig = np.max(shp.bounds.maxx) - res/2  
    tx = np.arange(lef, rig, res)
    
    bot = np.min(shp.bounds.miny)
    top = np.max(shp.bounds.maxy) 
    
    if top < bot:
        res = -res
        
    bot = bot + res/2
    top = top - res/2 
    ty = np.arange(bot, top, res)
    
    return tx, ty

def scene_to_array(sc,tx,ty,mask=None,no_data=0):
   
    """
    SCENE_TO_ARRAY
    
    Generate an numpy array (image stack) from a given Eodal SceneCollection object.
    The scenes are resampled on a costant coordinate grid allowing pixel analysis.
    Missing data location are marked as nans.

    Parameters
    ----------
    sc : Eodal SceneCollection
        The given Scene Collection generated from Eodal
    tx : Float Vector
        x coordinate vector for the resample grid.
        ex. tx = numpy.arange(100,150,10) # x coords from 100 to 150 with 10 resolution
    ty : Float Vector
        y coordinate vector for the resample grid.
    
    mask : np.array of size [ty,tx]
        if given, pixels valued False or 0 are set to no_data in the output grid.
    
    no_data : int or nan
        value for missing data in the grid

    Returns
    -------
    im : float 4D numpy array.
        Array containing the stack of all scenes.
        4 dimensions: [x, y, bands, scenes]

    """
    
    ts = sc.timestamps # time stamps for each image
    bands = sc[sc.timestamps[0]].band_names # bands
    im_size = [len(ty),len(tx)] # image size
    im = np.empty(np.hstack([im_size,len(bands),len(ts)])) # preallocate matrix

    for i, scene_iterator in enumerate(sc):
        
        # REGRID SCENE TO BBOX AND TARGET RESOLUTION
        scene = scene_iterator[1]        
        for idx, band_iterator in enumerate(scene):
            
            # extract data with masked ones = 0
            band = band_iterator[1]
            Gv = np.copy(band.values.data)             
            Gv[band.values.mask==1]=0
            
            #original grid coordinates
            ny,nx = np.shape(Gv)
            vx = band.coordinates['x']
            vy = band.coordinates['y']
            
            # flip vy vector to increasing values
            if vy[0] > vy[-1]:
                vy = np.flipud(vy)
           
            # create interpolator
            Gv_no_nans = Gv.copy()
            Gv_no_nans[np.isnan(Gv)] = 0
            f = interp2d(vx,vy,Gv_no_nans,kind='linear',fill_value=no_data)
            
            # interpolate band on the target grid
            Tv = f(tx,ty) #Tv = np.flipud(f(tx,ty))
            
             # assign interpolated band [i = scene , b = band]
            im[:,:,idx,i] = Tv.copy()
            del Tv
            
    # apply mask if given
    if mask is not None:
        im[np.logical_not(mask),:,:] = no_data
    
    return im

def imrisc(im,qmin=1,qmax=99): 
    
    """
    IMRISC
    Percentile-based 0-1 rescale for multiband images. 
    Useful for satellite image visualization.
    

    Parameters
    ----------
    im : Float Array
        The image to rescale, can be multiband on the 3rd dimension
    qmin : Float Scalar
        Percentile to set the bottom of the value range e.g. 0.01
    qmax : Float Scalar
        Percentile to set the top of the value range e.g. 0.99

    Returns
    -------Quantile
    im_out : Float Array
        Rescaled image
        
    EXAMPLE
    import matplotlib.pyplot as plt
    
    # with im being an [x,y,[r,g,b]] image
    
    plt.figure()
    plt.imshow(imrisc(im))
    

    """

    if len(np.shape(im))==2:
        band=im.copy()
        band2=band[~np.isnan(band)]
        vmin=np.percentile(band2,qmin)
        vmax=np.percentile(band2,qmax)
        band[band<vmin]=vmin
        band[band>vmax]=vmax
        band=(band-vmin)/(vmax-vmin)
        im_out=band
    else:
        im_out=im.copy()
        for i in range(np.shape(im)[2]):
            band=im[:,:,i].copy()
            band2=band[~np.isnan(band)]
            vmin=np.percentile(band2,qmin)
            vmax=np.percentile(band2,qmax)
            band[band<vmin]=vmin
            band[band>vmax]=vmax
            band=(band-vmin)/(vmax-vmin)
            im_out[:,:,i]=band
            
    return im_out

def evi(blue,red,nir):
    
    """
    EVI
    
    Calculates the Enhanced Vegetation Index (EVI). from scipy.interpolate import interp2d

    Huete et al. (2002) https://doi.org/10.1016/S0034-4257(02)00096-2

    Parameters
    ----------
    blue, red, nir : n-d numericals
        blue, red, near-infrared band images or values

    Returns
    -------
    im_out : n-d numerical
        EVI image or value
    """
    
    numerator = nir - red
    denominator = nir + 6 * red - 7.5 * blue + 1    
    evi = 2.5 * (numerator / denominator)
    
    # threshold values outside [-1,1] (artificial surfaces)
    evi[evi > 1.0] = 1.0
    evi[evi < -1.0] = -1.0
    
    return evi

def ndvi(red,nir):
    
    """
    NDVI
    
    Calculates the Normalized Difference Vegetation Index (NDVI).
    Rouse et al. 1974 https://ntrs.nasa.gov/citations/19740022614
    
    Parameters
    ----------
    red, nir : n-d numericals
        blue, red, near-infrared band images or values

    Returns
    -------
    im_out : n-d numerical
        NDVI image or value
    """
    
    ndvi = (nir - red) / (nir + red) 
    
    return ndvi

def annual_interp(time,data,time_res='doy',lb=0,rb=0,sttt=0,entt=365.25):
    
    """
    ANNUAL CURVE INTERPOLATION
    Generates the interpoalted curves for annual data. If more values are present
    with the same dates (ex. pixels from the same image), the median is taken.

    Parameters
    ----------
    time : vector
        any date of time vector the data
    data : vector
        data vector
    time_res : string, optional
        wanted time resolution among "doy", "week", and "month". The default is 'doy'.
    lb : scalar, optional
        left boundary of the interpolated curve at start time. The default is 0, 
        if lb = 'data' it is set to data[0]
    rb : scalar, optional
        right boundary of the interpolated curve at start time. The default is 0.
        if rb = 'data' it is set to data[-1]
    sttt : scalar, optional
        starting time for the interpolation. The default is 0.
    entt : scalar, optional
        ending time for the interpolation. The default is 365.25.

    Returns
    -------
    xv : vector
        interpolation time
    yv : vector
        interpolated values
    t_list : vector
        time vector of the interpolated data
    data : vector
        interpolated data (median)

    """
    
    t_list = np.unique(time) # data time line 
    
    # if more data are present with the same time, take the median
    qm = np.array([],dtype=float)
    for t in t_list:
        qm = np.hstack((qm,np.nanquantile(data[time==t],0.5)))
    
    # add start/ending 0 boundary conditions
    stbt = 0 # zero time in weeks
    if time_res == 'doy':
        enbt = 366 # end time in doys
        #sttt = 0 # start target time (beginning Mar) 
        #entt = 300 # end target time (end Oct)
        dt = 1 # daily granularity for interp
    elif time_res == 'week':
        enbt = 52.17857 # end time in weeks
        sttt = 9 # start target time (beginning Mar) 
        entt = 44 # end target time (end Oct)
        dt = 1/7 # daily granularity for interp
    elif time_res == 'month':
        enbt = 12
        sttt = 3 # start target time (beginning Mar) 
        entt = 10 # end target time (end Oct)
        dt = 1/30 # daily granularity for interp
    
    t_list_tmp = np.hstack([stbt,t_list,enbt])
    
    if lb == 'data':
        lb = qm[0]
    if rb == 'data':
        rb = qm[-1]
    data_tmp = np.hstack([lb,qm,rb]) 
    
    t_list_tmp,ind = np.unique(t_list_tmp, return_index=True)
    data_tmp = data_tmp[ind]
    
    # piecewise cubic hermitian interpolator
    ph = PchipInterpolator(t_list_tmp,data_tmp,extrapolate=False)
    
    # interpolation on target dates
    xv = np.arange(sttt,entt+dt,dt) # daily granularity
    yv = ph(xv)
    
    return xv,yv,t_list,qm # target time weeks, target interpolated data, data time, data
    
def annual_plot(dates,data,dcol,dlabel,time_res = 'doy',envelope=True,lb=0,rb=0,f_range=[-1,1]):
    """
    
    ANNUAL CURVE PLOT
    Generates the interpoalted curves for annual data. If more values are present
    with the same dates (ex. pixels from the same image), the median is taken.
    If envelope = True, the 0.25-0.75 quantile envelope is also plotted.
    The interpolated values are also given as output vectors

    Parameters
    ----------
    dates : vector
        dates or time vector
    data : vector
        ndvi or similar values to plot
    dcol : string
        color string for the plotted curve
    dlabel : string
        legend label for the curve
    time_res : string
        time resolution among 'month','week', or 'doy' 
    envelope : boolean, optional
        if = True the 0.25-0.75 quantile envelope is computed and plotted. 
        The default is True.
    lb : scalar, optional
        left boundary of the interpolated curve at start time. 
        The default is 0. If lb = 'data' it is set to data[0]
    rb : scalar, optional
        right boundary of the interpolated curve at start time. 
        The default is 0. If rb = 'data' it is set to data[-1]
    f_range: 2-element vector
        range outside which the ndvi median value is considered invalid. 
        Default is [-1,1], total NDVI range.

    Returns
    -------
    d_listm : vector
    time vector for the interpolated values
    q2i : vector
    interpolated 0.25 quantile values
    qmi : vector
    interpolated median values
    q1i : vector
    interpolated median values

    """
    d_list = np.unique(dates)
    plt.grid(axis='y',linestyle='--')
    
    qm = np.array([],dtype=float)
    for d in d_list:
        qm = np.hstack((qm,np.nanquantile(data[dates==d],0.5)))
    
    # filter out data with median outside given range
    fil = np.logical_and(qm > f_range[0],qm < f_range[1])
    d_list = d_list[fil]
    qm = qm[fil]
    
    # envelope interpolation
    if envelope==True:
        q1 = np.array([],dtype=float)
        q2 = np.array([],dtype=float)
        for d in d_list:
            q1 = np.hstack((q1,np.nanquantile(data[dates==d],0.75)))
            q2 = np.hstack((q2,np.nanquantile(data[dates==d],0.25)))
        d_list1,q1i,*tmp = annual_interp(d_list,q1,time_res=time_res,lb=lb,rb=rb)
        d_list2,q2i,*tmp = annual_interp(d_list,q2,time_res=time_res,lb=lb,rb=rb)
        q2i_f = np.flip(q2i)
        qi = np.hstack((q1i,q2i_f))
        d = np.hstack((d_list1,np.flip(d_list1)))
        d = d[~np.isnan(qi)]
        qi = qi[~np.isnan(qi)]
        plt.fill(d,qi,alpha=0.5,c=dcol)

    # median interpolation
    d_listm,qmi,*tmp = annual_interp(d_list,qm,time_res=time_res,lb=lb,rb=rb)
    plt.plot(d_listm,qmi,linestyle = '--', c=dcol,markersize=15,label=dlabel)
    plt.scatter(d_list,qm,c=dcol)
    
    if envelope == True:     
        return d_listm, q2i, qmi, q1i # time, q.25, q.5, q.75
    else:
        return d_listm, qmi # time, q.5


def annual_px_stats(dates,data,plot=True):
    
    if np.all(np.isnan(data)): # if all data is nan
        
        return [np.nan]*8 # nan for all output stats    
    
    # INDICATOR CURVE: data median time, median data, interp median time, interp median
    tm,qm,tmi,qmi = annual_px_plot(dates,data,'tab:blue', 'EVI',time_res='doy',f_range=[-0.1,1],plot=plot)
    
    # SOG
    sogm = quick_sog(tmi,qmi,th=0.05,pth=10)
    
    # EOS
    eosm = quick_eos(tmi,qmi,th=0.05,pth=10)
    
    if np.isnan(sogm) or np.isnan(eosm): # if sason cannot be detected
        
        return [np.nan]*8 # nan for all output stats
    
    # LOS
    if eosm-sogm > 0: # if a season is defined
        
        losm = eosm-sogm
    
    else:
        
        losm = np.nan
    
    # OTHERS STATS defined for the SOG-EOS interpolated curve portion
    
    # area under the indicator curve from SOG to 
    aucm = quick_auc(tmi,qmi,sttt=sogm,entt=eosm)
    
    # MSG: mean seasonal growth
    if aucm > 0 and losm > 0:
        
        msgm = aucm/losm
    
    else:
        
        msgm = np.nan
        
    # 90th quantile of the curve 
    p = 0.9
    q90 = quick_q(tmi, qmi, p, sttt=sogm, entt=eosm)
    
    # standard deviation of the green portion
    stdm = quick_std(tmi, qmi, sttt=sogm, entt=eosm)
    
    # indicator growth slope fittend on growing season only (213-th DOY, 1st of Aug)
    fp,C = curve_fit(gomp,
                      tmi[:213],
                      qmi[:213],
                      maxfev=100000,
                      bounds = ([0,0,0,0],[2,360,1,1]))
    
    sl = fp[2].copy()
    gom_time = np.arange(0,213)
    gom_data = gomp(gom_time,*fp)
    
    if plot==True:
        
        plt.plot(gom_time,gom_data,'--',c='tab:purple',label="Gompertz")
        
        plt.plot([eosm,eosm],[-0.1,0.1],'--',c='tab:brown')
        plt.text(eosm+1,-0.1,s='EOS',c='tab:brown',rotation = 'vertical')
        
        plt.plot([sogm,sogm],[-0.1,0.1],'--',c='tab:blue')
        plt.text(sogm+1,-0.1,s='SOG',c='tab:blue',rotation = 'vertical')
        
        plt.text(tmi[180],qmi[180]/2,s='AUC = ' + str(aucm)[:5],c='tab:blue')
        
        plt.plot([sogm,eosm],[q90,q90],'--',c='tab:orange')
        plt.text(sogm+10,q90+0.05,s='q90',c='tab:orange')
        
        # graph cosmetics
        plt.ylim([-0.15,1.1])
        plt.xlabel('Day of the year (DOY)')
        plt.ylabel('Spectral indicators [-1,1]')
        plt.legend(loc='upper right',prop={'size': 11})
        plt.grid(axis='y',linestyle='--',alpha=0.5)
        plt.grid(axis='x',linestyle='--',alpha=0.5)
        plt.tight_layout()
    
    return sogm, eosm, losm, aucm, msgm, sl, q90, stdm

def annual_px_plot(dates,data,dcol,dlabel,time_res = 'doy',lb=0,rb=0,f_range=[-1,1],plot=True):
    """

    SINGLE PIXEL ANNUAL CURVE PLOT
    Generates the interpolated single-pixel or single data series curve for annual data. 
    If more values are present with the same dates the median is taken.
    The interpolated values are also given as output vectors

    Parameters
    ----------
    dates : vector
        dates or time vector
    data : vector
        ndvi or similar values to plot
    dcol : string
        color string for the plotted curve
    dlabel : string
        legend label for the curve
    time_res : string
        time resolution among 'month','week', or 'doy' 
    lb : scalar, optional
        left boundary of the interpolated curve at start time. 
        The default is 0. If lb = 'data' it is set to data[0]
    rb : scalar, optional
        right boundary of the interpolated curve at start time. 
        The default is 0. If rb = 'data' it is set to data[-1]
    f_range: 2-element vector
        range outside which the ndvi median value is considered invalid. 
        Default is [-1,1], total NDVI range.
    plot: boolean
        wsith to generate the plot.

    Returns
    -------
    d_listm : vector
    time vector for the interpolated values
    qmi : vector
    interpolated median values
    
    """
    
    # generate data vectors
    d_list = np.unique(dates)
    qm = np.array([],dtype=float)
    
    for d in d_list:
        
        data_tmp = data[dates==d]
        
        if np.all(np.isnan(data_tmp)):
            
            qm = np.hstack((qm,np.nan))
                           
        else:
            
            qm = np.hstack((qm,np.nanquantile(data[dates==d],0.5)))
    
    # filter out data with median outside given range
    fil = np.logical_and(qm > f_range[0],qm < f_range[1])
    d_list = d_list[fil]
    qm = qm[fil]
    
    # median interpolation
    d_listm,qmi,*tmp = annual_interp(d_list,qm,time_res=time_res,lb=lb,rb=rb)
    
    if plot==True:
        plt.grid(axis='y',linestyle='--')
        plt.plot(d_listm,qmi,linestyle = '--', c=dcol,markersize=15,label=dlabel)
        plt.scatter(d_list,qm,c=dcol)
    
    return d_list, qm, d_listm, qmi # data time, median data, interp time, interp median

def quick_eos(tmi,qmi,th=0.1,pth=5,sttt=213,entt=365):
    """
    QUICK EOS (End Of Season)
    Computes the season end for given interpolated annual data and an NDVI 
    threshold considered the end of the greening season. 
    
    Parameters
    ----------
    tmi : vector
        doy time vector
    qmi : vector
        interpolated data vector
    th : scalar
        indicator threshold below which the browning time is reached. Default is 0.1
    pth : scalar
        number of time steps for which ndvi_th has to be passed in order to 
        define the brwning time. Default is 5 
    sttt : scalar, optional
        starting time for the interpolation. The default is 213 (half of season).
    entt : scalar, optional
        ending time for the interpolation. The default is 366 (end of year).

    Returns
    -------
    EOS for the given median annual curve

    """
    
    # extract curve portion
    st_ind = np.argwhere(tmi==sttt)[0][0]
    en_ind = np.argwhere(tmi==entt)[0][0]
    tmi_tmp = tmi[st_ind:en_ind]
    qmi_tmp = qmi[st_ind:en_ind]
    
    # compute eos
    gsw = False
    n = 0
    egm = np.nan
    for i in range(len(tmi_tmp)):
        
        if n==pth: 
        
            break
        
        elif qmi_tmp[i]<th and gsw==False:
        
            egm = tmi_tmp[i]
            gsw = True
            n = n+1
            
        elif qmi_tmp[i]<th and gsw==True:
            
            n = n+1
            
        elif  qmi_tmp[i]>=th:
            
            egm = entt
            #egm = np.nan
            gsw = False
            n = 0
 
    return egm # 0.5 quantile browning time

def quick_sog(tmi,qmi,th=0.1,pth=5,sttt=0,entt=213):
    """
    QUICK SOG (Start Of Greening)
    Computes the season end for given interpolated annual data and an NDVI 
    threshold considered the end of the greening season. 
    
    Parameters
    ----------
    tmi : vector
        doy time vector
    qmi : vector
        interpolated data vector
    th : scalar
        indicator threshold below which the browning time is reached. Default is 0.1
    pth : scalar
        number of time steps for which ndvi_th has to be passed in order to 
        define the brwning time. Default is 5 
    sttt : scalar, optional
        starting time for the interpolation. The default is 213 (half of season).
    entt : scalar, optional
        ending time for the interpolation. The default is 366 (end of year).

    Returns
    -------
    SOG for the given median annual curve

    """
    
    # extract curve portion
    st_ind = np.argwhere(tmi==sttt)[0][0]
    en_ind = np.argwhere(tmi==entt)[0][0]
    tmi_tmp = tmi[st_ind:en_ind]
    qmi_tmp = qmi[st_ind:en_ind]
    
    # compute sog    
    gsw = False
    n = 0
    egm = np.nan
    
    for i in range(len(tmi_tmp)):
    
        if n==pth:
            
            break
        
        elif qmi_tmp[i]>th and gsw==False:
            
            egm = tmi_tmp[i]
            gsw = True
            n = n+1
            
        elif qmi_tmp[i]>th and gsw==True:
            
            n = n+1
            
        elif  qmi_tmp[i]<=th:
            
            egm = np.nan
            gsw = False
            n = 0
        
    return egm # 0.5 quantile greening time

def quick_auc(tmi,qmi,sttt=0,entt=365):
    """
    AUC
    Computes the Area under the curve (AUC) for given interpolated annual curve. 
    
    Parameters
    ----------
    tmi : vector
        doy time vector
    qmi : vector
        interpolated data vector
    sttt : scalar, optional
        starting time for the interpolation. The default is 0.
    entt : scalar, optional
        ending time for the interpolation. The default is 365.25.

    Returns
    -------
    qsum : scalar
    AUC of the annual curve

    """
    
    # extract curve portion
    st_ind = np.argwhere(tmi==sttt)[0][0]
    en_ind = np.argwhere(tmi==entt)[0][0]
    qmi_tmp = qmi[st_ind:en_ind]
    
    # auc
    qmsum = np.cumsum(qmi_tmp)[-1]

    return qmsum # auc

def quick_q(tmi,qmi,p,sttt=0,entt=365):
    """
    QUICK QUANTILE
    Computes the p-th quantile for a given interpolated annual curve. 
    
    Parameters
    ----------
    tmi : vector
        doy time vector
    qmi : vector
        interpolated data vector
    p:
        value of the quantile, defined in [0,1].
        ex. q = 0.25 is the first quartile
    sttt : scalar, optional
        starting time for the interpolation. The default is 0.
    entt : scalar, optional
        ending time for the interpolation. The default is 365.25.

    Returns
    -------
    q : scalar
    p-th quantile of the annual curve

    """
    
    # extract curve portion
    st_ind = np.argwhere(tmi==sttt)[0][0]
    en_ind = np.argwhere(tmi==entt)[0][0]
    qmi_tmp = qmi[st_ind:en_ind]
    
    # quantile
    q = np.quantile(qmi_tmp,p)

    return q 

def quick_std(tmi,qmi,sttt=0,entt=365):
    """
    QUICK STANDARD DEVIATION
    Computes the standard deviation for a given interpolated annual curve. 
    
    Parameters
    ----------
    tmi : vector
        doy time vector
    qmi : vector
        interpolated data vector
    sttt : scalar, optional
        starting time for the interpolation. The default is 0.
    entt : scalar, optional
        ending time for the interpolation. The default is 365.25.

    Returns
    -------
    stdm : scalar
    stadard deviation of the selected annual curve portion

    """
    
    # extract curve portion
    st_ind = np.argwhere(tmi==sttt)[0][0]
    en_ind = np.argwhere(tmi==entt)[0][0]
    qmi_tmp = qmi[st_ind:en_ind]
    
    # quantile
    stdm = np.std(qmi_tmp)

    return stdm 

def auc(dates,data,time_res,envelope=True,sttt=0,entt=365.25):
    """
    AUC
    Computes the Area under the curve (AUC) for given annual data. 
    Data are interpolated as annual curve. If more values are present with
    the same dates (ex. pixels from the same image), the median is taken.
    If envelope = True, AUC is also computed for the 0.25-0.75 quantile 
    envelope curves.
    Parameters
    ----------
    time : vector
        any date of time vector the data
    data : vector
        data vector
    time_res : string, optional
        wanted time resolution among "doy", "week", and "month". The default is 'doy'.
    envelope : boolean, optional
        if = True AUC of the 0.25 and 0.75 quantile envelope curves are computed. 
        The default is True.
    sttt : scalar, optional
        starting time for the interpolation. The default is 0.
    entt : scalar, optional
        ending time for the interpolation. The default is 365.25.

    Returns
    -------
    q2sum : scalar
    AUC of the 0.25 quantile annual curve
    qmi : scalar
    AUC of the median annual curve
    q1i : scalar
    AUC of the 0.75 quantile annual curve

    """
    d_list = np.unique(dates)
    #plt.grid(axis='y',linestyle='--')
    
    if envelope==True:
        q1 = np.array([],dtype=float)
        q2 = np.array([],dtype=float)
        for d in d_list:
            q1 = np.hstack((q1,np.nanquantile(data[dates==d],0.75)))
            q2 = np.hstack((q2,np.nanquantile(data[dates==d],0.25)))
        d_list1,q1i,*tmp = annual_interp(d_list,q1,time_res=time_res,sttt=sttt,entt=entt)
        d_list2,q2i,*tmp = annual_interp(d_list,q2,time_res=time_res,sttt=sttt,entt=entt)
        q1sum = np.cumsum(q1i)[-1]
        q2sum = np.cumsum(q2i)[-1]
        
    qm = np.array([],dtype=float)
    for d in d_list:
        qm = np.hstack((qm,np.nanquantile(data[dates==d],0.5)))  
    d_listm,qmi,*tmp = annual_interp(d_list,qm,time_res=time_res,sttt=sttt,entt=entt)
    qmsum = np.cumsum(qmi)[-1]
    
    if envelope==True:
        return q2sum, qmsum, q1sum # q25, qm,q75
    else:
        return qmsum # qm


def sog(dates,data,time_res,envelope=True,ndvi_th=0.1,pth=5,sttt=0,entt=366):
    """
    SOG (Start Of Greening)
    Computes the SOG for given NDVI annual data and an NDVI 
    threshold considered the beginning of the greening season. 
    Data are interpolated as annual curve. If more values are present with
    the same dates (ex. pixels from the same image), the median is taken.
    If envelope = True, the greening start is computed also for the 0.25-0.75 
    quantile envelope curves.
    Parameters
    ----------
    dates : vector
        any date of time vector the data
    data : vector
        data vector
    time_res : string, optional
        wanted time resolution among "doy", "week", and "month". The default is 'doy'.
    envelope : boolean, optional
        if = True SOG of the 0.25 and 0.75 quantile envelope curves are computed. 
        The default is True.
    ndvi_th : scalar
        ndvi threshold below which the greening time is reached. Default is 0.1
    pth : scalar
        number of continuous time steps for which ndvi_th has to be passed in 
        order to define the greening time. Default is 5 
    sttt : scalar, optional
        starting time for the interpolation. The default is 0.
    entt : scalar, optional
        ending time for the interpolation. The default is 365.25.

    Returns
    -------
    q2sum : scalar
    SOG of the 0.25 quantile annual curve
    qmi : scalar
    AOG of the median annual curve
    q1i : scalar
    SOG of the 0.75 quantile annual curve

    """
    d_list = np.unique(dates)
    
    if envelope==True:
        q1 = np.array([],dtype=float)
        q2 = np.array([],dtype=float)
        for d in d_list:
            q1 = np.hstack((q1,np.nanquantile(data[dates==d],0.75)))
            q2 = np.hstack((q2,np.nanquantile(data[dates==d],0.25)))
        d_list1,q1i,*tmp = annual_interp(d_list,q1,time_res=time_res,sttt=sttt,entt=entt)
        d_list2,q2i,*tmp = annual_interp(d_list,q2,time_res=time_res,sttt=sttt,entt=entt)
        
        ndvi_th = 0.1
        gsw = False
        n = 0
        egq1 = np.nan
        for i in range(len(d_list1)):
            if n==5: 
                break
            elif q1i[i]>ndvi_th and gsw==False:
                egq1 = d_list1[i]
                gsw = True
                n = n+1
            elif q1i[i]>ndvi_th and gsw==True:
                n = n+1
            elif q1i[i]<=ndvi_th:
                egq1 = np.nan
                gsw = False
                n = 0
        
        gsw = False
        n = 0
        egq2 = np.nan
        for i in range(len(d_list2)):
            if n==5: 
                break
            elif q2i[i]>ndvi_th and gsw==False:
                egq2 = d_list1[i]
                gsw = True
                n = n+1
            elif q2i[i]>ndvi_th and gsw==True:
                n = n+1
            elif q2i[i]<=ndvi_th:
                egq2 = np.nan
                gsw = False
                n = 0
        
    qm = np.array([],dtype=float)
    for d in d_list:
        qm = np.hstack((qm,np.nanquantile(data[dates==d],0.5)))  
    d_listm,qmi,*tmp = annual_interp(d_list,qm,time_res=time_res,sttt=sttt,entt=entt)
    
    gsw = False
    n = 0
    egm = np.nan
    for i in range(len(d_listm)):
        if n==pth: 
            break
        elif qmi[i]>ndvi_th and gsw==False:
            egm = d_listm[i]
            gsw = True
            n = n+1
        elif qmi[i]>ndvi_th and gsw==True:
            n = n+1
        elif  qmi[i]<=ndvi_th:
            egm = np.nan
            gsw = False
            n = 0
        
    if envelope==True:
        return egq2,egm,egq1 # 0.25 0.5 0.7 greening time
    else:
        return egm # 0.5 quantile greening time

def greening_slope(dates,data,envelope=True,sttt=0,entt=213,time_res='doy',plot=False):
    """
    GREENING SLOPE
    Computes the greening slope for given NDVI annual data. 
    If more values are present with the same dates 
    (ex. pixels from the same image), the median is taken.
    If envelope = True, the greening slope is computed also for the 0.25-0.75 
    quantile envelope curves.
    
    Parameters
    ----------
    dates : vector
        any date of time vector the data
    data : vector
        data vector
    envelope : boolean, optional
        if = True the slope of the 0.25 and 0.75 quantile envelope curves are computed. 
        The default is True.

    Returns
    -------
    q2sum : scalar
    AUC of the 0.25 quantile annual curve
    qmi : scalar
    AUC of the median annual curve
    q1i : scalar
    AUC of the 0.75 quantile annual curve

    """
    d_list = np.unique(dates)
    d_list = d_list[np.logical_and(d_list>=sttt,d_list<=entt)]    
    
    if envelope==True:
        q1 = np.array([],dtype=float)
        q2 = np.array([],dtype=float)
        for d in d_list:
            q1 = np.hstack((q1,np.nanquantile(data[dates==d],0.75)))
            q2 = np.hstack((q2,np.nanquantile(data[dates==d],0.25)))
        d_list1,q1i,*tmp = annual_interp(d_list,q1,time_res=time_res,lb=0,rb='data',sttt=sttt,entt=entt)       
        d_list2,q2i,*tmp = annual_interp(d_list,q2,time_res=time_res,lb=0,rb='data',sttt=sttt,entt=entt)
        fp1,C = curve_fit(gomp,
                          d_list1,
                          q1i,
                          bounds = (
                              [0,50,0,0],
                              [2,200,1,1]
                              ))
        fp2,C = curve_fit(gomp,
                          d_list2,
                          q2i,
                          bounds = (
                              [0,50,0,0],
                              [2,200,1,1]
                              ))
    
    qm = np.array([],dtype=float)
    for d in d_list:
        qm = np.hstack((qm,np.nanquantile(data[dates==d],0.5)))
    d_listm,qmi,*tmp = annual_interp(d_list,qm,time_res=time_res,lb=0,rb='data',sttt=sttt,entt=entt)
    fpm,C = curve_fit(gomp,
                      d_listm,
                      qmi,
                      bounds = (
                              [0,50,0,0],
                              [2,200,1,1]
                              ))        
    
    # plot
    if plot == True:
        fx = np.arange(0,240)
        fy = gomp(fx,*fpm)
        plt.figure()
        plt.plot(d_listm,qmi,'o', markersize = 0.7)
        plt.plot(fx,fy, label = 'Gompertz on data')
    
    #output
    
    if envelope==True:
        return fp2[2],fpm[2],fp1[2] # 0.25 0.5 0.7 greening slope
    else:
        return fpm[2] # 0.5 quantile greening slipe

def greening_max(dates,data,envelope=True,sttt=0,entt=213,time_res='doy',plot=False):
    """
    GREENING MAX
    Computes the greening max for given NDVI annual data. 
    If more values are present with the same dates 
    (ex. pixels from the same image), the median is taken.
    If envelope = True, the greening max is computed also for the 0.25-0.75 
    quantile envelope curves.
    
    Parameters
    ----------
    dates : vector
        any date of time vector the data
    data : vector
        data vector
    envelope : boolean, optional
        if = True MAX of the 0.25 and 0.75 quantile envelope curves are computed. 
        The default is True.

    Returns
    -------
    q2sum : scalar
    AUC of the 0.25 quantile annual curve
    qmi : scalar
    AUC of the median annual curve
    q1i : scalar
    AUC of the 0.75 quantile annual curve

    """
    d_list = np.unique(dates)
    d_list = d_list[np.logical_and(d_list>=sttt,d_list<=entt)]    
    
    if envelope==True:
        q1 = np.array([],dtype=float)
        q2 = np.array([],dtype=float)
        for d in d_list:
            q1 = np.hstack((q1,np.nanquantile(data[dates==d],0.75)))
            q2 = np.hstack((q2,np.nanquantile(data[dates==d],0.25)))
        d_list1,q1i,*tmp = annual_interp(d_list,q1,time_res=time_res,lb=0,rb='data',sttt=sttt,entt=entt)       
        d_list2,q2i,*tmp = annual_interp(d_list,q2,time_res=time_res,lb=0,rb='data',sttt=sttt,entt=entt)
        fp1,C = curve_fit(gomp,
                          d_list1,
                          q1i,
                          bounds = (
                              [0,50,0,0],
                              [2,200,1,1]
                              ))
        fp2,C = curve_fit(gomp,
                          d_list2,
                          q2i,
                          bounds = (
                              [0,50,0,0],
                              [2,200,1,1]
                              ))
        
    qm = np.array([],dtype=float)
    for d in d_list:
        qm = np.hstack((qm,np.nanquantile(data[dates==d],0.5)))
    d_listm,qmi,*tmp = annual_interp(d_list,qm,time_res=time_res,lb=0,rb='data',sttt=sttt,entt=entt)
    fpm,C = curve_fit(gomp,
                      d_listm,
                      qmi,
                      bounds = (
                          [0,50,0,0],
                          [2,200,1,1]
                          ))
    # plot
    if plot == True:
        fx = np.arange(0,240)
        fy = gomp(fx,*fpm)
        plt.figure()
        plt.plot(d_listm,qmi,'o', markersize = 0.7)
        plt.plot(fx,fy, label = 'Gompertz on data')
    
    #output
    
    if envelope==True:
        return fp2[3],fpm[3],fp1[3] # 0.25 0.5 0.7 greening slope
    else:
        return fpm[3] # 0.5 quantile greening slipe

def eos(dates,data,time_res,envelope=True,ndvi_th=0.1,pth=5,sttt=213,entt=365):
    """
    EOS (End Of Season)
    Computes the season end for given NDVI annual data and an NDVI 
    threshold considered the end of the greening season. 
    Data are interpolated as annual curve. If more values are present with
    the same dates (ex. pixels from the same image), the median is taken.
    If envelope = True, the browning time is computed also for the 0.25-0.75 
    quantile envelope curves.
    Parameters
    ----------
    dates : vector
        any date of time vector the data
    data : vector
        data vector
    time_res : string, optional
        wanted time resolution among "doy", "week", and "month". The default is 'doy'.
    envelope : boolean, optional
        if = True EOS of the 0.25 and 0.75 quantile envelope curves are computed. 
        The default is True.
    ndvi_th : scalar
        ndvi threshold below which the browning time is reached. Default is 0.1
    pth : scalar
        number of time steps for which ndvi_th has to be passed in order to 
        define the brwning time. Default is 5 
    sttt : scalar, optional
        starting time for the interpolation. The default is 213 (half of season).
    entt : scalar, optional
        ending time for the interpolation. The default is 366 (end of year).

    Returns
    -------
    q2sum : scalar
    EOS (in time units given by time_res) of the 0.25 quantile annual curve
    qmi : scalar
    EOS of the median annual curve
    q1i : scalar
    EOS of the 0.75 quantile annual curve

    """
    d_list = np.unique(dates)
    #plt.grid(axis='y',linestyle='--')
    
    if envelope==True:
        q1 = np.array([],dtype=float)
        q2 = np.array([],dtype=float)
        for d in d_list:
            q1 = np.hstack((q1,np.nanquantile(data[dates==d],0.75)))
            q2 = np.hstack((q2,np.nanquantile(data[dates==d],0.25)))
        d_list1,q1i,*tmp = annual_interp(d_list,q1,time_res=time_res,sttt=sttt,entt=entt)
        d_list2,q2i,*tmp = annual_interp(d_list,q2,time_res=time_res,sttt=sttt,entt=entt)
        
        gsw = False
        n = 0
        egq1 = np.nan
        for i in range(len(d_list1)):
            if n==5: 
                break
            elif q1i[i]<ndvi_th and gsw==False:
                egq1 = d_list1[i]
                gsw = True
                n = n+1
            elif q1i[i]<ndvi_th and gsw==True:
                n = n+1
            elif q1i[i]>=ndvi_th:
                egq1 = np.nan
                gsw = False
                n = 0
        
        gsw = False
        n = 0
        egq2 = np.nan
        for i in range(len(d_list2)):
            if n==5: 
                break
            elif q2i[i]<ndvi_th and gsw==False:
                egq2 = d_list1[i]
                gsw = True
                n = n+1
            elif q2i[i]<ndvi_th and gsw==True:
                n = n+1
            elif q2i[i]>=ndvi_th:
                egq2 = np.nan
                gsw = False
                n = 0
        
    qm = np.array([],dtype=float)
    for d in d_list:
        qm = np.hstack((qm,np.nanquantile(data[dates==d],0.5)))  
    d_listm,qmi,*tmp = annual_interp(d_list,qm,time_res=time_res,sttt=sttt,entt=entt)
    
    #ndvi_th = 0.2
    #pth = 5
    gsw = False
    n = 0
    egm = np.nan
    for i in range(len(d_listm)):
        if n==pth: 
            break
        elif qmi[i]<ndvi_th and gsw==False:
            egm = d_listm[i]
            gsw = True
            n = n+1
        elif qmi[i]<ndvi_th and gsw==True:
            n = n+1
        elif  qmi[i]>=ndvi_th:
            egm = np.nan
            gsw = False
            n = 0
        
    if envelope==True:
        return egq2,egm,egq1 # 0.25 0.5 0.7 browning time
    else:
        return egm # 0.5 quantile browning time

def gomp(x,a,b,c,d):
    """
    GOMPERTZ
    1D sigmoidal function of the form:
        y = a(exp(-b*exp(c*x))+d

        
    Parameters
    ----------
    x : vector
        independent variable at which the function is evaluated
        
    a : scalar
        eight of the bell
        
    b : scalar
        x coordinates of the inflection point of the sigmoid slope
        
    c : scalars
        sigmoid slope
    
    d : vertical shift of the function

    Returns
    -------
    y : vector
        function evaluated at x

    """
    y = -a*(np.exp(-np.exp((x-b)*c)))+d
    return y 

def snow_plot(t,ts_data,year,dcol,stat='mean',lb='data',rb='data',slabel='snow depth'):
    
    y_tmp = []
    d_tmp = []
    for i in range(len(t)):
        y_tmp.append(t[i].year)
        d_tmp.append(t[i].timetuple().tm_yday)
    
    d_tmp = np.array(d_tmp)
    y_ind = np.in1d(y_tmp,year)
    dates = d_tmp[y_ind]
    data = ts_data[y_ind]
    if stat == 'cumsum':
        data = np.cumsum(data)
    xi,yi,*tmp = annual_interp(dates,data,time_res='doy',lb=lb,rb=rb,sttt=0)
    yi = yi/np.max(yi)-np.min(yi)
    plt.plot(xi,yi,dcol,label=slabel)
    
    return xi, yi

def reproject_raster(src,out_crs):
    """
    REPROJECT RASTER reproject a given rasterio object into a wanted CRS.

    Parameters
    ----------
    src : rasterio.io.DatasetReader
        rasterio dataset to reproject.
        For a geoTiff, it can be obtained from:    
        src = rasterio.open(file.tif,'r')
            
    out_crs : int
        epgs code of the wanted output CRS

    Returns
    -------
    dst : rasterio.io.DatasetReader
        output rasterio dataset written in-memory (rasterio MemoryFile)
        can be written to file with:
        
        out_meta = src.meta.copy()
        with rasterio.open('out_file.tif','w', **out_meta) as out_file: 
            out_file.write(dst.read().copy())
            
        out_file.close()

    """
    
    src_crs = src.crs
    transform, width, height = calculate_default_transform(src_crs, out_crs, src.width, src.height, *src.bounds)
    kwargs = src.meta.copy()
    
    memfile = MemoryFile()
    
    kwargs.update({
        #'driver':'Gtiff',
        'crs': out_crs,
        'transform': transform,
        'width': width,
        'height': height,
        "BigTIFF" : "yes"})
    
    dst = memfile.open(**kwargs)

          
    for i in range(1, src.count + 1):
        reproject(
            source=rasterio.band(src, i),
            destination=rasterio.band(dst, i),
            src_transform=src.transform,
            src_crs=src_crs,
            dst_transform=transform,
            dst_crs=out_crs,
            resampling=Resampling.nearest)
    
    return dst


def download_SA3D_STAC(
        bbox_path: str, 
        out_crs: str, 
        out_res: float,
        out_path: str,
        server_url: str =  'https://data.geo.admin.ch/api/stac/v0.9/collections/ch.swisstopo.swissalti3d',
        product_res_label: str = '2'
    ):
    """
    DOWNLOAD_SA3D_STAC downloads the SwissAlti3D product for the bounding box 
    of a given shapefile and output resolution. The projection grid will start 
    from the lower and left box bounds. 
    
    Parameters
    ----------
    bbox_path : str
        path to the input shapefile, it can be a .gpkg or .shp with georef files
        in any crs
    out_crs : int
        epgs code of the output CRS (e.g. 4326)
    out_res : float
        output resolution
    out_path : str
        output .tif file path to create
    server_url : str, optional
       Swisstopo STAC server url for the product.
       The default is 'https://data.geo.admin.ch/api/stac/v0.9/collections/ch.swisstopo.swissalti3d'.
    product_res_label : str, optional
        the original product comes in 0.5 or 2-m resolution. 
        The default is '2'.

    Returns
    -------
    None.
    The projected DEM is written in out_path
    
    Example
    -------
    download_SA3D_STAC(
            bbox_path = bbox_fname, # bbox in any crs
            out_crs = 2056, # wanted output crs EPGS code
            out_res = 10, # wanted output resolution (compatible with out crs)
            out_path = 'prova.tif',
            server_url = 'https://data.geo.admin.ch/api/stac/v0.9/collections/ch.swisstopo.swissalti3d',
            product_res_label = '2' # can only be 0.5 or 2, original product resolution 
        )

    """
    
    # RETRIEVE TILE LINKS FOR DOWNLOAD
    
    shp = gpd.read_file(bbox_path).to_crs('epsg:4326') # WG84 necessary to the query
    lef = np.min(shp.bounds.minx)
    rig = np.max(shp.bounds.maxx)
    bot = np.min(shp.bounds.miny)
    top = np.max(shp.bounds.maxy)
    
    # If bbox is big divide the bbox in a series of chunks to override server limtiations
    cell_size = 0.05 #temporary bounding size in degree to define the query chunks 
    xbb = np.arange(lef,rig,cell_size)
    if xbb[-1] < rig:
        xbb = np.append(xbb,rig)
    ybb = np.arange(bot,top,cell_size)
    if ybb[-1] < top:
        ybb = np.append(ybb,top)
    
    files = []
    for i in range(len(xbb)-1):
        for j in range(len(ybb)-1):
            bbox_tmp = [xbb[i],ybb[j],xbb[i+1],ybb[j+1]]
            # construct bbox specification required by STAC
            bbox_expr = f'{bbox_tmp[0]},{bbox_tmp[1]},{bbox_tmp[2]},{bbox_tmp[3]}'
            # construct API GET call
            url = server_url + '/items?bbox=' + bbox_expr
            # send the request and check response code
            res_get = requests.get(url)
            res_get.raise_for_status()         
            # get content and extract the tile URLs
            content = json.loads(res_get.content)
            features = content['features']
            for feature in features:
                assets = feature['assets']
                tif_pattern = re.compile(r"^.*\.(tif)$")
                tif_files = [tif_pattern.match(key) for key in assets.keys()]
                tif_files =[x for x in tif_files if x is not None]
                tif_file = [x.string if x.string.find(f'_{product_res_label}_') > 0 \
                            else None for x in tif_files]
                tif_file = [x for x in tif_file if x is not None][0]
                # get download link
                link = assets[tif_file]['href']
                files.append(link)
        
    # reprojects tiles in out crs
    file_handler = []
    for row in files:
        src = rasterio.open(row,'r')
        dst = reproject_raster(src,out_crs)
        file_handler.append(dst)
    
    shp = gpd.read_file(bbox_path).to_crs(out_crs) # WG84 necessary to the query
    lef = np.min(shp.bounds.minx)
    rig = np.max(shp.bounds.maxx)
    bot = np.min(shp.bounds.miny)
    top = np.max(shp.bounds.maxy)
    
    merge(datasets=file_handler, # list of dataset objects opened in 'r' mode
    bounds=(lef, bot, rig, top), # tuple
    nodata=0, # float
    dtype='uint16', # dtype
    res=out_res,
    resampling=Resampling.nearest,
    method='first', # strategy to combine overlapping rasters
    dst_path=out_path # str or PathLike to save raster
    )
    

def write_bbox(fpath,out_base_path):
    """
    Generates a bounding-box shapefile form a given geotiff / shp / gpkg file.

    Parameters
    ----------
    fpath : str
        original tif, shp, or gpkg file
    out_base_path : str
        Shapefile output base path (with no extensions, since multiple related
                                    fiels are generated)

    Returns
    -------
    None.
    
    """
    
    # read original file
    if  fpath[-3:] == 'tif' or  fpath[-4:] == 'tiff':
        
        dataset = rasterio.open(fpath)
        lef = dataset.bounds.left
        rig = dataset.bounds.right
        
        if dataset.bounds.bottom > dataset.bounds.top:
            top = dataset.bounds.bottom
            bot = dataset.bounds.top
        else:
            bot = dataset.bounds.bottom
            top = dataset.bounds.top
    
    elif fpath[-3:] == 'shp' or  fpath[-4:] == 'gpkg':
        
        dataset = gpd.read_file(fpath)
        lef = np.min(dataset.bounds.minx)
        rig = np.max(dataset.bounds.maxx)
        bot = np.min(dataset.bounds.miny)
        top = np.max(dataset.bounds.maxy)
    
    # write bbox shapefile
    bbox_fname = out_base_path + '.shp'
    w = shapefile.Writer(bbox_fname)
    w.field([],'C')
    w.poly([[[lef,bot],[lef,top],[rig,top],[rig,bot]]])
    w.record([])
    w.close()
    
    # write georeferenziation file
    esri_code = int(dataset.crs.to_string()[5:])
    dataset = None
    spatialRef = osr.SpatialReference()
    spatialRef.ImportFromEPSG(esri_code)
    spatialRef.MorphToESRI()
    file = open(out_base_path + '.prj', 'w')
    file.write(spatialRef.ExportToWkt())
    file.close()

def write_bbox_from_shp(fpath,out_base_path):
    """
    Generates a bounding-box shapefile form a given geotiff / shp / gpkg file.

    Parameters
    ----------
    fpath : str
        original tif, shp, or gpkg file
    out_base_path : str
        Shapefile output base path (with no extensions, since multiple related
                                    fiels are generated)

    Returns
    -------
    None.
    
    """
    
    

def rasterize_shp(tx:float,
                  ty:float,
                  shp_path:str,
                  field:str = '1',
                  no_data = 0,
                  njobs:int = 1,
                  target_crs:int = None):
    """
    RASTERIZE SHP project a polygon shapefile on a raster defined by the coord
    vectors tx and ty, which can be given or generated by make_grid_vec().

    Parameters
    ----------
    tx : float
        x-coord vector
    ty : float
        y-coord vector
    shp_path : str
        path to shapefile to rasterize
    field : str
        name of the field to give as value of the pixels covered by the polygons. 
        available fields can be see with:
        
        sf = gpd.read_file(shp_path) # read shape
        sf.columns # field names
        Default is '1', meaning constant 1 for all polygons pixels.
        
    no_data : numerical or object
        no data value in the raster. Default is 0.
    
    njobs : int
        number of parallel threads for parallelization
        Default is 1

    target_crs: int
        epsg code for the target crs for the output coordinate vectors tx and ty.
        Deafult is None. If None the original crs of shp_fname is used. 
    

    Returns
    -------
    rast : array
    array of size [ty,tx] with shapes projected with wanted field

    """
    
    
    # Step 1: Read Shapefile

    if target_crs!=None:
        gdf = gpd.read_file(shp_path).to_crs("EPSG:" + str(target_crs))
    else:
        gdf = gpd.read_file(shp_path)

    # not necessary to explode, present geometries will be rasterized in form
    # polygon or multipolygon. Explode would create double-level indexes for geometry
    # parts.
    
    # # Step 2: Explode MultiPart geometries
    # gdf_exploded = gdf.explode(index_parts=False)
    
    # # Step 3: Access Exploded Geometry
    # geometry = gdf_exploded.geometry
    
    # Step 4: Access the geomtery field to populate raster
    
    if field == '1': # costant 1s everywhere
        
        fi_re = np.ones(len(gdf.geometry))
    
    else: # or wanted field
    
        fi_re = np.array(gdf[field])
        
    #start = time.time() # start time

    xx, yy = np.meshgrid(tx,ty,indexing='xy')
    
    # geoseries of grid points
    s = gpd.GeoSeries.from_xy(xx.ravel(), yy.ravel(), crs="EPSG:2056")
    
    # preallocate raster
    rast = np.zeros_like(xx)
    rast[:] = no_data
        
    # Create a lock to synchronize multi-thread access to the matrix
    lock = threading.Lock()
    progress_bar = tqdm(total=len(gdf))
    
    def rast_poly(geom,s,rast,fi_re,lock,i,progress_bar):
        
        ind = np.array(geom[i].contains(s))
        
        with lock:
            rast.ravel()[ind] = fi_re[i] # assign corresponding filed value
            progress_bar.update(1)
            
    items = np.arange(len(gdf.geometry))
    
    # Create a ThreadPoolExecutor with a maximum of 4 worker threads
    with concurrent.futures.ThreadPoolExecutor(max_workers=njobs) as executor:
        # Submit each item in the list to the executor
        futures = [executor.submit(rast_poly, gdf.geometry, s, rast, fi_re, lock,i,progress_bar) for i in items]
        
        # Wait for all futures to complete
        concurrent.futures.wait(futures)

        # # Create a list of items to process
        # items = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        
        # # Create a matrix to store the results
        # matrix = [None] * (max(items) + 1)
        
        # # Create a lock to synchronize access to the matrix
        # lock = threading.Lock()
        
        # # Create a ThreadPoolExecutor with a maximum of 4 worker threads
        # with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        #     # Submit each item in the list to the executor
        #     futures = [executor.submit(process_item, item, matrix, lock) for item in items]
            
        #     # Wait for all futures to complete
        #     concurrent.futures.wait(futures)
     
    #  %%%%%%%5 LINEAR LOOP (OLD)
    # for i in range(len(gdf.geometry)):
    #     n = n+1
    #     ind = np.array(gdf.geometry[i].contains(s))
    #     rast.ravel()[ind] = fi_re[i] # assign corresponding filed value
    #     print(n)
    
    #  %%%%%%%%% OLD IMPLEMENTATION
        
    #end = time.time(), print(end - start) # end time

    # # Step 5: Extract Coordinates
    # shx = []
    # shy = []
    
    # for geom in geometry:
        
    #     shx_tmp = []
    #     shy_tmp = []
        
    #     for point in geom.exterior.coords:
            
    #         shx_tmp.append(point[0])
    #         shy_tmp.append(point[1])
        
    #     shx.append(np.array(shx_tmp))
    #     shy.append(np.array(shy_tmp))
        
            
            #x, y = point  # For 2D data (longitude, latitude)
            # For 3D data (longitude, latitude, elevation), use:
            # x, y, z = point
            #print(f"Longitude: {x}, Latitude: {y}")

    
    # # create shapefile object OLD
    # sf = Reader(shp_path,encoding='UTF-8') # specify encoding
    
    # # retrieve wanted field index to fill the raster
    # fi = np.array(sf.fields) # [field name, type, data length, decimal places]
    # fi_names = fi[:,0] # field names
    # fi_types = fi[:,1] # field types
    
    # fi_ind = fi_names == field # wanted field index
    # fi_type = fi_types[fi_ind][0] # wanted field type
    # fi_ind = fi_ind[1:] # remove initial field index (deletion flag not present in re)
    
    # # translate fi_type name into array_type name 
    # fi_type_list = np.array(['C','N','F','L','D'])
    # array_type_list = np.array(['str','float','float','Boolean','str'])
    # array_type = array_type_list[np.in1d(fi_type_list,fi_type)][0]

    # # retrieve wanted field records (associated to the shapes, contain all fields for every shape)
    # fi_re = np.array(sf.records())[:,fi_ind].astype(array_type)
    
    # # retrieve shapes
    # sh = np.array(sf.shapes()) # shape

    # # extract atribute and point vector for every polygon
    # shx = []
    # shy = []
    # for i in range(len(fi_re)):
       
    #     # consider only records with non-zero shape 
    #     if len(np.array(sf.geometry[i].boundary.coords.xy[0]))>0:
    #         shx.append(np.array(sf.geometry[i].boundary.coords.xy[0]))
    #         shy.append(np.array(sf.geometry[i].boundary.coords.xy[1]))
    #         # shx.append(np.array(sh[i].points)[:,0])
    #         # shy.append(np.array(sh[i].points)[:,1])

    # shx = np.array(shx)
    # shy = np.array(shy)
    
    # # project shapes on the target grid
    # xx, yy = np.meshgrid(tx,np.flip(ty),indexing='xy')
    # rast = np.zeros_like(xx)
    # rast[:] = no_data
    # points = np.array((xx.flatten(), yy.flatten())).T
    
    # #%%%%%%%%%%%%%%% PARALLEL IMPLEMENTATION
    # import concurrent.futures
    # import threading

    # # Define a function that will be executed in parallel
    # def process_item(item, matrix, lock):
    #     # Perform some processing on the item and write to the matrix
    #     result = item * item
    #     with lock:
    #         matrix[item] = result


    # # Create a list of items to process
    # items = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    # # Create a matrix to store the results
    # matrix = [None] * (max(items) + 1)
    
    # # Create a lock to synchronize access to the matrix
    # lock = threading.Lock()
    
    # # Create a ThreadPoolExecutor with a maximum of 4 worker threads
    # with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    #     # Submit each item in the list to the executor
    #     futures = [executor.submit(process_item, item, matrix, lock) for item in items]
        
    #     # Wait for all futures to complete
    #     concurrent.futures.wait(futures)

    
    return rast

# def write_geotiff(output_path,array,x_origin,y_origin,pixel_size,epsg=None,dtype=None,nodata_val=None,band_name=None):
    
    # # nodata_val, band_name should be lists/array even if containing one value
    # if dtype==None: # set data type from given format
    #     dtype=im.dtype.name
    # if np.ndim(im)<3:
    #     nb=1
    #     im=im[:,:,None]
    # else:
    #     nb=np.shape(im)[2]
    
    # driver =  GetDriverByName('GTiff') # generate driver    
    # outdata = driver.Create(filename, np.shape(im)[1], np.shape(im)[0], nb, GetDataTypeByName(dtype))
    # for i in range(nb): # write data
    #     if dtype!=None:
    #         outdata.GetRasterBand(i+1).WriteArray(im[:,:,i].astype(dtype))
    #     else:
    #         outdata.GetRasterBand(i+1).WriteArray(im[:,:,i])
            
    #     if nodata_val!=None:
    #         outdata.GetRasterBand(i+1).SetNoDataValue(nodata_val)
        
    #     if band_name!=None:
    #         outdata.GetRasterBand(i+1).SetDescription(band_name[i])
        
    # # GEOREFERENTIATION
    #  # coords
    # geotransform = (xmin, # x-coordinate of the upper-left corner of the upper-left pixel.
    #                 xres, # W-E pixel resolution / pixel width.
    #                 0, #row rotation (typically zero).
    #                 ymax, # y-coordinate of the upper-left corner of the upper-left pixel.
    #                 0, # column rotation (typically zero).
    #                 -yres) # N-S pixel resolution / pixel height (negative value for a north-up image).
    # outdata.SetGeoTransform(geotransform)    # specify coords
    # srs = osr.SpatialReference()            # establish encoding
    # srs.ImportFromEPSG(epsg)                # set CRS from EPSG code 
    # outdata.SetProjection(srs.ExportToWkt())
    # # save data
    # outdata.FlushCache()
    # outdata=None
    
def write_geotiff(out_path,array,x_origin,y_origin,pixel_size,epsg,band_names,nodata_val=None):
    """
    Write N-D array to geotiff.

    Parameters
    ----------
    out_path : str
        path of the out geotiff file to create.
    array : numerical
        2-D or 3-D array, where bands are stacked in the 3rd dimension.
    x_origin : float
        x-coord of the upper left corner of the upper left pixel.
    y_origin : float
        y-coord of the upper left corner of the upper left pixel..
    pixel_size : float
        pixel size
    epsg : str
        EPSG code of the CRS liked to the array coordinates
    band_names : list of str
        list of names to give to each band.
    nodata_val : type, optional
        nodata value present for missing data in the array. The default is None

    Returns
    -------
    None.

    """
    
    # Create the transform object
    transform = from_origin(x_origin, y_origin, pixel_size, pixel_size)
    
    # Define the coordinate reference system (CRS) for the GeoTIFF
    crs = 'EPSG:' + epsg  # WGS84

    
    if array.ndim==2:
        
        array = np.expand_dims(array, axis=2)
        
        
    nbands = array.shape[2]
    
    # Write the array to a GeoTIFF file
    with rasterio.open(
        out_path, 'w',
        driver='GTiff',
        height=array.shape[0],
        width=array.shape[1],
        count=nbands,  # Number of bands
        dtype=array.dtype,
        crs=crs,
        transform=transform
    ) as dst:
        
        for i in range(nbands):
            dst.write(np.squeeze(array[:, :, i]), i + 1)  # Write each band separately
            dst.set_band_description(i + 1, band_names[i])  # Set band name

def init_data_cube(zarr_path,tx,ty,BAND_LIST,res,crs,pid_rast,SIND_NAME,chunk_size=10):
    
    # Create a Zarr group
    data_cube = zarr.open_group(zarr_path,mode='w') # main group
    mc = 50 # memory chunk manually setup where necessary

    # image cube
    im_sh = (len(ty), len(tx), len(BAND_LIST)+1,0)
    im = np.empty(im_sh)
    data_cube.create_dataset('im',
                             data=im,
                             shape=im.shape,
                             dtype=im.dtype)
    
    # derived spectral indicator cube
    sind_sh = (len(ty), len(tx), 0)
    sind = np.empty(sind_sh)
    data_cube.create_dataset('sind', 
                     data=sind, # x, y, time 
                     shape=sind.shape,
                     chunks=(chunk_size,chunk_size,mc), # memory chunk = 1 for x and y because they pixels will be accessed separately)
                     dtype=sind.dtype) # data type
    
    # spectral indicator name
    data_cube.create_dataset('sind_name', 
                     data=SIND_NAME, 
                     dtype='str') # data type
    
    # date vector
    #im_date_sh = (0)
    im_date = pd.Series(dtype='datetime64[ns]') 
    data_cube.create_dataset('im_date',
                         data=im_date.values,  # pass the underlying numpy array
                         dtype=im_date.values.dtype)  # get the dtype from the numpy array

    # cloud percentage vector
    im_cloud_perc = pd.Series(dtype='float64') 
    data_cube.create_dataset('im_cloud_perc', 
                     data=im_cloud_perc.values,
                     dtype=im_cloud_perc.dtype) # data type
    
    # tx
    data_cube.create_dataset('tx', 
                     data=tx, 
                     #chunks=(mc), # memory chunks
                     dtype='float') # data type
    
    # ty
    data_cube.create_dataset('ty', 
                     data=ty, 
                     #chunks=(mc), # memory chunks
                     dtype='float') # data type
    
    # pid_rast
    data_cube.create_dataset('pid_rast', 
                     data=pid_rast,
                     dtype='int') # data type
    
    # bands
    data_cube.create_dataset('bands', 
                     data=BAND_LIST,
                     dtype='str') # data type
    
    # res
    data_cube.create_dataset('res', 
                     data=res,
                     dtype='float') # data type
    
    # CRS EPSG
    data_cube.create_dataset('crs', 
                     data=crs,
                     dtype='int') # data type
    
    # data_cube is automatically saved


def append_data_cube(zarr_path,
         im_date, # dates vector
         im_cloud_perc, # cloud pecentage vector
         im,  # images array [x,y,band,scene]
         sind # indicator array
         ):

    # Open the Zarr array in append mode
    z = zarr.open(zarr_path + '/im_date', mode='a')
    
    # Append the additional data
    z.append(im_date.values)
    
    # Open the Zarr array in append mode
    z = zarr.open(zarr_path + '/im_cloud_perc', mode='a')
    
    # Append the additional data
    z.append(im_cloud_perc.values)
    
    # Open the Zarr array in append mode
    z = zarr.open(zarr_path + '/im', mode='a')
    
    # Append the additional data
    z.append(im,axis=3)
    
    # Open the Zarr array in append mode
    z = zarr.open(zarr_path + '/sind', mode='a')
    
    # Append the additional data
    z.append(sind,axis=2)

def _proc_chunk(chunk_coords, time_vec, shm_name, shape):
    """Process a chunk of the array to compute cumulative sums."""
    existing_shm = shared_memory.SharedMemory(name=shm_name)
    data_cube_shared = np.ndarray(shape, dtype='float', buffer=existing_shm.buf)

    (i_start, i_end), (j_start, j_end) = chunk_coords
    evi_chunk = data_cube_shared[i_start:i_end, j_start:j_end, :]
    auc_chunk = np.full((i_end - i_start, j_end - j_start), np.nan, dtype='float32')

    for i in range(evi_chunk.shape[0]):
        for j in range(evi_chunk.shape[1]):
            pixel_data = evi_chunk[i, j, :]
            if not np.all(np.isnan(pixel_data)):
                
                sogm, eosm, losm, aucm, msgm, sl, q90, stdm = annual_px_stats(time_vec,
                                                                            # EVI[i,j,:],
                                                                            pixel_data,
                                                                            plot=False)                
                auc_chunk[i, j] = aucm  # Example operation
                
    existing_shm.close()
    return (i_start, i_end), (j_start, j_end), auc_chunk

def _update_progress(result, progress_bar, auc_rast, chunk_npix):
    """Callback function to update shared state with the result of _proc_chunk."""
    (i_start, i_end), (j_start, j_end), auc_chunk = result
    auc_rast[i_start:i_end, j_start:j_end] = auc_chunk.copy()
    progress_bar.update(chunk_npix)

def indicator_map(data_cube, time_vec, njobs=1, chunk_size=10,silent=False):
    """
    INDICATOR_MAP creates an indicator map from a given annual curves cube. 
    Used in greentrack_map
    
    Parameters
    ----------
    data_cube : zarr array
        data cube of dimensionality (x,y,DOY) with vegetation index values.
        every pixel present an annual curve for every DOY. 
    time_vec : array
        array of DOYs of the same size of np.shape(data_cube)[2].
    njobs : int
        number of threads used in the parallel computation. Defauklt is 1 
        (linear computation).
    chunk_size : int
        number of pixels per size of chunk processed and stored. memory will be 
        organized in chunks of size chunk_size x chunk_size.
    silent : boolean
        if true the progress bars will not be shown.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    
    # pixel amount per chunk
    chunk_npix = chunk_size * chunk_size

    # preallocate output raster
    auc_rast = zarr.create(shape=data_cube.shape[:2], dtype='float32', chunks=(chunk_size, chunk_size), fill_value=np.nan)

    # Define chunk coordinates, handle boundary conditions
    nrows, ncols = auc_rast.shape
    total_chunks = (nrows // chunk_size + (1 if nrows % chunk_size != 0 else 0)) * (ncols // chunk_size + (1 if ncols % chunk_size != 0 else 0))    
    chunk_coords = [
        ((i, min(i + chunk_size, nrows)), (j, min(j + chunk_size, ncols)))
        for i in range(0, nrows, chunk_size)
        for j in range(0, ncols, chunk_size)
    ]

    # Create shared memory for data_cube
    shm = shared_memory.SharedMemory(create=True, size=data_cube.nbytes)
    data_cube_shared = np.ndarray(data_cube.shape, dtype=data_cube.dtype, buffer=shm.buf)
    np.copyto(data_cube_shared, data_cube)

    # Parallel loop
    with tqdm(total=total_chunks * chunk_npix,disable=silent) as progress_bar:
        _update_progress_with_bar = partial(_update_progress, progress_bar=progress_bar, auc_rast=auc_rast, chunk_npix=chunk_npix)
        
        with multiprocessing.Pool(processes=njobs) as pool:
            # Submit tasks to the pool
            results = [
                pool.apply_async(_proc_chunk, args=(coords, time_vec, shm.name, data_cube.shape), callback=_update_progress_with_bar)
                for coords in chunk_coords
            ]
            
            # Ensure all tasks are completed
            for r in results:
                r.wait()

    # Clean up shared memory
    shm.close()
    shm.unlink()

    return auc_rast  # raster map
    