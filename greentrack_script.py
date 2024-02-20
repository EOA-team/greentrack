#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SIMPLE SCRIPT TO DOWNLOAD SENTINEL IMAGES ON A COMMON GRID AND EXTRACT THE NDVI CURVE

import numpy as np
import matplotlib.pyplot as plt
font = {'family' : 'DejaVu Sans',
        'size'   : 14}
from matplotlib import rc
rc('font', **font)
import geopandas as gpd
from datetime import datetime, timedelta
from eodal.config import get_settings
from eodal.core.sensors.sentinel2 import Sentinel2
from eodal.mapper.feature import Feature
from eodal.mapper.filter import Filter
from eodal.mapper.mapper import Mapper, MapperConfigs
from pathlib import Path
from typing import List
from pandas import Series #, concat
import geopandas
from scipy.interpolate import interp2d
import re
import os
from datetime import datetime as dt 
from scipy.optimize import curve_fit
import greentrack_tools as gtt

# STAC PROTOCOL SETTINGS: set to False to use a local data archive
Settings = get_settings()
Settings.USE_STAC = True

#%% INPUT PARAMS

## HERE you can create a loop to run all following code for a list of sites (parcels)
# giving SITE_NAME a different name every loop iteration

SITE_NAME = 'cortone' # base name for output files and folder

# file path of the bounding box (can be geopackage or shapefile with related files)
bbox_fname = 'data/parcels__posieux_5.gpkg'

# list  of years you want the data for, can also contain one year
year_list = [2018] # [2016,2017,2018]

# local path where output directory and files are saved
SAVE_DIR = 'export' # 

# coudy pixel percentage to discard sentinel tiles
CLOUD_TH = 30

# sentinel product name
S_NAME = 'sentinel2-msi'

# bands to select (list)
# See available bands for sentinel-2 L2A here:
# https://docs.sentinel-hub.com/api/latest/data/sentinel-2-l2a/
# SCL (Scene Classification Layer) is added as extra band

BAND_LIST = [
# 'B01',
'B02', # RGB
'B03',
'B04',
# 'B05',
# 'B06',
# 'B07',
'B08', # NIR
# 'B8A',
# 'B09',
# 'B11',
# 'B12'
]

# Reuse existing data: If set to True, the script will look for already downloaded images
# present in SAVE_DIR for SITE_NAME and YEAR. If False all images will be dowloaded
# from scratch
 
REUSE_DATA = False

# PREPROCESSING FUNCTION - EDIT TO ADD PREPROCESSING TO THE EODAL SCENES

def preprocess_sentinel2_scenes(
    ds: Sentinel2,  # this is an EODAL Sentinel2 scene, 
                    # i.e. a RasterCollection object
    target_resolution: int, 
    
    # ADD HERE MORE ARGUMENTS (E.G.packages for preprocessing the images) 
    # and add these also in the dictionary below  'scene_modifier_kwargs'
    ) -> Sentinel2:
    
    
    """
    PREPROCESS_SENTINEL2_SCENES
    
    Preprocessing module for the EOdal mapper.
    
    Resample Sentinel-2 scenes and mask clouds, shadows, and snow
    # resample scene
    .resample(inplace=True, target_resolution=target_resolution)
    
    ask clouds, shadows, and snowbased on the Scene Classification Layer (SCL).
    
    	NOTE:
    		Depending on your needs, the pre-processing function can be
    		fully customized using the full power of EOdal and its
    		interfacing libraries!
    
    	:param target_resolution:
    		spatial target resolution to resample all bands to.
    	:returns:
    		resampled, cloud-masked Sentinel-2 scene.
	"""
    
    # resample scene
    ds.resample(inplace=True, target_resolution=target_resolution) 
    
    # mask clouds, shadows, but leave snow (class 11), see page 304 https://sentinel.esa.int/documents/247904/685211/sentinel-2-products-specification-document
    # Label Classification
    # 0 NO_DATA
    # 1 SATURATED_OR_DEFECTIVE
    # 2 DARK_AREA_PIXELS
    # 3 CLOUD_SHADOWS
    # 4 VEGETATION
    # 5 BARE_SOILS
    # 6 WATER
    # 7 UNCLASSIFIED 
    # 8 CLOUD_MEDIUM_PROBABILITY
    # 9 CLOUD_HIGH_PROBABILITY 
    # 10 THIN_CIRRUS
    # 11 SNOW /ICE
    
    ds.mask_clouds_and_shadows(inplace=True, cloud_classes=[1, 2, 3, 7, 8, 9, 10])   # MASKED BY EODAL
    
    return ds

##############################################################################


#%% LOOP OVER YEARS

for k in range(len(year_list)):
    
    YEAR = year_list[k]
    
    print('#### Extracting data for site ' + SITE_NAME + ' year ' + str(YEAR))
    
    #% (EDIT) EODAL QUERY PARAMETERS
       
    # user-inputs
    # -------------------------- Collection -------------------------------
    collection: str = S_NAME
    
    # ------------------------- Time Range ---------------------------------
    time_start: datetime = datetime(YEAR,1,1)  		# year, month, day (incl.)
    time_end: datetime = datetime(YEAR,12,31)     	# year, month, day (incl.)
    
    # ---------------------- Spatial Feature  ------------------------------
    geom: Path = Path(bbox_fname) # BBOX as geometry for the query
    	
    # ------------------------- Metadata Filters ---------------------------
    metadata_filters: List[Filter] = [
      	Filter('cloudy_pixel_percentage','<', CLOUD_TH),
      	Filter('processing_level', '==', 'Level-2A')
        ]
    
    #  ---------------------- query params for STAC  ------------------------------
    
    scene_kwargs = {
        'scene_constructor': Sentinel2.from_safe,
        'scene_constructor_kwargs': {'band_selection': BAND_LIST},
        'scene_modifier': preprocess_sentinel2_scenes,
        'scene_modifier_kwargs': {'target_resolution': 10}
    }
        
    feature = Feature.from_geoseries(gpd.read_file(geom).geometry)
    
    #% DOWNLOAD THE IMAGES
    
    # path where results are saved
    OUT_PATH = SAVE_DIR + '/' + SITE_NAME + '_' + str(YEAR)
    if not os.path.exists(OUT_PATH):
        os.makedirs(OUT_PATH)
    
    # subfolder where sat images and temp data are saved
    DATA_PATH = 'data/' + SITE_NAME + '_' + str(YEAR)
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
    
        
    
    # split the wanted date range in approx 30-day chuncks to override download limit
    
    CHUNK_SIZE = 30
    date_vec = [time_start]
    date_new = time_start + timedelta(days = CHUNK_SIZE)
    
    n = 1
    while date_new < time_end and n < 100: # max 100 chunks
        date_vec.append(date_new)
        date_new = date_new + timedelta(days = CHUNK_SIZE)        
        n = n+1
    date_vec.append(time_end)
    
    
    
    #% define target grid based on original bbox (local crs) and target resolution
    
    res = scene_kwargs['scene_modifier_kwargs']['target_resolution']
    tx, ty = gtt.make_grid_vec(bbox_fname,res)
    
    im_date = Series([])
    im_cloud_perc = Series([])
    
    
    # DOWNLOAD DATA    
    
    if REUSE_DATA == False:
        
        # delete old downloaded files
        target_dir = DATA_PATH
        target_pattern = 's2'
        gtt.purge(target_dir, target_pattern)    
        
        # reset the counter to 0
        n = 0 # data chunk counter
        np.save(DATA_PATH + '/counter.npy',n)
        n_block = 0 # saved data chunk counter (a dedicated one because empty chunks are skipped)
        np.save(DATA_PATH + '/block_counter.npy',n)
    
    
    ############################################
    
    n = np.load(DATA_PATH + '/counter.npy') # counter to resume from last completed chunk
    n_block = np.load(DATA_PATH + '/block_counter.npy') # counter to resume from last completed chunk
    
    for i in range(n,len(date_vec)-1):
        
        data_fname = DATA_PATH + '/s2_data_' + str(n_block) + '_snow.npz'
        
        if REUSE_DATA == False or os.path.exists(data_fname) == False:
        
            print('DOWNLOADING DATA CHUNK '  + str(i) + ' of ' + str(len(date_vec)-2))
            mapper_configs = MapperConfigs(
                 collection = collection,
                 time_start = date_vec[i],
                 time_end = date_vec[i+1],
                 feature = feature,
                 metadata_filters = metadata_filters
             )
        
        
            # Create mapper
            mapper = Mapper(mapper_configs)
           
            try:
                mapper.query_scenes()
            
            except Exception as e: 
                
                # if no images available are found skip to the next data chunk
                if e.args[0] == "Querying STAC catalog failed: 'sensing_time'":
                    print('No images found, continuing to the next data chunk')
                    n = n+1 # update counter
                    np.save(DATA_PATH + '/counter.npy',n)
                    continue # skip this data chunk download
                
                else:
                    print(e) 
                    break
            
            # download the images
            mapper.load_scenes(scene_kwargs=scene_kwargs)
            
            # display image headers
            mapper.data

            
            if mapper.data is None:
                
                print('No images found, continuing to the next data chunk')
                n = n+1 # update counter
                np.save(DATA_PATH + '/counter.npy',n)
                
                continue # skip this data chunk download
            
            for _, scene in mapper.data:
                
                # reproject scene 
                shp = gpd.read_file(bbox_fname)
                scene.reproject(inplace=True, target_crs=shp.crs.to_epsg())
                
            # retrieve band names
            sc = mapper.data
            bands = sc[sc.timestamps[0]].band_names # bands
            
            # extract and save image dates, cloud percentage, and images
            # if any data is present
            if not mapper.data.empty:
                
                im = gtt.scene_to_array(mapper.data,tx,ty)

                if np.any(im!=0): # if any data is non zero
                    
                    im_date = mapper.metadata['sensing_time']
                    im_cloud_perc = im_cloud_perc
                    
                    # SAVE multiband image as a .npz file
                    np.savez(data_fname,
                             im_date = im_date, # dates vector
                             im_cloud_perc = im_cloud_perc, # cloud pecentage vector
                             bands = bands, # band names
                             im = im, # images array [x,y,band,scene]
                             tx = tx, # x coord vector
                             ty = ty,  # y coord vector
                             shp = shp # roi shapefile
                             )
                    
                    im[im==0] = np.nan
                    
                    #% compute NDVI, YOU CAN PUT HERE OTHER INDICES
                    # AND ADD THEM TO THE SAVED IMAGES
                    RED = np.squeeze(im[:,:,np.array(bands)==['B04'],:])
                    NIR = np.squeeze(im[:,:,np.array(bands)==['B08'],:])
                    NDVI = (NIR-RED)/(NIR+RED)
                    
                    # SAVE NDVI image as .npz, SAVE HERE NEW INDICES                
                    np.savez(DATA_PATH + '/s2_ndvi_' + str(n_block) + '_snow.npz',        
                             im_date = im_date, # dates vector
                             im_cloud_perc = im_cloud_perc, # cloud pecentage vector
                             im = NDVI, # images array [x,y,band,scene]
                             tx = tx, # x coord vector
                             ty = ty,  # y coord vector
                             shp = shp # roi shapefile
                             )
                    
                    n_block = n_block + 1
                    
                    # save block counter
                    np.save(DATA_PATH + '/block_counter.npy',n_block)
                    
                    del im, mapper, NDVI, RED, NIR
                
        n = n+1 # update counter
        np.save(DATA_PATH + '/counter.npy',n)

                 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% PREPROCESS %%%%%%%%%%%%%%%%%%%%%%%%
#% IMPORT GRID FROM FIRST SAT IMAGE CHUNK

    variables = np.load(DATA_PATH + '/s2_data_0_snow.npz')
    variables.allow_pickle=True
    locals().update(variables)
    del variables
    xx, yy = np.meshgrid(tx,ty,indexing='xy')
    res = tx[1]-tx[0]
    im_extent = [tx[0]-res/2,tx[-1]+res/2,ty[0]+res/2,ty[-1]-res/2]



#%% INITIALIZE DATABASE AS DICTIONARY
# # BASIC DICTIONARY STRUCTURE: EXA503MPLE
# base name for output files

# mydict = {
#     "ndvi": [0.1,0.2,0.5,0.8],
#     "unit":[2,3,5,6,9],
#     "date":[datetime(2016,2,1),datetime(2016,6,1),datetime(2017,2,1),datetime(2019,2,1)],
#     "shadow":[0,1,1,0]
#     }
#
# variables can then be extracted as numpy arrays,ex:
# ndvi = np.concatenate(mydict['ndvi'])
#

    m_list = np.arange(1,13) # months list now including winter
    
    
    mykeys = ['ndvi', # NDVI pixel value
              'date', # original image date
              'data_chunk_n', # data chunk the image belongs to
              'year', # image year
              'month', # image month
              'week', # image week
              'doy', # image day of the year
              'unit', # pixel vegetation unit
              'pid', # parcel id the pixel belongs to
              # 'shadow', # shadow indicator
              # 'elevation', # elevation
              # 'aspect', # aspect
              # 'slope', # slope
              # 'curvature', # curvature
              'x', # x coord
              'y'] # y coord
    
    #inizialize dictionarîes for shadow and sun data, dictionary structure is nested [unit][month]
    ndvi_dict = dict.fromkeys(mykeys)
    
    for i in ndvi_dict.keys():
          ndvi_dict[i] = list([])


    #%% EXTRACT NDVI FROM DATA CHUNKS, 
    # resuming from latest data chunk processed, q_tmp
    
    plt.close('all')
    
    # INITIALIZE SAVED DATA (COMMENT TO RESUME PREVIOUS JOB)
    np.savez(DATA_PATH + '/ndvi_dict_snow.npz',
              ndvi_dict = ndvi_dict,
              q_tmp = 0, # current data chunk to process, 0 at the beginning 
              )
    
    # RESUME SAVED DATA
    variables = np.load(DATA_PATH + '/ndvi_dict_snow.npz')
    variables.allow_pickle=True
    locals().update(variables)
    del variables
    ndvi_dict = ndvi_dict.all()
    
    # vt_ind = np.in1d(vt_rast,vt_list).reshape(np.shape(vt_rast))
    nblocks = np.load(DATA_PATH + '/block_counter.npy') # final block counter
    
    for q in range(q_tmp,nblocks): # q_tmp is loaded from ndvi_dict_snow above
        
        print('processing chunk ' + str(q) + ' of ' + str(nblocks-1))
        # import images chunk    
        # variables = np.load('data/s2_ndvi_' + str(q) + '.npz')
        variables = np.load(DATA_PATH + '/s2_ndvi_' + str(q) + '_snow.npz')
        variables.allow_pickle=True
        locals().update(variables)
        del variables
        
        if np.ndim(im)==2:
            im = im[:,:,None]
        nim = np.shape(im)[2]
        
        # skip whole data chunk if there is no image belonging to the month list
        month = []
        for i in range(nim):
            month.append(datetime.strptime(str(im_date[i])[:-13],'%Y-%m-%d %H:%M:%S').month)    
        
        if not(np.any(np.in1d(month,m_list))): # exclude data outside the wanted month range
                continue   
        
        
        # # Shadow computation (parallelized)
        # num_cores = multiprocessing.cpu_count()-2
        # shad = Parallel(n_jobs=num_cores,backend="multiprocessing")(delayed(compute_shadow)(dem,lon,lat,im_date[i],tzone,dx)for i in range(len(im_date))) # parallelized loop
        
        # populate data dictionary
        for i in range(nim):
            date_tmp = datetime.strptime(str(im_date[i])[:-13],'%Y-%m-%d %H:%M:%S')    
            
            # skip image if there is no image belonging to the month list
            if not(np.in1d(date_tmp.month,m_list)):
                continue
            
            # skip image if there if it contains too few data
            # data_ind = np.logical_and(vt_ind,~np.isnan(im[:,:,i]))
            data_ind = ~np.isnan(im[:,:,i])
            df = np.sum(data_ind)/len(data_ind.ravel()) # data fraction in the image
            dth = 0.1 # 10% of miniumum data required
            if df < dth:
                continue
                        
            ndvi_dict['date'].append([date_tmp]*np.sum(data_ind))
            ndvi_dict['data_chunk_n'].append([q]*np.sum(data_ind))
            ndvi_dict['year'].append([date_tmp.year]*np.sum(data_ind))
            ndvi_dict['month'].append([date_tmp.month]*np.sum(data_ind))
            ndvi_dict['week'].append([date_tmp.isocalendar()[1]]*np.sum(data_ind))
            ndvi_dict['doy'].append([date_tmp.timetuple().tm_yday]*np.sum(data_ind))
            ndvi_dict['ndvi'].append(im[data_ind,i])
            ndvi_dict['x'].append(xx[data_ind])
            ndvi_dict['y'].append(yy[data_ind])
            
            
        # save data
        np.savez(DATA_PATH + '/ndvi_dict_snow.npz',
                 ndvi_dict = ndvi_dict,
                  q_tmp = q+1, # current data chunk to process, 0 at the beginning 
                  )
    

    # UNCOMMENT TO REMOVE THE ORIGINAL IMAGES AFTER PREPROCESSING
    # os.remove(DATA_PATH + '/s2_ndvi_' + str(q) + '_snow.npz')
    # os.remove(DATA_PATH + '/s2_data_' + str(q) + '_snow.npz')
    
  
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% ANALYSIS %%%%%%%%%%%%%%%%%%%%%%%%%%% 
    
    
    #%% LOAD DATA BACK FROM PREPROCESSING
    
    # data dictionary
    variables = np.load(DATA_PATH + '/ndvi_dict_snow.npz')
    variables.allow_pickle=True
    locals().update(variables)
    del variables
    os.remove(DATA_PATH + '/ndvi_dict_snow.npz')
    ndvi_dict = ndvi_dict.all()
    
    date = np.concatenate(ndvi_dict['date'])
    doy = np.concatenate(ndvi_dict['doy'])
    month = np.concatenate(ndvi_dict['month'])
    ndvi = np.concatenate(ndvi_dict['ndvi'])
    # shadow = np.concatenate(ndvi_dict['shadow'])
    week = np.concatenate(ndvi_dict['week'])
    year = np.concatenate(ndvi_dict['year'])
    # elevation = np.concatenate(ndvi_dict['elevation'])
    # aspect = np.concatenate(ndvi_dict['aspect'])
    x = np.concatenate(ndvi_dict['x'])
    y = np.concatenate(ndvi_dict['y'])
    dcn = np.concatenate(ndvi_dict['data_chunk_n'])
    
    
    #%% ANNUAL NDVI CURVE FOR DIFFERENT YEARS
    
    ### INPUT PARAMS ########################################################
    
    # years to generate the ndvi annual curves (one curve per year)
    selected_years = [YEAR] #np.arange(2016,2024) # chosen dry/wet years 
    time_res = 'doy' # time resolution of the data display, it can be 'doy' 'week' or 'month'
    
    # aspect filter
    aspect_filter = False
    alb = 90 # lower aspect boundary to select the ROI pixels
    aub = 270 # upper aspect boundary to select the ROI pixels, = 365 for no filter
    
    # elevation filter
    elevation_filter = False
    elb = 0 # elevation boundaries to select the ROI pixels, = 0 for no filter
    eub = 9999 # elevation boundaries to select the ROI pixels, = 9999 for no filter
    
    # shadow filter
    shadow_filter = False # if True consider only pixels outside the mountain shadow
    
    # dates to exclude from the images (ex. snowfall days), leave empty for no dates excluded
    date_filter = False # if True exclude the following date list from the analysis
    excl_dates = []
    # excl_dates = ["2016-01-04 10:24:32",
    #               "2016-04-19 10:10:32",
    #               "2017-10-24 10:21:11",
    #               "2017-09-21 10:10:21",
    #               "2017-08-20 10:20:19",
    #               "2017-10-09 10:20:09",
    #               "2017-07-31 10:20:19",
    #               "2019-09-09 10:20:29",
    #               "2020-09-30 10:07:29",
    #               "2020-10-18 10:20:41",
    #               "2019-10-16 10:10:29",
    #               "2020-10-08 10:20:31",
    #               "2020-10-10 10:10:29",
    #               "2020-10-28 10:21:41",
    #               "2020-10-30 10:11:39",
    #               "2021-10-08 10:18:29",
    #               "2022-09-18 10:17:01"]
    
    for i in range(len(excl_dates)):
        excl_dates[i] = dt.strptime(excl_dates[i],'%Y-%m-%d %H:%M:%S')
    
    ###############################################################################
    
    #%% COMPUTE THE CURVES
    
    # initialize output data dictionary
    out_dict = dict()
    
    # time vector with given resolution
    t_tmp = np.concatenate(ndvi_dict[time_res])
    
    # data filters
    data_ind = [True]*len(ndvi)
    # if shadow_filter == True:
    #     sun_ind = shadow==0 # sun indicator
    #     data_ind = np.logical_and(data_ind,sun_ind)
    # if aspect_filter == True:
    #     asp_ind = np.logical_and(aspect > alb, aspect < aub)
    #     data_ind = np.logical_and(data_ind,asp_ind)
    # if elevation_filter == True:
    #     elev_ind = np.logical_and(elevation > elb, elevation < eub)
    #     data_ind = np.logical_and(data_ind,elev_ind)
    if date_filter == True:
        excl_ind = ~np.in1d(date,excl_dates) 
        data_ind = np.logical_and(data_ind,excl_ind)
    
    
    fig,ax = plt.subplots()
    n = 0
    
    for i in range(len(selected_years)):
        
        n = n+1
        
        # current data selection
        tmp_year = selected_years[i]
        y_ind = np.in1d(year,tmp_year) # dry year indicator
        data_ind_tmp = np.logical_and(data_ind,y_ind)
        ndvi_data = ndvi[data_ind_tmp]
        ndvi_time = t_tmp[data_ind_tmp]
        
        
        # PLOT
        plt.subplot(1,1,n)
        
        # ndvi plot
        qtime,q25,qm,q75 = gtt.annual_plot(ndvi_time,ndvi_data,'tab:green', 'NDVI ' + str(tmp_year),time_res=time_res,f_range=[-0.1,1])
        out_dict['ndvi_time_' + str(tmp_year)] = qtime
        out_dict['ndvi_q25_' + str(tmp_year)] = q25
        out_dict['ndvi_median_' + str(tmp_year)] = qm
        out_dict['ndvi_q75_' + str(tmp_year)] = q75
        out_dict['ndvi_data_time_' + str(tmp_year)] = ndvi_time
        out_dict['ndvi_data_' + str(tmp_year)] = ndvi_data
        
        
        # greening time
        gt = gtt.sog(ndvi_time,ndvi_data,time_res='doy',ndvi_th = 0.05,pth=10,envelope=False)
        plt.plot([gt,gt],[-0.1,0.1],'--',c='tab:green')
        plt.text(gt+1,-0.1,s='SOG',c='tab:green',rotation = 'vertical')
        out_dict['SOG_' + str(tmp_year)] = gt
        
        # browning time
        bt = gtt.eos(ndvi_time,ndvi_data,time_res='doy',ndvi_th = 0.05,pth=10,envelope=False)
        plt.plot([bt,bt],[-0.1,0.1],'--',c='tab:brown')
        plt.text(bt+1,-0.1,s='EOS',c='tab:brown',rotation = 'vertical')
        out_dict['EOS_' + str(tmp_year)] = bt
        
        # area under the ndvi curve from greening time
        aucm = gtt.auc(ndvi_time,ndvi_data,time_res=time_res,envelope=False,sttt=gt,entt=365.25)
        plt.text(qtime[180],qm[180]/2,s='AUC = ' + str(aucm)[:5],c='tab:green')
        
        # ndvi growth slope fittend on growing season only (213-th DOY)
        fp,C = curve_fit(gtt.gomp,
                          qtime[:213],
                          qm[:213])#,
                          #bounds = ([0,0,0,-0.1],[1,200,10,+0.1]))
        
        sl = fp[2].copy()
        gom_time = np.arange(0,213)
        gom_data = gtt.gomp(gom_time,*fp)
        plt.plot(gom_time,gom_data,'--',c='tab:orange',label="Gompertz")
        out_dict['slope_param_' + str(tmp_year)] = sl
    
        # graph cosmetics
        plt.ylim([-0.15,1.1])
        plt.xlabel('Day of the year (DOY)')
        plt.ylabel('NDVI [-1,1]')
        
        plt.legend(loc='upper right',prop={'size': 11})
        
        plt.grid(axis='y',linestyle='--',alpha=0.5)
        plt.grid(axis='x',linestyle='--',alpha=0.5)
    
    # END LOOP
    plt.tight_layout()
    plt.savefig(OUT_PATH + '/' + SITE_NAME + '_' + str(YEAR) + '.pdf')
    
    # SAVE DATA AS NPZ ARCHIVE TO PRESERVE THE DICTIONARY STRUCTURE
    fname = OUT_PATH + '/' + SITE_NAME + '_' + str(YEAR) + '.npz'
    np.savez(fname,
              out_dict = out_dict,
              )


# # %% INTERACTIVE NDVI CURVE TO VISUALIZE IMAGES BY CLICKING THE NDVI CURVE


# ### params
# time_res= 'doy' #  year, month, week, doy
# year_tmp = 2016 # to group years: np.array([[2016,2017],[2022,2023]])
# ###

# #sun_ind = shadow==0 # sun indicator
# #asp_ind = np.logical_and(aspect > alb, aspect < aub)

# f=plt.figure(figsize=[17.03,  6.8 ]) # f.get_size_inches()
# data_ind = np.in1d(year,year_tmp) # unit and year indicators
# #data_ind = np.logical_and(data_ind,unit==vt_tmp) # unit and year indicators
# #data_ind = np.logical_and(data_ind,shadow==0) # sun indicator
# #data_ind = np.logical_and(data_ind,asp_ind)
# #data_ind = np.logical_and(data_ind,excl_ind)
# ndvi_sun = ndvi[data_ind]
# time_sun = doy[data_ind]
# dcn_sun = dcn[data_ind]
# date_sun = date[data_ind]

# # plot
# h = plt.subplot(1,3,1)
# gtt.annual_plot(time_sun,ndvi_sun,'r','sun',time_res='doy')
# #plt.title(verb_labels[vt_list==vt_tmp])
# if time_res == 'month':
#     plt.xticks(ticks=np.arange(3,11),labels=['Mar','Apr','May','Jun','Jul','Aug','Sep','Oct'])
# plt.ylim([-0.15,1.1])
# plt.xlabel(time_res + ' number')#plt.xlabel('week number')
# plt.ylabel('NDVI [-1,1]')
# #f.text(0.05,0.97,str(i),fontsize=16,weight='bold')        
# plt.tight_layout()
# global h2
# h2 = plt.subplot(1,3,2)
# pcm = h2.imshow(np.empty([10,10])*np.nan,vmin=-0.1,vmax=0.8)
# h2.set_xticks([],[])
# h2.set_yticks([],[])
# #plt.colorbar(pcm,ax=h2,orientation='horizontal')
# global h3
# h3 = plt.subplot(1,3,3)
# pcm = h3.imshow(np.empty([10,10])*np.nan,vmin=-0.1,vmax=0.8)
# h3.set_xticks([],[])
# h3.set_yticks([],[])
# #plt.colorbar(pcm,ax=h3)
    
# # DEFINE CALLBACK FUNCTIONS
# def onclick(event):
#     global ix, iy, ni 
#     ix, iy = event.xdata, event.ydata
#     dx=(ix-time_sun)/np.std(time_sun) 
#     dy=(iy-ndvi_sun)/np.std(ndvi_sun)
#     D=np.sqrt(dx**2+dy**2)
#     ind=np.argmin(D)
#     print((time_sun[ind],ndvi_sun[ind]))
    
#     # detect the point in the plot and map
#     h.scatter(time_sun[ind],ndvi_sun[ind],marker='+',s=100,c='k') 
#     plt.draw()
    
#     # load related data chunk

#     variables = np.load(DATA_PATH + '/s2_ndvi_' + str(dcn_sun[ind].astype('int')) + '_snow.npz')
#     variables.allow_pickle=True
#     globals().update(variables)
#     del variables
    
#     for j in range(len(im_date)):
#         print(j)
#         if date_sun[ind] == dt.strptime(str(im_date[j])[:19],'%Y-%m-%d %H:%M:%S'):
#             print('image found!')
#             print('point date ' + str(date_sun[ind]))
#             print('found date ' + str(im_date[j])[:19])
#             break
    
#     print('j = ' + str(j) + ' data_chunk = ' + str(dcn_sun[ind]))
#     #h2 = plt.subplot(1,2,2)
#     h2.clear()
#     plt.draw()
#     if im.ndim==2:
#         h2.imshow(np.squeeze(im),vmin=-0.1,vmax=0.8,interpolation='None')
#         #data_ind = np.logical_and(~np.isnan(im),~npvar2 = tk.IntVar().isnan(vt_rast))
#         #df = np.sum(data_ind)/np.sum(~np.isnan(vt_rast)) # data fraction in the image
#     else:
#         h2.imshow(np.squeeze(im[:,:,j]),vmin=-0.1,vmax=0.8,interpolation='None')
#         #data_ind = np.logical_and(~np.isnan(im[:,:,j]),~np.isnan(vt_rast))
#         #df = np.sum(data_ind)/np.sum(~np.isnan(vt_rast)) # data fraction in the image
#     h2.set_title('ndvi ' + str(im_date[j])) # + '\n data fraction = ' + str(df)))
#     plt.tight_layout()
#     plt.draw()
    
#     variables = np.load(DATA_PATH + '/s2_data_' + str(dcn_sun[ind].astype('int')) + '_snow.npz')
#     variables.allow_pickle=True
#     globals().update(variables)
#     del variables
    
#     h3.clear()
#     plt.draw()
#     if im.ndim==2:
#         h3.imshow(imrisc(np.squeeze(im[:,:,[2,1,0],:]),2,98),interpolation='None')
#     else:
#         h3.imshow(imrisc(np.squeeze(im[:,:,[2,1,0],j]),2,98),interpolation='None')
#     h3.set_title('RGB')
#     plt.tight_layout()
#     plt.draw()

# def onclose(event): # on figure close do..
#     # disconnect listening functions
#     f.canvas.mpl_disconnect(cid)
#     f.canvas.mpl_disconnect(cid2)
#     print('figure closed')

# # CONNECT LISTENING FUNCTIONS TO FIGURE
# cid = f.canvas.mpl_connect('button_press_event', onclick)
# cid2 = f.canvas.mpl_connect('close_event', onclose)