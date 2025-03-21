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
#import pandas as pd
import geopandas
import os
from datetime import datetime as dt 
from scipy.optimize import curve_fit
import greentrack_tools as gtt

# STAC PROTOCOL SETTINGS: set to False to use a local data archive
Settings = get_settings()
Settings.USE_STAC = True

#%%  INPUT PARAMS  ###########################################################


## HERE you can create a loop to run all following code for a list of sites (parcels)
# giving SITE_NAME a different name every loop iteration

SITE_NAME = 'test' # base name for output files and folder

# shapefile path of the ROI (.gpkg or .shp)
shp_path = '/home/orianif/GEO/software/greentrack/data/parcels__posieux_5.gpkg'
bbox_fname = shp_path

# list  of years you want the data for, can also contain one year
year_list = [2022]

# local path where output directory and files are saved
SAVE_DIR = 'export'  

# coudy pixel percentage to discard sentinel tiles
CLOUD_TH = 30

# target image resolution: the unit must be compatible with the shapefile crs
# usually meters if planar (ex. EPGS:2025 LV95 for Switzerland) or degrees if geodetic (ex. WGS84)
res = 10

# sentinel product name
S_NAME = 'sentinel2-msi'

# bands to select (list)
# See available bands for sentinel-2 L2A here:
# https://docs.sentinel-hub.com/api/latest/data/sentinel-2-l2a/
# SCL (Scene Classification Layer) is added as extra band

BAND_LIST = [
# 'B01',
'B02', # BLUE
'B03', # GREEN
'B04', # RED
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
 
REUSE_DATA = False # if True, missing data will be freshly downloaded

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
    
    # resample scene (necessary for uniform all bands resolution)
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

#%% TARGET GRID AND TREAT MULTI POLYGON ROI

# target grid (accepts exploded shapefiles)
tx, ty = gtt.make_grid_vec(bbox_fname,res)

# if ROI is multi-polygons make a multipolygon 
poly = gpd.read_file(bbox_fname)

if  len(poly) > 1:
    
     # if mutlipolygon, use the convex hull as mono-polygon ROI and apply mask later 
     poly.dissolve().to_file('data/dissolved.shp')
     bbox_fname = 'data/dissolved.shp'
    
# mask for original polygons to apply later
mask = gtt.rasterize_shp(tx,ty,shp_path,'1',no_data=0, njobs = 1, target_crs=poly.crs.to_epsg())
    
    
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
        'scene_modifier_kwargs': {'target_resolution': 10} # keep standard 10m res here
    }
      

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
        np.save(DATA_PATH + '/block_counter.npy',n_block)
        
    n = np.load(DATA_PATH + '/counter.npy') # counter to resume from last completed chunk
    n_block = np.load(DATA_PATH + '/block_counter.npy') # counter to resume from last completed chunk
    
    for i in range(n,len(date_vec)-1):
        
        data_fname = DATA_PATH + '/s2_data_' + str(n_block) + '.npz'
        
        if REUSE_DATA == False or os.path.exists(data_fname) == False:
        
            print('DOWNLOADING DATA CHUNK '  + str(i) + ' of ' + str(len(date_vec)-2))
            
            feature = Feature.from_geoseries(gpd.read_file(geom).geometry)
        
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
                
                # project scenes on the target grid
                im = gtt.scene_to_array(mapper.data,tx,ty,mask=mask)

                if np.any(im!=0): # if any data is non zero
                    
                    im_date = mapper.metadata['sensing_time']
                    im_cloud_perc = mapper.metadata['cloudy_pixel_percentage']
                    
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
                    
                    # compute NDVI and EVI, YOU CAN PUT HERE OTHER INDICES
                    # AND ADD THEM TO THE SAVED IMAGES
                    
                    # red, nir, blue bands for all scenes together
                    RED = np.squeeze(im[:,:,np.array(bands)==['B04'],:])
                    NIR = np.squeeze(im[:,:,np.array(bands)==['B08'],:])
                    BLUE =  np.squeeze(im[:,:,np.array(bands)==['B02'],:])
                    
                    # indicators for all scenes together
                    NDVI = gtt.ndvi(red=RED,nir=NIR)
                    EVI = gtt.evi(blue=BLUE,red=RED,nir=NIR)
                    
                    # SAVE indicator images as .npz, ADD HERE NEW INDICES                
                    np.savez(DATA_PATH + '/s2_indicators_' + str(n_block) + '.npz',        
                             im_date = im_date, # dates vector
                             im_cloud_perc = im_cloud_perc, # cloud pecentage vector
                             NDVI = NDVI, # NDVI array [x,y,scene]
                             EVI = EVI, # EVI array [x,y,scene]
                             tx = tx, # x coord vector
                             ty = ty,  # y coord vector
                             shp = shp # roi shapefile
                             )
                    
                    # update data block counter
                    n_block = n_block + 1
                    
                    # save block counter
                    np.save(DATA_PATH + '/block_counter.npy',n_block)
                    
                    del im, mapper, NDVI, RED, NIR
                
        n = n+1 # update counter
        np.save(DATA_PATH + '/counter.npy',n)

    # IF ALL PIXELS ARE ZERO IN ALL IMAGES, NO IMAGE IS SAVED
    if n_block == 0:
       raise Exception("All found images contain only cloudy or zero (no data) pixels. Try changing the ROI polygon, the cloud-filter parameter, or the target year")
                 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% PREPROCESSING %%%%%%%%%%%%%%%%%%%%%%%%

    #% DEFINE TARGET GRID
    xx, yy = np.meshgrid(tx,ty,indexing='xy')
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
              'evi', # EVI pixel value
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
    pixel_dict = dict.fromkeys(mykeys)
    
    for i in pixel_dict.keys():
          pixel_dict[i] = list([])


    #%% EXTRACT NDVI FROM DATA CHUNKS, 
    # resuming from latest data chunk processed, q_tmp
    
    plt.close('all')
    
    # INITIALIZE SAVED DATA (COMMENT TO RESUME PREVIOUS JOB)
    np.savez(DATA_PATH + '/pixel_dict.npz',
              pixel_dict = pixel_dict,
              q_tmp = 0, # current data chunk to process, 0 at the beginning 
              )
    
    # RESUME SAVED DATA
    q_tmp = []
    variables = np.load(DATA_PATH + '/pixel_dict.npz')
    variables.allow_pickle=True
    locals().update(variables)
    del variables
    pixel_dict = pixel_dict.all()
    
    # vt_ind = np.in1d(vt_rast,vt_list).reshape(np.shape(vt_rast))
    nblocks = np.load(DATA_PATH + '/block_counter.npy') # final block counter
    
    for q in range(q_tmp,nblocks): # q_tmp is loaded from pixel_dict above
        
        print('processing chunk ' + str(q) + ' of ' + str(nblocks-1))
        # import images chunk    
        # variables = np.load('data/s2_ndvi_' + str(q) + '.npz')
        variables = np.load(DATA_PATH + '/s2_indicators_' + str(q) + '.npz')
        variables.allow_pickle=True
        locals().update(variables)
        del variables
        
        if np.ndim(NDVI)==2:
            NDVI = NDVI[:,:,None]
            EVI = EVI[:,:,None]
        nim = np.shape(NDVI)[2]
        
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
            data_ind = ~np.isnan(NDVI[:,:,i])
            #df = np.sum(data_ind)/len(data_ind.ravel()) # data fraction in the image
            df = np.sum(data_ind)/np.sum(mask) # data fraction in the image
            dth = 0.1 # 10% of miniumum data required
            if df < dth:
                continue
                        
            pixel_dict['date'].append([date_tmp]*np.sum(data_ind))
            pixel_dict['data_chunk_n'].append([q]*np.sum(data_ind))
            pixel_dict['year'].append([date_tmp.year]*np.sum(data_ind))
            pixel_dict['month'].append([date_tmp.month]*np.sum(data_ind))
            pixel_dict['week'].append([date_tmp.isocalendar()[1]]*np.sum(data_ind))
            pixel_dict['doy'].append([date_tmp.timetuple().tm_yday]*np.sum(data_ind))
            pixel_dict['ndvi'].append(NDVI[data_ind,i])
            pixel_dict['evi'].append(EVI[data_ind,i])
            pixel_dict['x'].append(xx[data_ind])
            pixel_dict['y'].append(yy[data_ind])
            
            
        # save data
        np.savez(DATA_PATH + '/pixel_dict.npz',
                 pixel_dict = pixel_dict,
                  q_tmp = q+1, # current data chunk to process, 0 at the beginning 
                  )
    

    # UNCOMMENT TO REMOVE THE ORIGINAL IMAGES AFTER PREPROCESSING
    # os.remove(DATA_PATH + '/s2_ndvi_' + str(q) + '.npz')
    # os.remove(DATA_PATH + '/s2_data_' + str(q) + '.npz')
    
  
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% ANALYSIS %%%%%%%%%%%%%%%%%%%%%%%%%%% 
    
    # LOAD DATA BACK FROM PREPROCESSING
    
    # data dictionary
    variables = np.load(DATA_PATH + '/pixel_dict.npz')
    variables.allow_pickle=True
    locals().update(variables)
    del variables
    pixel_dict = pixel_dict.all()
    
    date = np.concatenate(pixel_dict['date'])
    doy = np.concatenate(pixel_dict['doy'])
    month = np.concatenate(pixel_dict['month'])
    ndvi = np.concatenate(pixel_dict['ndvi'])
    evi = np.concatenate(pixel_dict['evi'])
    # shadow = np.concatenate(pixel_dict['shadow'])
    week = np.concatenate(pixel_dict['week'])
    year = np.concatenate(pixel_dict['year'])
    # elevation = np.concatenate(pixel_dict['elevation'])
    # aspect = np.concatenate(pixel_dict['aspect'])
    x = np.concatenate(pixel_dict['x'])
    y = np.concatenate(pixel_dict['y'])
    dcn = np.concatenate(pixel_dict['data_chunk_n'])
    
    
    #%% ANNUAL NDVI CURVE FOR DIFFERENT YEARS
        
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
    
    # example
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
    t_tmp = np.concatenate(pixel_dict[time_res])
    
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
        evi_data = evi[data_ind_tmp]
        evi_time = t_tmp[data_ind_tmp]
        
        # PLOT
        plt.subplot(1,1,n)

        ###### NDVI PLOT AND STATS
        
        # ndvi plot
        qtime,q25,qm,q75 = gtt.annual_plot(ndvi_time,ndvi_data,'tab:green', 'NDVI ' + str(tmp_year),time_res=time_res,f_range=[-0.1,1])
        out_dict['ndvi_time_' + str(tmp_year)] = qtime
        out_dict['ndvi_q25_' + str(tmp_year)] = q25
        out_dict['ndvi_median_' + str(tmp_year)] = qm
        out_dict['ndvi_q75_' + str(tmp_year)] = q75
        out_dict['ndvi_data_time_' + str(tmp_year)] = ndvi_time
        out_dict['ndvi_data_' + str(tmp_year)] = ndvi_data
        
        # SOG
        gt = gtt.sog(ndvi_time,ndvi_data,time_res='doy',ndvi_th = 0.05,pth=10,envelope=False)
        plt.plot([gt,gt],[-0.1,0.1],'--',c='tab:green')
        plt.text(gt+1,-0.1,s='SOG',c='tab:green',rotation = 'vertical')
        out_dict['ndvi_SOG_' + str(tmp_year)] = gt
        
        # EOS
        bt = gtt.eos(ndvi_time,ndvi_data,time_res='doy',ndvi_th = 0.05,pth=10,envelope=False)
        plt.plot([bt,bt],[-0.1,0.1],'--',c='tab:brown')
        plt.text(bt+1,-0.1,s='EOS',c='tab:brown',rotation = 'vertical')
        out_dict['ndvi_EOS_' + str(tmp_year)] = bt
        
        # area under the ndvi curve from SOG
        aucm = gtt.auc(ndvi_time,ndvi_data,time_res=time_res,envelope=False,sttt=gt,entt=365.25)
        plt.text(qtime[180],qm[180]/2,s='AUC = ' + str(aucm)[:5],c='tab:green')
        out_dict['ndvi_AUC_' + str(tmp_year)] = aucm
        
        # ndvi growth slope fittend on growing season only (213-th DOY)
        fp,C = curve_fit(gtt.gomp,
                          qtime[:213],
                          qm[:213],
                          maxfev=100000,
                          bounds = ([0,0,0,0],[2,360,1,1]))
        
        sl = fp[2].copy()
        gom_time = np.arange(0,213)
        gom_data = gtt.gomp(gom_time,*fp)
        plt.plot(gom_time,gom_data,'--',c='tab:orange',label="Gompertz")
        out_dict['ndvi_slope' + str(tmp_year)] = sl
        
        ###### EVI PLOT AND STATS
        
        # evi plot
        qtime,q25,qm,q75 = gtt.annual_plot(evi_time,evi_data,'tab:blue', 'EVI ' + str(tmp_year),time_res=time_res,f_range=[-0.1,1])
        out_dict['evi_time_' + str(tmp_year)] = qtime
        out_dict['evi_q25_' + str(tmp_year)] = q25
        out_dict['evi_median_' + str(tmp_year)] = qm
        out_dict['evi_q75_' + str(tmp_year)] = q75
        out_dict['evi_data_time_' + str(tmp_year)] = evi_time
        out_dict['evi_data_' + str(tmp_year)] = evi_data
        
        # SOG
        gt = gtt.sog(evi_time,evi_data,time_res='doy',ndvi_th = 0.05,pth=10,envelope=False)
        plt.plot([gt,gt],[-0.1,0.1],'--',c='tab:blue')
        plt.text(gt+1,-0.1,s='SOG',c='tab:blue',rotation = 'vertical')
        out_dict['evi_SOG_' + str(tmp_year)] = gt
        
        # EOS
        bt = gtt.eos(evi_time,evi_data,time_res='doy',ndvi_th = 0.05,pth=10,envelope=False)
        plt.plot([bt,bt],[-0.1,0.1],'--',c='tab:brown')
        plt.text(bt+1,-0.1,s='EOS',c='tab:brown',rotation = 'vertical')
        out_dict['evi_EOS_' + str(tmp_year)] = bt
        
        # area under the ndvi curve from SOG
        aucm = gtt.auc(evi_time,evi_data,time_res=time_res,envelope=False,sttt=gt,entt=365.25)
        plt.text(qtime[180],qm[180]/2,s='AUC = ' + str(aucm)[:5],c='tab:blue')
        
        # evi growth slope fittend on growing season only (213-th DOY)
        fp,C = curve_fit(gtt.gomp,
                          qtime[:213],
                          qm[:213],
                          maxfev=100000,
                          bounds = ([0,0,0,0],[2,360,1,1]))
        
        sl = fp[2].copy()
        gom_time = np.arange(0,213)
        gom_data = gtt.gomp(gom_time,*fp)
        plt.plot(gom_time,gom_data,'--',c='tab:purple',label="Gompertz")
        out_dict['slope_param_' + str(tmp_year)] = sl    
        
        # graph cosmetics
        plt.ylim([-0.15,1.1])
        plt.xlabel('Day of the year (DOY)')
        plt.ylabel('Spectral indicators [-1,1]')
        
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
