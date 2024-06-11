#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SIMPLE SCRIPT TO DOWNLOAD SENTINEL IMAGES ON A COMMON GRID AND EXTRACT THE NDVI CURVE

import numpy as np
import matplotlib.pyplot as plt
font = {'family' : 'DejaVu Sans',
        'size'   : 12}
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
import pandas as pd
import geopandas
import os
import greentrack_tools as gtt
import zarr

# STAC PROTOCOL SETTINGS: set to False to use a local data archive
Settings = get_settings()
Settings.USE_STAC = True

#%% 0)  INPUT PARAMS  ###########################################################

# giving SITE_NAME a different name every loop iteration
SITE_NAME = 'test' # base name for output files and folder

# shapefile path of the ROI (.gpkg or .shp)
shp_path = 'data/test_perimeter.gpkg'

# if True get data for the whole bounding box (no shape mask)
USE_BBOX = False

# switch to start from the last downloaded QUERY CHUNK 
# (ex. in case of interrupted download for server error) 
RESUME_DOWNLOAD = False

# list  of years you want the data for, can also contain one year
year_list = [2020]
# year_list = np.arange(2018,2024) # multiple years

# local path where output directory and files are saved
SAVE_DIR = 'export'  

# coudy pixel percentage to discard sentinel tiles
CLOUD_TH = 30

# target image resolution: the unit must be compatible with the shapefile crs
# usually meters if planar (ex. EPGS:2025 LV95 for Switzerland) or degrees if geodetic (ex. WGS84)
res = 10

# Number of threads used in parallel operations
njobs = 2

# Number of pixel-cunk size to process in every jobs
# every pixel array of sze chunk-size x chunk-size will be stored in the same 
# memory chunk and processed linearly by a cpu

chunk_size = 20 # here 20x20 pixel chunks are stored and processed

# sentinel product name
S_NAME = 'sentinel2-msi'

# spectral indicator to compute ('EVI' or 'NDVI')
SIND_NAME = 'EVI'

# BANDS TO SELECT
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


# EODAL PREPROCESSING FUNCTION

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

#%% 1) TARGET GRID AND IMPORT POLYGONS #######################################

# target grid vectors (accepts multipolygons or exploded shapefiles)
tx, ty = gtt.make_grid_vec(shp_path,res)

# read ROI shape and extract object (parcerls) ids
poly = gpd.read_file(shp_path)
pid_list = np.unique(poly['OBJECTID'])
print('Rasterizing polygons to target grid...')
pid_rast = gtt.rasterize_shp(tx,ty,shp_path,'OBJECTID',no_data=0,njobs=njobs)

# mask for original polygons to apply later
mask = pid_rast > 0
plt.imshow(mask)   

# if ROI is composed by multiple polygons make a multipolygon 
if  len(poly) > 1:
    
     # if mutlipolygon, use the convex hull as mono-polygon ROI and apply mask later 
     gs = poly.buffer(0) # make all geometries valid
     mpoly = gpd.GeoDataFrame(geometry=gs) # recreate goedataframe from geoseries
     mpoly.dissolve().to_file('data/dissolved.shp') # dissolve and save to temp file
     mpoly_path = 'data/dissolved.shp'

# write bbox if used as ROI
if USE_BBOX == True:
    
    # USE BBOX AS ROI
    out_base_path = 'data/bbox'
    gtt.write_bbox(mpoly_path,out_base_path)
    mpoly_path = out_base_path + '.shp'
    
# mask for original polygons to apply later
mask = pid_rast > 0
plt.imshow(mask)    
    
#%% 2)  DOWNLOAD THE IMAGES ##################################################

#% EODAL QUERY PARAMETERS
   
# user-inputs
# -------------------------- Collection -------------------------------
collection: str = S_NAME

# ---------------------- Spatial Feature  ------------------------------
geom: Path = Path(mpoly_path) # geometry for the query
feature = Feature.from_geoseries(gpd.read_file(geom).geometry) # as feature

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

# LOOP OVER YEARS

for k in range(len(year_list)):
    
    # DEFINE TIME RANGE
    YEAR = year_list[k] # current year
    
    print('#### Extracting data for site ' + SITE_NAME + ' year ' + str(YEAR))
    
    time_start: datetime = datetime(YEAR,1,1)  		# year, month, day (incl.)
    time_end: datetime = datetime(YEAR,12,31)     	# year, month, day (incl.)
    
    # DEFINE QUERY CHUNK SERIES
    # split the wanted date range in approx 30-day chuncks to override server 
    # download limit
    
    CHUNK_SIZE = 30 # size in days
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
    
    # CREATE OUTPUT PATHS
    
    # path where results are saved
    OUT_PATH = SAVE_DIR + '/' + SITE_NAME + '_' + str(YEAR)
    if not os.path.exists(OUT_PATH):
        os.makedirs(OUT_PATH)
    
    # path where sat images and temp data are saved
    DATA_PATH = 'data/' + SITE_NAME + '_' + str(YEAR)
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
    
    # DOWNLOAD DATA    
    
    if RESUME_DOWNLOAD == False: # if not used, clean old data
        
        # delete old downloaded files
        target_dir = DATA_PATH
        target_pattern = 's2'
        gtt.purge(target_dir, target_pattern)    
        
        # SET COUNTERS TO ZERO
        n = 0 # query counter
        np.save(DATA_PATH + '/counter.npy',n)
        
        # saved data chunk counter 
        # (different from previous because empty chunks are skipped)
        n_block = 0 
        np.save(DATA_PATH + '/block_counter.npy',n_block)
        
        # INIZIALIZE EMPTY ANNUAL DATA CUBE WITH METADATA
        zarr_path = DATA_PATH + '/data_group.zarr'
        crs = poly.crs.to_epsg()
        gtt.init_data_cube(zarr_path, 
                           tx, 
                           ty, 
                           BAND_LIST,
                           res,
                           crs,
                           pid_rast,
                           SIND_NAME,
                           chunk_size=chunk_size)
    
    # RESUME COUNTERS
    n = np.load(DATA_PATH + '/counter.npy') # query counter
    n_block = np.load(DATA_PATH + '/block_counter.npy') # saved data chunk
    
    # loop over data chunks
    for i in range(n,len(date_vec)-1):

        data_fname = DATA_PATH + '/s2_data_' + str(n_block) + '.npz'
        
        print('DOWNLOADING DATA CHUNK '  + str(i) + ' of ' + str(len(date_vec)-2))
        
        # eodal mapper configuration
        mapper_configs = MapperConfigs(
             collection = collection,
             time_start = date_vec[i],
             time_end = date_vec[i+1],
             feature = feature,
             metadata_filters = metadata_filters
         )
    
    
        # create mapper
        mapper = Mapper(mapper_configs)
       
        # send the query
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
        
        # warn of no images are found
        if mapper.data is None:
            
            print('No images found, continuing to the next data chunk')
            n = n+1 # update counter
            np.save(DATA_PATH + '/counter.npy',n)
            
            continue # skip this data chunk download
        
        # reproject scene to ROI CRS
        for _, scene in mapper.data:
            
            scene.reproject(inplace=True, target_crs=poly.crs.to_epsg())
            
        # retrieve band names
        sc = mapper.data
        bands = sc[sc.timestamps[0]].band_names # bands
        
### 3) CREATE THE IMAGE AND INDICATOR CUBES AND SAVE THEM ####################

        # if any data is present
        if not mapper.data.empty:
            
            # project scenes on the target grid
            im = gtt.scene_to_array(mapper.data,tx,ty)

            if np.any(im!=0): # if any data non-zero data is present
            
                # date vector
                im_date = pd.to_datetime(mapper.metadata['sensing_time'])
                
                # cloud percentage vector
                im_cloud_perc = mapper.metadata['cloudy_pixel_percentage']
                
                # set zeros to nan
                im[im==0] = np.nan
                
                # COMPUTE INDICATOR
                
                # red, nir, blue bands for all scenes together
                RED = np.squeeze(im[:,:,np.array(bands)==['B04'],:])
                NIR = np.squeeze(im[:,:,np.array(bands)==['B08'],:])
                BLUE =  np.squeeze(im[:,:,np.array(bands)==['B02'],:])
                
                # indicators for all scenes together
                #NDVI = gtt.ndvi(red=RED,nir=NIR)
                EVI = gtt.evi(blue=BLUE,red=RED,nir=NIR)
                
                if len(EVI.shape)==2:
                    EVI = np.expand_dims(EVI,axis=2)
                
                # APPEND INDICATOR TO DATA CUBE
                gtt.append_data_cube(zarr_path,
                         im_date = im_date, # dates vector
                         im_cloud_perc = im_cloud_perc, # cloud pecentage vector
                         im = im, # images array [x,y,band,scene]
                         sind = EVI,
                         )
                
                # UPDATE SAVED DATA BLOCK COUNTER
                n_block = n_block + 1
                np.save(DATA_PATH + '/block_counter.npy',n_block)
                
                # DELETE DATA IN MEMORY
                del im, mapper, RED, NIR
        
        # UPDATE QUERY COUNTER
        n = n+1 
        np.save(DATA_PATH + '/counter.npy',n)

    # IF ALL PIXELS ARE ZERO IN ALL IMAGES, NO IMAGE IS SAVED
    if n_block == 0:
       raise Exception("Only cloudy or zero pixels for the whole ROI, no data saved!")
    
#%% 4) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% ANALYSIS %%%%%%%%%%%%%%%%%%%%%%

for k in range(len(year_list)):
    
    YEAR = year_list[k] # current year

    # PATHS
    # path where results are saved
    OUT_PATH = SAVE_DIR + '/' + SITE_NAME + '_' + str(YEAR)
    if not os.path.exists(OUT_PATH):
        os.makedirs(OUT_PATH)
    
    # path where sat images and temp data are saved
    DATA_PATH = 'data/' + SITE_NAME + '_' + str(YEAR)
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
    
    # data cube path
    zarr_path = DATA_PATH + '/data_group.zarr'
    
    # IMPORT INDICATOR DATA
    data_cube = zarr.open(zarr_path + '/sind', mode='r')
    time_vec = np.array(zarr.open(zarr_path + '/im_date', mode='r'))
    time_vec = pd.to_datetime(time_vec).day_of_year
    
    # INDICATOR MAP - PARALLEL COMPUTATION 
    print('Computing indicator map...')    
    auc_rast = gtt.indicator_map(data_cube,time_vec,njobs,chunk_size)
    
    # PLOT MAP
    plt.figure()
    plt.imshow(auc_rast)
    OUT_PATH + '/auc_rast.zarr'
    
#### 5) SAVE MAP TO PERSISTENT ZARR FILE #####################################
    persistent_store = zarr.DirectoryStore(OUT_PATH + '/auc_rast.zarr')
    persistent_array = zarr.zeros_like(auc_rast, store=persistent_store, overwrite=True)
    persistent_array[:] = auc_rast[:] # zarr file automatically updated
    
    # # TO OPEN THE SAVED MAP
    # auc_rast = zarr.open(OUT_PATH + '/auc_rast.zarr')
    # plt.figure()
    # plt.imshow(auc_rast)
    

    
    