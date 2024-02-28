#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 11:50:12 2024

@author: Fabio Oriani, Agroscope, fabio.oriani@agroscope.admin.ch
"""



#%% INTERACTIVE NDVI CURVE TO VISUALIZE IMAGES BY CLICKING THE NDVI CURVE

import numpy as np
import matplotlib.pyplot as plt
font = {'family' : 'DejaVu Sans',
        'size'   : 14}
from matplotlib import rc
rc('font', **font)
from datetime import datetime as dt 
import greentrack_tools as gtt
import os

### params
SITE_NAME = "posieux"
time_res= 'doy' #  year, month, week, doy
YEAR = 2022 # to group years: np.array([[2016,2017],[2022,2023]])
###

DATA_PATH = 'data/' + SITE_NAME + '_' + str(YEAR)
if not os.path.exists(DATA_PATH):
    raise("data path not found, check the data folder name in data/")
    
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

#sun_ind = shadow==0 # sun indicator
#asp_ind = np.logical_and(aspect > alb, aspect < aub)

f=plt.figure(figsize=[17.03,  6.8 ]) # f.get_size_inches()
data_ind = np.in1d(year,YEAR) # unit and year indicators
#data_ind = np.logical_and(data_ind,unit==vt_tmp) # unit and year indicators
#data_ind = np.logical_and(data_ind,shadow==0) # sun indicator
#data_ind = np.logical_and(data_ind,asp_ind)
#data_ind = np.logical_and(data_ind,excl_ind)

indicator_data = ndvi[data_ind]
indicator_data_2 = evi[data_ind]
time_data = doy[data_ind]
dcn_data = dcn[data_ind]
date_data = date[data_ind]

# plot
h = plt.subplot(2,2,1)
gtt.annual_plot(time_data,indicator_data,'tab:green','NDVI ' + str(YEAR),time_res='doy')
gtt.annual_plot(time_data,indicator_data_2,'tab:blue','EVI ' + str(YEAR),time_res='doy')
plt.legend()
#plt.title(verb_labels[vt_list==vt_tmp])
if time_res == 'month':
    plt.xticks(ticks=np.arange(3,11),labels=['Mar','Apr','May','Jun','Jul','Aug','Sep','Oct'])
plt.ylim([-0.15,1.1])
plt.xlabel(time_res + ' number')#plt.xlabel('week number')
plt.ylabel('vegetation indices [-1,1]')
#f.text(0.05,0.97,str(i),fontsize=16,weight='bold')        
plt.tight_layout()

global h2
h2 = plt.subplot(2,2,2)
pcm = h2.imshow(np.empty([10,10])*np.nan,vmin=-0.1,vmax=0.8)
h2.set_xticks([],[])
h2.set_yticks([],[])
#plt.colorbar(pcm,ax=h2,orientation='horizontal')
global h3
h3 = plt.subplot(2,2,3)
pcm = h3.imshow(np.empty([10,10])*np.nan,vmin=-0.1,vmax=0.8)
h3.set_xticks([],[])
h3.set_yticks([],[])
#plt.colorbar(pcm,ax=h3)

global h4
h4 = plt.subplot(2,2,4)
pcm = h4.imshow(np.empty([10,10])*np.nan,vmin=-0.1,vmax=0.8)
h4.set_xticks([],[])
h4.set_yticks([],[])
#plt.colorbar(pcm,ax=h3)
    
# DEFINE CALLBACK FUNCTIONS
def onclick(event):
    
    global ix, iy, ni, p, p2
    ix, iy = event.xdata, event.ydata
    
    dx = abs(ix-time_data)#/np.std(time_data)
    #dx = np.hstack([dx,dx])    
    
    #dy1 = (iy-indicator_data)/np.std(indicator_data)
    #dy2 = (iy-indicator_data_2)/np.std(indicator_data_2)
    #dy = np.hstack([dy1,dy2])
    
    #D = np.sqrt(dx**2+dy**2)
    
    ind = np.argmin(dx)
    
    #x = np.hstack([time_data,time_data])
    #y = np.hstack([indicator_data,indicator_data_2])
    #print((x[ind],y[ind]))
    
    
    # detect the point in the plot and map
    if 'p' in globals():
        p[0].remove()
        p2.remove()
    #p = h.scatter(x[ind],y[ind],marker='+',s=100,c='r') 
    p = h.plot([time_data[ind],time_data[ind]],[0,1],'--r')
    plt.draw()
    
    # load related data chunk

    variables = np.load(DATA_PATH + '/s2_indicators_' + str(dcn_data[ind].astype('int')) + '.npz')
    variables.allow_pickle=True
    globals().update(variables)
    del variables
    
    for j in range(len(im_date)):
        print(j)
        if date_data[ind] == dt.strptime(str(im_date[j])[:19],'%Y-%m-%d %H:%M:%S'):
            print('image found!')
            print('point date ' + str(date_data[ind]))
            print('found date ' + str(im_date[j])[:19])
            break
    
    print('j = ' + str(j) + ' data_chunk = ' + str(dcn_data[ind]))
    #h2 = plt.subplot(1,2,2)
    h2.clear()
    plt.draw()
    if NDVI.ndim==2:
        h2.imshow(np.squeeze(NDVI),vmin=-0.1,vmax=0.8,interpolation='None')
        #data_ind = np.logical_and(~np.isnan(im),~npvar2 = tk.IntVar().isnan(vt_rast))
        #df = np.sum(data_ind)/np.sum(~np.isnan(vt_rast)) # data fraction in the image
    else:
        h2.imshow(np.squeeze(NDVI[:,:,j]),vmin=-0.1,vmax=0.8,interpolation='None')
        #data_ind = np.logical_and(~np.isnan(NDVI[:,:,j]),~np.isnan(vt_rast))
        #df = np.sum(data_ind)/np.sum(~np.isnan(vt_rast)) # data fraction in the image
    h2.set_title("NDVI") # + '\n data fraction = ' + str(df)))
    p2 = h.text(time_data[ind]+2,0.1,s=str(im_date[j])[:10],c='r')
    plt.tight_layout()
    plt.draw()
    
    h4.clear()
    plt.draw()
    if EVI.ndim==2:
        h4.imshow(np.squeeze(EVI),vmin=-0.1,vmax=0.8,interpolation='None')
        #data_ind = np.logical_and(~np.isnan(im),~npvar2 = tk.IntVar().isnan(vt_rast))
        #df = np.sum(data_ind)/np.sum(~np.isnan(vt_rast)) # data fraction in the image
    else:
        h4.imshow(np.squeeze(EVI[:,:,j]),vmin=-0.1,vmax=0.8,interpolation='None')
    #plt.colorbar()
        #data_ind = np.logical_and(~np.isnan(NDVI[:,:,j]),~np.isnan(vt_rast))
        #df = np.sum(data_ind)/np.sum(~np.isnan(vt_rast)) # data fraction in the image
    h4.set_title("EVI") # + '\n data fraction = ' + str(df)))
    plt.tight_layout()
    plt.draw()
    
    variables = np.load(DATA_PATH + '/s2_data_' + str(dcn_data[ind].astype('int')) + '.npz')
    variables.allow_pickle=True
    globals().update(variables)
    del variables
    
    h3.clear()
    plt.draw()
    if im.ndim==2:
        h3.imshow(gtt.imrisc(np.squeeze(im[:,:,[2,1,0]]),2,98),interpolation='None')
    else:
        h3.imshow(gtt.imrisc(np.squeeze(im[:,:,[2,1,0],j]),2,98),interpolation='None')
    h3.set_title('RGB')
    plt.tight_layout()
    plt.draw()

def onclose(event): # on figure close do..
    # disconnect listening functions
    f.canvas.mpl_disconnect(cid)
    f.canvas.mpl_disconnect(cid2)
    print('figure closed')

# CONNECT LISTENING FUNCTIONS TO FIGURE
cid = f.canvas.mpl_connect('button_press_event', onclick)
cid2 = f.canvas.mpl_connect('close_event', onclose)