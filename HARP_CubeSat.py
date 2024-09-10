#!/usr/bin/env python3

# Author: Nirandi Jayasinnghe
# Last Updated: Sep 3 2024

import numpy as np
from netCDF4 import Dataset 

# Project: Aerosol Retrievals using Multi-Angular Polarimeters in the Twilight Zone 
# Author: Nirandi Jayasinghe
# Reading HARP CubeSat Data into rslt dict to create SDATA
# Last Edited: Sep 3th 2024
"""
ang: (1,nwl) shaped variable which give angle ranges for each wavelength if necessary
CMask: give path to CMASK array for corresponding HARP CubeSat data file. 
       This variable will later be converted as Cloud Masking command
"""

file_path = '/data/home/njayasinghe/LES/GSFC-GRASP-Python-Interface/CubeSat/'
filename = 'HARP.20200730T112721.L1C.4.4KM.0719.V01h.nc'

def Read_HARP_CubeSat(file_path,filename,XPX, YPX, nwl, ang=None, CMask=None):

    ds = Dataset(file_path+filename, 'r')
    blue = {}
    green = {}
    nir = {}
    red = {}
    coordinates = {}

    blue = read_group(ds,blue,'blue')
    green = read_group(ds,green,'green')
    nir = read_group(ds,nir,'nir')
    red = read_group(ds,red,'red')
    coordinates =read_group(ds,coordinates,'Coordinates')

    Lat = coordinates['Latitude'][XPX,YPX]
    Lon = coordinates['Longitude'][XPX,YPX]

    if coordinates['LandMask'][XPX,YPX] == 1: 
        surf = 'land'
        print("Land pixel")
    else: 
        print("ocean pixel")
        surf = 'ocean'

    DT = filename.split('.')[1]
    date = DT.split('T')[0]
    time = DT.split('T')[-1]

    yy = date[:4] ; mm = date[4:6] ; dd = date[-2:] 
    hh = time[:2] ; mi = time[2:4] ; s = time[-2:]; ms = 00 

    








def read_px(data,XPX,YPX,ang=None):

    if ang == None:
        I = np.where(data['I'][:,XPX,YPX].data != data['I'][:,XPX,YPX].fill_value, data['I'][:,XPX,YPX].data, np.nan )
        Q = np.where(data['Q'][:,XPX,YPX].data != data['Q'][:,XPX,YPX].fill_value, data['Q'][:,XPX,YPX].data, np.nan )
        U = np.where(data['U'][:,XPX,YPX].data != data['U'][:,XPX,YPX].fill_value, data['U'][:,XPX,YPX].data, np.nan)
        DOLP = np.where(data['DOLP'][:,XPX,YPX].data != data['DOLP'][:,XPX,YPX].fill_value, data['DOLP'][:,XPX,YPX].data, np.nan)

        sza = np.where(data['Solar_Zenith'][:,XPX,YPX].data != data['Solar_Zenith'][:,XPX,YPX].fill_value, data['Solar_Zenith'][:,XPX,YPX].data,np.nan)
        saa = np.where(data['Solar_Azimuth'][:,XPX,YPX].data != data['Solar_Azimuth'][:,XPX,YPX].fill_value, data['Solar_Azimuth'][:,XPX,YPX].data, np.nan)
        vza = np.where(data['View_Zenith'][:,XPX,YPX].data != data['View_Zenith'][:,XPX,YPX].fill_value, data['View_Zenith'][:,XPX,YPX].data, np.nan)
        vaa = np.where(data['View_Azimuth'][:,XPX,YPX].data != data['View_Azimuth'][:,XPX,YPX].fill_value, data['View_Azimuth'][:,XPX,YPX].data, np.nan)

    else:
        I = np.where(data['I'][ang[0]:ang[1],XPX,YPX].data != data['I'][ang[0]:ang[1],XPX,YPX].fill_value, data['I'][ang[0]:ang[1],XPX,YPX].data, np.nan)
        Q = np.where(data['Q'][ang[0]:ang[1],:,XPX,YPX].data != data['Q'][ang[0]:ang[1],:,XPX,YPX].fill_value, data['Q'][ang[0]:ang[1],:,XPX,YPX].data, np.nan)
        U = np.where(data['U'][ang[0]:ang[1],:,XPX,YPX].data != data['U'][ang[0]:ang[1],:,XPX,YPX].fill_value, data['U'][ang[0]:ang[1],:,XPX,YPX].data, np.nan)
        DOLP = np.where(data['DOLP'][ang[0]:ang[1],XPX,YPX].data != data['DOLP'][ang[0]:ang[1],XPX,YPX].fill_value, data['DOLP'][ang[0]:ang[1],XPX,YPX].data, np.nan)

        sza = np.where(data['Solar_Zenith'][ang[0]:ang[1],XPX,YPX].data != data['Solar_Zenith'][ang[0]:ang[1],XPX,YPX].fill_value, data['Solar_Zenith'][ang[0]:ang[1],XPX,YPX].data,)
        saa = np.where(data['Solar_Azimuth'][ang[0]:ang[1],XPX,YPX].data != data['Solar_Azimuth'][ang[0]:ang[1],XPX,YPX].fill_value, data['Solar_Azimuth'][ang[0]:ang[1],XPX,YPX].data, np.nan)
        vza = np.where(data['View_Zenith'][ang[0]:ang[1],XPX,YPX].data != data['View_Zenith'][ang[0]:ang[1],XPX,YPX].fill_value, data['View_Zenith'][ang[0]:ang[1],XPX,YPX].data, np.nan)
        vaa = np.where(data['View_Azimuth'][ang[0]:ang[1],XPX,YPX].data != data['View_Azimuth'][ang[0]:ang[1],XPX,YPX].fill_value, data['View_Azimuth'][ang[0]:ang[1],XPX,YPX].data, np.nan)

    return I, Q, U, DOLP, sza, saa, vza, vaa
    

def read_group(ds, dict, group):
    Var_names = ds.groups[group].variables.keys()

    for name in Var_names: 
        dict[name] = ds.groups[group][name][:]

    return dict
