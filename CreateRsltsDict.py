"""
# Greema Regmi, UMBC
# Date: Jan 31, 2023

This code reads Polarimetric data from the Campaigns and runs GRASP. This code was created to Validate the Aerosol retrivals performed using Non Spherical Kernels (Hexahedral from TAMU DUST 2020)

"""

import os
  
    
import numpy as np
from matplotlib import pyplot as plt
import datetime as dt
import h5py 
import juliandate as jd


### Reading the Multiangle Polarimeter data ()

# Reads the Data from ORACLES and gives the rslt dictionary for GRASP
def Read_Data_RSP_Oracles(file_path,file_name,PixNo,TelNo, nwl,ang1, ang): #PixNo = Index of the pixel, #nwl = wavelength index, :nwl will be taken
    #Readinh the hdf file
    
    f1_MAP = h5py.File(file_path + file_name,'r+')  #reading hdf5 file
    # print("Keys: %s" % f1_MAP.keys())
    Data = f1_MAP['Data'] #Reading the data
    if ang == None: ang= 152
    if ang1 == None: ang1= 0
    
    # ang = 152 #No fo angles
    #Reading the Geometry
    # f1_MAP['Geometry'].keys()
    Latitude = f1_MAP['Geometry']['Collocated_Latitude']
    Longitude = f1_MAP['Geometry']['Collocated_Longitude']

    Lat = Latitude[TelNo,:]
    Lon = Longitude[TelNo,:]
    # Lat[Lat== -999.0] = np.nan
    # Lon[Lon== -999.0] = np.nan

    Ang_corr = np.ones((f1_MAP['Geometry']['Solar_Zenith'][TelNo,PixNo,ang1:ang].shape))*180  #correction for groud based to Grasp 

    Scattering_ang = f1_MAP['Geometry']['Scattering_Angle'][TelNo,PixNo,ang1:ang]
    Solar_Zenith =  f1_MAP['Geometry']['Solar_Zenith'][TelNo,PixNo,ang1:ang]
    Solar_Azimuth = f1_MAP['Geometry']['Solar_Azimuth'][TelNo,PixNo,ang1:ang]
    Viewing_Azimuth = f1_MAP['Geometry']['Viewing_Azimuth'][TelNo,PixNo,ang1:ang]
    Viewing_Zenith = Ang_corr - f1_MAP['Geometry']['Viewing_Zenith'][TelNo,PixNo,ang1:ang] # Theta_v <90
    # Viewing_Zenith = f1_MAP['Geometry']['Viewing_Zenith'][TelNo,PixNo,:ang]
    
    
    Relative_Azi = Solar_Azimuth - Viewing_Azimuth
    for i in range (len(Relative_Azi)) : 
        if Relative_Azi[i]<0: Relative_Azi[i] = 360 + Relative_Azi[i]
    

    # Relative_Azi = np.ones((Solar_Azimuth.shape))*180 + Solar_Azimuth - Viewing_Azimuth
    #Variables
    pol_ang =  Data['Angle_of_Polarization']
    wl = Data['Wavelength']
    if nwl == None: nwl = len(Data['Wavelength'][:])
    
    I1 = (Data['Intensity_1'][:]) #telescope 1 Normalized intensity (unitless)
    I1[(I1<0) & (I1>2)] = np.nan  #there are some negative intesity values in the file
    
    I2 = Data['Intensity_2'][:]  #telescope 2
    I2[(I1<0) & (I1>2)] = np.nan  #there are some negative intesity values in the file

    I = (I1+I2)/2  
    I[(I<0) & (I>2)] = 0   # averaging over telescope 1 and telescope 2
    Q = Data['Stokes_Q']
    U = Data['Stokes_U']
    DoLP = Data['DoLP']

    #Creating rslt dictionary for GRASP
    rslt ={}
    rslt['lambda'] = Data['Wavelength'][:nwl]/1000 # Wavelengths in um
    #converting modified julian date to julain date and then to gregorian
    jdv = f1_MAP['Geometry']['Measurement_Time'][TelNo,PixNo,0]+ 2400000.5  #Taking the time stamp for first angle
    
    yy,mm,dd,hh,mi,s,ms = jd.to_gregorian(jdv)
    rslt['datetime'] = dt.datetime(yy,mm,dd,hh,mi,s,ms) #Coverts julian to datetime
    jdv = f1_MAP['Geometry']['Measurement_Time'][TelNo,PixNo,0]+ 2400000.5  #Taking the time stamp for first angle
    
    # rslt['datetime'] = dt.datetime.now()
    rslt['longitude'] = Lon[PixNo]
    rslt['latitude'] = Lat[PixNo]

    # All the geometry arrays should be 2D, (angle, wl)
    rslt['sza'] = np.repeat(Solar_Zenith, nwl).reshape(len(Solar_Zenith), nwl)
    rslt['vis']= np.repeat(Viewing_Zenith, nwl).reshape(len(Viewing_Zenith), nwl) 
    rslt['sca_ang']= np.repeat( Scattering_ang, nwl).reshape(len(Scattering_ang), nwl)  #Nangles x Nwavelengths
    rslt['fis'] = np.repeat(Relative_Azi , nwl).reshape(len(Relative_Azi ), nwl)
    
    rslt['meas_I']= I[PixNo,ang1:ang,:nwl]

    rslt['meas_Q']= Q[PixNo,ang1:ang,:nwl]
    rslt['meas_U']= U[PixNo,ang1:ang,:nwl]
    rslt['DoLP'] = DoLP[PixNo, ang1:ang,:nwl]
    rslt['DoLP'][rslt['DoLP']<0] == np.nan

    #height key should not be used for altitude,

    rslt['OBS_hght']= f1_MAP['Geometry']['Collocated_Altitude'][TelNo,PixNo] # height of pixel in m
    if  rslt['OBS_hght'] < 0:  #if colocated attitude is less than 0 then that is set to 0
        rslt['OBS_hght'] = 0
        print(f"The collocated height was { rslt['OBS_hght']}, OBS_hght was set to 0 ")

    return rslt
 
def Read_Data_HSRL_Oracles(file_path,file_name,PixNo,height):
    #Creating rslt dictionary for GRASP
    f1= h5py.File(file_path + file_name,'r+')  #reading hdf5 file   
    latitude = f1['Nav_Data']['gps_lat'][:]
    longitude = f1['Nav_Data']['gps_lon'][:]
    altitude = f1['Nav_Data']['gps_alt'][:]
        
    PixNo = 0
    height = 0

    HSRL = f1['DataProducts']
    #backscatter 
    bscatter = np.array((HSRL['1064_bsc'][:], HSRL['532_bsc'][:], HSRL['355_bsc'][:]))
    #extinction
    ext = np.array((HSRL['1064_ext'][:], HSRL['532_ext'][:], HSRL['355_ext'][:]))
    LidarRatio = ext/bscatter #The lidar ratio is defined as the ratio of the extinction-to-backscatter coefficient

    Depol_ratio = np.array((HSRL['1064_dep'],HSRL['532_dep'],HSRL['355_dep']))
    
    rslt = {}
    rslt['lambda'] = np.array([1064,532,355])/1000
    rslt['LidarRatio'] = LidarRatio[:,PixNo,height] # for 3 wl 
    rslt['LidarDepol']= Depol_ratio[:,PixNo,height] # for 3 wl 
    rslt['height']= altitude[:]
    rslt['latitude'] = latitude[:]
    rslt['longitude']= longitude[:]
    

    return rslt
