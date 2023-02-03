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


### Reading the Multiangle Polarimeter data ()

## Notes on the data
# i1 : Normalized Intensity - Telescope 1 (W/m2/um/sr).
# i2 : Normalized Intensity - Telescope 2 (W/m2/um/sr).
# q : Normalized Polarized Intensity (Q) (W/m2/um/sr).
# u : Normalized Polarized Intensity (U) (W/m2/um/sr).
# P : Degree of Linear Polarization (fraction).
# Chi : Polarization Azimuth (degrees).


# Reads the Data from ORACLES and gives the rslt dictionary for GRASP
def Read_Data_Oracles(file_path,file_name,PixNo,TelNo):


    f1_MAP = h5py.File(file_path + file_name,'r+')  #reading hdf5 file
    # print("Keys: %s" % f1_MAP.keys())
    Data = f1_MAP['Data'] #Reading the data

    #Reading the Geometry
    # f1_MAP['Geometry'].keys()
    Latitude = f1_MAP['Geometry']['Ground_Latitude']
    Longitude = f1_MAP['Geometry']['Ground_Longitude']

    Lat = Latitude[TelNo,:,:]
    Lon = Longitude[TelNo,:,:]
    Lat[Lat== -999.0] = np.nan
    Lon[Lon== -999.0] = np.nan

    Scattering_ang = f1_MAP['Geometry']['Scattering_Angle'][TelNo,PixNo,:]
    Solar_Zenith = f1_MAP['Geometry']['Solar_Zenith'][TelNo,PixNo,:]
    Solar_Azimuth = f1_MAP['Geometry']['Solar_Azimuth'][TelNo,PixNo,:]
    Viewing_Azimuth = f1_MAP['Geometry']['Viewing_Azimuth'][TelNo,PixNo,:]
    Viewing_Zenith = f1_MAP['Geometry']['Viewing_Zenith'][TelNo,PixNo,:]
    # Altitude = f1_MAP['Geometry']['Mapped_Altitudes']
    Terrain_Height =  f1_MAP['Geometry']['Terrain_Height']
    #f1_MAP['Geometry'].keys()
    y = np.ones((Solar_Azimuth.shape))*180  #correction for groud based to Grasp 
    Relative_Azi = np.ones((Solar_Azimuth.shape))*180 + Solar_Azimuth - Viewing_Azimuth
    #Variables
    pol_ang =  Data['Angle_of_Polarization']
    wl = Data['Wavelength']
    I = Data['Intensity_2'] #telescope 2
    Q = Data['Stokes_Q']
    U = Data['Stokes_U']
    DoLP = Data['DoLP']

    #Creating rslt dictionary for GRASP
    rslt ={}
    rslt['lambda'] = Data['Wavelength'][:]
    nwl = len(rslt['lambda']) #no of wavelengths
    
    # rslt['datetime'] = f1_MAP['Geometry']['Measurement_Time'][TelNo,PixNo,:]
    rslt['longitude'] = Lon[PixNo,:]
    rslt['latitude'] = Lat[PixNo,:]

    
    rslt['sza'] = np.repeat(Solar_Zenith, nwl).reshape(len(Solar_Zenith), nwl)
    rslt['vis']= np.repeat(Viewing_Zenith, nwl).reshape(len(Viewing_Zenith), nwl) 
    rslt['sca_ang']= np.repeat( Scattering_ang, nwl).reshape(len(Scattering_ang), nwl)  #Nangles x Nwavelengths
    rslt['fis'] = np.repeat(Relative_Azi , nwl).reshape(len(Relative_Azi ), nwl)
    rslt['meas_I']= I[PixNo,:,:]
    rslt['meas_Q']= Q[PixNo,:,:]
    rslt['meas_U']= U[PixNo,:,:]
    rslt['DoLP'] = DoLP[PixNo,:,:]

    return rslt

