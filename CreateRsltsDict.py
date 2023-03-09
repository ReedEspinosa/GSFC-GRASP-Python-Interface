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

#The function Checks for Fill values or negative values and replaces them with nan. To check for negative values, set negative_check = True 
def checkFillVals(param, negative_check = None):
    
    param[:] = np.where(param[:] == -999, np.nan, param[:])
    
    if negative_check == True:
        param[:] = np.where(param[:] < 0 , np.nan, param[:])
        
    return param


### Reading the Multiangle Polarimeter data ()

# Reads the Data from ORACLES and gives the rslt dictionary for GRASP
def Read_Data_RSP_Oracles(file_path,file_name,PixNo,TelNo, nwl,ang1, ang): #PixNo = Index of the pixel, #nwl = wavelength index, :nwl will be taken
    
    #Reading the hdf file
    f1_MAP = h5py.File(file_path + file_name,'r+') 
    
    Data = f1_MAP['Data'] #Reading the data
    # if ang == None: ang= 152
    # if ang1 == None: ang1= 0
    
    #Variables
    wl = Data['Wavelength']
    if nwl == None: nwl = len(Data['Wavelength'][:])

    #Reading the Geometry
    Lat = f1_MAP['Geometry']['Collocated_Latitude'][TelNo,PixNo]
    Lon = f1_MAP['Geometry']['Collocated_Longitude'][TelNo,PixNo]


    Scattering_ang = checkFillVals(f1_MAP['Geometry']['Scattering_Angle'][TelNo,PixNo,ang1:ang])
    Solar_Zenith =  checkFillVals(f1_MAP['Geometry']['Solar_Zenith'][TelNo,PixNo,ang1:ang])
    
    #Converting sunlight azimuth to solar azimuth: 𝜃𝑠, 180- 𝜃𝑣 𝜙𝑠 = 𝜙𝑠 -180, 𝜙𝑣

    Solar_Azimuth = checkFillVals(f1_MAP['Geometry']['Solar_Azimuth'][TelNo,PixNo,ang1:ang]) - 180
    Viewing_Azimuth = checkFillVals(f1_MAP['Geometry']['Viewing_Azimuth'][TelNo,PixNo,ang1:ang])
    #Converting viewing zenith with respect to nadir to that wrt zenith
    
    Viewing_Zenith = 180 - checkFillVals(f1_MAP['Geometry']['Viewing_Zenith'][TelNo,PixNo,ang1:ang]) # Theta_v <90
            
#     sza =  np.radians(Solar_Zenith)
#     vza =   np.radians(Viewing_Zenith)
#     szi =  np.radians(Solar_Azimuth)
#     vzi =  np.radians(Viewing_Azimuth)


    #Caculating the relative azimuth using the scattering angle definition. This will assure that the range of relative azimuth is contained in 0-360 range
    # Relative_Azi = (180/np.pi)*(np.arccos((np.cos((Scattering_ang *np.pi)/180)  + np.cos(sza)*np.cos(vza))/(- np.sin(sza)*np.sin(vza)) ))

    Relative_Azi = Solar_Azimuth - Viewing_Azimuth
    for i in range (len(Relative_Azi)): 
        if Relative_Azi[i]<0 : Relative_Azi[i] =  Relative_Azi[i]+360
 
    
    
    I1 = checkFillVals(Data['Intensity_1'][PixNo,ang1:ang,:nwl],negative_check =True) #telescope 1 Normalized intensity (unitless)#there are some negative intesity values in the file
    I2 = checkFillVals(Data['Intensity_2'][PixNo,ang1:ang,:nwl],negative_check =True)  #telescope 2

    # Q and U in scattering plane 
    
    scat_sin = checkFillVals(f1_MAP['Geometry']['Sin_Rot_Scatt_Plane'][TelNo,PixNo,ang1:ang])
    scat_cos = checkFillVals(f1_MAP['Geometry']['Cos_Rot_Scatt_Plane'][TelNo,PixNo,ang1:ang])
    
    scat_sin1 = np.repeat(scat_sin, nwl).reshape(len(scat_sin), nwl)
    scat_cos1 = np.repeat(scat_cos, nwl).reshape(len(scat_cos), nwl)
    
    Q = checkFillVals(Data['Stokes_Q'][PixNo,ang1:ang,:nwl])
    U = checkFillVals(Data['Stokes_U'][PixNo,ang1:ang,:nwl])

   
    #Creating rslt dictionary for GRASP
    rslt ={}
    rslt['lambda'] = Data['Wavelength'][:nwl]/1000 # Wavelengths in um
    rslt['longitude'] = Lon
    rslt['latitude'] = Lat
    rslt['meas_I'] = (I1+I2)/2

    #Meridian plane 
    rslt['meas_Q'] = Q
    rslt['meas_U'] = U                               
    rslt['DoLP'] = checkFillVals(Data['DoLP'][PixNo,ang1:ang,:nwl],negative_check =True)
    
    #In Scattering plane 
    # rslt['meas_Q'] = U*scat_sin1 + Q*scat_cos1
    # rslt['meas_U'] = Q*scat_sin1 - U*scat_cos1
            
    #converting modified julian date to julain date and then to gregorian
    jdv = f1_MAP['Geometry']['Measurement_Time'][TelNo,PixNo,0]+ 2400000.5  #Taking the time stamp for first angle
    
    yy,mm,dd,hh,mi,s,ms = jd.to_gregorian(jdv)
    rslt['datetime'] = dt.datetime(yy,mm,dd,hh,mi,s,ms) #Coverts julian to datetime
    jdv = f1_MAP['Geometry']['Measurement_Time'][TelNo,PixNo,0]+ 2400000.5  #Taking the time stamp for first angle
    
    # rslt['datetime'] = dt.datetime.now()
    

    # All the geometry arrays should be 2D, (angle, wl)
    rslt['sza'] = np.repeat(Solar_Zenith, nwl).reshape(len(Solar_Zenith), nwl)
    rslt['vis']= np.repeat(Viewing_Zenith, nwl).reshape(len(Viewing_Zenith), nwl) 
    rslt['sca_ang']= np.repeat( Scattering_ang, nwl).reshape(len(Scattering_ang), nwl)  #Nangles x Nwavelengths
    rslt['fis'] = np.repeat(Relative_Azi , nwl).reshape(len(Relative_Azi ), nwl)
    
    # solar_distance = (f1_MAP['Platform']['Solar_Distance'][PixNo])**2 
    # const = solar_distance/(np.cos(np.radians(rslt['sza'])))
 

    #height key should not be used for altitude,
    rslt['OBS_hght']= f1_MAP['Platform']['Platform_Altitude'][PixNo] # height of pixel in m
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
    rslt['height']= altitude[PixNo]
    rslt['latitude'] = latitude[PixNo]
    rslt['longitude']= longitude[PixNo]
    # rslt['RangeLidar']=0

    rslt['datetime'] = dt.datetime.now()
    

    return rslt
