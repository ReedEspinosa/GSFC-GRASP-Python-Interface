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
import itertools
from scipy.interpolate import interp1d
import netCDF4 as nc
from numpy import nanmean
import yaml
import pandas as pd
from netCDF4 import Dataset

#The function Checks for Fill values or negative values and replaces them with nan. To check for negative values, set negative_check = True 
def checkFillVals(param , negative_check = None):
    param[:] = np.where(param[:] == -999, np.nan, param[:])
    if negative_check == True:
        param[:] = np.where(param[:] < 0 , np.nan, param[:])
    return param
#Checks for negative values and replaces them by nan
def HSRL_checkFillVals(param):
    param[:] = np.where(param[:] < 0 , np.nan, param[:])     
    return param

'''  GasAbsFn: Format of file: .nc, Description: file containing the value of combined optical depth for different gases in the atmosphere using radiatiove tranfer code
     altIndex = Type: integer , Description: index of the vertical height at which we want to calculate the absorption, In this case we've taken the maximum altitude of RSP aircraft 
     SpecResFn = Format of file: .txt ,Description: file containing the  response funtion of the instruent at a particular wl (Wlname = interger, wavelength in the file name ) '''

def Interpolate_Tau(Wlname,GasAbsFn,altIndex,SpecResFn):
    #Gas Absorption correction using UNL_VRTM (provided by Richard,UMBC)
    #Reading the NetCDF file for gas absorption from radiative tranfer code (UNLVRTM)
    ds = nc.Dataset(GasAbsFn)
#     Tau_Comb = np.sum(ds.variables['tauGas'][altIndex,:]) #Bulk gas absorption for different layers
    Wl = ds.variables['Lamdas'][:] # wavelength values corresponding to the gas absorption
    Tau_Comb_solar=  np.sum(ds.variables['tauGas'], axis=0)
    Tau_Comb_view=  np.sum(ds.variables['tauGas'][:altIndex,:], axis=0)   
    #Spectral response values for given RSP wl
    SpecResFn = SpecResFn[SpecResFn[:,0]>= min(Wl)]
    #1D interpolation across wavelength
    f = interp1d(Wl,Tau_Comb_solar,kind = 'linear')
    f2 = interp1d(Wl,Tau_Comb_view,kind = 'linear')

    # Evaluate the function at a new point
    wl_RSP = SpecResFn[:,0]
    tau_solar = f(wl_RSP) #Tau at given RSP response function wl
    tau_view = f2(wl_RSP) #Tau at given RSP response function wl
     
    return tau_solar, tau_view, wl_RSP, SpecResFn[:,1]

#This function will return the Transmittance for all the solar and viewing geometries
def Abs_Correction(Solar_Zenith,Viewing_Zenith,Wlname,GasAbsFn,altIndex,SpecResFn):
    
    intp =Interpolate_Tau(Wlname,GasAbsFn,altIndex,SpecResFn)
    Tau_Comb_solar = intp[0]
    Tau_Comb_view = intp[1]# Tau interpolated to the RSP response function wavelengths
    
    RSP_wl = intp[2]
    SzenNo = len(Solar_Zenith) # no of angles measured by RSP
    
    C_factor_solar = np.zeros((SzenNo,len(RSP_wl))) #  angles x wl
    C_factor_view = np.zeros((SzenNo,len(RSP_wl))) #  angles x wl

    G_s = 1/np.cos(np.radians(Solar_Zenith))
    G_v = 1/np.cos(np.radians(Viewing_Zenith))

    for i in range(SzenNo):
        C_factor_solar[i,:] = np.exp(-(G_s[i])*Tau_Comb_solar) #Based on solar zenith angle
        C_factor_view[i,:] = np.exp(-(G_v[i])*Tau_Comb_view)

    return  C_factor_solar, C_factor_view


### Reading the Multiangle Polarimeter data ()

# Reads the Data from ORACLES and gives the rslt dictionary for GRASP
def Read_Data_RSP_Oracles(file_path,file_name,PixNo,ang1,ang2,TelNo, nwl,GasAbsFn): #PixNo = Index of the pixel, #nwl = wavelength index, :nwl will be taken
    #ang 1 & ang 2 : index for selected range of view angles.
    #TelNo : telescope index. RSP has 2 telescopes. 
    
    #Reading the hdf file
    f1_MAP = h5py.File(file_path + file_name,'r+') 
    Data = f1_MAP['Data'] #Reading the data
    
    #Variables
    wl = Data['Wavelength'] #Wavelength
    if nwl == None: nwl = len(Data['Wavelength'][:]) # User could either provide the number of wavelengths (which is also index of the wl), or it will just take the number of wavelength values in the variable " Wavelength"

    #Reading the Geometry
    Lat = f1_MAP['Geometry']['Collocated_Latitude'][TelNo,PixNo]
    Lon = f1_MAP['Geometry']['Collocated_Longitude'][TelNo,PixNo]   
    #All the angles are converted to GRASP's definition of Genometry which is different than that of RSP
    vza = 180-f1_MAP['Geometry']['Viewing_Zenith'][TelNo,PixNo,ang1:ang2]

    #This can be used to filter scattering angles if required
    vza[f1_MAP['Geometry']['Nadir_Index'][0,PixNo]:] = - vza[f1_MAP['Geometry']['Nadir_Index'][0,PixNo]:]
    Angfilter = (vza>= -45) & (vza<= 45) # taking only the values of view zenith from -65 to 45

    #Angles are checked for nans 
    Scattering_ang = checkFillVals(f1_MAP['Geometry']['Scattering_Angle'][TelNo,PixNo,ang1:ang2]  )
    Solar_Zenith =  checkFillVals(f1_MAP['Geometry']['Solar_Zenith'][TelNo,PixNo,ang1:ang2]  )
    
    #Converting sunlight azimuth to solar azimuth: 𝜃𝑠, 180- 𝜃𝑣 𝜙𝑠 = 𝜙𝑠 -180, 𝜙𝑣

    Solar_Azimuth = checkFillVals(f1_MAP['Geometry']['Solar_Azimuth'][TelNo,PixNo,ang1:ang2], ) - 180
    Viewing_Azimuth = checkFillVals(f1_MAP['Geometry']['Viewing_Azimuth'][TelNo,PixNo,ang1:ang2]  )
   
    #Converting viewing zenith with respect to nadir to that wrt zenith
    Viewing_Zenith = 180 - checkFillVals(f1_MAP['Geometry']['Viewing_Zenith'][TelNo,PixNo,ang1:ang2]  ) # Theta_v <90
    
    #Converting values into radians to caculate the relative azimuth angles        
    sza =  np.radians(Solar_Zenith)
    vza =   np.radians(Viewing_Zenith)
    szi =  np.radians(Solar_Azimuth)
    vzi =  np.radians(Viewing_Azimuth)
    
    Relative_Azi = (180/np.pi)*(np.arccos((np.cos((Scattering_ang *np.pi)/180)  + np.cos(sza)*np.cos(vza))/(- np.sin(sza)*np.sin(vza)) ))

    # Relative_Azi = Solar_Azimuth - Viewing_Azimuth
    # for i in range (len(Relative_Azi)): 
    #     if Relative_Azi[i]<0 : Relative_Azi[i] =  Relative_Azi[i]+360
    RSP_wlf = [410, 470, 555, 670, 865, 960, 1590, 1880, 2250] #wl as in the file name of response functions
    
    # CorFac1 = np.ones((np.sum(Angfilter),nwl))
    # CorFac2 = np.ones((np.sum(Angfilter),nwl))
    # fig, ax1 = plt.subplots()
    # ax2 = ax1.twinx() 
    # for j in range(nwl):
        
    #     if j == 8:
    #         Solar_Zenith = f1_MAP['Geometry']['Solar_Zenith'][1,PixNo,ang1:ang2]
    #         Viewing_Zenith = f1_MAP['Geometry']['Viewing_Zenith'][1,PixNo,ang1:ang2]
            
    #     Wlname =  RSP_wlf[j]
    #     print(Wlname)
    #     altIndex = 7 #v I need to improve this and make it more general, altitude index where the altidue t

    #     SpecResFn = np.loadtxt(f'/home/gregmi/ORACLES/RSP_Spectral_Response/{Wlname}.txt')
    #     intp =Interpolate_Tau(Wlname,GasAbsFn,altIndex,SpecResFn)
    # RSP_wl = intp[2]
    # resFunc = intp[3]/np.max(intp[3])
    # Trans1 = Abs_Correction(Solar_Zenith,Viewing_Zenith,Wlname,GasAbsFn,altIndex,SpecResFn)[0]
    # Trans2 = Abs_Correction(Solar_Zenith,Viewing_Zenith,Wlname,GasAbsFn,altIndex,SpecResFn)[1]
    
    # ax1.plot(RSP_wl,Trans1[0,:],lw =0.2)
    # ax2.plot(RSP_wl,resFunc, label=f"{RSP_wlf[j]} ")
    # plt.legend()
    
    # for i in range(ang2-ang1):
    #     CorFac1[i,j] = np.sum(Trans1[i,1:]*resFunc[1:]* (np.diff(RSP_wl)))/np.sum(resFunc[1:]* (np.diff(RSP_wl)))
    #     CorFac2[i,j] = np.sum(Trans2[i,1:]*resFunc[1:]* (np.diff(RSP_wl)))/np.sum(resFunc[1:]* (np.diff(RSP_wl)))
            
    # corrFac = (CorFac1+CorFac2)/np.nanmax(CorFac1+CorFac2) #Noramalized correction factore

    I1 = (checkFillVals(Data['Intensity_1'][PixNo,ang1:ang2,:nwl]  , negative_check =True))# / corrFac telescope 1 Normalized intensity (unitless)#there are some negative intesity values in the file
    # I1 = I1/CorFac2
    I2 = (checkFillVals(Data['Intensity_2'][PixNo,ang1:ang2,:nwl]  ,negative_check =True))# #telescope 2
    # I2 = I2/CorFac2
    # Q and U in scattering plane 


   
    #Creating rslt dictionary for GRASP
    rslt ={}
    rslt['lambda'] = Data['Wavelength'][:nwl]/1000 # Wavelengths in um
    rslt['longitude'] = Lon
    rslt['latitude'] = Lat
    rslt['meas_I'] = (I1+I2)/2  

    '''This should be changed  '''

    # rslt['meas_P'] = rslt['meas_I'] *checkFillVals(Data['DoLP'][PixNo,ang1:ang2,:nwl]  ,negative_check =True)/100
    rslt['meas_P'] = checkFillVals(Data['DoLP'][PixNo,ang1:ang2,:nwl]  ,negative_check =True)/100    #relative value P/I
    #converting modified julian date to julain date and then to gregorian
    jdv = f1_MAP['Geometry']['Measurement_Time'][TelNo,PixNo,0]+ 2400000.5  #Taking the time stamp for first angle
    
    yy,mm,dd,hh,mi,s,ms = jd.to_gregorian(jdv)
    rslt['datetime'] = dt.datetime(yy,mm,dd,hh,mi,s,ms) #Coverts julian to datetime
    jdv = f1_MAP['Geometry']['Measurement_Time'][TelNo,PixNo,0]+ 2400000.5  #Taking the time stamp for first angle
    

    # All the geometry arrays should be 2D, (angle, wl) --> solar zenith angle, wavelength
    rslt['sza'] = np.repeat(Solar_Zenith, nwl).reshape(len(Solar_Zenith), nwl) #repeat for number of view angles
    rslt['vis']= np.repeat(Viewing_Zenith, nwl).reshape(len(Viewing_Zenith), nwl) 
    rslt['sca_ang']= np.repeat( Scattering_ang, nwl).reshape(len(Scattering_ang), nwl)  #Nangles x Nwavelengths
    rslt['fis'] = np.repeat(Relative_Azi , nwl).reshape(len(Relative_Azi ), nwl)
    rslt['land_prct'] =0 #0% land Percentage
    # solar_distance = (f1_MAP['Platform']['Solar_Distance'][PixNo])**2 
    # const = solar_distance/(np.cos(np.radians(rslt['sza'])))
 
    #height key should not be used for altitude,
    rslt['OBS_hght']= f1_MAP['Platform']['Platform_Altitude'][PixNo] # height of pixel in m
    # print(rslt['OBS_hght'])
    if  rslt['OBS_hght'] < 0:  #if colocated attitude is less than 0 then that is set to 0
        rslt['OBS_hght'] = 0
        print(f"The collocated height was { rslt['OBS_hght']}, OBS_hght was set to 0 ")
    f1_MAP.close()
    return rslt

########################################## Reading LES Data ##########################################
# Project: Aerosol impact in Twilight Zones using LES 
# Author: Nirandi Jayasinghe
# Reading LES Data into rslt dict to create SDATA
# Last Edited: Aug 6th 2024

def Read_LES_Data(file_path, filename, XPX, YPX, nwl,RT):

    #Reading Data
    Data = Dataset(file_path+filename, 'r')
    Var_Names = Data.variables.keys()

    data = {}

    #Reading all the variables
    for name in Var_Names: 
        data[name] = Data.variables[name][:]
    
    wl = data["Rayleigh_Wavelength"].data #wavelengths
    if nwl == None : nwl = len(wl)  #no of wavelengths

    Lat = 41.5600 #dummy value 
    Lon = -5.2500 #dummy value

    #Assuming angles for LES data are defined similar to GRASP definition.
    #view zenith angle
    vza = data["View_Zenith_Angle"].data
    #scatteting angle
    scat_ang = data["scat_ang"].data
    #solar zenith angle
    sza = data["Solar_Zenith_Angle"].data
    #solar azimuth angle
    saa = data["Solar_Azimuth_Angle"].data
    #view azimuth angle
    vaa = data["View_Azimuth_Angle"].data

    #converting angles to radians to calculate relative azimuth angles
    SZA = np.radians(sza)
    VZA = np.radians(vza)
    SAA = np.radians(saa)
    VAA = np.radians(vaa)

    Rel_Azimuth = (180/np.pi)*(np.arccos((np.cos((scat_ang *np.pi)/180) + np.cos(SZA)*np.cos(VZA))/(- np.sin(SZA)*np.sin(VZA))))

    if (RT==1):
        # Dimentions:[nVZA,nwl,SZA,Stoke,nX,nY]
        meas_I = data["RT_1D_Reflectance"].data[:,:nwl,0,0,XPX:XPX+1,YPX:YPX+1]
        meas_Q = data["RT_1D_Reflectance"].data[:,:nwl,0,1,XPX:XPX+1,YPX:YPX+1]
        meas_U = data["RT_1D_Reflectance"].data[:,:nwl,0,2,XPX:XPX+1,YPX:YPX+1]
        DOLP = np.sqrt(meas_Q*meas_Q+meas_U*meas_U)/meas_I

    if (RT==3):
        # Dimentions:[nVZA,nwl,SZA,Stoke,nX,nY]
        meas_I = data["RT_3D_Reflectance"].data[:,:nwl,0,0,XPX:XPX+1,YPX:YPX+1]
        meas_Q = data["RT_3D_Reflectance"].data[:,:nwl,0,1,XPX:XPX+1,YPX:YPX+1]
        meas_U = data["RT_3D_Reflectance"].data[:,:nwl,0,2,XPX:XPX+1,YPX:YPX+1]
        DOLP = np.sqrt(meas_Q*meas_Q+meas_U*meas_U)/meas_I

    #creating Results Dictionary for GRASP

    rslt = {}

    rslt['lambda'] = wl #wavelength is in um
    rslt['longitude'] = Lon
    rslt['latitude'] = Lat
    rslt['meas_I'] = meas_I
    rslt['meas_Q'] = meas_Q
    rslt['meas_U'] = meas_U
    rslt['meas_P'] = DOLP

    yy = 2024 ; mm = 8 ; dd = 8 ; hh = 12 ; mi = 00 ; s = 00; ms = 00
    rslt['datetime'] = dt.datetime(yy,mm,dd,hh,mi,s,ms)
  
    # All the geometry arrays should be 2D, (angle, wl) --> solar zenith angle, wavelength
    rslt['sza'] = np.repeat(sza,nwl).reshape(len(sza),nwl)
    rslt['vis'] = np.repeat(vza,nwl).reshape(len(vza),nwl)
    rslt['sca_ang'] = np.repeat(scat_ang,nwl).reshape(len(scat_ang),nwl)
    rslt['fis'] = np.repeat(Rel_Azimuth,nwl).reshape(len(Rel_Azimuth),nwl)
    rslt['land_prct'] = 0 #over ocean 

    rslt['OBS_hght'] = data['Sensor_Position'].data*1000 #sensor position is 20 km for LES data

    return rslt

    ########################################## END of LES DATA READING ##########################################

def Read_Data_HSRL_Oracles(file_path,file_name,PixNo):

    f1= h5py.File(file_path + file_name,'r+')  #reading Lidar measurements 
    HSRL = f1['DataProducts']
    latitude,longitude = f1['Nav_Data']['gps_lat'][:],f1['Nav_Data']['gps_lon'][:]
    AirAlt = f1['Nav_Data']['gps_alt'][PixNo] #Altitude of the aircraft
    print(AirAlt)

    Data_dic ={} #This dictionary stores 
    inp = ['355_ext','532_ext','1064_ext','355_bsc_Sa','532_bsc_Sa','1064_bsc_Sa','355_dep', '532_dep','1064_dep']
    #Setting negative values to zero, Teh negative values are due to low signal so we can replace with 0 without loss of info.
    for i in range (len(inp)):
        Data_dic[f'{inp[i]}'] = HSRL[f'{inp[i]}'][PixNo]
        HSRL_checkFillVals(Data_dic[f'{inp[i]}']) # set all negative values to zero
        if (inp[i] == '355_dep') or (inp[i] == '532_dep') or (inp[i] == '1064_dep'):
            Data_dic[f'{inp[i]}'] = np.where(HSRL[f'{inp[i]}'][PixNo][:]>= 0.6 , np.nan, HSRL[f'{inp[i]}'][PixNo])  #CH: This should be changed, 0.7 has been set arbitarily

    #Caculating range, Range is defined as distance from the instrument to the aersosol layer, i.e. range at instrument heright = 0. We have to make sure that The range is in decending order
    Data_dic['Altitude'] = AirAlt - HSRL['Altitude'][0] 
    df_new = pd.DataFrame(Data_dic)
    df_new.interpolate(inplace=True, limit_area= 'inside')

    #Filtering and removing the pixels with bad data: 
    Removed_index = []  # Removed_index holds indices of pixels to be removed
    # Filter values greater than flight altitude
    Removed_index.append(np.nonzero(np.array(df_new['Altitude'][:]) > AirAlt)[0][:])
    # Filter values less than or equal to zero altitude
    Removed_index.append(np.nonzero(np.array(df_new['Altitude'][:]) <= 0)[0][:])
    # Filter values less than or equal to zero range
    Removed_index.append(np.nonzero(f1['UserInput']['range_interp'][PixNo] <= 0)[0])
    # Filter values less than 1800 in range interpolation
    Removed_index.append(np.nonzero(f1['UserInput']['range_interp'][PixNo] < 1800)[0])
    # Filter NaN values in range interpolation
    Removed_index.append(np.nonzero(np.isnan(f1['UserInput']['range_interp'][PixNo]))[0])
    # Filter NaN values in low gain signal limit data mask
    Removed_index.append(np.nonzero(np.isnan(HSRL["mask_low"][PixNo]))[0])

    # Cloud Correction
    CloudCorr_1064 = np.nonzero(np.isnan(HSRL["1064_bsc_cloud_screened"][PixNo]))[0]
    CloudCorr_355 = np.nonzero(np.isnan(HSRL["355_bsc_cloud_screened"][PixNo]))[0]
    CloudCorr_532 = np.nonzero(np.isnan(HSRL["532_bsc_cloud_screened"][PixNo]))[0]
    
    Removed_index.append(CloudCorr_1064)
    Removed_index.append(CloudCorr_355)
    Removed_index.append(CloudCorr_532)
    Removed_index.append(np.nonzero(np.isnan(HSRL['355_ext'][PixNo]))[0])

    # Concatenate all removed indices and remove duplicates
    rm_pix=[]
    for lis in Removed_index:
        rm_pix += list(lis)[:] # concatenating the lists
    rm_pix = np.unique(rm_pix)

    # Create dictionaries to hold filtered and interpolated data
    del_dict = {}
    # Delete removed pixels and set negative values to zero for each data type
    inp2 = ['355_ext','532_ext','1064_ext','355_bsc_Sa','532_bsc_Sa','1064_bsc_Sa','355_dep', '532_dep','1064_dep', 'Altitude']
    for i in range (10):
        del_dict[f'{inp2[i]}'] = np.delete(np.array(df_new[f'{inp2[i]}']), rm_pix)

    df_mean = pd.DataFrame()
    npoints = 10 #no of height pixels averaged 
    Mod_value = np.array(del_dict['Altitude']).shape[0] % npoints  #Skip these values for reshaping the array
    for i in range (10): #taking mean 
        df_mean[f'{inp2[i]}'] = nanmean(np.array(del_dict[f'{inp2[i]}'][Mod_value:]).reshape( int(np.array(del_dict[f'{inp2[i]}']).shape[0]/npoints),npoints),axis=1)

    for k in df_mean.keys():
        print(df_mean[k].shape)

    df = df_mean[:]
    rslt = {} # 
    height_shape = np.array(df['Altitude'][:]).shape[0] #to avoint the height of the sea salt, this should be removed 

    Range = np.ones((height_shape,3))
    Range[:,0] = df['Altitude'][:]
    Range[:,1] = df['Altitude'][:]
    Range[:,2] = df['Altitude'][:]  # in meters
    rslt['RangeLidar'] = Range

    Bext = np.ones((height_shape,3))
    Bext[:,0] = df['355_ext'][:]
    Bext[:,1] = df['532_ext'][:]
    Bext[:,2] = df['1064_ext'] [:]
    # Bext[0,2] = np.nan 

    Bsca = np.ones((height_shape,3))
    Bsca[:,0] = df['355_bsc_Sa'][:]
    Bsca[:,1] = df['532_bsc_Sa'] [:]
    Bsca[:,2] = df['1064_bsc_Sa'][:]

    # Nor_Bca = 
    # Bsca[0,2] = np.nan 

    Dep = np.ones((height_shape,3))
    Dep[:,0] = df['355_dep'][:]
    Dep[:,1] = df['532_dep'][:]
    Dep[:,2] = df['1064_dep'] [:]

    rslt['meas_VExt'] = Bext / 1000
    rslt['meas_VBS'] = Bsca / 1000 # converting units from km-1 tp m-1
    rslt['meas_DP'] = Dep*100  #_aer
    # print(rslt['meas_DP'])

    rslt['lambda'] = np.array([355,532,1064])/1000 #values of HSRL wl in um
    rslt['wl'] = np.array([355,532,1064])/1000
    rslt['datetime'] =dt.datetime.strptime(str(int(f1["header"]['date'][0][0]))+ np.str(f1['Nav_Data']['UTCtime2'][PixNo][0]),'%Y%m%d%H%M%S.%f')
    rslt['latitude'] = latitude[PixNo]
    rslt['longitude']= longitude[PixNo]
    rslt['OBS_hght']=  AirAlt# aircraft altitude. 
    rslt['land_prct'] = 0 #Ocean Surface

    #Substitude the actual value
    rslt['gaspar'] = np.ones((3))*0.0037 #MOlecular depolarization 
    f1.close() 
    return rslt

## Reading the Multiangle Polarimeter data ()

def Read_Data_HSRL_Oracles(file_path,file_name,PixNo):

    f1= h5py.File(file_path + file_name,'r+')  #reading Lidar measurements 
    HSRL = f1['DataProducts']
    latitude,longitude = f1['Nav_Data']['gps_lat'][:],f1['Nav_Data']['gps_lon'][:]
    AirAlt = f1['Nav_Data']['gps_alt'][PixNo] #Altitude of the aircraft
    print(AirAlt)

    Data_dic ={} #This dictionary stores 
    inp = ['355_ext','532_ext','1064_ext','355_bsc_Sa','532_bsc_Sa','1064_bsc_Sa','355_dep', '532_dep','1064_dep']
    #Setting negative values to zero, Teh negative values are due to low signal so we can replace with 0 without loss of info.
    for i in range (len(inp)):
        Data_dic[f'{inp[i]}'] = HSRL[f'{inp[i]}'][PixNo]
        HSRL_checkFillVals(Data_dic[f'{inp[i]}']) # set all negative values to zero
        if (inp[i] == '355_dep') or (inp[i] == '532_dep') or (inp[i] == '1064_dep'):
            Data_dic[f'{inp[i]}'] = np.where(HSRL[f'{inp[i]}'][PixNo][:]>= 0.6 , np.nan, HSRL[f'{inp[i]}'][PixNo])  #CH: This should be changed, 0.7 has been set arbitarily

    #Caculating range, Range is defined as distance from the instrument to the aersosol layer, i.e. range at instrument heright = 0. We have to make sure that The range is in decending order
    Data_dic['Altitude'] = AirAlt - HSRL['Altitude'][0] 
    df_new = pd.DataFrame(Data_dic)
    df_new.interpolate(inplace=True, limit_area= 'inside')

    # Filtering and removing the pixels with bad data: 
    Removed_index = []  # Removed_index holds indices of pixels to be removed
    # Filter values greater than flight altitude
    Removed_index.append(np.nonzero(np.array(df_new['Altitude'][:]) > AirAlt)[0][:])
    # Filter values less than or equal to zero altitude
    Removed_index.append(np.nonzero(np.array(df_new['Altitude'][:]) <= 0)[0][:])
    # Filter values less than or equal to zero range
    Removed_index.append(np.nonzero(f1['UserInput']['range_interp'][PixNo] <= 0)[0])
    # Filter values less than 1800 in range interpolation
    Removed_index.append(np.nonzero(f1['UserInput']['range_interp'][PixNo] < 1800)[0])
    # Filter NaN values in range interpolation
    Removed_index.append(np.nonzero(np.isnan(f1['UserInput']['range_interp'][PixNo]))[0])
    # Filter NaN values in low gain signal limit data mask
    Removed_index.append(np.nonzero(np.isnan(HSRL["mask_low"][PixNo]))[0])

    # Cloud Correction
    CloudCorr_1064 = np.nonzero(np.isnan(HSRL["1064_bsc_cloud_screened"][PixNo]))[0]
    CloudCorr_355 = np.nonzero(np.isnan(HSRL["355_bsc_cloud_screened"][PixNo]))[0]
    CloudCorr_532 = np.nonzero(np.isnan(HSRL["532_bsc_cloud_screened"][PixNo]))[0]
    
    Removed_index.append(CloudCorr_1064)
    Removed_index.append(CloudCorr_355)
    Removed_index.append(CloudCorr_532)
    Removed_index.append(np.nonzero(np.isnan(HSRL['355_ext'][PixNo]))[0])

    # Concatenate all removed indices and remove duplicates
    rm_pix=[]
    for lis in Removed_index:
        rm_pix += list(lis)[:] # concatenating the lists
    rm_pix = np.unique(rm_pix)

    # Create dictionaries to hold filtered and interpolated data
    del_dict = {}
    # Delete removed pixels and set negative values to zero for each data type
    inp2 = ['355_ext','532_ext','1064_ext','355_bsc_Sa','532_bsc_Sa','1064_bsc_Sa','355_dep', '532_dep','1064_dep', 'Altitude']
    for i in range (10):
        del_dict[f'{inp2[i]}'] = np.delete(np.array(df_new[f'{inp2[i]}']), rm_pix)

    df_mean = pd.DataFrame()
    npoints = 10 #no of height pixels averaged 
    Mod_value = np.array(del_dict['Altitude']).shape[0] % npoints  #Skip these values for reshaping the array
    for i in range (10): #taking mean 
        df_mean[f'{inp2[i]}'] = nanmean(np.array(del_dict[f'{inp2[i]}'][Mod_value:]).reshape( int(np.array(del_dict[f'{inp2[i]}']).shape[0]/npoints),npoints),axis=1)

    for k in df_mean.keys():
        print(df_mean[k].shape)

    df = df_mean[:]
    rslt = {} # 
    height_shape = np.array(df['Altitude'][:]).shape[0] #to avoint the height of the sea salt, this should be removed 

    Range = np.ones((height_shape,3))
    Range[:,0] = df['Altitude'][:]
    Range[:,1] = df['Altitude'][:]
    Range[:,2] = df['Altitude'][:]  # in meters
    rslt['RangeLidar'] = Range

    Bext = np.ones((height_shape,3))
    Bext[:,0] = (df['355_ext'][:]/1000)/(np.trapz(df['355_ext'][::-1]/1000, df['Altitude'][::-1]))
    Bext[:,1] = (df['532_ext'][:]/1000)/(np.trapz(df['532_ext'][::-1]/1000, df['Altitude'][::-1]))
    Bext[:,2] = (df['1064_ext'] [:]/1000)/(np.trapz(df['1064_ext'][::-1]/1000, df['Altitude'][::-1]))
    # Bext[0,2] = np.nan 

    Bsca = np.ones((height_shape,3))
    Bsca[:,0] =(df['355_bsc_Sa'][:]/1000)/(np.trapz(df['355_bsc_Sa'][::-1]/ 1000, df['Altitude'][::-1]))
    Bsca[:,1] = (df['532_bsc_Sa'] [:]/1000)/(np.trapz(df['532_bsc_Sa'][::-1]/ 1000, df['Altitude'][::-1]))
    Bsca[:,2] = (df['1064_bsc_Sa'][:]/1000)/(np.trapz(df['1064_bsc_Sa'][::-1]/ 1000, df['Altitude'][::-1]))

    # Nor_Bca = 
    # Bsca[0,2] = np.nan 

    Dep = np.ones((height_shape,3))
    Dep[:,0] = df['355_dep'][:]
    Dep[:,1] = df['532_dep'][:]
    Dep[:,2] = df['1064_dep'] [:]

    rslt['meas_VExt'] = Bext
    rslt['meas_VBS'] = Bsca # converting units from km-1 tp m-1
    rslt['meas_DP'] = Dep*100  #_aer
    # print(rslt['meas_DP'])

    rslt['lambda'] = np.array([355,532,1064])/1000 #values of HSRL wl in um
    rslt['wl'] = np.array([355,532,1064])/1000
    rslt['datetime'] =dt.datetime.strptime(str(int(f1["header"]['date'][0][0]))+ np.str(f1['Nav_Data']['UTCtime2'][PixNo][0]),'%Y%m%d%H%M%S.%f')
    rslt['latitude'] = latitude[PixNo]
    rslt['longitude']= longitude[PixNo]
    rslt['OBS_hght']=  AirAlt# aircraft altitude. 
    rslt['land_prct'] = 0 #Ocean Surface

    #Substitude the actual value
    rslt['gaspar'] = np.ones((3))*0.0037 #MOlecular depolarization 
    f1.close() 
    return rslt





