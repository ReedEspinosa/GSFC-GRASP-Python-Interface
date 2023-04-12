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

#The function Checks for Fill values or negative values and replaces them with nan. To check for negative values, set negative_check = True 
def checkFillVals(param,Angfilter, negative_check = None):
    
    param[:] = np.where(param[:] == -999, np.nan, param[:])
    
    if negative_check == True:
        param[:] = np.where(param[:] < 0 , np.nan, param[:])

    param = param[Angfilter]
        
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
def Read_Data_RSP_Oracles(file_path,file_name,PixNo,TelNo, nwl,GasAbsFn): #PixNo = Index of the pixel, #nwl = wavelength index, :nwl will be taken
    
    #Reading the hdf file
    f1_MAP = h5py.File(file_path + file_name,'r+') 
    
    Data = f1_MAP['Data'] #Reading the data
    
    #Variables
    wl = Data['Wavelength']
    if nwl == None: nwl = len(Data['Wavelength'][:])

    #Reading the Geometry
    Lat = f1_MAP['Geometry']['Collocated_Latitude'][TelNo,PixNo]
    Lon = f1_MAP['Geometry']['Collocated_Longitude'][TelNo,PixNo]
    #filtering angles
    vza = 180-f1_MAP['Geometry']['Viewing_Zenith'][TelNo,PixNo,:]
    vza[:f1_MAP['Geometry']['Nadir_Index'][0,PixNo]] = - vza[:f1_MAP['Geometry']['Nadir_Index'][0,PixNo]]
    Angfilter = (vza>= -65) & (vza<= 45) # taking only the values of view zenith from -65 to 45



    Scattering_ang = checkFillVals(f1_MAP['Geometry']['Scattering_Angle'][TelNo,PixNo,:],Angfilter)
    Solar_Zenith =  checkFillVals(f1_MAP['Geometry']['Solar_Zenith'][TelNo,PixNo,:],Angfilter)
    
    #Converting sunlight azimuth to solar azimuth: 𝜃𝑠, 180- 𝜃𝑣 𝜙𝑠 = 𝜙𝑠 -180, 𝜙𝑣

    Solar_Azimuth = checkFillVals(f1_MAP['Geometry']['Solar_Azimuth'][TelNo,PixNo,:],Angfilter) - 180
    Viewing_Azimuth = checkFillVals(f1_MAP['Geometry']['Viewing_Azimuth'][TelNo,PixNo,:],Angfilter)
    #Converting viewing zenith with respect to nadir to that wrt zenith
    
    Viewing_Zenith = 180 - checkFillVals(f1_MAP['Geometry']['Viewing_Zenith'][TelNo,PixNo,:],Angfilter) # Theta_v <90
            
    sza =  np.radians(Solar_Zenith)
    vza =   np.radians(Viewing_Zenith)
    szi =  np.radians(Solar_Azimuth)
    vzi =  np.radians(Viewing_Azimuth)
    
    Relative_Azi = (180/np.pi)*(np.arccos((np.cos((Scattering_ang *np.pi)/180)  + np.cos(sza)*np.cos(vza))/(- np.sin(sza)*np.sin(vza)) ))

    # Relative_Azi = Solar_Azimuth - Viewing_Azimuth
    # for i in range (len(Relative_Azi)): 
    #     if Relative_Azi[i]<0 : Relative_Azi[i] =  Relative_Azi[i]+360
    RSP_wlf = [410, 470, 555, 670, 865, 960, 1590, 1880, 2250] #wl as in the file name of response functions
    
    CorFac1 = np.ones((np.sum(Angfilter),nwl))
    CorFac2 = np.ones((np.sum(Angfilter),nwl))
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx() 
    for j in range(nwl):
        
        if j == 8:
            Solar_Zenith = f1_MAP['Geometry']['Solar_Zenith'][1,PixNo,:]
            Viewing_Zenith = f1_MAP['Geometry']['Viewing_Zenith'][1,PixNo,:]
            
        Wlname =  RSP_wlf[j]
        print(Wlname)
        altIndex = 7 #v I need to improve this and make it more general, altitude index where the altidue t

        SpecResFn = np.loadtxt(f'/home/gregmi/ORACLES/RSP_Spectral_Response/{Wlname}.txt')
        intp =Interpolate_Tau(Wlname,GasAbsFn,altIndex,SpecResFn)
    RSP_wl = intp[2]
    resFunc = intp[3]/np.max(intp[3])
    Trans1 = Abs_Correction(Solar_Zenith,Viewing_Zenith,Wlname,GasAbsFn,altIndex,SpecResFn)[0]
    Trans2 = Abs_Correction(Solar_Zenith,Viewing_Zenith,Wlname,GasAbsFn,altIndex,SpecResFn)[1]
    
    ax1.plot(RSP_wl,Trans1[0,:],lw =0.2)
    ax2.plot(RSP_wl,resFunc, label=f"{RSP_wlf[j]} ")
    plt.legend()
    
    for i in range(np.sum(Angfilter)):
        CorFac1[i,j] = np.sum(Trans1[i,1:]*resFunc[1:]* (np.diff(RSP_wl)))/np.sum(resFunc[1:]* (np.diff(RSP_wl)))
        CorFac2[i,j] = np.sum(Trans2[i,1:]*resFunc[1:]* (np.diff(RSP_wl)))/np.sum(resFunc[1:]* (np.diff(RSP_wl)))
            



    corrFac = (CorFac1+CorFac2)/np.nanmax(CorFac1+CorFac2) #Noramalized correction factore

    I1 = (checkFillVals(Data['Intensity_1'][PixNo,:,:nwl],Angfilter, negative_check =True)/ corrFac)# / corrFac telescope 1 Normalized intensity (unitless)#there are some negative intesity values in the file
    # I1 = I1/CorFac2
    I2 = (checkFillVals(Data['Intensity_2'][PixNo,:,:nwl],Angfilter,negative_check =True)/ corrFac)# #telescope 2
    # I2 = I2/CorFac2
    # Q and U in scattering plane 
    
    # scat_sin = checkFillVals(f1_MAP['Geometry']['Sin_Rot_Scatt_Plane'][TelNo,PixNo,ang1:ang])
    # scat_cos = checkFillVals(f1_MAP['Geometry']['Cos_Rot_Scatt_Plane'][TelNo,PixNo,ang1:ang])
    
    # scat_sin1 = np.repeat(scat_sin, nwl).reshape(len(scat_sin), nwl)
    # scat_cos1 = np.repeat(scat_cos, nwl).reshape(len(scat_cos), nwl)
    
    # Q = checkFillVals(Data['Stokes_Q'][PixNo,:,:nwl])
    # U = checkFillVals(Data['Stokes_U'][PixNo,:,:nwl])

   
    #Creating rslt dictionary for GRASP
    rslt ={}
    rslt['lambda'] = Data['Wavelength'][:nwl]/1000 # Wavelengths in um
    rslt['longitude'] = Lon
    rslt['latitude'] = Lat
    rslt['meas_I'] = (I1+I2)/2                              
    rslt['meas_P'] = rslt['meas_I'] *checkFillVals(Data['DoLP'][PixNo,:,:nwl],Angfilter,negative_check =True)/100
    
  
            
    #converting modified julian date to julain date and then to gregorian
    jdv = f1_MAP['Geometry']['Measurement_Time'][TelNo,PixNo,0]+ 2400000.5  #Taking the time stamp for first angle
    
    yy,mm,dd,hh,mi,s,ms = jd.to_gregorian(jdv)
    rslt['datetime'] = dt.datetime(yy,mm,dd,hh,mi,s,ms) #Coverts julian to datetime
    jdv = f1_MAP['Geometry']['Measurement_Time'][TelNo,PixNo,0]+ 2400000.5  #Taking the time stamp for first angle
    

    # All the geometry arrays should be 2D, (angle, wl)
    rslt['sza'] = np.repeat(Solar_Zenith, nwl).reshape(len(Solar_Zenith), nwl)
    rslt['vis']= np.repeat(Viewing_Zenith, nwl).reshape(len(Viewing_Zenith), nwl) 
    rslt['sca_ang']= np.repeat( Scattering_ang, nwl).reshape(len(Scattering_ang), nwl)  #Nangles x Nwavelengths
    rslt['fis'] = np.repeat(Relative_Azi , nwl).reshape(len(Relative_Azi ), nwl)
    rslt['land_prct'] =0 #0% land Percentage
    # solar_distance = (f1_MAP['Platform']['Solar_Distance'][PixNo])**2 
    # const = solar_distance/(np.cos(np.radians(rslt['sza'])))
 

    #height key should not be used for altitude,
    rslt['OBS_hght']= f1_MAP['Platform']['Platform_Altitude'][PixNo]- 1000 # height of pixel in m
    if  rslt['OBS_hght'] < 0:  #if colocated attitude is less than 0 then that is set to 0
        rslt['OBS_hght'] = 0
        print(f"The collocated height was { rslt['OBS_hght']}, OBS_hght was set to 0 ")

    return rslt
 


def Read_Data_HSRL_Oracles(file_path,file_name,PixNo):


    #Specify the pixel no
    PixNo = 1250

    f1= h5py.File(file_path + file_name,'r+')  #reading hdf5 file  

    #Lat and Lon values for that pixel
    latitude = f1['Nav_Data']['gps_lat'][:]
    longitude = f1['Nav_Data']['gps_lon'][:]
    altitude = f1['Nav_Data']['gps_alt'][:]

    #Reading the data Products
    HSRL = f1['DataProducts']


    #The data file has many nan values, val is the list that stores indices of all the nan values for all the parameters
    val = []
    inp =['355_ext','532_ext','1064_ext','355_aer_dep', '532_aer_dep','1064_aer_dep','355_bsc_Sa','532_bsc_Sa','1064_bsc_Sa']
    for i in range (9):
        value2 = HSRL[f'{inp[i]}'][PixNo,1:].reshape(56, 10).mean(axis=1)
        val.append(np.nonzero(np.isnan(value2))[0])#this will give the indices of values that are nan
        val.append(np.nonzero(value2<0)[0]) #Filter all negatives
    
    # val.append(np.nonzero(HSRL['355_ext'][PixNo,1:].reshape(56, 10).mean(axis=1)<0)[0])
    # val.append(np.nonzero(HSRL['355_bsc_Sa'][PixNo,1:].reshape(56, 10).mean(axis=1)<0)[0])
    remove_pixels = np.unique(list(itertools.chain.from_iterable(val))) 
                
    # Merging all the lists in the val file and then taking the unique values
    # values for all these pixels will be removed from the parameter file so that we no longer have nan values.


    rslt = {} 
    rslt['lambda'] = np.array([355,532,1064])/1000

    rslt['wl'] = np.array([355,532,1064])/1000
    height_shape = np.delete(HSRL['355_ext'][PixNo,1:].reshape(56, 10).mean(axis=1),remove_pixels) .shape[0]

    Range = np.ones((height_shape,3))
    Range[:,0] = np.delete(f1['UserInput']['range_interp'][PixNo,1:].reshape(56, 10).mean(axis=1),remove_pixels) 
    Range[:,1] = np.delete(f1['UserInput']['range_interp'][PixNo,1:].reshape(56, 10).mean(axis=1),remove_pixels) 
    Range[:,2] = np.delete(f1['UserInput']['range_interp'][PixNo,1:].reshape(56, 10).mean(axis=1),remove_pixels)   # in meters
    rslt['RangeLidar'] = Range

    Bext = np.ones((height_shape,3))
    Bext[:,0] = np.delete(HSRL['355_ext'][PixNo,1:].reshape(56, 10).mean(axis=1),remove_pixels) 
    Bext[:,1] = np.delete(HSRL['532_ext'][PixNo,1:].reshape(56, 10).mean(axis=1),remove_pixels) 
    Bext[:,2] = np.delete(HSRL['1064_ext'][PixNo,1:].reshape(56, 10).mean(axis=1),remove_pixels) 
    # rslt['meas_VExt'] = Bext

    Bsca = np.ones((height_shape,3))
    Bsca[:,0] = np.delete(HSRL['355_bsc_Sa'][PixNo,1:].reshape(56, 10).mean(axis=1),remove_pixels) 
    Bsca[:,1] = np.delete(HSRL['532_bsc_Sa'][PixNo,1:].reshape(56, 10).mean(axis=1),remove_pixels) 
    Bsca[:,2] = np.delete(HSRL['1064_bsc_Sa'][PixNo,1:].reshape(56, 10).mean(axis=1),remove_pixels) 
    # rslt['meas_VBS'] =Bsca
    # Bext = Bext/1000 #m-1
    # Bsca = Bsca/1000 #m-1

    height = altitude[PixNo] -  Range[:,0]
    #Normalizing the values 
    # for i in range(3):
        # Bext[:,i] = (Bext[:,i]/1000)/np.trapz(Bext[:,i],height)
        # Bsca[:,i] = (Bsca[:,i]/1000)/np.trapz(Bsca[:,i],height)
        
    rslt['meas_VExt'] = Bext / 1000
    rslt['meas_VBS'] = Bsca / 1000 # m-1


    Dep = np.ones((height_shape,3))
    Dep[:,0] = np.delete(HSRL['355_aer_dep'][PixNo,1:].reshape(56, 10).mean(axis=1),remove_pixels) 
    Dep[:,1] = np.delete(HSRL['532_aer_dep'][PixNo,1:].reshape(56, 10).mean(axis=1),remove_pixels) 
    Dep[:,2] = np.delete(HSRL['1064_aer_dep'][PixNo,1:].reshape(56, 10).mean(axis=1),remove_pixels) 
    rslt['meas_DP'] = Dep*100

    
    rslt['datetime'] =dt.datetime.strptime(file_name[10:-6]+ np.str(f1['Nav_Data']['UTCtime2'][PixNo][0]),'%Y%m%d%H%M%S.%f')
    rslt['latitude'] = latitude[PixNo]
    rslt['longitude']= longitude[PixNo]

    rslt['OBS_hght']=(altitude[PixNo] -  Range[:,0])[-1] # in m
    rslt['land_prct'] = 0 #Ocean Surface

    return rslt