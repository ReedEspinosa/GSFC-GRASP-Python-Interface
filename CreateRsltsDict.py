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

#The function Checks for Fill values or negative values and replaces them with nan. To check for negative values, set negative_check = True 
def checkFillVals(param , negative_check = None):
    param[:] = np.where(param[:] == -999, np.nan, param[:])
    if negative_check == True:
        param[:] = np.where(param[:] < 0 , np.nan, param[:])
    return param

#This sets the nan values to 0 
def HSRL2_checkFillVals(param):
    param[:] = np.where(param < 0, np.nan, param) #todo changed to nan/ March 14 was O
    param[:] = np.where(np.isnan(param),1e-10, param) 
    return param

#Checks for negative values and replaces them by nan
def HSRL_checkFillVals(param):
    param[:] = np.where(param[:] < 0 , np.nan, param[:])     
    return param

'''  GasAbsFn: Format of file: .nc, Description: file containing the value of combined optical depth for different gases in the atmosphere using radiatiove tranfer code
     altIndex = Type: integer , Description: index of the vertical height at which we want to calculate the absorption, In this case we've taken the maximum altitude of RSP aircraft 
     SpecResFn = Format of file: .txt ,Description: file containing the  response funtion of the instruent at a particular wl (Wlname = interger, wavelength in the file name ) '''
    
def VertP(Data, hgtInterv,inp2):
    
    df_new = pd.DataFrame(Data)
    Plot_avg_prof = True
    avgProf = pd.DataFrame()
    hgt = Data['Altitude']

    # inp = ['355_ext','532_ext','1064_ext','355_bsc','532_bsc','1064_bsc','355_dep', '532_dep','1064_dep', 'Altitude','Dust_Mixing_Ratio','Angstrom_Spherical','Angstrom_532_355','532_aer_dep']

    altitude_diff = np.gradient(Data['Altitude'])
    numInterv = int(hgtInterv / np.mean(altitude_diff))  # Calculate numInterv based on mean altitude difference
    Npoint = int(len(Data['Altitude'])/numInterv) # No of vertical bins after averaging
    
    #assert hgtInterv > min(Data['Altitude']), "hgtInterv should be higher than the minimum altitude value."
    

    # Create an array to store the averaged values
    for i in range(numInterv):
        strtVal = i*numInterv
        endVal = (i+1)*numInterv    

        hgtAvg = np.zeros(Npoint)
    for k in range (len(inp2)):
        a = 0   # Indexing variable
        averaged_values = np.zeros(Npoint)

        for i in range(Npoint-1):
            start_range = hgt[i * numInterv]
            end_range = hgt[(i + 1) * numInterv]
            indexAvg= np.where((hgt >= start_range) & (hgt < end_range)) #Index of values in the start-stop range 
        
    
                #Taking mean
            if len(indexAvg[0]) > 0:
                non_zero_vals = np.array(df_new[f'{inp2[k]}'])[indexAvg]
                non_zero_vals = non_zero_vals[non_zero_vals > 0]
            if len(non_zero_vals) > 0:
                averaged_values[a] = np.nanmean(non_zero_vals)

            if k ==0: # Avoiding repetition  # Todo: update np,mean to
                hgtAvg[a]= np.nanmean(np.array(df_new['Altitude'])[indexAvg][np.where(np.array(df_new['Altitude'])[indexAvg] != 0)])
            a=a+1 # Indexing variable
        #Storing the values of profile
        avgProf[f'{inp2[k]}'] = averaged_values[np.where(hgtAvg>0)] #Averaged profile values with height >0 
        avgProf['Altitude'] =  hgtAvg[np.where(hgtAvg>0)]
        
    ## GRASP requires the vertical profiles to be in decending order, so reversing the entire profile
    df = avgProf[::-1]
    
    # if Plot_avg_prof ==True:
    #     fig, axs = plt.subplots(nrows= 1, ncols=9, figsize=(20, 6), sharey = True)
    #     for i in range (0,len(inp)-1):
    #         axs[i].plot(Data[f'{inp[i]}'],Data['Altitude'], marker= '.', label = "Org" )
    #         axs[i].plot(df[f'{inp[i]}'],df['Altitude'], marker= '.' , label = "Avg Prof")

    #         axs[i].set_xlabel(f'{inp[i]}')
            
    #     axs[0].legend()
    #     axs[0].set_ylabel('Height m ') 
        
#     AProf =  df.to_dict()
    
    return df
#TODO make this more general
SpecResFnPath = '/home/gregmi/ORACLES/RSP_Spectral_Response/'

def Gaspar_MAP(RSP_wl,Airhgt,GasAbsFn,SpecResFnPath):
    """This function will caculate the total gas optical depth when the spectral Response function and Gas Optical Depth for different gases are 
    provided
    For this case, SRF is from RSP instruemnt and Gas tau is from UNL_VRTM """
    
    #Gas Absorption correction using UNL_VRTM (provided by Richard,UMBC)
    RSP_wlf = [410, 470, 555, 670, 865,960,1590,1880,2250]

    # RSP_wlf = [410, 470, 555, 670, 865]

    ds = nc.Dataset(GasAbsFn)
    Wl = ds.variables['Lamdas'][:] # wavelength values corresponding to the gas absorption
    Tau = ds.variables['tauGas'][:] #Optical depth of gases at different altitude upto 120 km
    Gashgt = ds.variables['Z'][:-1]

    GasAbsTau =np.zeros(len(RSP_wlf))

    # GasTau = np.ones((len(Wl)))
    # for i in range(len(Wl)):
    #     GasTau[i] = np.trapz(Tau[:,i],Gashgt)

    GasTau = np.sum(Tau,axis=0)


    for n in range(len(RSP_wlf)):

    #     #Reading the spectral response value for RSP
        SRFFile = SpecResFnPath + str(RSP_wlf[n]) +'.txt'
        SpecResFn = np.loadtxt(SRFFile)

    #     #Spectral response values for given RSP wl
        SpecResFn = SpecResFn[SpecResFn[:,0]>= min(Wl)]
        #     #1D interpolation across wavelength
        f = interp1d(Wl,GasTau,kind = 'linear')
        #     # Evaluate the function at a new point
        wl_RSP = SpecResFn[:,0]
        tau_solar = f(wl_RSP) #Tau at given RSP response function wl
        NormSRF = SpecResFn[:,1]  #normalizing the response function

        tau = np.trapz(tau_solar*NormSRF,wl_RSP )
        swl= np.trapz(NormSRF,wl_RSP)

        GasAbsTau[n]= tau/swl
    plt.plot(RSP_wlf,GasAbsTau)



            #Reading the NetCDF file for gas absorption from radiative tranfer code (UNLVRTM)
        


        # Tau_hgt = Tau  #gas optical depth in the profile 
        # Gashgtp= Gashgt

        # #this array will store the total tau value
        # ColumnTau = np.sum(Tau,axis=0 )  #total column optical depth below the aircraaft altitiude. 
        # GasAbsTau = np.zeros((len(RSP_wlf))) #this array will store the total tau value

        # for n in range(len(RSP_wl)):
        #     #Reading the spectral response value for RSP
        #     SpecResFn = np.loadtxt(f'/Users/greema/Desktop/UMBC/ORACLES/RSP_Spectral_Response/{RSP_wl[n]}.txt')
        #     SpecResFn = SpecResFn[SpecResFn[:,0]>= min(Wl)]
        # #     #1D interpolation across wavelength
        #     f = interp1d(Wl,GasTau,kind = 'linear')
        #     #     # Evaluate the function at a new point
        #     wl_RSP = SpecResFn[:,0]
        #     tau_solar = f(wl_RSP) #Tau at given RSP response function wl
        #     NormSRF = SpecResFn[:,1]  #normalizing the response function

        #     tau = np.trapz(tau_solar*NormSRF,wl_RSP )
        #     swl= np.trapz(NormSRF,wl_RSP)

        #     GasAbsTau[n]= tau/swl

        # plt.plot(RSP_wl,GasAbsTau)
    # #Reading the NetCDF file for gas absorption from radiative tranfer code (UNLVRTM)
    # ds = nc.Dataset(GasAbsFn)
    # Wl = ds.variables['Lamdas'][:] # wavelength values corresponding to the gas absorption
    # Tau = ds.variables['tauGas'][:] #Optical depth of gases at different altitude upto 120 km
    # Gashgt = ds.variables['Z'][:-1]

    # # Tau_hgt = Tau[np.where(Gashgt<Airhgt)]  #gas optical depth in the profile 
    # # Gashgtp= Gashgt[np.where(Gashgt<Airhgt)]

    # Tau_hgt = Tau  #gas optical depth in the profile 
    # Gashgtp= Gashgt


    # ColumnTau = np.sum(Tau,axis=0 )  #total column optical depth below the aircraaft altitiude. 
    # GasAbsTau = np.ones((len(RSP_wlf))) #this array will store the total tau value

    # for n in range(len(RSP_wlf)):
    #     #Reading the spectral response value for RSP
    #     SRFFile = SpecResFnPath + str(RSP_wlf[n]) +'.txt'
    #     SpecResFn = np.loadtxt(SRFFile)

    #     #Spectral response values for given RSP wl
    #     SpecResFn = SpecResFn[SpecResFn[:,0]>= min(Wl)]
    #     #1D interpolation across wavelength
    #     f = interp1d(Wl,ColumnTau,kind = 'linear')
    #     # Evaluate the function at a new point
    #     wl_RSP = SpecResFn[:,0]
    #     tau_solar = f(wl_RSP) #Tau at given RSP response function wl
    #     NormSRF = SpecResFn[:,1]  #normalizing the response function


    #     tautotal = np.trapz(tau_solar*NormSRF,wl_RSP)
    #     swl= np.trapz(NormSRF,wl_RSP)
        
    #     GasAbsTau[n]= tautotal/swl

    #     plt.plot(RSP_wlf,GasAbsTau)
    
    return GasAbsTau

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
    
    anglesIdx = np.arange(ang1,ang2,1)
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
    vza = 180-f1_MAP['Geometry']['Viewing_Zenith'][TelNo,PixNo,anglesIdx]

    #This can be used to filter scattering angles if required
    vza[f1_MAP['Geometry']['Nadir_Index'][0,PixNo]:] = - vza[f1_MAP['Geometry']['Nadir_Index'][0,PixNo]:]
    Angfilter = (vza>= -45) & (vza<= 45) # taking only the values of view zenith from -65 to 45

    #Angles are checked for nans 
    Scattering_ang = checkFillVals(f1_MAP['Geometry']['Scattering_Angle'][TelNo,PixNo,anglesIdx]  )
    Solar_Zenith =  checkFillVals(f1_MAP['Geometry']['Solar_Zenith'][TelNo,PixNo,anglesIdx]  )
    
    #Converting sunlight azimuth to solar azimuth: 𝜃𝑠, 180- 𝜃𝑣 𝜙𝑠 = 𝜙𝑠 -180, 𝜙𝑣

    Solar_Azimuth = checkFillVals(f1_MAP['Geometry']['Solar_Azimuth'][TelNo,PixNo,anglesIdx], ) - 180
    Viewing_Azimuth = checkFillVals(f1_MAP['Geometry']['Viewing_Azimuth'][TelNo,PixNo,anglesIdx]  )
   
    #Converting viewing zenith with respect to nadir to that wrt zenith
    Viewing_Zenith = 180 - checkFillVals(f1_MAP['Geometry']['Viewing_Zenith'][TelNo,PixNo,anglesIdx]  ) # Theta_v <90
    
    #Converting values into radians to caculate the relative azimuth angles        
    sza =  np.radians(Solar_Zenith)
    vza =   np.radians(Viewing_Zenith)
    szi =  np.radians(Solar_Azimuth)
    vzi =  np.radians(Viewing_Azimuth)
    
    Relative_Azi = (180/np.pi)*(np.arccos((np.cos((Scattering_ang *np.pi)/180)  + np.cos(sza)*np.cos(vza))/(- np.sin(sza)*np.sin(vza)) ))
    #TODO
    # Relative_Azi = Solar_Azimuth - Viewing_Azimuth
    # for i in range (len(Relative_Azi)): 
    #     if Relative_Azi[i]<0 : Relative_Azi[i] =  Relative_Azi[i]+360
    RSP_wlf = [410, 470, 555, 670, 865] #wl as in the file name of response functions
    
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

    I1 = (checkFillVals(Data['Intensity_1'][PixNo,anglesIdx,:nwl]  , negative_check =True))# / corrFac telescope 1 Normalized intensity (unitless)#there are some negative intesity values in the file
    # I1 = I1/CorFac2
    I2 = (checkFillVals(Data['Intensity_2'][PixNo,anglesIdx,:nwl]  ,negative_check =True))# #telescope 2
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
    rslt['meas_P'] = checkFillVals(Data['DoLP'][PixNo,anglesIdx,:nwl]  ,negative_check =True)/100    #relative value P/I
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
    rslt['OBS_hght']= f1_MAP['Platform']['Platform_Altitude'][PixNo] # height of pixel in m
    # print(rslt['OBS_hght'])
    if  rslt['OBS_hght'] < 0:  #if colocated attitude is less than 0 then that is set to 0
        rslt['OBS_hght'] = 0
        print(f"The collocated height was { rslt['OBS_hght']}, OBS_hght was set to 0 ")
    
    rslt['gaspar'] =Gaspar_MAP(wl,rslt['OBS_hght']/1000,GasAbsFn,SpecResFnPath)[:nwl]
    print(rslt['gaspar'])
    f1_MAP.close()
    return rslt

def Read_Data_HSRL_Oracles(file_path,file_name,PixNo,Plot_avg_prof = None):

    f1= h5py.File(file_path + file_name,'r+')  #reading Lidar measurements 
    HSRL = f1['DataProducts']
    #Latitude and longitude values 
    latitude,longitude = f1['Nav_Data']['gps_lat'][:],f1['Nav_Data']['gps_lon'][:]
    AirAlt = f1['Nav_Data']['gps_alt'][PixNo] #Altitude of the aircraft
    print(AirAlt)

    del_dict ={} #This dictionary stores 
    inp = ['355_ext','532_ext','1064_ext','355_bsc','532_bsc','1064_bsc','355_dep', '532_dep','1064_dep']
    #Setting negative values to zero, The negative values are due to low signal so we can replace with 0 without loss of info.
    for i in range (len(inp)):
        del_dict[f'{inp[i]}'] = HSRL[f'{inp[i]}'][PixNo]
        HSRL_checkFillVals(del_dict[f'{inp[i]}']) # set all negative values to zero
        if (inp[i] == '355_dep') or (inp[i] == '532_dep') or (inp[i] == '1064_dep'):
            del_dict[f'{inp[i]}'] = np.where(HSRL[f'{inp[i]}'][PixNo][:]>= 0.6 , np.nan, HSRL[f'{inp[i]}'][PixNo])  #CH: This should be changed, 0.7 has been set arbitarily

    #Caculating range, Range is defined as distance from the instrument to the aersosol layer, i.e. range at instrument heright = 0. We have to make sure that The range is in decending order
    del_dict['Altitude'] = HSRL['Altitude'][0]  # altitude above sea level
    # del_dict['Altitude'] = HSRL['Altitude'][0]  
    df_new = pd.DataFrame(del_dict)
    df_new.interpolate(inplace=True, limit_area= 'inside')

    BLHIndx = np.where(np.gradient(df_new['1064_dep'],df_new['Altitude']) ==np.nanmax(np.gradient(df_new['1064_dep'],df_new['Altitude']))) #index of the BLH
    print(BLHIndx)
    #Height limits
    # BLh = 1200 #Boundary layer height 
    BLh = np.array(df_new['Altitude'][:])[BLHIndx]

    #Filtering and removing the pixels with bad data: 
    Removed_index = []  # Removed_index holds indices of pixels to be removed
    # Filter values greater than flight altitude
    Removed_index.append(np.nonzero(np.array(df_new['Altitude'][:]) > AirAlt)[0][:])
    # Removed_index.append(np.nonzero(np.array(df_new['Altitude'][:]) < 1000)[0][:]) #this removes the profile below 1000m 
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
    del_dictt = {}
    # Delete removed pixels and set negative values to zero for each data type
    inp2 = ['355_ext','532_ext','1064_ext','355_bsc','532_bsc','1064_bsc','355_dep', '532_dep','1064_dep', 'Altitude','FMF']
    for i in range (len(inp2)): #
        del_dictt[f'{inp2[i]}'] = np.delete(np.array(df_new[f'{inp2[i]}']), rm_pix)
    avgProf = pd.DataFrame() # Vertical Averaged Profiles

    #Averaging the vertical profile to reduce noise
    Humidity = f1["State"]['Relative_Humidity'][PixNo]*100

    
    hgt = del_dictt['Altitude']
    
    
    hgtInterv = 150  # new height interval, for now it is set to 150 m

    altitude_diff = np.gradient(del_dictt['Altitude'])
    numInterv = int(hgtInterv / np.mean(altitude_diff))  # Calculate numInterv based on mean altitude difference
    Npoint = int(len(del_dictt['Altitude'])/numInterv) # No of vertical bins after averaging
    
    # Create an array to store the averaged values
    for i in range(numInterv):
        strtVal = i*numInterv
        endVal = (i+1)*numInterv    

        hgtAvg = np.zeros(Npoint)
    for k in range (len(inp)):
        a = 0   # Indexing variable
        averaged_values = np.zeros(Npoint)

        for i in range(Npoint):
            start_range = hgt[i * numInterv]
            end_range = hgt[(i + 1) * numInterv]
            indexAvg= np.where((hgt >= start_range) & (hgt < end_range)) #Index of values in the start-stop range 
            
            #Taking mean
            if len(indexAvg[0]) > 0:
                averaged_values[a]= np.mean(np.array(df_new[f'{inp[k]}'])[indexAvg])
            
            if k ==0: # Avoiding repetition 
                hgtAvg[a]= np.mean(np.array(df_new['Altitude'])[indexAvg])
            a=a+1 # Indexing variable
        #Storing the values of profile
        avgProf[f'{inp[k]}'] = averaged_values #Averaged profile values with height >0 
        avgProf['Altitude'] =  hgtAvg
        
    ## GRASP requires the vertical profiles to be in decendong order, so reversing the entire profile
    df = avgProf[::-1]
    
    
    if Plot_avg_prof ==True:
        fig, axs = plt.subplots(nrows= 1, ncols=9, figsize=(20, 6), sharey = True)
        for i in range (0,len(inp)):
            axs[i].plot(del_dictt[f'{inp[i]}'],del_dictt['Altitude'], marker= '.', label = "Org" )
            axs[i].plot(df[f'{inp[i]}'],df['Altitude'], marker= '.' , label = "Avg Prof")

            axs[i].set_xlabel(f'{inp[i]}')
            
        axs[0].legend()
        axs[0].set_ylabel('Height m ')  
        
        
    #Creating GRASP rslt dictionary for runGRASP.py    

    
    rslt = {} # This dictionary will store the values arranged in GRASP's format. 
    height_shape = np.array(df['Altitude'][:]).shape[0] #to avoid the height of the sea salt, this should be removed 

    Range = np.ones((height_shape,3))
    Range[:,0] = df['Altitude'][:]
    Range[:,1] = df['Altitude'][:]
    Range[:,2] = df['Altitude'][:]  # in meters
    rslt['RangeLidar'] = Range

    Bext = np.ones((height_shape,3)) 
    Bext[:,0] = df['355_ext'][:]
    Bext[:,1] = df['532_ext'][:]
    Bext[:,2] = df['1064_ext'][:]
    Bext[0,2] = np.nan 

    Bsca = np.ones((height_shape,3))
    Bsca[:,0] = df['355_bsc'][:]
    Bsca[:,1] = df['532_bsc'] [:]
    Bsca[:,2] = df['1064_bsc'][:]

    # Bsca[0,2] = np.nan #Setting one of the value in the array to nan so that GRASP will discard this measurement, we are doing this for HSRL because it is not a direct measuremnt

    Dep = np.ones((height_shape,3))
    Dep[:,0] = df['355_dep'][:]
    Dep[:,1] = df['532_dep'][:]
    Dep[:,2] = df['1064_dep'] [:]

    #Unit conversion 
    rslt['meas_VExt'] = Bext / 1000
    rslt['meas_VBS'] = Bsca / 1000 # converting units from km-1 to m-1
    rslt['meas_DP'] = Dep *100  # in percentage
    # print(rslt['meas_DP'])

    rslt['lambda'] = np.array([355,532,1064])/1000 #values of HSRL wl in um
    rslt['wl'] = np.array([355,532,1064])/1000
    rslt['datetime'] =dt.datetime.strptime(str(int(f1["header"]['date'][0][0]))+ str(f1['Nav_Data']['UTCtime2'][PixNo][0]),'%Y%m%d%H%M%S.%f')
    rslt['latitude'] = latitude[PixNo]
    rslt['longitude']= longitude[PixNo]
    rslt['OBS_hght']=  AirAlt# aircraft altitude in m
    rslt['land_prct'] = 0 #Ocean Surface

    #Substitude the actual value
    # rslt['gaspar'] = np.ones((3))*0.0037 #MOlecular depolarization 
    f1.close() 
    return rslt, BLh

#Read_Data_HSRL_Oracles Version .2 
# Using different averaging method for vertical profile averaging 
def Read_HARP2(file_path,file_name,PixNo,ang1,ang2):
    anglesIdx = np.arange(ang1,ang2,1)
    #Reading the hdf file
    f1_MAP = h5py.File(file_path + file_name,'r+') 
    #Creating rslt dictionary for GRASP
    
    
    rslt ={}
    rslt['lambda'] = Data['Wavelength'][:nwl]/1000 # Wavelengths in um
    rslt['longitude'] = Lon
    rslt['latitude'] = Lat
    rslt['meas_I'] = (I1+I2)/2  

    '''This should be changed  '''

    # rslt['meas_P'] = rslt['meas_I'] *checkFillVals(Data['DoLP'][PixNo,ang1:ang2,:nwl]  ,negative_check =True)/100
    rslt['meas_P'] = checkFillVals(Data['DoLP'][PixNo,anglesIdx,:nwl]  ,negative_check =True)/100    #relative value P/I
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
    rslt['OBS_hght']= f1_MAP['Platform']['Platform_Altitude'][PixNo] # height of pixel in m
    # print(rslt['OBS_hght'])
    if  rslt['OBS_hght'] < 0:  #if colocated attitude is less than 0 then that is set to 0
        rslt['OBS_hght'] = 0
        print(f"The collocated height was { rslt['OBS_hght']}, OBS_hght was set to 0 ")
    
    rslt['gaspar'] =Gaspar_MAP(wl,rslt['OBS_hght']/1000,GasAbsFn,SpecResFnPath)
    print(rslt['gaspar'])
    f1_MAP.close()
    return rslt



def Read_Data_HSRL_Oracles_Height(file_path,file_name,PixNo,gaspar =None,SimpleCase =None):

    """Note: This is the function that is being currently used to run GRASP for HSRL (ORACLES)"""

    #Specify the Path and the name of the HSRL file
    #Reading the HSRL data
    f1= h5py.File(file_path + file_name,'r+')  #reading Lidar measurements 

    HSRL = f1['DataProducts']
    Temp =  f1['State']['Temperature'][:]
    AirAlt = f1['Nav_Data']['gps_alt'][PixNo] #Altitude of the aircraft
    AerID = HSRL['Aerosol_ID'][PixNo][:]
    hgt = HSRL['Altitude'][:].flatten()
    
    DMR = HSRL['Dust_Mixing_Ratio'][:]
    DMR[np.where(np.isnan(DMR))[0]] = 0

    #Extinding the value of Vext to the aircraft altitude. To do so we assume lidar ratio closer to the aircraft to be constant, Vext at h > ~4000 to HAircradt  = LR * VBsc
    Vext532  = HSRL['532_ext'][PixNo][:]
    Vext355  = HSRL['355_ext'][PixNo][:]

    Bsca532  = HSRL['532_bsc'][PixNo][:]
    Bsca355  = HSRL['355_bsc'][PixNo][:]

    NanHgt355 = hgt[(np.where(np.isnan(Vext355)))[0]]  #Finding the index of the first nan value. 
    MaxExtIdx355 = int(np.where(hgt ==np.nanmin(NanHgt355[NanHgt355>3000]))[0]) -10

    NanHgt532 = hgt[(np.where(np.isnan(Vext532)))[0]]  #Finding the index of the first nan value. 
    MaxExtIdx532 = int(np.where(hgt == np.nanmin(NanHgt532[NanHgt532>3000]))[0]) -10
   
    # Calculating the lodar ratio
    LR_532_cal =Vext532/Bsca532
    LR_355_cal =Vext355/Bsca355

    #Caculatig the height above which the lidar ratio is too high:

    LR_max = hgt[(np.where(LR_532_cal>=100))[0]]
    print('Height above which LR at 532> 100', np.nanmin(LR_max[LR_max>4000]))

    plt.plot(LR_532_cal ,hgt)
    plt.plot(LR_355_cal,hgt)

    # Assuming the lidar ratio to be constant
    # Vext532hmax = LR_532_cal[MaxExtIdx532-10]*Bsca532[MaxExtIdx355-2:]
    # Vext355hmax  = LR_355_cal[MaxExtIdx355-10]*Bsca355[MaxExtIdx355-2:]
    # # Vext532[MaxExtIdx355-2:] = Vext532hmax 

    #Color ratio

    ColorRatioA =Vext355/Vext532
    ColorRatioA =Vext532/Vext355  #for 532

    # Vext355hmax  = LR_355_cal[MaxExtIdx355-10]*Bsca355[MaxExtIdx355:]
    ColorRatio = np.mean(ColorRatioA[np.where((hgt>4000) & (hgt<4550))[0]])
    # ColorRatio = np.nanmax(ColorRatioA)
    print(ColorRatio)
    # # LR_355_cal =Vext355/Bsca355
    # #Assuming the lidar ratio to be constant
    # # Vext532hmax = ColorRatio[MaxExtIdx355-2]*Vext355hmax[MaxExtIdx355-1:]
    Vext355hmax = ColorRatio*Vext532[MaxExtIdx355-2:]
    
     
    #Replacing the nan values by the caculate Vext from LR and Backscatter
    # Vext532[MaxExtIdx355-2:] = Vext532hmax 
    Vext355[MaxExtIdx355-2:] = Vext355hmax

    # DMRpar = HSRL['532_aer_dep'][PixNo][:]
    # DMRpar[MaxExtIdx355-1:] = HSRL['1064_dep'][PixNo][MaxExtIdx355-1:]


    # fig,axs  = plt.subplots(1,2, figsize =(10,6), sharey =True)

    # axs[0].plot(Vext532hmax, hgt[MaxExtIdx532-1:],marker ='o',c= '#888859', label ='cal')
    # axs[0].plot(HSRL['532_ext'][PixNo][:],hgt,c='k')


    # axs[1].plot(Vext355hmax,hgt[MaxExtIdx355-1:],c= '#888859',marker ='o', label='cal')
    # axs[1].plot(HSRL['355_ext'][PixNo][:],hgt,c='k')


    # axs[0].set_xlabel('532 Ext km-1')
    # axs[1].set_xlabel('355 Ext km-1')

    # axs[0].set_ylabel('Altitude m')
    # # axs[1].set_ylabel('Altitude m')

    # # print(HSRL['532_ext'][PixNo][:])

    # # print(Vext532hmax )
    # axs[0].legend()
    # axs[1].legend()
    


    NanHgt = hgt[(np.where(np.isnan(HSRL['532_ext'][PixNo][:])))[0]] #finding the upper limit of the profile height (filtering for high noise) 
    # UpH =   np.nanmin(NanHgt[NanHgt>3000]) 

    #Index of values below aircraft and above ground
    hmaxInd = np.min(np.where(HSRL['Altitude'][0] >=AirAlt)[0])  #-1000 os added to avoid values vary close to the aircraft
   
    #Minimun index is the max height where the values are not nan, taking 500 as a dummy value for height. 
    hminInd = int(np.where(hgt ==np.nanmax(NanHgt[NanHgt<1000]))[0]) # -1 is to set the min value to 0 so that we dont have extra aod
    
    print('Height range of the profile',hmaxInd,hminInd)
    print('Height range of the profile',AirAlt)

    # print(AerID)
    # HSRL['Dust_Mixing_Ratio'][PixNo][:][np.where(AerID > 8)] = np.nanmax(HSRL['Dust_Mixing_Ratio'][PixNo][:])
    # print(HSRL['Dust_Mixing_Ratio'][PixNo][:][np.where(AerID > 8)])
    #Horizontal Averaging the Angstrom exponent data to reduce noise  and caculating the fine mode fraction
    # AEpix5= HSRL['Angstrom_532_355'][PixNo-5:PixNo+5][:]
   
    AEpix5= HSRL['Angstrom_532_355'][PixNo-3:PixNo+3][:]/HSRL['Angstrom_Spherical'][PixNo-3:PixNo+3][:]
    avgFMF = np.nanmean(AEpix5 , axis = 0)

    # Caculating the fine mode froaction of 

    # Voth532 = (1-DMR[PixNo-3:PixNo+3])*HSRL['532_ext'][PixNo-3:PixNo+3]
    # Voth355 = (1-DMR[PixNo-3:PixNo+3])*HSRL['355_ext'][PixNo-3:PixNo+3]
    # AngVoth = -np.log(Voth532/Voth355)/np.log(532/355)
    # FMFVoth = AngVoth/HSRL['Angstrom_Spherical'][PixNo-3:PixNo+3][:]
    
    Voth532 = (1-DMR[PixNo])*Vext532
    Voth355 = (1-DMR[PixNo])*Vext355
    # Voth355 = (1-DMR[PixNo-3:PixNo+3])*HSRL['355_ext'][PixNo-3:PixNo+3]

    AngVoth = -np.log(Voth532/Voth355)/np.log(532/355)
    FMFVoth = AngVoth/HSRL['Angstrom_Spherical'][PixNo][:]
    AvgFMFVoth = FMFVoth 

    

   
    Data_dic ={} #This dictionary stores all the HSRL variables used for GRASP forward simulation
    inp = ['355_ext','532_ext','1064_ext','355_bsc','532_bsc','1064_bsc','355_dep', '532_dep','1064_dep','Dust_Mixing_Ratio' ,'532_aer_dep']
    inp2 = ['355_ext','532_ext','1064_ext','355_bsc','532_bsc','1064_bsc','355_dep', '532_dep','1064_dep', 'Altitude','Dust_Mixing_Ratio','532_aer_dep','FMF'] #Key words for Datadic
    
    
    Data_dic['Altitude'] = HSRL2_checkFillVals(HSRL['Altitude'][0][hminInd:hmaxInd])  # Height of the aerosol layer from the sea level
    # Data_dic['FMF'] = HSRL2_checkFillVals(avgFMF[hminInd:hmaxInd])
    Data_dic['FMF'] = HSRL2_checkFillVals(AvgFMFVoth[hminInd:hmaxInd])
    

    #Setting negative values to zero, The negative values are due to low signal so we can replace with 0 without loss of info.
    
    
    for i in range (len(inp)):
        

        Data_dic[f'{inp[i]}'] = HSRL[f'{inp[i]}'][PixNo][hminInd:hmaxInd] #Values within the reasonable heights
        df_new = pd.DataFrame(Data_dic)
    
    # print(Data_dic['532_ext'][:])

 #Specify the Path and the name of the HSRL file
    #Reading the HSRL data
    #Dividing the height into 3 layers> Boundary layer, dust dominated layer( for ORACLES case, spt 22, 2018), and high layer
    Data_dic['532_ext'] = Vext532[hminInd:hmaxInd]
    Data_dic['355_ext'] = Vext355[hminInd:hmaxInd]
    # Data_dic['532_aer_dep'] =DMRpar[hminInd:hmaxInd] #Replacing the nan values of aerosol DP at 532 with aer DP at 1064 #TODO change this 

    # Data_dic['532_aer_dep'][np.where(Data_dic['Altitude']>hgtNanHgt355 )[0]] = Data_dic['532_dep'][hminInd:hmaxInd][np.where(Data_dic['Altitude']>NanHgt355 )[0]]
    

    del_dictt = Data_dic     
    #Calculating the marine boundary layer height, Because we are interested in aerosol shape, usin Dp as a parameter to determine boundary layer height
    
    """Another potential parameter for boundary layer height is the relative humidity, as it potentially helps to differentiate the layer with spherical and non-spherical layer
    Note: this is this case specific (marine in the boundary layer (spherical) and dust layer above(NonSph)) and cannot be applied to other cases
    """

    #TODO : cacluating blh with respect to temp or humidity

    BLHIndx = np.where(np.gradient(df_new['1064_dep'],df_new['Altitude']) ==np.nanmax(np.gradient(df_new['1064_dep'],df_new['Altitude']))) #index of the BLH
    # print(BLHIndx)
    #Height limits
    # BLh = 1200 #Boundary layer height 
    BLh = np.array(df_new['Altitude'][:])[BLHIndx] #Boundary layer height 
    # print(df_new['Altitude'][:])
    # print(BLh)
    # fig,axs  = plt.subplots(1,2, figsize =(10,6), sharey =True)
    # axs[0].plot(Data_dic['532_ext'][:],Data_dic['Altitude'],c='k')
    # axs[1].plot(Data_dic['355_ext'][:],Data_dic['Altitude'],c='k')


    # axs[0].set_xlabel('532 Ext km-1')
    # axs[1].set_xlabel('355 Ext km-1')

    # axs[0].set_ylabel('Altitude m')
    # # axs[1].set_ylabel('Altitude m')


    # axs[0].legend()
    # axs[1].legend()
    



     #3000 is a dummy value 

    # UpH = np.nanmax(df_new['Altitude'][:]) # avoid low signals
    # BLh =300
    belowBL ={}
    MidAtm={}
    UpAtm={}

    #TODO make this part more general
    #No of height grids: divinding the profile into two layer: below and above the boundary layer heights.
   
    MidInv = 150 #150 #No of grids below the boundary layer
    UpInv = 800
    BLIntv = 110 #No of grids below the boundary layer
    avgProf = pd.DataFrame()
    hgt = del_dictt['Altitude']
   
    # UpH = hgt[MaxExtIdx355] #Height where Vext355 is nan

    UpH = np.nanmin(LR_max[LR_max>4000]) #Height where LR at 532>100
    # UpH = hgt[MaxExtIdx532] #Height where Vext355 is nan

    #Dividing the profile  #This needs to be modified
    for i in range (len(inp2)): 
        belowBL[f'{inp2[i]}'] = del_dictt[f'{inp2[i]}'][np.where(hgt<BLh)]
        # MidAtm[f'{inp2[i]}'] = del_dictt[f'{inp2[i]}'][np.where((hgt>= BLh))]
        MidAtm[f'{inp2[i]}'] = del_dictt[f'{inp2[i]}'][np.where((hgt>= BLh) &(hgt< UpH))]
        UpAtm[f'{inp2[i]}'] = del_dictt[f'{inp2[i]}'][np.where(hgt>= UpH)]
    
    #Average profiles
    BLProf= VertP(belowBL, BLIntv,inp2)
    MidProf= VertP(MidAtm, MidInv,inp2)
    UpProf = VertP(UpAtm, UpInv,inp2)
    AppendZeroTOA = 0

    plt.plot()
    # UpProf= VertP(UpAtm, UpInv)
    inp = ['355_ext','532_ext','1064_ext','355_bsc','532_bsc','1064_bsc','355_dep', '532_dep','1064_dep', 'Dust_Mixing_Ratio' ,'FMF','532_aer_dep']

    FullAvgProf = {}
    for i in range (len(inp2)): 
        FullAvgProf[inp2[i]] =  np.concatenate((np.array(UpProf[inp2[i]]), np.array(MidProf[inp2[i]]),np.array(BLProf[inp2[i]])), axis=0)
        # FullAvgProf[inp2[i]] =  np.concatenate(( np.array(MidProf[inp2[i]]),np.array(BLProf[inp2[i]])), axis=0)
        # GRASP extrapolates the top ofthe profile to TOA, to stop that we will append a 0 point on top of the profile.
        if inp2[i] != 'Altitude':
            FullAvgProf[inp2[i]][0] = AppendZeroTOA  # The HSRL datahas artifacts/ noisy at altitudes near the aircraft. 
        if 'dep' in inp2[i]:
            FullAvgProf[inp2[i]][0] = 0.0037
            

    # FullAvgProf['Altitude'][0] = AirAlt-1000
        

        # FullAvgProf[inp2[i]] =  np.concatenate(( np.array(MidProf[inp2[i]]),np.array(BLProf[inp2[i]])), axis=0)
        # FullAvgProf[inp2[i]] =   np.array(MidProf[inp2[i]])
       
    
    fig, axs = plt.subplots(nrows= 1, ncols=len(inp), figsize=(20, 6), sharey = True)
    for i in range (0,len(inp)):
        axs[i].plot(del_dictt[f'{inp[i]}'],del_dictt['Altitude'], marker= '.', label = "Org" )
        axs[i].plot(FullAvgProf[f'{inp[i]}'],FullAvgProf['Altitude'], marker= '.' , label = "Avg Prof")

        axs[i].set_xlabel(f'{inp[i]}')

    axs[0].legend()
    axs[0].set_ylabel('Height m ') 


    fig, axs = plt.subplots(nrows= 1, ncols=len(inp), figsize=(20, 6), sharey = True)
    for i in range (0,len(inp)):
        axs[i].plot(del_dictt[f'{inp[i]}'],del_dictt['Altitude'], marker= '.', label = "Org" )
        # axs[i].plot(belowBL[f'{inp[i]}'],belowBL['Altitude'], marker= '.' , label = "bl")
        axs[i].plot(MidAtm[f'{inp[i]}'],MidAtm['Altitude'], marker= '.' , label = "bl")
        # axs[i].plot(UpAtm[f'{inp[i]}'],UpAtm['Altitude'], marker= '.' , label = "bl")
        axs[i].set_xlabel(f'{inp[i]}')
    axs[0].legend()
    axs[0].set_ylabel('Height m ') 
    # Creating GRASP rslt dictionary for runGRASP.py    
    
    
    
    df = pd.DataFrame(FullAvgProf)
    # df = pd.DataFrame(df_new)
    rslt = {} # 

    DstMR = df['Dust_Mixing_Ratio'][:]
    # AEsph = df['Angstrom_Spherical'][:]
    aDP= df['532_aer_dep'][:]

    AEt= df['FMF'][:]

    #dust mixing ratio from paper
   
    
    height_shape = np.array(df['Altitude'][:]).shape[0]   #setting the lowermos value to zero to avoid GRASP intgration

    Range = np.ones((height_shape,3))
    Range[:,0] = df['Altitude'][:]
    Range[:,1] = df['Altitude'][:]
    Range[:,2] = df['Altitude'][:]  # in meters
    rslt['RangeLidar'] = Range
    

    Bext = np.zeros((height_shape,3))
    Bext[:,0] = df['355_ext'][:]
    Bext[:,1] = df['532_ext'][:]
    Bext[:,2] = df['1064_ext'] [:]
    Bext[0,2] = np.nan  #Setting one of the value in the array to nan so that GRASP will discard this measurement

    Bsca = np.zeros((height_shape,3))
    Bsca[:,0] = df['355_bsc'][:]
    Bsca[:,1] = df['532_bsc'] [:]
    Bsca[:,2] = df['1064_bsc'][:]
    # Bsca[0,2] = np.nan
  
    
    Dep = np.zeros((height_shape,3))
    # Dep[-2:,0] = df['355_dep'][height_shape-2:] #Total depolarization ratio
    # Dep[-2:,1] = df['532_dep'][height_shape-2:]
    # Dep[-2:,2] = df['1064_dep'] [height_shape-2:]
    
    Dep[:,0] = df['355_dep'][:]  #Total depolarization ratio
    Dep[:,1] = df['532_dep'][:]
    Dep[:,2] = df['1064_dep'] [:]
    

    #Unit conversion 
    rslt['meas_VExt'] = Bext / 1000
    rslt['meas_VBS'] = Bsca / 1000 # converting units from km-1 to m-1
    rslt['meas_DP'] = Dep *100  # in percentage


    rslt['lambda'] = np.array([355,532,1064])/1000 #values of HSRL wl in um
    rslt['wl'] = np.array([355,532,1064])/1000
    rslt['datetime'] =dt.datetime.strptime(str(int(f1["header"]['date'][0][0]))+ str(f1['Nav_Data']['UTCtime2'][PixNo][0]),'%Y%m%d%H%M%S.%f')
    rslt['latitude'] = f1['Nav_Data']['gps_lat'][PixNo]
    rslt['longitude']= f1['Nav_Data']['gps_lon'][PixNo]
    rslt['OBS_hght']= AirAlt # aircraft altitude in m
    rslt['land_prct'] = 0 #Ocean Surface

    # if gaspar ==True: # Molecular depolarization correction 
    rslt['gaspar'] = np.array([0.0037,0.0037,0.0037])

    if SimpleCase == True:
        Bsca[0,2] = np.nan #Setting one of the value in the array to nan so that GRASP will discard this measurement, we are doing this for HSRL because it is not a direct measuremnt
        Bext[0,2] = np.nan  #Setting one of the value in the array to nan so that GRASP will discard this measurement

        
    
     # TODO check the units for molecular depol correction.   
    # TODO make this general

    f1.close()
    return rslt,BLh,DstMR,aDP,AEt,AerID, Temp






def Read_Data_HSRL_Oracles_Height_No355(file_path,file_name,PixNo,gaspar =None,SimpleCase =None):

    """Note: This is the function that is being currently used to run GRASP for HSRL (ORACLES)"""

    #Specify the Path and the name of the HSRL file
    #Reading the HSRL data
    f1= h5py.File(file_path + file_name,'r+')  #reading Lidar measurements 

    HSRL = f1['DataProducts']
    Temp =  f1['State']['Temperature'][:]
    AirAlt = f1['Nav_Data']['gps_alt'][PixNo] #Altitude of the aircraft
    AerID = HSRL['Aerosol_ID'][PixNo][:]
    hgt = HSRL['Altitude'][:].flatten()
    
    DMR = HSRL['Dust_Mixing_Ratio'][:]
    DMR[np.where(np.isnan(DMR))[0]] = 0

    #Extinding the value of Vext to the aircraft altitude. To do so we assume lidar ratio closer to the aircraft to be constant, Vext at h > ~4000 to HAircradt  = LR * VBsc
    Vext532  = HSRL['532_ext'][PixNo][:]
    Vext355  = HSRL['355_ext'][PixNo][:]

    Bsca532  = HSRL['532_bsc'][PixNo][:]
    Bsca355  = HSRL['355_bsc'][PixNo][:]

    # NanHgt355 = hgt[(np.where(np.isnan(Vext355)))[0]]  #Finding the index of the first nan value. 
    # MaxExtIdx355 = int(np.where(hgt ==np.nanmin(NanHgt355[NanHgt355>3000]))[0]) -10

    NanHgt532 = hgt[(np.where(np.isnan(Vext532)))[0]]  #Finding the index of the first nan value. 
    MaxExtIdx532 = int(np.where(hgt == np.nanmin(NanHgt532[NanHgt532>3000]))[0]) -10
   
    # Calculating the lodar ratio
    LR_532_cal =Vext532/Bsca532
    # LR_355_cal =Vext355/Bsca355

    #Caculatig the height above which the lidar ratio is too high:

    LR_max = hgt[(np.where(LR_532_cal>=100))[0]]
    print('Height above which LR at 532> 100', np.nanmin(LR_max[LR_max>4000]))

    plt.plot(LR_532_cal ,hgt)
    # plt.plot(LR_355_cal,hgt)

    # # Assuming the lidar ratio to be constant
    # Vext532hmax = LR_532_cal[MaxExtIdx532-10]*Bsca532[MaxExtIdx355-2:]
    # Vext355hmax  = LR_355_cal[MaxExtIdx355-10]*Bsca355[MaxExtIdx355-2:]
    # # Vext532[MaxExtIdx355-2:] = Vext532hmax 

    #Color ratio

    # ColorRatioA =Vext355/Vext532
    # ColorRatioA =Vext532/Vext355  #for 532

    # # # Vext355hmax  = LR_355_cal[MaxExtIdx355-10]*Bsca355[MaxExtIdx355:]
    # ColorRatio = np.mean(ColorRatioA[np.where((hgt>4000) & (hgt<4550))[0]])
    # # ColorRatio = np.nanmax(ColorRatioA)
    # print(ColorRatio)
    # # # LR_355_cal =Vext355/Bsca355
    # # #Assuming the lidar ratio to be constant
    # # # Vext532hmax = ColorRatio[MaxExtIdx355-2]*Vext355hmax[MaxExtIdx355-1:]
    # Vext355hmax = ColorRatio*Vext532[MaxExtIdx355-2:]
    
     
    # #Replacing the nan values by the caculate Vext from LR and Backscatter
    # # Vext532[MaxExtIdx355-2:] = Vext532hmax 
    # Vext355[MaxExtIdx355-2:] = Vext355hmax

    # # DMRpar = HSRL['532_aer_dep'][PixNo][:]
    # DMRpar[MaxExtIdx355-1:] = HSRL['1064_dep'][PixNo][MaxExtIdx355-1:]


    # fig,axs  = plt.subplots(1,2, figsize =(10,6), sharey =True)

    # axs[0].plot(Vext532hmax, hgt[MaxExtIdx532-1:],marker ='o',c= '#888859', label ='cal')
    # axs[0].plot(HSRL['532_ext'][PixNo][:],hgt,c='k')


    # axs[1].plot(Vext355hmax,hgt[MaxExtIdx355-1:],c= '#888859',marker ='o', label='cal')
    # axs[1].plot(HSRL['355_ext'][PixNo][:],hgt,c='k')


    # axs[0].set_xlabel('532 Ext km-1')
    # axs[1].set_xlabel('355 Ext km-1')

    # axs[0].set_ylabel('Altitude m')
    # # axs[1].set_ylabel('Altitude m')

    # # print(HSRL['532_ext'][PixNo][:])

    # # print(Vext532hmax )
    # axs[0].legend()
    # axs[1].legend()
    


    NanHgt = hgt[(np.where(np.isnan(HSRL['532_ext'][PixNo][:])))[0]] #finding the upper limit of the profile height (filtering for high noise) 
    # UpH =   np.nanmin(NanHgt[NanHgt>3000]) 

    #Index of values below aircraft and above ground
    hmaxInd = np.min(np.where(HSRL['Altitude'][0] >=AirAlt)[0])  #-1000 os added to avoid values vary close to the aircraft
   
    #Minimun index is the max height where the values are not nan, taking 500 as a dummy value for height. 
    hminInd = int(np.where(hgt ==np.nanmax(NanHgt[NanHgt<1000]))[0]) # -1 is to set the min value to 0 so that we dont have extra aod
    
    print('Height range of the profile',hmaxInd,hminInd)
    print('Height range of the profile',AirAlt)

    # print(AerID)
    # HSRL['Dust_Mixing_Ratio'][PixNo][:][np.where(AerID > 8)] = np.nanmax(HSRL['Dust_Mixing_Ratio'][PixNo][:])
    # print(HSRL['Dust_Mixing_Ratio'][PixNo][:][np.where(AerID > 8)])
    #Horizontal Averaging the Angstrom exponent data to reduce noise  and caculating the fine mode fraction
    # AEpix5= HSRL['Angstrom_532_355'][PixNo-5:PixNo+5][:]
   
    AEpix5= HSRL['Angstrom_532_355'][PixNo-3:PixNo+3][:]/HSRL['Angstrom_Spherical'][PixNo-3:PixNo+3][:]
    avgFMF = np.nanmean(AEpix5 , axis = 0)

    # Caculating the fine mode froaction of 

    # Voth532 = (1-DMR[PixNo-3:PixNo+3])*HSRL['532_ext'][PixNo-3:PixNo+3]
    # Voth355 = (1-DMR[PixNo-3:PixNo+3])*HSRL['355_ext'][PixNo-3:PixNo+3]
    # AngVoth = -np.log(Voth532/Voth355)/np.log(532/355)
    # FMFVoth = AngVoth/HSRL['Angstrom_Spherical'][PixNo-3:PixNo+3][:]
    
    Voth532 = (1-DMR[PixNo])*Vext532
    Voth355 = (1-DMR[PixNo])*Vext355
    # Voth355 = (1-DMR[PixNo-3:PixNo+3])*HSRL['355_ext'][PixNo-3:PixNo+3]

    AngVoth = -np.log(Voth532/Voth355)/np.log(532/355)
    FMFVoth = AngVoth/HSRL['Angstrom_Spherical'][PixNo][:]
    AvgFMFVoth = FMFVoth 

    

   
    Data_dic ={} #This dictionary stores all the HSRL variables used for GRASP forward simulation
    inp = ['355_ext','532_ext','1064_ext','355_bsc','532_bsc','1064_bsc','355_dep', '532_dep','1064_dep','Dust_Mixing_Ratio' ,'532_aer_dep']
    inp2 = ['355_ext','532_ext','1064_ext','355_bsc','532_bsc','1064_bsc','355_dep', '532_dep','1064_dep', 'Altitude','Dust_Mixing_Ratio','532_aer_dep','FMF'] #Key words for Datadic
    
    
    Data_dic['Altitude'] = HSRL2_checkFillVals(HSRL['Altitude'][0][hminInd:hmaxInd])  # Height of the aerosol layer from the sea level
    # Data_dic['FMF'] = HSRL2_checkFillVals(avgFMF[hminInd:hmaxInd])
    Data_dic['FMF'] = HSRL2_checkFillVals(AvgFMFVoth[hminInd:hmaxInd])
    

    #Setting negative values to zero, The negative values are due to low signal so we can replace with 0 without loss of info.
    
    
    for i in range (len(inp)):
        

        Data_dic[f'{inp[i]}'] = HSRL[f'{inp[i]}'][PixNo][hminInd:hmaxInd] #Values within the reasonable heights
        df_new = pd.DataFrame(Data_dic)
    
    # print(Data_dic['532_ext'][:])

 #Specify the Path and the name of the HSRL file
    #Reading the HSRL data
    #Dividing the height into 3 layers> Boundary layer, dust dominated layer( for ORACLES case, spt 22, 2018), and high layer
    Data_dic['532_ext'] = Vext532[hminInd:hmaxInd]
    Data_dic['355_ext'] = Vext355[hminInd:hmaxInd]
    # Data_dic['532_aer_dep'] =DMRpar[hminInd:hmaxInd] #Replacing the nan values of aerosol DP at 532 with aer DP at 1064 #TODO change this 

    # Data_dic['532_aer_dep'][np.where(Data_dic['Altitude']>hgtNanHgt355 )[0]] = Data_dic['532_dep'][hminInd:hmaxInd][np.where(Data_dic['Altitude']>NanHgt355 )[0]]
    

    del_dictt = Data_dic     
    #Calculating the marine boundary layer height, Because we are interested in aerosol shape, usin Dp as a parameter to determine boundary layer height
    
    """Another potential parameter for boundary layer height is the relative humidity, as it potentially helps to differentiate the layer with spherical and non-spherical layer
    Note: this is this case specific (marine in the boundary layer (spherical) and dust layer above(NonSph)) and cannot be applied to other cases
    """

    #TODO : cacluating blh with respect to temp or humidity

    BLHIndx = np.where(np.gradient(df_new['1064_dep'],df_new['Altitude']) ==np.nanmax(np.gradient(df_new['1064_dep'],df_new['Altitude']))) #index of the BLH
    # print(BLHIndx)
    #Height limits
    # BLh = 1200 #Boundary layer height 
    BLh = np.array(df_new['Altitude'][:])[BLHIndx] #Boundary layer height 
    # print(df_new['Altitude'][:])
    # print(BLh)
    # fig,axs  = plt.subplots(1,2, figsize =(10,6), sharey =True)
    # axs[0].plot(Data_dic['532_ext'][:],Data_dic['Altitude'],c='k')
    # axs[1].plot(Data_dic['355_ext'][:],Data_dic['Altitude'],c='k')


    # axs[0].set_xlabel('532 Ext km-1')
    # axs[1].set_xlabel('355 Ext km-1')

    # axs[0].set_ylabel('Altitude m')
    # # axs[1].set_ylabel('Altitude m')


    # axs[0].legend()
    # axs[1].legend()
    



     #3000 is a dummy value 

    # UpH = np.nanmax(df_new['Altitude'][:]) # avoid low signals
    # BLh =300
    belowBL ={}
    MidAtm={}
    UpAtm={}

    #TODO make this part more general
    #No of height grids: divinding the profile into two layer: below and above the boundary layer heights.
   
    MidInv = 150 #150 #No of grids below the boundary layer
    UpInv = 800
    BLIntv = 110 #No of grids below the boundary layer
    avgProf = pd.DataFrame()
    hgt = del_dictt['Altitude']
   
    # UpH = hgt[MaxExtIdx355] #Height where Vext355 is nan

    UpH = np.nanmin(LR_max[LR_max>4000]) #Height where LR at 532>100
    # UpH = hgt[MaxExtIdx532] #Height where Vext355 is nan

    #Dividing the profile  #This needs to be modified
    for i in range (len(inp2)): 
        belowBL[f'{inp2[i]}'] = del_dictt[f'{inp2[i]}'][np.where(hgt<BLh)]
        # MidAtm[f'{inp2[i]}'] = del_dictt[f'{inp2[i]}'][np.where((hgt>= BLh))]
        MidAtm[f'{inp2[i]}'] = del_dictt[f'{inp2[i]}'][np.where((hgt>= BLh) &(hgt< UpH))]
        UpAtm[f'{inp2[i]}'] = del_dictt[f'{inp2[i]}'][np.where(hgt>= UpH)]
    
    #Average profiles
    BLProf= VertP(belowBL, BLIntv,inp2)
    MidProf= VertP(MidAtm, MidInv,inp2)
    UpProf = VertP(UpAtm, UpInv,inp2)
    AppendZeroTOA = 0

    plt.plot()
    # UpProf= VertP(UpAtm, UpInv)
    inp = ['355_ext','532_ext','1064_ext','355_bsc','532_bsc','1064_bsc','355_dep', '532_dep','1064_dep', 'Dust_Mixing_Ratio' ,'FMF','532_aer_dep']

    FullAvgProf = {}
    for i in range (len(inp2)): 
        FullAvgProf[inp2[i]] =  np.concatenate((np.array(UpProf[inp2[i]]), np.array(MidProf[inp2[i]]),np.array(BLProf[inp2[i]])), axis=0)
        # FullAvgProf[inp2[i]] =  np.concatenate(( np.array(MidProf[inp2[i]]),np.array(BLProf[inp2[i]])), axis=0)
        # GRASP extrapolates the top ofthe profile to TOA, to stop that we will append a 0 point on top of the profile.
        if inp2[i] != 'Altitude':
            FullAvgProf[inp2[i]][0] = AppendZeroTOA  # The HSRL datahas artifacts/ noisy at altitudes near the aircraft. 
        if 'dep' in inp2[i]:
            FullAvgProf[inp2[i]][0] = 0.0037
            

    # FullAvgProf['Altitude'][0] = AirAlt-1000
        

        # FullAvgProf[inp2[i]] =  np.concatenate(( np.array(MidProf[inp2[i]]),np.array(BLProf[inp2[i]])), axis=0)
        # FullAvgProf[inp2[i]] =   np.array(MidProf[inp2[i]])
       
    
    fig, axs = plt.subplots(nrows= 1, ncols=len(inp), figsize=(20, 6), sharey = True)
    for i in range (0,len(inp)):
        axs[i].plot(del_dictt[f'{inp[i]}'],del_dictt['Altitude'], marker= '.', label = "Org" )
        axs[i].plot(FullAvgProf[f'{inp[i]}'],FullAvgProf['Altitude'], marker= '.' , label = "Avg Prof")

        axs[i].set_xlabel(f'{inp[i]}')

    axs[0].legend()
    axs[0].set_ylabel('Height m ') 


    fig, axs = plt.subplots(nrows= 1, ncols=len(inp), figsize=(20, 6), sharey = True)
    for i in range (0,len(inp)):
        axs[i].plot(del_dictt[f'{inp[i]}'],del_dictt['Altitude'], marker= '.', label = "Org" )
        # axs[i].plot(belowBL[f'{inp[i]}'],belowBL['Altitude'], marker= '.' , label = "bl")
        axs[i].plot(MidAtm[f'{inp[i]}'],MidAtm['Altitude'], marker= '.' , label = "bl")
        # axs[i].plot(UpAtm[f'{inp[i]}'],UpAtm['Altitude'], marker= '.' , label = "bl")
        axs[i].set_xlabel(f'{inp[i]}')
    axs[0].legend()
    axs[0].set_ylabel('Height m ') 
    # Creating GRASP rslt dictionary for runGRASP.py    
    
    
    
    df = pd.DataFrame(FullAvgProf)
    # df = pd.DataFrame(df_new)
    rslt = {} # 

    DstMR = df['Dust_Mixing_Ratio'][:]
    # AEsph = df['Angstrom_Spherical'][:]
    aDP= df['532_aer_dep'][:]

    AEt= df['FMF'][:]

    #dust mixing ratio from paper
   
    
    height_shape = np.array(df['Altitude'][:]).shape[0]   #setting the lowermos value to zero to avoid GRASP intgration

    Range = np.ones((height_shape,2))
    # Range[:,0] = df['Altitude'][:]
    Range[:,0] = df['Altitude'][:]
    Range[:,1] = df['Altitude'][:]  # in meters
    rslt['RangeLidar'] = Range
    

    Bext = np.zeros((height_shape,2))
    # Bext[:,0] = df['355_ext'][:]
    Bext[:,0] = df['532_ext'][:]
    Bext[:,1] = df['1064_ext'] [:]
    Bext[0,2] = np.nan  #Setting one of the value in the array to nan so that GRASP will discard this measurement

    Bsca = np.zeros((height_shape,2))
    # Bsca[:,0] = df['355_bsc'][:]
    Bsca[:,0] = df['532_bsc'] [:]
    Bsca[:,1] = df['1064_bsc'][:]
    # Bsca[0,2] = np.nan
  
    
    Dep = np.zeros((height_shape,2))
    # Dep[-2:,0] = df['355_dep'][height_shape-2:] #Total depolarization ratio
    # Dep[-2:,1] = df['532_dep'][height_shape-2:]
    # Dep[-2:,2] = df['1064_dep'] [height_shape-2:]
    
    # Dep[:,0] = df['355_dep'][:]  #Total depolarization ratio
    Dep[:,0] = df['532_dep'][:]
    Dep[:,1] = df['1064_dep'] [:]
    #Unit conversion 
    rslt['meas_VExt'] = Bext / 1000
    rslt['meas_VBS'] = Bsca / 1000 # converting units from km-1 to m-1
    rslt['meas_DP'] = Dep *100  # in percentage


    rslt['lambda'] = np.array([532,1064])/1000 #values of HSRL wl in um
    rslt['wl'] = np.array([532,1064])/1000
    rslt['datetime'] =dt.datetime.strptime(str(int(f1["header"]['date'][0][0]))+ str(f1['Nav_Data']['UTCtime2'][PixNo][0]),'%Y%m%d%H%M%S.%f')
    rslt['latitude'] = f1['Nav_Data']['gps_lat'][PixNo]
    rslt['longitude']= f1['Nav_Data']['gps_lon'][PixNo]
    rslt['OBS_hght']= AirAlt # aircraft altitude in m
    rslt['land_prct'] = 0 #Ocean Surface

    # if gaspar ==True: # Molecular depolarization correction 
    rslt['gaspar'] = np.array([0.0037,0.0037,0.0037])

    if SimpleCase == True:
        Bsca[0,2] = np.nan #Setting one of the value in the array to nan so that GRASP will discard this measurement, we are doing this for HSRL because it is not a direct measuremnt
        Bext[0,2] = np.nan  #Setting one of the value in the array to nan so that GRASP will discard this measurement

        
    
     # TODO check the units for molecular depol correction.   
    # TODO make this general

    f1.close()
    return rslt,BLh,DstMR,aDP,AEt,AerID, Temp




# def Read_Data_HSRL_Oracles_Height_V2_1(file_path,file_name,PixNo):

#  #Specify the Path and the name of the HSRL file
#     #Reading the HSRL data
    
#     f1= h5py.File(file_path + file_name,'r+')  #reading Lidar measurements 
#     HSRL = f1['DataProducts']
#     #Latitude and longitude values 
#     latitude,longitude = f1['Nav_Data']['gps_lat'][:],f1['Nav_Data']['gps_lon'][:]
#     AirAlt = f1['Nav_Data']['gps_alt'][PixNo] #Altitude of the aircraft
#     print(AirAlt)

#     del_dict ={} #This dictionary stores 
#     inp = ['355_ext','532_ext','1064_ext','355_bsc','532_bsc','1064_bsc','355_dep', '532_dep','1064_dep']
#     #Setting negative values to zero, The negative values are due to low signal so we can replace with 0 without loss of info.
#     for i in range (len(inp)):
#         del_dict[f'{inp[i]}'] = HSRL[f'{inp[i]}'][PixNo]
#         HSRL_checkFillVals(del_dict[f'{inp[i]}']) # set all negative values to zero
#         if (inp[i] == '355_dep') or (inp[i] == '532_dep') or (inp[i] == '1064_dep'):
#             del_dict[f'{inp[i]}'] = np.where(HSRL[f'{inp[i]}'][PixNo][:]>= 0.6 , np.nan, HSRL[f'{inp[i]}'][PixNo])  #CH: This should be changed, 0.7 has been set arbitarily

#     #Caculating range, Range is defined as distance from the instrument to the aersosol layer, i.e. range at instrument heright = 0. We have to make sure that The range is in decending order
#     del_dict['Altitude'] = HSRL['Altitude'][0]  # altitude above sea level
#     # del_dict['Altitude'] = HSRL['Altitude'][0]  
#     df_new = pd.DataFrame(del_dict)
#     df_new.interpolate(inplace=True, limit_area= 'inside')

#     #Filtering and removing the pixels with bad data: 
#     Removed_index = []  # Removed_index holds indices of pixels to be removed
#     # Filter values greater than flight altitude
#     Removed_index.append(np.nonzero(np.array(df_new['Altitude'][:]) > AirAlt)[0][:])
#     # Removed_index.append(np.nonzero(np.array(df_new['Altitude'][:]) < 1000)[0][:]) #this removes the profile below 1000m 
#     # Filter values less than or equal to zero altitude
#     Removed_index.append(np.nonzero(np.array(df_new['Altitude'][:]) <= 0)[0][:])
#     # Filter values less than or equal to zero range
#     Removed_index.append(np.nonzero(f1['UserInput']['range_interp'][PixNo] <= 0)[0])
#     # Filter values less than 1800 in range interpolation
#     Removed_index.append(np.nonzero(f1['UserInput']['range_interp'][PixNo] < 1800)[0])
#     # Filter NaN values in range interpolation
#     Removed_index.append(np.nonzero(np.isnan(f1['UserInput']['range_interp'][PixNo]))[0])
#     # Filter NaN values in low gain signal limit data mask
#     Removed_index.append(np.nonzero(np.isnan(HSRL["mask_low"][PixNo]))[0])

#     # Cloud Correction
#     CloudCorr_1064 = np.nonzero(np.isnan(HSRL["1064_bsc_cloud_screened"][PixNo]))[0]
#     CloudCorr_355 = np.nonzero(np.isnan(HSRL["355_bsc_cloud_screened"][PixNo]))[0]
#     CloudCorr_532 = np.nonzero(np.isnan(HSRL["532_bsc_cloud_screened"][PixNo]))[0]

#     Removed_index.append(CloudCorr_1064)
#     Removed_index.append(CloudCorr_355)
#     Removed_index.append(CloudCorr_532)
#     Removed_index.append(np.nonzero(np.isnan(HSRL['355_ext'][PixNo]))[0])

#     # Concatenate all removed indices and remove duplicates
#     rm_pix=[]
#     for lis in Removed_index:
#         rm_pix += list(lis)[:] # concatenating the lists
#     rm_pix = np.unique(rm_pix)

#     # Create dictionaries to hold filtered and interpolated data
#     del_dictt = {}
#     # Delete removed pixels and set negative values to zero for each data type
#     inp2 = ['355_ext','532_ext','1064_ext','355_bsc','532_bsc','1064_bsc','355_dep', '532_dep','1064_dep', 'Altitude']
#     for i in range (len(inp2)): #
#         del_dictt[f'{inp2[i]}'] = np.delete(np.array(df_new[f'{inp2[i]}']), rm_pix)

            
#     #Height limits
#     BLh = 1200
#     UpH = 4000

#     belowBL ={}
#     MidAtm={}
#     UpAtm={}


#     hgt = del_dictt['Altitude']

#     for i in range (len(inp2)): 
#         belowBL[f'{inp2[i]}'] = del_dictt[f'{inp2[i]}'][np.where(hgt<= BLh)]
#         MidAtm[f'{inp2[i]}'] = del_dictt[f'{inp2[i]}'][np.where((hgt> BLh) &(hgt< UpH))]
#         UpAtm[f'{inp2[i]}'] = del_dictt[f'{inp2[i]}'][np.where(hgt> UpH)]
        
#     #TODO make this part more general
#     #No of height grids in each ses in meters
#     BLiIntv = 50
#     MidInv = 150
#     UpInv = 200
#     BLIntv = 110


#     BLProf= VertP(belowBL, BLIntv)
#     MidProf= VertP(MidAtm, MidInv)
#     UpProf= VertP(UpAtm, UpInv)
#     inp = ['355_ext','532_ext','1064_ext','355_bsc','532_bsc','1064_bsc','355_dep', '532_dep','1064_dep']

#     FullAvgProf = {}

#     for i in range (len(inp2)): 
#         FullAvgProf[inp2[i]] =  np.concatenate((np.array(UpProf[inp2[i]]), np.array(MidProf[inp2[i]]),np.array(BLProf[inp2[i]])), axis=0)
        
#     # fig, axs = plt.subplots(nrows= 1, ncols=9, figsize=(20, 6), sharey = True)
#     # for i in range (0,len(inp)):
#     #     axs[i].plot(del_dictt[f'{inp[i]}'],del_dictt['Altitude'], marker= '.', label = "Org" )
#     #     axs[i].plot(FullAvgProf[f'{inp[i]}'],FullAvgProf['Altitude'], marker= '.' , label = "Avg Prof")

#     #     axs[i].set_xlabel(f'{inp[i]}')

#     # axs[0].legend()
#     # axs[0].set_ylabel('Height m ') 




#     # fig, axs = plt.subplots(nrows= 1, ncols=9, figsize=(20, 6), sharey = True)
#     # for i in range (0,len(inp)):
#     #     axs[i].plot(del_dictt[f'{inp[i]}'],del_dictt['Altitude'], marker= '.', label = "Org" )
#     #     axs[i].plot(belowBL[f'{inp[i]}'],belowBL['Altitude'], marker= '.' , label = "bl")
#     #     axs[i].plot(MidAtm[f'{inp[i]}'],MidAtm['Altitude'], marker= '.' , label = "bl")
#     #     axs[i].plot(UpAtm[f'{inp[i]}'],UpAtm['Altitude'], marker= '.' , label = "bl")
        

#     #     axs[i].set_xlabel(f'{inp[i]}')

#     # axs[0].legend()
#     # axs[0].set_ylabel('Height m ') 
#     #Creating GRASP rslt dictionary for runGRASP.py    
#     df = pd.DataFrame(FullAvgProf)
#     rslt = {} # 
#     height_shape = np.array(df['Altitude'][:]).shape[0]   #setting the lowermos value to zero to avoid GRASP intgration

#     Range = np.zeros((height_shape,3))
#     Range[:,0] = df['Altitude'][:]
#     Range[:,1] = df['Altitude'][:]
#     Range[:,2] = df['Altitude'][:]  # in meters
#     rslt['RangeLidar'] = Range
    
#     Bext = np.zeros((height_shape,3))
    
    
#     Bext[:,0] = df['355_ext'][:]
#     Bext[:,1] = df['532_ext'][:]
#     Bext[:,2] = df['1064_ext'] [:]
#     Bext[0,2] = np.nan  #Setting one of the value in the array to nan so that GRASP will discard this measurement

#     Bsca = np.zeros((height_shape,3))
#     Bsca[:,0] = df['355_bsc'][:]
#     Bsca[:,1] = df['532_bsc'] [:]
#     Bsca[:,2] = df['1064_bsc'][:]
#     Bsca[0,2] = np.nan #Setting one of the value in the array to nan so that GRASP will discard this measurement, we are doing this for HSRL because it is not a direct measuremnt

#     Dep = np.zeros((height_shape,3))
#     Dep[:,0] = df['355_dep'][:]  #Total depolarization ratio
#     Dep[:,1] = df['532_dep'][:]
#     Dep[:,2] = df['1064_dep'] [:]

#     #Unit conversion 
#     rslt['meas_VExt'] = Bext / 1000
#     rslt['meas_VBS'] = Bsca / 1000 # converting units from km-1 to m-1
#     rslt['meas_DP'] = Dep *100  # in percentage


#     rslt['lambda'] = np.array([355,532,1064])/1000 #values of HSRL wl in um
#     rslt['wl'] = np.array([355,532,1064])/1000
#     rslt['datetime'] =dt.datetime.strptime(str(int(f1["header"]['date'][0][0]))+ str(f1['Nav_Data']['UTCtime2'][PixNo][0]),'%Y%m%d%H%M%S.%f')
#     rslt['latitude'] = f1['Nav_Data']['gps_lat'][PixNo]
#     rslt['longitude']= f1['Nav_Data']['gps_lon'][PixNo]
#     rslt['OBS_hght']= AirAlt # aircraft altitude in m
#     rslt['land_prct'] = 0 #Ocean Surface

#     f1.close()
#     return rslt

def Read_Data_HSRL_constHgt(file_path,file_name,PixNo):

 #Specify the Path and the name of the HSRL file
    #Reading the HSRL data
    f1= h5py.File(file_path + file_name,'r+')  #reading Lidar measurements 

    HSRL = f1['DataProducts']
    AirAlt = f1['Nav_Data']['gps_alt'][PixNo] #Altitude of the aircraft

    Data_dic ={} #This dictionary stores all the HSRL variables used for GRASP forward simulation
    inp = ['355_ext','532_ext','1064_ext','355_bsc','532_bsc','1064_bsc','355_dep', '532_dep','1064_dep']

    #Setting negative values to zero, The negative values are due to low signal so we can replace with 0 without loss of info.
    for i in range (len(inp)):
        
        #Index of values below aircraft and above ground
        hmaxInd = np.max(np.where(HSRL['Altitude'][0] <=AirAlt-1200)[0])
        hminInd = np.min(np.where(HSRL['Altitude'][0] >0)[0]) #filtering the values that have negative vales for the height on the data
        
        #Storing Bsca, Bext and DP values for all HSRL wavelengths
        Data_dic[f'{inp[i]}'] = HSRL[f'{inp[i]}'][PixNo][hminInd:hmaxInd]
        
        HSRL2_checkFillVals(Data_dic[f'{inp[i]}']) # set all negative values to zero
        Data_dic['Altitude'] = HSRL2_checkFillVals(HSRL['Altitude'][0][hminInd:hmaxInd])  # Height of the aerosol layer from the sea level
        df_new = pd.DataFrame(Data_dic)

    Plot_avg_prof = True
    avgProf = pd.DataFrame() # Vertical Averaged Profiles

    #Averaging the vertical profile to reduce noise
    hgt = Data_dic['Altitude']
    hgtInterv = 200  # new height interval, for now it is set to 150 m


    altitude_diff = np.gradient(Data_dic['Altitude'])
    numInterv = int(hgtInterv / np.mean(altitude_diff))  # Calculate numInterv based on mean altitude difference
    Npoint = int(len(Data_dic['Altitude'])/numInterv) # No of vertical bins after averaging

    # Create an array to store the averaged values
    for i in range(numInterv):
        strtVal = i*numInterv
        endVal = (i+1)*numInterv    

        hgtAvg = np.zeros(Npoint)
    for k in range (len(inp)):
        a = 0   # Indexing variable
        averaged_values = np.zeros(Npoint)

        for i in range(Npoint):
            start_range = hgt[i * numInterv]
            end_range = hgt[(i + 1) * numInterv]
            indexAvg= np.where((hgt >= start_range) & (hgt < end_range)) #Index of values in the start-stop range 
            
            #Taking mean
            if len(indexAvg[0]) > 0:
                averaged_values[a]= np.mean(np.array(df_new[f'{inp[k]}'])[indexAvg])
            
            if k ==0: # Avoiding repetition 
                hgtAvg[a]= np.mean(np.array(df_new['Altitude'])[indexAvg])
            a=a+1 # Indexing variable
        #Storing the values of profile
        avgProf[f'{inp[k]}'] = averaged_values[np.where(hgtAvg>0)] #Averaged profile values with height >0 
        avgProf['Altitude'] =  hgtAvg[np.where(hgtAvg>0)]
        
    ## GRASP requires the vertical profiles to be in decendong order, so reversing the entire profile
    df = avgProf[::-1]
    if Plot_avg_prof ==True:
        fig, axs = plt.subplots(nrows= 1, ncols=9, figsize=(20, 6), sharey = True)
        for i in range (0,len(inp)):
            axs[i].plot(Data_dic[f'{inp[i]}'],Data_dic['Altitude'], marker= '.', label = "Org" )
            axs[i].plot(df[f'{inp[i]}'],df['Altitude'], marker= '.' , label = "Avg Prof")

            axs[i].set_xlabel(f'{inp[i]}')
            
        axs[0].legend()
        axs[0].set_ylabel('Height m ')  
        
        
    #Creating GRASP rslt dictionary for runGRASP.py    

    rslt = {} # 
    height_shape = np.array(df['Altitude'][:]).shape[0]   #setting the lowermos value to zero to avoid GRASP intgration

    Range = np.zeros((height_shape,3))
    Range[:,0] = df['Altitude'][:]
    Range[:,1] = df['Altitude'][:]
    Range[:,2] = df['Altitude'][:]  # in meters
    rslt['RangeLidar'] = Range

    Bext = np.zeros((height_shape,3))
    Bext[:,0] = df['355_ext'][:]
    Bext[:,1] = df['532_ext'][:]
    Bext[:,2] = df['1064_ext'] [:]
    Bext[0,2] = np.nan  #Setting one of the value in the array to nan so that GRASP will discard this measurement

    Bsca = np.zeros((height_shape,3))
    Bsca[:,0] = df['355_bsc'][:]
    Bsca[:,1] = df['532_bsc'] [:]
    Bsca[:,2] = df['1064_bsc'][:]

    Bsca[0,2] = np.nan #Setting one of the value in the array to nan so that GRASP will discard this measurement, we are doing this for HSRL because it is not a direct measuremnt

    Dep = np.zeros((height_shape,3))
    Dep[:,0] = df['355_dep'][:]  #Total depolarization ratio
    Dep[:,1] = df['532_dep'][:]
    Dep[:,2] = df['1064_dep'] [:]

    #Unit conversion 
    rslt['meas_VExt'] = Bext / 1000
    rslt['meas_VBS'] = Bsca / 1000 # converting units from km-1 to m-1
    rslt['meas_DP'] = Dep *100  # in percentage


    rslt['lambda'] = np.array([355,532,1064])/1000 #values of HSRL wl in um
    rslt['wl'] = np.array([355,532,1064])/1000
    rslt['datetime'] =dt.datetime.strptime(str(int(f1["header"]['date'][0][0]))+ str(f1['Nav_Data']['UTCtime2'][PixNo][0]),'%Y%m%d%H%M%S.%f')
    rslt['latitude'] = f1['Nav_Data']['gps_lat'][PixNo]
    rslt['longitude']= f1['Nav_Data']['gps_lon'][PixNo]
    rslt['OBS_hght']= AirAlt # aircraft altitude in m
    rslt['land_prct'] = 0 #Ocean Surface

    f1.close()
    return rslt




# #The function Checks for Fill values or negative values and replaces them with nan. To check for negative values, set negative_check = True 
# def checkFillVals(param , negative_check = None):
#     param[:] = np.where(param[:] == -999, np.nan, param[:])
#     if negative_check == True:
#         param[:] = np.where(param[:] < 0 , np.nan, param[:])
#     return param

# #This sets the nan values to 0 
# def HSRL2_checkFillVals(param):
#     param[:] = np.where(param < 0, 0, param)
#     param[:] = np.where(np.isnan(param), 0, param)
#     return param

# #Checks for negative values and replaces them by nan
# def HSRL_checkFillVals(param):
#     param[:] = np.where(param[:] < 0 , np.nan, param[:])     
#     return param




# ## Reading the Multiangle Polarimeter data ()

# def Read_Data_HSRL_Oracles(file_path,file_name,PixNo):

#     f1= h5py.File(file_path + file_name,'r+')  #reading Lidar measurements 
#     HSRL = f1['DataProducts']
#     #Latitude and longitude values 
#     latitude,longitude = f1['Nav_Data']['gps_lat'][:],f1['Nav_Data']['gps_lon'][:]
#     AirAlt = f1['Nav_Data']['gps_alt'][PixNo] #Altitude of the aircraft
#     print(AirAlt)

#     Data_dic ={} #This dictionary stores 
#     inp = ['355_ext','532_ext','1064_ext','355_bsc_Sa','532_bsc_Sa','1064_bsc_Sa','355_dep', '532_dep','1064_dep']
#     #Setting negative values to zero, The negative values are due to low signal so we can replace with 0 without loss of info.
#     for i in range (len(inp)):
#         Data_dic[f'{inp[i]}'] = HSRL[f'{inp[i]}'][PixNo]
#         HSRL_checkFillVals(Data_dic[f'{inp[i]}']) # set all negative values to zero
#         if (inp[i] == '355_dep') or (inp[i] == '532_dep') or (inp[i] == '1064_dep'):
#             Data_dic[f'{inp[i]}'] = np.where(HSRL[f'{inp[i]}'][PixNo][:]>= 0.6 , np.nan, HSRL[f'{inp[i]}'][PixNo])  #CH: This should be changed, 0.7 has been set arbitarily

#     #Caculating range, Range is defined as distance from the instrument to the aersosol layer, i.e. range at instrument heright = 0. We have to make sure that The range is in decending order
#     Data_dic['Altitude'] = HSRL['Altitude'][0]  # altitude above sea level
#     # Data_dic['Altitude'] = HSRL['Altitude'][0]  
#     df_new = pd.DataFrame(Data_dic)
#     df_new.interpolate(inplace=True, limit_area= 'inside')

#     #Filtering and removing the pixels with bad data: 
#     Removed_index = []  # Removed_index holds indices of pixels to be removed
#     # Filter values greater than flight altitude
#     Removed_index.append(np.nonzero(np.array(df_new['Altitude'][:]) > AirAlt)[0][:])
#     # Filter values less than or equal to zero altitude
#     Removed_index.append(np.nonzero(np.array(df_new['Altitude'][:]) <= 0)[0][:])
#     # Filter values less than or equal to zero range
#     Removed_index.append(np.nonzero(f1['UserInput']['range_interp'][PixNo] <= 0)[0])
#     # Filter values less than 1800 in range interpolation
#     Removed_index.append(np.nonzero(f1['UserInput']['range_interp'][PixNo] < 1800)[0])
#     # Filter NaN values in range interpolation
#     Removed_index.append(np.nonzero(np.isnan(f1['UserInput']['range_interp'][PixNo]))[0])
#     # Filter NaN values in low gain signal limit data mask
#     Removed_index.append(np.nonzero(np.isnan(HSRL["mask_low"][PixNo]))[0])

#     # Cloud Correction
#     CloudCorr_1064 = np.nonzero(np.isnan(HSRL["1064_bsc_cloud_screened"][PixNo]))[0]
#     CloudCorr_355 = np.nonzero(np.isnan(HSRL["355_bsc_cloud_screened"][PixNo]))[0]
#     CloudCorr_532 = np.nonzero(np.isnan(HSRL["532_bsc_cloud_screened"][PixNo]))[0]
    
#     Removed_index.append(CloudCorr_1064)
#     Removed_index.append(CloudCorr_355)
#     Removed_index.append(CloudCorr_532)
#     Removed_index.append(np.nonzero(np.isnan(HSRL['355_ext'][PixNo]))[0])

#     # Concatenate all removed indices and remove duplicates
#     rm_pix=[]
#     for lis in Removed_index:
#         rm_pix += list(lis)[:] # concatenating the lists
#     rm_pix = np.unique(rm_pix)

#     # Create dictionaries to hold filtered and interpolated data
#     del_dict = {}
#     # Delete removed pixels and set negative values to zero for each data type
#     inp2 = ['355_ext','532_ext','1064_ext','355_bsc_Sa','532_bsc_Sa','1064_bsc_Sa','355_dep', '532_dep','1064_dep', 'Altitude']
#     for i in range (len(inp2)): #
#         del_dict[f'{inp2[i]}'] = np.delete(np.array(df_new[f'{inp2[i]}']), rm_pix)

#     df_mean = pd.DataFrame()
#     npoints = 10 #no of height pixels averaged 
#     Mod_value = np.array(del_dict['Altitude']).shape[0] % npoints  #Skip these values for reshaping the array
#     for i in range (len(inp2)): #taking mean 
#         df_mean[f'{inp2[i]}'] = nanmean(np.array(del_dict[f'{inp2[i]}'][Mod_value:]).reshape( int(np.array(del_dict[f'{inp2[i]}']).shape[0]/npoints),npoints),axis=1)

#     for k in df_mean.keys():
#         print(df_mean[k].shape)

#     df = df_mean[::-1] # GRASP requires the vertical profiles to be arranged in descending order, hence reversing the data.
    
    
#     rslt = {} # This dictionary will store the values arranged in GRASP's format. 
#     height_shape = np.array(df['Altitude'][:]).shape[0] #to avoid the height of the sea salt, this should be removed 

#     Range = np.ones((height_shape,3))
#     Range[:,0] = df['Altitude'][:]
#     Range[:,1] = df['Altitude'][:]
#     Range[:,2] = df['Altitude'][:]  # in meters
#     rslt['RangeLidar'] = Range

#     Bext = np.ones((height_shape,3))
#     Bext[:,0] = df['355_ext'][:]
#     Bext[:,1] = df['532_ext'][:]
#     Bext[:,2] = df['1064_ext'] [:]
#     Bext[0,2] = np.nan 

#     Bsca = np.ones((height_shape,3))
#     Bsca[:,0] = df['355_bsc_Sa'][:]
#     Bsca[:,1] = df['532_bsc_Sa'] [:]
#     Bsca[:,2] = df['1064_bsc_Sa'][:]

#     Bsca[0,2] = np.nan 

#     Dep = np.ones((height_shape,3))
#     Dep[:,0] = df['355_dep'][:]
#     Dep[:,1] = df['532_dep'][:]
#     Dep[:,2] = df['1064_dep'] [:]

#     #Unit conversion 
#     rslt['meas_VExt'] = Bext / 1000
#     rslt['meas_VBS'] = Bsca / 1000 # converting units from km-1 to m-1
#     rslt['meas_DP'] = Dep *100  # in percentage
#     # print(rslt['meas_DP'])

#     rslt['lambda'] = np.array([355,532,1064])/1000 #values of HSRL wl in um
#     rslt['wl'] = np.array([355,532,1064])/1000
#     rslt['datetime'] =dt.datetime.strptime(str(int(f1["header"]['date'][0][0]))+ np.str(f1['Nav_Data']['UTCtime2'][PixNo][0]),'%Y%m%d%H%M%S.%f')
#     rslt['latitude'] = latitude[PixNo]
#     rslt['longitude']= longitude[PixNo]
#     rslt['OBS_hght']=  AirAlt# aircraft altitude in m
#     rslt['land_prct'] = 0 #Ocean Surface

#     #Substitude the actual value
#     # rslt['gaspar'] = np.ones((3))*0.0037 #MOlecular depolarization 
#     f1.close() 
#     return rslt