"""
# Greema Regmi, UMBC
# Date: Jan 31, 2023

This code reads Polarimetric data from the Campaigns and runs GRASP. This code was created to Validate the Aerosol retrivals performed using Non Spherical Kernels (Hexahedral from TAMU DUST 2020)

"""

import sys
from CreateRsltsDict import Read_Data_RSP_Oracles
from CreateRsltsDict import Read_Data_HSRL_Oracles
import netCDF4 as nc
from runGRASP import graspDB, graspRun, pixel, graspYAML
from matplotlib import pyplot as plt
import os
if os.uname()[1]=='uranus': plt.switch_backend('agg')
import numpy as np
import datetime as dt
from numpy import nanmean
%matplotlib inline
# %load_ext autoreload
# %autoreload 2
# %reload_ext autoreload

# %run -d -b runGRASP.py:LINENUM scriptToRun.py
# %load_ext autoreload
# %autoreload 2

import numpy as np
# import pandas as pd
from matplotlib import pyplot as plt
import h5py 
sys.path.append("/home/gregmi/git/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases")
from architectureMap import returnPixel
from Plot_ORACLES import PltGRASPoutput
import yaml

# file_path = '/home/gregmi/ORACLES/ more_case/'
# file_name = 'RSP1-P3_L1C-RSPCOL-CollocatedRadiances_20181010T124329Z_V003-20210422T002644Z.h5'
# # RSP Case: August 1
# file_path = "/home/gregmi/ORACLES/RSP2-L1C_P3_20170801_R02"  #Path to the ORACLE data file
# file_name =  "/RSP2-P3_L1C-RSPCOL-CollocatedRadiances_20170801T130949Z_V002-20210624T034642Z.h5" #Name of the ORACLES file
# # file_name = '/RSP2-P3_L1C-RSPCOL-CollocatedRadiances_20170801T142830Z_V002-20210624T034755Z.h5'

# # RSP  Case: Sep 22
file_path = "/home/gregmi/ORACLES/RSP1-L1C_P3_20180922_R03/"  #Path to the ORACLE data file
file_name =  "/RSP1-P3_L1C-RSPCOL-CollocatedRadiances_20180922T151106Z_V003-20210421T233946Z.h5" #Name of the ORACLES file

HSRLfile_path = "/home/gregmi/ORACLES/HSRL" #Path to the ORACLE data file 
HSRLfile_name =  "/HSRL2_P3_20180922_R2.h5" #Name of the ORACLES file
GasAbsFn = '/home/gregmi/ORACLES/UNL_VRTM/shortwave_gas.unlvrtm.nc'



#This is required if we want to configure the HSRL yaml file based on the GRASP output for the RSP
noMod =2  #number of aerosol mode, here 2 for fine+coarse mode configuration
maxr=1.05  #set max and min value : here max = 1% incease, min 1% decrease : this is a very narrow distribution
minr =0.95
a=1 #no of char # this is a varible added to char and modes to avoid char[0]/ mod[0] which doesnt exist


def update_HSRLyaml(YamlFileName, RSP_rslt, noMod, maxr, minr, a, Kernel_type):

    # Load the YAML file for HSRL
    with open(YamlFileName, 'r') as f:  
        data = yaml.safe_load(f)

    YamlChar =[]  #This list stores the name of the charater types in the yaml files
    noYmlChar = np.arange(1,7) #No of aerosol characters types in the yaml file (This can be adjusted based on the parameters we want to change)
    
    for i in noYmlChar:

        YamlChar.append(data['retrieval']['constraints'][f'characteristic[{i}]']['type'])

    # RSP_rslt = np.load('RSP_sph.npy',allow_pickle= True).item()
    print(len(YamlChar))
    
    #change the yaml intitial conditions using the RSP GRASP output
    for i in range(len(YamlChar)): #loop over the character types in the list

        for noMd in range(noMod): #loop over the aerosol modes

    #         print(noMd,i)
            initCond = data['retrieval']['constraints'][f'characteristic[{i+a}]'][f'mode[{noMd+a}]']['initial_guess']
            if YamlChar[i] == 'aerosol_concentration':
                initCond['value'] = float(RSP_rslt['vol'][noMd]) #value from the GRASP result for RSP
               
            if YamlChar[i] == 'size_distribution_lognormal':
                initCond['value'] = float(RSP_rslt['rv'][noMd]),float(RSP_rslt['sigma'][noMd])
                initCond['max'] =float(RSP_rslt['rv'][noMd]*maxr),float(RSP_rslt['sigma'][noMd]*maxr)
                initCond['min'] =float(RSP_rslt['rv'][noMd]*minr),float(RSP_rslt['sigma'][noMd]*minr)
                print("done",YamlChar[i])
            if YamlChar[i] == 'real_part_of_refractive_index_spectral_dependent':
                initCond['index_of_wavelength_involved'] = [1,2,3]
                initCond['value'] =float(RSP_rslt['n'][noMd][0]),float(RSP_rslt['n'][noMd][2]),float(RSP_rslt['n'][noMd][4])
                initCond['max'] =float(RSP_rslt['n'][noMd][0]*maxr),float(RSP_rslt['n'][noMd][2]*maxr),float(RSP_rslt['n'][noMd][4]*maxr)
                initCond['min'] =float(RSP_rslt['n'][noMd][0]*minr),float(RSP_rslt['n'][noMd][2]*minr),float(RSP_rslt['n'][noMd][4]*minr)
                print("done",YamlChar[i])
            if YamlChar[i] == 'imaginary_part_of_refractive_index_spectral_dependent':
                initCond['index_of_wavelength_involved'] = [1,2,3]
                initCond['value'] =float(RSP_rslt['k'][noMd][0]),float(RSP_rslt['k'][noMd][2]),float(RSP_rslt['k'][noMd][4])
                initCond['max'] =float(RSP_rslt['k'][noMd][0]*maxr),float(RSP_rslt['k'][noMd][2]*maxr),float(RSP_rslt['k'][noMd][4]*maxr)
                initCond['min'] = float(RSP_rslt['k'][noMd][0]*minr),float(RSP_rslt['k'][noMd][2]*minr),float(RSP_rslt['k'][noMd][4]*minr)
                print("done",YamlChar[i])
            if YamlChar[i] == 'sphere_fraction':
                initCond['value'] = float(RSP_rslt['sph'][noMd]/100)
                initCond['max'] =float(RSP_rslt['sph'][noMd]/100*maxr) #GARSP output is in %
                initCond['min'] =float(RSP_rslt['sph'][noMd]/100*minr)
                print("done",YamlChar[i])


    if Kernel_type == "sphro":
        UpKerFile = 'Settings_Sphd_RSP_HSRL.yaml'
    if Kernel_type == "TAMU":
        UpKerFile = 'Settings_TAMU_RSP_HSRL.yaml'
    
    ymlPath = '/home/gregmi/git/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/'
    
    with open(ymlPath+UpKerFile, 'w') as f: #write the chnages to new yaml file
        yaml.safe_dump(data, f)
        
    return 

#Find the pixel index for nearest lat and lon for given LatH and LonH
def FindPix(LatH,LonH,Lat,Lon):
    
    # Assuming Lat, latH, Lon, and LonM are all NumPy arrays
    diffLat = np.abs(LatH - Lat) # Find the absolute difference between `Lat` and each element in `latH`
    indexLat = np.argwhere(diffLat == diffLat.min())[0] # Find the indices of all elements that minimize the difference

    diffLon = np.abs(LonH - Lon) # Find the absolute difference between `Lon` and each element in `LonM`
    indexLon = np.argwhere(diffLon == diffLon.min())[0] # Find the indices of all elements that minimize the difference
    
    return indexLat[0], indexLat[1]


def find_dust(HSRLfile_path, HSRLfile_name, plot=None):
    #BAsed
    # Open the HDF5 file in read mode
    f1 = h5py.File(HSRLfile_path + HSRLfile_name, 'r+')
    
    # Extract the Aerosol_ID data product
    Dust_pix = f1['DataProducts']['Aerosol_ID']
    
    # Create an empty list to store indices of dust pixels for each column
    dust_pixel = []
    
    # Loop over the columns in Dust_pix
    for i in range(Dust_pix.shape[1]):
        # Get the indices where the pixel value is 8 (dust)
        dust_pixel.append(np.where(Dust_pix[:, i] == 8)[0])

    # Concatenate the arrays along the first axis (rows)
    concatenated_array = np.concatenate(dust_pixel, axis=0)
    # Flatten the concatenated array to a 1D array
    all_dust_pixels = concatenated_array.flatten()
    
    # Find the unique values and their frequency counts in the flattened dust pixel array
    unique_values, counts = np.unique(all_dust_pixels, return_counts=True)
    
    # Filter out the dust pixel values where frequency count is less than 100
    dust_pix = unique_values[counts > 100]
    # Find the dust pixel value(s) with the highest frequency count
    max_dust = unique_values[counts == counts.max()]
    
    # If plot is True, create and display plots
    if plot == True:
        # Plot a bar diagram showing the frequency count of each dust pixel value
        plt.figure(figsize=(15,5))
        plt.bar(unique_values, counts)
        plt.xlabel('Values')
        plt.ylabel('Frequency')
        plt.show()
        
        # Create a contour plot of the Aerosol_ID data product and plot the dust pixel indices on it
        fig, ax = plt.subplots()
        c = ax.contourf(f1['DataProducts']['Aerosol_ID'][:].T, cmap='tab20b')
        ax.scatter(dust_pix, np.repeat((0), len(dust_pix)), c="k")
        plt.colorbar(c)
    
    # Close the HDF5 file
    f1.close()
    
    # Return the filtered dust pixel values and the dust pixel value(s) with the highest frequency count
    return dust_pix, max_dust

def RSP_Run(Kernel_type,PixNo,ang1,ang2,TelNo,nwl): 
        
        # binPathGRASP ='/home/shared/GRASP_GSFC/build_RSP_v112_noWarn/bin/grasp_app' # New Version GRASP Executable  
        krnlPath='/home/shared/GRASP_GSFC/src/retrieval/internal_files'
        # Kernel_type =  sphro is for the GRASP spheriod kernal, while TAMU is to run with Hexahedral Kernal
        if Kernel_type == "sphro":
            fwdModelYAMLpath = '/home/gregmi/git/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/settings_BCK_POLAR_2modes_Shape_ORACLE_DoLP.yml'
            # fwdModelYAMLpath ='/home/gregmi/git/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/settings_BCK_POLAR_2modes_Shape_ORACLE_DoLP_2COARSE.yml'
            binPathGRASP ='/home/shared/GRASP_GSFC/build_RSP_v112/bin/grasp_app' 
            savePath=f"/home/gregmi/ORACLES/RSP1-L1C_P3_20180922_R03_{Kernel_type}"
        
        if Kernel_type == "TAMU":
            fwdModelYAMLpath = '/home/gregmi/git/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/settings_BCK_POLAR_2modes_Shape_ORACLE_DoLP_dust.yml'
            # fwdModelYAMLpath ='/home/gregmi/git/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/settings_BCK_POLAR_2modes_Shape_ORACLE_DoLP_dust_2Coarse.yml'
            binPathGRASP ='/home/shared/GRASP_GSFC/build_HEX_v112/bin/grasp_app' #GRASP Executable
            savePath=f"/home/gregmi/ORACLES/RSP1-L1C_P3_20180922_R03_{Kernel_type}"

    
        #rslt is the GRASP rslt dictionary or contains GRASP Objects
        rslt = Read_Data_RSP_Oracles(file_path,file_name,PixNo,ang1,ang2,TelNo, nwl,GasAbsFn)

        maxCPU = 3 #maximum CPU allocated to run GRASP on server
        gRuns = []
        yamlObj = graspYAML(baseYAMLpath=fwdModelYAMLpath)
        #eventually have to adjust code for height, this works only for one pixel (single height value)
        gRuns.append(graspRun(pathYAML=yamlObj, releaseYAML=True )) # This should copy to new YAML object

        pix = pixel()
        pix.populateFromRslt(rslt, radianceNoiseFun=None, dataStage='meas', verbose=False)
        gRuns[-1].addPix(pix)
        gDB = graspDB(graspRunObjs=gRuns, maxCPU=maxCPU)

        #rslts contain all the results form the GRASP inverse run
        rslts, failPix = gDB.processData(binPathGRASP=binPathGRASP, savePath=None, krnlPathGRASP=krnlPath)
        return rslts



#Running the GRASP for spherical or hexahedral shape model for HSRL data
def HSLR_run(Kernel_type,HSRLfile_path,HSRLfile_name,PixNo, updateYaml= None):
        #Path to the kernel files
        krnlPath='/home/shared/GRASP_GSFC/src/retrieval/internal_files'
        
        if Kernel_type == "sphro":  #If spheriod model
            #Path to the yaml file for sphriod model
            fwdModelYAMLpath = '/home/gregmi/git/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/settings_BCK_POLARandLIDAR_10Vbins_2modes_ORACLES.yml'
            # fwdModelYAMLpath ='/home/gregmi/git/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/settings_BCK_POLARandLIDAR_10Vbins_2modes_ORACLES_2Coarse.yml'
            if updateYaml == True:  # True if init conditions for Yaml file for HSRL is updated from the GRASP output from RSP
                update_HSRLyaml(fwdModelYAMLpath, rslts_Sph[0], noMod, maxr, minr, a,Kernel_type)
                
                fwdModelYAMLpath ='/home/gregmi/git/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/Settings_Sphd_RSP_HSRL.yaml'
            # binPathGRASP = path toGRASP Executable for spheriod model
            binPathGRASP ='/home/shared/GRASP_GSFC/build_RSP_v112/bin/grasp_app' 
            savePath=f"/home/gregmi/ORACLES/HSRL1_P3_20180922_R03_{Kernel_type}"
        
        if Kernel_type == "TAMU":
            fwdModelYAMLpath = '/home/gregmi/git/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/settings_BCK_POLARandLIDAR_10Vbins_2modes_Tamu.yml'
            # fwdModelYAMLpath ='/home/gregmi/git/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/settings_BCK_POLARandLIDAR_10Vbins_2modes_Tamu_2Coarse.yml'
            if updateYaml == True:# True if init conditions for Yaml file for HSRL is updated from the GRASP output from RSP
                update_HSRLyaml(fwdModelYAMLpath, rslts_Tamu[0], noMod, maxr, minr, a,Kernel_type)
                fwdModelYAMLpath ='/home/gregmi/git/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/Settings_TAMU_RSP_HSRL.yaml'
            #Path to the GRASP Executable for TAMU
            binPathGRASP ='/home/shared/GRASP_GSFC/build_HEX_v112/bin/grasp_app' #GRASP Executable
            #Path to save output plot
            savePath=f"/home/gregmi/ORACLES/RSP1-L1C_P3_20180922_R03_{Kernel_type}"

    
        #rslt is the GRASP rslt dictionary or contains GRASP Objects
        rslt = Read_Data_HSRL_Oracles(HSRLfile_path,HSRLfile_name,PixNo)

        maxCPU = 3 #maximum CPU allocated to run GRASP on server
        gRuns = []
        yamlObj = graspYAML(baseYAMLpath=fwdModelYAMLpath)
        #eventually have to adjust code for height, this works only for one pixel (single height value)
        gRuns.append(graspRun(pathYAML=yamlObj, releaseYAML=True )) # This should copy to new YAML object

        pix = pixel()
        pix.populateFromRslt(rslt, radianceNoiseFun=None, dataStage= 'meas', verbose=False)
        gRuns[-1].addPix(pix)

        gDB = graspDB(graspRunObjs=gRuns, maxCPU=maxCPU)

        #rslts contain all the results form the GRASP inverse run
        rslts, failPix = gDB.processData(binPathGRASP=binPathGRASP, savePath=None, krnlPathGRASP=krnlPath)
        return rslts
    # height = 200 


def LidarAndMAP(Kernel_type,HSRLfile_path,HSRLfile_name,HSRLPixNo,file_path,file_name,RSP_PixNo,ang1,ang2,TelNo, nwl,GasAbsFn, updateYaml= None):
    
    krnlPath='/home/shared/GRASP_GSFC/src/retrieval/internal_files'

    if Kernel_type == "sphro":  #If spheriod model
        #Path to the yaml file for sphriod model
        fwdModelYAMLpath = '/home/gregmi/git/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/settings_BCK_LidarAndMAP_V.1.2.yml'
        # fwdModelYAMLpath ='/home/gregmi/git/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/settings_BCK_POLARandLIDAR_10Vbins_2modes_ORACLES_2Coarse.yml'
        if updateYaml == True:  # True if init conditions for Yaml file for HSRL is updated from the GRASP output from RSP
            update_HSRLyaml(fwdModelYAMLpath, rslts_Sph[0], noMod, maxr, minr, a,Kernel_type)
            
            fwdModelYAMLpath ='/home/gregmi/git/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/Settings_Sphd_RSP_HSRL.yaml'
        # binPathGRASP = path toGRASP Executable for spheriod model
        binPathGRASP ='/home/shared/GRASP_GSFC/build_RSP_v112/bin/grasp_app' 
        savePath=f"/home/gregmi/ORACLES/HSRL1_P3_20180922_R03_{Kernel_type}"
    
    if Kernel_type == "TAMU":
        fwdModelYAMLpath = '/home/gregmi/git/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/settings_BCK_LidarAndMAP_V.1.2_TAMU.yml'
        # fwdModelYAMLpath ='/home/gregmi/git/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/settings_BCK_POLARandLIDAR_10Vbins_2modes_Tamu_2Coarse.yml'
        if updateYaml == True:# True if init conditions for Yaml file for HSRL is updated from the GRASP output from RSP
            update_HSRLyaml(fwdModelYAMLpath, rslts_Tamu[0], noMod, maxr, minr, a,Kernel_type)
            fwdModelYAMLpath ='/home/gregmi/git/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/Settings_TAMU_RSP_HSRL.yaml'
        #Path to the GRASP Executable for TAMU
        binPathGRASP ='/home/shared/GRASP_GSFC/build_HEX_v112/bin/grasp_app' #GRASP Executable
        #Path to save output plot
        savePath=f"/home/gregmi/ORACLES/RSP1-L1C_P3_20180922_R03_{Kernel_type}"


    rslt_HSRL = Read_Data_HSRL_Oracles(HSRLfile_path,HSRLfile_name,HSRLPixNo)
    rslt_RSP = Read_Data_RSP_Oracles(file_path,file_name,RSP_PixNo,ang1,ang2,TelNo, nwl,GasAbsFn)
    
    rslt= {}  # Teh order of the data is  First lidar(number of wl ) and then Polarimter data 
    rslt['lambda'] = np.concatenate((rslt_HSRL['lambda'],rslt_RSP['lambda']))
    #Sort sotes theindex of the wavelength if arranged in ascending order
    sort = np.argsort(rslt['lambda']) 
    IndHSRL = rslt_HSRL['lambda'].shape[0]
    sort_Lidar, sort_MAP  = np.array([0,3,7]),np.array([1,2,4,5,6])
    

# The shape of the variables in RSPkeys and HSRLkeys should be equal to no of wavelength
#  Setting np.nan in place of the measurements for wavelengths for which there is no data
    RSPkeys = ['meas_I', 'meas_P','sza', 'vis', 'sca_ang', 'fis']
    HSRLkeys = ['RangeLidar','meas_VExt','meas_VBS','meas_DP']
    GenKeys= ['datetime','longitude', 'latitude', 'land_prct'] # Shape of these variables is not N wavelength
    
    #MAP measurement variables
    RSP_var = np.ones((rslt_RSP['meas_I'].shape[0],rslt['lambda'].shape[0])) * np.nan
    for keys in RSPkeys:

        for a in range(rslt_RSP[keys][:,0].shape[0]):
            RSP_var[a][sort_MAP] = rslt_RSP[keys][a]
        rslt[keys] = RSP_var
        RSP_var = np.ones((rslt_RSP['meas_I'].shape[0],rslt['lambda'].shape[0])) * np.nan


    #Lidar Measurements
    HSRL_var = np.ones((rslt_HSRL['meas_VExt'].shape[0],rslt['lambda'].shape[0]))* np.nan
    for keys1 in HSRLkeys:  
        for a in range(rslt_HSRL[keys1][:,0].shape[0]):
            
            HSRL_var[a][sort_Lidar] = rslt_HSRL[keys1][a]

        rslt[keys1] = HSRL_var
        # Refresh the array by Creating numpy nan array with shape of height x wl, Basically deleting all values
        HSRL_var = np.ones((rslt_HSRL['meas_VExt'].shape[0],rslt['lambda'].shape[0]))* np.nan

    for keys in GenKeys:
        rslt[keys] = rslt_RSP[keys]  #Adding the information about lat, lon, datetime and so on from RSP
    
    # rslt['OBS_hght'] = rslt_HSRL['OBS_hght']
    rslt['lambda'] = rslt['lambda'][sort]
    # print(rslt)

    maxCPU = 3 #maximum CPU allocated to run GRASP on server
    gRuns = []
    yamlObj = graspYAML(baseYAMLpath=fwdModelYAMLpath)
    #eventually have to adjust code for height, this works only for one pixel (single height value)
    gRuns.append(graspRun(pathYAML=yamlObj, releaseYAML=True )) # This should copy to new YAML object

    pix = pixel()
    pix.populateFromRslt(rslt, radianceNoiseFun=None, dataStage= 'meas', verbose=False)
    gRuns[-1].addPix(pix)

    gDB = graspDB(graspRunObjs=gRuns, maxCPU=maxCPU)

    #rslts contain all the results form the GRASP inverse run
    rslts, failPix = gDB.processData(binPathGRASP=binPathGRASP, savePath=None, krnlPathGRASP=krnlPath)
    return rslts

# # RSP_PixNo =13460
# RSP_PixNo = 13240
#     #Dusty pixel on 9/22
# # PixNo = find_dust(file_path,file_name)[1][0]
# TelNo = 0 # aggregated altitude. To obtain geometries corresponding to data from the 1880 nm channel, aggregation altitude should be set to 1, while aggregation altitude =0 should be used for all other channels.
# nwl = 5 # first  nwl wavelengths
# ang1 = 20
# ang2 = 120 # :ang angles  #Remove


# f1_MAP = h5py.File(file_path+file_name,'r+')   
# Data = f1_MAP['Data']
# LatRSP = f1_MAP['Geometry']['Collocated_Latitude'][TelNo,RSP_PixNo]
# LonRSP = f1_MAP['Geometry']['Collocated_Longitude'][TelNo,RSP_PixNo]
# f1_MAP.close()

# f1= h5py.File(HSRLfile_path + HSRLfile_name,'r+')  #reading hdf5 file  

# #Lat and Lon values for that pixel
# LatH = f1['Nav_Data']['gps_lat'][:]
# LonH = f1['Nav_Data']['gps_lon'][:]
# HSRLPixNo = FindPix(LatH,LonH,LatRSP,LonRSP)[0]
# f1.close()

# HSRLPixNo = 1154



for i in range(1):
    
    #Reading the ORACLE data for given pixel no, Tel_no = aggregated altitude
    #working pixels: 16800,16813 ,16814
    # RSP_PixNo = 2776  #4368  #Clear pixel 8/01 Pixel no of Lat,Lon that we are interested

    RSP_PixNo = 13240
     #Dusty pixel on 9/22
    # PixNo = find_dust(file_path,file_name)[1][0]
    TelNo = 0 # aggregated altitude. To obtain geometries corresponding to data from the 1880 nm channel, aggregation altitude should be set to 1, while aggregation altitude =0 should be used for all other channels.
    nwl = 5 # first  nwl wavelengths
    ang1 = 20
    ang2 = 120 # :ang angles  #Remove

    #  Kernel_type = Run(Kernel_type) for spheriod, Kernel_type = 'TAMU' for hexahedral
    rslts_Sph = RSP_Run("sphro",RSP_PixNo,ang1,ang2,TelNo,nwl)
    rslts_Tamu = RSP_Run("TAMU",RSP_PixNo,ang1,ang2,TelNo,nwl)

    
    f1_MAP = h5py.File(file_path+file_name,'r+')   
    Data = f1_MAP['Data']
    LatRSP = f1_MAP['Geometry']['Collocated_Latitude'][TelNo,RSP_PixNo]
    LonRSP = f1_MAP['Geometry']['Collocated_Longitude'][TelNo,RSP_PixNo]
    f1_MAP.close()
    f1= h5py.File(HSRLfile_path + HSRLfile_name,'r+')  #reading hdf5 file  

    #Lat and Lon values for that pixel
    LatH = f1['Nav_Data']['gps_lat'][:]
    LonH = f1['Nav_Data']['gps_lon'][:]

    f1.close()
    #Get the index of pixel taht corresponds to the RSP Lat Lon
    HSRLPixNo = FindPix(LatH,LonH,LatRSP,LonRSP)[0]  # Or can manually give the index of the pixel that you are intrested in

    # HSRLPixNo = 1154
    Retrieval_type = 'NosaltStrictConst_final'
    #Running GRASP for HSRL, HSRL_sphrod = for spheriod kernels,HSRL_Tamu = Hexahedral kernels
    HSRL_sphrod = HSLR_run("sphro",HSRLfile_path,HSRLfile_name,HSRLPixNo,updateYaml= True) 
    HSRL_Tamu = HSLR_run("TAMU",HSRLfile_path,HSRLfile_name,HSRLPixNo,updateYaml= True)

    LidarPolSph = LidarAndMAP('sphro',HSRLfile_path,HSRLfile_name,HSRLPixNo,file_path,file_name,RSP_PixNo,ang1,ang2,TelNo, nwl,GasAbsFn, updateYaml= None)
    LidarPolTAMU = LidarAndMAP('TAMU',HSRLfile_path,HSRLfile_name,HSRLPixNo,file_path,file_name,RSP_PixNo,ang1,ang2,TelNo, nwl,GasAbsFn, updateYaml= None)
    
    print('SPH',"tam" )
    print(HSRL_sphrod[0]['aod'],HSRL_Tamu[0]['aod'])
    
    
    #Plotting the results
    PltGRASPoutput(rslts_Sph, rslts_Tamu,file_name,PixNo = RSP_PixNo)

    def Plot_HSRL(HSRL_sphrod,LidarPolSph,HSRL_Tamu,LidarPolTAMU):
        plt.rcParams['font.size'] = '16'
        fig, axs= plt.subplots(nrows = 3, ncols =3, figsize= (18,10))
        for i in range(3):

            wave = np.str(HSRL_sphrod[0]['lambda'][i]) +"μm \n Range(km)"

            
            axs[i,0].plot(HSRL_sphrod[0]['meas_VBS'][:,i],HSRL_sphrod[0]['RangeLidar'][:,0]/1000, marker =">",color = "#3B270C", label ="Meas")
            axs[i,0].plot(HSRL_sphrod[0]['fit_VBS'][:,i],HSRL_sphrod[0]['RangeLidar'][:,0]/1000,color = "#025043", marker = "$O$",label ="Sphd")
            axs[i,0].plot(HSRL_Tamu[0]['fit_VBS'][:,i],HSRL_Tamu[0]['RangeLidar'][:,0]/1000,color = "#d24787",ls = "--", label="Hex", marker = "h")

            axs[i,1].plot(HSRL_sphrod[0]['meas_DP'][:,i],HSRL_sphrod[0]['RangeLidar'][:,0]/1000, marker =">",color = "#3B270C", label ="Meas")
            axs[i,1].plot(HSRL_sphrod[0]['fit_DP'][:,i],HSRL_sphrod[0]['RangeLidar'][:,0]/1000,color = "#025043", marker = "$O$")
            axs[i,1].plot(HSRL_Tamu[0]['fit_DP'][:,i],HSRL_Tamu[0]['RangeLidar'][:,0]/1000,color = "#d24787", ls = "--",marker = "h")
    
            axs[i,2].plot(HSRL_sphrod[0]['meas_VExt'][:,i],HSRL_sphrod[0]['RangeLidar'][:,0]/1000, marker =">",color = "#3B270C", label ="Meas")
            axs[i,2].plot(HSRL_sphrod[0]['fit_VExt'][:,i],HSRL_sphrod[0]['RangeLidar'][:,0]/1000,color = "#025043", marker = "$O$")
            axs[i,2].plot(HSRL_Tamu[0]['fit_VExt'][:,i],HSRL_Tamu[0]['RangeLidar']/1000,color = "#d24787",ls = "--", marker = "h")
            



            axs[0,0].set_title('VBS')
            axs[i,0].set_xlabel('VBS')
            axs[i,0].set_ylabel(wave)
            if i ==0:
                axs[0,0].legend()

            
            axs[i,1].plot(LidarPolSph[0]['meas_DP'][:,i],HSRL_sphrod[0]['RangeLidar'][:,0]/1000, marker =">",color = "#3B270C", label ="Meas")
            axs[i,1].plot(LidarPolSph[0]['fit_DP'][:,i],HSRL_sphrod[0]['RangeLidar'][:,0]/1000,color = "#025043", marker = "$O$")
            axs[0,1].set_title(f'DP')
            axs[i,1].set_xlabel('DP')
            # axs[i,1].set_ylabel('Range (km)')

            axs[i,2].plot(LidarPolSph[0]['meas_VExt'][:,i],HSRL_sphrod[0]['RangeLidar'][:,0]/1000, marker =">",color = "#3B270C", label ="Meas")
            axs[i,2].plot(LidarPolSph[0]['fit_VExt'][:,i],HSRL_sphrod[0]['RangeLidar'][:,0]/1000,color = "#025043", marker = "$O$")
            
            axs[0,2].set_title('VExt')
            axs[i,2].set_xlabel('VExt')
            # axs[i,2].set_ylabel('Range (km)')


            # axs[i,0].plot(HSRL_Tamu[0]['meas_VBS'][:,i],HSRL_Tamu[0]['RangeLidar'][:,0]/1000, ".b", label ="Meas TAMU")
            
            axs[i,0].plot(LidarPolTAMU[0]['fit_VBS'][:,i],HSRL_Tamu[0]['RangeLidar'][:,0]/1000,color = "#d24787",ls = "--", label="Hex", marker = "h")
            if i ==0:
                axs[0,0].legend()


            # axs[i,1].plot(HSRL_Tamu[0]['meas_DP'][:,i],HSRL_Tamu[0]['RangeLidar'][:,0]/1000, ".b", label ="Meas")
            
            axs[i,1].plot(LidarPolTAMU[0]['fit_DP'][:,i],HSRL_Tamu[0]['RangeLidar'][:,0]/1000,color = "#d24787", ls = "--",marker = "h")
        

            # axs[i,2].plot(HSRL_Tamu[0]['meas_VExt'][:,i],HSRL_Tamu[0]['RangeLidar'], ".b", label ="Meas")
            axs[i,2].plot(LidarPolTAMU[0]['fit_VExt'][:,i],HSRL_Tamu[0]['RangeLidar']/1000,color = "#d24787",ls = "--", marker = "h")
            

            plt.suptitle(f"Lat: {HSRL_Tamu[0]['latitude']},Lon:{HSRL_Tamu[0]['longitude']} Date: {HSRL_Tamu[0]['datetime']}\n ") #Initial condition strictly constrainted by RSP retrievals
            fig.savefig(f'/home/gregmi/ORACLES/HSRL_RSP/HSRL_{HSRLPixNo}_{RSP_PixNo}_{Retrieval_type}.png',dpi = 300)


        fig, axs = plt.subplots()
        axs.plot(HSRL_sphrod[0]['meas_DP'][:,2],HSRL_sphrod[0]['RangeLidar'][:,0]/1000, marker =">",color = "#3B270C", label ="Meas")
        axs.plot(HSRL_sphrod[0]['fit_DP'][:,2],HSRL_sphrod[0]['RangeLidar'][:,0]/1000,color = "#025043", marker = "$O$")
        axs.plot(HSRL_Tamu[0]['fit_DP'][:,2],HSRL_Tamu[0]['RangeLidar'][:,0]/1000,color = "#d24787", ls = "--",marker = "h")
        axs.set_xlabel('DP')
        # plt.suptitle(f" Initial conditions strictly\n constrainted by RSP retrievals ") #Initial condition  constrainted by RSP retrievals
        fig.savefig(f'/home/gregmi/ORACLES/HSRL_RSP/HSRLProfile_{HSRLPixNo}_{RSP_PixNo}_{Retrieval_type}.png',dpi = 300)

        # fig2= plt.plot()
        # plt.plot(rslts_Sph[0]['lambda'],rslts_Sph[0]['aod'],label = "RSP Sphd")
        # plt.plot(rslts_Tamu[0]['lambda'],rslts_Tamu[0]['aod'],label = "RSP Tamu")
        # plt.plot(HSRL_sphrod[0]['lambda'],HSRL_sphrod[0]['aod'],label = "HSRL Sphd")
        # plt.plot(HSRL_Tamu[0]['lambda'],HSRL_Tamu[0]['aod'],label = "HSRL TAMU")
        
        Spheriod = HSRL_sphrod[0]
        Hex= HSRL_Tamu[0]
        
        #Stokes Vectors Plot
        date_latlon = ['datetime', 'longitude', 'latitude']
        Xaxis = ['r','lambda','sca_ang','rv','height']
        Retrival = ['dVdlnr','aodMode','ssaMode','n', 'k']
        #['sigma', 'vol', 'aodMode','ssaMode', 'rEff', 'costVal']
        Angles =   ['sza', 'vis', 'fis','angle' ]
        Stokes =   ['meas_I', 'fit_I', 'meas_PoI', 'fit_PoI']
        Pij    = ['p11', 'p12', 'p22', 'p33'], 
        Lidar=  ['heightStd','g','LidarRatio','LidarDepol', 'gMode', 'LidarRatioMode', 'LidarDepolMode']


        
        # Plot the AOD data
        y = [0,1,2,0,1,2,]
        x = np.repeat((0,1),3)
        mode_v = ["fine", "coarse"]
        linestyle =[':', '-']

        cm_sp = ['#008080',"#C1E1C1" ]
        cm_t = ['#900C3F',"#FF5733" ]
        color_sph = '#0c7683'
        color_tamu = "#BC106F"

        #Retrivals:
        fig, axs = plt.subplots(nrows= 5, ncols=1, figsize=(7, 30))
        for i in range(len(Retrival)):
            for mode in range(Spheriod['r'].shape[0]): #for each modes
                if i ==0:
                    axs[i].plot(Spheriod['r'][mode], Spheriod[Retrival[i]][mode], marker = "$O$",color = cm_sp[mode],ls = linestyle[mode], label=f"Sphrod_{mode_v[mode]}")
                    axs[i].plot(Hex['r'][mode],Hex[Retrival[i]][mode], marker = "H", color = cm_t[mode] , ls = linestyle[mode],label=f"Hex_{mode_v[mode]}")
                    axs[i].set_xlabel('Radius')
                    axs[i].set_xscale("log")
                else:
                    axs[i].plot(Spheriod['lambda'], Spheriod[Retrival[i]][mode], marker = "$O$",color = cm_sp[mode],ls = linestyle[mode], label=f"Sphrod_{mode_v[mode]}")
                    axs[i].plot(Hex['lambda'],Hex[Retrival[i]][mode], marker = "H",color = cm_t[mode] , ls = linestyle[mode],label=f"Hex_{mode_v[mode]}")
                    axs[i].set_xticks(Spheriod['lambda'])
                    axs[i].set_xticklabels(['0.355', '0.532', '1.064'])

                    axs[i].xaxis.set_tick_params(labelbottom=False)
            axs[4].xaxis.set_tick_params(labelbottom=True)
            axs[i].set_ylabel(f'{Retrival[i]}')
            axs[4].set_xlabel(r'$\lambda$')

            axs[0].legend()

        lat_t = Hex['latitude']
        lon_t = Hex['longitude']
        dt_t = Hex['datetime']
        plt.suptitle(f'HSRL Retrievals\n Lat:{lat_t} Lon :{lon_t}\n Date: {dt_t}  Pixel:{HSRLPixNo} \n Initial condition strictly constrainted by RSP retrievals ') #\n 

        fig.savefig(f'/home/gregmi/ORACLES/HSRL_RSP/microph_{HSRLPixNo}_{Retrieval_type}.png',dpi = 300)
        print('HSRL: sph tamu; RSP: sph, tamu' )
        print(HSRL_sphrod[0]['costVal'],HSRL_Tamu[0]['costVal'], rslts_Sph[0]['costVal'],rslts_Tamu[0]['costVal'])
        return
    
    def PlotOutput_Separate():


        fig, axs= plt.subplots(nrows = 1, ncols =3, figsize= (18,6))
        for i in range(3):

            wave = np.str(HSRL_sphrod[0]['lambda'][i]) +"μm"
            axs[i].plot(HSRL_sphrod[0]['meas_VBS'][:,i],HSRL_sphrod[0]['RangeLidar'][:,0]/1000, marker =">",color = "#3B270C", label ="Meas")
            axs[i].plot(HSRL_sphrod[0]['fit_VBS'][:,i],HSRL_sphrod[0]['RangeLidar'][:,0]/1000,color = "#025043", marker = "$O$",label ="Sphd")
            axs[i].plot(HSRL_Tamu[0]['fit_VBS'][:,i],HSRL_Tamu[0]['RangeLidar'][:,0]/1000,color = "#d24787",ls = "--", label="Hex", marker = "h")

            axs[i].set_xlabel('VBS')
            axs[i].set_title(wave)
            if i ==0:
                axs[0].legend()
            plt.suptitle(f"Lat: {HSRL_Tamu[0]['latitude']},Lon:{HSRL_Tamu[0]['longitude']} Date: {HSRL_Tamu[0]['datetime']}\n ") #Initial condition strictly constrainted by RSP retrievals

        fig, axs= plt.subplots(nrows = 1, ncols =3, figsize= (18,6))
        for i in range(3):

            axs[i].plot(HSRL_sphrod[0]['meas_DP'][:,i],HSRL_sphrod[0]['RangeLidar'][:,0]/1000, marker =">",color = "#3B270C", label ="Meas")
            axs[i].plot(HSRL_sphrod[0]['fit_DP'][:,i],HSRL_sphrod[0]['RangeLidar'][:,0]/1000,color = "#025043", marker = "$O$",label ="Sphd")
            axs[i].plot(HSRL_Tamu[0]['fit_DP'][:,i],HSRL_Tamu[0]['RangeLidar'][:,0]/1000,color = "#d24787", ls = "--",marker = "h")
            axs[i].set_xlabel('DP %')
            axs[i].set_title(wave)
            if i ==0:
                axs[0].legend()
            plt.suptitle(f"Lat: {HSRL_Tamu[0]['latitude']},Lon:{HSRL_Tamu[0]['longitude']} Date: {HSRL_Tamu[0]['datetime']}\n ") #Initial condition strictly constrainted by RSP retrievals

        fig, axs= plt.subplots(nrows = 1, ncols =3, figsize= (18,6))
        for i in range(3):

            axs[i].plot(HSRL_sphrod[0]['meas_VExt'][:,i],HSRL_sphrod[0]['RangeLidar'][:,0]/1000, marker =">",color = "#3B270C", label ="Meas")
            axs[i].plot(HSRL_sphrod[0]['fit_VExt'][:,i],HSRL_sphrod[0]['RangeLidar'][:,0]/1000,color = "#025043", marker = "$O$",label ="Sphd")
            axs[i].plot(HSRL_Tamu[0]['fit_VExt'][:,i],HSRL_Tamu[0]['RangeLidar']/1000,color = "#d24787",ls = "--", marker = "h")
            axs[i].set_xlabel('VExt')
            axs[i].set_title(wave)
            if i ==0:
                axs[0].legend()
            plt.suptitle(f"Lat: {HSRL_Tamu[0]['latitude']},Lon:{HSRL_Tamu[0]['longitude']} Date: {HSRL_Tamu[0]['datetime']}\n ") #Initial condition strictly constrainted by RSP retrievals

        fig, axs= plt.subplots(nrows = 1, ncols =3, figsize= (18,6))
        IndexH = [0,3,7]
        for i in range(3):

            wave = np.str(LidarPolSph[0]['lambda'][IndexH[i]]) +"μm"
            axs[i].plot(LidarPolSph[0]['meas_VBS'][:,IndexH[i]],LidarPolSph[0]['RangeLidar'][:,0]/1000, marker =">",color = "#3B270C", label ="Meas")
            axs[i].plot(LidarPolSph[0]['fit_VBS'][:,IndexH[i]],LidarPolSph[0]['RangeLidar'][:,0]/1000,color = "#025043", marker = "$O$",label ="Sphd")
            axs[i].plot(LidarPolTAMU[0]['fit_VBS'][:,IndexH[i]],LidarPolTAMU[0]['RangeLidar'][:,0]/1000,color = "#d24787",ls = "--", label="Hex", marker = "h")

            axs[i].set_xlabel('VBS')
            axs[i].set_title(wave)
            if i ==0:
                axs[0].legend()
            plt.suptitle(f"Lat: {HSRL_Tamu[0]['latitude']},Lon:{HSRL_Tamu[0]['longitude']} Date: {HSRL_Tamu[0]['datetime']}\n ") #Initial condition strictly constrainted by RSP retrievals

        fig, axs= plt.subplots(nrows = 1, ncols =3, figsize= (18,6))
        for i in range(3):

            axs[i].plot(LidarPolSph[0]['meas_DP'][:,IndexH[i]],LidarPolSph[0]['RangeLidar'][:,0]/1000, marker =">",color = "#3B270C", label ="Meas")
            axs[i].plot(LidarPolSph[0]['fit_DP'][:,IndexH[i]],LidarPolSph[0]['RangeLidar'][:,0]/1000,color = "#025043", marker = "$O$",label ="Sphd")
            axs[i].plot(LidarPolTAMU[0]['fit_DP'][:,IndexH[i]],LidarPolTAMU[0]['RangeLidar'][:,0]/1000,color = "#d24787", ls = "--",marker = "h", label="Hex")
            axs[i].set_xlabel('DP %')
            axs[i].set_title(wave)
            if i ==0:
                axs[0].legend()
            plt.suptitle(f"Lat: {LidarPolTAMU[0]['latitude']},Lon:{LidarPolTAMU[0]['longitude']} Date: {HSRL_Tamu[0]['datetime']}\n Lidar+polarimeter Retrievals ") #Initial condition strictly constrainted by RSP retrievals

        fig, axs= plt.subplots(nrows = 1, ncols =3, figsize= (18,6))
        for i in range(3):

            axs[i].plot(LidarPolSph[0]['meas_VExt'][:,IndexH[i]],LidarPolSph[0]['RangeLidar'][:,0]/1000, marker =">",color = "#3B270C", label ="Meas")
            axs[i].plot(LidarPolSph[0]['fit_VExt'][:,IndexH[i]],LidarPolSph[0]['RangeLidar'][:,0]/1000,color = "#025043", marker = "$O$",label ="Sphd")
            axs[i].plot(LidarPolTAMU[0]['fit_VExt'][:,IndexH[i]],LidarPolTAMU[0]['RangeLidar']/1000,color = "#d24787",ls = "--", marker = "h", label="Hex")
            axs[i].set_xlabel('VExt')
            axs[i].set_title(wave)
            if i ==0:
                axs[0].legend()
            plt.suptitle(f"Lat: {LidarPolTAMU[0]['latitude']},Lon:{LidarPolTAMU[0]['longitude']} Date: {HSRL_Tamu[0]['datetime']}\n Lidar+polarimeter Retrievals  ") #Initial condition strictly constrainted by RSP retrievals







