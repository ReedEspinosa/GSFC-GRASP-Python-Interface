"""
# Greema Regmi, UMBC
# Date: Jan 31, 2023

This code reads Polarimetric data from the Campaigns and runs GRASP. This code was created to Validate the Aerosol retrivals performed using Non Spherical Kernels (Hexahedral from TAMU DUST 2020)

"""
# %load_ext autoreload
# %autoreload 2
# %reload_ext autoreload

# %run -d -b runGRASP.py:LINENUM scriptToRun.py
# %load_ext autoreload


import sys
import warnings
from CreateRsltsDict import Read_Data_RSP_Oracles, Read_Data_HSRL_constHgt
from CreateRsltsDict import Read_Data_HSRL_Oracles,Read_Data_HSRL_Oracles_Height,Read_Data_HSRL_constHgt
import netCDF4 as nc
from runGRASP import graspDB, graspRun, pixel, graspYAML
from matplotlib import pyplot as plt

import os
if os.uname()[1]=='uranus': plt.switch_backend('agg')
import numpy as np
import datetime as dt
from numpy import nanmean
import h5py 
sys.path.append("/home/gregmi/git/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases")
from architectureMap import returnPixel
from matplotlib.font_manager import FontProperties
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as mpatches
from matplotlib.ticker import ScalarFormatter
# from Plot_ORACLES import PltGRASPoutput, PlotRetrievals
import yaml

import pickle




# # Path to the Polarimeter data (RSP, In this case)

# #Case1
# file_path = "/home/gregmi/ORACLES/RSP1-L1C_P3_20180922_R03/"  #Path to the ORACLE data file
# file_name =  "/RSP1-P3_L1C-RSPCOL-CollocatedRadiances_20180922T151106Z_V003-20210421T233946Z.h5" #Name of the ORACLES file
# #Paths to the Lidar Data


# file_path = '/home/gregmi/ORACLES/Case2'
# file_name ='/RSP1-P3_L1C-RSPCOL-CollocatedRadiances_20180921T180327Z_V003-20210421T232647Z.h5'

# #Case 1: 22nd sept 2018, ORACLES
# HSRLfile_path = "/home/gregmi/ORACLES/HSRL" #Path to the ORACLE data file 
# HSRLfile_name =  "/HSRL2_P3_20180922_R2.h5" #Name of the ORACLES file

# #Case2: 21st sept 2018, ORACLES

# HSRLfile_path = '/home/gregmi/ORACLES/Case2'
# HSRLfile_name =  "/HSRL2_P3_20180921_R2.h5"


# #Path to the gas absorption (tau) values for gas absorption correction
# GasAbsFn = '/home/gregmi/ORACLES/UNL_VRTM/shortwave_gas.unlvrtm.nc'
# SpecResFnPath = '/home/gregmi/ORACLES/RSP_Spectral_Response/'
# #This is required if we want to configure the HSRL yaml file based on the GRASP output for the RSP
# noMod =3  #number of aerosol mode, here 2 for fine+coarse mode configuration
# maxr=1.05  #set max and min value : here max = 5% incease, min 5% decrease : this is a very narrow distribution
# minr =0.95
# a=1 #no of char # this is a varible added to char and modes to avoid char[0]/ mod[0] which doesnt exist

# # #Saving state variables and noises from yaml file
# # def Normalize_prof():

        
def VariableNoise(YamlFileName,nwl=3): #This will store the inforamtion of the characteristics and noises from the yaml file to a string for reference. 
    
    noMod =2  #number of aerosol mode, here 2 for fine+coarse mode configuration
    maxr=1.05  #set max and min value : here max = 5% incease, min 5% decrease : this is a very narrow distribution
    minr =0.95
    a=1 #no of char # this is a varible added to char and modes to avoid char[0]/ mod[0] which doesnt exist
    nwl=3
    Info = []
    with open(YamlFileName, 'r') as f:  
        data = yaml.safe_load(f)
    for i in range (1,nwl+1):
        Noise_yaml = data['retrieval']['inversion']['noises'][f'noise[{i}]']
        errType =  Noise_yaml['error_type']
        noiseName = Noise_yaml[f'measurement_type[1]']['type'] #type of measurement
        sd = Noise_yaml['standard_deviation'] #satndard divation for that meas typ
        NoiseInfo = f'The {errType} noise for {noiseName}  are {sd} '
        Info.append(NoiseInfo )

    #Characteristics'

    YamlChar =[]  #This list stores the name of the charater types in the yaml files
    noYmlChar = np.arange(1,7) #No of aerosol characters types in the yaml file (This can be adjusted based on the parameters we want to change)
    
    for i in noYmlChar:
        YamlChar.append(data['retrieval']['constraints'][f'characteristic[{i}]']['type'])
    print(len(YamlChar))
    
    #change the yaml intitial conditions using the RSP GRASP output
    for i in range(len(YamlChar)): #loop over the character types in the list
        for noMd in range(noMod): #loop over the aerosol modes (i.e 2 for fine and coarse)
            #State Varibles from yaml file: 
            typ= data['retrieval']['constraints'][f'characteristic[{i+1}]']['type']
            initCond = data['retrieval']['constraints'][f'characteristic[{i+a}]'][f'mode[{noMd+a}]']['initial_guess']
            # if YamlChar[i] == 'aerosol_concentration': #Update the in and max values from aerosol properties retrieved from the RSP measurements
            if YamlChar[i] == 'vertical_profile_normalized':
                val = [initCond['value'] [0],initCond['value'] [-1]]
                maxV = [initCond['max'][0],initCond['max'][-1] ]
                minV = [initCond['min'][0],initCond['min'][-1] ]
            else:
                
            #     initCond['value'] = float(RSP_rslt['vol'][noMd]) #value from the GRASP result for RSP
                val = initCond['value'] 
                maxV = initCond['max'] 
                minV = initCond['min'] 
            charInfo =  f'{typ} for mode {noMd+1} is {val} ,  ranging from  max: {maxV} to min : {minV} '
            Info.append(charInfo)
    
    #COmbining all the infroamtion into single sting
    combined_string = '-'

    for i in range (len(Info)):
        line = Info[i]
        combined_string = combined_string + "\n" + line

    return combined_string

def update_HSRLyaml(YamlFileName, RSP_rslt, noMod, maxr, minr, a, Kernel_type,ConsType):
    '''HSRL provides information on the aerosol in each height,
      and RSP measures the total column intergrated values. We can use the total column information to bound the HSRL retrievals
    '''
    #This function creates new yaml with initial conditions updated form microphysical properties of  Polarimeter retrievals

    # Load the YAML file for HSRL
    with open(YamlFileName, 'r') as f:  
        data = yaml.safe_load(f)

    YamlChar =[]  #This list stores the name of the charater types in the yaml files
    noYmlChar = np.arange(1,7) #No of aerosol characters types in the yaml file (This can be adjusted based on the parameters we want to change)
    
    for i in noYmlChar:
        YamlChar.append(data['retrieval']['constraints'][f'characteristic[{i}]']['type'])
    # RSP_rslt = np.load('RSP_sph.npy',allow_pickle= True).item()
    print(len(YamlChar))

    # if ManualChange == True:
    #     initCond = data['inversion']['constraints'][characteristic[]][f'mode[{noMd+a}]']['initial_guess']
    #     initCond = initCond['value'] = float(RSP_rslt['vol'][noMd]) 

    
    #change the yaml intitial conditions using the RSP GRASP output
    for i in range(len(YamlChar)): #loop over the character types in the list
        for noMd in range(noMod): #loop over the aerosol modes (i.e 2 for fine and coarse)
            #State Varibles from yaml file: 
            
            # data['retrieval']['constraints'][f'characteristic[0]'][f'mode[{noMd+a}]']['value'][0] =  1e-8
            # data['retrieval']['constraints'][f'characteristic[0]'][f'mode[{noMd+a}]']['max'][0] =1e-7
            # data['retrieval']['constraints'][f'characteristic[0]'][f'mode[{noMd+a}]']['min'][0] =1e-9
            # #     # print("Updating",YamlChar[i])
            initCond = data['retrieval']['constraints'][f'characteristic[{i+a}]'][f'mode[{noMd+a}]']['initial_guess']
            # if YamlChar[i] == 'aerosol_concentration': #Update the in and max values from aerosol properties retrieved from the RSP measurements
            # #     # initCond['max'] = float(np.max(RSP_rslt['vol'][noMd])*maxr) #value from the GRASP result for RSP
            # #     # initCond['min'] = float(RSP_rslt['vol'][noMd]*minr)
            #     initCond['value'] = float(RSP_rslt['vol'][noMd]) #value from the GRASP result for RSP
            if YamlChar[i] == 'size_distribution_lognormal':
                initCond['value'] = float(RSP_rslt['rv'][noMd]),float(RSP_rslt['sigma'][noMd])
                initCond['max'] =float(RSP_rslt['rv'][noMd]*maxr),float(RSP_rslt['sigma'][noMd]*maxr)
                initCond['min'] =float(RSP_rslt['rv'][noMd]*minr),float(RSP_rslt['sigma'][noMd]*minr)
                # print("Updating",YamlChar[i])
                if ConsType == 'strict': #This will set the retrieved parameteter to false. 
                    data['retrieval']['constraints'][f'characteristic[{i+a}]']['retrieved'] = 'false'
            
            if YamlChar[i] == 'real_part_of_refractive_index_spectral_dependent':
                initCond['index_of_wavelength_involved'] = [1,2,3]
                #Adding off sets because the HSRL abd RSP wavelengths dont match, this should be improved
                if noMd==0: offs = 0
                if noMd==1: offs = 0

                initCond['value'] =float(RSP_rslt['n'][noMd][0]+offs),float(RSP_rslt['n'][noMd][2]),float(RSP_rslt['n'][noMd][4])
                initCond['max'] =float((RSP_rslt['n'][noMd][0]+offs)*maxr),float(RSP_rslt['n'][noMd][2]*maxr),float(RSP_rslt['n'][noMd][4]*maxr)
                initCond['min'] =float((RSP_rslt['n'][noMd][0]+offs)*minr),float(RSP_rslt['n'][noMd][2]*minr),float(RSP_rslt['n'][noMd][4]*minr)
                # print("Updating",YamlChar[i])
                if ConsType == 'strict':  # tot set retrival in setting files to False
                    data['retrieval']['constraints'][f'characteristic[{i+a}]']['retrieved'] = 'false'
            
            if YamlChar[i] == 'imaginary_part_of_refractive_index_spectral_dependent':
                initCond['index_of_wavelength_involved'] = [1,2,3]
                initCond['value'] =float(RSP_rslt['k'][noMd][0]),float(RSP_rslt['k'][noMd][2]),float(RSP_rslt['k'][noMd][4])
                initCond['max'] =float(RSP_rslt['k'][noMd][0]*maxr),float(RSP_rslt['k'][noMd][2]*maxr),float(RSP_rslt['k'][noMd][4]*maxr)
                initCond['min'] = float(RSP_rslt['k'][noMd][0]*minr),float(RSP_rslt['k'][noMd][2]*minr),float(RSP_rslt['k'][noMd][4]*minr)
                # print("Updating",YamlChar[i])
                if ConsType == 'strict':
                    data['retrieval']['constraints'][f'characteristic[{i+a}]']['retrieved'] = 'false'
            
            if YamlChar[i] == 'sphere_fraction':
               
                initCond['value'] = float(RSP_rslt['sph'][noMd]/100)
                initCond['max'] =float(RSP_rslt['sph'][noMd] *maxr/100) #GARSP output is in %
                initCond['min'] =float(RSP_rslt['sph'][noMd] *minr/100)
                # print("Updating",YamlChar[i])
                if ConsType == 'strict':
                    data['retrieval']['constraints'][f'characteristic[{i+a}]']['retrieved'] = 'false'
            


    if Kernel_type == "sphro":
        UpKerFile = 'Settings_Sphd_RSP_HSRL.yaml' #for spheroidal kernel
    if Kernel_type == "TAMU":
        UpKerFile = 'Settings_TAMU_RSP_HSRL.yaml'#for hexahedral kernel
    
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

def RSP_Run(Kernel_type,file_path,file_name,PixNo,ang1,ang2,TelNo,nwl,GasAbsFn,ModeNo): 
        
        krnlPath='/home/shared/GRASP_GSFC/src/retrieval/internal_files'
        # Kernel_type =  sphro is for the GRASP spheriod kernal, while TAMU is to run with Hexahedral Kernal
        if Kernel_type == "sphro":

            if ModeNo == None or ModeNo == 2:
                fwdModelYAMLpath = '/home/gregmi/git/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/settings_BCK_DoLP_POLAR_2modes_SphrodShape_ORACLE.yml'
            
            if ModeNo == 3:
                # fwdModelYAMLpath ='/home/gregmi/git/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/settings_BCK_DoLP_POLAR_3modes_SphrodShape_ORACLE.yml'
                #Case2 
                fwdModelYAMLpath = '/home/gregmi/git/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/settings_BCK_DoLP_POLAR_3modes_SphrodShape_ORACLE_case2.yml'
            
             # binPathGRASP ='/home/shared/GRASP_GSFC/build_RSP_v112/bin/grasp_app' 
            binPathGRASP ='/home/shared/GRASP_GSFC/build_HEX_v112/bin/grasp_app' #GRASP Executable
           
            savePath=f"/home/gregmi/ORACLES/RSP1-L1C_P3_20180922_R03_{Kernel_type}"
        
        if Kernel_type == "TAMU":
            if ModeNo == None or ModeNo == 2:
                fwdModelYAMLpath = '/home/gregmi/git/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/settings_BCK_DoLP_POLAR_2modes_HexShape_ORACLE.yml'
            if ModeNo == 3:
                fwdModelYAMLpath ='/home/gregmi/git/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/settings_BCK_DoLP_POLAR_3modes_HexShape_ORACLE.yml'
                fwdModelYAMLpath = '/home/gregmi/git/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/settings_BCK_DoLP_POLAR_3modes_HexShape_ORACLE_case2.yml'
            binPathGRASP ='/home/shared/GRASP_GSFC/build_HEX_v112/bin/grasp_app' #GRASP Executable
            # binPathGRASP ='/home/shared/GRASP_GSFC/build_HexV112_4Modes/bin/grasp_app'
            savePath=f"/home/gregmi/ORACLES/RSP1-L1C_P3_20180922_R03_{Kernel_type}"

        #rslt is the GRASP rslt dictionary or contains GRASP Objects
        rslt = Read_Data_RSP_Oracles(file_path,file_name,PixNo,ang1,ang2,TelNo, nwl,GasAbsFn)
        print(rslt['OBS_hght'])
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
def HSLR_run(Kernel_type,HSRLfile_path,HSRLfile_name,HSRLPixNo, nwl,updateYaml= None,ConsType = None,releaseYAML =True, ModeNo=None,VertProfConstrain =None,Simplestcase =None,rslts_Sph=None):
        #Path to the kernel files
        krnlPath='/home/shared/GRASP_GSFC/src/retrieval/internal_files'
        if Kernel_type == "sphro":  #If spheroid model

            #Path to the yaml file for sphreroid model
            # fwdModelYAMLpath ='/home/gregmi/git/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/settings_BCK_POLAR_4modes_Shape_ORACLE1.yml'
            if ModeNo == None or ModeNo == 2:
                fwdModelYAMLpath ='/home/gregmi/git/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/settings_BCK_POLARandLIDAR_10Vbins_2modes_ORACLES.yml'
            if ModeNo == 3:
                fwdModelYAMLpath ='/home/gregmi/git/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/settings_BCK_POLAR_3modes_Shape_ORACLE.yml'
                fwdModelYAMLpath ='/home/gregmi/git/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/settings_BCK_POLAR_3modes_Shape_ORACLE_Case2.yml'

            
            if ModeNo ==4:
                fwdModelYAMLpath = '/home/gregmi/git/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/settings_BCK_POLAR_4modes_Shape_ORACLE1.yml'
            
            if updateYaml == True:  # True if init conditions for Yaml file for HSRL is updated from the GRASP output from RSP
                maxr=1.05  #set max and min value : here max = 5% incease, min 5% decrease : this is a very narrow distribution
                minr =0.95
                a=1
                
                update_HSRLyaml(fwdModelYAMLpath, rslts_Sph[0], ModeNo, maxr, minr, a,Kernel_type,ConsType)
                fwdModelYAMLpath ='/home/gregmi/git/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/Settings_Sphd_RSP_HSRL.yaml'
            
            
            # binPathGRASP = path toGRASP Executable for spheriod model
            binPathGRASP ='/home/shared/GRASP_GSFC/build_HEX_v112/bin/grasp_app'
            # binPathGRASP ='/home/shared/GRASP_GSFC/build_t/bin/grasp_app'
            # binPathGRASP ='/home/shared/GRASP_GSFC/build-tmu/bin/grasp_app' #GRASP Executable
            # binPathGRASP ='/home/shared/GRASP_GSFC/build_HEX_v112/bin/grasp_app'
            info = VariableNoise(fwdModelYAMLpath,nwl)
            # binPathGRASP ='/home/shared/GRASP_GSFC/build_RSP_v112/bin/grasp_app' 
            savePath=f"/home/gregmi/ORACLES/HSRL1_P3_20180922_R03_{Kernel_type}"
        
        if Kernel_type == "TAMU":

            if ModeNo == None or ModeNo == 2:
                fwdModelYAMLpath ='/home/gregmi/git/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/settings_BCK_POLARandLIDAR_10Vbins_2modes_Hex.yml'
            if ModeNo == 3:
                fwdModelYAMLpath ='/home/gregmi/git/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/settings_BCK_POLAR_3modes_Shape_Hex.yml'
                fwdModelYAMLpath ='/home/gregmi/git/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/settings_BCK_POLAR_3modes_Shape_Hex_Case2.yml'
            
            
            if ModeNo ==4:
                fwdModelYAMLpath = '/home/gregmi/git/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/settings_BCK_POLAR_4modes_Shape_HEX.yml'
            # fwdModelYAMLpath ='/home/gregmi/git/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/settings_BCK_POLAR_3modes_Shape_Hex.yml'
            # fwdModelYAMLpath = '/home/gregmi/git/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/settings_BCK_POLAR_3modes_Shape_ORACLE.yml'
            if updateYaml == True:# True if init conditions for Yaml file for HSRL is updated from the GRASP output from RSP
                maxr=1.05  #set max and min value : here max = 5% incease, min 5% decrease : this is a very narrow distribution
                minr =0.95
                a=1
                
                
                update_HSRLyaml(fwdModelYAMLpath, rslts_Tamu[0], ModeNo, maxr, minr, a,Kernel_type,ConsType)
                fwdModelYAMLpath ='/home/gregmi/git/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/Settings_TAMU_RSP_HSRL.yaml'
            #Path to the GRASP Executable for TAMU
            info = VariableNoise(fwdModelYAMLpath,nwl)
            # binPathGRASP ='/home/shared/GRASP_GSFC/build-tmu/bin/grasp_app' #GRASP Executable
            binPathGRASP ='/home/shared/GRASP_GSFC/build_HEX_v112/bin/grasp_app'
            # binPathGRASP ='/home/shared/GRASP_GSFC/build_HexV112_4Modes/bin/grasp_app'
            #Path to save output plot
            savePath=f"/home/gregmi/ORACLES/RSP1-L1C_P3_20180922_R03_{Kernel_type}"

        #This section is for the normalization paramteter in the yaml settings file
        #rslt is the GRASP rslt dictionary or contains GRASP Objects
        # DictHsrl = Read_Data_HSRL_Oracles_Height(HSRLfile_path,HSRLfile_name,HSRLPixNo)

        if Simplestcase == True:  #Height grid is constant and no gas correction applied
            DictHsrl = Read_Data_HSRL_Oracles_Height(HSRLfile_path,HSRLfile_name,HSRLPixNo,gaspar =None,SimpleCase = True)
        else: #Variable grid height and gas correction applied
            DictHsrl = Read_Data_HSRL_Oracles_Height(HSRLfile_path,HSRLfile_name,HSRLPixNo,gaspar =True)

        rslt = DictHsrl[0]
        max_alt = rslt['OBS_hght']
        Vext1 = rslt['meas_VExt'][:,0]
        Vext1[np.where(Vext1<=0)] = 1e-6
      
        #Updating the normalization values in the settings file. 
        with open(fwdModelYAMLpath, 'r') as f:  
            data = yaml.safe_load(f)
        
        if VertProfConstrain == True: #True of we want to apply vertical profile normalization

            # Vext1 = rslt['meas_VExt'][:,0]
            hgt =  rslt['RangeLidar'][:,0][:]
            DP1064= rslt['meas_DP'][:,2][:]

            #Boundary layer height
            BLH_indx = np.where(np.gradient(DP1064,hgt) == np.max(np.gradient(DP1064,hgt)))[0]
            BLH = hgt[np.where(np.gradient(DP1064,hgt) == np.max(np.gradient(DP1064,hgt)))]
             #altitude of the aircraft
            # print(rslt['OBS_hght'])
            
            DMR1 = DictHsrl[2]
            # print(DMR1)
            # DMR1[0][np.where(np.isnan(DMR1[0]))] = 0
            if np.any(DMR1 > 1):
                warnings.warn('DMR > 1, renormalizing', UserWarning)
                DMR = DMR1/np.nanmax(DMR1)
            else: 
                DMR= DMR1 

            # else:
            #     DMR = DMR1
                #Renormalize.
            Vext1[np.where(Vext1<=0)] = 1e-6
            Vextoth = abs(1-0.99999*DMR)*Vext1
            
            VextDst = Vext1 - Vextoth 
            # VextDst[np.where(VextDst<=0)] = 1e-6
            

            # DMR[DMR>1] = 1  # ratios must be 1
            # VextDst = 0.99999*DMR*Vext1

            aboveBL = Vextoth[:BLH_indx[0]]
            mean = np.nanmean(aboveBL[:50])
            belowBL = Vextoth[BLH_indx[0]:]



            Vextfine = np.concatenate((0.99999*aboveBL, np.ones(len(belowBL))*10**-8))
            VextSea = Vextoth -Vextfine

            # VBack = 0.00002*Vextoth
            # Voth = 0.999998*Vextoth
            # VextSea = np.concatenate((VBack[:BLH_indx[0]],Voth[BLH_indx[0]:]))
            # Vextfine =np.concatenate((Voth[:BLH_indx[0]],VBack[BLH_indx[0]:]))

            DstProf =VextDst/ np.trapz(VextDst[::-1],hgt[::-1])
            FineProf = Vextfine/np.trapz(Vextfine[::-1],hgt[::-1])
            SeaProf = VextSea/ np.trapz(VextSea[::-1],hgt[::-1])

            fig = plt.figure()
            plt.plot(VextDst,hgt, color = '#067084',label='Dust')
            plt.plot(VextSea,hgt,color ='#6e526b',label='Salt')
            plt.plot(Vextfine,hgt,color ='y',label='fine')
            plt.plot(Vext1,hgt,color='#8a9042',ls = '--',label='Total Ext')
            plt.plot(Vext1[BLH_indx],hgt[BLH_indx],color='#660000',marker = 'x',label='BLH')
            plt.legend()

            fig = plt.figure()
            plt.plot(FineProf,hgt,color ='y',label='fine')
            plt.plot(SeaProf,hgt,color ='#6e526b',label='Salt')
            plt.plot(DstProf,hgt, color = '#067084',label='Dust')
            plt.legend()

            for noMd in range(ModeNo+1): #loop over the aerosol modes (i.e 2 for fine and coarse)

                # MinVal = np.repeat(np.minimum(np.minimum(FineProf[FineProf>1.175494e-38],DstProf),SeaProf),10).tolist() # the size of the list will be adjusted later in the code
             # Updating the vertical profile norm values in yaml file: 
                if noMd ==1:  #Mode 1
                    # data['retrieval']['constraints'][f'characteristic[1]'][f'mode[{noMd}]']['initial_guess']['min'] =   MinVal  #Int his version of GRASP min and val value should be same for each modes
                    data['retrieval']['constraints'][f'characteristic[1]'][f'mode[{noMd}]']['initial_guess']['value'] =  FineProf.tolist()
                if noMd ==2: #Mode 2 
                    # data['retrieval']['constraints'][f'characteristic[1]'][f'mode[{noMd}]']['initial_guess']['min'] =   MinVal
                    data['retrieval']['constraints'][f'characteristic[1]'][f'mode[{noMd}]']['initial_guess']['value'] =  DstProf.tolist()
                if noMd ==3: 
                    # data['retrieval']['constraints'][f'characteristic[1]'][f'mode[{noMd}]']['initial_guess']['min'] =   MinVal
                    data['retrieval']['constraints'][f'characteristic[1]'][f'mode[{noMd}]']['initial_guess']['value'] =  SeaProf.tolist()
        
            if Kernel_type == "sphro":
                UpKerFile = 'settings_BCK_POLAR_3modes_Shape_Sph_Update.yml' #for spheroidal kernel
            if Kernel_type == "TAMU":
                UpKerFile = 'settings_BCK_POLAR_3modes_Shape_HEX_Update.yml'#for hexahedral kernel
        
            ymlPath = '/home/gregmi/git/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/'

            with open(ymlPath+UpKerFile, 'w') as f2: #write the chnages to new yaml file
                yaml.safe_dump(data, f2)
            #     # print("Updating",YamlChar[i])
                fwdModelYAMLpath = ymlPath+UpKerFile

            Vext = rslt['meas_VExt']
        
        #Running GRASP
        maxCPU = 3 #maximum CPU allocated to run GRASP on server
        gRuns = []
        yamlObj = graspYAML(baseYAMLpath= fwdModelYAMLpath)
        # ymlO = yamlObj._repeatElementsInField(fldName=fwdModelYAMLpath, Nrepeats =10, λonly=False)
        #eventually have to adjust code for height, this works only for one pixel (single height value)
        gRuns.append(graspRun(pathYAML=yamlObj, releaseYAML= releaseYAML)) # This should copy to new YAML object
        pix = pixel()
        pix.populateFromRslt(rslt, radianceNoiseFun=None, dataStage= 'meas', verbose=False)
        gRuns[-1].addPix(pix)
        gDB = graspDB(graspRunObjs=gRuns, maxCPU=maxCPU)
        #rslts contain all the results form the GRASP inverse run
        rslts, failPix = gDB.processData(binPathGRASP=binPathGRASP, savePath=None, krnlPathGRASP=krnlPath)
        
        return rslts, max_alt, info



def PlotRandomGuess(filename_npy, NoItr):
    noMod=3
    
    fig, ax = plt.subplots(nrows= 3, ncols=2, figsize=(20, 15))
# #     
    loaded_data = np.load(filename_npy, allow_pickle=True)
    
    costVal = []
    colors = cm.rainbow(costVal)
    c = ['#625460','#CB9129','#6495ED']
    
    ls = ['solid','solid','dashed' ]
    
    smoke = mpatches.Patch(color=c[0], label='fine')
    dust = mpatches.Patch(color=c[1], label='Non-sph dust')
    salt = mpatches.Patch(color=c[2], label='Sph sea salt')
    
    
#     for j in range(noMod):

    
#         for y, c in zip(costVal, colors):
#             plt.scatter(loaded_data[i][0]['lambda'], loaded_data[i][0]['n'][j], color=c)
  
    for i in range(NoItr):
        if len(loaded_data[i]) >0:
            costVal.append(loaded_data[i][0]['costVal'])
#             if loaded_data[i][0]['costVal'] < 4.5: 
#                 print(i), print(loaded_data[i][0]['costVal'])
#                 CombinedLidarPolPlot(loaded_data[i],loaded_data[i])

            for j in range(noMod):
                

                ax[0,1].plot(loaded_data[i][0]['lambda'],loaded_data[i][0]['n'][j], alpha = 0.5, color= c[j], ls = ls[j])
                ax[0,1].set_ylabel('n')
                ax[1,1].plot(loaded_data[i][0]['lambda'],loaded_data[i][0]['k'][j], alpha = 0.5, color= c[j], ls = ls[j])
                ax[1,1].set_ylabel('k')
                ax[1,0].plot(loaded_data[i][0]['lambda'],loaded_data[i][0]['aodMode'][j], alpha = 0.5, color= c[j], ls = ls[j])
                ax[1,0].set_ylabel('aodMode')
                ax[2,0].plot(loaded_data[i][0]['lambda'],loaded_data[i][0]['ssaMode'][j], alpha = 0.5, color= c[j], ls = ls[j])
                ax[2,0].set_ylabel('ssaMode')
                ax[2,1].plot(loaded_data[i][0]['r'][j], loaded_data[i][0]['dVdlnr'][j],color= c[j], ls = ls[j], alpha = 0.5,lw = 2)
               
                ax[2,1].set_xlabel(r'rv $ \mu m$')
                ax[2,1].set_ylabel('dVdlnr')
                ax[2,1].set_xscale("log")
        

            
            
    ax[0,0].hist(costVal, color ='#12106C') 
    ax[0,0].set_xlabel('CostVal')
    
    plt.suptitle(f'{filename_npy[:3]} RSP+HSRL Aerosol Retrieval')
    plt.legend(handles=[smoke,dust,salt]) 
    
    plt.savefig(f'{filename_npy[:5]}.png', dpi = 100)
    
    return costVal


def LidarAndMAP(Kernel_type,HSRLfile_path,HSRLfile_name,HSRLPixNo,file_path,file_name,RSP_PixNo,ang1,ang2,TelNo, nwl,GasAbsFn, ModeNo=None, updateYaml= None, RandinitGuess =None , NoItr=None,fnGuessRslt = None):
    failedmeas = 0 #Count of number of meas that failed
    krnlPath='/home/shared/GRASP_GSFC/src/retrieval/internal_files'

    if Kernel_type == "sphro":  #If spheriod model
        #Path to the yaml file for sphriod model
        if ModeNo == None or ModeNo == 2:
            fwdModelYAMLpath = '/home/gregmi/git/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/settings_BCK_LidarAndMAP_V.1.2.yml'
        if ModeNo == 3:
            fwdModelYAMLpath ='/home/gregmi/git/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/settings_BCK_LidarAndMAP_3modes_V.1.2.yml'
        if updateYaml == True:  # True if init conditions for Yaml file for HSRL is updated from the GRASP output from RSP
            update_HSRLyaml(fwdModelYAMLpath, rslts_Sph[0], noMod, maxr, minr, a,Kernel_type)
            
            fwdModelYAMLpath ='/home/gregmi/git/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/Settings_Sphd_RSP_HSRL.yaml'
        # binPathGRASP = path toGRASP Executable for spheriod model
        # binPathGRASP ='/home/shared/GRASP_GSFC/build_SphrdV112_Noise/bin/grasp_app'
        binPathGRASP ='/home/shared/GRASP_GSFC/build_HEX_v112/bin/grasp_app'  #This will work for both the kernels as it has more parameter wettings
       
        # binPathGRASP ='/home/shared/GRASP_GSFC/build_RSP_v112/bin/grasp_app' 
        savePath=f"/home/gregmi/ORACLES/HSRL1_P3_20180922_R03_{Kernel_type}"
        fnGuessRslt ='sphRandnew.npy'
    
    if Kernel_type == "TAMU":
        if ModeNo == None or ModeNo == 2:
            fwdModelYAMLpath = '/home/gregmi/git/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/settings_BCK_LidarAndMAP_V.1.2_TAMU.yml'
        if ModeNo == 3:
            fwdModelYAMLpath ='/home/gregmi/git/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/settings_BCK_LidarAndMAP_3modes_V.1.2_TAMU.yml'
        if updateYaml == True:# True if init conditions for Yaml file for HSRL is updated from the GRASP output from RSP
            update_HSRLyaml(fwdModelYAMLpath, rslts_Tamu[0], noMod, maxr, minr, a,Kernel_type)
            fwdModelYAMLpath ='/home/gregmi/git/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/Settings_TAMU_RSP_HSRL.yaml'

        #Path to the GRASP Executable for TAMU
        # binPathGRASP ='/home/shared/GRASP_GSFC/build_HexV112_Noise/bin/grasp_app' #Recompiled to account for more noise parameter
        binPathGRASP ='/home/shared/GRASP_GSFC/build_HEX_v112/bin/grasp_app' #GRASP Executable
        # binPathGRASP ='/home/shared/GRASP_GSFC/build_HEX_v112/bin/grasp_app'
        #Path to save output plot
        savePath=f"/home/gregmi/ORACLES/RSP1-L1C_P3_20180922_R03_{Kernel_type}"
        fnGuessRslt ='HexRandnew.npy'

# /tmp/tmpn596k7u8$
    rslt_HSRL_1 = Read_Data_HSRL_Oracles_Height(HSRLfile_path,HSRLfile_name,HSRLPixNo)
    rslt_HSRL =  rslt_HSRL_1[0]
    rslt_RSP = Read_Data_RSP_Oracles(file_path,file_name,RSP_PixNo,ang1,ang2,TelNo, nwl,GasAbsFn)
    
    rslt= {}  # Teh order of the data is  First lidar(number of wl ) and then Polarimter data 
    rslt['lambda'] = np.concatenate((rslt_HSRL['lambda'],rslt_RSP['lambda']))

    #Sort the index of the wavelength if arranged in ascending order, this is required by GRASP
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
        #adding values to sort_MAP index positions
        for a in range(rslt_RSP[keys][:,0].shape[0]):
            RSP_var[a][sort_MAP] = rslt_RSP[keys][a]
        rslt[keys] = RSP_var
        RSP_var = np.ones((rslt_RSP['meas_I'].shape[0],rslt['lambda'].shape[0])) * np.nan


    #Lidar Measurements
    HSRL_var = np.ones((rslt_HSRL['meas_VExt'].shape[0],rslt['lambda'].shape[0]))* np.nan
    for keys1 in HSRLkeys:  
        for a in range(rslt_HSRL[keys1][:,0].shape[0]):
            
            HSRL_var[a][sort_Lidar] = rslt_HSRL[keys1][a]

            # 'sza', 'vis','fis'
        rslt[keys1] = HSRL_var
        # Refresh the array by Creating numpy nan array with shape of height x wl, Basically deleting all values
        HSRL_var = np.ones((rslt_HSRL['meas_VExt'].shape[0],rslt['lambda'].shape[0]))* np.nan

    
    for keys in GenKeys:
        rslt[keys] = rslt_RSP[keys]  #Adding the information about lat, lon, datetime and so on from RSP
    
    rslt['OBS_hght'] = rslt_RSP['OBS_hght'] #adding the aircraft altitude 
    rslt['lambda'] = rslt['lambda'][sort]

    #TODO improve this code to work when only mol deopl or gasabs are provided and not both
    if 'gaspar' in  rslt_HSRL:
        gasparHSRL = np.zeros(len(rslt['lambda']))
        gasparHSRL[sort_Lidar] = rslt_HSRL['gaspar']
        
        rslt['gaspar'] = gasparHSRL
        print(rslt['gaspar'])

    if 'gaspar' in  rslt_RSP:
        gasparRSP = np.zeros(len(rslt['lambda']))
        gasparRSP[sort_MAP] = rslt_RSP['gaspar']

        rslt['gaspar'] = gasparRSP
        print(rslt['gaspar'])

    if 'gaspar' in  rslt_RSP and rslt_HSRL :
        gasparB = np.zeros(len(rslt['lambda']))
        gasparB[sort_MAP] = rslt_RSP['gaspar']
        gasparB[sort_Lidar] = rslt_HSRL['gaspar']
        rslt['gaspar'] = gasparB
        print(rslt['gaspar'])


    # rslt2 = rslt_HSRL_1[0]
    Vext1 = rslt_HSRL['meas_VExt'][:,0]
    hgt =  rslt_HSRL['RangeLidar'][:,0][:]
    DP1064= rslt_HSRL['meas_DP'][:,2][:]

    #Boundary layer height
    BLH_indx = np.where(np.gradient(DP1064,hgt) == np.max(np.gradient(DP1064,hgt)))[0]
    BLH = hgt[np.where(np.gradient(DP1064,hgt) == np.max(np.gradient(DP1064,hgt)))]

    
    DMR1 = rslt_HSRL_1[2]
    if np.any(DMR1 > 1):
        warnings.warn('DMR > 1, renormalizing', UserWarning)
        DMR = DMR1/np.nanmax(DMR1)
    else:
        DMR = DMR1
        #Renormalize.
    
    Vext1[np.where(Vext1<=0)] = 1e-6
    Vextoth = abs(1-0.9999*DMR)*Vext1
    VextDst = Vext1 - Vextoth 
    

    # DMR[DMR>1] = 1  # ratios must be 1
    # VextDst = 0.99999*DMR*Vext1
    
    VBack = 0.00002*Vextoth
    Voth = 0.99998*Vextoth

    VextSea = np.concatenate((VBack[:BLH_indx[0]],Voth[BLH_indx[0]:]))
    Vextfine =np.concatenate((Voth[:BLH_indx[0]],VBack[BLH_indx[0]:]))
    
    DstProf =VextDst/ np.trapz(VextDst[::-1],hgt[::-1])
    FineProf = Vextfine/np.trapz(Vextfine[::-1],hgt[::-1])
    SeaProf = VextSea/ np.trapz(VextSea[::-1],hgt[::-1])
    
    # #apriori estimate for the vertical profile
    fig = plt.figure()
    plt.plot(VextDst,hgt, color = '#067084',label='Dust')
    plt.plot(VextSea,hgt,color ='#6e526b',label='Salt')
    plt.plot(Vextfine,hgt,color ='y',label='fine')
    plt.plot(Vext1,hgt,color='#8a9042',ls = '--',label='Total Ext')
    plt.plot(Vext1[BLH_indx],hgt[BLH_indx],color='#660000',marker = 'x',label='BLH')
    plt.legend()

    fig = plt.figure()
    plt.plot(FineProf,hgt,color ='y',label='fine')
    plt.plot(SeaProf,hgt,color ='#6e526b',label='Salt')
    plt.plot(DstProf,hgt, color = '#067084',label='Dust')
    plt.legend()


    # plt.savefig(f'/home/gregmi/ORACLES/HSRL_RSP/DMRHSRL2Retrieval.png', dpi = 400)
    
    #Updating the normalization values in the settings file. 
    with open(fwdModelYAMLpath, 'r') as f:  
        data = yaml.safe_load(f)

    for noMd in range(4): #loop over the aerosol modes (i.e 2 for fine and coarse)
        
            #State Varibles from yaml file: 
        if noMd ==1:
            data['retrieval']['constraints'][f'characteristic[1]'][f'mode[{noMd}]']['initial_guess']['value'] =  FineProf.tolist()
        if noMd ==2:
            data['retrieval']['constraints'][f'characteristic[1]'][f'mode[{noMd}]']['initial_guess']['value'] =  DstProf.tolist()
        if noMd ==3:
            data['retrieval']['constraints'][f'characteristic[1]'][f'mode[{noMd}]']['initial_guess']['value'] =  SeaProf.tolist()
    
    if Kernel_type == "sphro":
        UpKerFile = 'settings_LIDARandPOLAR_3modes_Shape_Sph_Update.yml' #for spheroidal kernel
    if Kernel_type == "TAMU":
        UpKerFile = 'settings_LIDARandPOLAR_3modes_Shape_HEX_Update.yml'#for hexahedral kernel

    ymlPath = '/home/gregmi/git/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/'
    

    with open(ymlPath+UpKerFile, 'w') as f2: #write the chnages to new yaml file
        yaml.safe_dump(data, f2)
        #     # print("Updating",YamlChar[i])

    max_alt = rslt['OBS_hght'] #altitude of the aircraft
    print(rslt['OBS_hght'])

    Vext = rslt['meas_VExt']
    Updatedyaml = ymlPath+UpKerFile

    Finalrslts =[]

    if RandinitGuess == True:
        
        maxCPU = 3
        if NoItr==None: NoItr=1
        gRuns = [[]]*NoItr
        pix = pixel()
        pix.populateFromRslt(rslt, radianceNoiseFun=None, dataStage= 'meas', verbose=False)
        dblist = []
        
        for i in range(NoItr):
            try:
                print(i)
                gyaml = graspYAML(baseYAMLpath=fwdModelYAMLpath)
                yamlObjscramble = gyaml.scrambleInitialGuess(fracOfSpace=1, skipTypes=['vertical_profile_normalized','aerosol_concentration','surface_water_cox_munk_iso'])
                yamlObj = graspYAML(baseYAMLpath=fwdModelYAMLpath)
                gRuns = []
                gRuns.append(graspRun(pathYAML=yamlObj, releaseYAML= True )) # This should copy to new YAML object
                gRuns[-1].addPix(pix)
                gDB = graspDB(graspRunObjs=gRuns, maxCPU=maxCPU,maxT =1)
                dblist.append(gDB)#rslts contain all the results form the GRASP inverse run
                rslts, failPix = gDB.processData(binPathGRASP=binPathGRASP, savePath=None, krnlPathGRASP=krnlPath)

                Finalrslts.append(rslts)
                # if len(rslts) >0:
                # plot_HSRL(rslts[0],rslts[0], forward = True, retrieval = True, Createpdf = True,PdfName =f"/home/gregmi/ORACLES/rsltPdf/HSRL_Only_Plots_{i}.pdf", combinedVal =rslts[2]) 
                plot_HSRL(rslts[0],rslts[0], forward = False, retrieval =True, Createpdf = True,PdfName =f"/home/gregmi/ORACLES/rsltPdf/LIDARPOL_Plots_{i}.pdf")
                RSP_plot(rslts,rslts,RSP_PixNo,LIDARPOL= True,fn = f'Random{i}')


                                
                # nc_file = nc.Dataset('/home/gregmi/git/GSFC-GRASP-Python-Interface/IntialGuessHSRLRSP.nc', 'w', format='NETCDF4')
                # nc_file.close()
                
            except:
                failedmeas +=1
                pass

    else:
        maxCPU = 13 #maximum CPU allocated to run GRASP on server
        gRuns = []
        yamlObj = graspYAML(baseYAMLpath=Updatedyaml)

        

        #eventually have to adjust code for height, this works only for one pixel (single height value)
        gRuns.append(graspRun(pathYAML=yamlObj, releaseYAML= True )) # This should copy to new YAML object
        pix = pixel()
        pix.populateFromRslt(rslt, radianceNoiseFun=None, dataStage= 'meas', verbose=False)
        gRuns[-1].addPix(pix)
        gDB = graspDB(graspRunObjs=gRuns, maxCPU=maxCPU)
        #rslts contain all the results form the GRASP inverse run
        rslts, failPix = gDB.processData(binPathGRASP=binPathGRASP, savePath=None, krnlPathGRASP=krnlPath)
        
        Finalrslts.append(rslts)

    if fnGuessRslt == None:fnGuessRslt = 'try.npy'
        
    np.save(fnGuessRslt, Finalrslts)
    return rslts,Finalrslts,failedmeas,gRuns


def PlotRandomGuess(filename_npy, NoItr):
    fig, ax = plt.subplots(nrows= 3, ncols=2, figsize=(30, 10))
# #     
    loaded_data = np.load(filename_npy, allow_pickle=True)
    
    costVal = []

    for i in range(NoItr):
        if len(loaded_data[i]) >0:
            costVal.append(loaded_data[i][0]['costVal'])

            for j in range(noMod):
                ax[0,1].plot(loaded_data[i][0]['n'][i])
                ax[1,1].plot(loaded_data[i][0]['k'][i])
                ax[1,0].plot(loaded_data[i][0]['aodMode'][i])
                ax[2,0].plot(loaded_data[i][0]['ssaMode'][i])
    ax[0,0].hist(costVal)      
    return

def Netcdf(Finalrslts,NoItr):
    with nc.Dataset('/home/gregmi/git/GSFC-GRASP-Python-Interface/IntialGuessHSRLRSP.nc', 'a') as rootgrp:
        
        data = Finalrslts[1][0][0]
        # Create dimensions
        rootgrp.createDimension('r_dim', data['r'].shape[0])
        rootgrp.createDimension('range_dim', data['r'].shape[1])
        rootgrp.createDimension('d_range', data['meas_DP'].shape[0])
        rootgrp.createDimension('d_wl', data['range'].shape[1])
        rootgrp.createDimension('wlRSP', data['meas_I'].shape[0])
        rootgrp.createDimension('angle', data['meas_I'].shape[1])
        rootgrp.createDimension('allLambda', data['lambda'].shape[0])
        rootgrp.createDimension('iter_d', NoItr)

        rootgrp.createVariable('datetime', 'S19')
        rootgrp.createVariable('datetime', 'S19')
        rootgrp.createVariable('latitude', 'f4')
        rootgrp.createVariable('land_prct', 'f4')
        rootgrp.createVariable('r', 'f4', ('r_dim', 'range_dim','iter_d'))
        rootgrp.createVariable('dVdlnr', 'f4', ('r_dim', 'range_dim','iter_d'))
        rootgrp.createVariable('rv', 'f4', ('r_dim','iter_d'))
        rootgrp.createVariable('sigma', 'f4', ('r_dim','iter_d'))
        rootgrp.createVariable('vol', 'f4', ('r_dim','iter_d'))
        rootgrp.createVariable('sph', 'f4', ('r_dim','iter_d'))
        rootgrp.createVariable('range', 'f4', ('d_range','iter_d'))
        rootgrp.createVariable('beta_ext', 'f4', ('r_dim','d_range','iter_d'))
        rootgrp.createVariable('lambda', 'f4', 'allLambda')[:]
        rootgrp.createVariable('aod', 'f4',  ('allLambda','iter_d'))
        rootgrp.createVariable('aod_mode', 'f4', ('r_dim', 'allLambda','iter_d'))
        rootgrp.createVariable('ssa', 'f4',  ('allLambda','iter_d'))
        rootgrp.createVariable('ssa_mode', 'f4', ('r_dim', 'allLambda','iter_d'))
        rootgrp.createVariable('n', 'f4', ('r_dim','allLambda','iter_d'))
        rootgrp.createVariable('k', 'f4', ('r_dim','allLambda','iter_d'))
        rootgrp.createVariable('lidar_ratio', 'f4', ('allLambda','iter_d'))
        # rootgrp.createVariable('height', 'f4', ('d_range','d_wl'))[:] = data['height']
        # rootgrp.createVariable('r_eff', 'f4', ('r_dim', 'range_dim'))[:] = data['rEff']
        # rootgrp.createVariable('wtr_surf', 'f4', ('r_dim', 'range_dim'))[:] = data['wtrSurf']
        rootgrp.createVariable('cost_val', 'f4', ('r_dim','iter_d'))
        # rootgrp.createVariable('range_lidar', 'f4', ('d_range','d_wl'))[:] = data['RangeLidar']
        rootgrp.createVariable('meas_dp', 'f4', ('d_range','allLambda','iter_d'))
        rootgrp.createVariable('fit_dp', 'f4', ('d_range','allLambda','iter_d'))
        rootgrp.createVariable('meas_vext', 'f4', ('d_range','allLambda','iter_d'))
        rootgrp.createVariable('fit_vext', 'f4', ('d_range','allLambda','iter_d'))
        rootgrp.createVariable('meas_vbs', 'f4', ('d_range','allLambda','iter_d'))
        rootgrp.createVariable('fit_vbs', 'f4', ('d_range','allLambda','iter_d'))
        rootgrp.createVariable('sza', 'f4', ('angle','wlRSP','iter_d'))
        rootgrp.createVariable('vis', 'f4', ('angle','wlRSP','iter_d'))
        rootgrp.createVariable('fis', 'f4', ('angle','wlRSP','iter_d'))
        rootgrp.createVariable('sca_ang', 'f4', ('angle','wlRSP','iter_d'))
        rootgrp.createVariable('meas_i', 'f4', ('angle', 'wlRSP','iter_d'))
        rootgrp.createVariable('fit_i', 'f4', ('angle', 'wlRSP','iter_d'))
        rootgrp.createVariable('meas_p_rel', 'f4', ('angle', 'wlRSP','iter_d'))
        rootgrp.createVariable('fit_p_rel', 'f4', ('angle', 'wlRSP','iter_d'))


        for i in range(NoItr):
            data = Finalrslts[1][i][0]
        # Create variables and assign values
            rootgrp['datetime'] = np.array(str(data['datetime']), dtype='S19')
            rootgrp['longitude'][:] = data['longitude']
            rootgrp.createVariable('latitude', 'f4')[:] = data['latitude']
            rootgrp.createVariable('land_prct', 'f4')[:] = data['land_prct']
            rootgrp.createVariable('r', 'f4', ('r_dim', 'range_dim','iter_d'))[:,:,i] = data['r']
            rootgrp.createVariable('dVdlnr', 'f4', ('r_dim', 'range_dim','iter_d'))[:,:,i] = data['dVdlnr']
            rootgrp.createVariable('rv', 'f4', ('r_dim','iter_d'))[:,i] = data['rv']
            rootgrp.createVariable('sigma', 'f4', ('r_dim','iter_d'))[:,i] = data['sigma']
            rootgrp.createVariable('vol', 'f4', ('r_dim','iter_d'))[:,i] = data['vol']
            rootgrp.createVariable('sph', 'f4', ('r_dim','iter_d'))[:,i] = data['sph']
            rootgrp.createVariable('range', 'f4', ('d_range','iter_d'))[:,i] = data['range'][0]
            rootgrp.createVariable('beta_ext', 'f4', ('r_dim','d_range','iter_d'))[:,:,i] = data['βext']
            rootgrp.createVariable('lambda', 'f4', 'allLambda')[:] = data['lambda']
            rootgrp.createVariable('aod', 'f4',  ('allLambda','iter_d'))[:,i] = data['aod']
            rootgrp.createVariable('aod_mode', 'f4', ('r_dim', 'allLambda','iter_d'))[:,:,i] = data['aodMode']
            rootgrp.createVariable('ssa', 'f4',  ('allLambda','iter_d'))[:,i] = data['ssa']
            rootgrp.createVariable('ssa_mode', 'f4', ('r_dim', 'allLambda','iter_d'))[:,:,i] = data['ssaMode']
            rootgrp.createVariable('n', 'f4', ('r_dim','allLambda','iter_d'))[:,:,i] = data['n']
            rootgrp.createVariable('k', 'f4', ('r_dim','allLambda','iter_d'))[:,:,i] = data['k']
            rootgrp.createVariable('lidar_ratio', 'f4', ('allLambda','iter_d'))[:,i] = data['LidarRatio']
            # rootgrp.createVariable('height', 'f4', ('d_range','d_wl'))[:] = data['height']
            # rootgrp.createVariable('r_eff', 'f4', ('r_dim', 'range_dim'))[:] = data['rEff']
            # rootgrp.createVariable('wtr_surf', 'f4', ('r_dim', 'range_dim'))[:] = data['wtrSurf']
            rootgrp.createVariable('cost_val', 'f4', ('r_dim','iter_d'))[:,i] = data['costVal']
            # rootgrp.createVariable('range_lidar', 'f4', ('d_range','d_wl'))[:] = data['RangeLidar']
            rootgrp.createVariable('meas_dp', 'f4', ('d_range','allLambda','iter_d'))[:,:,i] = data['meas_DP']
            rootgrp.createVariable('fit_dp', 'f4', ('d_range','allLambda','iter_d'))[:,:,i] = data['fit_DP']
            rootgrp.createVariable('meas_vext', 'f4', ('d_range','allLambda','iter_d'))[:,:,i] = data['meas_VExt']
            rootgrp.createVariable('fit_vext', 'f4', ('d_range','allLambda','iter_d'))[:,:,i] = data['fit_VExt']
            rootgrp.createVariable('meas_vbs', 'f4', ('d_range','allLambda','iter_d'))[:,:,i] = data['meas_VBS']
            rootgrp.createVariable('fit_vbs', 'f4', ('d_range','allLambda','iter_d'))[:,:,i] = data['fit_VBS']
            rootgrp.createVariable('sza', 'f4', ('angle','wlRSP','iter_d'))[:,:,i] = data['sza']
            rootgrp.createVariable('vis', 'f4', ('angle','wlRSP','iter_d'))[:,:,i] = data['vis']
            rootgrp.createVariable('fis', 'f4', ('angle','wlRSP','iter_d'))[:,:,i] = data['fis']
            rootgrp.createVariable('sca_ang', 'f4', ('angle','wlRSP','iter_d'))[:,:,i] = data['sca_ang']
            rootgrp.createVariable('meas_i', 'f4', ('angle', 'wlRSP','iter_d'))[:,:,i] = data['meas_I']
            rootgrp.createVariable('fit_i', 'f4', ('angle', 'wlRSP','iter_d'))[:,:,i] = data['fit_I']
            rootgrp.createVariable('meas_p_rel', 'f4', ('angle', 'wlRSP','iter_d'))[:,:,i] = data['meas_P_rel']
            rootgrp.createVariable('fit_p_rel', 'f4', ('angle', 'wlRSP','iter_d'))[:,:,i] = data['fit_P_rel']
    
    # netcdf_file.close()    
    
    
    return


def randomInitGuess(Kernel_type,HSRLfile_path,HSRLfile_name,PixNo, nwl,updateYaml= None,ConsType = None,releaseYAML =True, ModeNo=None,NoItr=None):
    Finalrslts =[]
    krnlPath='/home/shared/GRASP_GSFC/src/retrieval/internal_files'

    if Kernel_type == "sphro":  #If spheroid model

        #Path to the yaml file for sphreroid model
        # fwdModelYAMLpath ='/home/gregmi/git/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/settings_BCK_POLAR_4modes_Shape_ORACLE1.yml'
        if ModeNo == None or ModeNo == 2:
            fwdModelYAMLpath ='/home/gregmi/git/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/settings_BCK_POLARandLIDAR_10Vbins_2modes_ORACLES.yml'
        if ModeNo == 3:
            fwdModelYAMLpath ='/home/gregmi/git/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/settings_BCK_POLAR_3modes_Shape_ORACLE.yml'
        if ModeNo ==4:
            fwdModelYAMLpath = '/home/gregmi/git/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/settings_BCK_POLAR_4modes_Shape_ORACLE1.yml'
        
        if updateYaml == True:  # True if init conditions for Yaml file for HSRL is updated from the GRASP output from RSP
            update_HSRLyaml(fwdModelYAMLpath, rslts_Sph[0], noMod, maxr, minr, a,Kernel_type,ConsType)
            fwdModelYAMLpath ='/home/gregmi/git/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/Settings_Sphd_RSP_HSRL.yaml'
        
        
        # binPathGRASP = path toGRASP Executable for spheriod model
        # binPathGRASP ='/home/shared/GRASP_GSFC/build_HEX_v112/bin/grasp_app'
        # binPathGRASP ='/home/shared/GRASP_GSFC/build_t/bin/grasp_app'
        # binPathGRASP ='/home/shared/GRASP_GSFC/build-tmu/bin/grasp_app' #GRASP Executable
        binPathGRASP ='/home/shared/GRASP_GSFC/build_HEX_v112/bin/grasp_app'
        
        
        info = VariableNoise(fwdModelYAMLpath,nwl)
        
        
        
        # binPathGRASP ='/home/shared/GRASP_GSFC/build_RSP_v112/bin/grasp_app' 
        savePath=f"/home/gregmi/ORACLES/HSRL1_P3_20180922_R03_{Kernel_type}"
    
    if Kernel_type == "TAMU":

        if ModeNo == None or ModeNo == 2:
            fwdModelYAMLpath ='/home/gregmi/git/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/settings_BCK_POLARandLIDAR_10Vbins_2modes_Hex.yml'
        if ModeNo == 3:
            fwdModelYAMLpath ='/home/gregmi/git/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/settings_BCK_POLAR_3modes_Shape_Hex.yml'
        if ModeNo ==4:
            fwdModelYAMLpath = '/home/gregmi/git/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/settings_BCK_POLAR_4modes_Shape_HEX.yml'
        # fwdModelYAMLpath ='/home/gregmi/git/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/settings_BCK_POLAR_3modes_Shape_Hex.yml'
        # fwdModelYAMLpath = '/home/gregmi/git/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/settings_BCK_POLAR_3modes_Shape_ORACLE.yml'
        if updateYaml == True:# True if init conditions for Yaml file for HSRL is updated from the GRASP output from RSP
            update_HSRLyaml(fwdModelYAMLpath, rslts_Tamu[0], noMod, maxr, minr, a,Kernel_type,ConsType)
            fwdModelYAMLpath ='/home/gregmi/git/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/Settings_TAMU_RSP_HSRL.yaml'
        #Path to the GRASP Executable for TAMU
        info = VariableNoise(fwdModelYAMLpath,nwl)
        # binPathGRASP ='/home/shared/GRASP_GSFC/build-tmu/bin/grasp_app' #GRASP Executable
        binPathGRASP ='/home/shared/GRASP_GSFC/build_HEX_v112/bin/grasp_app'
        # binPathGRASP ='/home/shared/GRASP_GSFC/build_HexV112_4Modes/bin/grasp_app'
        #Path to save output plot
        savePath=f"/home/gregmi/ORACLES/RSP1-L1C_P3_20180922_R03_{Kernel_type}"

    rslt_HSRL_1 = Read_Data_HSRL_Oracles_Height(HSRLfile_path,HSRLfile_name,HSRLPixNo)
    rslt=  rslt_HSRL_1[0]

    maxCPU =5
    if NoItr==None: NoItr=1
    
    
    gRuns = [[]]*NoItr
    pix = pixel()
    pix.populateFromRslt(rslt, radianceNoiseFun=None, dataStage= 'meas', verbose=False)
    dblist = []
    
    for i in range(NoItr):
        gyaml = graspYAML(baseYAMLpath=fwdModelYAMLpath)
        yamlObjscramble = gyaml.scrambleInitialGuess(fracOfSpace=1, skipTypes=['vertical_profile_normalized','aerosol_concentration'])
        yamlObj = graspYAML(baseYAMLpath=fwdModelYAMLpath)
        # yamlObj=graspYAML(baseYAMLpath=fwdModelYAMLpath)
        #eventually have to adjust code for height, this works only for one pixel (single height value)
        gRuns = []
        gRuns.append(graspRun(pathYAML=yamlObj, releaseYAML= True )) # This should copy to new YAML object
        gRuns[-1].addPix(pix)

        gDB = graspDB(graspRunObjs=gRuns, maxCPU=maxCPU,maxT =1)
        dblist.append(gDB)#rslts contain all the results form the GRASP inverse run
        rslts, failPix = gDB.processData(binPathGRASP=binPathGRASP, savePath=None, krnlPathGRASP=krnlPath)

        Finalrslts.append(rslts)
    return Finalrslts,gRuns,dblist

def VrtGrad(HSRL_sphrod):

    '''This function calculates and plots the gradient of the vertical profiles to gauge for the noise of the measurement'''
    plt.rcParams['font.size'] = '25'
    fig, ax= plt.subplots(nrows = 3, ncols =3,  figsize= (24,24))
    for i in range (3):
        ax[0,i].plot(np.diff(HSRL_sphrod[0][0]['meas_VExt'][:,i]),HSRL_sphrod[0][0]['RangeLidar'][:,0][1:]/1000 , lw = 3, marker = '.')
        ax[0,i].set_xlabel("VExt, m-1")
        ax[1,i].plot(np.diff(HSRL_sphrod[0][0]['meas_VBS'][:,i]),HSRL_sphrod[0][0]['RangeLidar'][:,0][1:]/1000, lw = 3, marker = '.')
        ax[1,i].set_xlabel("VBS")
        ax[2,i].plot(np.diff(HSRL_sphrod[0][0]['meas_DP'][:,i]),HSRL_sphrod[0][0]['RangeLidar'][:,0][1:]/1000, lw = 3, marker = '.')
        ax[2,i].set_xlabel("DP")

def errPlots(rslts_Sph,rslts_Tamu):

   #This fucntion the error between the measuremtn and the fit from RSP for all wl and all scattering angles.
    wl = rslts_Sph[0]['lambda']
    colorwl = ['#70369d','#4682B4','#01452c','#FF7F7F','#d4af37','#4c0000']
    fig, axs = plt.subplots(nrows= 1, ncols=2, figsize=(16, 9), sharey =True)
    for i in range(len(wl)):
        axs[0].plot(rslts_Sph[0]['sca_ang'][:,i], ErrSph[:,i] ,color = colorwl[i],marker = "$O$",markersize= 10, lw = 0.2,label=f"{np.round(wl[i],2)}")
        axs[1].plot(rslts_Tamu[0]['sca_ang'][:,i], ErrHex[:,i],color = colorwl[i], marker = 'H',markersize= 12, lw = 0.2,label=f"{np.round(wl[i],2)}" )
    axs[0].set_title('Spheriod')
    axs[1].set_title('Hexahedral')

    axs[0].plot(rslts_Sph[0]['sca_ang'][:,0], 100*UNCERT['I']*np.ones(len(ErrSph[:,i])) ,color = 'k',ls = '--', lw = 1,label=f"Meas Uncert")
    axs[1].plot(rslts_Sph[0]['sca_ang'][:,0], 100*UNCERT['I']*np.ones(len(ErrSph[:,i])) ,color = 'k',ls = '--', lw = 1,label=f"Meas Uncert")


    axs[0].set_ylabel('Error I %')
    # axs[1].set_ylabel('Error %')
    plt.legend( ncol=2)

    axs[0].set_xlabel(r'${\theta_s}$')
    axs[1].set_xlabel(r'${\theta_s}$')
    plt.suptitle("RSP-Only Case 1")
    plt.savefig(f'{file_name[2:]}_{RSP_PixNo}_ErrorI.png', dpi = 300)
#Plotting the fits and the retrievals

def plot_HSRL(HSRL_sphrod,HSRL_Tamu, UNCERT, forward = None, retrieval = None, Createpdf = None,PdfName =None, combinedVal = None, key1= None, key2 = None ,RSP_plot =None):


    

    """ HSRL_sphrod = GRASP output array for spheroid 
        HSRL_Tamu = GRASP output array for hexahedral
        forward =  set this to True to plot the measuremnts and  fits
        retrieval = set thid to True yo plot retrieved properties 
        Createpdf = True if you want to create a pdf with plots 
        PdfName = Name/path of the pdf file
        combinedVal = combined string of noises and characteristics from yaml file, this is the output form VariableNoise'
    """

    Hsph,HTam = HSRL_sphrod,HSRL_Tamu
    font_name = "Times New Roman"
    plt.rcParams['font.size'] = '14'


    NoMode = HSRL_sphrod['r'].shape[0]

    pdf_pages = PdfPages(PdfName) 

    if Createpdf == True:
        x = np.arange(10, 1000, 0.1)  #Adjusting the grid of the plot
        y = np.zeros(len(x))
        plt.figure(figsize=(30,15)) 
        #adding text inside the plot
        plt.text(-10, 0.000005,combinedVal , fontsize = 22)
        
        plt.plot(x, y, c='w')
        plt.xlabel("X-axis", fontsize = 15)
        plt.ylabel("Y-axis",fontsize = 15)
        
        
        pdf_pages.savefig()

        Index1 = [0,1,2]
        Index2 = [0,1,2]
        IndexH = [0,3,7]
        IndexRSP = [1,2,4,5,6]

        cm_sp = ['k','#8B4000', '#87C1FF']
        cm_t = ["#BC106F",'#E35335', 'b']
        # color_sph = "#025043"
        color_sph = '#0c7683'
        color_tamu = "#d24787"




        
        if len(Hsph['lambda']) > 3: 
            Index1 = IndexH #Input is Lidar+polarimter

        if len(HTam['lambda']) > 3: 
            Index2 = IndexH 
        
        if RSP_plot !=None:
            Index3 = IndexRSP
            
        if len(HTam['lambda']) !=len(Hsph['lambda']) :
            cm_sp = ['#AA98A9','#CB9129', 'c']
            cm_t = ['#900C3F','#8663bd', '#053F5C']
            color_tamu = '#DE970B'
            color_sph = '#1175A8'



    if forward == True:
        #Converting range to altitude
        altd = (Hsph['RangeLidar'][:,0])/1000 #altitude for spheriod
        altT = (Hsph['RangeLidar'][:,0])/1000 #altitude for hexahedra
        fig, axs= plt.subplots(nrows = 1, ncols =3, figsize= (15,6))  #TODO make it more general which adjust the no of rows based in numebr of wl
        plt.subplots_adjust(top=0.78)
        for i in range(3):
            wave = str(Hsph['lambda'][i]) +"μm"
            axs[i].errorbar(Hsph['meas_VBS'][:,i],altd,xerr= UNCERT['VBS'],color = "#695E93", capsize=3,capthick =1,alpha =0.4, label =f"{UNCERT['VBS']}")
            
            axs[i].plot(Hsph['meas_VBS'][:,Index1[i]],altd, marker =">",color = "#281C2D", label ="Meas")
            axs[i].plot(Hsph['fit_VBS'][:,Index1[i]],altd,color =color_sph, marker = "$O$",label =f"{key1}",alpha =0.8)
            axs[i].plot(HTam['fit_VBS'][:,Index2[i]],altd,color =  color_tamu,ls = "--", label=f"{key2}", marker = "h")

            axs[i].xaxis.set_major_formatter(ScalarFormatter(useMathText=True))

            # print(UNCERT)
            # axs[i].errorbar(Hsph['fit_VBS'][:,i],altd,xerr= UNCERT['VBS'],color = "#025043", marker = "$O$",label ="Sphd")
            # axs[i].errorbar(HTam['fit_VBS'][:,i],altd,xerr= UNCERT['VBS'],color = "#d24787",ls = "--", label="Hex", marker = "h")

            axs[i].set_xlabel(f'$ VBS (m^{-1}Sr^{-1})$',fontproperties=font_name)
            axs[0].set_ylabel('Height above ground (km)',fontproperties=font_name)

            axs[i].set_title(wave)
            if i ==0:
                axs[0].legend()
            plt.suptitle(f"HSRL Vertical Backscatter profile  \nLat,Lon: {HTam['latitude']}, {HTam['longitude']} Date: {HTam['datetime']}\n ",fontproperties=font_name) #Initial condition strictly constrainted by RSP retrievals
        pdf_pages.savefig()
        fig.savefig(f'/home/gregmi/ORACLES/HSRL_RSP/FIT_HSRL Vertical Backscatter profile{NoMode} .png',dpi = 300)
            # plt.tight_layout()
        fig, axs= plt.subplots(nrows = 1, ncols =3, figsize= (15,6))
        plt.subplots_adjust(top=0.78)
        for i in range(3):
            wave = str(Hsph['lambda'][i]) +"μm"

            axs[i].errorbar(Hsph['meas_DP'][:,Index1[i]],altd,xerr= UNCERT['DP'],color = '#695E93', capsize=4,capthick =1,alpha =0.6, label =f"{UNCERT['DP']}%")
            
            axs[i].plot(Hsph['meas_DP'][:,Index1[i]],altd, marker =">",color = "#281C2D", label ="Meas")
            axs[i].plot(Hsph['fit_DP'][:,Index1[i]],altd,color = color_sph, marker = "$O$",label =f"{key1}")
            axs[i].plot(HTam['fit_DP'][:,Index2[i]],altd,color = color_tamu, ls = "--",marker = "h",label=f"{key2}")
            
            # axs[i].errorbar(Hsph['fit_DP'][:,i],altd,xerr= UNCERT['DP'],fmt='-o',color = "#025043", marker = "$O$",label ="Sphd")
            # axs[i].errorbar(HTam['fit_DP'][:,i],altd,xerr= UNCERT['DP'],fmt='-o',color = "#d24787",ls = "--", label="Hex", marker = "h")


            axs[i].set_xlabel('DP %')
            axs[i].set_title(wave)
            if i ==0:
                axs[0].legend()
            axs[0].set_ylabel('Height above ground (km)',fontproperties=font_name)
            plt.suptitle(f"HSRL Depolarization Ratio \n Lat,Lon: {HTam['latitude']}, {HTam['longitude']}  Date: {HTam['datetime']}\n ",fontproperties=font_name) #Initial condition strictly constrainted by RSP retrievals
        pdf_pages.savefig()
        fig.savefig(f'/home/gregmi/ORACLES/HSRL_RSP/FIT_HSRL Depolarization Ratio{NoMode}.png',dpi = 300)
        fig, axs= plt.subplots(nrows = 1, ncols =2, figsize= (11,6))
        plt.subplots_adjust(top=0.78)
        for i in range(2):
            wave = str(Hsph['lambda'][i]) +"μm"

            axs[i].errorbar(Hsph['meas_VExt'][:,Index1[i]],altd,xerr= UNCERT['VEXT']*Hsph['meas_VExt'][:,i],color = '#695E93',capsize=4,capthick =1,alpha =0.7, label =f"{UNCERT['VEXT']*100}%")
            
            axs[i].plot(Hsph['meas_VExt'][:,Index1[i]],altd, marker =">",color = "#281C2D", label ="Meas")
            axs[i].plot(Hsph['fit_VExt'][:,Index1[i]],altd,color = color_sph, marker = "$O$",label =f"{key1}")
            axs[i].plot(HTam['fit_VExt'][:,Index2[i]],altd,color = color_tamu,ls = "--", marker = "h",label=f"{key2}")
            axs[i].xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
            # axs[i].errorbar(Hsph['fit_VExt'][:,i],altd,xerr= UNCERT['VEXT'],color = "#025043", marker = "$O$",label ="Sphd")
            # axs[i].errorbar(HTam['fit_VExt'][:,i],altd,xerr= UNCERT['VEXT'],color = "#d24787",ls = "--", label="Hex", marker = "h")

            
            
            axs[i].set_xlabel(f'$VExt (m^{-1})$',fontproperties=font_name)
            axs[0].set_ylabel('Height above ground (km)',fontproperties=font_name)
            axs[i].set_title(wave)
            if i ==0:
                axs[0].legend()
            plt.suptitle(f"HSRL Vertical Extinction profile\n Lat,Lon: {HTam['latitude']},{HTam['longitude']}  Date: {HTam['datetime']}\n ",fontproperties=font_name) #Initial condition strictly constrainted by RSP retrievals
        plt.tight_layout()
        pdf_pages.savefig()
        fig.savefig(f'/home/gregmi/ORACLES/HSRL_RSP/FIT_HSRL_Vertical_Ext_profile{NoMode}.png',dpi = 300)

    if retrieval == True:
        plt.rcParams['font.size'] = '35'
        
        Spheriod,Hex = HSRL_sphrod,HSRL_Tamu
        # plt.rcParams['font.size'] = '17'
        #Stokes Vectors Plot
        date_latlon = ['datetime', 'longitude', 'latitude']
        Xaxis = ['r','lambda','sca_ang','rv','height']
        Retrival = ['dVdlnr','aodMode','ssaMode','n', 'k']
        #['sigma', 'vol', 'aodMode','ssaMode', 'rEff', 'costVal']
        Angles =   ['sza', 'vis', 'fis','angle' ]
        Stokes =   ['meas_I', 'fit_I', 'meas_P_rel', 'fit_P_rel']
        Pij    = ['p11', 'p12', 'p22', 'p33'], 
        Lidar=  ['heightStd','g','LidarRatio','LidarDepol', 'gMode', 'LidarRatioMode', 'LidarDepolMode']

        # Plot the AOD data
        

        # Plot the AOD data
        y = [0,1,2,0,1,2,]
        x = np.repeat((0,1),3)
        if Spheriod['r'].shape[0] ==2 :
            mode_v = ["fine", "dust","marine", 'NonSphMarine']
        if Spheriod['r'].shape[0] ==3 :
            mode_v = ["fine", "dust","marine", 'NonSphMarine']
        linestyle =[':', '-','-.','-']
        mode_v = ["fine", "dust","marine", 'NonSphMarine']

        


        if key1== None:
            key1 ='Sphroid'
        if key2 == None:
            key2 = 'Hex'

        #Retrivals:
        fig, axs2 = plt.subplots(nrows= 3, ncols=2, figsize=(38, 20))
        for i in range(len(Retrival)):
            a,b = i%3,i%2

            lambda_ticks1 = np.round(Spheriod['lambda'], decimals=2)
            lambda_ticks2 = np.round(Hex['lambda'], decimals=2)

            lambda_ticks_str1 = [str(x) for x in lambda_ticks1]
            lambda_ticks_str2 = [str(x) for x in lambda_ticks2]
            
            for mode in range(Spheriod['r'].shape[0]): #for each modes
                if i ==0:

                    # axs2[a,b].errorbar(Spheriod['r'][mode], Spheriod[Retrival[i]][mode],xerr=UNCERT['rv'],capsize=5,capthick =2, marker = "$O$",color = cm_sp[mode],lw = 3,ls = linestyle[mode], label=f"Sphrod_{mode_v[mode]}")
                    # axs2[a,b].errorbar(Hex['r'][mode],Hex[Retrival[i]][mode], marker = "H",xerr=UNCERT['rv'],capsize=5,capthick =2, color = cm_t[mode] ,lw = 3, ls = linestyle[mode],label=f"Hex_{mode_v[mode]}")
                        
                    axs2[a,b].plot(Spheriod['r'][mode], Spheriod[Retrival[i]][mode], marker = "$O$",color = cm_sp[mode],lw = 5.5,ls = linestyle[mode], label=f"{key1}_{mode_v[mode]}")
                    axs2[a,b].plot(Hex['r'][mode],Hex[Retrival[i]][mode], marker = "H", color = cm_t[mode] ,lw = 5.5, ls = linestyle[mode],label=f"{key2}_{mode_v[mode]}")
                    # if RSP_plot != None:
                    #     axs2[a,b].plot(RSP_plot['r'][mode],RSP_plot[Retrival[i]][mode], marker = "H", color = cm_t[mode] ,lw = 5.5, ls = linestyle[mode],label=f"{key2}_{mode_v[mode]}")
                    

                    axs2[0,0].set_xlabel(r'rv $ \mu m$')
                    axs2[0,0].set_xscale("log")
                else:

                    axs2[a,b].errorbar(lambda_ticks1, Spheriod[Retrival[i]][mode],yerr=UNCERT[Retrival[i]], marker = "$O$",markeredgecolor='k',capsize=7,capthick =2,markersize=25, color = cm_sp[mode],lw = 6,ls = linestyle[mode], label=f"{key1}_{mode_v[mode]}")
                    axs2[a,b].errorbar(lambda_ticks2,Hex[Retrival[i]][mode],yerr=UNCERT[Retrival[i]],capsize=7,capthick =2,  marker = "H",markeredgecolor='k',markersize=25, color = cm_t[mode] ,lw = 6, ls = linestyle[mode],label=f"{key2}_{mode_v[mode]}")
                    
                    # axs[a,b].plot(Spheriod['lambda'], Spheriod[Retrival[i]][mode], marker = "$O$",color = cm_sp[mode],lw = 2,ls = linestyle[mode],markersize=15, label=f"Sphrod_{mode_v[mode]}")
                    # axs[a,b].plot(Hex['lambda'],Hex[Retrival[i]][mode], marker = "H",color = cm_t[mode] ,lw = 2,  ls = linestyle[mode],markersize=15, label=f"Hex_{mode_v[mode]}")
                    # lambda_ticks = np.round(Spheriod['lambda'], decimals=2)

                    
                    # lambda_ticks = np.round(Spheriod['lambda'], decimals=2)
                    # axs2[a,b].set_xticks(lambda_ticks_str2)
                    
                    # axs2[a,b].set_xticks(Spheriod['lambda'])
                    # axs[a,b].set_xticklabels(['0.41', '0.46', '0.55' , '0.67'  , '0.86'])
                    axs2[a,b].set_xlabel(r'$\lambda \mu m$')
                    # axs2[a,b].set_ylim(bottom=0)
            
            axs2[a,b].set_ylabel(f'{Retrival[i]}')
            
            
        # axs[2,1].plot(Spheriod['lambda'], Spheriod['aod'], marker = "$O$",color = color_sph,markersize=15,lw = 2, label=f"Sphroid")
        # axs[2,1].plot(Hex['lambda'], Hex['aod'], marker = "H", color = color_tamu ,markersize=15,lw = 2, label=f"Hexahedral")
        
        axs2[2,1].errorbar(lambda_ticks1, Spheriod['aod'],yerr=0.03+UNCERT['aod']*Spheriod['aod'], marker = "$O$",color = color_sph,markeredgecolor='k',capsize=7,capthick =2,markersize=25,lw = 4, label=f"{key1}")
        axs2[2,1].errorbar(lambda_ticks2, Hex['aod'],yerr=0.03+UNCERT['aod']* Hex['aod'], marker = "H", color = color_tamu ,lw = 4,markeredgecolor='k',capsize=7,capthick =2,markersize=25, label=f"{key2}")
        # axs2[2,1].set_xticks(lambda_ticks_str)  
        # axs[2,1].set_xticklabels(Spheriod['lambda'],rotation=45)
        # axs[2,1].set_xticklabels(['0.41', '0.46', '0.55' , '0.67'  , '0.86'])
        # meas_aod = []
        # for i in range (NoMode):
        #     meas_aod.append(np.trapz(Hsph['meas_VExt'][:,i][::-1],Hsph['range'][i,:][::-1] ))
        # meas_aodsph = []
        # for i in range (NoMode):
        #     meas_aodsph.append(np.trapz(Hsph['fit_VExt'][:,i][::-1],Hsph['range'][i,:][::-1] ))
        # meas_aodhex = []
        # for i in range (NoMode):
        #     meas_aodhex.append(np.trapz(HTam['fit_VExt'][:,i][::-1],HTam['range'][i,:][::-1] ))
        
        # axs[2,1].plot( Spheriod['lambda'],meas_aod,color = "k", ls = "--", marker = '*' ,markersize=20, label = " cal meas")    
        # axs[2,1].plot( Spheriod['lambda'],meas_aodsph,color = "r", ls = "-.", marker = '*' ,markersize=20, label = "cal sph")    
        # axs[2,1].plot( Spheriod['lambda'],meas_aod,color = "b", ls = "-.", marker = '*' ,markersize=20, label = "cal hex")    
       
        
        axs2[2,1].set_xlabel(r'$\lambda$')
        axs2[2,1].set_ylabel('Total AOD')
        axs2[0,0].legend(prop = { "size": 22 }, ncol=2)
        axs2[2,1].legend( ncol=2)
        lat_t = Hex['latitude']
        lon_t = Hex['longitude']
        dt_t = Hex['datetime']
        chisph,chihex = Spheriod['costVal'] ,Hex['costVal']

        if len(lambda_ticks_str2) <4:
            plt.suptitle(f" HSRL-2 Aerosol Retrieval \n  {lat_t}N,{lon_t}E  {dt_t}\n")
        if len(lambda_ticks_str2) >3:
            plt.suptitle(f" HSRL-2+ RSP Aerosol Retrieval \n  {lat_t}N,{lon_t}E  {dt_t}\n")
        
                          
        plt.subplots_adjust(top=0.99)
        plt.tight_layout()
  
        
        pdf_pages.savefig()
        pdf_pages.close()
        fig.savefig(f'/home/gregmi/ORACLES/HSRL_RSP/{dt_t}{NoMode}{key1}_{key2}_HSRL2Retrieval.png', dpi = 400)
        plt.rcParams['font.size'] = '15'
        fig, axs = plt.subplots(nrows= 1, ncols=2, figsize=(10, 5), sharey=True)
        
        for i in range(Spheriod['r'].shape[0]):
            #Hsph,HTam
            axs[0].plot(Hsph['βext'][i],Hsph['range'][i]/1000, label =i+1)
            axs[1].plot(HTam['βext'][i],HTam['range'][i]/1000, label =i+1)
            axs[1].set_xlabel('βext')
            axs[0].set_title('Spheriod')
            axs[0].set_xlabel('βext')
            axs[0].set_ylabel('Height km')
            axs[1].set_title('Hexahedral')
        plt.legend()
        plt.tight_layout()

    return




def CombinedLidarPolPlot(LidarPolSph,LidarPolTAMU,RSP_PixNo, UNCERT): #should be updated to 
    plt.rcParams['font.size'] = '14'
    fig, axs= plt.subplots(nrows = 1, ncols =3, figsize= (15,6))

    altd =LidarPolTAMU[0]['RangeLidar'][:,0]/1000
    HTam = LidarPolTAMU[0]
    plt.subplots_adjust(top=0.78)
    IndexH = [0,3,7]
    for i in range(3):

        wave = str(LidarPolSph[0]['lambda'][IndexH[i]]) +"μm"

        axs[i].plot(LidarPolSph[0]['meas_VBS'][:,IndexH[i]],(LidarPolSph[0]['RangeLidar'][:,0])/1000, marker =">",color = "#3B270C", label ="Meas")
        axs[i].errorbar(LidarPolSph[0]['meas_VBS'][:,IndexH[i]],(LidarPolSph[0]['RangeLidar'][:,0])/1000,xerr= UNCERT['VBS'],color = "#695E93", capsize=3,capthick =1,alpha =0.4, label =f"{UNCERT['VBS']}")
        axs[i].plot(LidarPolSph[0]['fit_VBS'][:,IndexH[i]],(LidarPolSph[0]['RangeLidar'][:,0])/1000, marker = "$O$",color = "#025043",label ="Sphd")
        axs[i].plot(LidarPolTAMU[0]['fit_VBS'][:,IndexH[i]],altd,color = "#d24787",ls = "--", label="Hex",marker = "h")

        axs[i].set_xlabel('VBS')
        axs[i].set_title(wave)
        if i ==0:
            axs[0].legend()
        plt.suptitle(f"Vertical Backscatter profile Fits(HSRL2+RSP)  \n {LidarPolTAMU[0]['latitude']}N,{LidarPolTAMU[0]['longitude']}E {HTam['datetime']} ") #Initial condition strictly constrainted by RSP retrievals
    fig.savefig(f'/home/gregmi/ORACLES/HSRL_RSP/FIT_Combined_DP.png',dpi = 300)

    fig, axs= plt.subplots(nrows = 1, ncols =3, figsize= (15,6))
    plt.subplots_adjust(top=0.78)
    for i in range(3):
        wave = str(LidarPolSph[0]['lambda'][IndexH[i]]) +"μm"

        axs[i].errorbar(LidarPolSph[0]['meas_DP'][:,IndexH[i]],(LidarPolSph[0]['RangeLidar'][:,0])/1000, marker =">",xerr= UNCERT['DP'],color = "#695E93", capsize=3,capthick =1,alpha =0.4, label =f"{UNCERT['DP']}%")
        

        axs[i].plot(LidarPolSph[0]['meas_DP'][:,IndexH[i]],(LidarPolSph[0]['RangeLidar'][:,0])/1000, marker =">",color = "#3B270C", label ="Meas")
        axs[i].plot(LidarPolSph[0]['fit_DP'][:,IndexH[i]],(LidarPolSph[0]['RangeLidar'][:,0])/1000,color = "#025043", marker = "$O$",label ="Sphd")
        axs[i].plot(LidarPolTAMU[0]['fit_DP'][:,IndexH[i]],altd,color = "#d24787", ls = "--",marker = "h", label="Hex")
        axs[i].set_xlabel('DP %')
        axs[i].set_title(wave)
        if i ==0:
            axs[0].legend()
        plt.suptitle(f" Depolarization Ratio Profile Fit(HSR2L+RSP) \n {LidarPolTAMU[0]['latitude']}N,{LidarPolTAMU[0]['longitude']}E {HTam['datetime']}") #Initial condition strictly constrainted by RSP retrievals
    fig.savefig(f'/home/gregmi/ORACLES/HSRL_RSP/FIT_Combined_VBS.png',dpi = 300)

    fig, axs= plt.subplots(nrows = 1, ncols =2, figsize= (9,6))
    plt.subplots_adjust(top=0.78)
    for i in range(2):
        wave = str(LidarPolSph[0]['lambda'][IndexH[i]]) +"μm"

        axs[i].errorbar(LidarPolSph[0]['meas_VExt'][:,IndexH[i]],(LidarPolSph[0]['RangeLidar'][:,0])/1000, marker =">",xerr= LidarPolSph[0]['meas_VExt'][:,IndexH[i]]*UNCERT['VEXT'],color = "#695E93", capsize=3,capthick =1,alpha =0.4, label =f"{UNCERT['VEXT']*100}%")
        
        axs[i].plot(LidarPolSph[0]['meas_VExt'][:,IndexH[i]],(LidarPolSph[0]['RangeLidar'][:,0])/1000, marker =">",color = "#3B270C", label ="Meas")
        axs[i].plot(LidarPolSph[0]['fit_VExt'][:,IndexH[i]],(LidarPolSph[0]['RangeLidar'][:,0])/1000,color = "#025043", marker = "$O$",label ="Sphd")
        axs[i].plot(LidarPolTAMU[0]['fit_VExt'][:,IndexH[i]],altd,color = "#d24787",ls = "--", marker = "h", label="Hex")
        axs[i].set_xlabel('VExt')
        axs[i].set_title(wave)
        if i ==0:
            axs[0].legend()
        plt.suptitle(f"Vertical Extinction Profile Fit (HSR2L+RSP) \n {LidarPolTAMU[0]['latitude']}N,{LidarPolTAMU[0]['longitude']}E {HTam['datetime']}\n") #Initial condition strictly constrainted by RSP retrievals
    
    
    fig.savefig(f'/home/gregmi/ORACLES/HSRL_RSP/FIT_Combined_Vertical_EXT_profile.png',dpi = 300)
    fig, axs= plt.subplots(nrows = 1, ncols =2, figsize= (10,5))
    for i in [1,2,4,5,6]:

        
        # axs[0].errorbar(LidarPolSph[0]['sca_ang'][:,i], LidarPolSph[0]['meas_I'][:,i],yerr=  LidarPolSph[0]['meas_I'][:,i]*UNCERT['I'],color = "#695E93", capsize=3,capthick =1,alpha =0.4, label =f"{UNCERT['I']}")
        axs[0].fill_between(LidarPolSph[0]['sca_ang'][:,i], LidarPolSph[0]['meas_I'][:,i], LidarPolSph[0]['meas_I'][:,i]*1.03, color = 'r',alpha=0.2, ls = "--",label="+3%")
        axs[0].fill_between(LidarPolSph[0]['sca_ang'][:,i], LidarPolSph[0]['meas_I'][:,i], LidarPolSph[0]['meas_I'][:,i]*0.97, color = 'b',alpha=0.2, ls = "--",label="-3%")
           
        axs[0].plot(LidarPolSph[0]['sca_ang'][:,i], LidarPolSph[0]['meas_I'][:,i],color = "#3B270C",lw= 2, label ="Meas")
        axs[0].plot(LidarPolSph[0]['sca_ang'][:,i],LidarPolSph[0]['fit_I'][:,i],color = "#025043",lw= 2, label ="Sphd")
        axs[0].plot(LidarPolSph[0]['sca_ang'][:,i],LidarPolTAMU[0]['fit_I'][:,i],color = "#d24787",lw= 2,ls = "--", label="Hex")
        axs[0].set_ylabel('I')
        axs[0].set_xlabel('Scattering angles')
        
        plt.suptitle(f"HSR2L+RSP Retrieval: I and DOLP fit at {LidarPolSph[0]['lambda'][i]}μm , \n {LidarPolTAMU[0]['latitude']}N,{LidarPolTAMU[0]['longitude']}E {HTam['datetime']} ") #Initial condition strictly constrainted by RSP retrievals
        axs[1].fill_between(LidarPolSph[0]['sca_ang'][:,i], LidarPolSph[0]['meas_P_rel'][:,i], LidarPolSph[0]['meas_P_rel'][:,i]+0.005, color = 'r',alpha=0.2, ls = "--",label="+3%")
        axs[1].fill_between(LidarPolSph[0]['sca_ang'][:,i], LidarPolSph[0]['meas_P_rel'][:,i], LidarPolSph[0]['meas_P_rel'][:,i]-0.005, color = 'b',alpha=0.2, ls = "--",label="-3%")
        
        axs[1].plot(LidarPolSph[0]['sca_ang'][:,i],LidarPolSph[0]['meas_P_rel'][:,i],color = "#3B270C",lw= 2, label ="Meas")
        axs[1].plot(LidarPolSph[0]['sca_ang'][:,i],LidarPolSph[0]['fit_P_rel'][:,i],color = "#025043",lw= 2,label ="Sphd")
        axs[1].plot(LidarPolSph[0]['sca_ang'][:,i],LidarPolTAMU[0]['fit_P_rel'][:,i],color = "#d24787",lw= 2,ls = "--", label="Hex")
        axs[1].set_ylabel('P/I')
        axs[1].set_xlabel('Scattering angles')
        axs[0].legend()
        # axs[1].set_title()
        fig.savefig(f'/home/gregmi/ORACLES/HSRL_RSP/FIT_{RSP_PixNo}_Combined_I_P_{i}.png',dpi = 300)

        fig, axs= plt.subplots(nrows = 1, ncols =2, figsize= (10,5))
       
def RSP_plot(rslts_Sph,rslts_Tamu,RSP_PixNo,UNCERT,rslts_Sph2=None,rslts_Tamu2=None, LIDARPOL= None, fn = "None",ModeComp = None): 
    
    if (rslts_Sph2!=None )and (rslts_Tamu2!=None):
        RepMode =2
    else:
        RepMode =1
    Spheriod,Hex = rslts_Sph[0],rslts_Tamu[0]

    cm_sp = ['k','#8B4000', '#87C1FF']
    cm_t = ["#BC106F",'#E35335', 'b']
    color_sph = '#0c7683'
    color_tamu = "#BC106F"

    fig, axs2 = plt.subplots(nrows= 3, ncols=2, figsize=(35, 20))
        


    for i in range(RepMode):
        if i ==1:
            Spheriod,Hex = rslts_Sph2[0],rslts_Tamu2[0]
            cm_sp = ['#b54cb5','#14411b', '#87C1FF']
            cm_t = ["#14411b",'#adbf4b', 'b']
            color_sph = '#adbf4b'
            color_tamu = "#936ecf"


        
        
        plt.rcParams['font.size'] = '26'
        plt.rcParams["figure.autolayout"] = True
        #Stokes Vectors Plot
        date_latlon = ['datetime', 'longitude', 'latitude']
        Xaxis = ['r','lambda','sca_ang','rv','height']
        Retrival = ['dVdlnr','aodMode','ssaMode','n', 'k']
        #['sigma', 'vol', 'aodMode','ssaMode', 'rEff', 'costVal']
        Angles =   ['sza', 'vis', 'fis','angle' ]
        Stokes =   ['meas_I', 'fit_I', 'meas_P_rel', 'fit_P_rel']
        Pij    = ['p11', 'p12', 'p22', 'p33'], 
        Lidar=  ['heightStd','g','LidarRatio','LidarDepol', 'gMode', 'LidarRatioMode', 'LidarDepolMode']

        # Plot the AOD data
        y = [0,1,2,0,1,2,]
        x = np.repeat((0,1),3)
        NoMode =Spheriod['r'].shape[0]
        if Spheriod['r'].shape[0] ==2 :
                mode_v = ["fine", "dust","marine"]
        if Spheriod['r'].shape[0] ==3 :
            mode_v = ["fine", "dust","marine"]
        linestyle =[':', '-','-.']
        

        lambda_ticks = np.round(Spheriod['lambda'], decimals=2)
        lambda_ticks_str = [str(x) for x in lambda_ticks]

        
        
        #Retrivals:
        for i in range(len(Retrival)):
            a,b = i%3,i%2
            for mode in range(Spheriod['r'].shape[0]): #for each modes
                if i ==0:

                    # axs2[a,b].errorbar(Spheriod['r'][mode], Spheriod[Retrival[i]][mode],xerr=UNCERT['rv'],capsize=5,capthick =2, marker = "$O$",color = cm_sp[mode],lw = 3,ls = linestyle[mode], label=f"Sphrod_{mode_v[mode]}")
                    # axs2[a,b].errorbar(Hex['r'][mode],Hex[Retrival[i]][mode], marker = "H",xerr=UNCERT['rv'],capsize=5,capthick =2, color = cm_t[mode] ,lw = 3, ls = linestyle[mode],label=f"Hex_{mode_v[mode]}")
                        
                    axs2[a,b].plot(Spheriod['r'][mode], Spheriod[Retrival[i]][mode], marker = "$O$",color = cm_sp[mode],lw = 2,ls = linestyle[mode], label=f"Sphrod_{mode_v[mode]}")
                    axs2[a,b].plot(Hex['r'][mode],Hex[Retrival[i]][mode], marker = "H", color = cm_t[mode] ,lw = 2, ls = linestyle[mode],label=f"Hex_{mode_v[mode]}")
                    
                    axs2[0,0].set_xlabel(r'rv $ \mu m$')
                    axs2[0,0].set_xscale("log")
                else:

                    axs2[a,b].errorbar(lambda_ticks_str, Spheriod[Retrival[i]][mode],yerr=UNCERT[Retrival[i]], marker = "$O$",markeredgecolor='k',capsize=7,capthick =2,markersize=25, color = cm_sp[mode],lw = 4,ls = linestyle[mode], label=f"Sphrod_{mode_v[mode]}")
                    axs2[a,b].errorbar(lambda_ticks_str,Hex[Retrival[i]][mode],yerr=UNCERT[Retrival[i]],capsize=7,capthick =2,  marker = "H",markeredgecolor='k',markersize=25, color = cm_t[mode] ,lw = 4, ls = linestyle[mode],label=f"Hex_{mode_v[mode]}")
                    
                    # axs[a,b].plot(Spheriod['lambda'], Spheriod[Retrival[i]][mode], marker = "$O$",color = cm_sp[mode],lw = 2,ls = linestyle[mode],markersize=15, label=f"Sphrod_{mode_v[mode]}")
                    # axs[a,b].plot(Hex['lambda'],Hex[Retrival[i]][mode], marker = "H",color = cm_t[mode] ,lw = 2,  ls = linestyle[mode],markersize=15, label=f"Hex_{mode_v[mode]}")
                    
                    axs2[a,b].set_xticks(lambda_ticks_str) 
                    # axs[a,b].set_xticklabels(['0.41', '0.46', '0.55' , '0.67'  , '0.86'])
                    axs2[a,b].set_xlabel(r'$\lambda \mu m$')
                    # axs2[a,b].set_xlim(0, np.maximum(Spheriod[Retrival[i]][mode],Hex[Retrival[i]][mode]))
            axs2[a,b].set_ylabel(f'{Retrival[i]}')
            
            
        # axs[2,1].plot(Spheriod['lambda'], Spheriod['aod'], marker = "$O$",color = color_sph,markersize=15,lw = 2, label=f"Sphroid")
        # axs[2,1].plot(Hex['lambda'], Hex['aod'], marker = "H", color = color_tamu ,markersize=15,lw = 2, label=f"Hexahedral")
        
        axs2[2,1].errorbar(lambda_ticks_str, Spheriod['aod'],yerr=0.03+UNCERT['aod']*Spheriod['aod'], marker = "$O$",color = color_sph,markeredgecolor='k',capsize=7,capthick =2,markersize=25,lw = 4, label=f"Sphroid")
        axs2[2,1].errorbar(lambda_ticks_str, Hex['aod'],yerr=0.03+UNCERT['aod']* Hex['aod'], marker = "H", color = color_tamu ,lw = 4,markeredgecolor='k',capsize=7,capthick =2,markersize=25, label=f"Hex")
            
        
        axs2[2,1].set_xticks(lambda_ticks_str)
        # axs[2,1].set_xticklabels(['0.41', '0.46', '0.55' , '0.67'  , '0.86'])
                
        axs2[2,1].set_xlabel(r'$\lambda$')
        axs2[2,1].set_ylabel('Total AOD')
        axs2[0,0].legend(prop = { "size": 21 }, ncol=2)
        axs2[2,1].legend()
        lat_t = Hex['latitude']
        lon_t = Hex['longitude']
        dt_t = Hex['datetime']
        plt.tight_layout(rect=[0, 0, 1, 1])
        # axs2.savefig(f'/home/gregmi/ORACLES/HSRL_RSP/{dt_t}{RepMode}_RSPRetrieval{RepMode}.png', dpi = 400)
        
        plt.suptitle(f'RSP Aerosol Retrieval \n  Lat:{lat_t} Lon :{lon_t}   Date: {dt_t}')
        if RepMode ==2:
            fig.savefig(f'/home/gregmi/ORACLES/HSRL_RSP/{dt_t}2+3_RSPRetrieval.png', dpi = 400)
        else:
            fig.savefig(f'/home/gregmi/ORACLES/HSRL_RSP/{dt_t}{NoMode}_RSPRetrieval.png', dpi = 400)

        #Stokes: 
        wl = rslts_Sph[0]['lambda'] 
        fig, axs = plt.subplots(nrows= 4, ncols=5, figsize=(35, 17),gridspec_kw={'height_ratios': [1, 0.3,1,0.3]}, sharex='col')
        # Plot the AOD data
        meas_P_rel = 'meas_P_rel'

        if LIDARPOL == True:
            wlIdx = [1,2,4,5,6]  #Lidar wavelngth indices
        else:
            wlIdx = np.arange(len(wl))


        for nwav in range(len(wlIdx)):
        # Plot the fit and measured I data
            
            axs[0, nwav].fill_between(Spheriod ['sca_ang'][:,wlIdx[nwav]],Spheriod ['meas_I'][:,wlIdx[nwav]], Spheriod ['meas_I'][:,wlIdx[nwav]]*1.03, color = 'r',alpha=0.2, ls = "--",label="+3%")
            axs[0, nwav].fill_between(Spheriod ['sca_ang'][:,wlIdx[nwav]],Spheriod ['meas_I'][:,wlIdx[nwav]], Spheriod ['meas_I'][:,wlIdx[nwav]]*0.97, color = "b",alpha=0.2, ls = "--",label="-3%")
            axs[0, nwav].plot(Spheriod['sca_ang'][:,wlIdx[nwav]], Spheriod['meas_I'][:,wlIdx[nwav]], color = "k", lw = 3, label="meas")

            axs[0, nwav].plot(Spheriod ['sca_ang'][:,wlIdx[nwav]], Spheriod ['fit_I'][:,wlIdx[nwav]],color =color_sph , lw = 3, ls = '--',label="fit sphrod")
            # axs[0, nwav].scatter(Spheriod ['sca_ang'][:,nwav][marker_indsp], Spheriod ['fit_I'][:,nwav][marker_indsp],color =color_sph , m = "o",label="fit sphrod")
            
            # axs[0, nwav].set_xlabel('Scattering angles (deg)')
            axs[0, 0].set_ylabel('I')
            # axs[0, nwav].legend()

            # Plot the fit and measured QoI data
            axs[2, nwav].plot(Spheriod ['sca_ang'][:,wlIdx[nwav]], Spheriod ['meas_P_rel'][:,wlIdx[nwav]],color = "k", lw = 3, label="meas")
            axs[2, nwav].plot(Spheriod ['sca_ang'][:,wlIdx[nwav]], Spheriod ['fit_P_rel'][:,wlIdx[nwav]], color =color_sph, lw = 3, ls = '--', label="fit sph")
            
            axs[2, nwav].fill_between(Spheriod ['sca_ang'][:,wlIdx[nwav]],(Spheriod ['meas_P_rel'][:,wlIdx[nwav]]), (Spheriod ['meas_P_rel'][:,wlIdx[nwav]])*1.03,color = 'r', alpha=0.2,ls = "--", label="+3%")
            axs[2, nwav].fill_between(Spheriod ['sca_ang'][:,wlIdx[nwav]],(Spheriod ['meas_P_rel'][:,wlIdx[nwav]]), (Spheriod ['meas_P_rel'][:,wlIdx[nwav]])*0.97,color = "b", alpha=0.2,ls = "--", label="-3%")
            # axs[2, nwav].set_xlabel('Scattering angles (deg)')
            axs[2, 0].set_ylabel('DOLP')
            axs[0, nwav].set_title(f"{wl[wlIdx[nwav]]} $\mu m$", fontsize = 22)
            
            axs[0, nwav].plot(Hex['sca_ang'][:,wlIdx[nwav]], Hex['fit_I'][:,wlIdx[nwav]],color =color_tamu , lw = 3, ls = "dashdot",label="fit Hex")
            axs[2, nwav].plot(Hex['sca_ang'][:,wlIdx[nwav]],Hex['fit_P_rel'][:,wlIdx[nwav]],color = color_tamu , lw = 3,ls = "dashdot", label = "fit Hex") 


            sphErr = 100 * abs(Spheriod['meas_I'][:,wlIdx[nwav]]-Spheriod ['fit_I'][:,wlIdx[nwav]] )/Spheriod['meas_I'][:,wlIdx[nwav]]
            HexErr = 100 * abs(Hex['meas_I'][:,nwav]-Hex['fit_I'][:,nwav] )/Hex['meas_I'][:,wlIdx[nwav]]
            
            axs[1, nwav].plot(Spheriod ['sca_ang'][:,wlIdx[nwav]], sphErr,color =color_sph , lw = 3,label="Sphrod")
            axs[1, nwav].plot(Hex ['sca_ang'][:,wlIdx[nwav]], HexErr,color = color_tamu, lw = 3 ,label="Hex")
            axs[1, 0].set_ylabel('Err I %')
            
    #Absolute error
            sphErrP =  abs(Spheriod['meas_P_rel'][:,wlIdx[nwav]]-Spheriod ['fit_P_rel'][:,wlIdx[nwav]])
            HexErrP =  abs(Hex['meas_P_rel'][:,wlIdx[nwav]]-Hex['fit_P_rel'][:,wlIdx[nwav]] )
            
            axs[3, nwav].plot(Spheriod ['sca_ang'][:,wlIdx[nwav]], sphErrP,color =color_sph , lw = 3,label="Sphrod")
            axs[3, nwav].plot(Hex ['sca_ang'][:,wlIdx[nwav]], HexErrP,color =color_tamu, lw = 3 ,label="Hex")
            
            axs[3, nwav].set_xlabel(r'$\theta$s(deg)')
            # axs[3, nwav].set_ylabel('Err P')
            # axs[3, nwav].legend()
            axs[3, nwav].set_xlabel(r'$\theta$s(deg)')
            axs[3, 0].set_ylabel('|Meas-fit|')
        
            # axs[1, nwav].set_title(f"{wl[nwav]}", fontsize = 14)

            axs[0, 4].legend(bbox_to_anchor=(1.05, 0.5), loc='center left')
            # axs[1, 4].legend(bbox_to_anchor=(1.05, 0.5), loc='center left')
            plt.suptitle(f'RSP Aerosol Retrieval \n  Lat:{lat_t} Lon :{lon_t}   Date: {dt_t}')
            if fn ==None : fn = "1"
            fig.savefig(f'/home/gregmi/ORACLES/HSRL_RSP/{dt_t}_RSPFits_{fn}.png', dpi = 400)

            plt.tight_layout(rect=[0, 0, 1, 1])



def Ext2Vconc(botLayer,topLayer,Nlayers,wl):

    dict_of_dicts = {}


#To simulate the coefficient to convert Vext to vconc
 
    botLayer = 2000
    topLayer = 4000
    Nlayers = 3

    VConcandExt = np.zeros((len(Volume),2))*np.nan


    krnlPath='/home/shared/GRASP_GSFC/src/retrieval/internal_files'
    fwdModelYAMLpath ='/home/gregmi/git/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/settings_dust_Vext_conc.yml'
    binPathGRASP ='/home/shared/GRASP_GSFC/build_HEX_v112/bin/grasp_app'
    ymlPath = '/home/gregmi/git/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/'
    UpKerFile =  'settings_dust_Vext_conc_dump.yml'

    singProf = np.linspace(botLayer, topLayer, Nlayers)[::-1]
    #Inputs for simulation
    wvls = [0.532] # wavelengths in μm
    msTyp = [36] # meas type VEXT
    sza = 0.01 # we assume vertical lidar

    nbvm = Nlayers*np.ones(len(msTyp), int)
    thtv = np.tile(singProf, len(msTyp))
    meas = np.r_[np.repeat(2.372179e-05, nbvm[0])]
    phi = np.repeat(0, len(thtv)) # currently we assume all observations fall within a plane
# errStr = [y for y in archName.lower().split('+') if 'lidar09' in y][0]
    nowPix = pixel(dt.datetime.now(), 1, 1, 0, 0, masl=0, land_prct=0)
    
    for wvl in wvls: # This will be expanded for wavelength dependent measurement types/geometry
        # errModel = functools.partial(addError, errStr, concase=concase, orbit=orbit, lidErrDir=lidErrDir) # this must link to an error model in addError() below
        nowPix.addMeas(wvl, msTyp, nbvm, sza, thtv, phi, meas,)


    Volume = np.arange(1e-5,2,10**-1)
   
    for i in range(len(Volume)):


    #change the value of concentration in the yaml file
        with open(fwdModelYAMLpath, 'r') as f:  
            data = yaml.safe_load(f) 
            data['retrieval']['constraints']['aerosol_concentration']['mode[1]']['value'][0] = Volume[i]
        f.close()
        
        with open(ymlPath+UpKerFile, 'w') as f: #write the chnages to new yaml file
            yaml.safe_dump(data, f)
        f.close()

        Newyaml = ymlPath+UpKerFile
        print('Using settings file at %s' % fwdModelYAMLpath)
        gr = graspRun(pathYAML= Newyaml, releaseYAML=True, verbose=True) # setup instance of graspRun class, add the above pixel, run grasp and read the output to gr.invRslt[0] (dict)
        gr.addPix(nowPix) # add the pixel we created above
        gr.runGRASP(binPathGRASP=binPathGRASP, krnlPathGRASP=krnlPath) # run grasp binary (output read to gr.invRslt[0] [dict])
        print('AOD at %5.3f μm was %6.4f.' % (gr.invRslt[0]['lambda'][-1],gr.invRslt[0]['aod'][-1])) # 

        VConcandExt[i,0], VConcandExt[i,1] = gr.invRslt[0]['vol'],gr.invRslt[0]['fit_VExt']

        dict_of_dicts[f'itr{i}'] = gr.invRslt[0]



    # Save to file using pickle
    with open('Vext2conc.pickle', 'wb') as f:
        pickle.dump(dict_of_dicts, f)


# def ComparisionPlots(LidarOnlyShape1,LidarOnlyShape2, CombShape1, CombShape2,key1 = None,key2= None,UNCERT= None): #Comapre the fits and retreivals between independendt and combined retreivals
    
#     IndexH = [0,3,7] #INdex for Lidar values in COmbined retrievals, there are nan values for values in RSP wavelength
#         #Converting range to altitude
#     altd = (LidarOnlyShape1['RangeLidar'][:,0])/1000 #altitude for spheriod
    
#     fig, axs= plt.subplots(nrows = 2, ncols =3, figsize= (15,12))  #TODO make it more general which adjust the no of rows based in numebr of wl
#     plt.subplots_adjust(top=0.78)
#     for i in range(3):
            
#         wave = str(LidarOnlyShape1['lambda'][i]) +"μm"
#         axs[i].errorbar(LidarOnlyShape1['meas_VBS'][:,i],altd,xerr= UNCERT['VBS'],color = "#695E93", capsize=3,capthick =1,alpha =0.4, label =f"{UNCERT['VBS']}")
#         axs[i].plot(LidarOnlyShape1['meas_VBS'][:,i],altd, marker =">",color = "#281C2D", label ="Meas")
#         axs[i].plot(LidarOnlyShape1['fit_VBS'][:,i],altd,color ="#025043", marker = "$O$",label =f"{key1}",alpha =0.8) #HSRL only
#         axs[i].plot(LidarOnlyShape1['fit_VBS'][:,IndexH[i]],altd,color ="#025043", marker = "$O$",label =f"{key2}",alpha =0.8) #Combined
        

        
        
#         axs[i].plot(HTam['fit_VBS'][:,i],altd,color =  "#d24787",ls = "--", label=f"{key2}", marker = "h")
#             axs[i].xaxis.set_major_formatter(ScalarFormatter(useMathText=True))

#             # print(UNCERT)
#             # axs[i].errorbar(Hsph['fit_VBS'][:,i],altd,xerr= UNCERT['VBS'],color = "#025043", marker = "$O$",label ="Sphd")
#             # axs[i].errorbar(HTam['fit_VBS'][:,i],altd,xerr= UNCERT['VBS'],color = "#d24787",ls = "--", label="Hex", marker = "h")

#             axs[i].set_xlabel(f'$ VBS (m^{-1}Sr^{-1})$',fontproperties=font_name)
#             axs[0].set_ylabel('Height above ground (km)',fontproperties=font_name)

#             axs[i].set_title(wave)
#             if i ==0:
#                 axs[0].legend()
#             plt.suptitle(f"HSRL Vertical Backscatter profile  \nLat,Lon: {HTam['latitude']}, {HTam['longitude']} Date: {HTam['datetime']}\n ",fontproperties=font_name) #Initial condition strictly constrainted by RSP retrievals
#         pdf_pages.savefig()
#         fig.savefig(f'/home/gregmi/ORACLES/HSRL_RSP/FIT_HSRL Vertical Backscatter profile{NoMode} .png',dpi = 300)
#             # plt.tight_layout()
#         fig, axs= plt.subplots(nrows = 1, ncols =3, figsize= (15,6))
#         plt.subplots_adjust(top=0.78)
#         for i in range(3):
#             wave = str(Hsph['lambda'][i]) +"μm"

#             axs[i].errorbar(Hsph['meas_DP'][:,i],altd,xerr= UNCERT['DP'],color = '#695E93', capsize=4,capthick =1,alpha =0.6, label =f"{UNCERT['DP']}%")
            
#             axs[i].plot(Hsph['meas_DP'][:,i],altd, marker =">",color = "#281C2D", label ="Meas")
#             axs[i].plot(Hsph['fit_DP'][:,i],altd,color = "#025043", marker = "$O$",label =f"{key1}")
#             axs[i].plot(HTam['fit_DP'][:,i],altd,color = "#d24787", ls = "--",marker = "h",label=f"{key2}")
            
#             # axs[i].errorbar(Hsph['fit_DP'][:,i],altd,xerr= UNCERT['DP'],fmt='-o',color = "#025043", marker = "$O$",label ="Sphd")
#             # axs[i].errorbar(HTam['fit_DP'][:,i],altd,xerr= UNCERT['DP'],fmt='-o',color = "#d24787",ls = "--", label="Hex", marker = "h")


#             axs[i].set_xlabel('DP %')
#             axs[i].set_title(wave)
#             if i ==0:
#                 axs[0].legend()
#             axs[0].set_ylabel('Height above ground (km)',fontproperties=font_name)
#             plt.suptitle(f"HSRL Depolarization Ratio \n Lat,Lon: {HTam['latitude']}, {HTam['longitude']}  Date: {HTam['datetime']}\n ",fontproperties=font_name) #Initial condition strictly constrainted by RSP retrievals
#         pdf_pages.savefig()
#         fig.savefig(f'/home/gregmi/ORACLES/HSRL_RSP/FIT_HSRL Depolarization Ratio{NoMode}.png',dpi = 300)
#         fig, axs= plt.subplots(nrows = 1, ncols =2, figsize= (11,6))
#         plt.subplots_adjust(top=0.78)
#         for i in range(2):
#             wave = str(Hsph['lambda'][i]) +"μm"

#             axs[i].errorbar(Hsph['meas_VExt'][:,i],altd,xerr= UNCERT['VEXT']*Hsph['meas_VExt'][:,i],color = '#695E93',capsize=4,capthick =1,alpha =0.7, label =f"{UNCERT['VEXT']*100}%")
            
#             axs[i].plot(Hsph['meas_VExt'][:,i],altd, marker =">",color = "#281C2D", label ="Meas")
#             axs[i].plot(Hsph['fit_VExt'][:,i],altd,color = "#025043", marker = "$O$",label =f"{key1}")
#             axs[i].plot(HTam['fit_VExt'][:,i],altd,color = "#d24787",ls = "--", marker = "h",label=f"{key2}")
#             axs[i].xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
#             # axs[i].errorbar(Hsph['fit_VExt'][:,i],altd,xerr= UNCERT['VEXT'],color = "#025043", marker = "$O$",label ="Sphd")
#             # axs[i].errorbar(HTam['fit_VExt'][:,i],altd,xerr= UNCERT['VEXT'],color = "#d24787",ls = "--", label="Hex", marker = "h")

            
            
#             axs[i].set_xlabel(f'$VExt (m^{-1})$',fontproperties=font_name)
#             axs[0].set_ylabel('Height above ground (km)',fontproperties=font_name)
#             axs[i].set_title(wave)
#             if i ==0:
#                 axs[0].legend()
#             plt.suptitle(f"HSRL Vertical Extinction profile\n Lat,Lon: {HTam['latitude']},{HTam['longitude']}  Date: {HTam['datetime']}\n ",fontproperties=font_name) #Initial condition strictly constrainted by RSP retrievals
#         plt.tight_layout()
#         pdf_pages.savefig()
#         fig.savefig(f'/home/gregmi/ORACLES/HSRL_RSP/FIT_HSRL_Vertical_Ext_profile{NoMode}.png',dpi = 300)







        
# for i in range(1):

#     #Reading the ORACLE data for given pixel no, Tel_no = aggregated altitude
#     #working pixels: 16800,16813 ,16814
#     # RSP_PixNo = 2776  #4368  #Clear pixel 8/01 Pixel no of Lat,Lon that we are interested

#     # RSP_PixNo = 13240
#     RSP_PixNo = 13200
#     #Case2

#     RSP_PixNo = 2687


#      #Dusty pixel on 9/22 
#     # PixNo = find_dust(file_path,file_name)[1][0]
#     TelNo = 0 # aggregated altitude. To obtain geometries corresponding to data from the 1880 nm channel, aggregation altitude should be set to 1, while aggregation altitude =0 should be used for all other channels.
#     nwl = 5 # first  nwl wavelengths
#     ang1 = 5
#     ang2 = 140 # :ang angles  #Remove

#     f1_MAP = h5py.File(file_path+file_name,'r+')   
#     Data = f1_MAP['Data']
#     LatRSP = f1_MAP['Geometry']['Collocated_Latitude'][TelNo,RSP_PixNo]
#     LonRSP = f1_MAP['Geometry']['Collocated_Longitude'][TelNo,RSP_PixNo]
#     f1_MAP.close()
#     f1= h5py.File(HSRLfile_path + HSRLfile_name,'r+')  #reading hdf5 file  
    
#     #Lat and Lon values for that pixel
#     LatH = f1['Nav_Data']['gps_lat'][:]
#     LonH = f1['Nav_Data']['gps_lon'][:]
#     f1.close()
#     # HSRLPixNo is the index of HSRL pixel taht corresponds to the RSP Lat Lon
#     HSRLPixNo = FindPix(LatH,LonH,LatRSP,LonRSP)[0]  # Or can manually give the index of the pixel that you are intrested in
 
# # #  Kernel_type = Run(Kernel_type) for spheriod, Kernel_type = 'TAMU' for hexahedral
#     rslts_Sph = RSP_Run("sphro",RSP_PixNo,ang1,ang2,TelNo,nwl)
#     rslts_Tamu = RSP_Run("TAMU",RSP_PixNo,ang1,ang2,TelNo,nwl)
#     RSP_plot(rslts_Sph,rslts_Tamu,RSP_PixNo)
  

#     HSRL_sphrod = HSLR_run("sphro",HSRLfile_path,HSRLfile_name,HSRLPixNo,nwl,ModeNo=3, updateYaml= False,releaseYAML= True)
#     plot_HSRL(HSRL_sphrod[0][0],HSRL_sphrod[0][0], forward = True, retrieval = True, Createpdf = True,PdfName ="/home/gregmi/ORACLES/rsltPdf/HSRL_Only_Plots_444.pdf", combinedVal =HSRL_sphrod[2]) 

#     HSRL_Tamu = HSLR_run("TAMU",HSRLfile_path,HSRLfile_name, HSRLPixNo,nwl,ModeNo=3,updateYaml= False,releaseYAML= True)
#     plot_HSRL(HSRL_Tamu[0][0],HSRL_Tamu[0][0], forward = True, retrieval = True, Createpdf = True,PdfName ="/home/gregmi/ORACLES/rsltPdf/HSRL_Only_Plots_444.pdf", combinedVal =HSRL_Tamu[2])

    
#     plot_HSRL(HSRL_sphrod[0][0],HSRL_Tamu[0][0], forward = True, retrieval = True, Createpdf = True,PdfName ="/home/gregmi/ORACLES/rsltPdf/HSRL_Only_Plots_444.pdf", combinedVal =HSRL_sphrod[2])

#     # # print('Cost Value Sph, tamu: ',  rslts_Sph[0]['costVal'], rslts_Tamu[0]['costVal'])
#     # RSP_plot(rslts_Sph,rslts_Tamu,RSP_PixNo)
#     # plot_HSRL(rslts_Sph[0],rslts_Tamu[0], forward = False, retrieval = True, Createpdf = True,PdfName =f"/home/gregmi/ORACLES/rsltPdf/RSP_only_{RSP_PixNo}.pdf")
    
# #     # Retrieval_type = 'NosaltStrictConst_final'
# #     # #Running GRASP for HSRL, HSRL_sphrod = for spheriod kernels,HSRL_Tamu = Hexahedral kernels
    
# #     print('Cost Value Sph, tamu: ',  HSRL_sphrod[0][0]['costVal'],HSRL_Tamu[0][0]['costVal'])
   
# # # #     # # #Constrining HSRL retrievals by 5% 
# #     HSRL_sphro_5 = HSLR_run("sphro",HSRLfile_path,HSRLfile_name,HSRLPixNo,nwl,ModeNo=3,updateYaml= True,releaseYAML= True) 
# #     HSRL_Tamu_5 = HSLR_run("TAMU",HSRLfile_path,HSRLfile_name,HSRLPixNo,nwl,ModeNo=3,updateYaml= True,releaseYAML= True)

# # #     print('Cost Value Sph, tamu: ',  HSRL_sphro_5[0][0]['costVal'],HSRL_Tamu_5[0][0]['costVal'])

# # # #     #Strictly Constrining HSRL retrievals to values from RSP retrievals
# #     HSRL_sphrod_strict = HSLR_run("sphro",HSRLfile_path,HSRLfile_name,HSRLPixNo,nwl,ModeNo=3,updateYaml= True, ConsType = 'strict',releaseYAML= True) 
# #     HSRL_Tamu_strict = HSLR_run("TAMU",HSRLfile_path,HSRLfile_name,HSRLPixNo,nwl,ModeNo=3,updateYaml= True, ConsType = 'strict',releaseYAML= True)

# # #     print('Cost Value Sph, tamu: ',  HSRL_sphrod_strict[0][0]['costVal'],HSRL_Tamu_strict[0][0]['costVal'])
    
#     # RIG = randomInitGuess('sphro',HSRLfile_path,HSRLfile_name,HSRLPixNo, nwl,releaseYAML =True, ModeNo=3)
#      #Lidar+pol combined retrieval
#     # LidarPolSph = LidarAndMAP('sphro',HSRLfile_path,HSRLfile_name,HSRLPixNo,file_path,file_name,RSP_PixNo,ang1,ang2,TelNo, nwl,GasAbsFn,ModeNo=3, updateYaml= None,RandinitGuess=True, NoItr =100 )
#     # # PlotRandomGuess('gregmi/git/GSFC-GRASP-Python-Interface/try.npy', 2,0)
#     LidarPolSph = LidarAndMAP('sphro',HSRLfile_path,HSRLfile_name,HSRLPixNo,file_path,file_name,RSP_PixNo,ang1,ang2,TelNo, nwl,GasAbsFn,ModeNo=3, updateYaml= None)
#     LidarPolTAMU = LidarAndMAP('TAMU',HSRLfile_path,HSRLfile_name,HSRLPixNo,file_path,file_name,RSP_PixNo,ang1,ang2,TelNo, nwl,GasAbsFn,ModeNo=3, updateYaml= None)
    
#     LidarPolTAMU = LidarAndMAP('TAMU',HSRLfile_path,HSRLfile_name,HSRLPixNo,file_path,file_name,RSP_PixNo,ang1,ang2,TelNo, nwl,GasAbsFn,ModeNo=3, updateYaml= None,RandinitGuess=True, NoItr =150)
#     LidarPolSph = LidarAndMAP('sphro',HSRLfile_path,HSRLfile_name,HSRLPixNo,file_path,file_name,RSP_PixNo,ang1,ang2,TelNo, nwl,GasAbsFn,ModeNo=3, updateYaml= None,RandinitGuess=True, NoItr =150)

    
    
    # # LidarPolSph = LidarAndMAP('sphro',HSRLfile_path,HSRLfile_name,HSRLPixNo,file_path,file_name,RSP_PixNo,ang1,ang2,TelNo, nwl,GasAbsFn,ModeNo=3, updateYaml= None,RandinitGuess=True, NoItr =150)
    # LidarPolTAMU = LidarAndMAP('TAMU',HSRLfile_path,HSRLfile_name,HSRLPixNo,file_path,file_name,RSP_PixNo,ang1,ang2,TelNo, nwl,GasAbsFn,ModeNo=3, updateYaml= None,RandinitGuess=True, NoItr =150)


    # RandinitGuess=True, NoItr =2 
    # print('Cost Value Sph, tamu: ',  LidarPolSph[0]['costVal'],LidarPolTAMU[0]['costVal'])
    # # print('SPH',"tam" )


    # %matplotlib inline
    # plot_HSRL(HSRL_sphrod[0][0],HSRL_Tamu[0][0], forward = True, retrieval = True, Createpdf = True,PdfName ="/home/gregmi/ORACLES/rsltPdf/HSRL_Only_Plots_444.pdf", combinedVal =HSRL_sphrod[2])
    # plot_HSRL(HSRL_sphro_5[0][0],HSRL_Tamu_5[0][0], forward = True, retrieval = True, Createpdf = True,PdfName ="/home/gregmi/ORACLES/rsltPdf/5%_HSRL_Plots_444.pdf", combinedVal =HSRL_sphrod[2])
    # plot_HSRL(HSRL_sphrod_strict[0][0],HSRL_Tamu_strict[0][0], forward = True, retrieval = True, Createpdf = True,PdfName ="/home/gregmi/ORACLES/rsltPdf/STR_HSRL_Plots_444.pdf", combinedVal =HSRL_sphrod[2])
    # plot_HSRL(LidarPolSph[0],LidarPolTAMU[0], forward = False, retrieval = True, Createpdf = True,PdfName ="/home/gregmi/ORACLES/rsltPdf/LIDARPOL_Plots_444.pdf")
    # plot_HSRL(LidarPolSph[0][0],LidarPolTAMU[0][0], forward = False, retrieval = True, Createpdf = True,PdfName ="/home/gregmi/ORACLES/rsltPdf/LIDARPOL_Plots_444.pdf")
    # CombinedLidarPolPlot(LidarPolSph[0],LidarPolTAMU[0])

#     plot_HSRL(LidarPolTAMU[0][0],LidarPolTAMU[0][0], forward = False, retrieval = True, Createpdf = True,PdfName ="/home/gregmi/ORACLES/rsltPdf/LIDARPOL_Plots_444.pdf")
# CombinedLidarPolPlot(LidarPolTAMU[0],LidarPolTAMU[0])
