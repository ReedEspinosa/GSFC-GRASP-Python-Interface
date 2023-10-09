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
# %autoreload 2
# %matplotlib inline

import sys
from CreateRsltsDict import Read_Data_RSP_Oracles
from CreateRsltsDict import Read_Data_HSRL_Oracles,Read_Data_HSRL_Oracles_Height
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
# from Plot_ORACLES import PltGRASPoutput, PlotRetrievals
import yaml
# %matplotlib inline



# Path to the Polarimeter data (RSP, In this case)
file_path = "/home/gregmi/ORACLES/RSP1-L1C_P3_20180922_R03/"  #Path to the ORACLE data file
file_name =  "/RSP1-P3_L1C-RSPCOL-CollocatedRadiances_20180922T151106Z_V003-20210421T233946Z.h5" #Name of the ORACLES file
#Paths to the Lidar Data
HSRLfile_path = "/home/gregmi/ORACLES/HSRL" #Path to the ORACLE data file 
HSRLfile_name =  "/HSRL2_P3_20180922_R2.h5" #Name of the ORACLES file
#Path to the gas absorption (tau) values for gas absorption correction
GasAbsFn = '/home/gregmi/ORACLES/UNL_VRTM/shortwave_gas.unlvrtm.nc'

#This is required if we want to configure the HSRL yaml file based on the GRASP output for the RSP
noMod =2  #number of aerosol mode, here 2 for fine+coarse mode configuration
maxr=1.05  #set max and min value : here max = 5% incease, min 5% decrease : this is a very narrow distribution
minr =0.95
a=1 #no of char # this is a varible added to char and modes to avoid char[0]/ mod[0] which doesnt exist

#Saving state variables and noises from yaml file

        
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
            #     # initCond['max'] = float(np.max(RSP_rslt['dVdlnr'][noMd])) #value from the GRASP result for RSP
            #     # initCond['min'] = float(RSP_rslt['dVdlnr'][noMd][1])
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

                initCond['value'] =float(RSP_rslt['n'][noMd][1]+offs),float(RSP_rslt['n'][noMd][2]),float(RSP_rslt['n'][noMd][4])
                initCond['max'] =float((RSP_rslt['n'][noMd][1]+offs)*maxr),float(RSP_rslt['n'][noMd][2]*maxr),float(RSP_rslt['n'][noMd][4]*maxr)
                initCond['min'] =float((RSP_rslt['n'][noMd][1]+offs)*minr),float(RSP_rslt['n'][noMd][2]*minr),float(RSP_rslt['n'][noMd][4]*minr)
                # print("Updating",YamlChar[i])
                if ConsType == 'strict':  # tot set retrival in setting files to False
                    data['retrieval']['constraints'][f'characteristic[{i+a}]']['retrieved'] = 'false'
            
            if YamlChar[i] == 'imaginary_part_of_refractive_index_spectral_dependent':
                initCond['index_of_wavelength_involved'] = [1,2,3]
                initCond['value'] =float(RSP_rslt['k'][noMd][1]),float(RSP_rslt['k'][noMd][2]),float(RSP_rslt['k'][noMd][4])
                initCond['max'] =float(RSP_rslt['k'][noMd][1]*maxr),float(RSP_rslt['k'][noMd][2]*maxr),float(RSP_rslt['k'][noMd][4]*maxr)
                initCond['min'] = float(RSP_rslt['k'][noMd][1]*minr),float(RSP_rslt['k'][noMd][2]*minr),float(RSP_rslt['k'][noMd][4]*minr)
                # print("Updating",YamlChar[i])
                if ConsType == 'strict':
                    data['retrieval']['constraints'][f'characteristic[{i+a}]']['retrieved'] = 'false'
            
            if YamlChar[i] == 'sphere_fraction':
                initCond['value'] = float(RSP_rslt['sph'][noMd]/100)
                initCond['max'] =float(RSP_rslt['sph'][noMd]/100 *maxr) #GARSP output is in %
                initCond['min'] =float(RSP_rslt['sph'][noMd]/100 *minr)
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
def RSP_Run(Kernel_type,PixNo,ang1,ang2,TelNo,nwl): 
        
        krnlPath='/home/shared/GRASP_GSFC/src/retrieval/internal_files'
        # Kernel_type =  sphro is for the GRASP spheriod kernal, while TAMU is to run with Hexahedral Kernal
        if Kernel_type == "sphro":
            fwdModelYAMLpath = '/home/gregmi/git/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/settings_BCK_DoLP_POLAR_2modes_SphrodShape_ORACLE.yml'
            # fwdModelYAMLpath ='/home/gregmi/git/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/settings_BCK_POLAR_2modes_Shape_ORACLE_DoLP_2COARSE.yml'
            # binPathGRASP ='/home/shared/GRASP_GSFC/build_RSP_v112/bin/grasp_app' 
            
            savePath=f"/home/gregmi/ORACLES/RSP1-L1C_P3_20180922_R03_{Kernel_type}"
        
        if Kernel_type == "TAMU":
            fwdModelYAMLpath = '/home/gregmi/git/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/settings_BCK_DoLP_POLAR_2modes_HexShape_ORACLE.yml'
            # fwdModelYAMLpath ='/home/gregmi/git/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/settings_BCK_POLAR_2modes_Shape_ORACLE_DoLP_dust_2Coarse.yml'
            # binPathGRASP ='/home/shared/GRASP_GSFC/build_HEX_v112/bin/grasp_app' #GRASP Executable
            binPathGRASP ='/home/shared/GRASP_GSFC/build_HexV112_4Modes/bin/grasp_app'
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
def HSLR_run(Kernel_type,HSRLfile_path,HSRLfile_name,PixNo,nwl, updateYaml= None,ConsType = None,releaseYAML=True):
        #Path to the kernel files
        krnlPath='/home/shared/GRASP_GSFC/src/retrieval/internal_files'
        
        if Kernel_type == "sphro":  #If spheroid model
            #Path to the yaml file for sphreroid model
            # fwdModelYAMLpath ='/home/gregmi/git/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/settings_BCK_POLAR_4modes_Shape_ORACLE1.yml'
            
            # fwdModelYAMLpath = '/home/gregmi/git/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/settings_BCK_POLARandLIDAR_10Vbins_2modes_ORACLES.yml'
            fwdModelYAMLpath ='/home/gregmi/git/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/settings_BCK_POLAR_3modes_Shape_ORACLE.yml'
            if updateYaml == True:  # True if init conditions for Yaml file for HSRL is updated from the GRASP output from RSP
                update_HSRLyaml(fwdModelYAMLpath, rslts_Sph[0], noMod, maxr, minr, a,Kernel_type,ConsType)
                fwdModelYAMLpath ='/home/gregmi/git/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/Settings_Sphd_RSP_HSRL.yaml'
            # binPathGRASP = path toGRASP Executable for spheriod model
            binPathGRASP ='/home/shared/GRASP_GSFC/build_HEX_v112/bin/grasp_app'
            info = VariableNoise(fwdModelYAMLpath,nwl)
            # binPathGRASP ='/home/shared/GRASP_GSFC/build_SPHV112_4Modes/bin/grasp_app'
           
            
            
            # binPathGRASP ='/home/shared/GRASP_GSFC/build_RSP_v112/bin/grasp_app' 
            savePath=f"/home/gregmi/ORACLES/HSRL1_P3_20180922_R03_{Kernel_type}"
        
        if Kernel_type == "TAMU":
            # fwdModelYAMLpath = '/home/gregmi/git/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/settings_BCK_POLARandLIDAR_10Vbins_2modes_Tamu.yml'
            fwdModelYAMLpath ='/home/gregmi/git/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/settings_BCK_POLAR_3modes_Shape_Hex.yml'
            # fwdModelYAMLpath = '/home/gregmi/git/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/settings_BCK_POLAR_3modes_Shape_ORACLE.yml'
            if updateYaml == True:# True if init conditions for Yaml file for HSRL is updated from the GRASP output from RSP
                update_HSRLyaml(fwdModelYAMLpath, rslts_Tamu[0], noMod, maxr, minr, a,Kernel_type,ConsType)
                fwdModelYAMLpath ='/home/gregmi/git/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/Settings_TAMU_RSP_HSRL.yaml'
            #Path to the GRASP Executable for TAMU
            info = VariableNoise(fwdModelYAMLpath,nwl)
            binPathGRASP ='/home/shared/GRASP_GSFC/build_HEX_v112/bin/grasp_app' #GRASP Executable
            # binPathGRASP ='/home/shared/GRASP_GSFC/build_HexV112_4Modes/bin/grasp_app'
            #Path to save output plot
            savePath=f"/home/gregmi/ORACLES/RSP1-L1C_P3_20180922_R03_{Kernel_type}"

    
        #rslt is the GRASP rslt dictionary or contains GRASP Objects
        rslt = Read_Data_HSRL_Oracles(HSRLfile_path,HSRLfile_name,PixNo)
        max_alt = rslt['OBS_hght']
        print(rslt['OBS_hght'])

        maxCPU = 3 #maximum CPU allocated to run GRASP on server
        gRuns = []
        yamlObj = graspYAML(baseYAMLpath=fwdModelYAMLpath)
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
        # binPathGRASP ='/home/shared/GRASP_GSFC/build_SphrdV112_Noise/bin/grasp_app'
        binPathGRASP ='/home/shared/GRASP_GSFC/build_HexV112_4Modes/bin/grasp_app'  #This will work for both the kernels as it has more parameter wettings
       
        # binPathGRASP ='/home/shared/GRASP_GSFC/build_RSP_v112/bin/grasp_app' 
        savePath=f"/home/gregmi/ORACLES/HSRL1_P3_20180922_R03_{Kernel_type}"
    
    if Kernel_type == "TAMU":
        fwdModelYAMLpath = '/home/gregmi/git/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/settings_BCK_LidarAndMAP_V.1.2_TAMU.yml'
        # fwdModelYAMLpath ='/home/gregmi/git/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/settings_BCK_POLARandLIDAR_10Vbins_2modes_Tamu_2Coarse.yml'
        if updateYaml == True:# True if init conditions for Yaml file for HSRL is updated from the GRASP output from RSP
            update_HSRLyaml(fwdModelYAMLpath, rslts_Tamu[0], noMod, maxr, minr, a,Kernel_type)
            fwdModelYAMLpath ='/home/gregmi/git/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/Settings_TAMU_RSP_HSRL.yaml'

        #Path to the GRASP Executable for TAMU
        # binPathGRASP ='/home/shared/GRASP_GSFC/build_HexV112_Noise/bin/grasp_app' #Recompiled to account for more noise parameter
        # binPathGRASP ='/home/shared/GRASP_GSFC/build_HEX_v112/bin/grasp_app' #GRASP Executable
        binPathGRASP ='/home/shared/GRASP_GSFC/build_HexV112_4Modes/bin/grasp_app'
        #Path to save output plot
        savePath=f"/home/gregmi/ORACLES/RSP1-L1C_P3_20180922_R03_{Kernel_type}"

# /tmp/tmpn596k7u8$
    rslt_HSRL = Read_Data_HSRL_Oracles(HSRLfile_path,HSRLfile_name,HSRLPixNo)
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
    # rslt['masl'] = 0  #height of the ground
    # print(rslt)

    maxCPU = 3 #maximum CPU allocated to run GRASP on server
    gRuns = []
    yamlObj = graspYAML(baseYAMLpath=fwdModelYAMLpath)
    #eventually have to adjust code for height, this works only for one pixel (single height value)
    gRuns.append(graspRun(pathYAML=yamlObj, releaseYAML= False )) # This should copy to new YAML object
    pix = pixel()
    pix.populateFromRslt(rslt, radianceNoiseFun=None, dataStage= 'meas', verbose=False)
    gRuns[-1].addPix(pix)
    gDB = graspDB(graspRunObjs=gRuns, maxCPU=maxCPU)
    #rslts contain all the results form the GRASP inverse run
    rslts, failPix = gDB.processData(binPathGRASP=binPathGRASP, savePath=None, krnlPathGRASP=krnlPath)
    return rslts

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


#Plotting the fits and the retrievals
def plot_HSRL(HSRL_sphrod,HSRL_Tamu, forward = None, retrieval = None, Createpdf = None,PdfName =None, combinedVal = None):

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
    plt.rcParams['font.size'] = '18'


    

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
            
    if forward == True:
        #Converting range to altitude
        altd = (Hsph['RangeLidar'][:,0])/1000 #altitude for spheriod
        altT = (Hsph['RangeLidar'][:,0])/1000 #altitude for hexahedra
        fig, axs= plt.subplots(nrows = 1, ncols =2, figsize= (11,6))  #TODO make it more general which adjust the no of rows based in numebr of wl
        plt.subplots_adjust(top=0.78)
        for i in range(2):
            wave = str(Hsph['lambda'][i]) +"μm"
            axs[i].plot(Hsph['meas_VBS'][:,i],altd, marker =">",color = "#3B270C", label ="Meas")
            axs[i].plot(Hsph['fit_VBS'][:,i],altd,color = "#025043", marker = "$O$",label ="Sphd")
            axs[i].plot(HTam['fit_VBS'][:,i],altd,color = "#d24787",ls = "--", label="Hex", marker = "h")

            axs[i].set_xlabel(f'$ VBS (m^{-1}Sr^{-1})$',fontproperties=font_name)
            axs[0].set_ylabel('Height above ground (km)',fontproperties=font_name)

            axs[i].set_title(wave)
            if i ==0:
                axs[0].legend()
            plt.suptitle(f"HSRL Vertical Backscatter profile  \nLat,Lon: {HTam['latitude']}, {HTam['longitude']} Date: {HTam['datetime']}\n ",fontproperties=font_name) #Initial condition strictly constrainted by RSP retrievals
        pdf_pages.savefig()
        fig.savefig(f'/home/gregmi/ORACLES/HSRL_RSP/FIT_{HSRLPixNo}_HSRL Vertical Backscatter profile .png',dpi = 300)
            # plt.tight_layout()
        fig, axs= plt.subplots(nrows = 1, ncols =3, figsize= (18,6))
        plt.subplots_adjust(top=0.78)
        for i in range(3):
            wave = str(Hsph['lambda'][i]) +"μm"
            axs[i].plot(Hsph['meas_DP'][:,i],altd, marker =">",color = "#3B270C", label ="Meas")
            axs[i].plot(Hsph['fit_DP'][:,i],altd,color = "#025043", marker = "$O$",label ="Sphd")
            axs[i].plot(HTam['fit_DP'][:,i],altd,color = "#d24787", ls = "--",marker = "h")
            axs[i].set_xlabel('DP %')
            axs[i].set_title(wave)
            if i ==0:
                axs[0].legend()
            axs[0].set_ylabel('Height above ground (km)',fontproperties=font_name)
            plt.suptitle(f"HSRL Depolarization Ratio \n Lat,Lon: {HTam['latitude']}, {HTam['longitude']}  Date: {HTam['datetime']}\n ",fontproperties=font_name) #Initial condition strictly constrainted by RSP retrievals
        pdf_pages.savefig()
        fig.savefig(f'/home/gregmi/ORACLES/HSRL_RSP/FIT_{HSRLPixNo}_HSRL Depolarization Ratio.png',dpi = 300)
        fig, axs= plt.subplots(nrows = 1, ncols =2, figsize= (11,6))
        plt.subplots_adjust(top=0.78)
        for i in range(2):
            wave = str(Hsph['lambda'][i]) +"μm"
            axs[i].plot(Hsph['meas_VExt'][:,i],altd, marker =">",color = "#3B270C", label ="Meas")
            axs[i].plot(Hsph['fit_VExt'][:,i],altd,color = "#025043", marker = "$O$",label ="Sphd")
            axs[i].plot(HTam['fit_VExt'][:,i],altd,color = "#d24787",ls = "--", marker = "h")
            axs[i].set_xlabel(f'$VExt (m^{-1})$',fontproperties=font_name)
            axs[0].set_ylabel('Height above ground (km)',fontproperties=font_name)
            axs[i].set_title(wave)
            if i ==0:
                axs[0].legend()
            plt.suptitle(f"HSRL Vertical Extinction profile\n Lat,Lon: {HTam['latitude']},{HTam['longitude']}  Date: {HTam['datetime']}\n ",fontproperties=font_name) #Initial condition strictly constrainted by RSP retrievals
        pdf_pages.savefig()
        fig.savefig(f'/home/gregmi/ORACLES/HSRL_RSP/FIT_{HSRLPixNo}_HSRL_Vertical_Ext_profile.png',dpi = 300)

    if retrieval == True:
        Spheriod,Hex = HSRL_sphrod,HSRL_Tamu
        plt.rcParams['font.size'] = '17'
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



        if Spheriod['r'].shape[0] ==2 :
            mode_v = ["fine","marine", "dust"]
        if Spheriod['r'].shape[0] ==3 :
            mode_v = ["fine","marine", "dust"]
        
        linestyle =[':', '-','-.']

        cm_sp = ['#008080',"#C1E1C1",'c' ]
        cm_t = ['#900C3F',"#FF5733" ,'b']
        
        color_sph = '#0c7683'
        color_tamu = "#BC106F"

        #Retrivals:
        fig, axs = plt.subplots(nrows= 3, ncols=2, figsize=(18, 10))
        for i in range(len(Retrival)):
            a,b = i%3,i%2
            for mode in range(Spheriod['r'].shape[0]): #for each modes
                if i ==0:
                    axs[a,b].plot(Spheriod['r'][mode], Spheriod[Retrival[i]][mode], marker = "$O$",color = cm_sp[mode],ls = linestyle[mode], label=f"Sphrod_{mode_v[mode]}")
                    axs[a,b].plot(Hex['r'][mode],Hex[Retrival[i]][mode], marker = "H", color = cm_t[mode] , ls = linestyle[mode],label=f"Hex_{mode_v[mode]}")
                    axs[0,0].set_xlabel('Radius')
                    axs[0,0].set_xscale("log")
                else:
                    axs[a,b].plot(Spheriod['lambda'], Spheriod[Retrival[i]][mode], marker = "$O$",color = cm_sp[mode],ls = linestyle[mode], label=f"Sphrod_{mode_v[mode]}")
                    axs[a,b].plot(Hex['lambda'],Hex[Retrival[i]][mode], marker = "H",color = cm_t[mode] , ls = linestyle[mode],label=f"Hex_{mode_v[mode]}")
                    axs[a,b].set_xticks(Spheriod['lambda'])
                    # axs[a,b].set_xticklabels(['0.41', '0.46', '0.55' , '0.67'  , '0.86'])
            axs[a,b].set_ylabel(f'{Retrival[i]}')
            axs[a,b].set_xlabel(r'$\lambda$')
            axs[0,0].legend()
            axs[2,1].plot(Spheriod['lambda'], Spheriod['aod'], marker = "$O$",color = cm_sp[mode],ls = linestyle[mode], label=f"Sphrod_{mode_v[mode]}")
            axs[2,1].plot(Hex['lambda'], Hex['aod'], marker = "H", color = cm_t[mode] , ls = linestyle[mode],label=f"Hex_{mode_v[mode]}")
            axs[2,1].set_xlabel(r'$\lambda$')
            axs[2,1].set_ylabel('Total AOD')
            


        lat_t = Hex['latitude']
        lon_t = Hex['longitude']
        dt_t = Hex['datetime']
        chisph,chihex = Spheriod['costVal'] ,Hex['costVal']
        plt.suptitle(f'HSRL2 Aerosol Retrieval \n  Lat:{lat_t} Lon :{lon_t}\n Date: {dt_t} \n Cost Function : {chisph,chihex }')
        pdf_pages.savefig()
        pdf_pages.close()
        fig.savefig(f'/home/gregmi/ORACLES/HSRL_RSP/{dt_t}_RSPRetrieval.png', dpi = 400)
        
        
    return

def CombinedLidarPolPlot(LidarPolSph,LidarPolTAMU): #should be updated to 
    fig, axs= plt.subplots(nrows = 1, ncols =3, figsize= (18,6))

    altd =LidarPolTAMU[0]['RangeLidar'][:,0]/1000
    HTam = LidarPolTAMU[0]
    plt.subplots_adjust(top=0.78)
    IndexH = [0,3,7]
    for i in range(3):

        wave = str(LidarPolSph[0]['lambda'][IndexH[i]]) +"μm"
        axs[i].plot(LidarPolSph[0]['meas_VBS'][:,IndexH[i]],(LidarPolSph[0]['RangeLidar'][:,0])/1000, marker =">",color = "#3B270C", label ="Meas")
        axs[i].plot(LidarPolSph[0]['fit_VBS'][:,IndexH[i]],(LidarPolSph[0]['RangeLidar'][:,0])/1000,color = "#025043", marker = "$O$",label ="Sphd")
        axs[i].plot(LidarPolTAMU[0]['fit_VBS'][:,IndexH[i]],altd,color = "#d24787",ls = "--", label="Hex", marker = "h")

        axs[i].set_xlabel('VBS')
        axs[i].set_title(wave)
        if i ==0:
            axs[0].legend()
        plt.suptitle(f"HSRL+RSP Vertical Backscatter profile  \n Lat: {HTam['latitude']},Lon:{HTam['longitude']} Date: {HTam['datetime']}\n ") #Initial condition strictly constrainted by RSP retrievals
    fig.savefig(f'/home/gregmi/ORACLES/HSRL_RSP/FIT_{HSRLPixNo}_Combined_DP.png',dpi = 300)

    fig, axs= plt.subplots(nrows = 1, ncols =3, figsize= (18,6))
    plt.subplots_adjust(top=0.78)
    for i in range(3):
        wave = str(LidarPolSph[0]['lambda'][IndexH[i]]) +"μm"
        axs[i].plot(LidarPolSph[0]['meas_DP'][:,IndexH[i]],(LidarPolSph[0]['RangeLidar'][:,0])/1000, marker =">",color = "#3B270C", label ="Meas")
        axs[i].plot(LidarPolSph[0]['fit_DP'][:,IndexH[i]],(LidarPolSph[0]['RangeLidar'][:,0])/1000,color = "#025043", marker = "$O$",label ="Sphd")
        axs[i].plot(LidarPolTAMU[0]['fit_DP'][:,IndexH[i]],altd,color = "#d24787", ls = "--",marker = "h", label="Hex")
        axs[i].set_xlabel('DP %')
        axs[i].set_title(wave)
        if i ==0:
            axs[0].legend()
        plt.suptitle(f"HSRL Depolarization Ratio \n Lat: {LidarPolTAMU[0]['latitude']},Lon:{LidarPolTAMU[0]['longitude']} Date: {HTam['datetime']}\n Lidar+polarimeter Retrievals ") #Initial condition strictly constrainted by RSP retrievals
    fig.savefig(f'/home/gregmi/ORACLES/HSRL_RSP/FIT_{HSRLPixNo}_Combined_VBS.png',dpi = 300)

    fig, axs= plt.subplots(nrows = 1, ncols =2, figsize= (18,6))
    plt.subplots_adjust(top=0.78)
    for i in range(2):
        wave = str(LidarPolSph[0]['lambda'][IndexH[i]]) +"μm"
        axs[i].plot(LidarPolSph[0]['meas_VExt'][:,IndexH[i]],(LidarPolSph[0]['RangeLidar'][:,0])/1000, marker =">",color = "#3B270C", label ="Meas")
        axs[i].plot(LidarPolSph[0]['fit_VExt'][:,IndexH[i]],(LidarPolSph[0]['RangeLidar'][:,0])/1000,color = "#025043", marker = "$O$",label ="Sphd")
        axs[i].plot(LidarPolTAMU[0]['fit_VExt'][:,IndexH[i]],altd,color = "#d24787",ls = "--", marker = "h", label="Hex")
        axs[i].set_xlabel('VExt')
        axs[i].set_title(wave)
        if i ==0:
            axs[0].legend()
        plt.suptitle(f"HSRL Vertical Extinction Profile Fit \n Lat: {LidarPolTAMU[0]['latitude']},Lon:{LidarPolTAMU[0]['longitude']} Date: {HTam['datetime']}\n Lidar+polarimeter Retrievals  ") #Initial condition strictly constrainted by RSP retrievals
    fig, axs= plt.subplots(nrows = 1, ncols =2, figsize= (18,6))
    
    fig.savefig(f'/home/gregmi/ORACLES/HSRL_RSP/FIT_{HSRLPixNo}_Combined_Vertical_EXT_profile.png',dpi = 300)

    for i in [1,2,4,5,6]:

        axs[0].plot(LidarPolSph[0]['sca_ang'][:,i], LidarPolSph[0]['meas_I'][:,i],marker =">",color = "#3B270C", label ="Meas")
        axs[0].plot(LidarPolSph[0]['sca_ang'][:,i],LidarPolSph[0]['fit_I'][:,i],color = "#025043", marker = "$O$",label ="Sphd")
        axs[0].plot(LidarPolSph[0]['sca_ang'][:,i],LidarPolTAMU[0]['fit_I'][:,i],color = "#d24787",ls = "--", marker = "h", label="Hex")
        axs[0].set_ylabel('I')
        axs[0].set_xlabel('Scattering angles')
        
        plt.suptitle(f"Wl: {LidarPolSph[0]['lambda'][i]}, Lat: {LidarPolTAMU[0]['latitude']},Lon:{LidarPolTAMU[0]['longitude']} Date: {HTam['datetime']}\n Lidar+polarimeter Retrievals  ") #Initial condition strictly constrainted by RSP retrievals
        
        axs[1].plot(LidarPolSph[0]['sca_ang'][:,i],LidarPolSph[0]['meas_P_rel'][:,i], marker =">",color = "#3B270C", label ="Meas")
        axs[1].plot(LidarPolSph[0]['sca_ang'][:,i],LidarPolSph[0]['fit_P_rel'][:,i],color = "#025043", marker = "$O$",label ="Sphd")
        axs[1].plot(LidarPolSph[0]['sca_ang'][:,i],LidarPolTAMU[0]['fit_P_rel'][:,i],color = "#d24787",ls = "--", marker = "h", label="Hex")
        axs[1].set_ylabel('P/I')
        axs[1].set_xlabel('Scattering angles')
        axs[0].legend()
        # axs[1].set_title()
        fig, axs= plt.subplots(nrows = 1, ncols =2, figsize= (18,6))
        fig.savefig(f'/home/gregmi/ORACLES/HSRL_RSP/FIT_{RSP_PixNo}_Combined_I_P_{i}.png',dpi = 300)


def RSP_plot(rslts_Sph,rslts_Tamu,RSP_PixNo, LIDARPOL= None): 
    
    Spheriod,Hex = rslts_Sph[0],rslts_Tamu[0]
    plt.rcParams['font.size'] = '17'
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
    mode_v = ["fine", "coarse"]
    linestyle =[':', '-']

    cm_sp = ['#008080',"#C1E1C1" ]
    cm_t = ['#900C3F',"#FF5733" ]
    color_sph = '#0c7683'
    color_tamu = "#BC106F"

    #Retrivals:
    fig, axs = plt.subplots(nrows= 3, ncols=2, figsize=(35, 15))
    for i in range(len(Retrival)):
        a,b = i%3,i%2
        for mode in range(Spheriod['r'].shape[0]): #for each modes
            if i ==0:
                axs[a,b].plot(Spheriod['r'][mode], Spheriod[Retrival[i]][mode], marker = "$O$",color = cm_sp[mode],ls = linestyle[mode], label=f"Sphrod_{mode_v[mode]}")
                axs[a,b].plot(Hex['r'][mode],Hex[Retrival[i]][mode], marker = "H", color = cm_t[mode] , ls = linestyle[mode],label=f"Hex_{mode_v[mode]}")
                axs[0,0].set_xlabel('Radius')
                axs[0,0].set_xscale("log")
            else:
                axs[a,b].plot(Spheriod['lambda'], Spheriod[Retrival[i]][mode], marker = "$O$",color = cm_sp[mode],ls = linestyle[mode], label=f"Sphrod_{mode_v[mode]}")
                axs[a,b].plot(Hex['lambda'],Hex[Retrival[i]][mode], marker = "H",color = cm_t[mode] , ls = linestyle[mode],label=f"Hex_{mode_v[mode]}")
                axs[a,b].set_xticks(Spheriod['lambda'])
                # axs[a,b].set_xticklabels(['0.41', '0.46', '0.55' , '0.67'  , '0.86'])
        axs[a,b].set_ylabel(f'{Retrival[i]}')
        axs[a,b].set_xlabel(r'$\lambda$')
        
        axs[2,1].plot(Spheriod['lambda'], Spheriod['aod'], marker = "$O$",color = cm_sp[mode],ls = linestyle[mode], label=f"Sphrod_{mode_v[mode]}")
        axs[2,1].plot(Hex['lambda'], Hex['aod'], marker = "H", color = cm_t[mode] , ls = linestyle[mode],label=f"Hex_{mode_v[mode]}")
        axs[2,1].set_xlabel(r'$\lambda$')
        axs[2,1].set_ylabel('Total AOD')
    axs[0,0].legend()
    lat_t = Hex['latitude']
    lon_t = Hex['longitude']
    dt_t = Hex['datetime']
    plt.suptitle(f'RSP Aerosol Retrieval \n  Lat:{lat_t} Lon :{lon_t}\n Date: {dt_t}')
    fig.savefig(f'/home/gregmi/ORACLES/HSRL_RSP/{dt_t}_RSPRetrieval.png', dpi = 400)

    #Stokes: 
    wl = rslts_Sph[0]['lambda'] 
    fig, axs = plt.subplots(nrows= 4, ncols=5, figsize=(30, 10),gridspec_kw={'height_ratios': [1, 0.3,1,0.3]}, sharex='col')
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
        axs[0, nwav].plot(Spheriod['sca_ang'][:,wlIdx[nwav]], Spheriod['meas_I'][:,wlIdx[nwav]], color = "k", lw = 1, label="meas")

        axs[0, nwav].plot(Spheriod ['sca_ang'][:,wlIdx[nwav]], Spheriod ['fit_I'][:,wlIdx[nwav]],color =color_sph , lw = 2, ls = '--',label="fit sphrod")
        # axs[0, nwav].scatter(Spheriod ['sca_ang'][:,nwav][marker_indsp], Spheriod ['fit_I'][:,nwav][marker_indsp],color =color_sph , m = "o",label="fit sphrod")
        
        # axs[0, nwav].set_xlabel('Scattering angles (deg)')
        axs[0, 0].set_ylabel('I')
        # axs[0, nwav].legend()

        # Plot the fit and measured QoI data
        axs[2, nwav].plot(Spheriod ['sca_ang'][:,wlIdx[nwav]], Spheriod ['meas_P_rel'][:,wlIdx[nwav]],color = "k", lw = 1, label="meas")
        axs[2, nwav].plot(Spheriod ['sca_ang'][:,wlIdx[nwav]], Spheriod ['fit_P_rel'][:,wlIdx[nwav]], color =color_sph, ls = '--', label="fit sph")
        
        axs[2, nwav].fill_between(Spheriod ['sca_ang'][:,wlIdx[nwav]],(Spheriod ['meas_P_rel'][:,wlIdx[nwav]]), (Spheriod ['meas_P_rel'][:,wlIdx[nwav]])*1.03,color = 'r', alpha=0.2,ls = "--", label="+3%")
        axs[2, nwav].fill_between(Spheriod ['sca_ang'][:,wlIdx[nwav]],(Spheriod ['meas_P_rel'][:,wlIdx[nwav]]), (Spheriod ['meas_P_rel'][:,wlIdx[nwav]])*0.97,color = "b", alpha=0.2,ls = "--", label="-3%")
        # axs[2, nwav].set_xlabel('Scattering angles (deg)')
        axs[2, 0].set_ylabel('DOLP')
        axs[0, nwav].set_title(f"{wl[wlIdx[nwav]]}", fontsize = 14)
        
        axs[0, nwav].plot(Hex['sca_ang'][:,wlIdx[nwav]], Hex['fit_I'][:,wlIdx[nwav]],color =color_tamu , lw = 2, ls = "dashdot",label="fit Hex")
        axs[2, nwav].plot(Hex['sca_ang'][:,wlIdx[nwav]],Hex['fit_P_rel'][:,wlIdx[nwav]],color = color_tamu , lw = 2,ls = "dashdot", label = "fit Hex") 


        sphErr = 100 * abs(Spheriod['meas_I'][:,wlIdx[nwav]]-Spheriod ['fit_I'][:,wlIdx[nwav]] )/Spheriod['meas_I'][:,wlIdx[nwav]]
        HexErr = 100 * abs(Hex['meas_I'][:,nwav]-Hex['fit_I'][:,nwav] )/Hex['meas_I'][:,wlIdx[nwav]]
        
        axs[1, nwav].plot(Spheriod ['sca_ang'][:,wlIdx[nwav]], sphErr,color =color_sph ,label="Sphrod")
        axs[1, nwav].plot(Hex ['sca_ang'][:,wlIdx[nwav]], HexErr,color = color_tamu ,label="Hex")
        axs[1, 0].set_ylabel('Err I %')
        

        sphErrP = 100 * abs(Spheriod['meas_P_rel'][:,wlIdx[nwav]]-Spheriod ['fit_P_rel'][:,wlIdx[nwav]])
        HexErrP = 100 * abs(Hex['meas_P_rel'][:,wlIdx[nwav]]-Hex['fit_P_rel'][:,wlIdx[nwav]] )
        
        axs[3, nwav].plot(Spheriod ['sca_ang'][:,wlIdx[nwav]], sphErrP,color =color_sph ,label="Sphrod")
        axs[3, nwav].plot(Hex ['sca_ang'][:,wlIdx[nwav]], HexErrP,color =color_tamu ,label="Hex")
        
        axs[3, nwav].set_xlabel('Scattering angles (deg)')
        # axs[3, nwav].set_ylabel('Err P')
        # axs[3, nwav].legend()
        axs[3, nwav].set_xlabel('Scattering angles (deg)')
        axs[3, 0].set_ylabel('|Meas-fit|')
        # axs[1, nwav].set_title(f"{wl[nwav]}", fontsize = 14)

        # axs[0, 4].legend(bbox_to_anchor=(1.05, 0.5), loc='center left')
        # axs[1, 4].legend(bbox_to_anchor=(1.05, 0.5), loc='center left')
        # # plt.tight_layout()


for i in range(1):

    #Reading the ORACLE data for given pixel no, Tel_no = aggregated altitude
    #working pixels: 16800,16813 ,16814
    # RSP_PixNo = 2776  #4368  #Clear pixel 8/01 Pixel no of Lat,Lon that we are interested

    # RSP_PixNo = 13240
    RSP_PixNo = 13200
     #Dusty pixel on 9/22
    # PixNo = find_dust(file_path,file_name)[1][0]
    TelNo = 0 # aggregated altitude. To obtain geometries corresponding to data from the 1880 nm channel, aggregation altitude should be set to 1, while aggregation altitude =0 should be used for all other channels.
    nwl = 5 # first  nwl wavelengths
    ang1 = 10
    ang2 = 130 # :ang angles  #Remove

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
    # HSRLPixNo is the index of HSRL pixel taht corresponds to the RSP Lat Lon
    HSRLPixNo = FindPix(LatH,LonH,LatRSP,LonRSP)[0]  # Or can manually give the index of the pixel that you are intrested in
 
# #  Kernel_type = Run(Kernel_type) for spheriod, Kernel_type = 'TAMU' for hexahedral
#     rslts_Sph = RSP_Run("sphro",RSP_PixNo,ang1,ang2,TelNo,nwl)
#     rslts_Tamu = RSP_Run("TAMU",RSP_PixNo,ang1,ang2,TelNo,nwl)

    # print('Cost Value Sph, tamu: ',  rslts_Sph[0]['costVal'], rslts_Tamu[0]['costVal'])
    # RSP_plot(rslts_Sph,rslts_Tamu,RSP_PixNo)
    # plot_HSRL(rslts_Sph[0],rslts_Tamu[0], forward = False, retrieval = True, Createpdf = True,PdfName =f"/home/gregmi/ORACLES/rsltPdf/RSP_only_{RSP_PixNo}.pdf")
    
#     # Retrieval_type = 'NosaltStrictConst_final'
#     # #Running GRASP for HSRL, HSRL_sphrod = for spheriod kernels,HSRL_Tamu = Hexahedral kernels
    HSRL_sphrod = HSLR_run("sphro",HSRLfile_path,HSRLfile_name,nwl,HSRLPixNo,updateYaml= False,releaseYAML= True) 
    HSRL_Tamu = HSLR_run("TAMU",HSRLfile_path,HSRLfile_name, nwl,HSRLPixNo,updateYaml= False,releaseYAML= True)

    print('Cost Value Sph, tamu: ',  HSRL_sphrod[0][0]['costVal'],HSRL_Tamu[0][0]['costVal'])
   
#     # # #Constrining HSRL retrievals by 5% 
    HSRL_sphro_5 = HSLR_run("sphro",HSRLfile_path,HSRLfile_name,HSRLPixNo,updateYaml= True) 
    HSRL_Tamu_5 = HSLR_run("TAMU",HSRLfile_path,HSRLfile_name,HSRLPixNo,updateYaml= True)

    print('Cost Value Sph, tamu: ',  HSRL_sphro_5[0][0]['costVal'],HSRL_Tamu_5[0][0]['costVal'])

#     #Strictly Constrining HSRL retrievals to values from RSP retrievals
    HSRL_sphrod_strict = HSLR_run("sphro",HSRLfile_path,HSRLfile_name,HSRLPixNo,updateYaml= True, ConsType = 'strict') 
    HSRL_Tamu_strict = HSLR_run("TAMU",HSRLfile_path,HSRLfile_name,HSRLPixNo,updateYaml= True, ConsType = 'strict')

    print('Cost Value Sph, tamu: ',  HSRL_sphrod_strict[0][0]['costVal'],HSRL_Tamu_strict[0][0]['costVal'])
    
    #Lidar+pol combined retrieval
    LidarPolSph = LidarAndMAP('sphro',HSRLfile_path,HSRLfile_name,HSRLPixNo,file_path,file_name,RSP_PixNo,ang1,ang2,TelNo, nwl,GasAbsFn, updateYaml= None)
    LidarPolTAMU = LidarAndMAP('TAMU',HSRLfile_path,HSRLfile_name,HSRLPixNo,file_path,file_name,RSP_PixNo,ang1,ang2,TelNo, nwl,GasAbsFn, updateYaml= None)
    
    print('Cost Value Sph, tamu: ',  LidarPolSph[0]['costVal'],LidarPolTAMU[0]['costVal'])
    # print('SPH',"tam" )


    # %matplotlib inline
    # plot_HSRL(HSRL_sphrod[0][0],HSRL_Tamu[0][0], forward = True, retrieval = True, Createpdf = True,PdfName ="/home/gregmi/ORACLES/rsltPdf/HSRL_Only_Plots_444.pdf", combinedVal =HSRL_sphrod[2])
    # plot_HSRL(HSRL_sphro_5[0][0],HSRL_Tamu_5[0][0], forward = True, retrieval = True, Createpdf = True,PdfName ="/home/gregmi/ORACLES/rsltPdf/5%_HSRL_Plots_444.pdf", combinedVal =HSRL_sphrod[2])
    # plot_HSRL(HSRL_sphrod_strict[0][0],HSRL_Tamu_strict[0][0], forward = True, retrieval = True, Createpdf = True,PdfName ="/home/gregmi/ORACLES/rsltPdf/STR_HSRL_Plots_444.pdf", combinedVal =HSRL_sphrod[2])
    # plot_HSRL(LidarPolSph[0],LidarPolTAMU[0], forward = False, retrieval = True, Createpdf = True,PdfName ="/home/gregmi/ORACLES/rsltPdf/LIDARPOL_Plots_444.pdf")
    # plot_HSRL(LidarPolSph[0],LidarPolTAMU[0], forward = False, retrieval = True, Createpdf = True,PdfName ="/home/gregmi/ORACLES/rsltPdf/LIDARPOL_Plots_444.pdf")
    # CombinedLidarPolPlot(LidarPolSph,LidarPolTAMU)

# #Running GRASP for HSRL, HSRL_sphrod = for spheriod kernels,HSRL_Tamu = Hexahedral kernels
# HSRL_sphrod = HSLR_run("sphro",HSRLfile_path,HSRLfile_name,HSRLPixNo,updateYaml= False) 
# HSRL_Tamu = HSLR_run("TAMU",HSRLfile_path,HSRLfile_name,HSRLPixNo,updateYaml= False)
# #Con



# for i in  (np.arange(1,5,1)):
#     #Reading the ORACLE data for given pixel no, Tel_no = aggregated altitude
#     #working pixels: 16800,16813 ,16814
#     # RSP_PixNo = 2776  #4368  #Clear pixel 8/01 Pixel no of Lat,Lon that we are interested

#     # RSP_PixNo = 13240
#     RSP_PixNo = 13200+i
#      #Dusty pixel on 9/22
#     # PixNo = find_dust(file_path,file_name)[1][0]
#     TelNo = 0 # aggregated altitude. To obtain geometries corresponding to data from the 1880 nm channel, aggregation altitude should be set to 1, while aggregation altitude =0 should be used for all other channels.
#     nwl = 5 # first  nwl wavelengths
#     ang1 = 10
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
#     print( HSRLPixNo)
# #     # Retrieval_type = 'NosaltStrictConst_final'
# #     # #Running GRASP for HSRL, HSRL_sphrod = for spheriod kernels,HSRL_Tamu = Hexahedral kernels
#     HSRL_sphrod = HSLR_run("sphro",HSRLfile_path,HSRLfile_name,HSRLPixNo,updateYaml= False) 
#     # HSRL_Tamu = HSLR_run("TAMU",HSRLfile_path,HSRLfile_name,HSRLPixNo,updateYaml= False)
#     # %matplotlib inline
#     plot_HSRL(HSRL_sphrod[0][0],HSRL_Tamu[0][0], forward = True, retrieval = True, Createpdf = True,PdfName ="/home/gregmi/ORACLES/rsltPdf/HSRL_Only_Plots_444.pdf", combinedVal =HSRL_sphrod[2])
   

#     print('Cost Value Sph, tamu: ',  HSRL_sphrod[0][0]['costVal'],HSRL_Tamu[0][0]['costVal'])
   
# # #
# # def Angle_sensitivity(RSP_PixNo):
  
# #     ang1 = 10
# #     LIDARPOL = False

# #     plt.rcParams['font.size'] = '17'
# #     #Stokes Vectors Plot
# #     date_latlon = ['datetime', 'longitude', 'latitude']
# #     Xaxis = ['r','lambda','sca_ang','rv','height']
# #     Retrival = ['dVdlnr','aodMode','ssaMode','n', 'k']
# #     #['sigma', 'vol', 'aodMode','ssaMode', 'rEff', 'costVal']
# #     Angles =   ['sza', 'vis', 'fis','angle' ]
# #     Stokes =   ['meas_I', 'fit_I', 'meas_P_rel', 'fit_P_rel']
# #     Pij    = ['p11', 'p12', 'p22', 'p33'], 
# #     Lidar=  ['heightStd','g','LidarRatio','LidarDepol', 'gMode', 'LidarRatioMode', 'LidarDepolMode']

# #     # Plot the AOD data
# #     y = [0,1,2,0,1,2,]
# #     x = np.repeat((0,1),3)
# #     mode_v = ["fine", "coarse"]
# #     linestyle =[':', '-']

# #     cm_sp = ['#008080',"#C1E1C1" ]
# #     cm_t = ['#900C3F',"#FF5733" ]
# #     color_sph = '#0c7683'
# #     color_tamu = "#BC106F"

# #         #Retrivals:
# #     fig, axs = plt.subplots(nrows= 3, ncols=2, figsize=(35, 15))
# #     for i in range(5):
# #         ang2 = 150 - 10*i # :ang angles  #Remove
# #         Spheriod,Hex = RSP_scat[f'RSP_{ang1}_{ang2}'][0], RSP_scat[f'RSP_{ang1}_{ang2}'][0]
# #         for i in range(len(Retrival)):
# #             a,b = i%3,i%2
# #             for mode in range(Spheriod['r'].shape[0]): #for each modes
# #                 if i ==0:
# #                     axs[a,b].plot(Spheriod['r'][mode], Spheriod[Retrival[i]][mode], marker = "$O$",color = cm_sp[mode],ls = linestyle[mode], label=f"Sphrod_{mode_v[mode]}")
# #                     axs[a,b].plot(Hex['r'][mode],Hex[Retrival[i]][mode], marker = "H", color = cm_t[mode] , ls = linestyle[mode],label=f"Hex_{mode_v[mode]}")
# #                     axs[0,0].set_xlabel('Radius')
# #                     axs[0,0].set_xscale("log")
# #                 else:
# #                     axs[a,b].plot(Spheriod['lambda'], Spheriod[Retrival[i]][mode], marker = "$O$",color = cm_sp[mode],ls = linestyle[mode], label=f"Sphrod_{mode_v[mode]}")
# #                     axs[a,b].plot(Hex['lambda'],Hex[Retrival[i]][mode], marker = "H",color = cm_t[mode] , ls = linestyle[mode],label=f"Hex_{mode_v[mode]}")
# #                     axs[a,b].set_xticks(Spheriod['lambda'])
# #                     # axs[a,b].set_xticklabels(['0.41', '0.46', '0.55' , '0.67'  , '0.86'])
# #             axs[a,b].set_ylabel(f'{Retrival[i]}')
# #             axs[a,b].set_xlabel(r'$\lambda$')
            
# #             axs[2,1].plot(Spheriod['lambda'], Spheriod['aod'], marker = "$O$",color = cm_sp[mode],ls = linestyle[mode], label=f"Sphrod_{mode_v[mode]}")
# #             axs[2,1].plot(Hex['lambda'], Hex['aod'], marker = "H", color = cm_t[mode] , ls = linestyle[mode],label=f"Hex_{mode_v[mode]}")
# #             axs[2,1].set_xlabel(r'$\lambda$')
# #             axs[2,1].set_ylabel('Total AOD')
# #         axs[0,0].legend()
# #         lat_t = Hex['latitude']
# #         lon_t = Hex['longitude']
# #         dt_t = Hex['datetime']
# #         plt.suptitle(f'RSP Aerosol Retrieval \n  Lat:{lat_t} Lon :{lon_t}\n Date: {dt_t}')
# #         fig.savefig(f'/home/gregmi/ORACLES/HSRL_RSP/{dt_t}_RSPRetrieval.png', dpi = 400)

# #         #Stokes: 
# #         wl = rslts_Sph[0]['lambda'] 
# #     fig, axs = plt.subplots(nrows= 4, ncols=5, figsize=(30, 10),gridspec_kw={'height_ratios': [1, 0.3,1,0.3]}, sharex='col')
# #     for i in range(5):
# #         ang2 = 150 - 10*i # :ang angles  #Remove
# #         Spheriod,Hex = RSP_scat[f'RSP_{ang1}_{ang2}'][0], RSP_scat[f'RSP_{ang1}_{ang2}'][0]
        

# #         # Plot the AOD data
# #         meas_P_rel = 'meas_P_rel'

# #         if LIDARPOL == True:
# #             wlIdx = [1,2,4,5,6]
# #         else:
# #             wlIdx = np.arange(len(wl))


# #         for nwav in range(len(wlIdx)):
# #         # Plot the fit and measured I data
            
# #             axs[0, nwav].fill_between(Spheriod ['sca_ang'][:,wlIdx[nwav]],Spheriod ['meas_I'][:,wlIdx[nwav]], Spheriod ['meas_I'][:,wlIdx[nwav]]*1.03, color = 'r',alpha=0.2, ls = "--",label="+3%")
# #             axs[0, nwav].fill_between(Spheriod ['sca_ang'][:,wlIdx[nwav]],Spheriod ['meas_I'][:,wlIdx[nwav]], Spheriod ['meas_I'][:,wlIdx[nwav]]*0.97, color = "b",alpha=0.2, ls = "--",label="-3%")
# #             axs[0, nwav].plot(Spheriod['sca_ang'][:,wlIdx[nwav]], Spheriod['meas_I'][:,wlIdx[nwav]], color = "k", lw = 1, label="meas")

# #             axs[0, nwav].plot(Spheriod ['sca_ang'][:,wlIdx[nwav]], Spheriod ['fit_I'][:,wlIdx[nwav]],color =color_sph , lw = 2, ls = '--',label="fit sphrod")
# #             # axs[0, nwav].scatter(Spheriod ['sca_ang'][:,nwav][marker_indsp], Spheriod ['fit_I'][:,nwav][marker_indsp],color =color_sph , m = "o",label="fit sphrod")
            
# #             # axs[0, nwav].set_xlabel('Scattering angles (deg)')
# #             axs[0, 0].set_ylabel('I')
# #             # axs[0, nwav].legend()

# #             # Plot the fit and measured QoI data
# #             axs[2, nwav].plot(Spheriod ['sca_ang'][:,wlIdx[nwav]], Spheriod ['meas_P_rel'][:,wlIdx[nwav]],color = "k", lw = 1, label="meas")
# #             axs[2, nwav].plot(Spheriod ['sca_ang'][:,wlIdx[nwav]], Spheriod ['fit_P_rel'][:,wlIdx[nwav]], color =color_sph, ls = '--', label="fit sph")
            
# #             axs[2, nwav].fill_between(Spheriod ['sca_ang'][:,wlIdx[nwav]],(Spheriod ['meas_P_rel'][:,wlIdx[nwav]]), (Spheriod ['meas_P_rel'][:,wlIdx[nwav]])*1.03,color = 'r', alpha=0.2,ls = "--", label="+3%")
# #             axs[2, nwav].fill_between(Spheriod ['sca_ang'][:,wlIdx[nwav]],(Spheriod ['meas_P_rel'][:,wlIdx[nwav]]), (Spheriod ['meas_P_rel'][:,wlIdx[nwav]])*0.97,color = "b", alpha=0.2,ls = "--", label="-3%")
# #             # axs[2, nwav].set_xlabel('Scattering angles (deg)')
# #             axs[2, 0].set_ylabel('DOLP')
# #             axs[0, nwav].set_title(f"{wl[wlIdx[nwav]]}", fontsize = 14)
            
# #             axs[0, nwav].plot(Hex['sca_ang'][:,wlIdx[nwav]], Hex['fit_I'][:,wlIdx[nwav]],color =color_tamu , lw = 2, ls = "dashdot",label="fit Hex")
# #             axs[2, nwav].plot(Hex['sca_ang'][:,wlIdx[nwav]],Hex['fit_P_rel'][:,wlIdx[nwav]],color = color_tamu , lw = 2,ls = "dashdot", label = "fit Hex") 

# #             sphErr = 100 * abs(Spheriod['meas_I'][:,wlIdx[nwav]]-Spheriod ['fit_I'][:,wlIdx[nwav]] )/Spheriod['meas_I'][:,wlIdx[nwav]]
# #             HexErr = 100 * abs(Hex['meas_I'][:,nwav]-Hex['fit_I'][:,nwav] )/Hex['meas_I'][:,wlIdx[nwav]]
            
# #             axs[1, nwav].plot(Spheriod ['sca_ang'][:,wlIdx[nwav]], sphErr,color =color_sph ,label="Sphrod")
# #             axs[1, nwav].plot(Hex ['sca_ang'][:,wlIdx[nwav]], HexErr,color = color_tamu ,label="Hex")
# #             axs[1, 0].set_ylabel('Err I %')
            

# #             sphErrP = 100 * abs(Spheriod['meas_P_rel'][:,wlIdx[nwav]]-Spheriod ['fit_P_rel'][:,wlIdx[nwav]])
# #             HexErrP = 100 * abs(Hex['meas_P_rel'][:,wlIdx[nwav]]-Hex['fit_P_rel'][:,wlIdx[nwav]] )
            
# #             axs[3, nwav].plot(Spheriod ['sca_ang'][:,wlIdx[nwav]], sphErrP,color =color_sph ,label="Sphrod")
# #             axs[3, nwav].plot(Hex ['sca_ang'][:,wlIdx[nwav]], HexErrP,color =color_tamu ,label="Hex")
            
# #             axs[3, nwav].set_xlabel('Scattering angles (deg)')
# #             # axs[3, nwav].set_ylabel('Err P')
# #             # axs[3, nwav].legend()
# #             axs[3, nwav].set_xlabel('Scattering angles (deg)')
# #             axs[3, 0].set_ylabel('|Meas-fit|')
# #             # axs[1, nwav].set_title(f"{wl[nwav]}", fontsize = 14)

# #             # axs[0, 4].legend(bbox_to_anchor=(1.05, 0.5), loc='center left')
# #             # axs[1, 4].legend(bbox_to_anchor=(1.05, 0.5), loc='center left')
# #             # # plt.tight_layout()



# # def ReadGRASPtxt(Outputfn):


