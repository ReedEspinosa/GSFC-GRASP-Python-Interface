"""
# Greema Regmi, UMBC
# Date: Jan 31, 2023

This code reads Polarimetric data from the Campaigns and runs GRASP. This code was created to Validate the Aerosol retrivals performed using Non Spherical Kernels (Hexahedral from TAMU DUST 2020)

"""

import sys
from CreateRsltsDict import Read_Data_RSP_Oracles
from CreateRsltsDict import Read_Data_HSRL_Oracles

from runGRASP import graspDB, graspRun, pixel, graspYAML
from matplotlib import pyplot as plt
import os
if os.uname()[1]=='uranus': plt.switch_backend('agg')
import numpy as np
import datetime as dt
%matplotlib inline
import numpy as np
# import pandas as pd
from matplotlib import pyplot as plt
import h5py 
sys.path.append("/home/gregmi/git/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases")
from architectureMap import returnPixel

# # RSP Case: August 1
# file_path = "/home/gregmi/ORACLES/RSP2-L1C_P3_20170801_R02"  #Path to the ORACLE data file
# file_name =  "/RSP2-P3_L1C-RSPCOL-CollocatedRadiances_20170801T130949Z_V002-20210624T034642Z.h5" #Name of the ORACLES file

# RSP  Case: Sep 22
file_path = "/home/gregmi/ORACLES/RSP1-L1C_P3_20180922_R03/"  #Path to the ORACLE data file
file_name =  "/RSP1-P3_L1C-RSPCOL-CollocatedRadiances_20180922T151106Z_V003-20210421T233946Z.h5" #Name of the ORACLES file

HSRLfile_path = "/home/gregmi/ORACLES/HSRL" #Path to the ORACLE data file 
HSRLfile_name =  "/HSRL2_P3_20170801_R1.h5" #Name of the ORACLES file



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

def RSP_Run(Kernel_type,PixNo,TelNo,nwl,ang): 
        
        # binPathGRASP ='/home/shared/GRASP_GSFC/build_RSP_v112_noWarn/bin/grasp_app' # New Version GRASP Executable  
        krnlPath='/home/shared/GRASP_GSFC/src/retrieval/internal_files'
        
        if Kernel_type == "sphro":
            fwdModelYAMLpath = '/home/gregmi/git/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/settings_BCK_POLAR_2modes_Shape_ORACLE.yml'
            # binPathGRASP ='/home/shared/GRASP_GSFC/build_RSP/bin/grasp_app' #GRASP Executable
            binPathGRASP ='/home/shared/GRASP_GSFC/build_RSP_v112/bin/grasp_app' 
            savePath=f"/home/gregmi/ORACLES/RSP1-L1C_P3_20180922_R03_{Kernel_type}"
        
        if Kernel_type == "TAMU":
            fwdModelYAMLpath = '/home/gregmi/git/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/settings_BCK_POLAR_2modes_Shape_ORACLE_tamu.yml'
            binPathGRASP ='/home/shared/GRASP_GSFC/build_megaharp01_mod/bin/grasp_app' #GRASP Executable
            # binPathGRASP ='/home/shared/GRASP_GSFC/build_RSP_v112_noWarn/bin/grasp_app' # New Version GRASP Executable  
            savePath=f"/home/gregmi/ORACLES/RSP1-L1C_P3_20180922_R03_{Kernel_type}"

    
        #rslt is the GRASP rslt dictionary or contains GRASP Objects

        
        rslt = Read_Data_RSP_Oracles(file_path,file_name,PixNo,TelNo,nwl,ang1,ang)

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
 
def HSLR_run(Kernel_type,HSRLfile_path,HSRLfile_name,PixNo,height):
        # binPathGRASP ='/home/shared/GRASP_GSFC/build_RSP_v112_noWarn/bin/grasp_app' # New Version GRASP Executable  
        krnlPath='/home/shared/GRASP_GSFC/src/retrieval/internal_files'
        
        if Kernel_type == "sphro":
            fwdModelYAMLpath = '/home/gregmi/git/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/settings_BCK_POLAR_2modes_Shape_ORACLE.yml'
            # binPathGRASP ='/home/shared/GRASP_GSFC/build_RSP/bin/grasp_app' #GRASP Executable
            binPathGRASP ='/home/shared/GRASP_GSFC/build_RSP_v112/bin/grasp_app' 
            savePath=f"/home/gregmi/ORACLES/RSP1-L1C_P3_20180922_R03_{Kernel_type}"
        
        if Kernel_type == "TAMU":
            fwdModelYAMLpath = '/home/gregmi/git/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/settings_BCK_POLAR_2modes_Shape_ORACLE_tamu.yml'
            binPathGRASP ='/home/shared/GRASP_GSFC/build_megaharp01_mod/bin/grasp_app' #GRASP Executable
            # binPathGRASP ='/home/shared/GRASP_GSFC/build_RSP_v112_noWarn/bin/grasp_app' # New Version GRASP Executable  
            savePath=f"/home/gregmi/ORACLES/RSP1-L1C_P3_20180922_R03_{Kernel_type}"

    
        #rslt is the GRASP rslt dictionary or contains GRASP Objects
        rslt = Read_Data_HSRL_Oracles(HSRLfile_path,HSRLfile_name,PixNo,height)

        maxCPU = 3 #maximum CPU allocated to run GRASP on server
        gRuns = []
        yamlObj = graspYAML(baseYAMLpath=fwdModelYAMLpath)
        #eventually have to adjust code for height, this works only for one pixel (single height value)
        gRuns.append(graspRun(pathYAML=yamlObj, releaseYAML=True )) # This should copy to new YAML object

        pix = pixel()
        pix.populateFromRslt(rslt, radianceNoiseFun=None, dataStage= None, verbose=False)
        gRuns[-1].addPix(pix)

        gDB = graspDB(graspRunObjs=gRuns, maxCPU=maxCPU)

        #rslts contain all the results form the GRASP inverse run
        rslts, failPix = gDB.processData(binPathGRASP=binPathGRASP, savePath=None, krnlPathGRASP=krnlPath)
        return rslts
    


PixNo = 10823  #Pixel no of Lat,Lon that we are interested
for i in range(5):
    fig,ax = plt.subplots(nrows = 3, ncols = 1, figsize =(8,15))
    #Reading the ORACLE data for given pixel no, Tel_no = aggregated altitude
    #working pixels: 16800,16813 ,16814

    PixNo = PixNo +2  #Pixel no of Lat,Lon that we are interested

    # PixNo = find_dust(file_path,file_name)[1][0]
    TelNo = 0 # aggregated altitude. To obtain geometries corresponding to data from the 1880 nm channel, aggregation altitude should be set to 1, while aggregation altitude =0 should be used for all other channels.
    nwl = 5 # first  nwl wavelengths
    ang1 = 0
    ang = 152 # :ang angles

    #  Kernel_type = Run(Kernel_type) for spheriod, Kernel_type = 'TAMU' for hexahedral
    rslts_Sph = RSP_Run("sphro",PixNo,TelNo,nwl,ang)
    rslts_Tamu = RSP_Run("TAMU",PixNo,TelNo,nwl,ang)
  
 

        # Define the data to plot
    AOD_sph = rslts_Sph[0]['aod']
    AOD_Tamu = rslts_Tamu[0]['aod']
    wavelen = rslts_Sph[0]['lambda']
    factor_tamu = 1
    nwav = 0

    # Set up the plot
    fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(12, 8))

    # Plot the AOD data
    axs[0, 0].plot(wavelen, AOD_sph, marker = "$O$",label="sph")
    axs[0, 0].plot(wavelen, AOD_Tamu, marker = "H", label="tamu")
    axs[0, 0].set_xlabel('Wavelength (nm)')
    axs[0, 0].set_ylabel('AOD')
    axs[0, 0].legend()

    # Plot the fit and measured I data
    axs[0, 1].plot(rslts_Sph[0]['sca_ang'][:,nwav], rslts_Sph[0]['fit_I'][:,nwav], label="fit sph")
    axs[0, 1].plot(rslts_Sph[0]['sca_ang'][:,nwav], rslts_Sph[0]['meas_I'][:,nwav], label="meas sph")
    axs[0, 1].plot(rslts_Tamu[0]['sca_ang'][:,nwav], rslts_Tamu [0]['fit_I'][:,nwav], label="tamu")
    axs[0, 1].plot(rslts_Tamu[0]['sca_ang'][:,nwav], rslts_Tamu[0]['meas_I'][:,nwav])
    axs[0, 1].set_xlabel('Scattering angles (deg)')
    axs[0, 1].set_ylabel('I')
    axs[0, 1].legend()

    # Plot the fit and measured QoI data
    axs[1, 0].plot(rslts_Sph[0]['sca_ang'][:,nwav], rslts_Sph[0]['fit_QoI'][:,nwav], label="fit sph")
    axs[1, 0].plot(rslts_Sph[0]['sca_ang'][:,nwav], rslts_Sph[0]['meas_QoI'][:,nwav], label="meas sph")
    axs[1, 0].plot(rslts_Tamu[0]['sca_ang'][:,nwav],rslts_Tamu [0]['fit_QoI'][:,nwav], label = "tamu") 
    axs[1, 0].set_xlabel('Scattering angles (deg)')
    axs[1, 0].set_ylabel('QoI')
    axs[1, 0].legend()
    
    
    # Plot the fit and measured UoI data
    axs[1,1].plot(rslts_Sph[0]['sca_ang'][:,nwav],rslts_Sph[0]['fit_UoI'][:,nwav], label = "fir sph")
    axs[1,1].plot(rslts_Sph[0]['sca_ang'][:,nwav],rslts_Sph[0]['meas_UoI'][:,nwav],label = "meas sph")
    axs[1,1].plot(rslts_Tamu[0]['sca_ang'][:,nwav],rslts_Tamu [0]['fit_UoI'][:,nwav], label = "tamu")
    axs[1,1].plot(rslts_Tamu[0]['sca_ang'][:,nwav],rslts_Tamu[0]['meas_UoI'][:,nwav])
    axs[1,1].set_xlabel('Scattering angles (deg)')
    axs[1,1].set_ylabel('UoI')
    axs[1,1].legend()


    # axs[2,0].plot(rslts_Sph[0]['sca_ang'][:,nwav],rslts_Sph[0]['DoLP'][:,nwav], label = "sph")
    # axs[2,0].plot(rslts_Sph[0]['sca_ang'][:,nwav],rslts_Tamu [0]['DoLP'][:,nwav], label = "tamu")
    # axs[2,0].legend()
    # axs[2,0].set_xlabel('Scattering angles')
    # axs[2,0].set_ylabel('DoLP')

    plt.suptitle(f'/home/gregmi/ORACLES/{file_name}_{PixNo}')

fig.savefig(f'/home/gregmi/ORACLES/{file_name}_{PixNo}.png')




    # height = 10
    
    # PixNo = 16
    # #Plotting the results:
    # HSRL_rslt = HSLR_run("sphro",file_path,file_name,PixNo,height)

