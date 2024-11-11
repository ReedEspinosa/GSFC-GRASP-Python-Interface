#!/usr/bin/env python3

# Author: Nirandi Jayasinnghe
# Last Updated: Sep 3 2024

# import functions and packages
import numpy as np
from netCDF4 import Dataset 
import sys
from CreateRsltsDict import Read_HARP_CubeSat
import netCDF4 as nc
from runGRASP import graspDB, graspRun, pixel, graspYAML
from matplotlib import pyplot as plt
import os
import datetime as dt
import yaml


file_path = '/data/home/njayasinghe/LES/GSFC-GRASP-Python-Interface/CubeSat/'
filename = 'HARP.20200730T112721.L1C.4.4KM.0719.V01h.nc'

# Project: Aerosol Retrievals using Multi-Angular Polarimeters in the Twilight Zone 
# Author: Nirandi Jayasinghe
# Reading HARP CubeSat Data into rslt dict to create SDATA
# Last Edited: Sep 3th 2024
"""
ang: (1,nwl) shaped variable which give angle ranges for each wavelength if necessary
CMask: give path to CMASK array for corresponding HARP CubeSat data file. 
       This variable will later be converted as Cloud Masking command
"""
def Read_HARP_CubeSat(file_path,filename,XPX, YPX, CMask=None):
    
    #Reading Data
    Data = Dataset(file_path+filename, 'r')

    start_time = Data.time_coverage_start
    wl = np.array([Data.groups['blue'].central_wavelength_in_nm/1000, Data.groups['green'].central_wavelength_in_nm/1000, 
            Data.groups['red'].central_wavelength_in_nm/1000, Data.groups['nir'].central_wavelength_in_nm/1000])
    nwl = len(wl)

    Lat = Data.groups['Coordinates']['Latitude'][XPX,YPX].data
    Lon = Data.groups['Coordinates']['Longitude'][XPX,YPX].data

    VZA = []; SZA = []; SAA = []; VAA = []; Rel_Azi = []
    I = []; Q = []; U = []; DOLP = []; Scat_Ang = []
    bands = ['blue','green', 'red', 'nir']

    for band in bands:
        if (band != 'Coordinates'):

            if CMask != None:

                VZA.append(np.where((Data.groups[band]['View_Zenith'][:,XPX,YPX].mask == False) & (CMask[band][:,XPX,YPX]==0),Data.groups[band]['View_Zenith'][:,XPX,YPX].data,np.nan))
                SZA.append(np.where((Data.groups[band]['Solar_Zenith'][:,XPX,YPX].mask == False) & (CMask[band][:,XPX,YPX]==0),Data.groups[band]['Solar_Zenith'][:,XPX,YPX].data,np.nan))
                SAA.append(np.where((Data.groups[band]['Solar_Azimuth'][:,XPX,YPX].mask == False) & (CMask[band][:,XPX,YPX]==0),Data.groups[band]['Solar_Azimuth'][:,XPX,YPX].data,np.nan))
                VAA.append(np.where((Data.groups[band]['View_Azimuth'][:,XPX,YPX].mask == False) & (CMask[band][:,XPX,YPX]==0),Data.groups[band]['View_Azimuth'][:,XPX,YPX].data,np.nan))
                Rel_Azi.append(abs(np.where((Data.groups[band]['Solar_Azimuth'][:,XPX,YPX].mask == False) & (CMask[band][:,XPX,YPX]==0),Data.groups[band]['Solar_Azimuth'][:,XPX,YPX].data,np.nan) 
                        -np.where((Data.groups[band]['View_Azimuth'][:,XPX,YPX].mask == False) & (CMask[band][:,XPX,YPX]==0),Data.groups[band]['View_Azimuth'][:,XPX,YPX].data,np.nan)))
                Scat_Ang.append(np.rad2deg(np.arccos(np.sin(np.radians(VZA[-1]))*np.sin(np.radians(SZA[-1]))*np.cos(np.radians(Rel_Azi[-1]-np.pi))-np.cos(np.radians(VZA[-1]))*np.cos(np.radians(SZA[-1])))))

                I.append(np.where((Data.groups[band]['I'][:,XPX,YPX].mask == False) & (CMask[band][:,XPX,YPX]==0),Data.groups[band]['I'][:,XPX,YPX].data,np.nan))
                Q.append(np.where((Data.groups[band]['Q'][:,XPX,YPX].mask == False) & (CMask[band][:,XPX,YPX]==0),Data.groups[band]['Q'][:,XPX,YPX].data,np.nan))
                U.append(np.where((Data.groups[band]['U'][:,XPX,YPX].mask == False) & (CMask[band][:,XPX,YPX]==0),Data.groups[band]['U'][:,XPX,YPX].data,np.nan))
                DOLP.append(np.where((Data.groups[band]['DOLP'][:,XPX,YPX].mask == False) & (CMask[band][:,XPX,YPX]==0),Data.groups[band]['DOLP'][:,XPX,YPX].data,np.nan))

            else: 

                VZA.append(np.where(Data.groups[band]['View_Zenith'][:,XPX,YPX].mask == False,Data.groups[band]['View_Zenith'][:,XPX,YPX].data,np.nan))
                SZA.append(np.where(Data.groups[band]['Solar_Zenith'][:,XPX,YPX].mask == False,Data.groups[band]['Solar_Zenith'][:,XPX,YPX].data,np.nan))
                SAA.append(np.where(Data.groups[band]['Solar_Azimuth'][:,XPX,YPX].mask == False,Data.groups[band]['Solar_Azimuth'][:,XPX,YPX].data,np.nan))
                VAA.append(np.where(Data.groups[band]['View_Azimuth'][:,XPX,YPX].mask == False,Data.groups[band]['View_Azimuth'][:,XPX,YPX].data,np.nan))
                Rel_Azi.append(abs(np.where(Data.groups[band]['Solar_Azimuth'][:,XPX,YPX].mask == False,Data.groups[band]['Solar_Azimuth'][:,XPX,YPX].data,np.nan) 
                        -np.where(Data.groups[band]['View_Azimuth'][:,XPX,YPX].mask == False,Data.groups[band]['View_Azimuth'][:,XPX,YPX].data,np.nan)))
                Scat_Ang.append(np.rad2deg(np.arccos(np.sin(np.radians(VZA[-1]))*np.sin(np.radians(SZA[-1]))*np.cos(np.radians(Rel_Azi[-1]-np.pi))-np.cos(np.radians(VZA[-1]))*np.cos(np.radians(SZA[-1])))))

                I.append(np.where(Data.groups[band]['I'][:,XPX,YPX].mask == False,Data.groups[band]['I'][:,XPX,YPX].data,np.nan))
                Q.append(np.where(Data.groups[band]['Q'][:,XPX,YPX].mask == False,Data.groups[band]['Q'][:,XPX,YPX].data,np.nan))
                U.append(np.where(Data.groups[band]['U'][:,XPX,YPX].mask == False,Data.groups[band]['U'][:,XPX,YPX].data,np.nan))
                DOLP.append(np.where(Data.groups[band]['DOLP'][:,XPX,YPX].mask == False,Data.groups[band]['DOLP'][:,XPX,YPX].data,np.nan))

            for k in range(len(VZA[-1])-1):
                if ((I[-1][k]==np.nan) & (Q[-1][k]==np.nan) & (U[-1][k]==np.nan) & (DOLP[-1][k]==np.nan)) : 
                    del I[-1][k]; del Q[-1][k]; del U[-1][k]; del DOLP[-1][k]
                    del VZA[-1][k]; del SZA[-1][k]; del SAA[-1][k]; del VAA[-1][k]; del Rel_Azi[-1][k]

            

    vza = np.array([np.array(VZA[0]),np.array(VZA[1]),np.array(VZA[2]),np.array(VZA[3])],dtype=object)
    sza = np.array([np.array(SZA[0]),np.array(SZA[1]),np.array(SZA[2]),np.array(SZA[3])],dtype=object)
    rel_az = np.array([np.array(Rel_Azi[0]),np.array(Rel_Azi[1]),np.array(Rel_Azi[2]),np.array(Rel_Azi[3])],dtype=object)
    scat_ang = np.array([np.array(Scat_Ang[0]),np.array(Scat_Ang[1]),np.array(Scat_Ang[2]),np.array(Scat_Ang[3])],dtype=object)

    meas_I =np.array([np.array(I[0]),np.array(I[1]),np.array(I[2]),np.array(I[3])],dtype=object) 
    meas_Q =np.array([np.array(Q[0]),np.array(Q[1]),np.array(Q[2]),np.array(Q[3])],dtype=object)
    meas_U =np.array([np.array(U[0]),np.array(U[1]),np.array(U[2]),np.array(U[3])],dtype=object)
    
    rslt = {}

    rslt['lambda'] = wl
    rslt['longitude'] = Lon
    rslt['latitude'] = Lat
    rslt['meas_I'] = meas_I
    rslt['meas_Q'] = meas_Q
    rslt['meas_U'] = meas_U
    #rslt['meas_P'] = meas_DOLP[:10,:].reshape(len(meas_DOLP[:10,:]),nwl)

    yy = int(start_time.split('-')[0]); mm = int(start_time.split('-')[1]); dd = int(start_time.split('-')[-1].split(':')[0].split('T')[0])
    hh = int(start_time.split('-')[-1].split(':')[0].split('T')[-1]); mi = int(start_time.split('-')[-1].split(':')[1]); s = int(start_time.split('-')[-1].split(':')[-1].split('U')[0])

    rslt['datetime'] = dt.datetime(yy,mm,dd,hh,mi,s,00)

    rslt['sza'] = sza
    rslt['vis'] = vza
    rslt['sca_ang'] = scat_ang
    rslt['fis'] = rel_az
    rslt['land_prct'] = 0

    rslt['OBS_hght'] = 20000

    return rslt

def CubeSat_Run(file_path,filename,YMLpath,XPX,YPX,nwl,savePath=None): # for LES px numbers, #wavelengths, 1D/3D RT
        
    krnlPath =  '/data/home/njayasinghe/grasp_local_noRayleigh/GRASP_GSFC/src/retrieval/internal_files'
    binPathGRASP = '/data/home/njayasinghe/grasp_local_noRayleigh/GRASP_GSFC/build/bin/grasp_app'

    if savePath == None: 
        #savePath=f'/data/ESI/User/njayasinghe/LES_Retrievals/'+str(RT)+'D/LES_XP_'+str(XPX)+'_YP_'+str(YPX)
        #savePath=f'/data/home/njayasinghe/LES/GSFC-GRASP-Python-Interface/tmp/Corrt_I_only_30_ang_XPX_'+str(XPX)+'_YP_'+str(YPX)
        savePath=f'/data/home/njayasinghe/LES/GSFC-GRASP-Python-Interface/CubeSat/tmp/output_inv_'+str(XPX)+'_'+str(YPX)   

    #rslt is the GRASP rslt dictionary or contains GRASP Objects
    #rslt = Read_LES_Data(LES_file_path, LES_filename, XPX, YPX, nwl,RT,ang)
    rslt = Read_HARP_CubeSat(file_path, filename, XPX, YPX)
    print(rslt.keys())
    #print(rslt['OBS_hght'])

    maxCPU = 3 #maximum CPU allocated to run GRASP on server
    gRuns = []
    yamlObj = graspYAML(baseYAMLpath=YMLpath)
    #eventually have to adjust code for height, this works only for one pixel (single height value)
    gRuns.append(graspRun(pathYAML=yamlObj, releaseYAML=True )) # This should copy to new YAML object
    pix = pixel() # taking the px class from runGrasp.
    pix.populateFromRslt(rslt, radianceNoiseFun=None, dataStage='meas', verbose=False) #Creates SDATA
    gRuns[-1].addPix(pix)
    gDB = graspDB(graspRunObjs=gRuns, maxCPU=maxCPU) # creates GRASP object

    #rslts contain all the results form the GRASP inverse run
    rslts, failPix = gDB.processData(binPathGRASP=binPathGRASP, savePath=savePath, krnlPathGRASP=krnlPath) # Runs GRASP
    #return rslts


savepath = 'LES/GSFC-GRASP-Python-Interface/CubeSat/settings_CubeSat_inversion.yml'
file_path = '/data/home/njayasinghe/HARP_CubeSat/Data/'
filename = 'HARP.20200730T112721.L1C.4.4KM.0719.V01h.nc'
nwl = 4
pathYAML = '/data/home/njayasinghe/LES/GSFC-GRASP-Python-Interface/settings_LES_inversion.yml'
#'/data/home/njayasinghe/LES/GSFC-GRASP-Python-Interface/CubeSat/settings_CubeSat_inversion.yml'


for xp in range(75,76,1):
    for yp in range(0,168,1):

        CubeSat_Run(file_path,filename,pathYAML,xp,yp,nwl)