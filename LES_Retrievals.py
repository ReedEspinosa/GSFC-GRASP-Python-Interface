#!/usr/bin/env python3

# Author: Nirandi Jayasinnghe
# Last Updated: Aug 14 2024

""" This script is created to run GRASP retrievals on LES scene from 1D /3D RT data."""

# import functions and packages
import sys
from CreateRsltsDict import Read_LES_Data
import netCDF4 as nc
from runGRASP import graspDB, graspRun, pixel, graspYAML
from matplotlib import pyplot as plt
import os
import numpy as np
import datetime as dt
import yaml
from netCDF4 import Dataset


# Runing GRASP for LES Data
LES_file_path = '/data/home/njayasinghe/LES/Data/Test/' #LES file_path
LES_filename = 'dharma_TCu_001620_SZA40_SPHI30_wvl0.669_NOAERO-4.nc' #LES filename

# Experiment LES Data
LES_filename = 'dharma_TCu_001620_SZA40_SPHI30_wvl0.669_NOAERO_SingleColumn.nc'

#YML file path
invModelYAMLpath = '/data/home/njayasinghe/LES/GSFC-GRASP-Python-Interface/settings_LES_inversion.yml'
fwdModelYAMLpath = '/data/home/njayasinghe/LES/GSFC-GRASP-Python-Interface/settings_LES_forward.yml'
#YMLpath = invModelYAMLpath 
YMLpath = fwdModelYAMLpath
# files are written to tmp folder in NYX node

def Read_LES_Exp(file_path, filename, nwl,RT,ang=10):

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

    exp = 1

    if (ang > 1):
        #Assuming angles for LES data are defined similar to GRASP definition.
        #view zenith angle
        vza = data["View_Zenith_Angle"].data[::ang]
        #scatteting angle
        scat_ang = data["scat_ang"].data[::ang]
        #solar zenith angle
        sza = data["Solar_Zenith_Angle"].data[::ang]
        #solar azimuth angle
        saa = data["Solar_Azimuth_Angle"].data[::ang]
        #view azimuth angle
        vaa = data["View_Azimuth_Angle"].data[::ang]

        #converting angles to radians to calculate relative azimuth angles
        SZA = np.radians(sza)
        VZA = np.radians(vza)
        SAA = np.radians(saa)
        VAA = np.radians(vaa)

        Rel_Azimuth = (180/np.pi)*(np.arccos((np.cos((scat_ang *np.pi)/180) + np.cos(SZA)*np.cos(VZA))/(- np.sin(SZA)*np.sin(VZA))))

        if (RT==1):
            # Dimentions:[nVZA,nwl,SZA,Stoke,nX,nY]
            meas_I = data["R_I"].data[::ang,exp]*np.cos(np.radians(sza))#*np.cos(np.radians(vza))
            meas_Q = data["R_Q"].data[::ang,exp]*np.cos(np.radians(sza))#*np.cos(np.radians(vza))
            meas_U = data["R_U"].data[::ang,exp]*np.cos(np.radians(sza))#*np.cos(np.radians(vza))

            if (exp == 3): 
                DOLP = np.sqrt(meas_Q*meas_Q+meas_U*meas_U)
            else: 
                DOLP = np.sqrt(meas_Q*meas_Q+meas_U*meas_U)/meas_I

    
    else:

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
            meas_I = data["R_I"].data[:,exp]*np.cos(np.radians(sza))
            meas_Q = data["R_Q"].data[:,exp]*np.cos(np.radians(sza))
            meas_U = data["R_U"].data[:,exp]*np.cos(np.radians(sza))

            if (exp == 3):
                DOLP = np.sqrt(meas_Q*meas_Q+meas_U*meas_U)
            else: 
                DOLP = np.sqrt(meas_Q*meas_Q+meas_U*meas_U)/meas_I


    #creating Results Dictionary for GRASP

    rslt = {}

    rslt['lambda'] = wl #wavelength is in um
    rslt['longitude'] = Lon
    rslt['latitude'] = Lat
    rslt['meas_I'] = np.repeat(meas_I,nwl).reshape(len(meas_I),nwl)
    #rslt['meas_Q'] = meas_Q
    #rslt['meas_U'] = meas_U
    rslt['meas_P'] = np.repeat(DOLP,nwl).reshape(len(DOLP),nwl)
    #plt.plot(DOLP)

    yy = 2024 ; mm = 8 ; dd = 8 ; hh = 12 ; mi = 00 ; s = 00; ms = 00 #dummy values
    rslt['datetime'] = dt.datetime(yy,mm,dd,hh,mi,s,ms)
  
    # All the geometry arrays should be 2D, (angle, wl) --> solar zenith angle, wavelength
    rslt['sza'] = np.repeat(sza,nwl).reshape(len(sza),nwl)
    rslt['vis'] = np.repeat(vza,nwl).reshape(len(vza),nwl)
    rslt['sca_ang'] = np.repeat(scat_ang,nwl).reshape(len(scat_ang),nwl)
    rslt['fis'] = np.repeat(Rel_Azimuth,nwl).reshape(len(Rel_Azimuth),nwl)
    rslt['land_prct'] = 0 #over ocean 

    rslt['OBS_hght'] = data['Sensor_Position'].data*1000 #sensor position is 20 km for LES data

    print(rslt.keys())

    return rslt


def LES_Run(LES_file_path,LES_filename,YMLpath,XPX,YPX,nwl,RT,ang=10,savePath=None): # for LES px numbers, #wavelengths, 1D/3D RT
        
    #krnlPath='/data/home/njayasinghe/grasp/src/retrieval/internal_files'
    #binPathGRASP = '/data/home/njayasinghe/grasp/build/bin/grasp_app'
    # no Rayleigh bin path
    krnlPath =  '/data/home/njayasinghe/grasp_local_noRayleigh/GRASP_GSFC/src/retrieval/internal_files'
    binPathGRASP = '/data/home/njayasinghe/grasp_local_noRayleigh/GRASP_GSFC/build/bin/grasp_app'

    if savePath == None: 
        #savePath=f'/data/ESI/User/njayasinghe/LES_Retrievals/'+str(RT)+'D/LES_XP_'+str(XPX)+'_YP_'+str(YPX)
        #savePath=f'/data/home/njayasinghe/LES/GSFC-GRASP-Python-Interface/tmp/Corrt_I_only_30_ang_XPX_'+str(XPX)+'_YP_'+str(YPX)
        savePath=f'/data/home/njayasinghe/LES/GSFC-GRASP-Python-Interface/tmp/Exp_1_I_DOLP_30_ang_XPX_30_YP_30_FWD_al'   

    #rslt is the GRASP rslt dictionary or contains GRASP Objects
    #rslt = Read_LES_Data(LES_file_path, LES_filename, XPX, YPX, nwl,RT,ang)
    rslt = Read_LES_Exp(LES_file_path, LES_filename, nwl,RT,ang)
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


def Run_retrievals(nwl,RT):
    for xp in range(75,76,1):
        for yp in range(0,168,1):

            LES_Run(LES_file_path,LES_filename,xp,yp,nwl,RT)



if __name__ == "__main__":

    from datetime import datetime
    import argparse

    #--------------------------------------------------------------=---
    #-- 1. Command-line arguments/options with argparse
    #--------------------------------------------------------------=---
    parser = argparse.ArgumentParser(
                formatter_class=argparse.RawTextHelpFormatter,
                description='HARP2 L1B Quicklook Processor',
                epilog="""
    EXIT Status:
        0   : All is well in the world
        1   : Dunno, something horrible occurred
                """)

    #-- required arguments
    parser.add_argument('--nwl',  type=int, required=True, help='number of wavelengths')
    parser.add_argument('--RT', type=int, required=True, help='1D or 3D RT')

    args = parser.parse_args()

    Run_retrievals(args.nwl,args.RT)

    
    


