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


# Runing GRASP for LES Data
LES_file_path = '/data/home/njayasinghe/LES/Data/Test/' #LES file_path
LES_filename = 'dharma_TCu_001620_SZA40_SPHI30_wvl0.669_NOAERO-3.nc' #LES filename
# files are written to tmp folder in NYX node

def LES_Run(LES_file_path,LES_filename,XPX,YPX,nwl,RT): # for LES px numbers, #wavelengths, 1D/3D RT
        
    krnlPath='/data/home/njayasinghe/grasp/src/retrieval/internal_files'
    #Base YAML template
    fwdModelYAMLpath = '/data/home/njayasinghe/LES/GSFC-GRASP-Python-Interface/settings_LES_inversion.yml'
    binPathGRASP = '/data/home/njayasinghe/grasp/build/bin/grasp_app'
    savePath=f'/data/ESI/User/njayasinghe/LES_Retrievals/'+str(RT)+'D/LES_XP_'+str(XPX)+'_YP_'+str(YPX)

    #rslt is the GRASP rslt dictionary or contains GRASP Objects
    rslt = Read_LES_Data(LES_file_path, LES_filename, XPX, YPX, nwl,RT)
    #print(rslt['OBS_hght'])

    maxCPU = 3 #maximum CPU allocated to run GRASP on server
    gRuns = []
    yamlObj = graspYAML(baseYAMLpath=fwdModelYAMLpath)
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
    for xp in range(0,168,1):
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

    
    


