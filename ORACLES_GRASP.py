"""
# Greema Regmi, UMBC
# Date: Jan 31, 2023

This code reads Polarimetric data from the Campaigns and runs GRASP. This code was created to Validate the Aerosol retrivals performed using Non Spherical Kernels (Hexahedral from TAMU DUST 2020)

"""

import sys
from CreateRsltsDict import Read_Data_Oracles

from runGRASP import graspDB, graspRun, pixel, graspYAML
from matplotlib import pyplot as plt
import os
if os.uname()[1]=='uranus': plt.switch_backend('agg')
import numpy as np
import datetime as dt

import numpy as np
# import pandas as pd
from matplotlib import pyplot as plt
import h5py 
sys.path.append("/home/gregmi/git/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases")
from architectureMap import returnPixel

#Reading the ORACLES data

file_path = "/home/gregmi/ORACLES/RSP1-L1C_P3_20180922_R03"                    #Path to the file
file_name =  "/RSP1-P3_L1C-RSPCOL-CollocatedRadiances_20180922T151106Z_V003-20210421T233946Z.h5" #Name of the file
fwdModelYAMLpath = '/home/gregmi/git/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/settings_BCK_POLAR_2modes_Shape_ORACLE.yml'
binPathGRASP ='/home/shared/GRASP_GSFC/build_RSP/bin/grasp_app' #GRASP Executable
# binPathGRASP ='/home/shared/GRASP_GSFC/build_RSP_v112_noWarn/bin/grasp_app' #GRASP Executable
krnlPath='/home/shared/GRASP_GSFC/src/retrieval/internal_files'
savePath="/home/gregmi/ORACLES/RSP1-L1C_P3_20180922_R03"

#Reading the ORACLE data for given pixel no, Tel_no = aggregated altitude
PixNo = 16800 #Pixel no of Lat,Lon that we are interested
TelNo = 0 # aggregated altitude. Use 
nwl = 5 # first  nwl wavelengths
ang = 152 # :ang angles

#rslt is the GRASP rslt dictionary or contains GRASP Objects
rslt = Read_Data_Oracles(file_path,file_name,PixNo,TelNo,nwl,ang)


maxCPU = 3 #maximum CPU allocated to run GRASP on server
gRuns = []
yamlObj = graspYAML(baseYAMLpath=fwdModelYAMLpath)
#eventually have to adjust code for height, this works only for one pixel (single height value)
gRuns.append(graspRun(pathYAML=yamlObj, releaseYAML=True, orbHghtKM= rslt['height'] )) # This should copy to new YAML object


pix = pixel()
pix.populateFromRslt(rslt, radianceNoiseFun=None, dataStage='meas', verbose=False)
gRuns[-1].addPix(pix)

gDB = graspDB(graspRunObjs=gRuns, maxCPU=maxCPU)

#rslts contain all the results form the GRASP inverse run
rslts, failPix = gDB.processData(binPathGRASP=binPathGRASP, savePath=None, krnlPathGRASP=krnlPath)
















