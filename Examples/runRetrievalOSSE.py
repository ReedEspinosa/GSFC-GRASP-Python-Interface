#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" This script will run a retrieval simulation using OSSE results and the osseData class """

# import some standard modules
import os
import sys
import functools

# add GRASP_scripts, MADCAP_Scripts and ACCP subfolder to path (assumes GRASP_scripts and MADCAP_scripts are in the same parent folder)
parentDir = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) # obtain THIS_FILE_PATH/../ in POSIX
sys.path.append(parentDir) # that should be GRASP_Scripts – add it to Python path
grandParentDir = os.path.dirname(parentDir)# THIS_FILE_PATH/../../ in POSIX (this is folder that contains GRASP_scripts and MADCAP_scripts
sys.path.append(os.path.join(grandParentDir, "MADCAP_scripts"))
sys.path.append(os.path.join(grandParentDir, "MADCAP_scripts","ACCP_ArchitectureAndCanonicalCases"))

# import top level class that peforms the actual retrieval simulation; defined in THIS_FILE_PATH/../simulateRetrieval.py
import simulateRetrieval as rs

# import the class that is used to read Patricia's OSSE data; defined in ...MADCAP_scripts/readOSSEnetCDF.py
from readOSSEnetCDF import osseData

# import returnPixel and addError functions with instrument definitions from ...MADCAP_scripts/ACCP_ArchitectureAndCanonicalCases/architectureMap.py
from architectureMap import returnPixel, addError

from runGRASP import graspYAML

# define other paths not having to do with the python code itself
basePath = os.environ['NOBACKUP']
bckYAMLpath = os.path.join(basePath, 'MADCAP_scripts','ACCP_ArchitectureAndCanonicalCases','settings_BCK_POLAR_2modes.yml') # location of retrieval YAML file
dirGRASP = os.path.join(basePath, 'grasp_open/build/bin/grasp') # location of the GRASP binary to use for retrievals
krnlPath = os.path.join(basePath, 'local/share/grasp/kernels') # location of GRASP kernel files
osseDataPath = '/discover/nobackup/projects/gmao/osse2/pub/c1440_NR/OBS/A-CCP/' # base path for A-CCP OSSE data (contains gpm and ss450 folders)

# randomize initial guess in YAML file before retrieving
maxCPU = 2

# randomize initial guess in YAML file before retrieving
rndIntialGuess = False

# True to use 10k randomly chosen samples for defined month, or False to use specific day and hour defined below
random = True 

# point in time to pull OSSE data (if random==true then day and hour below can be ignored)
year = 2006
month = 8
day = 1
hour = 0

# simulated orbit to use – gpm OR ss450
orbit = 'ss450'

# filter 
maxSZA = 90

oceanOnly = False

# name of instrument as defined in MADCAP_Scripts/ACCP_ArchitectureAndCanonicalCases/architectureMap.py
archName = 'polar07'

# If true no noise will be added to simulated measurements, else noise is added according to architectureMap.py (or Ed's simulations for lidar data)
noiseFree = True 

# general version tag to distinguish runs
vrsn = 100

# wavelengths (μm); if we only want specific λ set it here, otherwise use all netCDF files found
wvls = [0.355, 0.36, 0.38, 0.41, 0.532, 0.55, 0.67, 0.87, 1.064, 1.55, 1.65] 

# specific pixels to run; set to None to run all pixels (likely very slow) 
# [NOTE: I think these indices correspond to the data after filtering for SZA, oceanOnly, etc. (Thus, changing filtering will change which number index goes with what pixel)] 
pixInd = [7375, 1444, 1359, 929, 4654, 6574, 2786, 6461, 6897, 2010] # SS Aug 2006

# save output here instead of within osseDataPath (None to disable)
customOutDir = os.path.join(basePath, 'synced', 'Working', 'OSSE_Test_Run') 

# create and instance of osseData with pixels from the specified date/time and extract the forward calculations in GRASP_scripts rslts dictionary format
od = osseData(osseDataPath, orbit, year, month, day, hour, random=random, wvls=wvls, 
              lidarVersion=None, maxSZA=maxSZA, oceanOnly=oceanOnly, loadDust=False, verbose=True)
fwdData = od.osse2graspRslts(pixInd=pixInd, newLayers=None)

# build file name to save the results
savePath = od.fpDict['savePath'] % (vrsn, 'example', archName)
if customOutDir: savePath = os.path.join(customOutDir, os.path.basename(savePath))
print('-- Running simulation for ' + os.path.basename(savePath) + ' --') # we print this because savePath has useful information about OSSE data

# NEEDED???
yamlObj = graspYAML(bckYAMLpath, newTmpFile=('BCK_n%d' % nn))

# Pull noise model for instrument defined above (if noiseFree==False)
radNoiseFun = None if noiseFree else functools.partial(addError, 'polar07')

# define a new instance corresponding to this architecture and run the retrievals
simA = rs.simulation() 
simA.runSim(fwdData, yamlObj, maxCPU=maxCPU, maxT=20, savePath=savePath, 
            binPathGRASP=dirGRASP, intrnlFileGRASP=krnlPath, releaseYAML=True, lightSave=True, 
            rndIntialGuess=rndIntialGuess, radianceNoiseFun=radNoiseFun,
            workingFileSave=True, dryRun=True, verbose=True)





