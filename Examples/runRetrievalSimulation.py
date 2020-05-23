#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" This script will run a retrieval simulations on user defined scenes and instruments """

# import some basic stuff
import os
import sys
import pprint

# add MADCAP_Scripts and ACCP subfolder to path, assuming GRASP_scripts is in parent directory of MADCAP_scripts
parentDir = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) # THIS_FILE_PATH/../ in POSIX
sys.path.append(parentDir) # add GRASP_Scripts to Python path
grandParentDir = os.path.dirname(parentDir)# THIS_FILE_PATH/../../ in POSIX
sys.path.append(os.path.join(grandParentDir, "MADCAP_Analysis")) # frequently called "MADCAP_Scripts"
sys.path.append(os.path.join(grandParentDir, "MADCAP_Analysis/ACCP_ArchitectureAndCanonicalCases"))

# import top level class that peforms the retrieval simulation, defined in THIS_FILE_PATH/../simulateRetrieval.py
import simulateRetrieval as rs

# import returnPixel function with instrument definitions from .../MADCAP_Analysis/ACCP_ArchitectureAndCanonicalCases/architectureMap.py 
from architectureMap import returnPixel

# import setupConCaseYAML function with simulated scene definitions from .../MADCAP_Analysis/ACCP_ArchitectureAndCanonicalCases/canonicalCaseMap.py 
from canonicalCaseMap import setupConCaseYAML


# <><><> BEGIN BASIC CONFIGURATION SETTINGS <><><>

# Full path to save simulation results as a Python pickle
savePath = '/Users/wrespino/Desktop/exampleSimulationTest.pkl'

# Full path grasp binary
binGRASP = '/Users/wrespino/Synced/Local_Code_MacBook/grasp_open/build/bin/grasp'

# Full path grasp precomputed single scattering kernels
krnlPath = '/Users/wrespino/Synced/Local_Code_MacBook/grasp_open/src/retrieval/internal_files'

# Directory containing the foward and inversion YAML files you would like to use
ymlDir = '/Users/wrespino/Synced/Local_Code_MacBook/MADCAP_Analysis/ACCP_ArchitectureAndCanonicalCases/'
fwdModelYAMLpath = os.path.join(ymlDir, 'settings_FWD_IQU_3lambda_POL.yml') # foward YAML file
bckYAMLpath = os.path.join(ymlDir, 'settings_BCK_POLAR_2modes.yml') # inversion YAML file

# Other non-path related settings
Nsims = 4 # the number of inversion to perform, each with its own random noise
maxCPU = 2 # the number of processes to launch, effectivly the # of CPU cores you want to dedicate to the simulation
conCase = 'case06a' # conanical case scene to run, case06a-k should work (see all defintions in setupConCaseYAML function)
SZA = 30 # solar zenith (Note GRASP doesn't seem to be wild about θs=0; θs=0.1 is fine though)
Phi = 0 # relative azimuth angle, φsolar-φsensor
τFactor = 1.0 # scaling factor for total AOD
instrument = 'polar0700' # polar0700 has (almost) no noise, polar07 has ΔI=3%, ΔDoLP=0.5%; see returnPixel function for more options

# <><><> END BASIC CONFIGURATION SETTINGS <><><>


# create a dummy pixel object, conveying the measurement geometry, wavlengths, etc. (i.e. information in a GRASP SDATA file)
nowPix = returnPixel(instrument, sza=SZA, relPhi=Phi, nowPix=None)

# generate a YAML file with the forward model "truth" state variable values for this simulated scene
cstmFwdYAML, landPrct = setupConCaseYAML(conCase, nowPix, fwdModelYAMLpath, caseLoadFctr=τFactor)

# land percentage for the scene is stored in the dummy pixel object, not the YAML file
nowPix.land_prct = landPrct

# Define a new instance of the simulation class for the instrument defined by nowPix (an instance of the pixel class)
simA = rs.simulation(nowPix) 

# run the simulation, see below the definition of runSIM in simulateRetrieval.py for more input argument explanations
gObjFwd, gObjBck = simA.runSim(cstmFwdYAML, bckYAMLpath, Nsims, maxCPU=maxCPU, savePath=savePath, \
            binPathGRASP=binGRASP, intrnlFileGRASP=krnlPath, releaseYAML=True, lightSave=True, \
            rndIntialGuess=False, dryRun=False, workingFileSave=True, verbose=True)
    
# print some results to the console/terminal
wavelengthIndex = 3
wavelengthValue = simA.rsltFwd[0]['lambda'][wavelengthIndex]
print('RMS deviations (retrieved-truth) at wavelength of %5.3f μm:' % wavelengthValue)
pprint.pprint(simA.analyzeSim(0)[0])

# save the some simulated truth data to a NetCDF file (this functionality needs expanding)
gObjFwd.output2netCDF(savePath[:-3] + '_truth.nc4')

