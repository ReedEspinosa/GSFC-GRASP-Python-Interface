#!/usr/bin/env python3
# -*- coding: utf-8 -*-

discover = True 

import numpy as np
import runGRASP as rg
from netCDF4 import Dataset
import sys
if discover:
    syncPath = '/discover/nobackup/wrespino/synced/'
    sys.path.append("/discover/nobackup/wrespino/MADCAP_scripts")
else:
    syncPath = '/Users/wrespino/Synced/'
    sys.path.append("/Users/wrespino/Synced/Local_Code_MacBook/MADCAP_Analysis")
from MADCAP_functions import loadVARSnetCDF



netCDFpath = syncPath+'Remote_Sensing_Projects/MADCAP_CAPER/newOpticsTables/LUT-DUST/optics_DU.v15_6.nc' # just need for lambda's and dummy ext
savePath_netCDF = syncPath+'Remote_Sensing_Projects/MADCAP_CAPER/newOpticsTables/LUT-DUST/GRASP_LUT-DUST_V3.nc'
#loadPath_pkl = syncPath+'Remote_Sensing_Projects/MADCAP_CAPER/newOpticsTables/LUT-DUST/GRASP_LUT-DUST_V3.pkl'
loadPath_pkl = None

maxL = 9 # max number of wavelengths in a single GRASP run
Nangles = 181 # determined by GRASP kernels

# if loadPath_pkl is None grasp results is generated (not loaded) and the following are required:
binPathGRASP = '/discover/nobackup/wrespino/grasp_open/build/bin/grasp' if discover else '/usr/local/bin/grasp' # currently has lambda checks disabled
YAMLpath = syncPath+'Remote_Sensing_Projects/MADCAP_CAPER/newOpticsTables/LUT-DUST/settings_BCK_ExtSca_9lambda.yml'
lgnrmfld = 'retrieval.constraints.characteristic[2].mode[1].initial_guess.value'
RRIfld =   'retrieval.constraints.characteristic[3].mode[1].initial_guess.value' # should match setting in YAML file
IRIfld =   'retrieval.constraints.characteristic[4].mode[1].initial_guess.value'
maxCPU = 28 if discover else 3
#                   rv      sigma [currently: MADCAP DUST-LUT V3 (fitting total ext, not just spectral dependnece)]
szVars = np.array([[0.7145, 0.3281],
                   [1.2436, 0.2500],
                   [2.2969, 0.3439],
                   [5.3621, 0.7064],
                   [9.9686, 0.7271]])
nBnds = [1.301, 1.699] # taken from netCDF but forced to these bounds
kBnds = [1e-8, 0.499]


# DUMMY VALUES
msTyp = [12] 
nbvm = np.ones(len(msTyp))
thtv = np.zeros(len(msTyp))
phi = np.zeros(len(msTyp)) 
sza = 0

loaddVarNames = ['lambda', 'qext', 'qsca', 'refreal', 'refimag']
optTbl = loadVARSnetCDF(netCDFpath, loaddVarNames)
wvls = optTbl['lambda']*1e6
gspRun = []
Nlambda = len(wvls)
lEdg = np.r_[0:Nlambda:maxL]
Nbin = szVars.shape[0]
if loadPath_pkl: # only write previous calculations to netCDF
    rslts = rg.graspDB().loadResults(loadPath_pkl)
else: # perform calculations
    for bn in range(Nbin):
        for lstrt in lEdg:
            wvlsNow = wvls[lstrt:min(lstrt+maxL,Nlambda)]
            wvInds = np.r_[lstrt:min(lstrt+maxL,Nlambda)]
            gspRunNow = rg.graspRun(YAMLpath)    
            nowPix = rg.pixel(730123.0+bn, 1, 1, 0, 0, 0, 100)
            for wvl, wvInd in zip(wvlsNow, wvInds): # This will be expanded for wavelength dependent measurement types/geometry
                meas = np.r_[optTbl['qext'][bn,0,wvInd]]
                nowPix.addMeas(wvl, msTyp, nbvm, sza, thtv, phi, meas)
            n = optTbl['refreal'][bn,0,wvInds]
            n = np.minimum(n, nBnds[1])
            n = np.maximum(n, nBnds[0])
            k = -optTbl['refimag'][bn,0,wvInds]
            k = np.minimum(k, kBnds[1])
            k = np.maximum(k, kBnds[0])
            gspRunNow.yamlObj.access(lgnrmfld, szVars[bn])
            gspRunNow.yamlObj.access(RRIfld, n) 
            gspRunNow.yamlObj.access(IRIfld, k)
            gspRunNow.addPix(nowPix)
            gspRun.append(gspRunNow)
    gDB = rg.graspDB(gspRun, maxCPU)
    rslts = gDB.processData(savePath=savePath_netCDF[0:-2]+'pkl', binPathGRASP=binPathGRASP)

root_grp = Dataset(savePath_netCDF, 'w', format='NETCDF4')
#root_grp = Dataset('/Users/wrespino/Desktop/netCDF_TEST.nc', 'w', format='NETCDF4')
root_grp.description = 'Single scattering properties of dust bins derived with GRASP'

# dimensions
root_grp.createDimension('sizeBin', Nbin)
root_grp.createDimension('lambda', Nlambda)
root_grp.createDimension('angle', Nangles)

# variables
sizeBin = root_grp.createVariable('sizeBin', 'u1', ('sizeBin'))
sizeBin.units = 'none'
sizeBin.long_name = 'dust size bin index'
wavelength = root_grp.createVariable('lambda', 'f4', ('lambda'))
wavelength.units = 'um'
wavelength.long_name = 'wavelength'
angle = root_grp.createVariable('angle', 'f4', ('angle'))
angle.units = 'degree'
angle.long_name = 'scattering angle'
PMvarNms = ['p11', 'p12', 'p22', 'p33', 'p34', 'p44']
PMvars = dict()
for varNm in PMvarNms:
    PMvars[varNm] = root_grp.createVariable(varNm, 'f8', ('sizeBin', 'lambda', 'angle'))
    PMvars[varNm].units = 'sr-1'
    PMvars[varNm].long_name = varNm + ' phase matrix element' 
bext_vol = root_grp.createVariable('bext_vol', 'f8', ('sizeBin', 'lambda'))
bext_vol.units = 'm2·m-3 '
bext_vol.long_name = 'volume extinction efficiency'
bext_vol.description = "The extinction cross section per unit of particle volume.\
 This is also the extinction coefficient per unit of volume concentration.\
 bext_vol x ρ = βext where ρ is the particle density and βext is the mass extinction efficiency."
bsca_vol = root_grp.createVariable('bsca_vol', 'f8', ('sizeBin', 'lambda'))
bsca_vol.units = 'm2·m-3'
bsca_vol.long_name = 'volume scattering efficiency'
bsca_vol.description = "The scattering cross section per unit of particle volume.\
 This is also the scattering coefficient per unit of volume concentration.\
 bsca_vol x ρ = βsca where ρ is the particle density and βsca is the mass scattering efficiency."
refreal = root_grp.createVariable('refreal', 'f8', ('sizeBin', 'lambda'))
refreal.units = 'none'
refreal.long_name = 'real refractive index'
reafimag = root_grp.createVariable('reafimag', 'f8', ('sizeBin', 'lambda'))
reafimag.units = 'none'
reafimag.long_name = 'imaginary refractive index'
rv = root_grp.createVariable('rv', 'f8', ('sizeBin'))
rv.units = 'um'
rv.long_name = 'lognormal volume median radius'
sigma = root_grp.createVariable('sigma', 'f8', ('sizeBin'))
sigma.units = 'none'
sigma.long_name = 'lognormal sigma'
sph = root_grp.createVariable('sph', 'f8', ('sizeBin'))
sph.units = 'none'
sph.long_name = 'fraction of spherical particles'

# data
sizeBin[:] = np.r_[0:Nbin]
wavelength[:] = wvls[0:Nlambda]
angle[:] = np.linspace(0, 180, Nangles)
for i,rslt in enumerate(rslts):
    binInd = i//len(lEdg)
    lEdgInd = i%len(lEdg)
    if lEdgInd == 0:
        rv[binInd] = rslt['rv'][0]
        sigma[binInd] = rslt['sigma'][0]
        sph[binInd] = np.atleast_1d(rslt['sph'])[0]
    lInd = np.r_[lEdg[lEdgInd]:min(lEdg[lEdgInd]+maxL,Nlambda)]
    bext_vol[binInd, lInd] = rslt['aod']/rslt['vol'][0]
    bsca_vol[binInd, lInd] = rslt['ssa']*rslt['aod']/rslt['vol'][0]
    refreal[binInd, lInd] = rslt['n']  # HINT: WE DID NOT HAVE MODE SPECIFIC REF IND IN YAML
    reafimag[binInd, lInd] = rslt['k']
    for varNm in PMvarNms:
        for i,l in enumerate(lInd):
            PMvars[varNm][binInd, l, :] = rslt[varNm][:,0,i]
root_grp.close()


# SANITY CHECK PLOT
if discover: sys.exit()
from matplotlib import pyplot as plt
loaddVarNames = ['lambda', 'bext_vol', 'bsca_vol']
optTblNew = loadVARSnetCDF(savePath_netCDF, loaddVarNames)
fitInd = np.r_[2,4,5,6,7,8,9,11,13]
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(7, 3.5))
Vars = ['ext_vol', 'sca_vol']
for i, vnm in enumerate(Vars):
    vnmq = 'q'+vnm[:-4]
    vnmb = 'b'+vnm
    ax[i].plot(optTbl['lambda']*1e6, optTbl['q'+vnm[:-4]][:,0,:].T, '--')
    ax[i].set_prop_cycle(None)
    Scl = optTblNew['b'+vnm][:,8]/optTbl['q'+vnm[:-4]][:,0,8]
    print(Scl)
    ax[i].plot(optTblNew['lambda'], optTblNew['b'+vnm].T/Scl)
    ax[i].set_xlim([0.3, 3.0])
    ax[i].set_xlabel('wavelength')   
    ax[i].set_ylabel('$q_{' + vnm[:-4] + '}$')
ax[0].legend(['Mode %d' % int(x+1) for x in range(optTblNew['b'+vnm].shape[0])])
ax[0].set_prop_cycle(None)
ax[0].plot(optTblNew['lambda'][fitInd], np.atleast_2d(optTbl['qext'][:,0,fitInd]).T, 'x')
plt.suptitle('Orignal (dashed), target point (crosses), and new (solid) spectral dependences', fontsize=12)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])