#!/usr/bin/env python3
# -*- coding: utf-8 -*-
 
import numpy as np
import runGRASP as rg
from netCDF4 import Dataset
import sys
sys.path.append("/Users/wrespino/Synced/Local_Code_MacBook/MADCAP_Analysis")
from MADCAP_functions import loadVARSnetCDF
from matplotlib import pyplot as plt

binPathGRASP = '/Users/wrespino/Synced/Local_Code_MacBook/grasp_open/build/bin/grasp' # currently has lambda checks disabled
YAMLpath = '/Users/wrespino/Synced/Remote_Sensing_Projects/MADCAP_CAPER/newOpticsTables/LUT-DUST/settings_BCK_ExtSca_8lambda.yml'
netCDFpath = '/Users/wrespino/Synced/Remote_Sensing_Projects/MADCAP_CAPER/newOpticsTables/LUT-DUST/optics_DU.v15_6.nc' # just need for lambda's and dummy ext
savePath_netCDF = '/Users/wrespino/Synced/Remote_Sensing_Projects/MADCAP_CAPER/newOpticsTables/LUT-DUST/GRASP_LUT-DUST_V1.nc'
lgnrmfld = 'retrieval.constraints.characteristic[2].mode[1].initial_guess.value'
RRIfld =   'retrieval.constraints.characteristic[3].mode[1].initial_guess.value' # should match setting in YAML file
IRIfld =   'retrieval.constraints.characteristic[4].mode[1].initial_guess.value'

maxCPU = 3
unbound = False # each bin has its own YAML (YAMLpath[:-4:]+'_mode%d' % (bn+1)+'.yml')
maxL = 8 # max number of wavelengths in a single GRASP run
Nbin = 5 # number of bins, should be YAML file for each (YAMLpath[:-4:]+'_mode%d' % (bn+1)+'.yml')
Nangles = 181 # determined by GRASP kernels

#                   rv      sigma
szVars = np.array([[0.8125, 0.4236],
                   [1.4257, 0.2730],
                   [2.0688, 0.3232],
                   [3.0000, 0.5841],
                   [6.0249, 0.7479]])
nBnds = [1.301, 1.699]
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
#Nlambda = 18 # HACK
lEdg = np.r_[0:Nlambda:maxL]
for bn in range(Nbin):
    for lstrt in lEdg:
        wvlsNow = wvls[lstrt:lstrt+maxL]
        wvInds = np.r_[lstrt:lstrt+maxL]
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
bext = root_grp.createVariable('bext', 'f8', ('sizeBin', 'lambda'))
bext.units = 'm-1'
bext.long_name = 'extinction coefficient'
bsca = root_grp.createVariable('bsca', 'f8', ('sizeBin', 'lambda'))
bsca.units = 'm-1'
bsca.long_name = 'scattering coefficient'
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
vol = root_grp.createVariable('vol', 'f8', ('sizeBin'))
vol.units = 'm3/m3'
vol.long_name = 'volume concentration'
sph = root_grp.createVariable('sph', 'f8', ('sizeBin'))
sph.units = 'none'
sph.long_name = 'fraction of spherical particles'

# data
sizeBin[:] = np.r_[0:Nbin]
wavelength[:] = wvls[0:Nlambda]
angle[:] = np.r_[0:Nangles]
for i,rslt in enumerate(rslts):
    binInd = i//len(lEdg)
    lEdgInd = i%len(lEdg)
    if lEdgInd == 0:
        rv[binInd] = rslt['rv'][0]
        sigma[binInd] = rslt['sigma'][0]
        vol[binInd] = rslt['vol'][0]
        sph[binInd] = np.atleast_1d(rslt['sph'])[0]
    lInd = np.r_[lEdg[lEdgInd]:(lEdg[lEdgInd]+maxL)]
    bext[binInd, lInd] = rslt['aod'] 
    bsca[binInd, lInd] = rslt['ssa']*rslt['aod']
    refreal[binInd, lInd] = rslt['n']  # HINT: WE DID NOT HAVE MODE SPECIFIC REF IND IN YAML
    reafimag[binInd, lInd] = rslt['k']
    for varNm in PMvarNms:
        for i,l in enumerate(lInd):
            PMvars[varNm][binInd, l, :] = rslt[varNm][:,0,i]
#            if varNm == 'p11' and binInd==4:
#                if l%4 == 0:
#                    if l>0:
#                        plt.yscale('log')
#                        plt.legend(lgTxt)
#                    plt.figure()
#                    lgTxt = []
#                plt.plot(np.r_[0:Nangles], rslt[varNm][:,0,i])
#                lgTxt.append(str(rslt['lambda'][i]))
root_grp.close()


# SANITY CHECK PLOT
loaddVarNames = ['lambda', 'bext', 'bsca']
optTblNew = loadVARSnetCDF(savePath_netCDF, loaddVarNames)
fitInd = np.r_[2,4,5,6,7,8,9,11,13]
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(7, 3.5))
Vars = ['ext', 'sca']
for i, vnm in enumerate(Vars):
    vnmq = 'q'+vnm
    vnmb = 'b'+vnm
    ax[i].plot(optTbl['lambda']*1e6, optTbl['q'+vnm][:,0,:].T, '--')
    ax[i].set_prop_cycle(None)
    Scl = optTblNew['b'+vnm][:,8]/optTbl['q'+vnm][:,0,8]
    ax[i].plot(optTblNew['lambda'], optTblNew['b'+vnm].T/Scl)
    ax[i].set_xlim([0.3, 3.0])
    ax[i].set_xlabel('wavelength')   
    ax[i].set_ylabel('$q_{' + vnm + '}$')
ax[0].legend(['Mode %d' % int(x+1) for x in range(optTblNew['b'+vnm].shape[0])])
ax[0].set_prop_cycle(None)
ax[0].plot(optTblNew['lambda'][fitInd], np.atleast_2d(optTbl['qext'][:,0,fitInd]).T, 'x')
plt.suptitle('Orignal (dashed), target point (crosses), and new (solid) spectral dependences', fontsize=12)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])