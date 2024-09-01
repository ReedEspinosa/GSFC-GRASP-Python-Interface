from matplotlib import pyplot as plt
import scipy.ndimage as ndimage
import os
import sys
if os.uname()[1]=='uranus': plt.switch_backend('agg')
import numpy as np
import datetime as dt
from runGRASP import graspRun, pixel
RtrvSimParentDir = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) # we assume GSFC-GRASP-Python-Interface is in parent of GSFC-Retrieval-Simulators
sys.path.append(os.path.join(RtrvSimParentDir, 'GSFC-Retrieval-Simulators', 'ACCP_ArchitectureAndCanonicalCases'))
from canonicalCaseMap import setupConCaseYAML


# Path to the YAML file you want to use for the aerosol and surface definition
# baseYAML = '/Users/wrespino/Synced/Local_Code_MacBook/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/settings_FWD_IQU_POLAR_1lambda.yml'
baseYAML = ['/Users/wrespino/Synced/Local_Code_MacBook/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/settings_FWD_IQU_POLAR_1lambda.yml',
            '/Users/wrespino/Synced/Local_Code_MacBook/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/settings_FWD_IQU_POLAR_1lambda-mod.yml']

# paths to GRASP binary and kernels
binPathGRASP = '/Users/wrespino/Synced/Local_Code_MacBook/grasp_open/build/bin/grasp'
krnlPathGRASP = '/Users/wrespino/Synced/Local_Code_MacBook/grasp_open/src/retrieval/internal_files'

# path to save the figure (None to just display to screen)
# figSavePath = '/Users/wrespino/Documents/lev2CasePCA.png'
figSavePath = None

# wvls = [0.36, 0.38, 0.41, 0.55, 0.67, 0.87, 1.55, 1.65] # wavelengths in μm
wvls = [0.67] # wavelengths in μm
caseStrs = ['lwcloud', 'lwcloud-mod']
tauFactor = 1.0 # applies to all cases currently

sza = 30 # solar zenith angle
msTyp = [41, 42, 43] # grasp measurements types (I, Q, U) [must be in ascending order]
# azmthΑng = np.r_[0:180:5] # azimuth angles to simulate (0,10,...,175)
# vza = np.r_[0:75:5] # viewing zenith angles to simulate [in GRASP cord. sys.]
azmthΑng = np.r_[0:40:10,40:80:2,80:180:10] # azimuth angles to simulate (0,10,...,175)
vza = np.r_[0:17:1,17:25:2,25:75:5] # viewing zenith angles to simulate [in GRASP cord. sys.]
upwardLooking = False # False -> downward looking imagers, True -> upward looking

clrMapReg = plt.cm.jet # color map for base value plots
clrMapDiff = plt.cm.seismic # color map for difference plots
logOfReflectance = False # Plot the base 10 log of reflectance instead of actual value
resolveQU = True
blurrSigma = [0, 0, 0] # sigma coefs for Gaussian blur in plots-only [I, Q/DoLP, U]; all zeros for now blur 
# blurrSigma = [0.7, 0.55, 0.6] # sigma coefs for Gaussian blur in plots-only [I, Q/DoLP, U]; all zeros for now blur 
ttlStr = None # Top Title as string or None to skip
wvFrmt =  '($%4.2f\\mu m$)' # Format of Y-axis labels 
graspVerboseLev = 2

######## END INPUTS ########

# prepare inputs for GRASP
spectralCompare = len(wvls)>1
assert not (spectralCompare and len(caseStrs)>1), 'This script only compares wavelengths or cases, not both.' 
if upwardLooking: vza=180-vza
Nvza = len(vza)
Nazimth = len(azmthΑng)
thtv_g = np.tile(vza, len(msTyp)*len(azmthΑng)) # create list of all viewing zenith angles in (Nvza X Nazimth X 3) grid
phi_g = np.tile(np.concatenate([np.repeat(φ, len(vza)) for φ in azmthΑng]), len(msTyp)) # create list of all relative azimuth angles in (Nvza X Nazimth X 3) grid
nbvm = len(thtv_g)/len(msTyp)*np.ones(len(msTyp), int) # number of view angles per Stokes element (Nvza X Nazimth)
meas = np.r_[np.repeat(0.1, nbvm[0]), np.repeat(0.01, nbvm[1]), np.repeat(0.01, nbvm[2])] # dummy measurement values for I, Q and U, respectively 

# initiate the "pixel" object we want to simulate
nowPix = pixel(dt.datetime.now(), 1, 1, 0, 0, masl=0, land_prct=100)
for wvl in wvls: nowPix.addMeas(wvl, msTyp, nbvm, sza, thtv_g, phi_g, meas)

# setup "graspRun" object and run GRASP binary
invRslt = []
if isinstance(baseYAML, str): baseYAML = [baseYAML for _ in range(caseStrs)]
for yamlNow, caseStr in zip(baseYAML, caseStrs):
    print('Running forward model for %s case' % caseStr)
    fwdYAMLPath = setupConCaseYAML(caseStr, nowPix, yamlNow, caseLoadFctr=tauFactor)
    gObjFwd = graspRun(fwdYAMLPath, verbose=graspVerboseLev)
    gObjFwd.addPix(nowPix)
    gObjFwd.runGRASP(binPathGRASP=binPathGRASP, krnlPathGRASP=krnlPathGRASP)
    invRslt.append(np.take(gObjFwd.readOutput(),0)) # we need take because readOutput returns list, even if just one element
    print('%s AOD at %5.3f μm was %6.4f.\n' % (caseStr, invRslt[-1]['lambda'][-1],invRslt[-1]['aod'][-1])) # 
    
#  create hemisphere plots
pxInd = 0
Nwvl = len(wvls) if spectralCompare else len(caseStrs)
Ncol = 3 if resolveQU else 2
fig, ax = plt.subplots(2*Nwvl-1, Ncol, subplot_kw=dict(projection='polar'), figsize=(3*Ncol, 6+3*(Nwvl-1)))
if Nwvl == 1: ax = ax[None,:]
r = invRslt[pxInd]['vis'][:,0].reshape(Nazimth, Nvza)
if upwardLooking: r = 180 - r
theta = invRslt[pxInd]['fis'][:,0].reshape(Nazimth, Nvza)/180*np.pi
r = np.vstack([r, np.flipud(r)]) # mirror symmetric about principle plane
theta = np.vstack([theta, np.flipud(2*np.pi-theta)]) # mirror symmetric about principle plane
for i in range(Ncol): # loop over reflectance (i=0) and DoLP (i=1)
    data = []
    labels = []
    for l in range(2*Nwvl-1): # loop over subplot rows; first for each wavelength, then the differences between them
        wvInd = l if spectralCompare else 0
        pxInd = 0 if spectralCompare else l
        if l<Nwvl: 
            I = invRslt[pxInd]['fit_I'][:,wvInd]
            if i==0:
                data.append(I)
                if logOfReflectance: data[-1] = np.log10(data[-1]) # take log of reflectance
            else:
                Q = invRslt[pxInd]['fit_Q'][:,wvInd]
                U = invRslt[pxInd]['fit_U'][:,wvInd]
                if resolveQU:
                    data.append(Q/I if i==1 else U/I)
                else: # DoLP
                    data.append(100 * np.sqrt(Q**2+U**2)/I)
            lbl = (wvFrmt % wvls[wvInd]) if spectralCompare else '(%s)' % caseStrs[pxInd]
            labels.append(lbl)
            clrMin = data[-1].min()
            clrMax = 0.1 if data[-1].max()==clrMin else data[-1].max() # U in single scat has max=min
            clrMap = clrMapReg
        else: # this is a difference plot  
            data.append(data[l-Nwvl+1] - data[0])
            labels.append(labels[l-Nwvl+1] + " – " + labels[0])
            clrMax = np.abs(data[-1]).max()
            if clrMax==0: clrMax=0.1
            clrMin = -clrMax
            clrMap = clrMapDiff
        v = np.linspace(clrMin, clrMax, 255, endpoint=True)
        ticks = np.linspace(clrMin, clrMax, 3, endpoint=True)
        data2D = data[-1].reshape(Nazimth, Nvza)
        dataFullHemi = np.vstack([data2D, np.flipud(data2D)]) # mirror symmetric about principle plane
        if np.any(blurrSigma)>0: 
            dataFullHemi = ndimage.gaussian_filter(dataFullHemi, sigma=blurrSigma[i], order=0)
        c = ax[l,i].contourf(theta, r, dataFullHemi, v, cmap=clrMap)
        if upwardLooking: ax[l,i].plot(np.pi, sza, '.',  color=[1,1,0], markersize=10) # plot sun position (if upward looking observation)
        cb = plt.colorbar(c, orientation='horizontal', ax=ax[l,i], ticks=ticks)
        ax[l,i].set_ylim([0, r.max()])
        if i==0: ax[l,i].set_ylabel(labels[-1], labelpad=30)
Rtitle = 'log10(Reflectance)' if logOfReflectance else 'Reflectance'
ax[0,0].set_title(Rtitle, y=1.2) # placement not consistent across all backends
if resolveQU:
    ax[0,1].set_title('Q/I', y=1.2)
    ax[0,2].set_title('U/I', y=1.2)
else:
    ax[0,1].set_title('DoLP [%]', y=1.2)
if ttlStr is not None: plt.suptitle(ttlStr)
plt.tight_layout(rect=[0.01, 0.01,0.98, 0.98])

# display or save the new plots
if figSavePath is None:
    plt.ion()
    plt.show()
else:
    plt.savefig(figSavePath)
    print('Figure saved as %s' % figSavePath)
