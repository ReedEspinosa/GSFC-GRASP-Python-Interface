from runGRASP import graspRun, pixel
from matplotlib import pyplot as plt
import os
if os.uname()[1]=='uranus': plt.switch_backend('agg')
import numpy as np
import datetime as dt

# Path to the YAML file you want to use for the aerosol and surface definition
fwdModelYAMLpath = '/home/respinosa/Documents/settings_BiomassBurning.yml'

# paths to GRASP binary and kernels
binPathGRASP = '/home/respinosa/GRASP_GSFC/build/bin/grasp'
krnlPathGRASP = '/home/respinosa/GRASP_GSFC/src/retrieval/internal_files'

# path to save the figure (None to just display to screen)
figSavePath = '/home/respinosa/Documents/dust_hemisphere.png'
# figSavePath = None

sza = 30 # solar zenith angle
wvls = [0.34, 0.550] # wavelengths in μm
msTyp = [41, 42, 43] # grasp measurements types (I, Q, U) [must be in ascending order]
azmthΑng = np.r_[0:180:10] # azimuth angles to simulate (0,10,...,175)
vza = np.r_[0:75:5] # viewing zenith angles to simulate [in GRASP cord. sys.]
upwardLooking = False # False -> downward looking imagers, True -> upward looking

clrMapReg = plt.cm.jet # color map for base value plots
clrMapDiff = plt.cm.seismic # color map for difference plots
logOfReflectance = True # Plot the base 10 log of reflectance instead of actual value
ttlStr = None # Top Title as string or None to skip
wvStrFrmt =  '($%4.2f\\mu m$)' # Format of Y-axis labels 


######## END INPUTS ########

# prepare inputs for GRASP
if upwardLooking: vza=180-vza
Nvza = len(vza)
Nazimth = len(azmthΑng)
thtv_g = np.tile(vza, len(msTyp)*len(azmthΑng)) # create list of all viewing zenith angles in (Nvza X Nazimth X 3) grid
phi_g = np.tile(np.concatenate([np.repeat(φ, len(vza)) for φ in azmthΑng]), len(msTyp)) # create list of all relative azimuth angles in (Nvza X Nazimth X 3) grid
nbvm = len(thtv_g)/len(msTyp)*np.ones(len(msTyp), int) # number of view angles per Stokes element (Nvza X Nazimth)
meas = np.r_[np.repeat(0.1, nbvm[0]), np.repeat(0.01, nbvm[1]), np.repeat(0.01, nbvm[2])] # dummy measurement values for I, Q and U, respectively 

# initiate the "pixel" object we want to simulate
nowPix = pixel(dt.datetime.now(), 1, 1, 0, 0, masl=0, land_prct=0)
for wvl in wvls: nowPix.addMeas(wvl, msTyp, nbvm, sza, thtv_g, phi_g, meas)

# setup "graspRun" object and run GRASP binary
print('Using settings file at %s' % fwdModelYAMLpath)
gr = graspRun(pathYAML=fwdModelYAMLpath, releaseYAML=True, verbose=True) # setup instance of graspRun class, add the above pixel, run grasp and read the output to gr.invRslt[0] (dict)
gr.addPix(nowPix) # add the pixel we created above
gr.runGRASP(binPathGRASP=binPathGRASP, krnlPathGRASP=krnlPathGRASP) # run grasp binary (output read to gr.invRslt[0] [dict])
print('AOD at %5.3f μm was %6.4f.' % (gr.invRslt[0]['lambda'][-1],gr.invRslt[0]['aod'][-1])) # 

#  create hemisphere plots
Nwvl = len(wvls)
fig, ax = plt.subplots(2*Nwvl-1, 2, subplot_kw=dict(projection='polar'), figsize=(6,6+3*(Nwvl-1)))
if Nwvl == 1: ax = ax[None,:]
pxInd = 0
r = gr.invRslt[pxInd]['vis'][:,0].reshape(Nazimth, Nvza)
if upwardLooking: r = 180 - r
theta = gr.invRslt[pxInd]['fis'][:,0].reshape(Nazimth, Nvza)/180*np.pi
r = np.vstack([r, np.flipud(r)]) # mirror symmetric about principle plane
theta = np.vstack([theta, np.flipud(2*np.pi-theta)]) # mirror symmetric about principle plane
for i in range(2): # loop over reflectance (i=0) and DoLP (i=1)
    data = []
    labels = []
    for l in range(2*Nwvl-1): # loop over subplot rows; first for each wavelength, then the differences between them
        if l<Nwvl: 
            wvStr = wvStrFrmt % wvls[l]
            data.append(gr.invRslt[pxInd]['fit_I'][:,l])
            if i==1: # we need to calculate DoLP
                Q = gr.invRslt[pxInd]['fit_Q'][:,l]
                U = gr.invRslt[pxInd]['fit_U'][:,l]
                data[-1] = (np.sqrt(Q**2+U**2)/data[-1]) * 100
            labels.append(wvStr)
            if i==0 and logOfReflectance: data[-1] = np.log10(data[-1]) # take log of reflectance
            clrMin = data[-1].min()
            clrMax = data[-1].max()
            clrMap = clrMapReg
        else: # this is a difference plot  
            data.append(data[l-Nwvl+1] - data[0])
            labels.append(labels[l-Nwvl+1] + " – " + labels[0])
            clrMax = np.abs(data[-1]).max()
            clrMin = -clrMax
            clrMap = clrMapDiff
        v = np.linspace(clrMin, clrMax, 255, endpoint=True)
        ticks = np.linspace(clrMin, clrMax, 3, endpoint=True)
        data2D = data[-1].reshape(Nazimth, Nvza)
        dataFullHemisphere = np.vstack([data2D, np.flipud(data2D)]) # mirror symmetric about principle plane
        c = ax[l,i].contourf(theta, r, dataFullHemisphere, v, cmap=clrMap)
        if upwardLooking: ax[l,i].plot(np.pi, sza, '.',  color=[1,1,0], markersize=10) # plot sun position (if upward looking observation)
        cb = plt.colorbar(c, orientation='horizontal', ax=ax[l,i], ticks=ticks)
        ax[l,i].set_ylim([0, r.max()])
        if i==0: ax[l,i].set_ylabel(labels[-1], labelpad=30)
Rtitle = 'log10(Reflectance)' if logOfReflectance else 'Reflectance'
ax[0,0].set_title(Rtitle, y=1.2) # placement not consistent across all backends
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
