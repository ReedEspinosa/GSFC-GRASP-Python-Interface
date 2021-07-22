from runGRASP import graspRun, pixel
from matplotlib import pyplot as plt
import numpy as np
import datetime as dt

# Path to the YAML file you want to use for the aerosol and surface definition
# fwdModelYAMLpath = '/Users/wrespino/Synced/Local_Code_MacBook/MADCAP_Analysis/ACCP_ArchitectureAndCanonicalCases/settings_FWD_IQU_POLAR_1lambda.yml'
fwdModelYAMLpath = '/Users/wrespino/Downloads/settings_Dust.yml'

# paths to your GRASP binary and kernels (replace everything up to grasp_open with the path to your GRASP repository)
# binPathGRASP = '/home/cdeleon/grasp/build/bin/grasp'
# krnlPathGRASP = '/home/cdeleon/grasp/src/retrieval/internal_files'
binPathGRASP = '/Users/wrespino/Synced/Local_Code_MacBook/grasp_open/build/bin/grasp'
krnlPathGRASP = '/Users/wrespino/Synced/Local_Code_MacBook/grasp_open/src/retrieval/internal_files'

ttlStr = 'Dust (multi scattering)' # Top Title
wvStrFrmt =  '($%4.2f\\mu m$)' # Format of Y-axis labels 
sza = 30 # solar zenith angle
wvls = [0.34, 0.550] # wavelengths in μm
msTyp = [41, 42, 43] # grasp measurements types (I, Q, U) [must be in ascending order]
azmthΑng = np.r_[0:180:10] # azimuth angles to simulate (0,10,...,175)
vza = 180-np.r_[0:75:5] # viewing zenith angles to simulate [in GRASP cord. sys.]
clrMapReg = plt.cm.jet # color map for plots
clrMapDiff = plt.cm.seismic # color map for plots

# prep geometry inputs for GRASP
Nvza = len(vza)
Nazimth = len(azmthΑng)
thtv_g = np.tile(vza, len(msTyp)*len(azmthΑng))
phi_g = np.tile(np.concatenate([np.repeat(φ, len(vza)) for φ in azmthΑng]), len(msTyp))
nbvm = len(thtv_g)/len(msTyp)*np.ones(len(msTyp), int)
meas = np.r_[np.repeat(0.1, nbvm[0]), np.repeat(0.01, nbvm[1]), np.repeat(0.01, nbvm[2])] # dummy values

# define the "pixel" we want to simulate
nowPix = pixel(dt.datetime.now(), 1, 1, 0, 0, masl=0, land_prct=100)
for wvl in wvls: nowPix.addMeas(wvl, msTyp, nbvm, sza, thtv_g, phi_g, meas)

# setup instance of graspRun class, add the above pixel, run grasp and read the output to gr.invRslt[0] (dict)
# releaseYAML=True tmeans that the python will adjust the YAML file to make it correspond to Nwvls (if it does not already)
gr = graspRun(pathYAML=fwdModelYAMLpath, releaseYAML=True, verbose=True)
gr.addPix(nowPix)
gr.runGRASP(binPathGRASP=binPathGRASP, krnlPathGRASP=krnlPathGRASP)
print('AOD at %5.3f μm was %6.4f.' % (gr.invRslt[0]['lambda'][-1],gr.invRslt[0]['aod'][-1]))

# hemisphere plotting code
Nwvl = len(wvls)
# print(Nwvl)
fig, ax = plt.subplots(2*Nwvl-1, 2, subplot_kw=dict(projection='polar'), figsize=(6,6+3*(Nwvl-1)))
if Nwvl == 1: ax = ax[None,:]
pxInd = 0
r = gr.invRslt[pxInd]['vis'][:,0].reshape(Nazimth, Nvza)
if vza.max()>90: r = 180 - r
theta = gr.invRslt[pxInd]['fis'][:,0].reshape(Nazimth, Nvza)/180*np.pi
r = np.vstack([r, np.flipud(r)]) # mirror symmetric about principle plane
theta = np.vstack([theta, np.flipud(2*np.pi-theta)]) # mirror symmetric about principle plane
for i in range(2):
    data = []
    labels = []
    for l in range(2*Nwvl-1):
        if l<Nwvl:
            wvStr = wvStrFrmt % wvls[l]
            data.append(gr.invRslt[pxInd]['fit_I'][:,l])
            if i==1: # we need to find DoLP
                Q = gr.invRslt[pxInd]['fit_Q'][:,l]
                U = gr.invRslt[pxInd]['fit_U'][:,l]
                data[-1] = (np.sqrt(Q**2+U**2)/data[-1]) * 100
            labels.append(wvStr)
            if i==0: data[-1] = np.log10(data[-1])
            clrMin = data[-1].min()
            clrMax = data[-1].max()
            clrMap = clrMapReg
        else: # this is a difference plot      
#             if i==1:
            data.append(data[l-Nwvl+1] - data[0])
            labels.append(labels[l-Nwvl+1] + " – " + labels[0])
  #           else:
#                 data.append(1 - (data[l-Nwvl+1]/data[0]))
#                 labels.append("1 - " + labels[l-Nwvl+1] + "/" + labels[0])
            clrMax = np.abs(data[-1]).max()
            clrMin = -clrMax
            clrMap = clrMapDiff
        v = np.linspace(clrMin, clrMax, 200, endpoint=True)
        ticks = np.linspace(clrMin, clrMax, 3, endpoint=True)
        data2D = data[-1].reshape(Nazimth, Nvza)
        dataFullHemisphere = np.vstack([data2D, np.flipud(data2D)]) # mirror symmetric about principle plane
        c = ax[l,i].contourf(theta, r, dataFullHemisphere, v, cmap=clrMap)
        ax[l,i].plot(np.pi, sza, '.',  color=[1,1,0], markersize=10)
        cb = plt.colorbar(c, orientation='horizontal', ax=ax[l,i], ticks=ticks)
        ax[l,i].set_ylim([0, r.max()])
        if i==0: ax[l,i].set_ylabel(labels[-1], labelpad=30)
ax[0,0].set_title('log10(Reflectance)')
ax[0,1].set_title('DoLP [%]')
# plt.suptitle(ttlStr)
plt.tight_layout(rect=[0.01, 0.01,0.98, 0.98])
plt.ion()
plt.show()
