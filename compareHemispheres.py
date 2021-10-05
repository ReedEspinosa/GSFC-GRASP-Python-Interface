from runGRASP import graspDB, graspRun, pixel
from matplotlib import pyplot as plt
import os
if os.uname()[1]=='uranus': plt.switch_backend('agg')
import numpy as np
import datetime as dt

# Path to the YAML file you want to use for the aerosol and surface definition
YAMLpaths = ['/Users/wrespino/Synced/MADCAP_CAPER/PCA_landModels/settings_FWD_IQU_POLAR_1lambda_PCA_0.yml', #0
             '/Users/wrespino/Synced/MADCAP_CAPER/PCA_landModels/settings_FWD_IQU_POLAR_1lambda_PCA_5.yml',
             '/Users/wrespino/Synced/MADCAP_CAPER/PCA_landModels/settings_FWD_IQU_POLAR_1lambda_PCA_18.yml',
             '/Users/wrespino/Synced/MADCAP_CAPER/PCA_landModels/settings_FWD_IQU_POLAR_1lambda_RTLS_orig_0.yml', #3
             '/Users/wrespino/Synced/MADCAP_CAPER/PCA_landModels/settings_FWD_IQU_POLAR_1lambda_RTLS_orig_5.yml',
             '/Users/wrespino/Synced/MADCAP_CAPER/PCA_landModels/settings_FWD_IQU_POLAR_1lambda_RTLS_orig_18.yml',
             '/Users/wrespino/Synced/MADCAP_CAPER/PCA_landModels/settings_FWD_IQU_POLAR_1lambda_RTLS_polder_0.yml', #6
             '/Users/wrespino/Synced/MADCAP_CAPER/PCA_landModels/settings_FWD_IQU_POLAR_1lambda_RTLS_polder_5.yml',
             '/Users/wrespino/Synced/MADCAP_CAPER/PCA_landModels/settings_FWD_IQU_POLAR_1lambda_RTLS_polder_18.yml']

# paths to GRASP binary and kernels
binPathGRASP = '/Users/wrespino/Synced/Local_Code_MacBook/grasp_open/build/bin/grasp'
krnlPathGRASP = '/Users/wrespino/Synced/Local_Code_MacBook/grasp_open/src/retrieval/internal_files'

# path to save the figure (None to just display to screen) – NOT SETUP YET
# figSavePath = '/Users/wrespino/Documents/lev2CasePCA.png'
figSavePath = None

graspSavePath = '/Users/wrespino/Documents/compareRTLS_bigOne.pkl'

hemiTrueInd = 3
hemiDiffInd = 0 # diff this YAML file against the one above; maps to YAMLpaths
histYamlInd = [[3,0,6],[4,1,7],[5,2,8]] # yaml file triplets to show hist of errors against[[true,pca,polder],...]
histYamlLegend = ['Conventional','New Method']
pxInd = 4 # pxInd to plot; maps to szas below
wvInd = 3 # wvInd to plot; maps to wvls below
szas = [23.6, 26.6, 29.9, 32.7, 36.3, 40.2, 44.4, 48.8, 53.9] # solar zenith angle
wvls = [0.36, 0.38, 0.41, 0.55, 0.67, 0.87, 1.55, 1.65] # wavelengths in μm
msTyp = [41] # grasp measurements types (I, Q, U) [must be in ascending order]
azmthΑng = np.r_[0:180:10] # azimuth angles to simulate (0,10,...,175)
vza = np.r_[0:65:5] # viewing zenith angles to simulate [in GRASP cord. sys.]
upwardLooking = False # False -> downward looking imagers, True -> upward looking

# clrMapReg = plt.cm.jet
# clrMapReg = plt.cm.viridis
clrMapReg = plt.cm.plasma
# clrMapReg = plt.cm.cividis
clrMapDiff = plt.cm.seismic # color map for difference plots


######## END INPUTS ########

# prepare inputs for GRASP
if upwardLooking: vza=180-vza
Nvza = len(vza)
Nazimth = len(azmthΑng)
thtv_g = np.tile(vza, len(msTyp)*len(azmthΑng)) # create list of all viewing zenith angles in (Nvza X Nazimth X 3) grid
phi_g = np.tile(np.concatenate([np.repeat(φ, len(vza)) for φ in azmthΑng]), len(msTyp)) # create list of all relative azimuth angles in (Nvza X Nazimth X 3) grid
nbvm = len(thtv_g)/len(msTyp)*np.ones(len(msTyp), int) # number of view angles per Stokes element (Nvza X Nazimth)
meas = np.r_[np.repeat(0.1, nbvm[0])]

# setup "graspRun" object and run GRASP binary
gr =[]
for yamlPath in YAMLpaths:
    print('Using settings file at %s' % yamlPath)
    gr.append(graspRun(pathYAML=yamlPath, releaseYAML=True, verbose=True)) # setup instance of graspRun class, add the above pixel, run grasp and read the output to gr.invRslt[0] (dict)
    for sza in szas:
        # initiate the "pixel" object we want to simulate
        nowPix = pixel(dt.datetime.now(), 1, 1, 0, 0, masl=0, land_prct=100)
        for wvl in wvls: nowPix.addMeas(wvl, msTyp, nbvm, sza, thtv_g, phi_g, meas)
        gr[-1].addPix(nowPix) # add the pixel we created above
gDB = graspDB(gr, maxCPU=12)
gDB.processData(binPathGRASP=binPathGRASP, savePath=graspSavePath, krnlPathGRASP=krnlPathGRASP)

for i,yamlPath in enumerate(YAMLpaths):
    wave = gDB.grObjs[i].invRslt[pxInd]['lambda'][wvInd]
    sza = gDB.grObjs[i].invRslt[pxInd]['sza'][0,wvInd]
    aod = gDB.grObjs[i].invRslt[pxInd]['aod'][wvInd]
    print('AOD at %5.3f μm was %6.4f with sza of %4.1f° (%s)' % (wave, aod, sza, os.path.split(yamlPath)[1]))

#  create hemisphere plots
tickLocals = [20, 40, 60]
FS = 10

Nwvl = len(wvls)
fig, ax = plt.subplots(1, 2, subplot_kw=dict(projection='polar'), figsize=(4.5,2.6))

r = gDB.grObjs[hemiTrueInd].invRslt[pxInd]['vis'][:,0].reshape(Nazimth, Nvza)
if upwardLooking: r = 180 - r
theta = gDB.grObjs[hemiTrueInd].invRslt[pxInd]['fis'][:,0].reshape(Nazimth, Nvza)/180*np.pi
r = np.vstack([r, np.flipud(r)]) # mirror symmetric about principle plane
theta = np.vstack([theta, np.flipud(2*np.pi-theta)]) # mirror symmetric about principle plane

# plot truth
data2D = gDB.grObjs[hemiTrueInd].invRslt[pxInd]['fit_I'][:,wvInd].reshape(Nazimth, Nvza)
clrMin = np.floor(gDB.grObjs[hemiTrueInd].invRslt[pxInd]['fit_I'][:,wvInd].min()*100)/100
clrMax = np.ceil(gDB.grObjs[hemiTrueInd].invRslt[pxInd]['fit_I'][:,wvInd].max()*100)/100
v = np.linspace(clrMin, clrMax, 255, endpoint=True)
ticks = np.linspace(clrMin, clrMax, 3, endpoint=True)
dataFullHemisphere = np.vstack([data2D, np.flipud(data2D)]) # mirror symmetric about principle plane
cnt = ax[0].contourf(theta, r, dataFullHemisphere, v, cmap=clrMapReg)
for c in cnt.collections: # need to prevent contours from showing when exporting to PDF
    c.set_edgecolor("face")
cb = plt.colorbar(cnt, orientation='horizontal', ax=ax[0], ticks=ticks)
ax[0].set_yticks(tickLocals)
ax[0].set_ylim([0, r.max()])
ax[0].set_title('Reflectance Truth', y=1.2, fontsize=FS)
ax[0].xaxis.set_tick_params(pad=1.5)

# plot diff from truth
dataDiff2D = 100*(gDB.grObjs[hemiDiffInd].invRslt[pxInd]['fit_I'][:,wvInd].reshape(Nazimth, Nvza) - data2D)/data2D
clrMax = np.ceil(np.abs(dataDiff2D).max()*10)/10
clrMax = 1.5
clrMin = -clrMax
clrMap = clrMapDiff
v = np.linspace(clrMin, clrMax, 255, endpoint=True)
ticks = np.linspace(clrMin, clrMax, 3, endpoint=True)
dataFullHemisphere = np.vstack([dataDiff2D, np.flipud(dataDiff2D)]) # mirror symmetric about principle plane
cnt = ax[1].contourf(theta, r, dataFullHemisphere, v, cmap=clrMapDiff)
for c in cnt.collections:
    c.set_edgecolor("face")
cb = plt.colorbar(cnt, orientation='horizontal', ax=ax[1], ticks=ticks)
ax[1].set_yticks(tickLocals)
ax[1].set_ylim([0, r.max()])
# ax[1].set_title('(New Method – Truth)/Truth [%]', y=1.2)
ax[1].set_title('PC Method Relative Error', y=1.2, fontsize=FS)
cb.set_ticklabels(['%3.1f%%' % x for x in cb.get_ticks()]) # add % to colorbar labels
ax[1].xaxis.set_tick_params(pad=1.5)
plt.tight_layout()


# display or save the new plots
if figSavePath is None:
    plt.ion()
    plt.show()
else:
    plt.savefig(figSavePath)
    print('Figure saved as %s' % figSavePath)
