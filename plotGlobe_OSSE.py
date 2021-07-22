import numpy as np
from matplotlib import pyplot as plt
from simulateRetrieval import simulation
from mpl_toolkits.basemap import Basemap

#mergePATH = '/discover/nobackup/wrespino/synced/Working/OSSE_Test_Run/ss450-g5nr.leV210.GRASP.example.polarimeter07.20060802_1000z.pkl' 
mergePATH = '/discover/nobackup/wrespino/synced/Working/OSSE_Test_Run/MERGED_ss450-g5nr.leV210.GRASP.example.polarimeter07.200608ALL_ALLz.pkl' 
simBase = simulation(picklePath=mergePATH)
print('Loading from %s - %d' % (mergePATH, len(simBase.rsltBck)))

plt.figure(figsize=(12, 6.5))
m = Basemap(projection='merc', resolution='c', llcrnrlon=-180, llcrnrlat=-60, urcrnrlon=180, urcrnrlat=80)
m.bluemarble(scale=1);
#m.shadedrelief(scale=0.2)
#m.fillcontinents(color=np.r_[176,204,180]/255,lake_color='white')
lon = [rb['longitude'] for rb in simBase.rsltBck]
lat = [rb['latitude'] for rb in simBase.rsltBck]
# vals = [rb['sph'] for rb in simBase.rsltFwd]
# label = 'SPH'
vals = np.log10([rb['costVal'] for rb in simBase.rsltBck])
label = 'log10(Cost Value)'
# vals = [rb['vol'][1]/rb['vol'].sum() for rb in simBase.rsltBck]
# label = 'SS Fraction'
# vals = np.log10([rb['aod'][3] for rb in simBase.rsltFwd])
# label = 'log10(AOD [550nm])'
# vals = np.log10([rb['vol'][2] for rb in simBase.rsltBck])
# label = 'log10(Dust Vol)'
x, y = m(lon, lat)
plt.scatter(x, y, c=vals, s=2, cmap='YlOrRd') # 'YlOrRd', 'seismic'
cbar = plt.colorbar()
cbar.set_label(label, fontsize=14)
plt.tight_layout()

figSaveName = mergePATH.replace('.pkl', '_MAP.png')
print('Saving map to: %s' % figSaveName)
plt.savefig(figSaveName)
plt.show()
