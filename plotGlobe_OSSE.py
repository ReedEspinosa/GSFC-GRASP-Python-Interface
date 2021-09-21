import numpy as np
from matplotlib import pyplot as plt
from simulateRetrieval import simulation
from mpl_toolkits.basemap import Basemap

mergePATH = '/discover/nobackup/wrespino/OSSE_results_working/MERGED_ss450-g5nr.leV30ALL.GRASP.example.polarimeter07.random.2006ALL_0000z.pkl'
simBase = simulation(picklePath=mergePATH)
print('Loading from %s - %d' % (mergePATH, len(simBase.rsltBck)))

plt.figure(figsize=(12, 6.5))
m = Basemap(projection='merc', resolution='c', llcrnrlon=-180, llcrnrlat=-70, urcrnrlon=180, urcrnrlat=70)
m.bluemarble(scale=1);
#m.shadedrelief(scale=0.2)
#m.fillcontinents(color=np.r_[176,204,180]/255,lake_color='white')
lon = [rb['longitude'] for rb in simBase.rsltBck]
lat = [rb['latitude'] for rb in simBase.rsltBck]
# vals = [rb['sph'] for rb in simBase.rsltFwd]
# label = 'SPH'
# vals = np.log10([rb['costVal'] for rb in simBase.rsltBck])
# label = 'log10(Cost Value)'
# vals = [rb['vol'][1]/rb['vol'].sum() for rb in simBase.rsltBck]
# label = 'SS Fraction'
# vals = np.log10([rb['aod'][3] for rb in simBase.rsltFwd])
# label = 'log10(AOD [550nm])'
# vals = np.log10([rb['vol'][2] for rb in simBase.rsltBck])
# label = 'log10(Dust Vol)'
#vals = np.sqrt([rb['rEffMode'][1]/rf['rEffMode'][1]-1 for rb,rf in zip(simBase.rsltBck, simBase.rsltFwd)])**0.6
vals = np.array([rb['rEffMode'][1]-rf['rEffMode'][1] for rb,rf in zip(simBase.rsltBck, simBase.rsltFwd)])
label = 'Reff Error'

x, y = m(lon, lat)
plt.scatter(x, y, c=vals, s=1, cmap='YlOrRd') # 'YlOrRd', 'seismic'
cbar = plt.colorbar()
cbar.set_label(label, fontsize=14)
plt.tight_layout()

figSaveName = mergePATH.replace('.pkl', '_MAP.png')
print('Saving map to: %s' % figSaveName)
plt.savefig(figSaveName)
plt.ion()
plt.show()

