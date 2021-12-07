import numpy as np
from matplotlib import pyplot as plt
from simulateRetrieval import simulation
from mpl_toolkits.basemap import Basemap

mergePATH = '/discover/nobackup/wrespino/OSSE_results_working/MERGED_ss450-g5nr.leV210.GRASP.example.polarimeter07.200608ALL_1000z.pkl'
# mergePATH = '/discover/nobackup/wrespino/OSSE_results_working/MERGED_ss450-g5nr.leV210.GRASP.example.polarimeter07.200608ALL_ALLz.pkl'
simBase = simulation(picklePath=mergePATH)
print('Loading from %s - %d' % (mergePATH, len(simBase.rsltBck)))
simBase.classifyAerosolType(verbose=True)

waveInd = 2
# AOD
# true = np.array([rb['aod'][waveInd] for rb in simBase.rsltBck])
# rtrv = np.array([rf['aod'][waveInd] for rf in simBase.rsltFwd])
# label = 'AOD (λ=%4.2fμm)' % simBase.rsltFwd[0]['lambda'][waveInd]

# 1-SSA
true = 1-np.array([rb['ssa'][waveInd] for rb in simBase.rsltBck])
rtrv = 1-np.array([rf['ssa'][waveInd] for rf in simBase.rsltFwd])
label = 'Coalbedo (λ=%4.2fμm)' % simBase.rsltFwd[0]['lambda'][waveInd]

# AAOD
true = np.asarray([(1-rf['ssa'][waveInd])*rf['aod'][waveInd] for rf in simBase.rsltFwd])
rtrv = np.asarray([(1-rb['ssa'][waveInd])*rb['aod'][waveInd] for rb in simBase.rsltBck])
label = 'AAOD (λ=%4.2fμm)' % simBase.rsltFwd[0]['lambda'][waveInd]

errFun = lambda t,r : np.sqrt(np.mean((t-r)**2))
aeroType = np.array([rf['aeroType'] for rf in simBase.rsltFwd])
version = 'V0'

labels = [
    'Dust   ',
    'PolDust',
    'Marine',
    'Urb/Ind',
    'BB-White',
    'BB-Dark',
    'All Types']

colors = np.array(
      [[0.86459569, 0.0514    , 0.05659828, 1.        ],
       [1.        , 0.35213633, 0.02035453, 1.        ],
       [0.13613483, 0.39992585, 0.65427592, 1.        ],
       [0.850949428, 0.50214365, 0.65874433, 1.        ],
       [0.30,      0.72, 0.30, 1.        ],
       [0.35    , 0.42 , 0.2, 1.        ],
       [0.3    , 0.3 , 0.3, 1.        ]]) # all
       
dxPerType = 7

fig, ax1 = plt.subplots(figsize=(7, 2.6))
Ntype = colors.shape[0]
Npix = []
for i in np.r_[0:(Ntype*dxPerType):dxPerType]:
    if i/dxPerType==(Ntype-1):
        indKeep = slice(None)
        NpixNow = len(true)
    else:
        indKeep = aeroType==round(i/dxPerType)
        NpixNow = indKeep.sum()
    muTr = np.mean(true[indKeep])
    muRt = np.mean(rtrv[indKeep])
    sigTr = np.std(true[indKeep])
    rmseRt = errFun(rtrv[indKeep], true[indKeep])
    
    typeColor = colors[int(round(i/dxPerType))]

    whtExp = 0.29 if i==0 else 0.7
#     ax1.bar(i,   muTr, color=typeColor**whtExp, edgecolor='white')
#     ax1.bar(i+1, muRt, color=typeColor**1.25, edgecolor='white')
#     ax1.bar(i+2, sigTr, color=typeColor**whtExp, edgecolor='white', hatch='\\')
#     ax1.bar(i+3, rmseRt, color=typeColor**1.25, edgecolor='white', hatch='\\')

#     ax1.bar(i,   muTr, color=typeColor**(whtExp*0.5), edgecolor='white')
#     ax1.bar(i+1, muRt, color=typeColor**(whtExp*0.5), edgecolor='white', hatch='\\')
    ax1.bar(i,   muTr, color=typeColor**1.25, edgecolor='white', alpha=0.5)
    ax1.bar(i+1, muRt, color=typeColor**1.25, edgecolor='white', hatch='\\', alpha=0.5)
    ax1.bar(i+2, sigTr, color=typeColor**1.25, edgecolor='white')
    ax1.bar(i+3, rmseRt, color=typeColor**1.25, edgecolor='white', hatch='\\')

    print('Type %2d – N=%7d' % (round(i/dxPerType), NpixNow))
    Npix.append(NpixNow)

x0 = (colors.shape[0]-1)*dxPerType-2
yMax = ax1.get_ylim()[1]
ax1.plot([x0,x0], [0, yMax], '--', color='gray')
ax1.set_ylim([0, yMax])

ax1.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=True) # labels along the bottom edge are off
    
labels = [label+'\n(%d)' % n for label,n in zip(labels, Npix)]
ax1.set_xticks(np.r_[2:(Ntype*dxPerType+2):dxPerType])
ax1.set_xticklabels(labels)
ax1.set_ylabel(label)
plt.tight_layout()

fig.savefig('/discover/nobackup/wrespino/synced/Working/AGU2021_Plots/errorBarPlots_%s_%s.pdf' % (label[0:-11], version))
plt.ion()
plt.show()
