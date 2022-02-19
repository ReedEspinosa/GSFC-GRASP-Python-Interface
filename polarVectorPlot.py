import numpy as np
import matplotlib.pyplot as plt
import PyMieScatt as ps

fig, ax = plt.subplots(subplot_kw=dict(polar=True))

kw = dict(arrowstyle="-|>", color='k', linewidth=1)
kw = dict(color='k', width=0.4,headwidth=2.5,headlength=3.0,alpha=0.3)


m = complex(1.59, 1e-4)
colors = np.asarray([
			[0.224, 0.416, 1.000],
			[0.145, 0.835, 0.184],
			[1.000, 0.471, 0.171]])**2

wavs = [473, 532, 671] # nm

colors = np.asarray([
			[0.224, 0.416, 1.000],
			
			[1.000, 0.471, 0.171]])**3

wavs = [473, 671] # nm


dp = [903] # diameter of size bin, nm
ndp = [1.0] # number in size bin

maxS11 = 0
for i,(wav, clrNow) in enumerate(zip(wavs,colors)):
	theta,sl,sr,su = ps.SF_SD(m, wav, dp, ndp)
	S11=0.5*(sl+sr)
	# P11 = 2*S11/np.trapz(S11*np.sin(theta), theta) # normalized P11
	P11 = (S11**0.25) # transform absolute P11
	kw['color'] = np.asarray(clrNow)
	stepSize = 10
	startInd = int(i*stepSize/len(wavs))
	contLineSep = 0.017
	for angle, radius in zip(theta[startInd:360:stepSize], P11[startInd:360:stepSize]):
		ax.annotate("", xy=(angle, radius - contLineSep*np.max(P11)), xytext=(0, 0), arrowprops=kw)
		ax.annotate("", xy=(2*np.pi-angle, radius - contLineSep*np.max(P11)), xytext=(0, 0), arrowprops=kw)
	maxS11 = max(maxS11, np.max(P11))
	contLineWidth = 0.7
	ax.plot(theta, P11, '-', color=clrNow, linewidth=contLineWidth)
	ax.plot(2*np.pi-theta, P11, '-', color=clrNow, linewidth=contLineWidth)
ax.set_ylim(0, maxS11) # we assume first wavelength has strongest scattering
ax.set_yticks([])
ax.set_xticks([])
ax.spines['polar'].set_visible(False)
ax.plot([0],[0],'o',color='w',markersize=14,zorder=4)
# ax.annotate("", xy=(np.pi, maxS11/25), xytext=(np.pi, 0.8*maxS11), arrowprops={'arrowstyle':'-|>', 'color':'k','linewidth':2.0},zorder=5)
ax.annotate("", xy=(np.pi, maxS11/25), xytext=(np.pi, 0.8*maxS11), arrowprops={'arrowstyle':'-|>', 'color':[0.4,0.4,0.4],'linewidth':2.0},zorder=5)
ax.plot([0],[0],'o',color='k',markersize=12,zorder=6)
plt.ion()
plt.show()
