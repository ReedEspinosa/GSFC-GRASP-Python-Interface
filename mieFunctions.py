#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 13:36:12 2019

@author: wrespino
"""

import PyMieScatt as ps 
import numpy as np
import matplotlib.pyplot as plt

# https://pymiescatt.readthedocs.io
#SR = |S1|^2
#SL = |S2|^2
#SU = 0.5(SR+SL)
#S11 = 0.5 (|S2|^2 + |S1|^2)
#S12 = 0.5 (|S2|^2 - |S1|^2)  [S12/S11=1 -> unpolarized light is scattered into 100% polarized light oriented perpendicular to scattering plane]
#S33 = 0.5 (S2^* S1 + S2 S1^*)
#S34 = 0.5 (S1 S2^* - S2 S1^*)

#m = 1.35528135+6.287e-7j # complex
m = complex(rslt[0]['n'], rslt[0]['k'])
wav = 865 # nm
#dp = [903,1000] # diamter of size bin, nm
#ndp = [1e6,1e1] # number in size bin

intrpRadii = np.logspace(np.log10(minR),np.log10(maxR),500)
dp = intrpRadii*2000
ndp = dvdlnr(intrpRadii)/(intrpRadii**4)


theta,sl,sr,su = ps.SF_SD(m, wav, dp, ndp)
S11=0.5*(sl+sr)
S12=-0.5*(sl-sr) # minus makes positive S12 polarized in scattering plane
#plt.figure()
#plt.plot(theta, -S12/S11)

P11 = 2*S11/np.trapz(S11*np.sin(theta), theta)

#should make PSD class with r,dNdr and type (dndr,dvdr,etc.)
