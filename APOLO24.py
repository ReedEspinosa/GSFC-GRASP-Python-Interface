


"""

Greema Regmi, UMBC, 2024

This code is written for my presentation in APOLO 2024. 

The code reads a single pixel info from the RSP and HSRL data from ORACLES campaign 2018. The pixel no are set to look at the dust layer over the ocean. 


We will use functions from ORACLES_GRASP.py to run GRASP for RSP and HSRL seperately

The code performs 2-step RSP and HSRL retrieval: Usoing retrieved states from RSP to constraine the search space of HSRL


"""

import sys
from CreateRsltsDict import Read_Data_RSP_Oracles, Read_Data_HSRL_Oracles_Height
from ORACLES_GRASP import AeroProfNorm_sc2,FindPix, RSP_Run,  HSLR_run, LidarAndMAP, plot_HSRL,RSP_plot,CombinedLidarPolPlot,Ext2Vconc
import netCDF4 as nc
import netCDF4 as nc
from runGRASP import graspDB, graspRun, pixel, graspYAML
from matplotlib import pyplot as plt
import os
if os.uname()[1]=='uranus': plt.switch_backend('agg')
import numpy as np
import datetime as dt
from numpy import nanmean
import h5py 
sys.path.append("/home/gregmi/git/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases")
from architectureMap import returnPixel
import yaml
import pickle
from netCDF4 import Dataset
from scipy.interpolate import interp1d


from ORACLES_GRASP import Gaussian_fits, update_HSRLyaml, CombineHSRLandRSPrslt
%matplotlib inline






#.....................................................................................
#.....................................................................................

# Running GRASP for single sensor RSP and HSRL retrievals over the dust scene

#.....................................................................................
#.....................................................................................


# Provide the information from RSP here: 
RSPfile_path = "/home/gregmi/ORACLES/RSP1-L1C_P3_20180922_R03/"  #Path to the ORACLE data file
RSPfile_name =  "/RSP1-P3_L1C-RSPCOL-CollocatedRadiances_20180922T151106Z_V003-20210421T233946Z.h5" #Name of the ORACLES file
RSP_PixNo = 13201
TelNo = 0 # aggregated altitude. To obtain geometries corresponding to data from the 1880 nm channel, aggregation altitude should be set to 1, while aggregation altitude =0 should be used for all other channels.
nwl = 5 # first  nwl wavelengths

#Angular range in index (Sca_ang[ang1]: Sca_ang[ang2])
ang1 = 10
ang2 = 120 



#HSRL
HSRLfile_path = "/home/gregmi/ORACLES/HSRL" #Path to the ORACLE  HSRL data file 
HSRLfile_name =  "/HSRL2_P3_20180922_R2.h5" #Name of the ORACLES HSRL  file

#Path to the gas absorption (tau) values for gas absorption correction, of gas absoprtion is done using GRASP
GasAbsFn = '/home/gregmi/ORACLES/UNL_VRTM/shortwave_gas.unlvrtm.nc'

#Path to the Spectral response finction of RSP
SpecResFnPath = '/home/gregmi/ORACLES/RSP_Spectral_Response/'


#This is required if we want to configure the HSRL yaml file based on the GRASP output for the RSP
noMod =3  #number of aerosol mode, here 2 for fine+coarse mode configuration
 #no of char # this is a varible added to char and modes to avoid char[0]/ mod[0] which doesnt exist

#Uncertainity values for error bars, Taken from AOS, This is just for plotting the error bars. Not used in the calculation 

UNCERT ={}
UNCERT['aodMode'] = 0
UNCERT['aod']= 0
UNCERT['ssaMode'] = 0
UNCERT['k'] = 0
UNCERT['n'] = 0
UNCERT['DP'] = 1
UNCERT['VBS'] =  2e-7     #2.4e-6        #2e-7 # rel %
UNCERT['VEXT'] = 1e-05  #rel% 10%
UNCERT['rv'] = 0
UNCERT['I'] = 0.03
UNCERT['DoLP'] = 0.005



#.....................................................................................
#Find  HSRL PIXEL corresponding to RSP co ordiantes
#.....................................................................................

'Reading the RSP data, HSRL PixNO calculates the Pixel no fo the HSRL that corresponds to the lat and lon of RSP'

f1_MAP = h5py.File(RSPfile_path+RSPfile_name,'r+')   
Data = f1_MAP['Data']
LatRSP = f1_MAP['Geometry']['Collocated_Latitude'][TelNo,RSP_PixNo]
LonRSP = f1_MAP['Geometry']['Collocated_Longitude'][TelNo,RSP_PixNo]
f1_MAP.close()
f1= h5py.File(HSRLfile_path + HSRLfile_name,'r+')  #reading hdf5 file  

#Lat and Lon values for that pixel
LatH = f1['Nav_Data']['gps_lat'][:]
LonH = f1['Nav_Data']['gps_lon'][:]
f1.close()
# HSRLPixNo is the index of HSRL pixel taht corresponds to the RSP Lat Lon
HSRLPixNo = FindPix(LatH,LonH,LatRSP,LonRSP)[0] # Or can manually give the index of the pixel that you are intrested in
#.....................................................................................
#.....................................................................................



#.....................................................................................
#Running GRASP for RSP for two shape models
#.....................................................................................


rslts_Sph = RSP_Run("sphro",RSPfile_path,RSPfile_name,RSP_PixNo,ang1,ang2,TelNo,nwl,GasAbsFn,ModeNo=3)
rslts_Tamu = RSP_Run("TAMU",RSPfile_path,RSPfile_name,RSP_PixNo,ang1,ang2,TelNo,nwl,GasAbsFn,ModeNo=3)


#Plot the data
RSP_plot(rslts_Sph,rslts_Tamu,RSP_PixNo,UNCERT)


#.....................................................................................
#Running GRASP for HSRL for two shape models
#.....................................................................................



HSRL_sphrodT = HSLR_run("sphro",HSRLfile_path,HSRLfile_name,HSRLPixNo,nwl,ModeNo=3, updateYaml= False,releaseYAML= True, VertProfConstrain = True,LagrangianOnly = True)
HSRL_TamuT = HSLR_run("TAMU",HSRLfile_path,HSRLfile_name, HSRLPixNo,nwl,ModeNo=3,updateYaml= False,releaseYAML= True, VertProfConstrain = True,LagrangianOnly = True)

plot_HSRL(HSRL_sphrodT[0][0],HSRL_TamuT[0][0],UNCERT, forward = True, retrieval = True, Createpdf = True,PdfName ="/home/gregmi/ORACLES/rsltPdf/HSRL_Only_Plots_444.pdf", combinedVal =HSRL_TamuT[2])





#.....................................................................................

# PART 2: 2-step retrievals:

#1. Taking the retrieved values from RSP and using that to constrain HSRL:
#   But, first, we interpoalte the RSP values to HSRL wavelengths  (because they are different)


#.....................................................................................

## Interpolating retrieved values from RSP to HSRL wavelengths. 

def Interpolate(Inst1Retrievals, Inst2wl, NoMode = None, Plot=False, Inst1name = None, Inst2name = None):

    """

    Inst1Retrievals =GRASP retrieval result contaning the state variables
    
    Inst2wl = Wavelength to interpolate the Inst1Retrievals to 

    NoMode = Number of aerosol modes, if not provided then NoMode = 3


    Returns: 

    IntpDict = Dict with interpolated n and k 
    DictIntru2 = Dict with interpolated n and k along with other retrieved  values from instrument 1

    """

    IntpDict = {}


    if NoMode == None:
        NoMode = 3
    Instru2n, Instru2k = np.ones((NoMode,len(Inst2wl))),np.ones((NoMode,len(Inst2wl)))
    Instru1n,Instru1k, Inst1Lwl =Inst1Retrievals['n'],Inst1Retrievals['k'], Inst1Retrievals['lambda']

    for mode in range(len(Instru1n)): #loop for each mode
        
        fn = interp1d( Inst1Lwl,Instru1n[mode], kind='linear',fill_value="extrapolate")
        fk = interp1d( Inst1Lwl,Instru1k[mode], kind='linear',fill_value="extrapolate")
        Instru2n[mode] = fn(Inst2wl)
        Instru2k[mode] = fk(Inst2wl)
    
    if Plot == True:
        fig,ax = plt.subplots(2,3, figsize = (10,5) )
        for mode in range(len(Instru1n)):



            ax[0,mode].plot(Inst1Lwl, Instru1n[mode],'-',marker = 'o', label=Inst1name)
            ax[0,mode].plot(Inst2wl,Instru2n[mode], '-',marker = 'o', label= Inst2name)
            
            ax[1,mode].plot(Inst1Lwl, Instru1k[mode],'-',marker = 'o', label=Inst1name)
            ax[1,mode].plot(Inst2wl,Instru2k[mode], '-',marker = 'o', label= Inst2name)
                
                
        ax[0,0].legend()
        

    IntpDict['n'] = Instru2n
    IntpDict['k'] = Instru2k

    #Create a new dictinoary with the interpoated values. 

    DictIntru2 = {}

    DictIntru2['lambda'] = Inst2wl
    DictIntru2['n'] = Instru2n
    DictIntru2['k'] = Instru2k
    DictIntru2['rv'] = Inst1Retrievals['rv']
    DictIntru2['sigma'] = Inst1Retrievals['sigma']
    # DictIntru2['vol'] = Inst1Retrievals['vol']
    DictIntru2['sph'] = Inst1Retrievals['sph']



    return IntpDict,DictIntru2 


#.....................................................................................
## Interpolating retrieved values from RSP to HSRL wavelengths. 

Updatedict_sph = Interpolate(Inst1Retrievals =rslts_Sph[0] , Inst2wl =HSRL_sphrodT[0][0]['lambda'] , NoMode =3 , Plot= True, Inst1name = ' RSP', Inst2name = 'HSRL')
Updatedict_hex = Interpolate(Inst1Retrievals =rslts_Tamu[0] , Inst2wl =HSRL_sphrodT[0][0]['lambda'] , NoMode =3 , Plot= True, Inst1name = ' RSP', Inst2name = 'HSRL')

#.....................................................................................


#result dictionary for the measurements: 

rslt_RSP = Read_Data_RSP_Oracles(RSPfile_path, RSPfile_name,RSP_PixNo,ang1,ang2,TelNo, nwl,GasAbsFn)
rslt_HSRL1 = Read_Data_HSRL_Oracles_Height(HSRLfile_path,HSRLfile_name,HSRLPixNo)


#.....................................................................................



fwdModelYAMLpath ='/home/gregmi/git/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/settings_BCK_POLAR_3modes_Shape_ORACLE_NohgtConst.yml'
ymlPath = '/home/gregmi/git/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/'

krnlPath='/home/shared/GRASP_GSFC/src/retrieval/internal_files'
binPathGRASP ='/home/shared/GRASP_GSFC/build_HEX_v112/bin/grasp_app'


#Updated ymal Path is the name of the file that will contsin the new updated yaml 
UpdatedymlPath = UpdatedymlPath = '/home/gregmi/git/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/settings_LIDARandPOLAR_3modes_Shape_Sph_Update.yml'







"""    

def update_HSRLyaml(UpdatedymlPath, YamlFileName: str, noMod: int, Kernel_type: str,  
 GRASPModel = None, AeroProf =None, ConsType= None, YamlChar=None, maxr=None, minr=None, 
 NewVarDict: dict = None, DataIdxtoUpdate=None)

"""




Allrslt = {} #This will store all the GRASP results for 5%, Strictly constrained case. 



Up1 = update_HSRLyaml(UpdatedymlPath=UpdatedymlPath , YamlFileName = fwdModelYAMLpath , noMod = noMod , Kernel_type = 'hex', NewVarDict = Updatedict_hex[1],maxr=1.05, minr=0.95,  GRASPModel = 'bck')
pix = pixel()
pix.populateFromRslt(rslt_HSRL1[0], radianceNoiseFun=None, dataStage='meas', verbose=False)

# Set up graspRun class, add pixel, and run GRASP
gr1 = graspRun(pathYAML=Up1, releaseYAML=True, verbose=True)
gr1.addPix(pix)  # Add pixel to the graspRun instance
gr1.runGRASP(binPathGRASP=binPathGRASP, krnlPathGRASP=krnlPath)  # Run GRASP

# Print AOD at last wavelength
print('AOD at %5.3f μm was %6.4f.' % (gr1.invRslt[0]['lambda'][-1], gr1.invRslt[0]['aod'][-1]))

Allrslt['hex5'] = gr1.invRslt[0]


#...........................................................
#...........................................................


Up2= update_HSRLyaml(UpdatedymlPath=UpdatedymlPath , YamlFileName = fwdModelYAMLpath , noMod = noMod , Kernel_type = 'spheroid', NewVarDict = Updatedict_sph[1],maxr=1.05, minr=0.95,  GRASPModel = 'bck')
pix = pixel()
pix.populateFromRslt(rslt_HSRL1[0], radianceNoiseFun=None, dataStage='meas', verbose=False)

# Set up graspRun class, add pixel, and run GRASP
gr2 = graspRun(pathYAML=Up2, releaseYAML=True, verbose=True)
gr2.addPix(pix)  # Add pixel to the graspRun instance
gr2.runGRASP(binPathGRASP=binPathGRASP, krnlPathGRASP=krnlPath)  # Run GRASP

# Print AOD at last wavelength
print('AOD at %5.3f μm was %6.4f.' % (gr2.invRslt[0]['lambda'][-1], gr2.invRslt[0]['aod'][-1]))

Allrslt['sph5'] = gr2.invRslt[0]


#...........................................................
#...........................................................



Up3 = update_HSRLyaml(UpdatedymlPath=UpdatedymlPath , YamlFileName = fwdModelYAMLpath , noMod = noMod , Kernel_type = 'hex', NewVarDict = Updatedict_hex[1],maxr=1.001, minr=0.999,  GRASPModel = 'bck')
pix = pixel()
pix.populateFromRslt(rslt_HSRL1[0], radianceNoiseFun=None, dataStage='meas', verbose=False)

# Set up graspRun class, add pixel, and run GRASP
gr3 = graspRun(pathYAML=Up3, releaseYAML=True, verbose=True)
gr3.addPix(pix)  # Add pixel to the graspRun instance
gr3.runGRASP(binPathGRASP=binPathGRASP, krnlPathGRASP=krnlPath)  # Run GRASP

# Print AOD at last wavelength
print('AOD at %5.3f μm was %6.4f.' % (gr3.invRslt[0]['lambda'][-1], gr3.invRslt[0]['aod'][-1]))

Allrslt['hexStrict'] = gr3.invRslt[0]


#...........................................................
#...........................................................


Up4 = update_HSRLyaml(UpdatedymlPath=UpdatedymlPath , YamlFileName = fwdModelYAMLpath , noMod = noMod , Kernel_type = 'spheroid', NewVarDict = Updatedict_sph[1],maxr=1.001, minr=0.999,   GRASPModel = 'bck')
pix = pixel()
pix.populateFromRslt(rslt_HSRL1[0], radianceNoiseFun=None, dataStage='meas', verbose=False)

# Set up graspRun class, add pixel, and run GRASP
gr4 = graspRun(pathYAML=Up4, releaseYAML=True, verbose=True)
gr4.addPix(pix)  # Add pixel to the graspRun instance
gr4.runGRASP(binPathGRASP=binPathGRASP, krnlPathGRASP=krnlPath)  # Run GRASP

# Print AOD at last wavelength
print('AOD at %5.3f μm was %6.4f.' % (gr4.invRslt[0]['lambda'][-1], gr4.invRslt[0]['aod'][-1]))

Allrslt['sphStrict'] = gr4.invRslt[0]



#...............................................

#Plotting

#...............................................

plot_HSRL(Allrslt['sph5'],Allrslt['hex5'],UNCERT, forward = True, retrieval = True, Createpdf = True,PdfName ="/home/gregmi/ORACLES/rsltPdf/HSRL_Only_Plots_444.pdf", combinedVal =gr.invRslt[0])
plot_HSRL(Allrslt['sphStrict'],Allrslt['hexStrict'],UNCERT, forward = True, retrieval = True, Createpdf = True,PdfName ="/home/gregmi/ORACLES/rsltPdf/HSRL_Only_Plots_444.pdf", combinedVal =gr.invRslt[0])





#****************


fwdModelYAMLpath ='/home/gregmi/git/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/settings_BCK_POLAR_3modes_Shape_ORACLE_NohgtConst.yml'

Up = update_HSRLyaml(UpdatedymlPath=UpdatedymlPath , YamlFileName = fwdModelYAMLpath , noMod = noMod , Kernel_type = 'hex', NewVarDict = Updatedict_hex[1],maxr=1.05, minr=0.95,  GRASPModel = 'bck')
pix = pixel()
pix.populateFromRslt(rslt_HSRL1[0], radianceNoiseFun=None, dataStage='meas', verbose=False)

# Set up graspRun class, add pixel, and run GRASP
gr = graspRun(pathYAML=Up, releaseYAML=True, verbose=True)
gr.addPix(pix)  # Add pixel to the graspRun instance
gr.runGRASP(binPathGRASP=binPathGRASP, krnlPathGRASP=krnlPath)  # Run GRASP

# Print AOD at last wavelength
print('AOD at %5.3f μm was %6.4f.' % (gr.invRslt[0]['lambda'][-1], gr.invRslt[0]['aod'][-1]))

Allrslt['NoHgtConsthex5'] = gr.invRslt[0]


#...........................................................
#...........................................................


Up = update_HSRLyaml(UpdatedymlPath=UpdatedymlPath , YamlFileName = fwdModelYAMLpath , noMod = noMod , Kernel_type = 'spheroid', NewVarDict = Updatedict_sph,maxr=1.05, minr=0.95,  GRASPModel = 'bck')
pix = pixel()
pix.populateFromRslt(rslt_HSRL1[0], radianceNoiseFun=None, dataStage='meas', verbose=False)

# Set up graspRun class, add pixel, and run GRASP
gr = graspRun(pathYAML=Up, releaseYAML=True, verbose=True)
gr.addPix(pix)  # Add pixel to the graspRun instance
gr.runGRASP(binPathGRASP=binPathGRASP, krnlPathGRASP=krnlPath)  # Run GRASP

# Print AOD at last wavelength
print('AOD at %5.3f μm was %6.4f.' % (gr.invRslt[0]['lambda'][-1], gr.invRslt[0]['aod'][-1]))

Allrslt['NoHgtConstsph5'] = gr.invRslt[0]





#_______________________________________________________________________________________________________________________
#_______________________________________________________________________________________________________________________
# Constrain RSP search space using HSRL data
#_______________________________________________________________________________________________________________________
#_______________________________________________________________________________________________________________________

