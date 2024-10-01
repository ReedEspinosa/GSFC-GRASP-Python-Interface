

"""
# Greema Regmi, UMBC
# Date: Dec27, 2023
"""

import sys
from matplotlib import pyplot as plt
import numpy as np
import datetime as dt
from numpy import nanmean
import h5py 
sys.path.append("/home/gregmi/git/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases")
from architectureMap import returnPixel
from ORACLES_GRASP import FindPix, RSP_Run,  HSLR_run, LidarAndMAP, plot_HSRL,RSP_plot,CombinedLidarPolPlot,Ext2Vconc
import yaml
%matplotlib inline
from runGRASP import graspRun, pixel
import math

from netCDF4 import Dataset
import matplotlib.patches as mpatches
from matplotlib.ticker import ScalarFormatter
from scipy.interpolate import interp1d
import re



# Case1: 21st sept 2018, ORACLES
# file_path = '/home/gregmi/ORACLES/Case2'
# file_name ='/RSP1-P3_L1C-RSPCOL-CollocatedRadiances_20180921T180327Z_V003-20210421T232647Z.h5'
# HSRLfile_path = '/home/gregmi/ORACLES/Case2'
# HSRLfile_name =  "/HSRL2_P3_20180921_R2.h5"
# RSP_PixNo =  2900 #2687
# ang1 = 60
# ang2 = 120



# Case2: 1st AUG 2017, ORACLES
# file_path = '/home/gregmi/ORACLES/Case2O'
# file_name ='/RSP2-P3_L1C-RSPCOL-CollocatedRadiances_20170801T145459Z_V002-20210624T034922Z.h5'
# HSRLfile_path = '/home/gregmi/ORACLES/Case2O'
# HSRLfile_name =  "/HSRL2_P3_20170801_R1.h5"
# RSP_PixNo = 5690 #6404 #6013 #5280 #5690 #4644 #5690 #3600 # best5690 #5680  #6393 #6013 #6558 #6030#766 #6558 #6393   # worked 4656 #4644#5685 #6558
# ang1 = 80
# ang2 = 130



# Case 3: 26th, Oct 2018, ORACLES : 
# file_path = '/home/gregmi/ORACLES/Case_5_Oct_26'
# file_name ='/RSP1-P3_L1C-RSPCOL-CollocatedRadiances_20181026T102651Z_V003-20210514T201940Z.h5'
# HSRLfile_path = '/home/gregmi/ORACLES/Case_5_Oct_26'
# HSRLfile_name = '/HSRL2_P3_20181026_R2.h5'
# RSP_PixNo =  1980 #1950 #1975
# ang1 = 20
# ang2 = 135




# # # # ###Case 4: 24th Sept 2018, ORACLES
# file_path = '/home/gregmi/ORACLES/Sept24/'
# file_name ='RSP1-P3_L1C-RSPCOL-CollocatedRadiances_20180924T090316Z_V003-20210421T233034Z.h5'
# HSRLfile_path = '/home/gregmi/ORACLES/Sept24/'
# HSRLfile_name =  "HSRL2_P3_20180924_R2.h5"
# RSP_PixNo = 382
# TelNo = 0 # aggregated altitude. To obtain geometries corresponding to data from the 1880 nm channel, aggregation altitude should be set to 1, while aggregation altitude =0 should be used for all other channels.
# nwl = 5 # first  nwl wavelengths
# ang1 = 10
# ang2 = 135



# # '''##Case 5: 22nd sept 2018, ORACLES
# # # # #RSP'''
file_path = "/home/gregmi/ORACLES/RSP1-L1C_P3_20180922_R03/"  #Path to the ORACLE data file
file_name =  "/RSP1-P3_L1C-RSPCOL-CollocatedRadiances_20180922T151106Z_V003-20210421T233946Z.h5" #Name of the ORACLES file
#HSRL
HSRLfile_path = "/home/gregmi/ORACLES/HSRL" #Path to the ORACLE data file 
HSRLfile_name =  "/HSRL2_P3_20180922_R2.h5" #Name of the ORACLES file
RSP_PixNo = 13201
TelNo = 0 # aggregated altitude. To obtain geometries corresponding to data from the 1880 nm channel, aggregation altitude should be set to 1, while aggregation altitude =0 should be used for all other channels.
nwl = 5 # first  nwl wavelengths
ang1 = 10
ang2 = 120 # :ang angles  #Remove

#Uncertainity values for error bars, Take from AOS 
# UNCERT ={}
# UNCERT['aodMode'] = 0.1
# UNCERT['aod']= 0.1
# UNCERT['ssaMode'] = 0.03
# UNCERT['k'] = 0.003
# UNCERT['n'] = 0.025
# UNCERT['DP'] = 1
# UNCERT['VBS'] =  2.4e-6        #2e-7 # rel %
# UNCERT['VEXT'] = 0.1   #rel% 10%
# UNCERT['rv'] = 0.05
# UNCERT['I'] = 0.03
# UNCERT['DoLP'] = 0.005


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



#Path to the gas absorption (tau) values for gas absorption correction
GasAbsFn = '/home/gregmi/ORACLES/UNL_VRTM/shortwave_gas.unlvrtm.nc'
SpecResFnPath = '/home/gregmi/ORACLES/RSP_Spectral_Response/'
#This is required if we want to configure the HSRL yaml file based on the GRASP output for the RSP
noMod =3  #number of aerosol mode, here 2 for fine+coarse mode configuration
 #no of char # this is a varible added to char and modes to avoid char[0]/ mod[0] which doesnt exist

# AprioriLagrange = PixIndx
AprioriLagrange = [5e-2,1e-2,5e-1,1e-1,1]

SphLagrangian =[]
TAMULagrangian =[]

SphRSP =[]
TAMURSP =[]
RSPIndx=[]

SphJoint =[]
TAMUJoint =[]
jointIndx =[]

SphHSRL =[]
TAMUHSRL =[]
PixIndx =[]


for i in range(1):

    RSP_PixNo = RSP_PixNo +1
    TelNo = 0 # aggregated altitude. To obtain geometries corresponding to data from the 1880 nm channel, aggregation altitude should be set to 1, while aggregation altitude =0 should be used for all other channels.
    nwl = 5 # first  nwl wavelengths
    # ang1 = 20
    # ang2 = 100 # :ang angles  #Remove

    f1_MAP = h5py.File(file_path+file_name,'r+')   
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
    HSRLPixNo = FindPix(LatH,LonH,LatRSP,LonRSP)[0] + i # Or can manually give the index of the pixel that you are intrested in


# # for i in range(3):
#     HSRLPixNo = HSRLPixNo+1
#     RSP only retrieval
 
# # # #  Kernel_type = Run(Kernel_type) for spheriod, Kernel_type = 'TAMU' for hexahedral
    rslts_Sph = RSP_Run("sphro",file_path,file_name,RSP_PixNo,ang1,ang2,TelNo,nwl,GasAbsFn,ModeNo=3)
#     # rslts_Sph2 = RSP_Run("sphro",file_path,file_name,RSP_PixNo,ang1,ang2,TelNo,nwl,GasAbsFn,ModeNo=2)
    rslts_Tamu = RSP_Run("TAMU",file_path,file_name,RSP_PixNo,ang1,ang2,TelNo,nwl,GasAbsFn,ModeNo=3)
    # # rslts_Tamu2 = RSP_Run("TAMU",file_path,file_name,RSP_PixNo,ang1,ang2,TelNo,nwl,GasAbsFn,ModeNo=2)
    RSP_plot(rslts_Sph,rslts_Tamu,RSP_PixNo,UNCERT)
   

# # # #     # SphRSP.append(rslts_Sph)
# # # #     # TAMURSP.append(rslts_Tamu)
# # # #     # RSPIndx.append(RSP_PixNo)


# # # #    #HSRL2 only retrieval
    
    HSRL_sphrodT = HSLR_run("sphro",HSRLfile_path,HSRLfile_name,HSRLPixNo,nwl,ModeNo=3, updateYaml= False,releaseYAML= True, VertProfConstrain = True,LagrangianOnly = True,  AprioriLagrange =  AprioriLagrange[i])
    # plot_HSRL(HSRL_sphrodT[0][0],HSRL_sphrodT[0][0], UNCERT,forward = True, retrieval = True, Createpdf = True,PdfName ="/home/gregmi/ORACLES/rsltPdf/HSRL_Only_Plots_444.pdf", combinedVal =HSRL_sphrodT[2]) 
    HSRL_TamuT = HSLR_run("TAMU",HSRLfile_path,HSRLfile_name, HSRLPixNo,nwl,ModeNo=3,updateYaml= False,releaseYAML= True, VertProfConstrain = True,LagrangianOnly = True,  AprioriLagrange =  AprioriLagrange[i])
    # # plot_HSRL(HSRL_Tamu[0][0],HSRL_Tamu[0][0], forward = True, retrieval = True, Createpdf = True,PdfName ="/home/gregmi/ORACLES/rsltPdf/HSRL_Only_Plots_444.pdf", combinedVal =HSRL_Tamu[2])    
    plot_HSRL(HSRL_sphrodT[0][0],HSRL_TamuT[0][0],UNCERT, forward = True, retrieval = True, Createpdf = True,PdfName ="/home/gregmi/ORACLES/rsltPdf/HSRL_Only_Plots_444.pdf", combinedVal =HSRL_TamuT[2])

#     # # SphHSRL.append(HSRL_sphrodT)
    # TAMUHSRL.append(HSRL_TamuT)
    # PixIndx.append(HSRLPixNo)

# #    # joint RSP + HSRL2  retrieval 
  # # # PlotRandomGuess('gregmi/git/GSFC-GRASP-Python-Interface/try.npy', 2,0)
    # LidarPolTAMU = LidarAndMAP('TAMU',HSRLfile_path,HSRLfile_name,HSRLPixNo,file_path,file_name,RSP_PixNo,ang1,ang2,TelNo, nwl,GasAbsFn,ModeNo=3, updateYaml= None)
    
    LidarPolSph = LidarAndMAP('sphro',HSRLfile_path,HSRLfile_name,HSRLPixNo,file_path,file_name,RSP_PixNo,ang1,ang2,TelNo, nwl,GasAbsFn,ModeNo=3, updateYaml= None)
    
    SphJoint.append(LidarPolSph)
    TAMUJoint.append(LidarPolTAMU)

    CombinedLidarPolPlot(LidarPolSph[0],LidarPolTAMU[0],RSP_PixNo,UNCERT)
    plot_HSRL(LidarPolSph[0][0],LidarPolTAMU[0][0],UNCERT, forward = True, retrieval = True, Createpdf = True,PdfName ="/home/gregmi/ORACLES/rsltPdf/LIDARPOL_Plots_444.pdf")
    RSP_plot(LidarPolSph[0],LidarPolTAMU[0],RSP_PixNo,UNCERT,LIDARPOL=True)




# def ShowRSPdata(LidarPolSph[0]):
    
#     print("Solar zenith", LidarPolSph[0][0]['sza'])
#     print("Sca Angle interval: ",np.diff(LidarPolSph[0][0]['sca_ang'][:,1]))
#     print("Sca Angle interval: ",len(LidarPolSph[0][0]['sca_ang'][:,1]))

#     retrun

# # # # # # # #2modes"
# # #     HSRL_sphrod = HSLR_run("sphro",HSRLfile_path,HSRLfile_name,HSRLPixNo,nwl,ModeNo=2, updateYaml= False,releaseYAML= True)
# #     # plot_HSRL(HSRL_sphrod[0][0],HSRL_sphrod[0][0], forward = True, retrieval = True, Createpdf = True,PdfName ="/home/gregmi/ORACLES/rsltPdf/HSRL_Only_Plots_444.pdf", combinedVal =HSRL_sphrod[2]) 
#     HSRL_Tamu = HSLR_run("TAMU",HSRLfile_path,HSRLfile_name, HSRLPixNo,nwl,ModeNo=2,updateYaml= False,releaseYAML= True)


    # # print('Cost Value Sph, tamu: ',  rslts_Sph[0]['costVal'], rslts_Tamu[0]['costVal'])
    # RSP_plot(rslts_Sph,rslts_Tamu,RSP_PixNo)
    # plot_HSRL(rslts_Sph[0],rslts_Tamu[0], forward = False, retrieval = True, Createpdf = True,PdfName =f"/home/gregmi/ORACLES/rsltPdf/RSP_only_{RSP_PixNo}.pdf")
    
#     # Retrieval_type = 'NosaltStrictConst_final'
#     # #Running GRASP for HSRL, HSRL_sphrod = for spheriod kernels,HSRL_Tamu = Hexahedral kernels
    
#     print('Cost Value Sph, tamu: ',  HSRL_sphrod[0][0]['costVal'],HSRL_Tamu[0][0]['costVal'])
   
# # #     # # #Constrining HSRL retrievals by 5% 
#     HSRL_sphro_5 = HSLR_run("sphro",HSRLfile_path,HSRLfile_name,HSRLPixNo,nwl,ModeNo=3,updateYaml= True,releaseYAML= True) 
#     HSRL_Tamu_5 = HSLR_run("TAMU",HSRLfile_path,HSRLfile_name,HSRLPixNo,nwl,ModeNo=3,updateYaml= True,releaseYAML= True)

# # #     print('Cost Value Sph, tamu: ',  HSRL_sphro_5[0][0]['costVal'],HSRL_Tamu_5[0][0]['costVal'])

# # # #     #Strictly Constrining HSRL retrievals to values from RSP retrievals
#     HSRL_sphrod_strict = HSLR_run("sphro",HSRLfile_path,HSRLfile_name,HSRLPixNo,nwl,ModeNo=3,updateYaml= True, ConsType = 'strict',releaseYAML= True) 
#     HSRL_Tamu_strict = HSLR_run("TAMU",HSRLfile_path,HSRLfile_name,HSRLPixNo,nwl,ModeNo=3,updateYaml= True, ConsType = 'strict',releaseYAML= True)

# # #     print('Cost Value Sph, tamu: ',  HSRL_sphrod_strict[0][0]['costVal'],HSRL_Tamu_strict[0][0]['costVal'])
    
    # RIG = randomInitGuess('sphro',HSRLfile_path,HSRLfile_name,HSRLPixNo, nwl,releaseYAML =True, ModeNo=3)
     #Lidar+pol combined retrieval
    # LidarPolSph = LidarAndMAP('sphro',HSRLfile_path,HSRLfile_name,HSRLPixNo,file_path,file_name,RSP_PixNo,ang1,ang2,TelNo, nwl,GasAbsFn,ModeNo=3, updateYaml= None,RandinitGuess=True, NoItr =100 )
  




# LidarPolTAMU = LidarAndMAP('TAMU',HSRLfile_path,HSRLfile_name,HSRLPixNo,file_path,file_name,RSP_PixNo,ang1,ang2,TelNo, nwl,GasAbsFn,ModeNo=3, updateYaml= None,RandinitGuess=True, NoItr =150)
    # LidarPolSph = LidarAndMAP('sphro',HSRLfile_path,HSRLfile_name,HSRLPixNo,file_path,file_name,RSP_PixNo,ang1,ang2,TelNo, nwl,GasAbsFn,ModeNo=3, updateYaml= None,RandinitGuess=True, NoItr =150)
    # plot_HSRL(HSRL_sphrodT[0][0],HSRL_TamuT[0][0], UNCERT, forward = True, retrieval = True, Createpdf = True,PdfName ="/home/gregmi/ORACLES/rsltPdf/HSRL_Only_Plots_444.pdf", combinedVal =HSRL_sphrod[2])
    # plot_HSRL(HSRL_sphrodT[0][0],HSRL_TamuT[0][0],UNCERT, forward = True, retrieval = True, Createpdf = True,PdfName ="/home/gregmi/ORACLES/rsltPdf/HSRL_Only_Plots_444.pdf", combinedVal =HSRL_TamuT[2])




# # # #Vertical Prof Not constrained 
# # #     HSRL_sphrod = HSLR_run("sphro",HSRLfile_path,HSRLfile_name,HSRLPixNo,nwl,ModeNo=2, updateYaml= False, releaseYAML= True)
# # #     # plot_HSRL(HSRL_sphrod[0][0],HSRL_sphrod[0][0], forward = True, retrieval = True, Createpdf = True,PdfName ="/home/gregmi/ORACLES/rsltPdf/HSRL_Only_Plots_444.pdf", combinedVal =HSRL_sphrod[2]) 
# # #     HSRL_Tamu = HSLR_run("TAMU",HSRLfile_path,HSRLfile_name, HSRLPixNo,nwl,ModeNo=2,updateYaml= False,releaseYAML= True)
# # #     # plot_HSRL(HSRL_Tamu[0][0],HSRL_Tamu[0][0], forward = True, retrieval = True, Createpdf = True,PdfName ="/home/gregmi/ORACLES/rsltPdf/HSRL_Only_Plots_444.pdf", combinedVal =HSRL_Tamu[2])
   
# # # # # #Vertical Prof constrained 



def CombineHSRLandRSPrslt(BckHSRL,BckRSP,CharName=None):

    '''Combines the retrievals from HSRL and RSP'''

    if CharName == None:
        CharName = ['n','k','aodMode', 'ssaMode']
        WlIndepChar = ['rv', 'sigma', 'sph', 'vol']

    CombRetrievalDict ={}
   
    HSRLwl = BckHSRL['lambda']  #Wl for HSRL
    RSPwl = BckRSP['lambda']   #Wl for RSP

    #Combining the wavelengths
    Combinedwl = np.concatenate((HSRLwl,RSPwl))

    #Sorting the wavelength in ascending order.
    Combinedsortwl = Combinedwl[np.argsort(Combinedwl) ]

    #Index of each instrument in the 
    HSRLIdx = np.where(np.isin( Combinedsortwl, HSRLwl))
    RSPIdx = np.where(np.isin( Combinedsortwl, RSPwl))

    CombRetrievalDict['lambda'] = Combinedsortwl

    for Char in CharName:
        CombRetrievalDict[f'{Char}'] = np.zeros(( (len(BckHSRL['rv'])),len(Combinedsortwl),))
        
        for mode in range(len(BckHSRL['rv'])):  #mode = No of aerosol modes. 
            CombRetrievalDict[f'{Char}'] [mode][HSRLIdx] = BckHSRL[f'{Char}'] [mode]
            CombRetrievalDict[f'{Char}'] [mode][RSPIdx] = BckRSP[f'{Char}'] [mode]
        
    for Wlindep in  WlIndepChar: 
        CombRetrievalDict[f'HSRL_{Wlindep}'] = BckHSRL[f'{Wlindep}']
        CombRetrievalDict[f'RSP_{Wlindep}'] = BckRSP[f'{Wlindep}']


    return CombRetrievalDict


def update_HSRLyaml(ymlPath,UpKerFile, YamlFileName: str, noMod: int, Kernel_type: str,  
                    a=1, ConsType= None, YamlChar=None, maxr=None, minr=None, 
                    NewVarDict: dict = None, DataIdxtoUpdate=None): 
    """
    Update the YAML file for HSRL with initial conditions from polarimeter retrievals.
    
    Arguments:
    YamlFileName: str         -- Path to the YAML file to be updated.
    noMod: int                -- Number of aerosol modes to iterate over.
    Kernel_type: str          -- Type of kernel ('spheroid' or 'hex').
    ConsType: str             -- Constraint type ('strict' to fix retrieval).
    YamlChar: list or None    -- Characteristics in the YAML file (optional).
    maxr: float               -- Factor for maximum values.
    minr: float               -- Factor for minimum values.
    NewVarDict: dict          -- New variables to update the YAML with.
    RSP_rslt: dict            -- RSP results containing aerosol properties.
    LagrangianOnly: bool      -- Whether to only update Lagrangian multipliers.
    RSPUpdate: bool           -- Whether to update initial conditions from RSP retrievals.
    DataIdxtoUpdate: list     -- Index of data to update (optional).
    AprioriLagrange: float    -- Lagrange multiplier for a priori estimates.
    a: int                    -- Offset for characteristic indices (default: 0).
    ymlPath: str              -- Path to save the updated YAML file (default: '').
    UpKerFile: str            -- Filename for the updated YAML file (default: '').

    Returns:
    None -- Writes the updated YAML data to a new file.
    """
    
    if maxr is None: 
        maxr = 1

    if minr is None: 
        minr = 1

    # Load the YAML file
    with open(YamlFileName, 'r') as f:  
        data = yaml.safe_load(f)
    
    if Kernel_type =='spheroid':
        data['retrieval']['forward_model']['phase_matrix']['kernels_folder'] = 'KERNELS_BASE'
    if Kernel_type =='hex':
        data['retrieval']['forward_model']['phase_matrix']['kernels_folder'] = 'Ver_sph'

    if YamlChar is None:
        # Find the number of characteristics in the settings (Yaml) file
        NoChar = []  # No of characteristics in the YAML file
        for key, value in data['retrieval']['constraints'].items():
            # Match the strings with "characteristics"
            match = re.match(r'characteristic\[\d+\]', key)
            if match:
                # Extract the number 
                numbers = re.findall(r'\d+', match.group())
                NoChar.append(int(numbers[0]))

        # All the characteristics in the settings file
        YamlChar = [data['retrieval']['constraints'][f'characteristic[{i}]']['type'] for i in NoChar]

    assert NewVarDict is not None, "NewVarDict must not be None"

    for i, char_type in enumerate(YamlChar):
        for noMd in range(noMod):
           
            initCond = data['retrieval']['constraints'][f'characteristic[{i + a}]'][f'mode[{noMd + a}]']['initial_guess']
            
            if char_type == 'size_distribution_lognormal':
                try:
                    if len(NewVarDict['rv']) != 0 and len(NewVarDict['sigma']) != 0:
                        # Check if noMd is within range
                        if noMd < len(NewVarDict['rv']) and noMd < len(NewVarDict['sigma']):
                            initCond['value'] = (float(NewVarDict['rv'][noMd]), float(NewVarDict['sigma'][noMd]))
                            initCond['max'] = (float(NewVarDict['rv'][noMd] * maxr), float(NewVarDict['sigma'][noMd] * maxr))
                            initCond['min'] = (float(NewVarDict['rv'][noMd] * minr), float(NewVarDict['sigma'][noMd] * minr))
                        else:
                            print("Error: noMd index out of range for rv or sigma.")
                    else:
                        print("Warning: rv or sigma lists are empty.")
                except Exception as e:
                    print(f"An error occurred: {e}")
                    continue

                if ConsType == 'strict':
                    data['retrieval']['constraints'][f'characteristic[{i + a}]']['retrieved'] = 'false'
            
            elif char_type == 'real_part_of_refractive_index_spectral_dependent':
                
                try:
                    if len(NewVarDict['n'])!=0:

                        if DataIdxtoUpdate is None:
                            DataIdxtoUpdate = [i for i in range(len(NewVarDict['lambda']))]

                        initCond['value'] = [float(NewVarDict['n'][noMd][i]) for i in DataIdxtoUpdate]
                        initCond['max'] = [float(NewVarDict['n'][noMd][i] * maxr) for i in DataIdxtoUpdate]
                        initCond['min'] = [float(NewVarDict['n'][noMd][i] * minr) for i in DataIdxtoUpdate]
                        initCond['index_of_wavelength_involved'] = [i for i in range(len(NewVarDict['lambda']))]

                    if ConsType == 'strict':
                        data['retrieval']['constraints'][f'characteristic[{i + a}]']['retrieved'] = 'false'
                except Exception as e:
                    print(f"An error occurred: {e}")
                    continue
            elif char_type == 'imaginary_part_of_refractive_index_spectral_dependent':
                try:
                    if len(NewVarDict['k'])!=0:

                        if DataIdxtoUpdate is None:
                            DataIdxtoUpdate = [i for i in range(len(NewVarDict['lambda']))]

                        initCond['value'] = [float(NewVarDict['k'][noMd][i]) for i in DataIdxtoUpdate]
                        initCond['max'] = [float(NewVarDict['k'][noMd][i] * maxr) for i in DataIdxtoUpdate]
                        initCond['min'] = [float(NewVarDict['k'][noMd][i] * minr) for i in DataIdxtoUpdate]
                        initCond['index_of_wavelength_involved'] = [i for i in range(len(NewVarDict['lambda']))]

                    if ConsType == 'strict':
                        data['retrieval']['constraints'][f'characteristic[{i + a}]']['retrieved'] = 'false'
                except Exception as e:
                    print(f"An error occurred: {e}")
                    continue
            elif char_type == 'sphere_fraction':

                try:
                    if len(NewVarDict['sph'])!=0:

                        initCond['value'] = float(NewVarDict['sph'][noMd] / 100)
                        initCond['max'] = float(NewVarDict['sph'][noMd] * maxr / 100)
                        initCond['min'] = float(NewVarDict['sph'][noMd] * minr / 100)

                        if ConsType == 'strict':
                            data['retrieval']['constraints'][f'characteristic[{i + a}]']['retrieved'] = 'false'
                except Exception as e:
                    print(f"An error occurred: {e}")
                    continue
    # Save the updated YAML file
    with open(ymlPath + UpKerFile, 'w') as f:
        yaml.safe_dump(data, f)

    print(f"YAML file updated and saved as {UpKerFile}")

    return UpKerFile


def RunGRASPwUpdateYaml(Shape: str, ymlPath: str, fwdModelYAMLpath: str, UpKerFile: str, binPathGRASP: str, krnlPath: str, UpdateDict, rsltDict,  Plot = False):
    """
    Updates the YAML file, runs GRASP simulation, and appends the results to SaveRsltDict.
    
    Parameters:
        ymlPath (str): The base path for the YAML files.
        fwdModelYAMLpath (str): Path to the forward model YAML file.
        UpKerFile (str): The name of the updated kernel file.
        rsltDict: rsltdict to populate the pixel.
        binPathGRASP (str): Path to the GRASP binary.
        krnlPath (str): Path to the GRASP kernel.
        UpdateDictHex (dict): Dictionary of new variables to update the kernel.
        SaveRsltDict (list): List to store the results.

    Returns:
        None
    """
    
    # Update the YAML with the provided parameters in UpdateDictHex
    update_HSRLyaml(ymlPath, UpKerFile, YamlFileName=fwdModelYAMLpath, 
                    minr=0.995, maxr=1.015, noMod=3, Kernel_type = Shape, 
                    NewVarDict=UpdateDict, a=1)
    
    # Define updated YAML path
    Updatedyaml = ymlPath + UpKerFile
    
    # Create and populate a pixel from HSRL result
    pix = pixel()
    pix.populateFromRslt(rsltDict, radianceNoiseFun=None, dataStage='meas', verbose=False)
    
    # Set up graspRun class, add pixel, and run GRASP
    gr = graspRun(pathYAML=Updatedyaml, releaseYAML=True, verbose=True)
    gr.addPix(pix)  # Add pixel to the graspRun instance
    gr.runGRASP(binPathGRASP=binPathGRASP, krnlPathGRASP=krnlPath)  # Run GRASP
    
    # Print AOD at last wavelength
    print('AOD at %5.3f μm was %6.4f.' % (gr.invRslt[0]['lambda'][-1], gr.invRslt[0]['aod'][-1]))
     
    if Plot == True:
        plot_HSRL(gr.invRslt[0],gr.invRslt[0],UNCERT, forward = True, retrieval = True, Createpdf = True,PdfName ="/home/gregmi/ORACLES/rsltPdf/HSRL_Only_Plots_444.pdf", combinedVal =HSRL_TamuT[2])
    
    return gr.invRslt[0]


def Interpolate(HSRLRslt, RSPwl, NoMode = None):

    IntpDict = {}

    if NoMode == None:
        NoMode = 3
    RSPn, RSPk = np.ones((NoMode,len(RSPwl))),np.ones((NoMode,len(RSPwl)))
    HSRLn,HSRLk, HSRLwl =HSRLRslt['n'],HSRLRslt['k'], HSRLRslt['lambda']

    for mode in range(len(HSRLn)): #loop for each mode
        
        fn = interp1d( HSRLwl,HSRLn[mode], kind='linear',fill_value="extrapolate")
        fk = interp1d( HSRLwl,HSRLk[mode], kind='linear',fill_value="extrapolate")
        RSPn[mode] = fn(RSPwl)
        RSPk[mode] = fk(RSPwl)

        plt.plot(HSRLwl, HSRLn[mode],'-',marker = 'o', label='HSRL')
        plt.plot(RSPwl,RSPn[mode], '-',marker = 'o', label='RSP')
        plt.legend()
        plt.show()

    IntpDict['n'] =  RSPn
    IntpDict['k'] =  RSPk

    return IntpDict


# All_cases ={}
# fwdModelYAMLpath = '/home/gregmi/git/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/settings_BCK_DoLP_POLAR_3modes_HexShape_ORACLE_case2.yml'

# Shape = 'spheroid'
# rsltDict = rslt_RSP
# UpdateDict = {'rv': HSRL_sphrodT[0][0]['rv'],'sigma': HSRL_sphrodT[0][0]['sigma'] }

# Sph_HSRLsizetoRSP= RunGRASPwUpdateYaml(Shape, ymlPath, fwdModelYAMLpath, UpKerFile, binPathGRASP, krnlPath, UpdateDict, rsltDict,  Plot = False)

# All_cases['Sph_HSRLsizetoRSP'].append(Sph_HSRLsizetoRSP)







#To simulate the coefficient to convert Vext to vconc

# krnlPath='/home/shared/GRASP_GSFC/src/retrieval/internal_files'
# fwdModelYAMLpath ='/home/gregmi/git/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/settings_dust_Vext_conc.yml'
# binPathGRASP ='/home/shared/GRASP_GSFC/build_HEX_v112/bin/grasp_app'

# botLayer = 300
# topLayer = 5000
# Nlayers = 15

# # singProf = np.linspace(botLayer, topLayer, Nlayers)[::-1]
# #Inputs for simulation
# wvls = [0.532] # wavelengths in μm
# msTyp = [36] # meas type VEXT
# sza = 0.01 # we assume vertical lidar

# nbvm = Nlayers*np.ones(len(msTyp), int)
# thtv = np.tile(singProf, len(msTyp))
# meas = np.r_[np.repeat(2.372179e-05, nbvm[0])]
# phi = np.repeat(0, len(thtv)) # currently we assume all observations fall within a plane
# # errStr = [y for y in archName.lower().split('+') if 'lidar09' in y][0]
# nowPix = pixel(dt.datetime.now(), 1, 1, 0, 0, masl=0, land_prct=0)

# for wvl in wvls: # This will be expanded for wavelength dependent measurement types/geometry
#     # errModel = functools.partial(addError, errStr, concase=concase, orbit=orbit, lidErrDir=lidErrDir) # this must link to an error model in addError() below
#     nowPix.addMeas(wvl, msTyp, nbvm, sza, thtv, phi, meas,)

# print('Using settings file at %s' % fwdModelYAMLpath)
# gr = graspRun(pathYAML=fwdModelYAMLpath, releaseYAML=True, verbose=True) # setup instance of graspRun class, add the above pixel, run grasp and read the output to gr.invRslt[0] (dict)
# gr.addPix(nowPix) # add the pixel we created above
# gr.runGRASP(binPathGRASP=binPathGRASP, krnlPathGRASP=krnlPath) # run grasp binary (output read to gr.invRslt[0] [dict])
# print('AOD at %5.3f μm was %6.4f.' % (gr.invRslt[0]['lambda'][-1],gr.invRslt[0]['aod'][-1])) # 

def PlotSensitivity(SphLagrangian,TAMULagrangian, variableName):
        Index1 = [0,1,2]
        Index2 = [0,1,2]
        # AprioriLagrange = [1e-2,1e-1,1]

        fig, axs1= plt.subplots(nrows = 1, ncols =3, figsize= (20,10))  #TODO make it more general which adjust the no of rows based in numebr of wl
        plt.subplots_adjust(top=0.78)

        fig, axs2= plt.subplots(nrows = 1, ncols =3, figsize= (20,10))
        plt.subplots_adjust(top=0.78)
        
        fig, axs= plt.subplots(nrows = 1, ncols =2, figsize= (15,10))
       
        for j in range(len(variableName)):
            # Hsph = SphLagrangian[j][0][0]
            # HTam = TAMULagrangian[j][0][0]

            Hsph = SphLagrangian[j][0][0]
            HTam = TAMULagrangian[j][0][0]




    #Converting range to altitude
            altd = (Hsph['RangeLidar'][:,0])/1000 #altitude for spheriod
            altT = (Hsph['RangeLidar'][:,0])/1000 #altitude for hexahedra
            
            for i in range(3):
                wave = str(Hsph['lambda'][i]) +"μm"
                axs1[i].plot(Hsph['fit_VBS'][:,Index1[i]],altd, marker = "$O$",label =f"Sph{variableName[j]}",alpha =0.8)
                axs1[i].plot(HTam['fit_VBS'][:,Index2[i]],altd,ls = "--", label=f"Hex{variableName[j]}", marker = "h")

                axs1[i].xaxis.set_major_formatter(ScalarFormatter(useMathText=True))

                # print(UNCERT)
                # axs[i].errorbar(Hsph['fit_VBS'][:,i],altd,xerr= UNCERT['VBS'],color = "#025043", marker = "$O$",label ="Sphd")
                # axs[i].errorbar(HTam['fit_VBS'][:,i],altd,xerr= UNCERT['VBS'],color = "#d24787",ls = "--", label="Hex", marker = "h")

                axs1[i].set_xlabel(f'$ VBS (m^{-1}Sr^{-1})$')
                axs1[0].set_ylabel('Height above ground (km)')

                axs1[i].set_title(wave)

                if j ==0:
                    axs1[i].errorbar(Hsph['meas_VBS'][:,i],altd,xerr= UNCERT['VBS'],color = "#695E93", capsize=3,capthick =1,alpha =0.4, label =f"{UNCERT['VBS']}")
                
                    axs1[i].plot(Hsph['meas_VBS'][:,Index1[i]],altd, marker =">",color = "#281C2D", label ="Meas")
                

                if (i ==0 ):
                    # axs1[i].errorbar(Hsph['meas_VBS'][:,i],altd,xerr= UNCERT['VBS'],color = "#695E93", capsize=3,capthick =1,alpha =0.4, label =f"{UNCERT['VBS']}")
                
                    # axs1[i].plot(Hsph['meas_VBS'][:,Index1[i]],altd, marker =">",color = "#281C2D", label ="Meas")
                
                    axs1[0].legend(prop = { "size": 10 })
                # plt.suptitle(f"HSRL Vertical Backscatter profile  \nLat,Lon: {HTam['latitude']}, {HTam['longitude']} Date: {HTam['datetime']}\n ") #Initial condition strictly constrainted by RSP retrievals
            # pdf_pages.savefig()
            fig.savefig(f'/home/gregmi/ORACLES/HSRL_RSP/FIT_HSRL Vertical Backscatter profile_Lagrangian_multiplier .png',dpi = 300)
                # plt.tight_layout()
            

            # AOD = np.trapz(Hsph['meas_VExt'],altd)
            for i in range(3):
                wave = str(Hsph['lambda'][i]) +"μm"

                axs2[i].plot(Hsph['fit_DP'][:,Index1[i]],altd, marker = "$O$",label =f"Sph{variableName[j]}")
                axs2[i].plot(HTam['fit_DP'][:,Index2[i]],altd, ls = "--",marker = "h",label=f"Hex{variableName[j]}")
                
                # axs[i].errorbar(Hsph['fit_DP'][:,i],altd,xerr= UNCERT['DP'],fmt='-o',color = "#025043", marker = "$O$",label ="Sphd")
                # axs[i].errorbar(HTam['fit_DP'][:,i],altd,xerr= UNCERT['DP'],fmt='-o',color = "#d24787",ls = "--", label="Hex", marker = "h")


                axs2[i].set_xlabel('DP %')
                axs2[i].set_title(wave)
                if j ==0:
                    axs2[i].errorbar(Hsph['meas_DP'][:,Index1[i]],altd,xerr= UNCERT['DP'],color = '#695E93', capsize=4,capthick =1,alpha =0.6, label =f"{UNCERT['DP']}%")
                    
                    axs2[i].plot(Hsph['meas_DP'][:,Index1[i]],altd, marker =">",color = "#281C2D", label ="Meas")
               
                if (i ==0 ):
                    axs2[0].legend(prop = { "size": 10 })
                axs2[0].set_ylabel('Height above ground (km)')
                # plt.suptitle(f"HSRL Depolarization Ratio \n Lat,Lon: {HTam['latitude']}, {HTam['longitude']}  Date: {HTam['datetime']}\n ") #Initial condition strictly constrainted by RSP retrievals
            # pdf_pages.savefig()
            fig.savefig(f'/home/gregmi/ORACLES/HSRL_RSP/FIT_HSRL Depolarization Ratio_LagrangianMultiplier.png',dpi = 300)
            plt.subplots_adjust(top=0.78)
            for i in range(2):
                
                wave = str(Hsph['lambda'][i]) +"μm"
                # axs[i].xaxis.set_major_formatter(ScalarFormatter(useMathText=True))

                axs[i].plot(1000*Hsph['fit_VExt'][:,Index1[i]],altd, marker = "$O$",label =f"Sph{variableName[j]}")
                axs[i].plot(1000*HTam['fit_VExt'][:,Index2[i]],altd,ls = "--", marker = "h",label=f"Hex{variableName[j]}")
                
                axs[i].xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
                axs[i].ticklabel_format(style="sci", axis="x")
                # axs[i].errorbar(Hsph['fit_VExt'][:,i],altd,xerr= UNCERT['VEXT'],color = "#025043", marker = "$O$",label ="Sphd")
                # axs[i].errorbar(HTam['fit_VExt'][:,i],altd,xerr= UNCERT['VEXT'],color = "#d24787",ls = "--", label="Hex", marker = "h")

                
                
                axs[i].set_xlabel(f'$VExt (km^{-1})$')
                axs[0].set_ylabel('Height above ground (km)')
                axs[i].set_title(wave)
                axs[i].xaxis.set_major_formatter(ScalarFormatter(useMathText=True))


                if j ==0:
                    axs[i].errorbar(1000*Hsph['meas_VExt'][:,Index1[i]],altd,xerr= UNCERT['VEXT']*1000*Hsph['meas_VExt'][:,i],color = '#695E93',capsize=4,capthick =1,alpha =0.7, label =f"{UNCERT['VEXT']*100}%")
                
                    axs[i].plot(1000*Hsph['meas_VExt'][:,Index1[i]],altd, marker =">",color = "#281C2D", label ="Meas")
                
                if i ==0:
                   
                    axs[0].legend(prop = { "size": 10 })
                # plt.suptitle(f"HSRL Vertical Extinction profile TAMU\n Lat,Lon: {HTam['latitude']},{HTam['longitude']}  Date: {HTam['datetime']}\n ") #Initial condition strictly constrainted by RSP retrievals
            plt.tight_layout()
            # pdf_pages.savefig()
            fig.savefig(f'/home/gregmi/ORACLES/HSRL_RSP/FIT_HSRL_Vertical_Ext_profile_Var1.png',dpi = 300)

def PlotSensitivity2(SphLagrangian,TAMULagrangian):
        Index1 = [0,1,2]
        Index2 = [0,1,2]
        # AprioriLagrange = [1e-2,1e-1,1]

        fig, axs1= plt.subplots(nrows = 1, ncols =3, figsize= (15,6))  #TODO make it more general which adjust the no of rows based in numebr of wl
        plt.subplots_adjust(top=0.78)

        fig, axs2= plt.subplots(nrows = 1, ncols =3, figsize= (15,6))
        plt.subplots_adjust(top=0.78)
        
        fig, axs= plt.subplots(nrows = 1, ncols =2, figsize= (11,6))
       
        for j in range(5):
            Hsph = SphLagrangian[j][0][0]
            HTam = TAMULagrangian[j][0][0]




    #Converting range to altitude
            altd = (Hsph['RangeLidar'][:,0])/1000 #altitude for spheriod
            altT = (Hsph['RangeLidar'][:,0])/1000 #altitude for hexahedra
            
            for i in range(3):
                wave = str(Hsph['lambda'][i]) +"μm"
                axs1[i].plot(Hsph['fit_VBS'][:,Index1[i]],altd, marker = "$O$",label =f"{VariName[j]}",alpha =0.8)
                # axs1[i].plot(HTam['fit_VBS'][:,Index2[i]],altd,ls = "--", label=f"{AprioriLagrange[j]}", marker = "h")

                axs1[i].xaxis.set_major_formatter(ScalarFormatter(useMathText=True))

                # print(UNCERT)
                # axs[i].errorbar(Hsph['fit_VBS'][:,i],altd,xerr= UNCERT['VBS'],color = "#025043", marker = "$O$",label ="Sphd")
                # axs[i].errorbar(HTam['fit_VBS'][:,i],altd,xerr= UNCERT['VBS'],color = "#d24787",ls = "--", label="Hex", marker = "h")

                axs1[i].set_xlabel(f'$ VBS (m^{-1}Sr^{-1})$')
                axs1[0].set_ylabel('Height above ground (km)')

                axs1[i].set_title(wave)

                if j ==0:
                    axs1[i].errorbar(Hsph['meas_VBS'][:,i],altd,xerr= UNCERT['VBS'],color = "#695E93", capsize=3,capthick =1,alpha =0.4, label =f"{UNCERT['VBS']}")
                
                    axs1[i].plot(Hsph['meas_VBS'][:,Index1[i]],altd, marker =">",color = "#281C2D", label ="Meas")
                

                if (i ==0 ):
                    # axs1[i].errorbar(Hsph['meas_VBS'][:,i],altd,xerr= UNCERT['VBS'],color = "#695E93", capsize=3,capthick =1,alpha =0.4, label =f"{UNCERT['VBS']}")
                
                    # axs1[i].plot(Hsph['meas_VBS'][:,Index1[i]],altd, marker =">",color = "#281C2D", label ="Meas")
                
                    axs1[0].legend()
                plt.suptitle(f"HSRL Vertical Backscatter profile  \nLat,Lon: {HTam['latitude']}, {HTam['longitude']} Date: {HTam['datetime']}\n ") #Initial condition strictly constrainted by RSP retrievals
            # pdf_pages.savefig()
            fig.savefig(f'/home/gregmi/ORACLES/HSRL_RSP/FIT_HSRL Vertical Backscatter profile_Lagrangian_multiplier .png',dpi = 300)
                # plt.tight_layout()
            

            # AOD = np.trapz(Hsph['meas_VExt'],altd)
            for i in range(3):
                wave = str(Hsph['lambda'][i]) +"μm"

                axs2[i].plot(Hsph['fit_DP'][:,Index1[i]],altd, marker = "$O$",label =f"{AprioriLagrange[j]}")
                # axs2[i].plot(HTam['fit_DP'][:,Index2[i]],altd, ls = "--",marker = "h",label=f"{AprioriLagrange[j]}")
                
                # axs[i].errorbar(Hsph['fit_DP'][:,i],altd,xerr= UNCERT['DP'],fmt='-o',color = "#025043", marker = "$O$",label ="Sphd")
                # axs[i].errorbar(HTam['fit_DP'][:,i],altd,xerr= UNCERT['DP'],fmt='-o',color = "#d24787",ls = "--", label="Hex", marker = "h")


                axs2[i].set_xlabel('DP %')
                axs2[i].set_title(wave)
                if j ==0:
                    axs2[i].errorbar(Hsph['meas_DP'][:,Index1[i]],altd,xerr= UNCERT['DP'],color = '#695E93', capsize=4,capthick =1,alpha =0.6, label =f"{UNCERT['DP']}%")
                    
                    axs2[i].plot(Hsph['meas_DP'][:,Index1[i]],altd, marker =">",color = "#281C2D", label ="Meas")
               
                if (i ==0 ):
                    axs2[0].legend()
                axs2[0].set_ylabel('Height above ground (km)')
                plt.suptitle(f"HSRL Depolarization Ratio \n Lat,Lon: {HTam['latitude']}, {HTam['longitude']}  Date: {HTam['datetime']}\n ") #Initial condition strictly constrainted by RSP retrievals
            # pdf_pages.savefig()
            fig.savefig(f'/home/gregmi/ORACLES/HSRL_RSP/FIT_HSRL Depolarization Ratio_LagrangianMultiplier.png',dpi = 300)
            plt.subplots_adjust(top=0.78)
            for i in range(2):
                wave = str(Hsph['lambda'][i]) +"μm"

                axs[i].plot(Hsph['fit_VExt'][:,Index1[i]],altd, marker = "$O$",label =f"{AprioriLagrange[j]}")
                # axs[i].plot(HTam['fit_VExt'][:,Index2[i]],altd,ls = "--", marker = "h",label=f"{AprioriLagrange[j]}")
                
                axs[i].xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
                axs[i].ticklabel_format(style="sci", axis="x")
                # axs[i].errorbar(Hsph['fit_VExt'][:,i],altd,xerr= UNCERT['VEXT'],color = "#025043", marker = "$O$",label ="Sphd")
                # axs[i].errorbar(HTam['fit_VExt'][:,i],altd,xerr= UNCERT['VEXT'],color = "#d24787",ls = "--", label="Hex", marker = "h")

                
                
                axs[i].set_xlabel(f'$VExt (m^{-1})$')
                axs[0].set_ylabel('Height above ground (km)')
                axs[i].set_title(wave)

                if j ==0:
                    axs[i].errorbar(Hsph['meas_VExt'][:,Index1[i]],altd,xerr= UNCERT['VEXT']*Hsph['meas_VExt'][:,i],color = '#695E93',capsize=4,capthick =1,alpha =0.7, label =f"{UNCERT['VEXT']*100}%")
                
                    axs[i].plot(Hsph['meas_VExt'][:,Index1[i]],altd, marker =">",color = "#281C2D", label ="Meas")
                
                if i ==0:
                   
                    axs[0].legend()
                plt.suptitle(f"HSRL Vertical Extinction profile TAMU\n Lat,Lon: {HTam['latitude']},{HTam['longitude']}  Date: {HTam['datetime']}\n ") #Initial condition strictly constrainted by RSP retrievals
            plt.tight_layout()
            # pdf_pages.savefig()
            fig.savefig(f'/home/gregmi/ORACLES/HSRL_RSP/FIT_HSRL_Vertical_Ext_profile_lagragian_multiplier.png',dpi = 300)


def errPlots(rslts_Sph,rslts_Tamu):


    ErrSph = 100*(abs(rslts_Sph[0]['meas_I'] - rslts_Sph[0]['fit_I']))/rslts_Sph[0]['meas_I']
    ErrHex = 100*(abs(rslts_Tamu[0]['meas_I'] - rslts_Tamu[0]['fit_I']))/rslts_Tamu[0]['meas_I']


    #This fucntion the error between the measuremtn and the fit from RSP for all wl and all scattering angles.
    wl = rslts_Sph[0]['lambda']
    colorwl = ['#70369d','#4682B4','#01452c','#FF7F7F','#d4af37','#4c0000']
    colorwl = ['#14411b','#94673E','#00B2FF',"#882255",'#FE5200','#8F64B3',"#DEA520",'#44440F','#212869']
            
    
    
    fig, axs = plt.subplots(nrows= 1, ncols=2, figsize=(16, 9), sharey =True)
    for i in range(len(wl)):
        axs[0].plot(rslts_Sph[0]['sca_ang'][:,i], ErrSph[:,i] ,color = colorwl[i],marker = "$O$",markersize= 10, lw = 0.2,label=f"{np.round(wl[i],2)}")
        axs[1].plot(rslts_Tamu[0]['sca_ang'][:,i], ErrHex[:,i],color = colorwl[i], marker = 'H',markersize= 12, lw = 0.2,label=f"{np.round(wl[i],2)}" )
    axs[0].set_title('Spheriod')
    axs[1].set_title('Hexahedral')

    axs[0].plot(rslts_Sph[0]['sca_ang'][:,0], 100*UNCERT['I']*np.ones(len(ErrSph[:,i])) ,color = 'k',ls = '--', lw = 1,label=f"Meas Uncert")
    axs[1].plot(rslts_Sph[0]['sca_ang'][:,0], 100*UNCERT['I']*np.ones(len(ErrSph[:,i])) ,color = 'k',ls = '--', lw = 1,label=f"Meas Uncert")


    axs[0].set_ylabel('Error I %')
    # axs[1].set_ylabel('Error %')
    plt.legend( ncol=2)

    axs[0].set_xlabel(r'${\theta_s}$')
    axs[1].set_xlabel(r'${\theta_s}$')
    plt.suptitle("RSP-Only Case 1")
    # plt.savefig(f'{file_name[2:]}_{RSP_PixNo}_ErrorI.png', dpi = 300)

    #Absolute err because DOLP are in %
    ErrSphP = 100*(abs(rslts_Sph[0]['meas_P_rel'] - rslts_Sph[0]['fit_P_rel']))
    ErrHexP = 100*(abs(rslts_Tamu[0]['meas_P_rel'] - rslts_Tamu[0]['fit_P_rel']))

        
    
    fig, axs = plt.subplots(nrows= 1, ncols=2, figsize=(16, 9), sharey =True)
    for i in range(len(wl)):
        axs[0].plot(rslts_Sph[0]['sca_ang'][:,i], ErrSphP[:,i] ,color = colorwl[i],marker = "$O$",markersize= 10, lw = 0.2,label=f"{np.round(wl[i],2)}")
        axs[1].plot(rslts_Tamu[0]['sca_ang'][:,i], ErrHexP[:,i],color = colorwl[i], marker = 'H',markersize= 12, lw = 0.2,label=f"{np.round(wl[i],2)}" )
    axs[0].set_title('Spheriod')
    axs[1].set_title('Hexahedral')

    axs[0].plot(rslts_Sph[0]['sca_ang'][:,0], 100*UNCERT['DoLP']*np.ones(len(ErrSph[:,i])) ,color = 'k',ls = '--', lw = 1,label=f"Meas Uncert")
    axs[1].plot(rslts_Sph[0]['sca_ang'][:,0], 100*UNCERT['DoLP']*np.ones(len(ErrSph[:,i])) ,color = 'k',ls = '--', lw = 1,label=f"Meas Uncert")


    axs[0].set_ylabel('Error DoLP %')
    # axs[1].set_ylabel('Error %')
    plt.legend( ncol=2)

    axs[0].set_xlabel(r'${\theta_s}$')
    axs[1].set_xlabel(r'${\theta_s}$')
    plt.suptitle("RSP-Only Case 1")

def PlotSingle(rslts_Sph,HSRL_sphrodT):

    #Combine the data from Lidar only and Map only 

    sort_MAP = np.array([1, 2, 4, 5, 6])
    sort_Lidar = np.array([0, 3, 7])

    keys = ['lambda', 'aodMode', 'ssaMode', 'n', 'k', 'aod']
    combdict ={}

    combdict['lidarVol'] = HSRL_sphrodT[0][0]['dVdlnr']
    combdict['lidarR'] = HSRL_sphrodT[0][0]['r']
    
    combdict['MapVol'] = rslts_Sph[0]['dVdlnr']
    combdict['MapR'] = rslts_Sph[0]['r']
    for key in keys:

        

        if len(rslts_Sph[0][key].shape)==1:
            Val2 = np.ones((8))
            Val2[sort_MAP ] =rslts_Sph[0][key]
            Val2[sort_Lidar] = HSRL_sphrodT[0][0][key]

            combdict[key] = Val2

        else:
            Val2 = np.ones((rslts_Sph[0][key].shape[0],8))
            Val2[:,sort_MAP ] =rslts_Sph[0][key]
            Val2[:,sort_Lidar] = HSRL_sphrodT[0][0][key]

            combdict[key] = Val2

    return combdict 


def Plotcomb(rslts_Sph2,rslts_Tamu2,LidarPolSph,LidarPolTAMU):


    # plot_HSRL(LidarPolSph[0][0],LidarPolTAMU[0][0],UNCERT, forward = False, retrieval = True, Createpdf = True,PdfName ="/home/gregmi/ORACLES/rsltPdf/LIDARPOL_Plots_444.pdf")
    
    if (rslts_Sph2!=None )and (rslts_Tamu2!=None):
        RepMode =2
    else:
        RepMode =1


    Spheriod,Hex = rslts_Sph2,rslts_Tamu2

    cm_sp = ['k','#8B4000', '#87C1FF']
    cm_t = ["#BC106F",'#E35335', 'b']
    color_sph = '#0c7683'
    color_tamu = "#BC106F"

    fig, axs2 = plt.subplots(nrows= 3, ncols=2, figsize=(35, 20))
    fig, axsErr = plt.subplots(nrows= 3, ncols=2, figsize=(20, 15))
        


    for i in range(RepMode):
        if i ==1:
            Spheriod,Hex = rslts_Sph2,rslts_Tamu2
            cm_sp = ['#b54cb5','#14411b', '#87C1FF']
            cm_t = ["#14411b",'#adbf4b', 'b']
            color_sph = '#adbf4b'
            color_tamu = "#936ecf"

        plt.rcParams['font.size'] = '26'
        plt.rcParams["figure.autolayout"] = True
        #Stokes Vectors Plot
        date_latlon = ['datetime', 'longitude', 'latitude']
        Xaxis = ['MapR','lambda','sca_ang','rv','height']
        Retrival = ['dVdlnr','aodMode','ssaMode','n', 'k']
        RetrivalMAP = ['MapVol','aodMode','ssaMode','n', 'k']
        RetrivalLidar = ['lidarVol','aodMode','ssaMode','n', 'k']
        #['sigma', 'vol', 'aodMode','ssaMode', 'rEff', 'costVal']
        Angles =   ['sza', 'vis', 'fis','angle' ]
        Stokes =   ['meas_I', 'fit_I', 'meas_P_rel', 'fit_P_rel']
        Pij    = ['p11', 'p12', 'p22', 'p33'], 
        Lidar=  ['heightStd','g','LidarRatio','LidarDepol', 'gMode', 'LidarRatioMode', 'LidarDepolMode']

        # Plot the AOD data
        y = [0,1,2,0,1,2,]
        x = np.repeat((0,1),3)
        NoMode =Spheriod['MapR'].shape[0]
        if Spheriod['MapR'].shape[0] ==2 :
                mode_v = ["fine", "dust","marine"]
        if Spheriod['MapR'].shape[0] ==3 :
            mode_v = ["fine", "dust","marine"]
        linestyle =[':', '-','-.']
        

        lambda_ticks = np.round(Spheriod['lambda'], decimals=2)
        lambda_ticks_str = [str(x) for x in lambda_ticks]

        
        
        #Retrivals:
        for i in range(len(Retrival)):
            a,b = i%3,i%2
            for mode in range(Spheriod['MapR'].shape[0]): #for each modes

                ErrMap = Spheriod[RetrivalMAP[i]][mode] - Hex[RetrivalMAP[i]][mode]
                ErrLidar = Spheriod[RetrivalLidar[i]][mode] - Hex[RetrivalLidar[i]][mode]
               
                    
                if i ==0:

                    cm_sp2 = ['#b54cb5','#14411b', '#87C1FF']
                    cm_t2 = ["#14411b",'#adbf4b', 'b']
                    color_sph2 = '#adbf4b'
                    color_tamu2 = "#936ecf"

                    

                    # axs2[a,b].errorbar(Spheriod['r'][mode], Spheriod[Retrival[i]][mode],xerr=UNCERT['rv'],capsize=5,capthick =2, marker = "$O$",color = cm_sp[mode],lw = 3,ls = linestyle[mode], label=f"Sphrod_{mode_v[mode]}")
                    # axs2[a,b].errorbar(Hex['r'][mode],Hex[Retrival[i]][mode], marker = "H",xerr=UNCERT['rv'],capsize=5,capthick =2, color = cm_t[mode] ,lw = 3, ls = linestyle[mode],label=f"Hex_{mode_v[mode]}")
                        
                    axs2[a,b].plot(Spheriod['MapR'][mode], Spheriod['MapVol'][mode], marker = "$O$",color = cm_sp[mode],lw = 3,ls = linestyle[mode], label=f"RSP_Sphrod_{mode_v[mode]}")
                    axs2[a,b].plot(Hex['MapR'][mode],Hex['MapVol'][mode], marker = "H", color=cm_sp2[mode] ,lw = 3, ls = linestyle[mode],label=f"RSP_Hex_{mode_v[mode]}")
                    

                    axs2[a,b].plot(Spheriod['lidarR'][mode], Spheriod['lidarVol'][mode], marker = "$O$",color = cm_t[mode],lw = 3,ls = linestyle[mode], label=f"HSRL_Sphrod_{mode_v[mode]}")
                    axs2[a,b].plot(Hex['lidarR'][mode],Hex['lidarVol'][mode], marker = "H", color = cm_t2[mode]  ,lw = 3, ls = linestyle[mode],label=f"HSRL_Hex_{mode_v[mode]}")
                    
                    # DiffShape = Spheriod[Retrival[i]][mode] - Hex[Retrival[i]][mode]
                    # Err = DiffShape  # how much does hexhahderal vary  from spheriod

                    # if DiffShape>0: markerErr = "$S>$"
                    # if DiffShape<=0: markerErr =  "."

                    axsErr[a,b].plot(Spheriod['MapR'][mode], ErrMap,  marker = "o",color = cm_sp[mode],lw = 5,ls = linestyle[mode], label=f"{mode_v[mode]}")
                    axsErr[a,b].plot(Spheriod['MapR'][mode], ErrLidar,  marker = "o",color = cm_sp[mode],lw = 5,ls = linestyle[mode], label=f"{mode_v[mode]}")
                    
                    
                    
                    axs2[0,0].set_xlabel(r'rv $ \mu m$')
                    axs2[0,0].set_xscale("log")

                    axsErr[a,b].set_xlabel(r'rv $ \mu m$')
                    axsErr[a,b].set_xscale("log")


                    
                else:

                    axs2[a,b].errorbar(lambda_ticks_str, Spheriod[Retrival[i]][mode],yerr=UNCERT[Retrival[i]], marker = "$O$",markeredgecolor='k',capsize=7,capthick =2,markersize=25, color = cm_sp[mode],lw = 4,ls = linestyle[mode], label=f"Sphrod_{mode_v[mode]}")
                    axs2[a,b].errorbar(lambda_ticks_str,Hex[Retrival[i]][mode],yerr=UNCERT[Retrival[i]],capsize=7,capthick =2,  marker = "H",markeredgecolor='k',markersize=25, color = cm_t[mode] ,lw = 4, ls = linestyle[mode],label=f"Hex_{mode_v[mode]}")
                    
                    
                    # DiffShape = Spheriod[Retrival[i]][mode] - Hex[Retrival[i]][mode]
                    # Err = DiffShape  # how much does hexhahderal vary  from spheriod

                    # if DiffShape>0: markerErr = "$S>$"
                    # if DiffShape<=0: markerErr =  "."

                    axsErr[a,b].errorbar(lambda_ticks_str, ErrMap,  marker = "o",color = cm_sp[mode],lw = 5,ls = linestyle[mode], label=f"{mode_v[mode]}")
                    axsErr[a,b].set_xticks(lambda_ticks_str) 
                    # axs[a,b].set_xticklabels(['0.41', '0.46', '0.55' , '0.67'  , '0.86'])
                    axsErr[a,b].set_xlabel(r'$\lambda \mu m$')
                    
                    
                    # axs[a,b].plot(Spheriod['lambda'], Spheriod[Retrival[i]][mode], marker = "$O$",color = cm_sp[mode],lw = 2,ls = linestyle[mode],markersize=15, label=f"Sphrod_{mode_v[mode]}")
                    # axs[a,b].plot(Hex['lambda'],Hex[Retrival[i]][mode], marker = "H",color = cm_t[mode] ,lw = 2,  ls = linestyle[mode],markersize=15, label=f"Hex_{mode_v[mode]}")
                    
                    axs2[a,b].set_xticks(lambda_ticks_str) 
                    # axs[a,b].set_xticklabels(['0.41', '0.46', '0.55' , '0.67'  , '0.86'])
                    axs2[a,b].set_xlabel(r'$\lambda \mu m$')
                    # axs2[a,b].set_xlim(0, np.maximum(Spheriod[Retrival[i]][mode],Hex[Retrival[i]][mode]))
                    # fill = np.arange(np.min(Spheriod[Retrival[i]]), np.max(Spheriod[Retrival[i]]))
                    # # Fill between y-coordinates 0.35 and 0.4, for x-coordinates from 0 to 1
                    # axs2[a, b].fill(np.repeat(lambda_ticks_str[0], len(fill)), fill, color='gray', linewidth=5)

                    # # Plot a line with x-coordinate 0.53, y-coordinate np.min(Spheriod[Retrival[i]]), and 0.55, np.max(Spheriod[Retrival[i]])
                    # # Fill the region between the lines at 0.53 and 0.55
                    # axs2[a, b].fill_between('0.53', '0.55', Spheriod[Retrival[i]], color='gray', alpha=0.3)

                    # # Fill the region between the lines at 1 and 1.06
                    # axs2[a, b].fill_between('1', '1.06', Spheriod[Retrival[i]], color='gray', alpha=0.3)
                    #                     # Plot a line with x-coordinate 1, y-coordinate np.min(Spheriod[Retrival[i]]), and 1.06, np.max(Spheriod[Retrival[i]])
                                        # axs2[a, b].plot('1.06',fill, color='gray', linewidth=5, alpha=0.3)
                    axs2[a,b].set_ylabel(f'{Retrival[i]}')
                                
            
        # axs[2,1].plot(Spheriod['lambda'], Spheriod['aod'], marker = "$O$",color = color_sph,markersize=15,lw = 2, label=f"Sphroid")
        # axs[2,1].plot(Hex['lambda'], Hex['aod'], marker = "H", color = color_tamu ,markersize=15,lw = 2, label=f"Hexahedral")
        
        axs2[2,1].errorbar(lambda_ticks_str, Spheriod['aod'],yerr=0.03+UNCERT['aod']*Spheriod['aod'], marker = "$O$",color = color_sph,markeredgecolor='k',capsize=7,capthick =2,markersize=25,lw = 4, label=f"Sphroid")
        axs2[2,1].errorbar(lambda_ticks_str, Hex['aod'],yerr=0.03+UNCERT['aod']* Hex['aod'], marker = "H", color = color_tamu ,lw = 4,markeredgecolor='k',capsize=7,capthick =2,markersize=25, label=f"Hex")
        axsErr[2,1].errorbar(lambda_ticks_str, ErrMap, marker = "o", color = color_tamu ,lw = 5,markeredgecolor='k',capsize=5,capthick =2,markersize=25, label=f"Hex")
            
        
        axs2[2,1].set_xticks(lambda_ticks_str)
        # axs[2,1].set_xticklabels(['0.41', '0.46', '0.55' , '0.67'  , '0.86'])
        axsErr[2,1].set_xlabel(r'$\lambda$')
        axsErr[2,1].set_ylabel('Total AOD')    
        axs2[2,1].set_xlabel(r'$\lambda$')
        axs2[2,1].set_ylabel('Total AOD')
        axs2[0,0].legend(prop = { "size": 21 }, ncol=2)
        axsErr[0,0].legend(prop = { "size": 21 }, ncol=1)
        axs2[2,1].legend()
        # lat_t = Hex['latitude']
        # lon_t = Hex['longitude']
        # dt_t = Hex['datetime']
        plt.tight_layout(rect=[0, 0, 1, 1])
        
        axs2.figure.savefig(f'/home/gregmi/ORACLES/HSRL_RSP/{RepMode}_RSPRetrieval{RepMode}.png', dpi = 400)
        
        # plt.suptitle(f'RSP Aerosol Retrieval \n  Lat:{lat_t} Lon :{lon_t}   Date: {dt_t}')
        if RepMode ==2:
            fig.savefig(f'/home/gregmi/ORACLES/HSRL_RSP/2+3_RSPRetrieval.png', dpi = 400)
        else:
            fig.savefig(f'/home/gregmi/ORACLES/HSRL_RSP/{dt_t}{NoMode}_RSPRetrieval.png', dpi = 400)

def PlotcombEachMode(rslts_Sph2,rslts_Tamu2,HSRL_sphrodT,HSRL_TamuT,LidarPolSph,LidarPolTAMU,costValCal ):

    "Plots the retrieval results from RSP only, Lidar only and comboned in the same plot"

    plt.rcParams['font.size'] = '55'
    # plt.rcParams["figure.autolayout"] = True
    plt.rcParams['font.weight'] = 'normal'
   
    #Stokes Vectors Plot
    date_latlon = ['datetime', 'longitude', 'latitude']
    Xaxis = ['MapR','lambda','sca_ang','rv','height']
    Retrival = ['dVdlnr','n', 'k','aodMode','ssaMode']
    RetrivalMAP = ['MapVol','aodMode','ssaMode','n', 'k']
    RetrivalLidar = ['lidarVol','aodMode','ssaMode','n', 'k']
    #['sigma', 'vol', 'aodMode','ssaMode', 'rEff', 'costVal']
    Angles =   ['sza', 'vis', 'fis','angle' ]
    Stokes =   ['meas_I', 'fit_I', 'meas_P_rel', 'fit_P_rel']
    Pij    = ['p11', 'p12', 'p22', 'p33'], 
    Lidar=  ['heightStd','g','LidarRatio','LidarDepol', 'gMode', 'LidarRatioMode', 'LidarDepolMode']

    RSPIdx = np.array([1, 2, 4, 5, 6])
    HSRLIdx = np.array([0, 3, 7])
    # plot_HSRL(LidarPolSph[0][0],LidarPolTAMU[0][0],UNCERT, forward = False, retrieval = True, Createpdf = True,PdfName ="/home/gregmi/ORACLES/rsltPdf/LIDARPOL_Plots_444.pdf")
    
    if (LidarPolTAMU!= None )and (LidarPolSph!=None):  #When plotting joint retrieval with individual retrieval
        RepMode =2
        # Spheriod = LidarPolSph[0][0]
        # Hex =LidarPolTAMU[0][0]

    else:
        RepMode =1
    
    
    Spheriod = PlotSingle(rslts_Sph2,HSRL_sphrodT)
    Hex = PlotSingle(rslts_Tamu2,HSRL_TamuT)


    markerSph = ['o','.','.','o','.','.','.','o']
    markerHex = ['D','H','H','D','H','H','H','D']


    # markerSph2 = ['$0$','$0$','$0$','$0$','$0$','$0$','$0$','$0$']
    # markerHex2 = ['D','D','D','D','D','D','H','D']

    fig, axs2 = plt.subplots(nrows= 6, ncols=3, figsize=(65, 80))
    # fig, axsErr = plt.subplots(nrows= 6, ncols=3, figsize=(10, 15))
        
    fig.patch.set_alpha(0.0)
    # Plot the AOD data
    y = [0,1,2,0,1,2,]
    x = np.repeat((0,1),3)

    

    lambda_ticks = np.round(Spheriod['lambda'], decimals=2)

    # lambda_ticks_str = np.arange(1,9,1)
    lambda_ticks_str = [str(x) for x in lambda_ticks]
    lambda_ticks_str[0]= '0.35'

    NoMode =Spheriod['MapR'].shape[0]
    if Spheriod['MapR'].shape[0] ==2 :
            mode_v = ["Fine", "Dust","Marine"]
    if Spheriod['MapR'].shape[0] ==3 :
        mode_v = ["Fine", "Dust","Marine"]
    linestyle =['-', '-','-.']

    # cm_spj = ['#4459AA','#568FA0', 'r']
    # cm_tj = ["#FB6807",'#C18BB7', 'y']



    for RetNo in range(RepMode):

        if  RetNo ==1: 
            Spheriod,Hex = LidarPolSph[0][0],LidarPolTAMU[0][0]

            markerSph = ['$0$','$0$','$0$','$0$','$0$','$0$','$0$','$0$']
            markerHex = ['D','D','D','D','D','D','D','D']

            # markerfacecolor = cm_t[mode]
            markerfaceChex = ['none','none','none']
           


        if  RetNo ==0:

            cm_t = ['#EAAAF9','#2A847A','#00B2FF']
            cm_sp = ["#A6D854",'#1F244E','#23969A']


            cm_t2 = ["#882255",'#D55E00','#8F64B3']
            cm_sp2 = ["k",'k','#212869']

            markerfaceChex = cm_t

            


            # cm_sp2 = ['b','#BFBF2A', '#844772']
            # cm_t2 = ["#14411b",'#936ecf', '#FFA500']


            Spheriod,Hex = PlotSingle(rslts_Sph2,HSRL_sphrodT),PlotSingle(rslts_Tamu2,HSRL_TamuT)
            # cm_sp = ['#b54cb5','#14411b', '#87C1FF']
            # cm_t = ["#14411b",'#adbf4b', 'b']
            color_sph = '#adbf4b'
            color_tamu = "#936ecf"

        


        # a =-1
        
        #Retrivals:
        for i in range(len(Retrival)):
            
            a = i
            # if a ==8: a=-1
            # a = i%3
            for mode in range(NoMode): #for each modes
                
                b = mode
                
                if  RetNo ==0: 
                    ErrMap = Spheriod[RetrivalMAP[i]][mode] - Hex[RetrivalMAP[i]][mode]
                    ErrLidar = Spheriod[RetrivalLidar[i]][mode] - Hex[RetrivalLidar[i]][mode]
                    keyInd = 'Single'
                
                if  RetNo ==1:
                    ErrMap = Spheriod[Retrival[i]][mode] - Hex[Retrival[i]][mode]
                    ErrLidar = Spheriod[Retrival[i]][mode] - Hex[Retrival[i]][mode]
                    keyInd = 'Joint'

                if i ==0 and RetNo ==0: #Plotting single sensor retrievals


                    
                    # cm_sp2 = ['b','#14411b', '#14411b']
                    # cm_t2 = ["#14411b",'#936ecf', '#FFA500']


                    color_sph2 = '#adbf4b'
                    color_tamu2 = "#936ecf"

                    color_instrument = ['#FBD381','#A78236','#CDEDFF','#404790','#CE357D','#711E1E']

                #Plotting RSP
                    axs2[a,b].plot(Spheriod['MapR'][mode], Spheriod['MapVol'][mode], marker = "$O$",color = color_instrument[0],lw = 15,ls = linestyle[mode], label=f"RSP_Sphd")
                    axs2[a,b].plot(Hex['MapR'][mode],Hex['MapVol'][mode], marker = "H", color=color_instrument[1] ,lw = 10, ls = linestyle[mode],label=f"RSP_Hex")
                    
                #Plotting HSRL-2
                    axs2[a,b].plot(Spheriod['lidarR'][mode], Spheriod['lidarVol'][mode], marker = "$O$",color = color_instrument[2],lw = 15,ls = linestyle[mode], label=f"HSRL_Sphd")
                    axs2[a,b].plot(Hex['lidarR'][mode],Hex['lidarVol'][mode], marker = "H", color = color_instrument[3]  ,lw = 15, ls = linestyle[mode],label=f"HSRL_Hex")
                    
                    # axsErr[a,b].plot(Spheriod['MapR'][mode], ErrMap,  marker = "o",color = cm_sp[mode],lw = 15,ls = linestyle[mode], label=f"{mode_v[mode]}")
                    # axsErr[a,b].plot(Spheriod['MapR'][mode], ErrLidar,  marker = "o",color = cm_t2[mode],lw = 15,ls = linestyle[mode], label=f"{mode_v[mode]}")
                    
                    axs2[a,b].set_xlabel(r'rv $ \mu m$')
                    axs2[a,b].set_ylabel(r'dVdlnr')
                    axs2[a,b].set_xscale("log")
                    axs2[a,b].set_title(f'{mode_v[mode]}', weight='bold')

                    # axsErr[a,b].set_xlabel(r'rv $ \mu m$')
                    # axsErr[a,b].set_xscale("log")
                    if mode ==0:
                        axs2[a,b].legend()

                if i ==0 and RetNo ==1: #Plotting single sensor retrievals

                    cm_sp= cm_sp2
                    cm_t = cm_t2


                  
                    axs2[a,b].plot(Spheriod['r'][mode], Spheriod[Retrival[i]][mode], marker = "$O$",color = color_instrument[4],lw = 15,ls = linestyle[mode], label=f"Joint_Sphd")
                    axs2[a,b].plot(Hex['r'][mode],Hex[Retrival[i]][mode], marker = "H", color=color_instrument[5] ,lw = 15, ls = linestyle[mode],label=f"Joint_Hex")
                    

                    # axs2[a,b].plot(Spheriod['lidarR'][mode], Spheriod['lidarVol'][mode], marker = "$O$",color = cm_sp2[mode],lw = 5,ls = linestyle[mode], label=f"HSRL_Sphrod")
                    # axs2[a,b].plot(Hex['lidarR'][mode],Hex['lidarVol'][mode], marker = "H", color = cm_t2[mode]  ,lw = 5, ls = linestyle[mode],label=f"HSRL_Hex")
                    
                    # axsErr[a,b].plot(Spheriod['MapR'][mode], ErrMap,  marker = "o",color = cm_sp[mode],lw = 5,ls = linestyle[mode], label=f"{mode_v[mode]}")
                    # axsErr[a,b].plot(Spheriod['MapR'][mode], ErrLidar,  marker = "o",color = cm_t2[mode],lw = 5,ls = linestyle[mode], label=f"{mode_v[mode]}")
                    
                    axs2[a,b].set_xlabel(r'rv $ \mu m$', weight='bold')
                    axs2[a,b].set_ylabel(r'dVdlnr', weight='bold')
                    axs2[a,b].set_xscale("log")
                    axs2[a,b].set_title(f'{mode_v[mode]}', weight='bold')

                    # axsErr[a,b].set_xlabel(r'rv $ \mu m$')
                    # axsErr[a,b].set_xscale("log")
                    if mode ==0:
                        axs2[a,b].legend()
                    # axs2[a,b].legend(prop = { "size": 22 },loc='upper center', bbox_to_anchor=(0.5, -0.4), shadow=True, ncol=3)

                if i>0:
                    
                    axs2[a,b].errorbar(lambda_ticks_str,Hex[Retrival[i]][mode],yerr=UNCERT[Retrival[i]], color = cm_t[mode] ,lw = 1, ls = linestyle[mode])
                    axs2[a,b].errorbar(lambda_ticks_str, Spheriod[Retrival[i]][mode],yerr=UNCERT[Retrival[i]], color = cm_sp[mode],lw = 1,ls = linestyle[mode])
                    

                    for scp in range(len(lambda_ticks_str)):
                        axs2[a,b].errorbar(lambda_ticks_str[scp],Hex[Retrival[i]][mode][scp],capsize=7,capthick =2,  marker =  markerHex[scp], markersize=50, markeredgecolor =cm_t[1] ,markerfacecolor= markerfaceChex[1],markeredgewidth=10,lw = 1,alpha = 0.8, ls = linestyle[mode])
                        axs2[a,b].errorbar(lambda_ticks_str[scp], Spheriod[Retrival[i]][mode][scp], marker = markerSph[scp],markeredgecolor='k',capsize=7,capthick =2,markersize=55,alpha = 0.8,color = cm_sp[1],lw = 1,ls = linestyle[mode])
                        
                    for scp2 in range(1):
                        axs2[a,b].errorbar(lambda_ticks_str[scp2],Hex[Retrival[i]][mode][scp2],capsize=7,capthick =2,  marker =  markerHex[scp], markersize=50, markeredgecolor = cm_t[1],markeredgewidth=10 , markerfacecolor= markerfaceChex[1], lw = 1,alpha = 0.8, ls = linestyle[mode], label=f"{keyInd}Hex")
                        axs2[a,b].errorbar(lambda_ticks_str[scp2], Spheriod[Retrival[i]][mode][scp2], marker = markerSph[scp],markeredgecolor='k',capsize=7,capthick =2,markersize=55,alpha = 0.8,color = cm_sp[1],lw = 1,ls = linestyle[mode], label=f"{keyInd}Sphd")
                      
                        if scp2 ==0:
                            if a ==1 and b ==1:
                                axs2[a,b].legend(loc='best', prop = { "size": 45 },ncol=1)
                                # axs2[a,b].legend(prop = { "size": 21 },loc='upper center', bbox_to_anchor=(0.5, -0.4), shadow=True, ncol=2)
                   

        
                    # axsErr[a,b].errorbar(lambda_ticks_str, ErrMap,  marker = "o",color = cm_sp[mode],lw = 5,ls = linestyle[mode], label=f"{mode_v[mode]}")
                    # axsErr[a,b].set_xticks(lambda_ticks_str) 
                    # # axs[a,b].set_xticklabels(['0.41', '0.46', '0.55' , '0.67'  , '0.86'])
                    # axsErr[a,b].set_xlabel(r'$\lambda \mu m$')
                     
                    axs2[a,b].set_xticks(lambda_ticks_str) 
                    # axs[a,b].set_xticklabels(['0.41', '0.46', '0.55' , '0.67'  , '0.86'])
                    
                    axs2[a,b].set_xlabel(r'$\lambda \mu m$')

                    # axs2[a,b].legend()
                    # axs2[a,b].set_xlim(0, np.maximum(Spheriod[Retrival[i]][mode],Hex[Retrival[i]][mode]))
                    # fill = np.arange(np.min(Spheriod[Retrival[i]]), np.max(Spheriod[Retrival[i]]))
                    # # Fill between y-coordinates 0.35 and 0.4, for x-coordinates from 0 to 1
                    # axs2[a, b].fill(np.repeat(lambda_ticks_str[0], len(fill)), fill, color='gray', linewidth=5)

                    # # Plot a line with x-coordinate 0.53, y-coordinate np.min(Spheriod[Retrival[i]]), and 0.55, np.max(Spheriod[Retrival[i]])
                    # # Fill the region between the lines at 0.53 and 0.55
                    # axs2[a, b].fill_between('0.53', '0.55', Spheriod[Retrival[i]], color='gray', alpha=0.3)

                    # # Fill the region between the lines at 1 and 1.06
                    # axs2[a, b].fill_between('1', '1.06', Spheriod[Retrival[i]], color='gray', alpha=0.3)
                    #                     # Plot a line with x-coordinate 1, y-coordinate np.min(Spheriod[Retrival[i]]), and 1.06, np.max(Spheriod[Retrival[i]])
                                        # axs2[a, b].plot('1.06',fill, color='gray', linewidth=5, alpha=0.3)
                    if b ==0:
                        axs2[a,b].set_ylabel(f'{Retrival[i]}', weight='bold')
                    if a ==0:
                        axs2[a,b].set_title(f'{mode_v[mode]}', weight='bold')

                    
                    # axs2[a,b].legend(ncol = 4)
                    
        axs2[5,1].errorbar(lambda_ticks_str, Spheriod['aod'], lw = 1)
        axs2[5,1].errorbar(lambda_ticks_str, Hex['aod'], lw = 1)
                              
        for scp in range(len(lambda_ticks_str)):
            axs2[5,1].errorbar(lambda_ticks_str[scp], Spheriod['aod'][scp], marker = markerSph[scp],markeredgecolor='k',capsize=7,capthick =2,markersize=50,  color = cm_sp[1])
            axs2[5,1].errorbar(lambda_ticks_str[scp],Hex['aod'][scp],capsize=7,capthick =2,  marker =  markerHex[scp], markersize=50, markeredgecolor =cm_t[1] ,markerfacecolor= markerfaceChex[1],markeredgewidth=10,lw = 1,alpha = 0.8)
            # if scp ==1:
            #     axs2[7,1].legend()
        for scp in range(1):
            axs2[5,1].errorbar(lambda_ticks_str[scp], Spheriod['aod'][scp], marker = markerSph[scp],markeredgecolor='k',capsize=7,capthick =2,markersize=50, color = cm_sp[1])
            axs2[5,1].errorbar(lambda_ticks_str[scp],Hex['aod'][scp],capsize=7,capthick =2,  marker =  markerHex[scp], markersize=50, markeredgecolor =cm_t[1] ,markerfacecolor= markerfaceChex[1],markeredgewidth=10,lw = 1,alpha = 0.8)
            
            # if scp ==1:
            axs2[5,1].legend(prop = { "size": 13 })
                    
        # axs2[7,1].errorbar(lambda_ticks_str, Spheriod['aod'],yerr=0.03+UNCERT['aod']*Spheriod['aod'], marker = "$O$",color = cm_sp[mode],markeredgecolor='k',capsize=7,capthick =2,markersize=25,lw = 4, label=f"{keyInd}Sphd")
        # axs2[7,1].errorbar(lambda_ticks_str, Hex['aod'],yerr=0.03+UNCERT['aod']* Hex['aod'], marker = "H", color = cm_t[mode] ,lw = 4,markeredgecolor='k',capsize=7,capthick =2,markersize=25, label=f"{keyInd}Hex")
        # axsErr[7,1].errorbar(lambda_ticks_str, ErrMap, marker = "o", color = color_tamu ,lw = 5,markeredgecolor='k',capsize=5,capthick =2,markersize=25, label=f"Hex")
            
        
        axs2[5,1].set_xticks(lambda_ticks_str)
        # axs[2,1].set_xticklabels(['0.41', '0.46', '0.55' , '0.67'  , '0.86'])
        # axsErr[5,1].set_xlabel(r'$\lambda$', weight='bold')
        # axsErr[5,1].set_ylabel('Total AOD', weight='bold')    
        axs2[5,1].set_xlabel(r'$\lambda$', weight='bold')
        axs2[5,1].set_ylabel('Total AOD', weight='bold')

        # axs2[8,1].legend(prop = { "size": 21 }, ncol=2)
        # axsErr[8,1].legend(prop = { "size": 21 }, ncol=1)
        axs2[5,1].legend(ncol=2)

        color_instrument = ['#FBD381','#A78236','#CDEDFF','#404790','#CE357D','#711E1E'] #Color blind friendly scheme for different retrieval techniques< :rsponlt, hsrl only and combined

        if RetNo ==0: #For Ldiar only retrieval

            width = 0.1 #width of the bar
            
            for Nmode in range (NoMode):
                

                x = Nmode+1

                axs2[5,2].bar(x,rslts_Sph[0]['sph'] [Nmode], color=color_instrument[0],width = width, label = "RSPsph")
                axs2[5,2].bar(x+ width,rslts_Tamu[0]['sph'][Nmode],color=color_instrument[1], width = width, label = "RSPhex")
                axs2[5,2].bar(x+ 2*width,HSRL_sphrodT[0][0]['sph'][Nmode],color=color_instrument[2], width = width, label = "HSRLsph")
                axs2[5,2].bar(x+ 3*width,HSRL_TamuT[0][0]['sph'][Nmode], color=color_instrument[3],width = width, label = "HSRLhex")

                axs2[5,2].bar(x+ 4*width,LidarPolSph[0][0]['sph'][Nmode],width = width,  color = color_instrument[4],label = "Jointsph")
                axs2[5,2].bar(x+ 5*width,LidarPolTAMU[0][0]['sph'][Nmode],width = width, color = color_instrument[5],label = "JointHex")
                # if Nmode ==0: 
                #     axs2[5,2].legend(prop = { "size": 18 },ncol=1)

                # axs2[6,1].legend(prop = { "size": 21 },ncol=1)
            axs2[5,2].set_xlabel("Aerosol type", weight='bold')
            axs2[5,2].set_ylabel('Spherical Frac', weight='bold')



            # axs2[5,0].bar('RSP \n SPh',rslts_Sph[0]['costVal'],width = 0.5,  color=color_instrument[0], label = "RSPsph")
            # axs2[5,0].bar('RSP\n Hex',rslts_Tamu[0]['costVal'],width = 0.5, color=color_instrument[1],label = "RSPhex")
            # axs2[5,0].bar('HSRL\n SPh',HSRL_sphrodT[0][0]['costVal'],width = 0.5, color=color_instrument[2],label = "RSPsph")
            # axs2[5,0].bar('HSRL \nHex',HSRL_TamuT[0][0]['costVal'],width = 0.5, color=color_instrument[3],label = "RSPhex")
            # # axs2[8,1].legend()
            # # axs2[8,1].set_xlabel("CostVal")
            # axs2[5,0].set_ylabel("CostVal", weight='bold')
            # axs2[5, 0].tick_params(axis='x', rotation=90)


            # axs2[5,2].scatter(mode_v,rslts_Sph[0]['sph'], color=cm_sp[mode], marker = "$RO$",s=1500,label = "RSPsph")
            # axs2[5,2].scatter(mode_v,rslts_Tamu[0]['sph'],color=cm_t[mode], marker = "$RH$",s=1500, label = "RSPhex")
            # axs2[5,2].scatter(mode_v,HSRL_sphrodT[0][0]['sph'],color=cm_sp2[mode],  marker = "$HO$",s=1500, label = "HSRLsph")
            # axs2[5,2].scatter(mode_v,HSRL_TamuT[0][0]['sph'], color=cm_t2[mode],marker = "$HH$",s=1500, label = "HSRLhex")
            # # axs2[6,1].legend(prop = { "size": 21 },ncol=1)
            # axs2[5,2].set_xlabel("Aerosol type", weight='bold')
            # axs2[5,2].set_ylabel('Spherical Frac', weight='bold')


            # axs2[5,0].scatter('RSP SPh',rslts_Sph[0]['costVal'], color=cm_sp[mode],s=1400, label = "RSPsph")
            # axs2[5,0].scatter('RSP Hex',rslts_Tamu[0]['costVal'], color=cm_t[mode],s=1400,label = "RSPhex")
            # axs2[5,0].scatter('HSRL SPh',HSRL_sphrodT[0][0]['costVal'], color=cm_sp2[mode],s=1400,label = "RSPsph")
            # axs2[5,0].scatter('HSRL Hex',HSRL_TamuT[0][0]['costVal'], color=cm_t2[mode],s=1400,label = "RSPhex")
            # # axs2[8,1].legend()
            # # axs2[8,1].set_xlabel("CostVal")
            # axs2[5,0].set_ylabel("CostVal", weight='bold')
        
        if RetNo ==1:

            
        
            
            axs2[5,2].set_xlabel("Aerosol type", weight='bold')
            axs2[5,2].set_ylabel('Spherical Frac', weight='bold')
            axs2[5,2].set_yscale('log')
            axs2[5,2].set_xticks(np.arange(1,4,1)+0.25, labels=mode_v)


        axs2[5,0].bar('RSP \n Sph',costValCal['RSP_sph'],width = 0.5,  color=color_instrument[0], label = "RSPsph")
        axs2[5,0].bar('RSP\n Hex',costValCal['RSP_hex'],width = 0.5, color=color_instrument[1],label = "RSPhex")
        axs2[5,0].bar('HSRL\n Sph',costValCal['HSRL_sph'],width = 0.5, color=color_instrument[2],label = "RSPsph")
        axs2[5,0].bar('HSRL \nHex',costValCal['HSRL_hex'],width = 0.5, color=color_instrument[3],label = "RSPhex")
        axs2[5,0].bar('RSP+HSRL\n Sph',costValCal['J_sph'],width = 0.5, color=color_instrument[4], label = "RSP+HSRL sph")
        axs2[5,0].bar('RSP+HSRL \nHex',costValCal['J_hex'],width = 0.5, color=color_instrument[5],label = "RSP+HSRL hex")
           
       
        # axs2[8,1].legend()
        # axs2[8,1].set_xlabel("CostVal")
        axs2[5,0].set_ylabel("CostVal", weight='bold')
        axs2[5, 0].tick_params(axis='x', rotation=90)


    #Aod values for dust, amrine and fine calculated based on the aerosol classification scheme.
    AerClassAOD= HSRL_sphrodT[3]

    x = ['0.35','0.53']

    axs2[3,0].scatter(x,(AerClassAOD['Aod_fine_355'],AerClassAOD['Aod_fine_532']), s = 3500, lw=3, marker ="X", zorder=6, edgecolors='k', color ='#E4FFA5')

    axs2[3,0].fill_between([-0.5, 0.5], 0, AerClassAOD['Aod_oth_355'], color='#E4FFA5', alpha=0.3)

    axs2[3,0].fill_between([2.5, 3.5], 0, AerClassAOD['Aod_oth_532'], color='#E4FFA5', alpha=0.3)

    axs2[3,1].scatter(x,(AerClassAOD['Aod_dst_355'],AerClassAOD['Aod_dst_532']), s = 3500, marker ="X",lw=3, zorder=6, edgecolors='k',color ='#E4FFA5')

    axs2[3,1].fill_between([-0.5, 0.5], 0, AerClassAOD['Aod_T_355'], color='#E4FFA5', alpha=0.3)

    axs2[3,1].fill_between([2.5, 3.5], 0, AerClassAOD['Aod_T_532'], color='#E4FFA5', alpha=0.3)

    axs2[3,2].scatter(x,(AerClassAOD['Aod_fine_355'],AerClassAOD['Aod_sea_532']), s = 3500,lw=3, marker ='X',color ='#E4FFA5', edgecolors='k', zorder=6)

    axs2[3,2].fill_between([-0.5, 0.5], 0, AerClassAOD['Aod_below_BLH_355'], color='#E4FFA5', alpha=0.3)

    axs2[3,2].fill_between([2.5, 3.5], 0, AerClassAOD['Aod_below_BLH_532'], color='#E4FFA5', alpha=0.3)


    axs2[5,1].scatter(x[0],AerClassAOD['Aod_T_355'], s = 3500, color ='#E4FFA5' , marker ="X", lw=3, zorder=6, edgecolors='k', label ="HSRL AOD")

    axs2[5,1].scatter(x[1],AerClassAOD['Aod_T_532'], s = 3500, color ='#E4FFA5' , marker ="X",lw=3, edgecolors='k',  zorder=6)





            # # axs2[8,1].legend()
            # # axs2[8,1].set_xlabel("CostVal")
            # axs2[5,0].set_ylabel("CostVal", weight='bold')

        # axs2[6,1].legend(prop = { "size": 18 },loc='upper center', bbox_to_anchor=(0.5, -0.4), shadow=True, ncol=4)
        # axs2[8,1].legend(prop = { "size": 18 },loc='upper center', bbox_to_anchor=(0.5, -0.4), shadow=True, ncol=4)
        # axs2[7,1].legend(prop = { "size": 18 },loc='upper center', bbox_to_anchor=(0.5, -0.4), shadow=True, ncol=4)
                 


        # lat_t = Hex['latitude']
        # lon_t = Hex['longitude']
        # dt_t = Hex['datetime']
        # plt.tight_layout(rect=[0, 0, 1, 1])

    plt.show()
        
    fig.savefig(f'/home/gregmi/ORACLES/HSRL_RSP/AllRetrieval{RepMode}_case1.png', dpi = 200,  transparent=True)
    
    # plt.suptitle(f'RSP Aerosol Retrieval \n  Lat:{lat_t} Lon :{lon_t}   Date: {dt_t}')
    # if RepMode ==2:
    #     fig.savefig(f'/home/gregmi/ORACLES/HSRL_RSP/2+3_RSPRetrieval.png', dpi = 300)
    # else:
    #     fig.savefig(f'/home/gregmi/ORACLES/HSRL_RSP/{NoMode}_RSPRetrieval.png', dpi = 300)


PlotcombEachMode(rslts_Sph,rslts_Tamu,HSRL_sphrodT,HSRL_TamuT,LidarPolSph,LidarPolTAMU,costValCal)



def ErrRetrieval(rslts_Sph,rslts_Tamu,rslts_Sph2=None,rslts_Tamu2= None):


    Spheriod,Hex = rslts_Sph[0],rslts_Tamu[0]

    if (rslts_Sph2!=None )and (rslts_Tamu2!=None):

        RepMode =2
    else:
        RepMode =1
        
    lambda_ticks = np.round(Spheriod['lambda'], decimals=2)
    lambda_ticks_str = [str(x) for x in lambda_ticks]
    
    
    cm_sp = ['k','#8B4000', '#87C1FF']
    cm_t = ["#BC106F",'#E35335', 'b']
    color_sph = '#0c7683'
    color_tamu = "#BC106F"
    
    fig, axsErr = plt.subplots(nrows= 3, ncols=2, figsize=(20, 15))
    for i in range(RepMode):
        
        if i ==1:
            
            
            # lambda_ticks_str = [str(x) for x in lambda_ticks]
            Spheriod,Hex = rslts_Sph2[0],rslts_Tamu2[0]
            lambda_ticks_str = np.round(Spheriod['lambda'], decimals=2)
            
            

        plt.rcParams['font.size'] = '26'
        plt.rcParams["figure.autolayout"] = True
        #Stokes Vectors Plot

        Retrival = ['dVdlnr','aodMode','ssaMode','n', 'k']
        #['sigma', 'vol', 'aodMode','ssaMode', 'rEff', 'costVal']
      
        # Plot the AOD data
        y = [0,1,2,0,1,2,]
        x = np.repeat((0,1),3)
        NoMode =Spheriod['r'].shape[0]
        if Spheriod['r'].shape[0] ==2 :
                mode_v = ["fine", "dust","marine"]
        if Spheriod['r'].shape[0] ==3 :
            mode_v = ["fine", "dust","marine"]
        linestyle =[':', '-','-.']
        
        #Retrivals:
        for i in range(len(Retrival)):
            a,b = i%3,i%2
            for mode in range(Spheriod['r'].shape[0]): #for each modes

                Err = Spheriod[Retrival[i]][mode] - Hex[Retrival[i]][mode]
                   
                if i ==0:

                
                    axsErr[a,b].plot(Spheriod['r'][mode], Err,  marker = "o",color = cm_sp[mode],lw = 5,ls = linestyle[mode], label=f"{mode_v[mode]}")
                    
                    axsErr[a,b].set_xlabel(r'rv $ \mu m$')
                    axsErr[a,b].set_xscale("log")


                else:

                    axsErr[a,b].scatter(lambda_ticks_str, Err,  marker = "o",color = cm_sp[mode],lw = 5,ls = linestyle[mode], label=f"{mode_v[mode]}")
                    axsErr[a,b].set_xticks(lambda_ticks_str) 
                    # axs[a,b].set_xticklabels(['0.41', '0.46', '0.55' , '0.67'  , '0.86'])
                    axsErr[a,b].set_xlabel(r'$\lambda \mu m$')
                    
                    

            axsErr[a,b].set_ylabel(f'{Retrival[i]}')
            
        axsErr[2,1].scatter(lambda_ticks_str, Err, marker = "o", color = color_tamu ,lw = 5, label=f"Hex")
          
 
        # axs[2,1].set_xticklabels(['0.41', '0.46', '0.55' , '0.67'  , '0.86'])
        axsErr[2,1].set_xlabel(r'$\lambda$')
        axsErr[2,1].set_ylabel('Total AOD')    
        axsErr[0,0].legend(prop = { "size": 21 }, ncol=1)
 
        lat_t = Hex['latitude']
        lon_t = Hex['longitude']
        dt_t = Hex['datetime']
        plt.tight_layout(rect=[0, 0, 1, 1])
        # axs2.savefig(f'/home/gregmi/ORACLES/HSRL_RSP/{dt_t}{RepMode}_RSPRetrieval{RepMode}.png', dpi = 400)
        
        plt.suptitle(f'RSP Aerosol Retrieval \n  Lat:{lat_t} Lon :{lon_t}   Date: {dt_t}')
        if RepMode ==2:
            fig.savefig(f'/home/gregmi/ORACLES/HSRL_RSP/{dt_t}2+3_RSPRetrieval.png', dpi = 400)
        else:
            fig.savefig(f'/home/gregmi/ORACLES/HSRL_RSP/{dt_t}{NoMode}_RSPRetrieval.png', dpi = 400)

def Error_bar(SphHSRL,HexHSRL,variableName):
    RepMode = 0

    fig, axs2 = plt.subplots(nrows= 3, ncols=2, figsize=(35, 20))
    fig, axsErr = plt.subplots(nrows= 3, ncols=2, figsize=(20, 15))


    for No in range (len(variableName)):    
        # Spheriod,Hex = SphHSRL[No],HexHSRL[No]

        cm_t = ['#14411b','#94673E','#00B2FF']
        cm_sp = ['#8A898E','#DCD98D','#23969A']


        cm_t2 = ["#882255",'#FE5200','#8F64B3']
        cm_sp2 = ["#DEA520",'#44440F','#212869']

        # cm_sp = ['#AA98A9','#CB9129', 'c']
        # cm_t = ['#900C3F','#8663bd', '#053F5C']

        color_sph = '#0c7683'
        color_tamu = "#BC106F"

        # Spheriod,Hex = SphHSRL[No][0][0],HexHSRL[No][0][0]
        Spheriod,Hex = SphHSRL[No][0],HexHSRL[No][0]
        
            


        # for i in range(RepMode):
        #     if i ==1:
        #         
        cm_sp = ['#b54cb5','#14411b', '#87C1FF']
        cm_t = ["#14411b",'#FF4500', 'b']
        # color_sph = '#adbf4b'
        # color_tamu = "#936ecf"

        plt.rcParams['font.size'] = '26'
        plt.rcParams["figure.autolayout"] = True
        #Stokes Vectors Plot
        date_latlon = ['datetime', 'longitude', 'latitude']
        Xaxis = ['MapR','lambda','sca_ang','rv','height']
        Retrival = ['dVdlnr','aodMode','ssaMode','n', 'k']
        RetrivalMAP = ['MapVol','aodMode','ssaMode','n', 'k']
        RetrivalLidar = ['lidarVol','aodMode','ssaMode','n', 'k']
        #['sigma', 'vol', 'aodMode','ssaMode', 'rEff', 'costVal']
        Angles =   ['sza', 'vis', 'fis','angle' ]
        Stokes =   ['meas_I', 'fit_I', 'meas_P_rel', 'fit_P_rel']
        Pij    = ['p11', 'p12', 'p22', 'p33'], 
        Lidar=  ['heightStd','g','LidarRatio','LidarDepol', 'gMode', 'LidarRatioMode', 'LidarDepolMode']

        # Plot the AOD data
        y = [0,1,2,0,1,2,]
        x = np.repeat((0,1),3)
        NoMode =Spheriod['r'].shape[0]
        if Spheriod['r'].shape[0] ==2 :
                mode_v = ["fine", "dust","marine"]
        if Spheriod['r'].shape[0] ==3 :
            mode_v = ["fine", "dust","marine"]
        linestyle =[':', '-','-.']
        

        lambda_ticks = np.round(Spheriod['lambda'], decimals=2)
        lambda_ticks_str = [str(x) for x in lambda_ticks]

        
        
        #Retrivals:
        for i in range(len(Retrival)):
            a,b = i%3,i%2
            for mode in range(Spheriod['r'].shape[0]): #for each modes

                ErrMap = Spheriod[Retrival[i]][mode] - Hex[Retrival[i]][mode]
                ErrLidar = Spheriod[Retrival[i]][mode] - Hex[Retrival[i]][mode]
            
                    
                if i ==0:

                    # cm_sp = ['#4459AA','#14411b', '#87C1FF']
                    # cm_t = ["#14411b",'#adbf4b', '#9BB54C']

                    # cm_sp2 = ['#b54cb5','#14411b', '#87C1FF']
                    # cm_t2 = ["#14411b",'#adbf4b', 'b']

                    # cm_sp2 = ['b','#BFBF2A', '#844772']
                    # cm_t2 = ["#14411b",'#936ecf', '#FFA500']
                    # color_sph2 = '#adbf4b'
                    

                    

                    # axs2[a,b].errorbar(Spheriod['r'][mode], Spheriod[Retrival[i]][mode],xerr=UNCERT['rv'],capsize=5,capthick =2, marker = "$O$",color = cm_sp[mode],lw = 3,ls = linestyle[mode], label=f"Sphrod_{mode_v[mode]}")
                    # axs2[a,b].errorbar(Hex['r'][mode],Hex[Retrival[i]][mode], marker = "H",xerr=UNCERT['rv'],capsize=5,capthick =2, color = cm_t[mode] ,lw = 3, ls = linestyle[mode],label=f"Hex_{mode_v[mode]}")
                        
                    axs2[a,b].plot(Spheriod['r'][mode], Spheriod[Retrival[i]][mode], marker = "$O$",color = cm_sp[mode],lw = 3,ls = linestyle[mode], label=f"RSP_Sphrod_{mode_v[mode]}")
                    axs2[a,b].plot(Hex['r'][mode],Hex[Retrival[i]][mode], marker = "H", color=cm_t[mode] ,lw = 3, ls = linestyle[mode],label=f"RSP_Hex_{mode_v[mode]}")
                    

                    # axs2[a,b].plot(Spheriod['r'][mode], Spheriod[Retrival[i]][mode], marker = "$O$",color = cm_t[mode],lw = 3,ls = linestyle[mode], label=f"HSRL_Sphrod_{mode_v[mode]}")
                    # axs2[a,b].plot(Hex['r'][mode],Hex[Retrival[i]][mode], marker = "H", color = cm_t2[mode]  ,lw = 3, ls = linestyle[mode],label=f"HSRL_Hex_{mode_v[mode]}")
                    
                    # DiffShape = Spheriod[Retrival[i]][mode] - Hex[Retrival[i]][mode]
                    # Err = DiffShape  # how much does hexhahderal vary  from spheriod

                    # if DiffShape>0: markerErr = "$S>$"
                    # if DiffShape<=0: markerErr =  "."

                    axsErr[a,b].plot(Spheriod['r'][mode], ErrMap,  marker = "o",color = cm_sp[mode],lw = 5,ls = linestyle[mode], label=f"{mode_v[mode]}")
                    axsErr[a,b].plot(Spheriod['r'][mode], ErrLidar,  marker = "o",color = cm_sp[mode],lw = 5,ls = linestyle[mode], label=f"{mode_v[mode]}")
                    
                    
                    
                    axs2[0,0].set_xlabel(r'rv $ \mu m$')
                    axs2[0,0].set_xscale("log")

                    axsErr[a,b].set_xlabel(r'rv $ \mu m$')
                    axsErr[a,b].set_xscale("log")


                    
                else:

                    axs2[a,b].errorbar(lambda_ticks_str, Spheriod[Retrival[i]][mode],yerr=UNCERT[Retrival[i]], marker = "$O$",markeredgecolor='k',capsize=7,capthick =2,markersize=25, color = cm_sp[mode],lw = 4,ls = linestyle[mode], label=f"Sphrod_{mode_v[mode]}")
                    axs2[a,b].errorbar(lambda_ticks_str,Hex[Retrival[i]][mode],yerr=UNCERT[Retrival[i]],capsize=7,capthick =2,  marker = "H",markeredgecolor='k',markersize=25, color = cm_t[mode] ,lw = 4, ls = linestyle[mode],label=f"Hex_{mode_v[mode]}")
                    
                    
                    # DiffShape = Spheriod[Retrival[i]][mode] - Hex[Retrival[i]][mode]
                    # Err = DiffShape  # how much does hexhahderal vary  from spheriod

                    # if DiffShape>0: markerErr = "$S>$"
                    # if DiffShape<=0: markerErr =  "."

                    axsErr[a,b].errorbar(lambda_ticks_str, ErrMap,  marker = "o",color = cm_sp[mode],lw = 5,ls = linestyle[mode], label=f"{mode_v[mode]}")
                    axsErr[a,b].set_xticks(lambda_ticks_str) 
                    # axs[a,b].set_xticklabels(['0.41', '0.46', '0.55' , '0.67'  , '0.86'])
                    axsErr[a,b].set_xlabel(r'$\lambda \mu m$')
                    
                    
                    # axs[a,b].plot(Spheriod['lambda'], Spheriod[Retrival[i]][mode], marker = "$O$",color = cm_sp[mode],lw = 2,ls = linestyle[mode],markersize=15, label=f"Sphrod_{mode_v[mode]}")
                    # axs[a,b].plot(Hex['lambda'],Hex[Retrival[i]][mode], marker = "H",color = cm_t[mode] ,lw = 2,  ls = linestyle[mode],markersize=15, label=f"Hex_{mode_v[mode]}")
                    
                    axs2[a,b].set_xticks(lambda_ticks_str) 
                    # axs[a,b].set_xticklabels(['0.41', '0.46', '0.55' , '0.67'  , '0.86'])
                    axs2[a,b].set_xlabel(r'$\lambda \mu m$')
                    # axs2[a,b].set_xlim(0, np.maximum(Spheriod[Retrival[i]][mode],Hex[Retrival[i]][mode]))
                    # fill = np.arange(np.min(Spheriod[Retrival[i]]), np.max(Spheriod[Retrival[i]]))
                    # # Fill between y-coordinates 0.35 and 0.4, for x-coordinates from 0 to 1
                    # axs2[a, b].fill(np.repeat(lambda_ticks_str[0], len(fill)), fill, color='gray', linewidth=5)

                    # # Plot a line with x-coordinate 0.53, y-coordinate np.min(Spheriod[Retrival[i]]), and 0.55, np.max(Spheriod[Retrival[i]])
                    # # Fill the region between the lines at 0.53 and 0.55
                    # axs2[a, b].fill_between('0.53', '0.55', Spheriod[Retrival[i]], color='gray', alpha=0.3)

                    # # Fill the region between the lines at 1 and 1.06
                    # axs2[a, b].fill_between('1', '1.06', Spheriod[Retrival[i]], color='gray', alpha=0.3)
                    #                     # Plot a line with x-coordinate 1, y-coordinate np.min(Spheriod[Retrival[i]]), and 1.06, np.max(Spheriod[Retrival[i]])
                                        # axs2[a, b].plot('1.06',fill, color='gray', linewidth=5, alpha=0.3)
                    axs2[a,b].set_ylabel(f'{Retrival[i]}')
                                
            
        # axs[2,1].plot(Spheriod['lambda'], Spheriod['aod'], marker = "$O$",color = color_sph,markersize=15,lw = 2, label=f"Sphroid")
        # axs[2,1].plot(Hex['lambda'], Hex['aod'], marker = "H", color = color_tamu ,markersize=15,lw = 2, label=f"Hexahedral")
        
        axs2[2,1].errorbar(lambda_ticks_str, Spheriod['aod'],yerr=0.03+UNCERT['aod']*Spheriod['aod'], marker = "$O$",color = color_sph,markeredgecolor='k',capsize=7,capthick =2,markersize=25,lw = 4, label=f"Sphroid")
        axs2[2,1].errorbar(lambda_ticks_str, Hex['aod'],yerr=0.03+UNCERT['aod']* Hex['aod'], marker = "H", color = color_tamu ,lw = 4,markeredgecolor='k',capsize=7,capthick =2,markersize=25, label=f"Hex")
        axsErr[2,1].errorbar(lambda_ticks_str, ErrMap, marker = "o", color = color_tamu ,lw = 5,markeredgecolor='k',capsize=5,capthick =2,markersize=25, label=f"Hex")
            
        
        axs2[2,1].set_xticks(lambda_ticks_str)
        # axs[2,1].set_xticklabels(['0.41', '0.46', '0.55' , '0.67'  , '0.86'])
        axsErr[2,1].set_xlabel(r'$\lambda$')
        axsErr[2,1].set_ylabel('Total AOD')    
        axs2[2,1].set_xlabel(r'$\lambda$')
        axs2[2,1].set_ylabel('Total AOD')
        # axsErr[0,0].legend(prop = { "size": 21 }, ncol=1)
        if No ==0:
            axs2[2,1].legend()
            axs2[0,0].legend(prop = { "size": 21 }, ncol=2)
        
        # lat_t = Hex['latitude']
        # lon_t = Hex['longitude']
        # dt_t = Hex['datetime']
        plt.tight_layout(rect=[0, 0, 1, 1])
        
    plt.savefig(f'/home/gregmi/ORACLES/HSRL_RSP/{RepMode}_RSPRetrieval{RepMode}.png', dpi = 400)
        

def PlotNeighbouringPIxs() :  



    HSRLPixNo = FindPix(LatH,LonH,LatRSP,LonRSP)[0]

    SphHSRL =[]
    TAMUHSRL =[]
    PixIndx =[]

    for i in range(4):
        HSRLPixNo = HSRLPixNo+i
        HSRL_sphrodT = HSLR_run("sphro",HSRLfile_path,HSRLfile_name,HSRLPixNo,nwl,ModeNo=3, updateYaml= False,releaseYAML= True, VertProfConstrain = True,LagrangianOnly = True,  AprioriLagrange =  AprioriLagrange[i])
        # plot_HSRL(HSRL_sphrodT[0][0],HSRL_sphrodT[0][0], UNCERT,forward = True, retrieval = True, Createpdf = True,PdfName ="/home/gregmi/ORACLES/rsltPdf/HSRL_Only_Plots_444.pdf", combinedVal =HSRL_sphrodT[2]) 
        HSRL_TamuT = HSLR_run("TAMU",HSRLfile_path,HSRLfile_name, HSRLPixNo,nwl,ModeNo=3,updateYaml= False,releaseYAML= True, VertProfConstrain = True,LagrangianOnly = True,  AprioriLagrange =  AprioriLagrange[i])
        # plot_HSRL(HSRL_Tamu[0][0],HSRL_Tamu[0][0], forward = True, retrieval = True, Createpdf = True,PdfName ="/home/gregmi/ORACLES/rsltPdf/HSRL_Only_Plots_444.pdf", combinedVal =HSRL_Tamu[2])    
        plot_HSRL(HSRL_sphrodT[0][0],HSRL_TamuT[0][0],UNCERT, forward = True, retrieval = True, Createpdf = True,PdfName ="/home/gregmi/ORACLES/rsltPdf/HSRL_Only_Plots_444.pdf", combinedVal =HSRL_TamuT[2])

        SphHSRL.append(HSRL_sphrodT)
        TAMUHSRL.append(HSRL_TamuT)
        PixIndx.append(HSRLPixNo) 

def PlotHSRLResidual(HSRL_sphrod,HSRL_Tamu,key1=None,key2=None):
    Hsph,HTam = HSRL_sphrod,HSRL_Tamu
    font_name = "Times New Roman"
    plt.rcParams['font.size'] = '14'


    NoMode = HSRL_sphrod['r'].shape[0]

    Index1,Index2 = [0,1,2],[0,1,2] # Index for using just HSRL retrieval
 
    IndexH, IndexRSP= [0,3,7],[1,2,4,5,6] #Index when using joint RSP+HSRL retrievaL
 
    #Colors for each aerosol modes
    cm_sp = ['k','#8B4000', '#87C1FF']  #Spheroid
    cm_t = ["#BC106F",'#E35335', 'b']  #Hex
    # color_sph = "#025043"
    color_sph = '#0c7683'
    color_tamu = "#d24787"
    
    if len(Hsph['lambda']) > 3:  #TODO Make this more general
        Index1 = IndexH #Input is Lidar+polarimter

    if len(HTam['lambda']) > 3: 
        Index2 = IndexH #Input is Lidar+polarimter
    
    if RSP_plot !=None:
        Index3 = IndexRSP
        
    if len(HTam['lambda']) !=len(Hsph['lambda']) :
        cm_sp = ['#AA98A9','#CB9129', 'c']
        cm_t = ['#900C3F','#8663bd', '#053F5C']
        color_tamu = '#DE970B'
        color_sph = '#1175A8'

#Plotting the fits if forward is set to True

    #Converting range to altitude
    altd = (Hsph['RangeLidar'][:,0])/1000 #altitude for spheriod
    altT = (Hsph['RangeLidar'][:,0])/1000 #altitude for hexahedra
    
    
    fig, axs= plt.subplots(nrows = 1, ncols =3, figsize= (15,6))  #TODO make it more general which adjust the no of rows based in numebr of wl
    plt.subplots_adjust(top=0.78)
    for i in range(3):
        wave = str(Hsph['lambda'][i]) +"μm"
        # axs[0].errorbar(Hsph['meas_VBS'][:,i],altd,xerr= UNCERT['VBS'],color = "#695E93", capsize=3,capthick =1,alpha =0.4, label =f"{UNCERT['VBS']}")
        
        # # axs[0].plot(Hsph['meas_VBS'][:,Index1[i]],altd, marker =">",color = "#281C2D", label ="Meas")
        axs[0].plot(abs(Hsph['meas_VBS'][:,Index1[i]]-Hsph['fit_VBS'][:,Index1[i]]),altd,color =cm_sp[i], marker = "$O$",label =f"sph{HTam['lambda'][i]}",alpha =0.8)
        axs[0].plot(abs(HTam['meas_VBS'][:,Index1[i]]-HTam['fit_VBS'][:,Index2[i]]),altd,color = cm_t[i],ls = "--", label=f"hex{HTam['lambda'][i]}", marker = "h")

        axs[0].xaxis.set_major_formatter(ScalarFormatter(useMathText=True))

        # print(UNCERT)
        # axs[i].errorbar(Hsph['fit_VBS'][:,i],altd,xerr= UNCERT['VBS'],color = "#025043", marker = "$O$",label ="Sphd")
        # axs[i].errorbar(HTam['fit_VBS'][:,i],altd,xerr= UNCERT['VBS'],color = "#d24787",ls = "--", label="Hex", marker = "h")

        axs[0].set_xlabel(f'$ Err VBS (m^{-1}Sr^{-1})$')
        axs[0].set_ylabel('Height above ground (km)')

        axs[0].set_title('VBS')
        
        axs[0].legend()
    #     plt.suptitle(f"HSRL Vertical Backscatter profile  \nLat,Lon: {HTam['latitude']}, {HTam['longitude']} Date: {HTam['datetime']}\n ",fontproperties=font_name) #Initial condition strictly constrainted by RSP retrievals
    # # pdf_pages.savefig()
    

    for i in range(3):
        wave = str(Hsph['lambda'][i]) +"μm"

        # axs[1].errorbar(Hsph['meas_DP'][:,Index1[i]],altd,xerr= UNCERT['DP'],color = '#695E93', capsize=4,capthick =1,alpha =0.6, label =f"{UNCERT['DP']}%")
        
        axs[1].plot(abs(Hsph['meas_DP'][:,Index1[i]]-Hsph['fit_DP'][:,Index1[i]]),altd,color = cm_sp[i], marker = "$O$",label =f"{key1}")
        axs[1].plot(abs(Hsph['meas_DP'][:,Index1[i]]-HTam['fit_DP'][:,Index2[i]]),altd,color = cm_t[i], ls = "--",marker = "h",label=f"{key2}")
        
        axs[1].set_xlabel('Err DP %')
        axs[1].set_title('DP')
        # if i ==0:
        #     axs[1].legend()
        axs[0].set_ylabel('Height above ground (km)')
    #     plt.suptitle(f"HSRL Depolarization Ratio \n Lat,Lon: {HTam['latitude']}, {HTam['longitude']}  Date: {HTam['datetime']}\n ") #Initial condition strictly constrainted by RSP retrievals
    # # pdf_pages.savefig()
    # fig.savefig(f'/home/gregmi/ORACLES/HSRL_RSP/FIT_HSRL Depolarization Ratio{NoMode}.png',dpi = 300)
    # fig, axs= plt.subplots(nrows = 1, ncols =2, figsize= (11,6))
    plt.subplots_adjust(top=0.78)
    for i in range(2):
        wave = str(Hsph['lambda'][i]) +"μm"

        # axs[2].errorbar(Hsph['meas_VExt'][:,Index1[i]],altd,xerr= UNCERT['VEXT']*Hsph['meas_VExt'][:,i],color = '#695E93',capsize=4,capthick =1,alpha =0.7, label =f"{UNCERT['VEXT']*100}%")
        
        # axs[2].plot(Hsph['meas_VExt'][:,Index1[i]],altd, marker =">",color = "#281C2D", label ="Meas")
        axs[2].plot(abs(Hsph['meas_VExt'][:,Index1[i]]-Hsph['fit_VExt'][:,Index1[i]]),altd,color = cm_sp[i], marker = "$O$",label =f"{key1}")
        axs[2].plot(abs(HTam['meas_VExt'][:,Index1[i]]-HTam['fit_VExt'][:,Index2[i]]),altd,color = cm_t[i],ls = "--", marker = "h",label=f"{key2}")
        
        axs[2].xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        axs[2].ticklabel_format(style="sci", axis="x")
          
        axs[2].set_xlabel(f'$VExt (m^{-1})$',fontproperties=font_name)
        axs[2].set_ylabel('Height above ground (km)',fontproperties=font_name)
        axs[2].set_title('VExt')
        # if i ==0:
        #     axs[0].legend()
        plt.suptitle(f"HSRL Vertical Extinction profile\n Lat,Lon: {HTam['latitude']},{HTam['longitude']}  Date: {HTam['datetime']}\n ",fontproperties=font_name) #Initial condition strictly constrainted by RSP retrievals
    plt.tight_layout()
    # pdf_pages.savefig()
    # fig.savefig(f'/home/gregmi/ORACLES/HSRL_RSP/FIT_HSRL_Vertical_Ext_profile{NoMode}.png',dpi = 300)
    fig.savefig(f'/home/gregmi/ORACLES/HSRL_RSP/FIT_HSRL_VBS_Err{NoMode}.png',dpi = 300)



def PlotDiffLidarCases(Case1S, Case1H, Case2S, Case2H,CaseJ1S, CaseJ1H,CaseJ2S, CaseJ2H, NoOfCases = None):
    plt.rcParams['font.size'] = '20'
    if NoOfCases == None:
        NoOfCases = 1
    fig3, axs3= plt.subplots(nrows = 1, ncols =3, figsize= (16.5,8))
    fig1, axs1= plt.subplots(nrows = 1, ncols =3, figsize= (16.5,8))  #TODO make it more general which adjust the no of rows based in numebr of wl
    fig2, axs2= plt.subplots(nrows = 1, ncols =3, figsize= (16.5,8))
          
    itr = 2*NoOfCases
    
    for it in range (itr):
        if it ==0:
            Hsph,HTam = Case1S, Case1H
            color_sph = '#5A6CE2'
            color_tamu = "#A20D6A"

            key = 'HSRL'

            markerm = '>'
            markers ='$O$'
            markerh ="D"
            ms = 7
            mssph = 12

        if it ==1: 
            Hsph,HTam = Case2S, Case2H
            color_tamu = '#FFB3FF' 
            color_sph = "#A6D854"

            key = 'Joint'

            ms = 7
            mssph = 12
         


            markerm = '>'
            markers ='$O$'
            markerh ="D"
        
        if it ==2:
            Hsph,HTam = CaseJ1S, CaseJ1H
            color_sph = '#A6D854'
            color_tamu = "#7072E0"

            markerm = 'd'
            markers ='$+$'
            markerh ="o"

        if it ==3:
            Hsph,HTam = CaseJ2S, CaseJ2H
            color_sph = '#DFEE88'
            color_tamu = "#AA4499"

            markerm = 'd'
            markers ='$d$'
            markerh ="*"


            

            
        font_name = "Times New Roman"
       

        NoMode = Hsph['r'].shape[0]
        Index1 = [0,1,2]
        Index2 = [0,1,2]
        IndexH = [0,3,7]
        IndexRSP = [1,2,4,5,6]


        
        if len(Hsph['lambda']) > 3: 
            Index1 = IndexH #Input is Lidar+polarimter
            

        if len(HTam['lambda']) > 3: 
            Index2 = IndexH 
            # color_tamu = '#DE970B'
            # color_sph = '#1175A8'
            
        
        if RSP_plot !=None:
            Index3 = IndexRSP
            
        if len(HTam['lambda']) !=len(Hsph['lambda']) :
            cm_sp = ['#AA98A9','#CB9129', 'c']
            cm_t = ['#900C3F','#8663bd', '#053F5C']
            # color_tamu = '#DE970B'
            # color_sph = '#1175A8'

    #Plotting the fits if forward is set to True

        #Converting range to altitude
        altd = (Hsph['RangeLidar'][:,0])/1000 #altitude for spheriod
        altT = (Hsph['RangeLidar'][:,0])/1000 #altitude for hexahedra


        # altd2 = (Hsph2['RangeLidar'][:,0])/1000 #altitude for spheriod
        # altT2 = (Hsph2['RangeLidar'][:,0])/1000 #altitude for hexahedra
        
        # fig, axs1= plt.subplots(nrows = 1, ncols =3, figsize= (15,6))  #TODO make it more general which adjust the no of rows based in numebr of wl
        plt.subplots_adjust(top=0.78)
        for i in range(3):
            wave = str(Hsph['lambda'][i]) +"μm"

            # if it==0:
            #     axs1[i].errorbar(Hsph['meas_VBS'][:,i],altd,xerr= UNCERT['VBS'],color = "k", capsize=3,capthick =1,alpha =0.4, label =f"{UNCERT['VBS']}")
            
            #     axs1[i].plot(Hsph['meas_VBS'][:,Index1[i]],altd, marker =markerm,markersize= ms,color = "#281C2D", label ="Meas")
            axs1[i].plot(HTam['fit_VBS'][:,Index2[i]],altd,color =  color_tamu,ls = "--", label=f"Hex{key}", marker = markerh,markersize= ms,)
            axs1[i].plot(Hsph['fit_VBS'][:,Index1[i]],altd,color =color_sph, marker = markers,markersize= mssph,label =f"Sph{key}",alpha =0.8)
            if it==1:
                
                axs1[i].plot(Hsph['meas_VBS'][:,Index1[i]],altd,lw =3,  marker =markerm,markersize= ms,color = "#281C2D", label ="Meas")
                axs1[i].errorbar(Hsph['meas_VBS'][:,Index1[i]],altd,xerr= UNCERT['VBS'],color = "k", capsize=3,capthick =1,alpha =1, label =f"{UNCERT['VBS']}")
            if it==0 :  
                axs1[i].set_title(wave)
            
            # axs1[i].xscale("log")
            # axs1[i].set_xscale("log")


            # axs[i].plot(Hsph2['meas_VBS'][:,Index1[i]],altd2, marker =">",color = "#281C2D", label ="C2Meas")
            # axs[i].plot(Hsph2['fit_VBS'][:,Index1[i]],altd2,color =color_sph, marker = "$O$",label =f"C2Sph",alpha =0.8)
            # axs[i].plot(HTam2['fit_VBS'][:,Index2[i]],altd2,color =  color_tamu,ls = "--", label =f"C2Hex", marker = "h")

            axs1[i].xaxis.set_major_formatter(ScalarFormatter(useMathText=True))

            # print(UNCERT)
            # axs[i].errorbar(Hsph['fit_VBS'][:,i],altd,xerr= UNCERT['VBS'],color = "#025043", marker = "$O$",label ="Sphd")
            # axs[i].errorbar(HTam['fit_VBS'][:,i],altd,xerr= UNCERT['VBS'],color = "#d24787",ls = "--", label="Hex", marker = "h")

            axs1[i].set_xlabel(f'$ VBS (m^{-1}Sr^{-1})$',fontproperties=font_name)
            axs1[0].set_ylabel('Altitude(km)',fontproperties=font_name)
            # axs1[i].xscale("log")
            
            
            # plt.suptitle(f"HSRL Vertical Backscatter profile  \nLat,Lon: {HTam['latitude']}, {HTam['longitude']} Date: {HTam['datetime']}\n ",fontproperties=font_name) #Initial condition strictly constrainted by RSP retrievals
        # pdf_pages.savefig()

        if i ==0:
                axs1[0].legend(prop = { "size": 18 }, ncol=1)
        fig1.savefig(f'/home/gregmi/ORACLES/HSRL_RSP/HSRLonlyandCombVertical_Backscatter_profile_comp{NoMode}.png',dpi = 300)
            # plt.tight_layout()
        # fig, axs2= plt.subplots(nrows = 1, ncols =3, figsize= (15,6))
        plt.subplots_adjust(top=0.78)

        # AOD = np.trapz(Hsph['meas_VExt'],altd)
        for i in range(3):
            wave = str(Hsph['lambda'][i]) +"μm"

            axs2[i].plot(HTam['fit_DP'][:,Index2[i]],altd,color = color_tamu, ls = "--",marker = markerh,markersize= ms,label=f"Hex{key}")
            axs2[i].plot(Hsph['fit_DP'][:,Index1[i]],altd,color = color_sph, marker = markers,markersize= mssph,label =f"Sph{key}")
            if it ==1:

                axs2[i].errorbar(Hsph['meas_DP'][:,Index1[i]],altd,xerr= UNCERT['DP'],color = 'k', capsize=4,capthick =1,alpha =0.5, label =f"{UNCERT['DP']}%")
                
                axs2[i].plot(Hsph['meas_DP'][:,Index1[i]],altd, marker =markerm,markersize= ms,lw =3, color = "#281C2D", label ="C1Meas")
            if it==0 : 
                axs2[i].set_title(wave)
            # axs[i].plot(Hsph2['meas_DP'][:,Index1[i]],altd2, marker =">",color = "#281C2D", label ="C2Meas")
            # axs[i].plot(Hsph2['fit_DP'][:,Index1[i]],altd2,color = color_sph, marker = "$O$",label =f"C2Sph")
            # axs[i].plot(HTam2['fit_DP'][:,Index2[i]],altd2,color = color_tamu, ls = "--",marker = "h",label=f"C2Hex")
        

            # axs[i].errorbar(Hsph['fit_DP'][:,i],altd,xerr= UNCERT['DP'],fmt='-o',color = "#025043", marker = "$O$",label ="Sphd")
            # axs[i].errorbar(HTam['fit_DP'][:,i],altd,xerr= UNCERT['DP'],fmt='-o',color = "#d24787",ls = "--", label="Hex", marker = "h")


            axs2[i].set_xlabel('DP %')
            if i ==0:
                axs2[0].legend(prop = { "size": 18 }, ncol=1)

            
            # if i ==0:
                # axs2[0].legend(prop = { "size": 12 }, ncol=2)
            axs2[0].set_ylabel('Altitude(km)',fontproperties=font_name)
            # plt.suptitle(f"HSRL Depolarization Ratio \n Lat,Lon: {HTam['latitude']}, {HTam['longitude']}  Date: {HTam['datetime']}\n ",fontproperties=font_name) #Initial condition strictly constrainted by RSP retrievals
        # pdf_pages.savefig()
        fig2.savefig(f'/home/gregmi/ORACLES/HSRL_RSP/HSRLonlyandComb_Depolarization_Ratio_{NoMode}.png',dpi = 300)
        # fig, axs3= plt.subplots(nrows = 1, ncols =2, figsize= (11,6))
        # plt.subplots_adjust(top=0.78)
        for i in range(2):

            axs3[i].ticklabel_format(style="sci", axis="x")
            wave = str(Hsph['lambda'][i]) +"μm"
            axs3[i].plot(1000*HTam['fit_VExt'][:,Index2[i]],altd,color = color_tamu,ls = "--", marker = markerh,markersize= ms,label=f"Hex{key}")
            axs3[i].plot(1000*Hsph['fit_VExt'][:,Index1[i]],altd,color = color_sph, markersize= mssph, marker =  markers,label =f"Sph{key}")
            if it ==1:
                axs3[i].errorbar(1000*Hsph['meas_VExt'][:,Index1[i]],altd,xerr= 1000*UNCERT['VEXT'],color = 'k',capsize=4,capthick =1,alpha =0.5, label =f"{UNCERT['VEXT']*100}%")
                
                axs3[i].plot(1000*Hsph['meas_VExt'][:,Index1[i]],altd, marker =">",lw = 3, markersize= ms,color = "#281C2D", label ="C1Meas")
            if it==0 : 
                axs3[i].set_title(wave)
            # axs3[i].xscale("log")
            # axs3[i].set_xscale("log")
            # axs[i].plot(Hsph2['meas_VExt'][:,Index1[i]],altd2, marker =">",color = "#281C2D", label ="C2Meas")
            # axs[i].plot(Hsph2['fit_VExt'][:,Index1[i]],altd2,color = color_sph, marker = "$O$",label =f"C2Sph")
            # axs[i].plot(HTam2['fit_VExt'][:,Index2[i]],altd2,color = color_tamu,ls = "--", marker = "h",label=f"C2Hex")
            



            axs3[i].xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
            
            # axs[i].errorbar(Hsph['fit_VExt'][:,i],altd,xerr= UNCERT['VEXT'],color = "#025043", marker = "$O$",label ="Sphd")
            # axs[i].errorbar(HTam['fit_VExt'][:,i],altd,xerr= UNCERT['VEXT'],color = "#d24787",ls = "--", label="Hex", marker = "h")

            
            
            axs3[i].set_xlabel(f'$VExt (km^{-1})$',fontproperties=font_name)
            axs3[0].set_ylabel('Altitude(km)',fontproperties=font_name)
            

            if i ==0:
                axs3[0].legend(prop = { "size": 18 }, ncol=1)
            # if i ==0:
                # axs3[0].legend(prop = { "size": 12 }, ncol=2)
            # plt.suptitle(f"HSRL Vertical Extinction profile\n Lat,Lon: {HTam['latitude']},{HTam['longitude']}  Date: {HTam['datetime']}\n ",fontproperties=font_name) #Initial condition strictly constrainted by RSP retrievals
        plt.tight_layout()
        # pdf_pages.savefig()
        fig3.savefig(f'/home/gregmi/ORACLES/HSRL_RSP/HSRLonlyandComb{NoMode}.png',dpi = 300)

PlotDiffLidarCases(HSRL_sphrodT[0][0], HSRL_TamuT[0][0] ,LidarPolSph[0][0], LidarPolTAMU[0][0],HSRL_sphrodT[0][0], HSRL_TamuT[0][0],LidarPolSph[0][0], LidarPolTAMU[0][0], NoOfCases = 1)

def PlotDiffRSPCases(rslts_Sph2,rslts_Tamu2,LidarPolSph,LidarPolTAMU, fn = None):

    fig, axs = plt.subplots(nrows= 4, ncols=5, figsize=(35, 17),gridspec_kw={'height_ratios': [1, 0.3,1,0.3]}, sharex='col')
    plt.rcParams['font.size'] = '26'
    plt.rcParams["figure.autolayout"] = True

    

  
    for i in range (2):
        if i == 0:
            Spheriod,Hex = rslts_Sph[0],rslts_Tamu[0]
            color_sph = '#5A6CE2'
            color_tamu = "#A20D6A"
            wl = rslts_Sph[0]['lambda'] 
            wlIdx = np.arange(len(wl))
            key = "RSP"
            print(wl,wlIdx )



        if i == 1:
            Spheriod,Hex = LidarPolSph[0][0], LidarPolTAMU[0][0]
            color_tamu = '#FFB3FF' 
            color_sph = "#A6D854"
            key = "RSP+HSRL2"

            wlIdx = [1,2,4,5,6]  #Lidar wavelngth indices

            print(wl,wlIdx )

            # key = 'Joint'

            # ms = 7
            # mssph = 12
        
            # markerm = '>'
            # markers ='$O$'
            # markerh ="D"
      
        #Stokes: 
            
             # Plot the AOD data
            meas_P_rel = 'meas_P_rel'

        
        for nwav in range(len(wlIdx)):
        # Plot the fit and measured I data
            
            # axs[0, nwav].fill_between(Spheriod ['sca_ang'][:,wlIdx[nwav]],Spheriod ['meas_I'][:,wlIdx[nwav]], Spheriod ['meas_I'][:,wlIdx[nwav]]*1.03, color = 'k',alpha=0.1, ls = "--",label="+3%")
            # axs[0, nwav].fill_between(Spheriod ['sca_ang'][:,wlIdx[nwav]],Spheriod ['meas_I'][:,wlIdx[nwav]], Spheriod ['meas_I'][:,wlIdx[nwav]]*0.97, color = "k",alpha=0.1, ls = "--",label="-3%")
            
            axs[0, nwav].plot(Spheriod ['sca_ang'][:,wlIdx[nwav]], Spheriod ['fit_I'][:,wlIdx[nwav]],color =color_sph , lw = 5, ls = '--',label=f"{key} sph")
            # axs[0, nwav].scatter(Spheriod ['sca_ang'][:,nwav][marker_indsp], Spheriod ['fit_I'][:,nwav][marker_indsp],color =color_sph , m = "o",label="fit sphrod")
            
            # axs[0, nwav].set_xlabel('Scattering angles (deg)')
            axs[0, 0].set_ylabel('I')
            # axs[0, nwav].legend()

            # Plot the fit and measured QoI data
            axs[2, nwav].plot(Spheriod ['sca_ang'][:,wlIdx[nwav]], Spheriod ['meas_P_rel'][:,wlIdx[nwav]],color = "k", lw = 5, label="meas")
            axs[2, nwav].plot(Spheriod ['sca_ang'][:,wlIdx[nwav]], Spheriod ['fit_P_rel'][:,wlIdx[nwav]], color =color_sph, lw = 5, ls = '--', label=f"{key} sph")
            
          
            axs[2, 0].set_ylabel('DOLP')
            if i ==0:
                axs[0, nwav].set_title(f"{wl[wlIdx[nwav]]} $\mu m$", fontsize = 22)
            
            axs[0, nwav].plot(Hex['sca_ang'][:,wlIdx[nwav]], Hex['fit_I'][:,wlIdx[nwav]],color =color_tamu , lw = 5, ls = "dashdot",label=f"{key}Hex")
            axs[0, nwav].plot(Spheriod['sca_ang'][:,wlIdx[nwav]], Spheriod['meas_I'][:,wlIdx[nwav]], color = "k", lw = 5, label="meas")

           
            axs[2, nwav].plot(Hex['sca_ang'][:,wlIdx[nwav]],Hex['fit_P_rel'][:,wlIdx[nwav]],color = color_tamu , lw = 5,ls = "dashdot", label = f"{key} Hex") 


            sphErr = 100 * abs(Spheriod['meas_I'][:,wlIdx[nwav]]-Spheriod ['fit_I'][:,wlIdx[nwav]] )/Spheriod['meas_I'][:,wlIdx[nwav]]
            HexErr = 100 * abs(Hex['meas_I'][:,wlIdx[nwav]]-Hex ['fit_I'][:,wlIdx[nwav]] )/Hex['meas_I'][:,wlIdx[nwav]]
            
            axs[1, nwav].plot(Spheriod ['sca_ang'][:,wlIdx[nwav]], sphErr,color =color_sph , lw = 5,label="Sphrod")
            axs[1, nwav].plot(Hex ['sca_ang'][:,wlIdx[nwav]], HexErr,color = color_tamu, lw = 5 ,label="Hex")
            axs[1, 0].set_ylabel('Err I %')
            
    #Absolute error
            sphErrP =  abs(Spheriod['meas_P_rel'][:,wlIdx[nwav]]-Spheriod ['fit_P_rel'][:,wlIdx[nwav]])
            HexErrP =  abs(Hex['meas_P_rel'][:,wlIdx[nwav]]-Hex['fit_P_rel'][:,wlIdx[nwav]] )
            
            axs[3, nwav].plot(Spheriod ['sca_ang'][:,wlIdx[nwav]], sphErrP,color =color_sph , lw = 5,label="Sphrod")
            axs[3, nwav].plot(Hex ['sca_ang'][:,wlIdx[nwav]], HexErrP,color =color_tamu, lw = 5 ,label="Hex")
            
            axs[3, nwav].set_xlabel(r'$\theta$s(deg)')
            # axs[3, nwav].set_ylabel('Err P')
            # axs[3, nwav].legend()
            axs[3, nwav].set_xlabel(r'$\theta$s(deg)')
            axs[3, 0].set_ylabel('|Meas-fit|')
        
            # axs[1, nwav].set_title(f"{wl[nwav]}", fontsize = 14)

            axs[0, 4].legend(bbox_to_anchor=(1.05, 0.5), loc='center left')
            # axs[1, 4].legend(bbox_to_anchor=(1.05, 0.5), loc='center left')


            lat_t = Hex['latitude']
            lon_t = Hex['longitude']
            dt_t = Hex['datetime']
            plt.tight_layout(rect=[0, 0, 1, 1])
            # plt.suptitle(f'RSP Aerosol Retrieval \n  Lat:{lat_t} Lon :{lon_t}   Date: {dt_t}')
            if fn ==None : fn = "1"
            fig.savefig(f'/home/gregmi/ORACLES/HSRL_RSP/Combined_{dt_t}_RSPFits_{fn}.png', dpi = 400)

                
# def ErrorEstimates(rslts_Sph,rslts_Tamu,HSRL_sphrodT,HSRL_TamuT,LidarPolSph,LidarPolTAMU):




from scipy import stats
import pandas as pd

def ErrHeatMap(rslt1, rslt2):
    Retrival = ['aod','ssa','aodMode','ssaMode','n', 'k']
    # Retrival = ['rv','sigma','sph']
    wlIdx = len(rslt1['lambda'])
    Xticks =[]

    ModeName = ['fine', 'dust', 'marine']
    df = pd.DataFrame()
    for i,key in enumerate(Retrival):

        
        for mode in range(3) : #Calculating error for each mode
            if key == 'aod' or key =='ssa':
                df.loc[key, rslt1['lambda']] = rslt1[key] - rslt2[key]
                

            
            if key== 'dVdlnr': #if keys contain the phase mode

                t_stat, p_value = stats.ttest_ind(rslt1[key],rslt2[key])

                df.loc[key, rslt1['lambda']] = np.repeat(t_stat,len(wlIdx))
                # if i ==0:
                #     Xticks.append(key)


            if key == 'rv' or key =='sigma'or key =='sph'or key =='aod'or key =='ssa':
                err = rslt1[key][mode] - rslt2[key][mode]
                

                df.loc[f'{key[:3]}_{ModeName[mode]}', rslt1['lambda']] = np.repeat(err,wlIdx)
                # if i ==0:
                #     Xticks.append(key)
            else:
                df.loc[f'{key[:3]}_{ModeName[mode]}', rslt1['lambda']] = rslt1[key][mode] - rslt2[key][mode]
                # if i ==0:
                #     Xticks.append(f'{key[:3]}_{ModeName[mode]}')
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(df.astype(float), annot=True, cmap='coolwarm', annot_kws={"weight": "bold"}, cbar_kws={'label': 'Error'}, xticklabels=rslt1['lambda'])
    plt.title('Heatmap of Alsolute Errors (Spheroid-Hex) Shape Models')
    # plt.xticks(rslt1['lambda'])
    # plt.xticks(np.unique(Xticks))
    
    plt.show()



    return df

df=ErrHeatMap(rslts_Sph[0],rslts_Tamu[0])

#Caculating the chi-square



def chiSquare(meas,fit,measErr):
    chisquare = np.sum(((meas-fit)/measErr)**2) 

    return chisquare



costValCal = {} #stores chi square value for different retrieval techniques 
chiHSrl ={}



Rsltdic = HSRL_sphrodT[0][0]
MeasKeys = ['VBS','DP','VExt']
MeasErrhsrl= [3e-7,1,1e-5]

for i in range(len(MeasKeys )):
    for wlIndx in range(len(Rsltdic[f'meas_{MeasKeys[i]}'][i,:])):
        chiHSrl[f'{MeasKeys[i]}_{wlIndx}']=chiSquare(Rsltdic[f'meas_{MeasKeys[i]}'][:,wlIndx],Rsltdic[f'fit_{MeasKeys[i]}'][:,wlIndx],MeasErrhsrl[i])

NTotalHSRL = 8* len(Rsltdic[f'meas_{MeasKeys[i]}'][:,0])  
costValCal['HSRL_sph'] = sum(value for value in chiHSrl.values() if not math.isnan(value))/NTotalHSRL 


chiHSrl_hex ={}
Rsltdic = HSRL_TamuT[0][0]
for i in range(len(MeasKeys )):
    for wlIndx in range(len(Rsltdic[f'meas_{MeasKeys[i]}'][i,:])):
        chiHSrl_hex[f'{MeasKeys[i]}_{wlIndx}']=chiSquare(Rsltdic[f'meas_{MeasKeys[i]}'][:,wlIndx],Rsltdic[f'fit_{MeasKeys[i]}'][:,wlIndx],MeasErrhsrl[i])
        
costValCal['HSRL_hex'] = sum(value for value in chiHSrl_hex.values() if not math.isnan(value))/NTotalHSRL 



chiRSP_hex ={}
MeasKeys = ['I','P_rel']
measErrRSP = [0.03, 0.005]
Rsltdic = rslts_Tamu[0]

for i in range(len(MeasKeys )):
    for wlIndx in range(len(Rsltdic[f'meas_{MeasKeys[i]}'][i,:])):
        
        if MeasKeys[i] == "I":  #relative error
            chiRSP_hex[f'{MeasKeys[i]}_{wlIndx}']=chiSquare(Rsltdic[f'meas_{MeasKeys[i]}'][:,wlIndx],Rsltdic[f'fit_{MeasKeys[i]}'][:,wlIndx],Rsltdic[f'meas_{MeasKeys[i]}'][:,wlIndx]*measErrRSP[i]) #Relative err

        else:
            chiRSP_hex[f'{MeasKeys[i]}_{wlIndx}']=chiSquare(Rsltdic[f'meas_{MeasKeys[i]}'][:,wlIndx],Rsltdic[f'fit_{MeasKeys[i]}'][:,wlIndx],measErrRSP[i])


        
NTotalRSP = 10* len(Rsltdic[f'meas_{MeasKeys[i]}'][:,0])
costValCal['RSP_hex'] = sum(value for value in chiRSP_hex.values() if not math.isnan(value))/NTotalRSP
costValCal



chiRSP_sph ={}
MeasKeys = ['I','P_rel']
Rsltdic = rslts_Sph[0]

for i in range(len(MeasKeys )):
    for wlIndx in range(len(Rsltdic[f'meas_{MeasKeys[i]}'][i,:])):

        if MeasKeys[i] == "I":  #relative error

            chiRSP_sph[f'{MeasKeys[i]}_{wlIndx}']=chiSquare(Rsltdic[f'meas_{MeasKeys[i]}'][:,wlIndx],Rsltdic[f'fit_{MeasKeys[i]}'][:,wlIndx],Rsltdic[f'meas_{MeasKeys[i]}'][:,wlIndx]*measErrRSP[i])

        else:
            chiRSP_sph[f'{MeasKeys[i]}_{wlIndx}']=chiSquare(Rsltdic[f'meas_{MeasKeys[i]}'][:,wlIndx],Rsltdic[f'fit_{MeasKeys[i]}'][:,wlIndx],measErrRSP[i])
        
costValCal['RSP_sph'] = sum(value for value in chiRSP_sph.values() if not math.isnan(value))/ NTotalRSP
costValCal



HSRL_keys = ['meas_VBS', 'meas_DP', 'meas_VExt']
RSP_keys =['meas_I', 'meas_P_rel']

J_sph ={}
MeasKeys = ['VBS','DP','VExt','I','P_rel']
MeasErrJ = [3e-7,1,1e-5,0.03, 0.005]
Rsltdic = LidarPolSph[0][0]

for i in range(len(MeasKeys)):
    for wlIndx in range(len(Rsltdic[f'meas_{MeasKeys[i]}'][i,:])):

        if [f'meas_{MeasKeys[i]}'] == "meas_I":  #relative error
            Chi=chiSquare(Rsltdic[f'meas_{MeasKeys[i]}'][:,wlIndx],Rsltdic[f'fit_{MeasKeys[i]}'][:,wlIndx],MeasErrJ[i]*Rsltdic[f'meas_{MeasKeys[i]}'][:,wlIndx])
            J_sph[f'{MeasKeys[i]}_{wlIndx}'] = Chi
        else:

            Chi=chiSquare(Rsltdic[f'meas_{MeasKeys[i]}'][:,wlIndx],Rsltdic[f'fit_{MeasKeys[i]}'][:,wlIndx],MeasErrJ[i])
            J_sph[f'{MeasKeys[i]}_{wlIndx}'] = Chi



chiVBS = (J_sph['VBS_0'] + J_sph['VBS_3']+J_sph['VBS_7'])/(3*len(Rsltdic['meas_VBS'][:,0]))
chiDP= (J_sph['DP_0'] + J_sph['DP_3']+J_sph['DP_7'])/(3*len(Rsltdic['meas_DP'][:,0]))
chiVExt = (J_sph['VExt_0'] + J_sph['VExt_3'])/(2*len(Rsltdic['meas_VExt'][:,0]))

chiI =  (J_sph['I_1']+ J_sph['I_2']+ J_sph['I_4']+ J_sph['I_5']+ J_sph['I_6'])/(5*len(Rsltdic['meas_I'][:,0]))
chiDolp =  (J_sph['P_rel_1']+ J_sph['P_rel_4']+ J_sph['P_rel_5']+ J_sph['P_rel_6']+ J_sph['P_rel_2'])/(5*len(Rsltdic['meas_P_rel'][:,0]))

costValCal['J_sph'] = (chiVBS+chiDP+chiVExt+chiI+chiDolp)/5
costValCal

    

J_hex ={}
MeasKeys = ['VBS','DP','VExt','I','P_rel']
Rsltdic = LidarPolTAMU[0][0]


for i in range(len(MeasKeys)):
    for wlIndx in range(len(Rsltdic[f'meas_{MeasKeys[i]}'][i,:])):

        if [f'meas_{MeasKeys[i]}'] == "meas_I":  #relative error
            Chi=chiSquare(Rsltdic[f'meas_{MeasKeys[i]}'][:,wlIndx],Rsltdic[f'fit_{MeasKeys[i]}'][:,wlIndx],MeasErrJ[i]*Rsltdic[f'meas_{MeasKeys[i]}'][:,wlIndx])
            J_hex[f'{MeasKeys[i]}_{wlIndx}'] = Chi
        else:

            Chi=chiSquare(Rsltdic[f'meas_{MeasKeys[i]}'][:,wlIndx],Rsltdic[f'fit_{MeasKeys[i]}'][:,wlIndx],MeasErrJ[i])
            J_hex[f'meas_{MeasKeys[i]}_{wlIndx}'] = Chi



HchiVBS = (J_hex['meas_VBS_0'] + J_hex['meas_VBS_3']+J_hex['meas_VBS_7'])/(3*len(Rsltdic['meas_VBS'][:,0]))
HchiDP= (J_hex['meas_DP_0'] + J_hex['meas_DP_3']+J_hex['meas_DP_7'])/(3*len(Rsltdic['meas_DP'][:,0]))
HchiVExt = (J_hex['meas_VExt_0'] + J_hex['meas_VExt_3'])/(2*len(Rsltdic['meas_VExt'][:,0]))

HchiI =  (J_hex['meas_I_1']+ J_hex['meas_I_2']+ J_hex['meas_I_4']+ J_hex['meas_I_5']+ J_hex['meas_I_6'])/(5*len(Rsltdic['meas_I'][:,0]))
HchiDolp = (J_hex['meas_P_rel_1']+ J_hex['meas_P_rel_4']+ J_hex['meas_P_rel_5']+ J_hex['meas_P_rel_6']+ J_hex['meas_P_rel_2'])/(5*len(Rsltdic['meas_P_rel'][:,0]))

costValCal['J_hex'] = (HchiVBS+HchiDP+HchiVExt+HchiI+HchiDolp)/5
costValCal


def compareWithHiGEAR(NoMeas,SavImgName ):

    # plt.rcParams['font.size'] = '16'


    file_name_APS = '/home/gregmi/ORACLES/HiGEAR/APS_P3_20180924_R2.nc'
    file_name_DMA = '/home/gregmi/ORACLES/HiGEAR/DMA_P3_20180924_R0.nc'
    file_name_HW = '/home/gregmi/ORACLES/HiGEAR/Howell-corrected-UHSAS_P3_20180924_R0.nc'
    file_name_LDMA = '/home/gregmi/ORACLES/HiGEAR/LDMA_P3_20180924_R0.nc'
    file_name_UHSAS = '/home/gregmi/ORACLES/HiGEAR/UHSAS_P3_20180924_R2.nc'

    Idx_APS = 214   #  Time : 09:07:26 
    Idx_UHSAS = 2556  #09:07:37.62000
    Idx_LDMA = 37    #09:07:31.139201
    Idx_HW = 2556    #09:07:37 
    Idx_DMA = 30    #09:06:47


    DvDlnr_APS,r_APS = Read_HiGear(file_name_APS, Idx_APS)
    DvDlnrHW,rHW= Read_HiGear(file_name_HW, Idx_HW)
    DvDlnrLDMA ,rLDMA = Read_HiGear(file_name_LDMA, Idx_LDMA)
    DvDlnrDM,rDM = Read_HiGear(file_name_DMA, Idx_DMA)
    DvDlnrUHSAS,rUHSAS = Read_HiGear(file_name_UHSAS, Idx_UHSAS)


    plt.scatter(rHW, DvDlnrHW, label ='HW')
    plt.scatter(rDM , DvDlnrDM, label ='DM')
    plt.scatter(r_APS , DvDlnr_APS, label ='APS')
    plt.plot(rLDMA , DvDlnrLDMA, label ='LDMA')

    # plt.scatter(rUHSAS , DvDlnrUHSAS, label ='UHSAS')
    # plt.xscale('log')
    plt.xscale('log')

    fig, axs2 = plt.subplots(nrows= 1, ncols=1, figsize=(12, 8))
        
    plt.rcParams['font.size'] = '20'
    for No in range(NoMeas):

        color_instrument = ['#FBD381','#A78236','#CDEDFF','#404790','#CE357D','#711E1E'] #Color blind friendly scheme for different retrieval techniques< :rsponlt, hsrl only and combined


        if No ==0:
            Spheriod,Hex = LidarPolSph[0][0],LidarPolTAMU[0][0]
            cm_sp = color_instrument[4]
            cm_t = color_instrument[5]
            key = 'HSRL+RSP'
        if No ==1:
            Spheriod,Hex = HSRL_sphrodT[0][0],HSRL_TamuT[0][0]
            cm_sp = color_instrument[2]
            cm_t = color_instrument[3]
            key = "HSRL"
        if No ==2:
            Spheriod,Hex = rslts_Sph[0],rslts_Tamu[0]
            cm_sp = color_instrument[0]
            cm_t = color_instrument[1]
            key='RSP'



        # plt.rcParams['font.size'] = '17'
        #Stokes Vectors Plot
        Retrival = ['dVdlnr','aodMode','ssaMode','n', 'k']
        #['sigma', 'vol', 'aodMode','ssaMode', 'rEff', 'costVal']
        Angles =   ['sza', 'vis', 'fis','angle' ]
        Stokes =   ['meas_I', 'fit_I', 'meas_P_rel', 'fit_P_rel']
        Pij    = ['p11', 'p12', 'p22', 'p33'], 
        Lidar=  ['heightStd','g','LidarRatio','LidarDepol', 'gMode', 'LidarRatioMode', 'LidarDepolMode']

        # Plot the AOD data


        # Plot the AOD data
        y = [0,1,2,0,1,2,]
        x = np.repeat((0,1),3)
        if Spheriod['r'].shape[0] ==2 :
            mode_v = ["fine", "dust","marine", 'NonSphMarine']
        if Spheriod['r'].shape[0] ==3 :
            mode_v = ["fine", "dust","marine", 'NonSphMarine']
      


        # cm_sp = ['k','#8B4000', '#87C1FF']
        # cm_t = ["#BC106F",'#E35335', 'b']

        #Retrivals:
        for i in range(len(Retrival)):
            a,b = i%3,i%2

            lambda_ticks1 = np.round(Spheriod['lambda'], decimals=2)
            lambda_ticks2 = np.round(Hex['lambda'], decimals=2)

            
            for mode in range(1): #for each modes
                if i ==0:

                    # axs2[a,b].errorbar(Spheriod['r'][mode], Spheriod[Retrival[i]][mode],xerr=UNCERT['rv'],capsize=5,capthick =2, marker = "$O$",color = cm_sp[mode],lw = 3,ls = linestyle[mode], label=f"Sphrod_{mode_v[mode]}")
                    # axs2[a,b].errorbar(Hex['r'][mode],Hex[Retrival[i]][mode], marker = "H",xerr=UNCERT['rv'],capsize=5,capthick =2, color = cm_t[mode] ,lw = 3, ls = linestyle[mode],label=f"Hex_{mode_v[mode]}")
                        
                    axs2.plot(Spheriod['r'][mode], Spheriod[Retrival[i]][mode], marker = "$O$",color = cm_sp,lw = 5.5, alpha = 0.8, label=f"{key}_Sph")
                    axs2.plot(Hex['r'][mode],Hex[Retrival[i]][mode], marker = "H", color = cm_t,lw = 5.5, alpha = 0.8,label=f"{key}_Hex")
                    # if RSP_plot != None:
                    #     axs2[a,b].plot(RSP_plot['r'][mode],RSP_plot[Retrival[i]][mode], marker = "H", color = cm_t[mode] ,lw = 5.5, ls = linestyle[mode],label=f"{key2}_{mode_v[mode]}")



    axs2.plot(rDM , DvDlnrDM, color = 'k', lw = 3, label ='HiGEAR-DM')
    # axs2.scatter(r_APS , DvDlnr_APS, color = '#0D0084', label ='APS')
    axs2.plot(rLDMA , DvDlnrLDMA, color = 'b',lw = 3, label ='HiGEAR-LDMA')
    # axs2.scatter(rUHSAS , DvDlnrUHSAS, color = '#843900', label ='UHSAS')
    axs2.scatter(rHW, DvDlnrHW, color = '#C70039',  label ='HiGEAR-HW')
   
    # axs2.errorbar(rHW, DvDlnrHW,  yerr= 0.1*DvDlnrHW, xerr=0.05*rHW,color = '#D8BB1B')
    # axs2.errorbar(rDM , DvDlnrDM,yerr= 0.1*DvDlnrDM, xerr=0.05*rDM, color = '#3EB3C7')
    # axs2.errorbar(r_APS , DvDlnr_APS,yerr= 0.1*DvDlnr_APS, xerr=0.05*r_APS,color = '#0D0084')
    # axs2.errorbar(rLDMA , DvDlnrLDMA,yerr= 0.1*DvDlnrLDMA, xerr=0.05*rLDMA, color = '#656564')

    # axs2.errorbar(rUHSAS , DvDlnrUHSAS,yerr= 0.1*DvDlnrUHSAS, xerr=0.05*rUHSAS,color = '#843900')

    axs2.set_xlim(7*10**-3,1)
    axs2.set_xlabel(r"Rv $\mu$m")
    axs2.set_ylabel(r"dV/dlnr")

    # plt.xscale('log')
    axs2.set_xscale("log")
    axs2.legend()


    plt.savefig(f'/home/gregmi/git/Results/FineMode{SavImgName}.png', dpi=200 )




def Interpolate(HSRLRslt, RSPwl, NoMode = None):

    if NoMode == None:
        NoMode = 3
    RSPn, RSPk = np.ones((NoMode,len(RSPwl))),np.ones((NoMode,len(RSPwl)))
    HSRLn,HSRLk, HSRLwl =HSRLRslt['n'],HSRLRslt['k'], HSRLRslt['lambda']

    for mode in range(len(HSRLn)): #loop for each mode
        
        fn = interp1d( HSRLwl,HSRLn[mode], kind='linear',fill_value="extrapolate")
        fk = interp1d( HSRLwl,HSRLk[mode], kind='linear',fill_value="extrapolate")
        RSPn[mode] = fn(RSPwl)
        RSPk[mode] = fk(RSPwl)

        plt.plot(HSRLwl, HSRLn[mode],'-',marker = 'o', label='HSRL')
        plt.plot(RSPwl,RSPn[mode], '-',marker = 'o', label='RSP')
        plt.legend()
        plt.show()


    return RSPn, RSPk


        
def UpdateRSPfromHSRL():


