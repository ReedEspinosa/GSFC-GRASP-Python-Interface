import sys
import warnings
from CreateRsltsDict import Read_Data_RSP_Oracles, Read_Data_HSRL_Oracles_Height
from ORACLES_GRASP import AeroProfNorm_sc2,FindPix, RSP_Run,  HSLR_run, LidarAndMAP, plot_HSRL,RSP_plot,CombinedLidarPolPlot,Ext2Vconc


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
from matplotlib.font_manager import FontProperties
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as mpatches
from matplotlib.ticker import ScalarFormatter
# from Plot_ORACLES import PltGRASPoutput, PlotRetrievals
import yaml
import pickle
# import datetimes
from netCDF4 import Dataset
%matplotlib inline



GasAbsFn = '/home/gregmi/ORACLES/UNL_VRTM/shortwave_gas.unlvrtm.nc'


def Combine_MAPandLidar_Rsltdict(rslt_RSP,rslt_HSRL, HSRLPixNo):

    """This funtion combines the rslt dict from Lidar and Polarimeter togetehr into a single rslt dictionary
        
        The rslt dict for RSP and HSRL are colocated.
        rslt_RSP = result dict for polarimeter
        rslt_HSRL = result dict fot Lidar
        TelNo  =  (Specific to Research Scanning Polarimeter) RSP has two telescopes, we will average the I and DoLP
    
    """
    
    rslt= {}  # Teh order of the data is  First lidar(number of wl ) and then Polarimter data 
    rslt['lambda'] = np.concatenate((rslt_HSRL['lambda'],rslt_RSP['lambda']))

    #Sort the index of the wavelength if arranged in ascending order, this is required by GRASP
    sort = np.argsort(rslt['lambda']) 
    sort_Lidar, sort_MAP  = np.array([0,3,7]),np.array([1,2,4,5,6])  #TODO Make this more general
    
    
# The shape of the variables in RSPkeys and HSRLkeys should be equal to no of wavelength
#  Setting np.nan in place of the measurements for wavelengths for which there is no data
    RSPkeys = ['meas_I', 'meas_P','sza', 'vis', 'sca_ang', 'fis']
    HSRLkeys = ['RangeLidar','meas_VExt','meas_VBS','meas_DP']
    GenKeys= ['datetime','longitude', 'latitude', 'land_prct'] # Shape of these variables is not N wavelength
    


    #MAP measurement variables 
    RSP_var = np.ones((rslt_RSP['meas_I'].shape[0],rslt['lambda'].shape[0])) * np.nan
    for keys in RSPkeys:
        #adding values to sort_MAP index positions
        for a in range(rslt_RSP[keys][:,0].shape[0]):
            RSP_var[a][sort_MAP] = rslt_RSP[keys][a]
        rslt[keys] = RSP_var
        RSP_var = np.ones((rslt_RSP['meas_I'].shape[0],rslt['lambda'].shape[0])) * np.nan

    #Lidar Measurements
    HSRL_var = np.ones((rslt_HSRL['meas_VExt'].shape[0],rslt['lambda'].shape[0]))* np.nan
    for keys1 in HSRLkeys:  
        for a in range(rslt_HSRL[keys1][:,0].shape[0]):
            
            HSRL_var[a][sort_Lidar] = rslt_HSRL[keys1][a]

            # 'sza', 'vis','fis'
        rslt[keys1] = HSRL_var
        # Refresh the array by Creating numpy nan array with shape of height x wl, Basically deleting all values
        HSRL_var = np.ones((rslt_HSRL['meas_VExt'].shape[0],rslt['lambda'].shape[0]))* np.nan

    
    for keys in GenKeys:
        rslt[keys] = rslt_RSP[keys]  #Adding the information about lat, lon, datetime and so on from RSP
    
    rslt['OBS_hght'] = rslt_RSP['OBS_hght'] #adding the aircraft altitude 
    rslt['lambda'] = rslt['lambda'][sort]

    #TODO improve this code to work when only mol deopl or gasabs are provided and not both
    if 'gaspar' in  rslt_HSRL:
        gasparHSRL = np.zeros(len(rslt['lambda']))
        gasparHSRL[sort_Lidar] = rslt_HSRL['gaspar']
        
        rslt['gaspar'] = gasparHSRL
        # print(rslt['gaspar'])

    if 'gaspar' in  rslt_RSP:
        gasparRSP = np.zeros(len(rslt['lambda']))
        gasparRSP[sort_MAP] = rslt_RSP['gaspar']

        rslt['gaspar'] = gasparRSP
        # print(rslt['gaspar'])

    if 'gaspar' in  rslt_RSP and rslt_HSRL :
        gasparB = np.zeros(len(rslt['lambda']))
        gasparB[sort_MAP] = rslt_RSP['gaspar']
        gasparB[sort_Lidar] = rslt_HSRL['gaspar']
        rslt['gaspar'] = gasparB
        # print(rslt['gaspar'])


    AeroProf = AeroProfNorm_sc2(rslt_HSRL1)
    
    # AeroProf =AeroProfNorm_FMF(rslt_HSRL_1)
    FineProfext,DstProfext,SeaProfext = AeroProf[0],AeroProf[1],AeroProf[2]



    rslt['VertProf_Mode1'] = AeroProf[0]  #Fine mode 
    rslt['VertProf_Mode2'] = AeroProf[1]
    rslt['VertProf_Mode3'] = AeroProf[2]


    return rslt






# # '''##Case 5: 22nd sept 2018, ORACLES
# # # # #RSP'''
file_path = "/home/gregmi/ORACLES/RSP1-L1C_P3_20180922_R03/"  #Path to the ORACLE data file
file_name =  "/RSP1-P3_L1C-RSPCOL-CollocatedRadiances_20180922T151106Z_V003-20210421T233946Z.h5" #Name of the ORACLES file
#HSRL
HSRLfile_path = "/home/gregmi/ORACLES/HSRL" #Path to the ORACLE data file 
HSRLfile_name =  "/HSRL2_P3_20180922_R2.h5" #Name of the ORACLES file
RSP_PixNo = 13201

nwl = 5 # first  nwl wavelengths
ang1 = 10
ang2 = 120 # :ang angles  #Remove
TelNo = 0 # aggregated altitude. To obtain geometries corresponding to data from the 1880 nm channel, aggregation altitude should be set to 1, while aggregation altitude =0 should be used for all other channels.

krnlPath='/home/shared/GRASP_GSFC/src/retrieval/internal_files'
binPathGRASP ='/home/shared/GRASP_GSFC/build_HEX_v112/bin/grasp_app'
fwdModelYAMLpath ='/home/gregmi/git/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/settings_BCK_LidarAndMAP_3modes_V.1.2_Simulate.yml'


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
HSRLPixNo = FindPix(LatH,LonH,LatRSP,LonRSP)[0]  # Or can manually give the index of the pixel that you are intrested in
rslt_HSRL = Read_Data_HSRL_Oracles_Height(HSRLfile_path,HSRLfile_name,HSRLPixNo)[0]
rslt_HSRL1 = Read_Data_HSRL_Oracles_Height(HSRLfile_path,HSRLfile_name,HSRLPixNo)
rslt_RSP = Read_Data_RSP_Oracles(file_path,file_name,RSP_PixNo,ang1,ang2,TelNo, nwl,GasAbsFn)


rslt =  Combine_MAPandLidar_Rsltdict(rslt_RSP,rslt_HSRL, HSRLPixNo)

FineSize = np.linspace(0.01, 0.1, 5)
# DustSize = np.linspace(0.1, 5, 15)[::-1]
# SphFrac = np.linspace(0.0001, 0.999, 3)

RRI = np.linspace(1.45, 1.6, 3)



ChangeVariable = FineSize
ChangeVariable2 = RRI





UpKerFile = 'settings_LIDARandPOLAR_3modes_Shape_Sph_Update.yml'
Full_dict= []


for Itr2 in range(len(ChangeVariable2)):
    DictRslt =[]



    for Itr in range(len(ChangeVariable)):
        #Updating the normalization values in the settings file. 
        with open(fwdModelYAMLpath, 'r') as f:  
            data = yaml.safe_load(f)

        for noMd in range(4): #loop over the aerosol modes (i.e 2 for fine and coarse)

            
                #State Varibles from yaml file: 
            if noMd ==1:
                data['retrieval']['constraints'][f'characteristic[1]'][f'mode[{noMd}]']['initial_guess']['value'] =  rslt['VertProf_Mode1'].tolist()
                data['retrieval']['constraints'][f'characteristic[3]'][f'mode[{noMd}]']['initial_guess']['min'][0] = float(0.9999*ChangeVariable[Itr])
                data['retrieval']['constraints'][f'characteristic[3]'][f'mode[{noMd}]']['initial_guess']['max'][0] = float(1.001*ChangeVariable[Itr])
                data['retrieval']['constraints'][f'characteristic[3]'][f'mode[{noMd}]']['initial_guess']['value'][0] = float(ChangeVariable[Itr])  #changing the value 
                
                #Changing the volume concentration to keep the AOD constant.
                data['retrieval']['constraints'][f'characteristic[2]'][f'mode[{noMd}]']['initial_guess']['min'][0] = float(0.9999*ConcForConstAOD[Itr2,Itr])
                data['retrieval']['constraints'][f'characteristic[2]'][f'mode[{noMd}]']['initial_guess']['max'][0] = float(1.001*ConcForConstAOD[Itr2,Itr])
                data['retrieval']['constraints'][f'characteristic[2]'][f'mode[{noMd}]']['initial_guess']['value'][0] = float(ConcForConstAOD[Itr2,Itr])
                
                # data['retrieval']['constraints'][f'characteristic[7]'][f'mode[{noMd}]']['initial_guess']['min'] = float(0.9999*SphFrac[Itr])
                # data['retrieval']['constraints'][f'characteristic[7]'][f'mode[{noMd}]']['initial_guess']['max'] = float(1.001*SphFrac[Itr])
                # data['retrieval']['constraints'][f'characteristic[7]'][f'mode[{noMd}]']['initial_guess']['value'] = float(SphFrac[Itr])  #changing the value 


                data['retrieval']['constraints'][f'characteristic[4]'][f'mode[{noMd}]']['initial_guess']['min'] = float(0.9999*ChangeVariable2[Itr2]),float(0.9999*ChangeVariable2[Itr2]),float(0.9999*ChangeVariable2[Itr2]),float(0.9999*ChangeVariable2[Itr2]),float(0.9999*ChangeVariable2[Itr2]),float(0.9999*ChangeVariable2[Itr2]),float(0.9999*ChangeVariable2[Itr2])
                data['retrieval']['constraints'][f'characteristic[4]'][f'mode[{noMd}]']['initial_guess']['max'] = float(1.001*ChangeVariable2[Itr2]),float(1.001*ChangeVariable2[Itr2]),float(1.001*ChangeVariable2[Itr2]),float(1.001*ChangeVariable2[Itr2]),float(1.001*ChangeVariable2[Itr2]),float(1.001*ChangeVariable2[Itr2]),float(1.001*ChangeVariable2[Itr2])
                data['retrieval']['constraints'][f'characteristic[4]'][f'mode[{noMd}]']['initial_guess']['value'] =float(ChangeVariable2[Itr2]),float(ChangeVariable2[Itr2]),float(ChangeVariable2[Itr2]),float(ChangeVariable2[Itr2]),float(ChangeVariable2[Itr2]),float(ChangeVariable2[Itr2]),float(ChangeVariable2[Itr2]),float(ChangeVariable2[Itr2])



            
            if noMd ==2:
                data['retrieval']['constraints'][f'characteristic[1]'][f'mode[{noMd}]']['initial_guess']['value'] =  rslt['VertProf_Mode2'].tolist()

            #     data['retrieval']['constraints'][f'characteristic[3]'][f'mode[{noMd}]']['initial_guess']['min'][0] = float(0.9999*ChangeVariable[Itr])
            #     data['retrieval']['constraints'][f'characteristic[3]'][f'mode[{noMd}]']['initial_guess']['max'][0] = float(1.001*ChangeVariable[Itr])
            #     data['retrieval']['constraints'][f'characteristic[3]'][f'mode[{noMd}]']['initial_guess']['value'][0] = float(ChangeVariable[Itr])  #changing the value 

            if noMd ==3:
                data['retrieval']['constraints'][f'characteristic[1]'][f'mode[{noMd}]']['initial_guess']['value'] = rslt['VertProf_Mode3'].tolist()


        ymlPath = '/home/gregmi/git/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/'


        with open(ymlPath+UpKerFile, 'w') as f2: #write the chnages to new yaml file
            yaml.safe_dump(data, f2)
            #     # print("Updating",YamlChar[i])
        Updatedyaml = ymlPath+UpKerFile

        pix = pixel()
        pix.populateFromRslt(rslt, radianceNoiseFun=None, dataStage= 'meas', verbose=False)
        gr = graspRun(pathYAML= Updatedyaml, releaseYAML=True, verbose=True) # setup instance of graspRun class, add the above pixel, run grasp and read the output to gr.invRslt[0] (dict)
        gr.addPix(pix) # add the pixel we created above
        gr.runGRASP(binPathGRASP=binPathGRASP, krnlPathGRASP=krnlPath) # run grasp binary (output read to gr.invRslt[0] [dict])
        print('AOD at %5.3f μm was %6.4f.' % (gr.invRslt[0]['lambda'][-1],gr.invRslt[0]['aod'][-1])) # 

        DictRslt.append(gr.invRslt[0])




    colors = [
        "#FFD700",  # Gold
        "#FFA500",  # Orange
        "#FF8C00",  # Dark Orange
        "#FF6347",  # Tomato
        "r",  # Orange Red

        
        "#00BFFF",  # Deep Sky Blue
        "#1E90FF",  # Dodger Blue
    
        "#5F9EA0",  # Cadet Blue
        "#00CED1",  # Dark Turquoise
        "#6495ED",  # Cornflower Blue
        "#1E3A8A",  # Deep Blue

        "#6A5ACD",  # Slate Blue
        "b",  # Medium Slate Blue
        
        "#8A2BE2",  # Blue Violet
        "m",  # Dark Magenta
    ]



    plt.rcParams['font.size'] = '17'

    sort_MAP = np.array([1, 2, 4, 5, 6])

    fig,ax = plt.subplots(2,5, figsize = (32,10), sharex = True)
    for Itr in range(len(ChangeVariable)):
        
        for i in range(5):
            if Itr == 0: ax[0,i].plot(DictRslt[Itr]['sca_ang'][:,sort_MAP[i]],DictRslt[Itr]['meas_I'][:,sort_MAP[i]], color ='k',lw = 3,marker = '.', label ='meas')
            
            ax[0,i].plot(DictRslt[Itr]['sca_ang'][:,sort_MAP[i]],DictRslt[Itr]['fit_I'][:,sort_MAP[i]], color = colors[Itr] ,label = np.round(ChangeVariable[Itr],3))
            # ax[0,i].scatter(DictRslt[Itr]['sca_ang'][:,sort_MAP[i]],DictRslt[Itr]['fit_I'][:,sort_MAP[i]], color = colors[Itr] ,label = np.round(ChangeVariable[Itr],3))

            if Itr == 0: ax[1,i].plot(DictRslt[Itr]['sca_ang'][:,sort_MAP[i]],DictRslt[Itr]['meas_P_rel'][:,sort_MAP[i]],color ='k',lw = 3,  label ='meas')
            
            ax[1,i].plot(DictRslt[Itr]['sca_ang'][:,sort_MAP[i]],DictRslt[Itr]['fit_P_rel'][:,sort_MAP[i]], color = colors[Itr], label = np.round(ChangeVariable[Itr],3))

        ax[0,0].set_ylabel("I")
        ax[1,0].set_ylabel("DoLP")

    for i in range(5):ax[0,i].set_title(f"{DictRslt[Itr]['lambda'][sort_MAP][i]} $\mu$m")
    for i in range(5):ax[1,i].set_xlabel(r"$\theta_{s}$")
    # plt.tight_layout()
    plt.subplots_adjust(right=0.7)

    # Create a common legend for all subplots, positioned to the right of the figure
    ax[0,4].legend(loc='center left', bbox_to_anchor=(1.11, -0.25), fancybox=True)





    sort_Lidar = np.array([0, 3, 7])
    fig,ax = plt.subplots(3,3, figsize = (18,18), sharey = True)
    for Itr in range(len(ChangeVariable)):
        
        for i in range(3):
            if Itr == 0: ax[0,i].plot(DictRslt[Itr]['meas_VExt'][:,sort_Lidar[i]]*1000,DictRslt[Itr]['range'][i,:],color ='k', lw = 3, label ='meas')
            
            ax[0,i].plot(DictRslt[Itr]['fit_VExt'][:,sort_Lidar[i]]*1000,DictRslt[Itr]['range'][i,:], color = colors[Itr], label =np.round(ChangeVariable[Itr],3))
            
            if Itr == 0: ax[1,i].plot(DictRslt[Itr]['meas_VBS'][:,sort_Lidar[i]],DictRslt[Itr]['range'][i,:],color ='k',lw = 3,  label ='meas')
            
            ax[1,i].plot(DictRslt[Itr]['fit_VBS'][:,sort_Lidar[i]],DictRslt[Itr]['range'][i,:], color = colors[Itr], label = np.round(ChangeVariable[Itr],3))

            if Itr == 0: ax[2,i].plot(DictRslt[Itr]['meas_DP'][:,sort_Lidar[i]],DictRslt[Itr]['range'][i,:],color ='k',lw = 3,  label ='meas')
            
            ax[2,i].plot(DictRslt[Itr]['fit_DP'][:,sort_Lidar[i]],DictRslt[Itr]['range'][i,:], color = colors[Itr], label =np.round(ChangeVariable[Itr],3))

    plt.subplots_adjust(right=0.8)

    # Create a common legend for all subplots, positioned to the right of the figure
    ax[0,2].legend(loc='center left', bbox_to_anchor=(1.2, -0.25), fancybox=True)

    ax[0,0].set_ylabel("Height")
    for i in range(2):ax[0,i].set_xlabel(r"$\alpha$")
    for i in range(3):ax[1,i].set_xlabel(r"$\beta$")
    for i in range(3):ax[2,i].set_xlabel(r"$\delta$ %")
    for i in range(3):ax[0,i].set_title(f"{DictRslt[Itr]['lambda'][sort_Lidar][i]} $\mu$m")






    fig,ax = plt.subplots(5,3, figsize = (12,12))
    for Itr in range(len(FineSize)):
        for i in range(3):
            # ax[i].plot(DictRslt[Itr]['sca_ang'][:,i],DictRslt[0]['meas_I'][:,i])
            ax[0,i].plot(DictRslt[Itr]['r'][i],DictRslt[Itr][ 'dVdlnr'][i], color = colors[Itr])
            ax[0,i].set_xscale('log')


            ax[1,i].plot(DictRslt[Itr]['lambda'],DictRslt[Itr][ 'n'][i], color = colors[Itr])
            ax[2,i].plot(DictRslt[Itr]['lambda'],DictRslt[Itr][ 'k'][i], color = colors[Itr])
            ax[3,i].plot(DictRslt[Itr]['lambda'],DictRslt[Itr][ 'ssaMode'][i], color = colors[Itr])
            ax[4,i].plot(DictRslt[Itr]['lambda'],DictRslt[Itr][ 'aodMode'][i], color = colors[Itr])

            
    plt.tight_layout()


    fig,ax = plt.subplots(1,4, figsize = (10,8))

    txt_title = 'n = '+ str(DictRslt[Itr]['n'])+ ' \n k = '+ str(DictRslt[Itr]['k'])+ r'\n $\sigma$ = '+ str(DictRslt[Itr]['sigma'])+ '\n r = '+ str(DictRslt[Itr]['rv'])+ '\n sph = '+ str(DictRslt[Itr][ 'sph'])
    for Itr in range(len(FineSize)):
        ax[0].plot(DictRslt[Itr]['r'][0],DictRslt[Itr][ 'dVdlnr'][0], color = colors[Itr])
        ax[1].plot(DictRslt[Itr]['lambda'],DictRslt[Itr][ 'ssaMode'][0], color = colors[Itr])
        ax[2].plot(DictRslt[Itr]['lambda'],DictRslt[Itr][ 'aodMode'][0], color = colors[Itr])
        ax[3].scatter(Itr, DictRslt[Itr][ 'costVal'])

        ax[0].set_xscale('log')



    plt.suptitle(txt_title)

            
    plt.tight_layout()

    Full_dict.append(DictRslt)
    print(Itr2)


ConcForConstAOD = np.ones((len(ChangeVariable2),len(ChangeVariable)))
checkrv = np.ones((len(ChangeVariable2),len(ChangeVariable)))

for j in range(len(ChangeVariable)):
    for i in range(len(ChangeVariable2)):
        MeasAOD = abs(np.trapz(Full_dict[i][j]['meas_VExt'][:,0], Full_dict[i][j]['range'][0,:]))
        print(Full_dict[i][j]['vol'][0]*Full_dict[i][j]['aodMode'][0][0]* MeasAOD)
        ConcForConstAOD[i,j] = Full_dict[i][j]['vol'][0]*Full_dict[i][j]['aodMode'][0][0]* MeasAOD
        checkrv[i,j] = Full_dict[i][j]['rv'][0]


























def Simulate_RSPandHSRL(Kernel_type,HSRLfile_path,HSRLfile_name,HSRLPixNo,file_path,file_name,RSP_PixNo,ang1,ang2,TelNo, nwl,GasAbsFn, ModeNo=None, updateYaml= None, RandinitGuess =None , NoItr=None,fnGuessRslt = None):
    
    """Kernel_type = "sphro" or "TAMU" for spehriod kernel or TAMU_sphericity_0.7 hexahedral

    """


    `failedmeas = 0 #Count of number of meas that failed

        #Path to the Kernel Files
        krnlPath='/home/shared/GRASP_GSFC/src/retrieval/internal_files'

        if Kernel_type == "sphro":  #If spheriod model
            #Path to the yaml file for sphriod model
            if ModeNo == None or ModeNo == 2:
                fwdModelYAMLpath = '/home/gregmi/git/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/settings_BCK_LidarAndMAP_V.1.2.yml'
            if ModeNo == 3:
                fwdModelYAMLpath ='/home/gregmi/git/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/settings_BCK_LidarAndMAP_3modes_V.1.2.yml'
            if updateYaml == True:  # True if init conditions for Yaml file for HSRL is updated from the GRASP output from RSP
                update_HSRLyaml(fwdModelYAMLpath, rslts_Sph[0], noMod, maxr, minr, a,Kernel_type)
                
                fwdModelYAMLpath ='/home/gregmi/git/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/Settings_Sphd_RSP_HSRL.yaml'
            binPathGRASP ='/home/shared/GRASP_GSFC/build_HEX_v112/bin/grasp_app'  #This will work for both the kernels as it has more parameter wettings
            savePath=f"/home/gregmi/ORACLES/HSRL1_P3_20180922_R03_{Kernel_type}"
            fnGuessRslt ='sphRandnew.npy'
        
        if Kernel_type == "TAMU": #Using Hexhderal shape kernel.
            if ModeNo == None or ModeNo == 2:
                fwdModelYAMLpath = '/home/gregmi/git/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/settings_BCK_LidarAndMAP_V.1.2_TAMU.yml'
            if ModeNo == 3:
                fwdModelYAMLpath ='/home/gregmi/git/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/settings_BCK_LidarAndMAP_3modes_V.1.2_TAMU.yml'
            if updateYaml == True:# True if init conditions for Yaml file for HSRL is updated from the GRASP output from RSP
                update_HSRLyaml(fwdModelYAMLpath, rslts_Tamu[0], noMod, maxr, minr, a,Kernel_type)
                fwdModelYAMLpath ='/home/gregmi/git/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/Settings_TAMU_RSP_HSRL.yaml'

            #Path to the GRASP Executable for TAMU
            # binPathGRASP ='/home/shared/GRASP_GSFC/build_HexV112_Noise/bin/grasp_app' #Recompiled to account for more noise parameter
            binPathGRASP ='/home/shared/GRASP_GSFC/build_HEX_v112/bin/grasp_app' #GRASP Executable
            # binPathGRASP ='/home/shared/GRASP_GSFC/build_HEX_v112/bin/grasp_app'
            #Path to save output plot
            savePath=f"/home/gregmi/ORACLES/RSP1-L1C_P3_20180922_R03_{Kernel_type}"
            fnGuessRslt ='HexRandnew.npy'


        #Updating the normalization values in the settings file. 
        with open(fwdModelYAMLpath, 'r') as f:  
            data = yaml.safe_load(f)

        for noMd in range(4): #loop over the aerosol modes (i.e 2 for fine and coarse)
            
                #State Varibles from yaml file: 
            if noMd ==1:
                data['retrieval']['constraints'][f'characteristic[1]'][f'mode[{noMd}]']['initial_guess']['value'] =  FineProfext.tolist()
            if noMd ==2:
                data['retrieval']['constraints'][f'characteristic[1]'][f'mode[{noMd}]']['initial_guess']['value'] =  DstProfext.tolist()
            if noMd ==3:
                data['retrieval']['constraints'][f'characteristic[1]'][f'mode[{noMd}]']['initial_guess']['value'] =  SeaProfext.tolist()
        
        if Kernel_type == "sphro":
            UpKerFile = 'settings_LIDARandPOLAR_3modes_Shape_Sph_Update.yml' #for spheroidal kernel
        if Kernel_type == "TAMU":
            UpKerFile = 'settings_LIDARandPOLAR_3modes_Shape_HEX_Update.yml'#for hexahedral kernel

        ymlPath = '/home/gregmi/git/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/'
        

        with open(ymlPath+UpKerFile, 'w') as f2: #write the chnages to new yaml file
            yaml.safe_dump(data, f2)
            #     # print("Updating",YamlChar[i])
        Updatedyaml = ymlPath+UpKerFile

        Finalrslts =[]

        


        maxCPU = 13 #maximum CPU allocated to run GRASP on server
        gRuns = []
        yamlObj = graspYAML(baseYAMLpath=Updatedyaml)

        

        #eventually have to adjust code for height, this works only for one pixel (single height value)
        gRuns.append(graspRun(pathYAML=yamlObj, releaseYAML= True )) # This should copy to new YAML object
        # rslt['meas_DP'][1,-1] = np.nan
        pix = pixel()
        pix.populateFromRslt(rslt, radianceNoiseFun=None, dataStage= 'meas', verbose=False)
        gRuns[-1].addPix(pix)
        gDB = graspDB(graspRunObjs=gRuns, maxCPU=maxCPU)
        #rslts contain all the results form the GRASP inverse run
        rslts, failPix = gDB.processData(binPathGRASP=binPathGRASP, savePath=None, krnlPathGRASP=krnlPath)
        
        Finalrslts.append(rslts)`

    


















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


#Path to the gas absorption (tau) values for gas absorption correction
GasAbsFn = '/home/gregmi/ORACLES/UNL_VRTM/shortwave_gas.unlvrtm.nc'



rslt_RSP = Read_Data_RSP_Oracles(file_path,file_name,RSP_PixNo,ang1,ang2,TelNo, nwl,GasAbsFn)
rslt_HSRL = Read_Data_HSRL_Oracles_Height(HSRLfile_path,HSRLfile_name,HSRLPixNo)






def RunFwdModel_LIDARandMAP(rslt_RSP):

    "Running forward model for Lidar and Map"



    Rslt_dicts = {}
    # sphericalFract = np.arange(1e-5, 0.99,0.01)
    "Designed to characterize the fine mode overestimation in Lidar and MAP"
    #Path to GRASP executables and Kernels

    krnlPath='/home/shared/GRASP_GSFC/src/retrieval/internal_files'
    binPathGRASP ='/home/shared/GRASP_GSFC/build_HEX_v112/bin/grasp_app'
    ymlPath = '/home/gregmi/git/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/'
    UpKerFile =  'settings_dust_Vext_conc_dump.yml'  #Yaml file after updating the state vectors
    fwdModelYAMLpath ='/home/gregmi/git/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/settings_dust_Vext_conc.yml'
    
   
#Geometry
    singProf = np.linspace(botLayer, topLayer, Nlayers)[::-1]
    #Inputs for simulation
    wvls = list(rslt_RSP['lambda']) # wavelengths in μm
    msTyp = [36] # meas type VEXT
    sza = 0.01 # we assume vertical lidar

    nbvm = Nlayers*np.ones(len(msTyp), int)
    thtv = np.tile(singProf, len(msTyp))
    meas = np.r_[np.repeat(2.372179e-05, nbvm[0])]
    phi = np.repeat(0, len(thtv)) # currently we assume all observations fall within a plane
# errStr = [y for y in archName.lower().split('+') if 'lidar09' in y][0]
    nowPix = pixel(dt.datetime.now(), 1, 1, 0, 0, masl=0, land_prct=0)
    
    for wvl in wvls: # This will be expanded for wavelength dependent measurement types/geometry
        # errModel = functools.partial(addError, errStr, concase=concase, orbit=orbit, lidErrDir=lidErrDir) # this must link to an error model in addError() below
        nowPix.addMeas(wvl, msTyp, nbvm, sza, thtv, phi, meas,)


    for i in range(len(Volume)):
        try:
#change the value of concentration in the yaml file
            with open(fwdModelYAMLpath, 'r') as f:  
                data = yaml.safe_load(f) 
                # print(data)
                data['retrieval']['constraints']['characteristic[2]']['mode[1]']['initial_guess']['value'][0] = float(Volume[i])
                data['retrieval']['constraints']['characteristic[3]']['mode[1]']['initial_guess']['value'][0],data['retrieval']['constraints']['characteristic[3]']['mode[1]']['initial_guess']['value'][1] = float(rv),float(sigma)
                data['retrieval']['constraints']['characteristic[4]']['mode[1]']['initial_guess']['value'][0] = float(n)
                data['retrieval']['constraints']['characteristic[5]']['mode[1]']['initial_guess']['value'][0] = float(k)
                data['retrieval']['constraints']['characteristic[6]']['mode[1]']['initial_guess']['value'][0] = float(sf)
                data['retrieval']['forward_model']['phase_matrix']['kernels_folder'] = Kernel

            f.close()
            
            with open(ymlPath+UpKerFile, 'w') as f: #write the chnages to new yaml file
                yaml.safe_dump(data, f)
            f.close()

#New yaml file after updating state vectors/variables
            Newyaml = ymlPath+UpKerFile
            #Running GRASP
            print('Using settings file at %s' % Newyaml)
            gr = graspRun(pathYAML= Newyaml, releaseYAML=True, verbose=True) # setup instance of graspRun class, add the above pixel, run grasp and read the output to gr.invRslt[0] (dict)
            gr.addPix(nowPix) # add the pixel we created above
            gr.runGRASP(binPathGRASP=binPathGRASP, krnlPathGRASP=krnlPath) # run grasp binary (output read to gr.invRslt[0] [dict])
            print('AOD at %5.3f μm was %6.4f.' % (gr.invRslt[0]['lambda'][-1],gr.invRslt[0]['aod'][-1])) # 

            #Saving Vext and Volume conc
            VConcandExt[i,0], VConcandExt[i,1] = gr.invRslt[0]['vol'],gr.invRslt[0]['fit_VExt'][1]
            
            #Saving the dictonary as pkl file
            dict_of_dicts[f'itr{i}'] = gr.invRslt[0]
            with open(f'/home/gregmi/ORACLES/Case2O/Vext2conc/Vext2conc_{i}_{Shape}.pickle', 'wb') as f:
                pickle.dump(gr.invRslt[0], f)
            f.close()

        except Exception as e:
            print(f"An error occurred for Conc = {Volume[i]}: {e}")
        continue
   
    # Save to file using pickle
    with open(f'/home/gregmi/ORACLES/Case2O/Vext2conc/Vext2conc_all_{Shape}.pickle', 'wb') as f:
        pickle.dump(dict_of_dicts, f)
    gr.invRslt[0]

