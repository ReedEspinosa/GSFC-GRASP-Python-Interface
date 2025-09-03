
import sys
from matplotlib import pyplot as plt
import numpy as np
import datetime as dt
from numpy import nanmean
import h5py 
sys.path.append("/data/home/gregmi/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases")
from architectureMap import returnPixel
from ORACLES_GRASP import FindPix, RSP_Run,  HSLR_run, LidarAndMAP, plot_HSRL,RSP_plot,CombinedLidarPolPlot,Ext2Vconc, RSP_Run_General
from ORACLES_GRASP import AeroProfNorm_sc2,AeroClassAOD

import yaml
# %matplotlib inline
from runGRASP import graspRun, pixel
import math

from netCDF4 import Dataset
import matplotlib.patches as mpatches
from matplotlib.ticker import ScalarFormatter
from scipy.interpolate import interp1d
import re
from itertools import cycle


from CreateRsltsDict import Read_Data_RSP_Oracles
from CreateRsltsDict import Read_Data_HSRL_Oracles_Height
import netCDF4 as nc
from runGRASP import graspDB, graspRun, pixel, graspYAML

import os
import pickle


UNCERT ={}
UNCERT['aodMode'] = 0
UNCERT['aod']= 0
UNCERT['ssaMode'] = 0
UNCERT['k'] = 0
UNCERT['n'] = 0
UNCERT['DP'] = 1
UNCERT['VBS'] =  2e-7     #2.4e-6        #2e-7 # abs %
UNCERT['VEXT'] = 1e-05  #rel% 10%
UNCERT['rv'] = 0
UNCERT['I'] = 0.03
UNCERT['DoLP'] = 0.005



# #Path to RSP filename
# file_path = "/data/home/gregmi/Data/ORACLES/RSP/" 
# file_name = 'RSP1-P3_L1C-RSPCOL-CollocatedRadiances_20180924T090316Z_V003-20210421T233034Z.h5'
# RSP_Start_PixNo = 382
# TelNo = 0 # aggregated altitude. To obtain geometries corresponding to data from the 1880 nm channel, aggregation altitude should be set to 1, while aggregation altitude =0 should be used for all other channels.
# nwl = 5 # first  nwl wavelengths
# ang1 = 10
# ang2 = 135


# #Path to HSRL data
# HSRLfile_path = "/data/home/gregmi/Data/ORACLES/HSRL2" #Path to the ORACLE data file 
# HSRLfile_name =  '/HSRL2_P3_20180924_R2.h5'




def UpdateKrnlYaml(YamlPath, UpdatedYamlPath, KrnlName, PrintAddInfo = False): 
        
        #Updates the Kernel name in the yaml file. 
        
        #Reading the yml file to change the kernel
        with open(YamlPath, 'r') as f:  
            data = yaml.safe_load(f) 
            data['retrieval']['forward_model']['particles']['phase_matrix']['kernels_folder'] = KrnlName  # KrnlName should be string
            data['retrieval']['general']['path_to_internal_files'] = UpdatedYamlPath
        
            f.close()
            UpdatedYaml = UpdatedYamlPath+"settings_BCK_MAP_3modes_Hex_kernelChange.yml"
            with open(UpdatedYaml, 'w') as f2:
                yaml.safe_dump(data, f2)
            f2.close()
            print(f"Kernel successfully updated : {UpdatedYaml}")

            if PrintAddInfo == True:
                 
                 # Set it to true to print the name of the kernel
                 #O pen the updated yaml file and print the name of the kernel from the file
                 with open(UpdatedYaml, 'r') as f3:  
                    data2 =  yaml.safe_load(f3) 
                    KrnlName  = data2['retrieval']['forward_model']['particles']['phase_matrix']['kernels_folder']
                    f3.close()

                    print(f"The kernel name has been changed to  {KrnlName}")

        return UpdatedYaml


def runRSP(RSP_file_path,RSP_file_name, RSP_PixNo, ang1,ang2,TelNo, RSPwlIdx,GasAbsFn,SpecResFnPath, binPathGRASP, krnlPath, fwdModelYAMLpath):

    maxCPU = 10 #maximum CPU allocated to run GRASP on server
    gRuns = []

    #rslt is the GRASP rslt dictionary or contains GRASP Objects
    rslt = Read_Data_RSP_Oracles(RSP_file_path, RSP_file_name,RSP_PixNo,ang1,ang2,TelNo, RSPwlIdx,GasAbsFn,SpecResFnPath)
    yamlObj = graspYAML(baseYAMLpath=fwdModelYAMLpath)
    #eventually have to adjust code for height, this works only for one pixel (single height value)
    gRuns.append(graspRun(pathYAML=yamlObj, releaseYAML=True )) # This should copy to new YAML object
    pix = pixel()
    pix.populateFromRslt(rslt, radianceNoiseFun=None, dataStage='meas', verbose=False)
    gRuns[-1].addPix(pix)
    gDB = graspDB(graspRunObjs=gRuns, maxCPU=maxCPU)

    #rslts contain all the results form the GRASP inverse run
    rslts, failPix = gDB.processData(binPathGRASP=binPathGRASP, savePath=None, krnlPathGRASP=krnlPath)
    return rslts

       
def RSP_plot_all(Rslt_dict, RSP_PixNo, UNCERT, fn=None, LIDARPOL=False):
    """
    Overlay retrieval results from all entries in Rslt_dict
    on the same plots, using gradient colors.
    """

    # Gradient colormap instead of fixed list
    cmap = plt.get_cmap("magma")
    n = len(Rslt_dict)

    linestyle = [':', '-', '-.', '--']

    # --------------------------
    # 1) RETRIEVAL PARAMETER PLOTS
    # --------------------------
    fig, axs2 = plt.subplots(nrows=3, ncols=2, figsize=(35, 20))
    plt.rcParams['font.size'] = '26'
    plt.rcParams["figure.autolayout"] = True

    Retrival = ['dVdlnr','aodMode','ssaMode','n', 'k']

    for key_idx, (key, rslts) in enumerate(Rslt_dict.items()):
        Spheriod = rslts[0]
        color = cmap(key_idx / (n-1))  # gradient color

        # modes
        mode_v = ["fine", "dust", "marine"] if Spheriod['r'].shape[0] == 3 else ["fine", "dust"]
        lambda_ticks = np.round(Spheriod['lambda'], decimals=2)
        lambda_ticks_str = [str(x) for x in lambda_ticks]

        # Loop through retrieval quantities
        for i, ret in enumerate(Retrival):
            a, b = i % 3, i % 2
            for mode in range(Spheriod['r'].shape[0]):
                if ret == 'dVdlnr':
                    axs2[a,b].plot(
                        Spheriod['r'][mode], Spheriod[ret][mode],
                        marker="$O$", color=color,
                        lw=2, ls=linestyle[mode], label=f"{key}_{mode_v[mode]}"
                    )
                    axs2[a,b].set_xlabel(r'rv $ \mu m$')
                    axs2[a,b].set_xscale("log")
                else:
                    axs2[a,b].errorbar(
                        lambda_ticks_str, Spheriod[ret][mode], 
                        yerr=UNCERT[ret],
                        marker="$O$", markeredgecolor='k', capsize=7,
                        capthick=2, markersize=15,
                        color=color, 
                        lw=3, ls=linestyle[mode], 
                        label=f"{key}_{mode_v[mode]}"
                    )
                    axs2[a,b].set_xticks(lambda_ticks_str)
                    axs2[a,b].set_xlabel(r'$\lambda \mu m$')
            axs2[a,b].set_ylabel(ret)

        # Plot total AOD
        axs2[2,1].errorbar(
            lambda_ticks_str, Spheriod['aod'],
            yerr=0.03 + UNCERT['aod'] * Spheriod['aod'],
            marker="$O$", markeredgecolor='k', capsize=7,
            capthick=2, markersize=15,
            color=color,
            lw=3, label=f"{key}_AOD"
        )

    axs2[2,1].set_xlabel(r'$\lambda$')
    axs2[2,1].set_ylabel('Total AOD')
    axs2[0,0].legend(prop={"size": 18}, ncol=2)

    # metadata from first retrieval
    first = list(Rslt_dict.values())[0][0]
    lat_t, lon_t, dt_t = first['latitude'], first['longitude'], first['datetime']
    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.suptitle(f'RSP Aerosol Retrievals \n Lat:{lat_t} Lon:{lon_t} Date:{dt_t}')

    out_fn = fn if fn else f"{dt_t}_AllRetrievals.png"
    fig.savefig(out_fn, dpi=300)

    # --------------------------
    # 2) STOKES PLOTS (all retrievals overlayed)
    # --------------------------
    wl = first['lambda']
    fig, axs = plt.subplots(
        nrows=4, ncols=len(wl),
        figsize=(len(wl)*7, 17),
        gridspec_kw={'height_ratios': [1, 0.3, 1, 0.3]},
        sharex='col'
    )

    wlIdx = [1,2,4,5,6] if LIDARPOL else np.arange(len(wl))

    for key_idx, (key, rslts) in enumerate(Rslt_dict.items()):
        Spheriod = rslts[0]
        color = cmap(key_idx / (n-1))  # gradient color

        for nwav in range(len(wlIdx)):
            idx = wlIdx[nwav]

            # I
            axs[0, nwav].plot(Spheriod['sca_ang'][:,idx], Spheriod['meas_I'][:,idx], 
                              color="k", lw=3, label="meas" if key_idx==0 else None)
            axs[0, nwav].plot(Spheriod['sca_ang'][:,idx], Spheriod['fit_I'][:,idx],
                              color=color, lw=3, ls='--', label=f"{key}_fit")
            axs[0, nwav].set_ylabel('I')

            # I error
            sphErr = 100 * abs(Spheriod['meas_I'][:,idx]-Spheriod['fit_I'][:,idx]) / Spheriod['meas_I'][:,idx]
            axs[1, nwav].plot(Spheriod['sca_ang'][:,idx], sphErr,
                              color=color, lw=3, label=f"{key}")
            axs[1,0].set_ylabel('Err I %')

            # DOLP
            axs[2, nwav].plot(Spheriod['sca_ang'][:,idx], Spheriod['meas_P_rel'][:,idx], 
                              color="k", lw=3, label="meas" if key_idx==0 else None)
            axs[2, nwav].plot(Spheriod['sca_ang'][:,idx], Spheriod['fit_P_rel'][:,idx],
                              color=color, lw=3, ls='--', label=f"{key}_fit")
            axs[2,0].set_ylabel('DOLP')

            # DOLP error
            sphErrP = abs(Spheriod['meas_P_rel'][:,idx] - Spheriod['fit_P_rel'][:,idx])
            axs[3, nwav].plot(Spheriod['sca_ang'][:,idx], sphErrP,
                              color=color, lw=3, label=f"{key}")
            axs[3,0].set_ylabel('|Meas-fit|')
            axs[3,nwav].set_xlabel(r'$\theta$ (deg)')

            axs[0, nwav].set_title(f"{wl[idx]} $\mu m$", fontsize=22)

    axs[0, len(wlIdx)-1].legend(bbox_to_anchor=(1.05, 0.5), loc='center left')
    plt.suptitle(f'RSP Stokes Fits \n Lat:{lat_t} Lon:{lon_t} Date:{dt_t}')

    out_fn2 = fn if fn else f"{dt_t}_AllRetrievals_Stokes.png"
    fig.savefig(out_fn2, dpi=300)

    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.show()


def runHSRL(HSRLfile_path,HSRLfile_name, HSRLPixNo,ModeNo, binPathGRASP, krnlPath, fwdModelYAMLpath,releaseYAML =True ):

    maxCPU = 10 #maximum CPU allocated to run GRASP on server
    gRuns = []

    #rslt is the GRASP rslt dictionary or contains GRASP Objects
    DictHsrl = Read_Data_HSRL_Oracles_Height(HSRLfile_path,HSRLfile_name,HSRLPixNo,gaspar =True)
    rslt = DictHsrl[0]


    with open(fwdModelYAMLpath, 'r') as f:  
            data = yaml.safe_load(f)
        
            AeroProf = AeroProfNorm_sc2(DictHsrl) #vertical profiles of Dust, marine and fine contribution to the total aod caculated based on the dust mixing ratio and fine mode fraction.
            AerClassAOD = AeroClassAOD(DictHsrl)
    
            FineProfext,DstProfext,SeaProfext = AeroProf[0],AeroProf[1],AeroProf[2]
    

            # apriori = np.ones(len(FineProfext.tolist()))*1e-2
         
            for noMd in range(ModeNo+1): #loop over the aerosol modes (i.e 2 for fine and coarse)

                # MinVal = np.repeat(np.minimum(np.minimum(FineProf[FineProf>1.175494e-38],DstProf),SeaProf),10).tolist() # the size of the list will be adjusted later in the code
             # Updating the vertical profile norm values in yaml file: 
                if noMd ==1:  #Mode 1
                    data['retrieval']['constraints'][f'characteristic[1]'][f'mode[{noMd}]']['initial_guess']['value'] =  FineProfext.tolist()
                    
                if noMd ==2: #Mode 2 
                    data['retrieval']['constraints'][f'characteristic[1]'][f'mode[{noMd}]']['initial_guess']['value'] =  DstProfext.tolist()
        
                if noMd ==3: 
                    data['retrieval']['constraints'][f'characteristic[1]'][f'mode[{noMd}]']['initial_guess']['value'] =  SeaProfext.tolist()
                    
        
            UpKerFile = 'settings_BCK_POLAR_3modes_Shape_Sph_Update.yml' #for spheroidal kernel
            ymlPath = '/data/home/gregmi/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/GRASPV2/'

            with open(ymlPath+UpKerFile, 'w') as f2: #write the chnages to new yaml file
                yaml.safe_dump(data, f2)


    fwdModelYAMLpath2 = ymlPath+UpKerFile
    yamlObj = graspYAML(baseYAMLpath= fwdModelYAMLpath2)

    gRuns.append(graspRun(pathYAML=yamlObj, releaseYAML= releaseYAML)) # This should copy to new YAML object
    pix = pixel()
    pix.populateFromRslt(rslt, radianceNoiseFun=None, dataStage= 'meas', verbose=False)
    gRuns[-1].addPix(pix)
    gDB = graspDB(graspRunObjs=gRuns, maxCPU=maxCPU)
    #rslts contain all the results form the GRASP inverse run
    rslts, failPix = gDB.processData(binPathGRASP=binPathGRASP, savePath=None, krnlPathGRASP=krnlPath)
        
    return rslts, AerClassAOD 


def runRSPandHSRL(krnlPath,fwdModelYAMLpath,binPathGRASP,HSRLfile_path,HSRLfile_name,HSRLPixNo ,RSP_file_path,RSP_file_name,RSP_PixNo,ang1,ang2,TelNo, RSPwlIdx,GasAbsFn,SpecResFnPath,NoDP =None, RandinitGuess=None):

    rslt_HSRL_1 = Read_Data_HSRL_Oracles_Height(HSRLfile_path,HSRLfile_name,HSRLPixNo)
    rslt_HSRL =  rslt_HSRL_1[0]
    
    rslt_RSP = Read_Data_RSP_Oracles(RSP_file_path,RSP_file_name,RSP_PixNo,ang1,ang2,TelNo, RSPwlIdx,GasAbsFn,SpecResFnPath)
    
    rslt= {}  # Teh order of the data is  First lidar(number of wl ) and then Polarimter data 
    rslt['lambda'] = np.concatenate((rslt_HSRL['lambda'],rslt_RSP['lambda']))

    sort = np.argsort(rslt['lambda']) 
    sort_Lidar, sort_MAP  = np.array([0,3,7]),np.array([1,2,4,5,6])
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

    # rslt['OBS_hght'] =  12000   #Delete this immediately!!!!!!
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



    AeroProf = AeroProfNorm_sc2(rslt_HSRL_1)
    
    # AeroProf =AeroProfNorm_FMF(rslt_HSRL_1)
    FineProfext,DstProfext,SeaProfext = AeroProf[0],AeroProf[1],AeroProf[2]

    # plt.savefig(f'/home/gregmi/ORACLES/HSRL_RSP/DMRHSRL2Retrieval.png', dpi = 400)
    
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
    
    
    UpKerFile = 'settings_LIDARandPOLAR_3modes_Shape_DiffSphecity_Update.yml' #for spheroidal kernel
    ymlPath = '/data/home/gregmi/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/GRASPV2/'
    

    with open(ymlPath+UpKerFile, 'w') as f2: #write the chnages to new yaml file
        yaml.safe_dump(data, f2)
  
    Updatedyaml = ymlPath+UpKerFile

    Finalrslts =[]

    if NoDP ==True: 
        del rslt['meas_DP']

    if RandinitGuess == True:
        
        maxCPU = 3
        if NoItr==None: NoItr=1
        gRuns = [[]]*NoItr
        pix = pixel()
        pix.populateFromRslt(rslt, radianceNoiseFun=None, dataStage= 'meas', verbose=False)
        dblist = []
        
        for i in range(NoItr):
            try:
                # print(i)
                gyaml = graspYAML(baseYAMLpath=fwdModelYAMLpath)
                yamlObjscramble = gyaml.scrambleInitialGuess(fracOfSpace=1, skipTypes=['vertical_profile_normalized','aerosol_concentration','surface_water_cox_munk_iso'])
                yamlObj = graspYAML(baseYAMLpath=fwdModelYAMLpath)
                gRuns = []
                gRuns.append(graspRun(pathYAML=yamlObj, releaseYAML= True )) # This should copy to new YAML object
                gRuns[-1].addPix(pix)
                gDB = graspDB(graspRunObjs=gRuns, maxCPU=maxCPU,maxT =1)
                dblist.append(gDB)#rslts contain all the results form the GRASP inverse run
                rslts, failPix = gDB.processData(binPathGRASP=binPathGRASP, savePath=None, krnlPathGRASP=krnlPath)

                Finalrslts.append(rslts)
                # if len(rslts) >0:
                # plot_HSRL(rslts[0],rslts[0], forward = True, retrieval = True, Createpdf = True,PdfName =f"/home/gregmi/ORACLES/rsltPdf/HSRL_Only_Plots_{i}.pdf", combinedVal =rslts[2]) 
                plot_HSRL(rslts[0],rslts[0], forward = False, retrieval =True, Createpdf = True,PdfName =f"/home/gregmi/ORACLES/rsltPdf/LIDARPOL_Plots_{i}.pdf")
                RSP_plot(rslts,rslts,RSP_PixNo,LIDARPOL= True,fn = f'Random{i}')


                                
                # nc_file = nc.Dataset('/home/gregmi/git/GSFC-GRASP-Python-Interface/IntialGuessHSRLRSP.nc', 'w', format='NETCDF4')
                # nc_file.close()
                
            except:
                failedmeas +=1
                pass

    else:
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
        
        Finalrslts.append(rslts)

    if fnGuessRslt == None:fnGuessRslt = 'try.npy'
        
    np.save(fnGuessRslt, Finalrslts)

    return rslts,Finalrslts,failedmeas,gRuns



                        
'''##Case 5: 22nd sept 2018, ORACLES. RSP'''

RSP_file_path = "/data/home/gregmi/Data/ORACLES/RSP/"  #Path to the ORACLE data file
RSP_file_name =  "RSP1-P3_L1C-RSPCOL-CollocatedRadiances_20180922T151106Z_V003-20210421T233946Z.h5" #Name of the ORACLES file
RSP_PixNo = 13201
TelNo = 0 # aggregated altitude. To obtain geometries corresponding to data from the 1880 nm channel, aggregation altitude should be set to 1, while aggregation altitude =0 should be used for all other channels.
nwl = [0,1,2,3,4,6,8] # first  nwl wavelengths
ang1 = 10
ang2 = 120 # :ang angles  #Remove

#HSRL
HSRLfile_path = "/data/home/gregmi/Data/ORACLES/HSRL2" #Path to the ORACLE data file 
HSRLfile_name =  "/HSRL2_P3_20180922_R2.h5" #Name of the ORACLES file

#Path to the gas absorption (tau) values for gas absorption correction
GasAbsFn = '/data/home/gregmi/Data/GasAbs/shortwave_gas.unlvrtm.nc'
SpecResFnPath = '/data/home/gregmi/Data/ORACLES/RSP/RSP_Spectral_Response/'
fwdModelYAMLpath = '/data/home/gregmi/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/GRASPV2/settings_BCK_MAP_3modes_SphrodShape_ORACLES.yml'

noMod =3  #number of aerosol mode, here 2 for fine+coarse mode configuration
AprioriLagrange = [5e-2,1e-2,5e-1,1e-1,1]


krnlPath = '/data/ESI/User/gregmi/GRASP_V2/grasp-dev-rtm-v120-new-inter/src/retrieval/internal_files/Hex_all_sph_kernels'
internalPath =  '/data/ESI/User/gregmi/GRASP_V2/grasp-dev-rtm-v120-new-inter/src/retrieval/internal_files/'
YamlPath = '/data/home/gregmi/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/GRASPV2/settings_BCK_MAP_3modes_Sphd_ORACLES.yml' #Original yaml file (ymal before updting the kernel name)
binPathGRASP = '/data/home/gregmi/GRASP_V2/grasp-dev-rtm-v120-new-inter/bin/grasp_app'
UpdatedYamlPath = '/data/home/gregmi/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/GRASPV2/'


HSRL_YamlPath = '/data/home/gregmi/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/GRASPV2/settings_BCK_LIDAR_3modes_Hex_ORACLE.yml'
RSPandHSRLyamlPath = '/data/home/gregmi/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/GRASPV2/settings_BCK_LidarAndMAP_3modes_sphd_GV2.yml'


nwlIdx = [0,1,2,3,4,6,8] #Index of the waveleg\nghts to be used in the retrieval


RSP_PixNo = RSP_PixNo +1
RSPwlIdx = nwlIdx[:5] # first  nwl wavelengths

f1_MAP = h5py.File(RSP_file_path+RSP_file_name,'r+')   
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
ModeNo =3


# HexKrnlNames = [f for f in os.listdir(krnlPath) if os.path.isdir(os.path.join(krnlPath, f))]

HexKrnlNames = ['0.695', '0.704', '0.713', '0.722', '0.731','0.74', '0.749', '0.758', '0.767', '0.776', '0.785']

HSRL_Rslt_dict ={}

for i, KrnlName in enumerate(HexKrnlNames): 
    fwdModelYAMLpath = UpdateKrnlYaml(HSRL_YamlPath, UpdatedYamlPath, KrnlName, PrintAddInfo = False)
    rslts_hex = runHSRL(HSRLfile_path,HSRLfile_name, HSRLPixNo,ModeNo, binPathGRASP, krnlPath, fwdModelYAMLpath, releaseYAML =True )
    HSRL_Rslt_dict[f'{KrnlName}'] = rslts_hex[0]



    with open(f'/data/home/gregmi/GSFC-GRASP-Python-Interface/PhD_SQ_2/rslt/HSRL_Rslt{i}.pickle', 'wb') as f:
        pickle.dump( HSRL_Rslt_dict[f'{KrnlName}'],f)
        f.close()
    



RSP_Rslt_dict ={}
for i, KrnlName in enumerate(HexKrnlNames): 
     
    fwdModelYAMLpath = UpdateKrnlYaml(YamlPath, UpdatedYamlPath, KrnlName, PrintAddInfo = False)
    rslts_hex = runRSP(RSP_file_path,RSP_file_name, RSP_PixNo, ang1,ang2,TelNo, RSPwlIdx,GasAbsFn,SpecResFnPath, binPathGRASP,krnlPath, fwdModelYAMLpath)
    RSP_Rslt_dict[f'{KrnlName}'] = rslts_hex
    with open(f'/data/home/gregmi/GSFC-GRASP-Python-Interface/PhD_SQ_2/rslt/Rslt{i}.pickle', 'wb') as f:
        pickle.dump( RSP_Rslt_dict[f'{KrnlName}'],f)
        f.close()



RSPandHSRL_Rslt_dict ={}

for i, KrnlName in enumerate(HexKrnlNames): 
    fwdModelYAMLpath = UpdateKrnlYaml(RSPandHSRLyamlPath, UpdatedYamlPath, KrnlName, PrintAddInfo = False)
    rslts_hex = runRSPandHSRL(krnlPath,fwdModelYAMLpath,binPathGRASP,HSRLfile_path,HSRLfile_name,HSRLPixNo ,RSP_file_path,RSP_file_name,RSP_PixNo,ang1,ang2,TelNo, RSPwlIdx,GasAbsFn,SpecResFnPath,NoDP =None, RandinitGuess=None)
    RSPandHSRL_Rslt_dict[f'{KrnlName}'] = rslts_hex[0]



    with open(f'/data/home/gregmi/GSFC-GRASP-Python-Interface/PhD_SQ_2/rslt/RSPandHSRL_Rslt{i}.pickle', 'wb') as f:
        pickle.dump( RSPandHSRL_Rslt_dict[f'{KrnlName}'],f)
        f.close()


       
def RSP_plot_all(Rslt_dict, RSP_PixNo, UNCERT, fn=None, LIDARPOL=False):
    """
    Overlay retrieval results from all entries in Rslt_dict
    on the same plots, using gradient colors.
    """

    # Gradient colormap instead of fixed list
    cmap = plt.get_cmap("magma")
    n = len(Rslt_dict)

    linestyle = [':', '-', '-.', '--']

    # --------------------------
    # 1) RETRIEVAL PARAMETER PLOTS
    # --------------------------
    fig, axs2 = plt.subplots(nrows=3, ncols=2, figsize=(35, 20))
    plt.rcParams['font.size'] = '26'
    plt.rcParams["figure.autolayout"] = True

    Retrival = ['dVdlnr','aodMode','ssaMode','n', 'k']

    for key_idx, (key, rslts) in enumerate(Rslt_dict.items()):
        Spheriod = rslts[0]
        color = cmap(key_idx / (n-1))  # gradient color

        # modes
        mode_v = ["fine", "dust", "marine"] if Spheriod['r'].shape[0] == 3 else ["fine", "dust"]
        lambda_ticks = np.round(Spheriod['lambda'], decimals=2)
        lambda_ticks_str = [str(x) for x in lambda_ticks]

        # Loop through retrieval quantities
        for i, ret in enumerate(Retrival):
            a, b = i % 3, i % 2
            for mode in range(Spheriod['r'].shape[0]):
                if ret == 'dVdlnr':
                    axs2[a,b].plot(
                        Spheriod['r'][mode], Spheriod[ret][mode],
                        marker="$O$", color=color,
                        lw=2, ls=linestyle[mode], label=f"{key}_{mode_v[mode]}"
                    )
                    axs2[a,b].set_xlabel(r'rv $ \mu m$')
                    axs2[a,b].set_xscale("log")
                else:
                    axs2[a,b].errorbar(
                        lambda_ticks_str, Spheriod[ret][mode], 
                        yerr=UNCERT[ret],
                        marker="$O$", markeredgecolor='k', capsize=7,
                        capthick=2, markersize=15,
                        color=color, 
                        lw=3, ls=linestyle[mode], 
                        label=f"{key}_{mode_v[mode]}"
                    )
                    axs2[a,b].set_xticks(lambda_ticks_str)
                    axs2[a,b].set_xlabel(r'$\lambda \mu m$')
            axs2[a,b].set_ylabel(ret)

        # Plot total AOD
        axs2[2,1].errorbar(
            lambda_ticks_str, Spheriod['aod'],
            yerr=0.03 + UNCERT['aod'] * Spheriod['aod'],
            marker="$O$", markeredgecolor='k', capsize=7,
            capthick=2, markersize=15,
            color=color,
            lw=3, label=f"{key}_AOD"
        )

    axs2[2,1].set_xlabel(r'$\lambda$')
    axs2[2,1].set_ylabel('Total AOD')
    axs2[0,0].legend(prop={"size": 18}, ncol=2)

    # metadata from first retrieval
    first = list(Rslt_dict.values())[0][0]
    lat_t, lon_t, dt_t = first['latitude'], first['longitude'], first['datetime']
    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.suptitle(f'RSP Aerosol Retrievals \n Lat:{lat_t} Lon:{lon_t} Date:{dt_t}')

    out_fn = fn if fn else f"{dt_t}_AllRetrievals.png"
    fig.savefig(out_fn, dpi=300)

    # --------------------------
    # 2) STOKES PLOTS (all retrievals overlayed)
    # --------------------------
    wl = first['lambda']
    fig, axs = plt.subplots(
        nrows=4, ncols=len(wl),
        figsize=(len(wl)*7, 17),
        gridspec_kw={'height_ratios': [1, 0.3, 1, 0.3]},
        sharex='col'
    )

    wlIdx = [1,2,4,5,6] if LIDARPOL else np.arange(len(wl))

    for key_idx, (key, rslts) in enumerate(Rslt_dict.items()):
        Spheriod = rslts[0]
        color = cmap(key_idx / (n-1))  # gradient color

        for nwav in range(len(wlIdx)):
            idx = wlIdx[nwav]

            # I
            axs[0, nwav].plot(Spheriod['sca_ang'][:,idx], Spheriod['meas_I'][:,idx], 
                              color="k", lw=3, label="meas" if key_idx==0 else None)
            axs[0, nwav].plot(Spheriod['sca_ang'][:,idx], Spheriod['fit_I'][:,idx],
                              color=color, lw=3, ls='--', label=f"{key}_fit")
            axs[0, nwav].set_ylabel('I')

            # I error
            sphErr = 100 * abs(Spheriod['meas_I'][:,idx]-Spheriod['fit_I'][:,idx]) / Spheriod['meas_I'][:,idx]
            axs[1, nwav].plot(Spheriod['sca_ang'][:,idx], sphErr,
                              color=color, lw=3, label=f"{key}")
            axs[1,0].set_ylabel('Err I %')

            # DOLP
            axs[2, nwav].plot(Spheriod['sca_ang'][:,idx], Spheriod['meas_P_rel'][:,idx], 
                              color="k", lw=3, label="meas" if key_idx==0 else None)
            axs[2, nwav].plot(Spheriod['sca_ang'][:,idx], Spheriod['fit_P_rel'][:,idx],
                              color=color, lw=3, ls='--', label=f"{key}_fit")
            axs[2,0].set_ylabel('DOLP')

            # DOLP error
            sphErrP = abs(Spheriod['meas_P_rel'][:,idx] - Spheriod['fit_P_rel'][:,idx])
            axs[3, nwav].plot(Spheriod['sca_ang'][:,idx], sphErrP,
                              color=color, lw=3, label=f"{key}")
            axs[3,0].set_ylabel('|Meas-fit|')
            axs[3,nwav].set_xlabel(r'$\theta$ (deg)')

            axs[0, nwav].set_title(f"{wl[idx]} $\mu m$", fontsize=22)

    axs[0, len(wlIdx)-1].legend(bbox_to_anchor=(1.05, 0.5), loc='center left')
    plt.suptitle(f'RSP Stokes Fits \n Lat:{lat_t} Lon:{lon_t} Date:{dt_t}')

    out_fn2 = fn if fn else f"{dt_t}_AllRetrievals_Stokes.png"
    fig.savefig(out_fn2, dpi=300)

    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.show()

RSP_plot_all(RSP_Rslt_dict, RSP_PixNo, UNCERT, fn=None, LIDARPOL=False)


       
def HSRL_plot_all(Rslt_dict, RSP_PixNo, UNCERT, fn=None, LIDARPOL=False):
    """
    Overlay retrieval results from all entries in Rslt_dict
    on the same plots, using gradient colors.
    """

    # Gradient colormap instead of fixed list
    cmap = plt.get_cmap("magma")
    n = len(Rslt_dict)

    linestyle = [':', '-', '-.', '--']

    # --------------------------
    # 1) RETRIEVAL PARAMETER PLOTS
    # --------------------------
    fig, axs2 = plt.subplots(nrows=3, ncols=2, figsize=(35, 20))
    plt.rcParams['font.size'] = '26'
    plt.rcParams["figure.autolayout"] = True

    Retrival = ['dVdlnr','aodMode','ssaMode','n', 'k']

    for key_idx, (key, rslts) in enumerate(Rslt_dict.items()):
        Spheriod = rslts[0]
        color = cmap(key_idx / (n-1))  # gradient color

        # modes
        mode_v = ["fine", "dust", "marine"] 
        lambda_ticks = np.round([0.355,0.532,1.064], decimals=2)
        lambda_ticks_str = [str(x) for x in lambda_ticks]

        # Loop through retrieval quantities
        for i, ret in enumerate(Retrival):
            a, b = i % 3, i % 2
            for mode in range(len( mode_v )):
                if ret == 'dVdlnr':
                    axs2[a,b].plot(
                        Spheriod['r'][mode], Spheriod[ret][mode],
                        marker="$O$", color=color,
                        lw=2, ls=linestyle[mode], label=f"{key}_{mode_v[mode]}"
                    )
                    axs2[a,b].set_xlabel(r'rv $ \mu m$')
                    axs2[a,b].set_xscale("log")
                else:
                    axs2[a,b].errorbar(
                        lambda_ticks_str, Spheriod[ret][mode], 
                        yerr=UNCERT[ret],
                        marker="$O$", markeredgecolor='k', capsize=7,
                        capthick=2, markersize=15,
                        color=color, 
                        lw=3, ls=linestyle[mode], 
                        label=f"{key}_{mode_v[mode]}"
                    )
                    axs2[a,b].set_xticks(lambda_ticks_str)
                    axs2[a,b].set_xlabel(r'$\lambda \mu m$')
            axs2[a,b].set_ylabel(ret)

        # Plot total AOD
        axs2[2,1].errorbar(
            lambda_ticks_str, Spheriod['aod'],
            yerr=0.03 + UNCERT['aod'] * Spheriod['aod'],
            marker="$O$", markeredgecolor='k', capsize=7,
            capthick=2, markersize=15,
            color=color,
            lw=3, label=f"{key}_AOD"
        )

    axs2[2,1].set_xlabel(r'$\lambda$')
    axs2[2,1].set_ylabel('Total AOD')
    axs2[0,0].legend(prop={"size": 18}, ncol=2)

    # metadata from first retrieval
    first = list(Rslt_dict.values())[0][0]
    lat_t, lon_t, dt_t = first['latitude'], first['longitude'], first['datetime']
    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.suptitle(f'RSP Aerosol Retrievals \n Lat:{lat_t} Lon:{lon_t} Date:{dt_t}')

    out_fn = fn if fn else f"{dt_t}_AllRetrievals.png"
    fig.savefig(out_fn, dpi=300)


def plot_HSRL_profiles(Rslt_dict, UNCERT, fn_prefix="HSRL_Profile"):

    wl = ['355','532','1064']

    fig, axs = plt.subplots(1, len(wl), figsize=(5*len(wl), 6))
    fig, axs2 = plt.subplots(1, len(wl), figsize=(5*len(wl), 6))
    fig, axs3 = plt.subplots(1, len(wl), figsize=(5*len(wl), 6))

    """
    Plot vertical profiles (VBS, DP, VExt) for each retrieval in Rslt_dict.
    Parameters:
        Rslt_dict : dict
            Retrieval results keyed by retrieval case/label.
        UNCERT : dict
            Dictionary with uncertainties: 'VBS', 'DP', 'VEXT'.
        fn_prefix : str
            Prefix for output PNG filenames.
    """
    cmap = plt.get_cmap("magma")
    n = len(Rslt_dict)

    for key_idx, (key, rslts) in enumerate(Rslt_dict.items()):
        sph = rslts[0]
        color = cmap(key_idx / (n-1))
        altd = sph['RangeLidar'][:, 0] / 1000  # km
        wl = sph['lambda']

        # --- Backscatter ---
        
        for i, ax in enumerate(np.atleast_1d(axs)):
            ax.errorbar(sph['meas_VBS'][:, i], altd, xerr=UNCERT['VBS'],
                        color="#695E93", capsize=3, alpha=0.4,)
            ax.plot(sph['meas_VBS'][:, i], altd, ">", color="#281C2D")
            ax.scatter(sph['fit_VBS'][:, i], altd,marker=  "$O$", color=color, label=f"{key}")
            ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
            ax.set_xlabel(r'$VBS\ (m^{-1}sr^{-1})$')
            ax.set_title(f"{wl[i]:.2f} μm")
        axs[0].set_ylabel("Height (km)")
        axs[0].legend()
        plt.suptitle(f"HSRL Vertical Backscatter Profile\n Lat:{sph['latitude']} Lon:{sph['longitude']} Date:{sph['datetime']}")
        plt.tight_layout()
        fig.savefig(f"{fn_prefix}_VBS_{key}.png", dpi=300)
       

        # --- Depolarization ---
        
        
        for i in range(len(wl)):
            axs2[i].errorbar(sph['meas_DP'][:, i], altd, xerr=UNCERT['DP'],
                        color="#695E93", capsize=4, alpha=0.6)
            axs2[i].plot(sph['meas_DP'][:, i], altd, ">", color="#281C2D")
            axs2[i].scatter(sph['fit_DP'][:, i], altd,  marker=  "$O$", color=color, label=f"{key}")
            axs2[i].set_xlabel("DP (%)")
            axs2[i].set_title(f"{wl[i]:.2f} μm")
        axs2[0].set_ylabel("Height (km)")
        axs2[0].legend()
        plt.suptitle(f"HSRL Depolarization Ratio\n Lat:{sph['latitude']} Lon:{sph['longitude']} Date:{sph['datetime']}")
        plt.tight_layout()
        fig.savefig(f"{fn_prefix}_DP_{key}.png", dpi=300)
 

        # --- Extinction ---
        
        for i in range(len(wl)):

            axs3[i].errorbar(sph['meas_VExt'][:, i], altd, 
                        xerr=UNCERT['VEXT'] * sph['meas_VExt'][:, i],
                        color="#695E93", capsize=4, alpha=0.7)
            axs3[i].plot(sph['meas_VExt'][:, i], altd, ">", color="#281C2D")
            axs3[i].scatter(sph['fit_VExt'][:, i], altd, marker=  "$O$", color=color, label=f"{key}")
            axs3[i].xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
            axs3[i].set_xlabel(r'$VExt\ (m^{-1})$')
            axs3[i].set_title(f"{wl[i]:.2f} μm")
        axs3[0].set_ylabel("Height (km)")
        axs3[0].legend()
        plt.suptitle(f"HSRL Vertical Extinction Profile\n Lat:{sph['latitude']} Lon:{sph['longitude']} Date:{sph['datetime']}")
        plt.tight_layout()
        fig.savefig(f"{fn_prefix}_VExt_{key}.png", dpi=300)
   
# plot_HSRL_profiles(Rslt_dict, UNCERT, fn_prefix="HSRL_Profile")

# #Degree of sphericity
# for i in range(1):

#     RSP_PixNo = RSP_PixNo +1
#     TelNo = 0 # aggregated altitude. To obtain geometries corresponding to data from the 1880 nm channel, aggregation altitude should be set to 1, while aggregation altitude =0 should be used for all other channels.
#     RSPwlIdx = nwlIdx[:i+5] # first  nwl wavelengths

#     f1_MAP = h5py.File(RSP_file_path+RSP_file_name,'r+')   
#     Data = f1_MAP['Data']
#     LatRSP = f1_MAP['Geometry']['Collocated_Latitude'][TelNo,RSP_PixNo]
#     LonRSP = f1_MAP['Geometry']['Collocated_Longitude'][TelNo,RSP_PixNo]
#     f1_MAP.close()
#     rslts_Tamu = RSP_Run("TAMU",RSP_file_path,RSP_file_name,RSP_PixNo,ang1,ang2,TelNo,RSPwlIdx,GasAbsFn,SpecResFnPath,ModeNo=3)



# rslts_Sph = runRSP(RSP_file_path,RSP_file_name, RSP_PixNo, ang1,ang2,TelNo, RSPwlIdx,GasAbsFn,SpecResFnPath, binPathGRASP,krnlPath, fwdModelYAMLpath)




       
def HSRL_plot_all(Rslt_dict, RSP_PixNo, UNCERT, fn=None, LIDARPOL=False):
    """
    Overlay retrieval results from all entries in Rslt_dict
    on the same plots, using gradient colors.
    """

    # Gradient colormap instead of fixed list
    cmap = plt.get_cmap("magma")
    n = len(Rslt_dict)

    linestyle = [':', '-', '-.', '--']

    # --------------------------
    # 1) RETRIEVAL PARAMETER PLOTS
    # --------------------------
    fig, axs2 = plt.subplots(nrows=3, ncols=2, figsize=(35, 20))
    plt.rcParams['font.size'] = '26'
    plt.rcParams["figure.autolayout"] = True

    Retrival = ['dVdlnr','aodMode','ssaMode','n', 'k']

    for key_idx, (key, rslts) in enumerate(Rslt_dict.items()):
        Spheriod = rslts[0]
        color = cmap(key_idx / (n-1))  # gradient color

        # modes
        mode_v = ["fine", "dust", "marine"] 
        lambda_ticks = np.round([0.355,0.532,1.064], decimals=2)
        lambda_ticks_str = [str(x) for x in lambda_ticks]

        # Loop through retrieval quantities
        for i, ret in enumerate(Retrival):
            a, b = i % 3, i % 2
            for mode in range(len( mode_v )):
                if ret == 'dVdlnr':
                    axs2[a,b].plot(
                        Spheriod['r'][mode], Spheriod[ret][mode],
                        marker="$O$", color=color,
                        lw=2, ls=linestyle[mode], label=f"{key}_{mode_v[mode]}"
                    )
                    axs2[a,b].set_xlabel(r'rv $ \mu m$')
                    axs2[a,b].set_xscale("log")
                else:
                    axs2[a,b].errorbar(
                        lambda_ticks_str, Spheriod[ret][mode], 
                        yerr=UNCERT[ret],
                        marker="$O$", markeredgecolor='k', capsize=7,
                        capthick=2, markersize=15,
                        color=color, 
                        lw=3, ls=linestyle[mode], 
                        label=f"{key}_{mode_v[mode]}"
                    )
                    axs2[a,b].set_xticks(lambda_ticks_str)
                    axs2[a,b].set_xlabel(r'$\lambda \mu m$')
            axs2[a,b].set_ylabel(ret)

        # Plot total AOD
        axs2[2,1].errorbar(
            lambda_ticks_str, Spheriod['aod'],
            yerr=0.03 + UNCERT['aod'] * Spheriod['aod'],
            marker="$O$", markeredgecolor='k', capsize=7,
            capthick=2, markersize=15,
            color=color,
            lw=3, label=f"{key}_AOD"
        )

    axs2[2,1].set_xlabel(r'$\lambda$')
    axs2[2,1].set_ylabel('Total AOD')
    axs2[0,0].legend(prop={"size": 18}, ncol=2)

    # metadata from first retrieval
    first = list(Rslt_dict.values())[0][0]
    lat_t, lon_t, dt_t = first['latitude'], first['longitude'], first['datetime']
    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.suptitle(f'HSRL Aerosol Retrievals \n Lat:{lat_t} Lon:{lon_t} Date:{dt_t}')

    out_fn = fn if fn else f"{dt_t}_HSRL_AllRetrievals.png"
    fig.savefig(out_fn, dpi=300)




def plot_HSRL_profiles(Rslt_dict, UNCERT, fn_prefix="HSRL_Profile"):

    plt.rcParams['font.size'] = '20'

    wl = ['355','532','1064']

    fig, axs = plt.subplots(1, len(wl), figsize=(5*len(wl), 6))
    fig, axs2 = plt.subplots(1, len(wl), figsize=(5*len(wl), 6))
    fig, axs3 = plt.subplots(1, len(wl), figsize=(5*len(wl), 6))

    """
    Plot vertical profiles (VBS, DP, VExt) for each retrieval in Rslt_dict.
    Parameters:
        Rslt_dict : dict
            Retrieval results keyed by retrieval case/label.
        UNCERT : dict
            Dictionary with uncertainties: 'VBS', 'DP', 'VEXT'.
        fn_prefix : str
            Prefix for output PNG filenames.
    """
    cmap = plt.get_cmap("magma")
    n = len(Rslt_dict)


    sorted_dict = {k: Rslt_dict[k] for k in sorted(Rslt_dict.keys(), key=float)}



    for key_idx, (key, rslts) in enumerate(sorted_dict.items()):
        sph = rslts[0]
        color = cmap(key_idx / (n-1))
        altd = sph['RangeLidar'][:, 0] / 1000  # km
        wl = sph['lambda']

        # --- Backscatter ---
        
        for i, ax in enumerate(np.atleast_1d(axs)):
            ax.errorbar(sph['meas_VBS'][:, i], altd, xerr=UNCERT['VBS'],
                        color="#695E93", capsize=3, alpha=0.4,)
            ax.plot(sph['meas_VBS'][:, i], altd, ">", color="#281C2D")
            ax.scatter(sph['fit_VBS'][:, i], altd,marker=  "$O$", color=color, label=f"{key}")
            ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
            ax.set_xlabel(r'$VBS\ (m^{-1}sr^{-1})$')
            ax.set_title(f"{wl[i]:.2f} μm")
        axs[0].set_ylabel("Height (km)")
        # axs[0].legend()
     
        plt.tight_layout()
        # fig.savefig(f"{fn_prefix}_VBS_{key}.png", dpi=300)
       

        # --- Depolarization ---
        
        
        for i in range(len(wl)):
            axs2[i].errorbar(sph['meas_DP'][:, i], altd, xerr=UNCERT['DP'],
                        color="#695E93", capsize=4, alpha=0.6)
            axs2[i].plot(sph['meas_DP'][:, i], altd, ">", color="#281C2D")
            axs2[i].scatter(sph['fit_DP'][:, i], altd,  marker=  "$O$", color=color, label=f"{key}")
            axs2[i].set_xlabel("DP (%)")
            axs2[i].set_title(f"{wl[i]:.2f} μm")
        axs2[0].set_ylabel("Height (km)")
        axs2[2].legend(fontsize=12)
      
        plt.tight_layout()
        # fig.savefig(f"{fn_prefix}_DP_{key}.png", dpi=300)
 

        # --- Extinction ---
        
        for i in range(len(wl)):

            axs3[i].errorbar(sph['meas_VExt'][:, i], altd, 
                        xerr=UNCERT['VEXT'] * sph['meas_VExt'][:, i],
                        color="#695E93", capsize=4, alpha=0.7)
            axs3[i].plot(sph['meas_VExt'][:, i], altd, ">", color="#281C2D")
            axs3[i].scatter(sph['fit_VExt'][:, i], altd, marker=  "$O$", color=color, label=f"{key}")
            axs3[i].set_xlabel(r'$VExt\ (m^{-1})$')
            axs3[i].set_title(f"{wl[i]:.2f} μm")
        axs3[0].set_ylabel("Height (km)")

        axs3[2].legend(fontsize=12)
      
        plt.tight_layout()
    fig.savefig(f"{fn_prefix}_VExt_{key}.png", dpi=300)
   
plot_HSRL_profiles(HSRL_Rslt_dict, UNCERT, fn_prefix="HSRL_Profile")


       
def HSRL_plot_all(Rslt_dict, RSP_PixNo, UNCERT, fn=None, LIDARPOL=False,No_Mode =1):
    """
    Overlay retrieval results from all entries in Rslt_dict
    on the same plots, using gradient colors.
    """

    # Gradient colormap instead of fixed list
    cmap = plt.get_cmap("magma")
    n = len(Rslt_dict)

    linestyle = ['-', '-', '-', '--']

    # --------------------------
    # 1) RETRIEVAL PARAMETER PLOTS
    # --------------------------
    fig, axs2 = plt.subplots(nrows=3, ncols=2, figsize=(35, 20))
    plt.rcParams['font.size'] = '26'
    plt.rcParams["figure.autolayout"] = True

    Retrival = ['dVdlnr','aodMode','ssaMode','n', 'k']

    for key_idx, (key, rslts) in enumerate(Rslt_dict.items()):
        Spheriod = rslts[0]
        color = cmap(key_idx / (n-1))  # gradient color

        # modes
        mode_vall = ["fine", "dust", "marine"] 

        mode_v =mode_vall[No_Mode-1]

        lambda_ticks = np.round([0.355,0.532,1.064], decimals=2)
        lambda_ticks_str = [str(x) for x in lambda_ticks]

        # Loop through retrieval quantities
        for i, ret in enumerate(Retrival):
            a, b = i % 3, i % 2
            mode = No_Mode-1
            if ret == 'dVdlnr':
                axs2[a,b].plot(
                    Spheriod['r'][mode], Spheriod[ret][mode],
                    marker="$O$", color=color,
                    lw=2, ls=linestyle[mode], label=f"{key}"
                )
                axs2[a,b].set_xlabel(r'rv $ \mu m$')
                axs2[a,b].set_xscale("log")
            else:
                axs2[a,b].errorbar(
                    lambda_ticks_str, Spheriod[ret][mode], 
                    yerr=UNCERT[ret],
                    marker="$O$", markeredgecolor='k', capsize=7,
                    capthick=2, markersize=15,
                    color=color, 
                    lw=3, ls=linestyle[mode], 
                    label=f"{key}_{mode_v[mode]}"
                )
                axs2[a,b].set_xticks(lambda_ticks_str)
                axs2[a,b].set_xlabel(r'$\lambda \mu m$')
        axs2[a,b].set_ylabel(ret)

        # Plot total AOD
        axs2[2,1].errorbar(
            lambda_ticks_str, Spheriod['aod'],
            yerr=0.03 + UNCERT['aod'] * Spheriod['aod'],
            marker="$O$", markeredgecolor='k', capsize=7,
            capthick=2, markersize=15,
            color=color,
            lw=3, label=f"{key}_AOD"
        )

    axs2[2,1].set_xlabel(r'$\lambda$')
    axs2[2,1].set_ylabel('Total AOD')
    axs2[0,0].legend(prop={"size": 18}, ncol=2)

    # metadata from first retrieval
    first = list(Rslt_dict.values())[0][0]
    lat_t, lon_t, dt_t = first['latitude'], first['longitude'], first['datetime']
    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.suptitle(f'HSRL Aerosol Retrievals \n Lat:{lat_t} Lon:{lon_t} Date:{dt_t}')

    out_fn = fn if fn else f"{dt_t}_HSRL_AllRetrievals.png"
    fig.savefig(out_fn, dpi=300)

HSRL_plot_all(HSRL_Rslt_dict, RSP_PixNo, UNCERT, fn=None, LIDARPOL=False,No_Mode =1)