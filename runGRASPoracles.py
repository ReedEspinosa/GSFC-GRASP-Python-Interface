

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

import matplotlib.patches as mpatches
from matplotlib.ticker import ScalarFormatter



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



# # # ###Case 4: 24th Sept 2018, ORACLES
# file_path = '/home/gregmi/ORACLES/Sept24/'
# file_name ='RSP1-P3_L1C-RSPCOL-CollocatedRadiances_20180924T090316Z_V003-20210421T233034Z.h5'
# HSRLfile_path = '/home/gregmi/ORACLES/Sept24/'
# HSRLfile_name =  "HSRL2_P3_20180924_R2.h5"
# RSP_PixNo = 380
# TelNo = 0 # aggregated altitude. To obtain geometries corresponding to data from the 1880 nm channel, aggregation altitude should be set to 1, while aggregation altitude =0 should be used for all other channels.
# nwl = 5 # first  nwl wavelengths
# ang1 = 10
# ang2 = 135

    

# '''##Case 5: 22nd sept 2018, ORACLES
# # #RSP'''
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
UNCERT ={}
UNCERT['aodMode'] = 0.1
UNCERT['aod']= 0.1
UNCERT['ssaMode'] = 0.03
UNCERT['k'] = 0.003
UNCERT['n'] = 0.025
UNCERT['DP'] = 1
UNCERT['VBS'] =  2.4e-6        #2e-7 # rel %
UNCERT['VEXT'] = 0.1   #rel% 10%
UNCERT['rv'] = 0.05
UNCERT['I'] = 0.03
UNCERT['DoLP'] = 0.005


#Path to the gas absorption (tau) values for gas absorption correction
GasAbsFn = '/home/gregmi/ORACLES/UNL_VRTM/shortwave_gas.unlvrtm.nc'
SpecResFnPath = '/home/gregmi/ORACLES/RSP_Spectral_Response/'
#This is required if we want to configure the HSRL yaml file based on the GRASP output for the RSP
noMod =3  #number of aerosol mode, here 2 for fine+coarse mode configuration
 #no of char # this is a varible added to char and modes to avoid char[0]/ mod[0] which doesnt exist
AprioriLagrange = [5e-2,1e-2,5e-1,1e-1,1]

SphLagrangian =[]
TAMULagrangian =[]


for i in range(1):
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
    HSRLPixNo = FindPix(LatH,LonH,LatRSP,LonRSP)[0]  # Or can manually give the index of the pixel that you are intrested in



    #RSP only retrieval
 
# #  Kernel_type = Run(Kernel_type) for spheriod, Kernel_type = 'TAMU' for hexahedral
    rslts_Sph = RSP_Run("sphro",file_path,file_name,RSP_PixNo,ang1,ang2,TelNo,nwl,GasAbsFn,ModeNo=3)
#     # rslts_Sph2 = RSP_Run("sphro",file_path,file_name,RSP_PixNo,ang1,ang2,TelNo,nwl,GasAbsFn,ModeNo=2)
    rslts_Tamu = RSP_Run("TAMU",file_path,file_name,RSP_PixNo,ang1,ang2,TelNo,nwl,GasAbsFn,ModeNo=3)
#     # # rslts_Tamu2 = RSP_Run("TAMU",file_path,file_name,RSP_PixNo,ang1,ang2,TelNo,nwl,GasAbsFn,ModeNo=2)
    RSP_plot(rslts_Sph,rslts_Tamu,RSP_PixNo,UNCERT)
   


   #HSRL2 only retrieval
    
    HSRL_sphrodT = HSLR_run("sphro",HSRLfile_path,HSRLfile_name,HSRLPixNo,nwl,ModeNo=3, updateYaml= False,releaseYAML= True, VertProfConstrain = True,LagrangianOnly = True,  AprioriLagrange =  AprioriLagrange[i])
    # plot_HSRL(HSRL_sphrodT[0][0],HSRL_sphrodT[0][0], UNCERT,forward = True, retrieval = True, Createpdf = True,PdfName ="/home/gregmi/ORACLES/rsltPdf/HSRL_Only_Plots_444.pdf", combinedVal =HSRL_sphrodT[2]) 
    HSRL_TamuT = HSLR_run("TAMU",HSRLfile_path,HSRLfile_name, HSRLPixNo,nwl,ModeNo=3,updateYaml= False,releaseYAML= True, VertProfConstrain = True,LagrangianOnly = True,  AprioriLagrange =  AprioriLagrange[i])
    # plot_HSRL(HSRL_Tamu[0][0],HSRL_Tamu[0][0], forward = True, retrieval = True, Createpdf = True,PdfName ="/home/gregmi/ORACLES/rsltPdf/HSRL_Only_Plots_444.pdf", combinedVal =HSRL_Tamu[2])    
    plot_HSRL(HSRL_sphrodT[0][0],HSRL_TamuT[0][0],UNCERT, forward = True, retrieval = True, Createpdf = True,PdfName ="/home/gregmi/ORACLES/rsltPdf/HSRL_Only_Plots_444.pdf", combinedVal =HSRL_TamuT[2])


   # joint RSP + HSRL2  retrieval 
  # # # PlotRandomGuess('gregmi/git/GSFC-GRASP-Python-Interface/try.npy', 2,0)
    LidarPolSph = LidarAndMAP('sphro',HSRLfile_path,HSRLfile_name,HSRLPixNo,file_path,file_name,RSP_PixNo,ang1,ang2,TelNo, nwl,GasAbsFn,ModeNo=3, updateYaml= None)
    LidarPolTAMU = LidarAndMAP('TAMU',HSRLfile_path,HSRLfile_name,HSRLPixNo,file_path,file_name,RSP_PixNo,ang1,ang2,TelNo, nwl,GasAbsFn,ModeNo=3, updateYaml= None)
    
    

    CombinedLidarPolPlot(LidarPolSph[0],LidarPolTAMU[0],RSP_PixNo,UNCERT)
    plot_HSRL(LidarPolSph[0][0],LidarPolTAMU[0][0],UNCERT, forward = True, retrieval = True, Createpdf = True,PdfName ="/home/gregmi/ORACLES/rsltPdf/LIDARPOL_Plots_444.pdf")
    RSP_plot(LidarPolTAMU[0],LidarPolTAMU[0],RSP_PixNo,UNCERT,LIDARPOL=True)







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

def PlotSensitivity(SphLagrangian,TAMULagrangian):
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
                # axs1[i].plot(Hsph['fit_VBS'][:,Index1[i]],altd, marker = "$O$",label =f"{AprioriLagrange[j]}",alpha =0.8)
                axs1[i].plot(HTam['fit_VBS'][:,Index2[i]],altd,ls = "--", label=f"{AprioriLagrange[j]}", marker = "h")

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

                # axs2[i].plot(Hsph['fit_DP'][:,Index1[i]],altd, marker = "$O$",label =f"{AprioriLagrange[j]}")
                axs2[i].plot(HTam['fit_DP'][:,Index2[i]],altd, ls = "--",marker = "h",label=f"{AprioriLagrange[j]}")
                
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

                # axs[i].plot(Hsph['fit_VExt'][:,Index1[i]],altd, marker = "$O$",label =f"{AprioriLagrange[j]}")
                axs[i].plot(HTam['fit_VExt'][:,Index2[i]],altd,ls = "--", marker = "h",label=f"{AprioriLagrange[j]}")
                
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
                axs1[i].plot(Hsph['fit_VBS'][:,Index1[i]],altd, marker = "$O$",label =f"{AprioriLagrange[j]}",alpha =0.8)
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



def PlotcombEachMode(rslts_Sph2,rslts_Tamu2,HSRL_sphrodT,HSRL_TamuT,LidarPolSph=None,LidarPolTAMU=None):
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


    # plot_HSRL(LidarPolSph[0][0],LidarPolTAMU[0][0],UNCERT, forward = False, retrieval = True, Createpdf = True,PdfName ="/home/gregmi/ORACLES/rsltPdf/LIDARPOL_Plots_444.pdf")
    
    if (LidarPolTAMU!=None )and (LidarPolSph!=None):  #When plotting joint retrieval with individual retrieval
        RepMode =2
        # Spheriod = LidarPolSph[0][0]
        # Hex =LidarPolTAMU[0][0]

    else:
        RepMode =1
    
    
    Spheriod = PlotSingle(rslts_Sph,HSRL_sphrodT)
    Hex = PlotSingle(rslts_Tamu2,HSRL_TamuT)



    
    cm_sp = ['k','#8B4000', '#87C1FF']
    cm_t = ["#BC106F",'#E35335', 'b']
    color_sph = '#0c7683'
    color_tamu = "#BC106F"

    fig, axs2 = plt.subplots(nrows= 9, ncols=2, figsize=(25, 45))
    fig, axsErr = plt.subplots(nrows= 9, ncols=2, figsize=(10, 15))
        

    # Plot the AOD data
    y = [0,1,2,0,1,2,]
    x = np.repeat((0,1),3)
    

    # markerSPh = ["$*$","$O$","$O$","$*$","$O$","$O$","$O$","$*$"]
    # markerHex = np.repeat("H", len(Spheriod['lambda']))
    # markerHex[0],markerHex[3],markerHex[7] = "</_\>","</_\>","</_\>"

    

    lambda_ticks = np.round(Spheriod['lambda'], decimals=2)
    lambda_ticks_str = [str(x) for x in lambda_ticks]

    NoMode =Spheriod['MapR'].shape[0]
    if Spheriod['MapR'].shape[0] ==2 :
            mode_v = ["Fine", "Dust","Marine"]
    if Spheriod['MapR'].shape[0] ==3 :
        mode_v = ["Fine", "Dust","Marine"]
    linestyle =[':', '-','-.']



    for RetNo in range(RepMode):

        if  RetNo ==1: 
            Spheriod,Hex = LidarPolSph[0][0],LidarPolTAMU[0][0]


            cm_sp = ['#757565','#5F381A', '#4BCBE2']
            cm_t =  ["#882255",'#D44B15', '#1E346D']


            # cm_sp2 = ['b','#BFBF2A', '#844772']
            # cm_t2 = ["#14411b",'#936ecf', '#FFA500']

           


        if  RetNo ==0:

            cm_sp = ['#4459AA','#14411b', '#87C1FF']
            cm_t = ["#14411b",'#adbf4b', '#9BB54C']


            cm_sp2 = ['b','#BFBF2A', '#844772']
            cm_t2 = ["#14411b",'#936ecf', '#FFA500']


            Spheriod,Hex = PlotSingle(rslts_Sph,HSRL_sphrodT),PlotSingle(rslts_Tamu2,HSRL_TamuT)
            # cm_sp = ['#b54cb5','#14411b', '#87C1FF']
            # cm_t = ["#14411b",'#adbf4b', 'b']
            color_sph = '#adbf4b'
            color_tamu = "#936ecf"

        


        a =-1
        
        #Retrivals:
        for i in range(len(Retrival)):
            b =i%2
            if a ==8: a=-1
            # a = i%3
            for mode in range(NoMode): #for each modes
                
                a = a+1
                
                if  RetNo ==0: 
                    ErrMap = Spheriod[RetrivalMAP[i]][mode] - Hex[RetrivalMAP[i]][mode]
                    ErrLidar = Spheriod[RetrivalLidar[i]][mode] - Hex[RetrivalLidar[i]][mode]
                    keyInd = 'Single'
                
                if  RetNo ==1:
                    ErrMap = Spheriod[Retrival[i]][mode] - Hex[Retrival[i]][mode]
                    ErrLidar = Spheriod[Retrival[i]][mode] - Hex[Retrival[i]][mode]
                    keyInd = 'Joint'

                if i ==0 and RetNo ==0: #Plotting single sensor retrievals


                    
                    cm_sp2 = ['b','#14411b', '#14411b']
                    cm_t2 = ["#14411b",'#936ecf', '#FFA500']
                    color_sph2 = '#adbf4b'
                    color_tamu2 = "#936ecf"

                  
                    axs2[a,b].plot(Spheriod['MapR'][mode], Spheriod['MapVol'][mode], marker = "$O$",color = cm_sp[mode],lw = 5,ls = linestyle[mode], label=f"RSP_Sphd")
                    axs2[a,b].plot(Hex['MapR'][mode],Hex['MapVol'][mode], marker = "H", color=cm_t[mode] ,lw = 5, ls = linestyle[mode],label=f"RSP_Hex")
                    

                    axs2[a,b].plot(Spheriod['lidarR'][mode], Spheriod['lidarVol'][mode], marker = "$O$",color = cm_sp2[mode],lw = 5,ls = linestyle[mode], label=f"HSRL_Sphd")
                    axs2[a,b].plot(Hex['lidarR'][mode],Hex['lidarVol'][mode], marker = "H", color = cm_t2[mode]  ,lw = 5, ls = linestyle[mode],label=f"HSRL_Hex")
                    
                    axsErr[a,b].plot(Spheriod['MapR'][mode], ErrMap,  marker = "o",color = cm_sp[mode],lw = 5,ls = linestyle[mode], label=f"{mode_v[mode]}")
                    axsErr[a,b].plot(Spheriod['MapR'][mode], ErrLidar,  marker = "o",color = cm_t2[mode],lw = 5,ls = linestyle[mode], label=f"{mode_v[mode]}")
                    
                    axs2[a,b].set_xlabel(r'rv $ \mu m$')
                    axs2[a,b].set_ylabel(r'dVdlnr')
                    axs2[a,b].set_xscale("log")
                    axs2[a,b].set_title(f'{mode_v[mode]}')

                    axsErr[a,b].set_xlabel(r'rv $ \mu m$')
                    axsErr[a,b].set_xscale("log")
                    axs2[a,b].legend()

                if i ==0 and RetNo ==1: #Plotting single sensor retrievals

                    cm_sp2 = ['#211363','#50438E','#476DE2']
                    cm_t2 = ['#BBC97C','#D29F4B','#87799C']
                    color_sph2 = '#adbf4b'
                    color_tamu2 = "#936ecf"

                  
                    axs2[a,b].plot(Spheriod['r'][mode], Spheriod[Retrival[i]][mode], marker = "$O$",color = cm_sp[mode],lw = 5,ls = linestyle[mode], label=f"Joint_Sphd")
                    axs2[a,b].plot(Hex['r'][mode],Hex[Retrival[i]][mode], marker = "H", color=cm_t[mode] ,lw = 5, ls = linestyle[mode],label=f"Joint_Hex")
                    

                    # axs2[a,b].plot(Spheriod['lidarR'][mode], Spheriod['lidarVol'][mode], marker = "$O$",color = cm_sp2[mode],lw = 5,ls = linestyle[mode], label=f"HSRL_Sphrod")
                    # axs2[a,b].plot(Hex['lidarR'][mode],Hex['lidarVol'][mode], marker = "H", color = cm_t2[mode]  ,lw = 5, ls = linestyle[mode],label=f"HSRL_Hex")
                    
                    # axsErr[a,b].plot(Spheriod['MapR'][mode], ErrMap,  marker = "o",color = cm_sp[mode],lw = 5,ls = linestyle[mode], label=f"{mode_v[mode]}")
                    # axsErr[a,b].plot(Spheriod['MapR'][mode], ErrLidar,  marker = "o",color = cm_t2[mode],lw = 5,ls = linestyle[mode], label=f"{mode_v[mode]}")
                    
                    axs2[a,b].set_xlabel(r'rv $ \mu m$')
                    axs2[a,b].set_ylabel(r'dVdlnr')
                    axs2[a,b].set_xscale("log")
                    axs2[a,b].set_title(f'{mode_v[mode]}')

                    # axsErr[a,b].set_xlabel(r'rv $ \mu m$')
                    # axsErr[a,b].set_xscale("log")
                    axs2[a,b].legend(prop = { "size": 22 },loc='upper center', bbox_to_anchor=(0.5, -0.4), shadow=True, ncol=3)

                if i>0:
                    
                    axs2[a,b].errorbar(lambda_ticks_str, Spheriod[Retrival[i]][mode],yerr=UNCERT[Retrival[i]], marker = "$O$",markeredgecolor='k',capsize=7,capthick =2,markersize=25, color = cm_sp[mode],lw = 4,ls = linestyle[mode], label=f"{keyInd}Sphd")
                    axs2[a,b].errorbar(lambda_ticks_str,Hex[Retrival[i]][mode],yerr=UNCERT[Retrival[i]],capsize=7,capthick =2,  marker = "H",markeredgecolor='k',markersize=25, color = cm_t[mode] ,lw = 4, ls = linestyle[mode],label=f"{keyInd}Hex")
                    
        
                    axsErr[a,b].errorbar(lambda_ticks_str, ErrMap,  marker = "o",color = cm_sp[mode],lw = 5,ls = linestyle[mode], label=f"{mode_v[mode]}")
                    axsErr[a,b].set_xticks(lambda_ticks_str) 
                    # axs[a,b].set_xticklabels(['0.41', '0.46', '0.55' , '0.67'  , '0.86'])
                    axsErr[a,b].set_xlabel(r'$\lambda \mu m$')
                     
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
                    axs2[a,b].set_title(f'{mode_v[mode]}')

                    
                    axs2[a,b].legend(prop = { "size": 21 },loc='upper center', bbox_to_anchor=(0.5, -0.4), shadow=True, ncol=4)
                    # axs2[a,b].legend(ncol = 4)
                    
                                
                
        axs2[7,1].errorbar(lambda_ticks_str, Spheriod['aod'],yerr=0.03+UNCERT['aod']*Spheriod['aod'], marker = "$O$",color = cm_sp[mode],markeredgecolor='k',capsize=7,capthick =2,markersize=25,lw = 4, label=f"{keyInd}Sphd")
        axs2[7,1].errorbar(lambda_ticks_str, Hex['aod'],yerr=0.03+UNCERT['aod']* Hex['aod'], marker = "H", color = cm_t[mode] ,lw = 4,markeredgecolor='k',capsize=7,capthick =2,markersize=25, label=f"{keyInd}Hex")
        axsErr[7,1].errorbar(lambda_ticks_str, ErrMap, marker = "o", color = color_tamu ,lw = 5,markeredgecolor='k',capsize=5,capthick =2,markersize=25, label=f"Hex")
            
        
        axs2[7,1].set_xticks(lambda_ticks_str)
        # axs[2,1].set_xticklabels(['0.41', '0.46', '0.55' , '0.67'  , '0.86'])
        axsErr[7,1].set_xlabel(r'$\lambda$')
        axsErr[7,1].set_ylabel('Total AOD')    
        axs2[7,1].set_xlabel(r'$\lambda$')
        axs2[7,1].set_ylabel('Total AOD')

        axs2[8,1].legend(prop = { "size": 21 }, ncol=2)
        axsErr[8,1].legend(prop = { "size": 21 }, ncol=1)
        axs2[7,1].legend()

        if RetNo ==0:
            axs2[6,1].scatter(mode_v,rslts_Sph[0]['sph'], color=cm_sp[mode], marker = "$RO$",s=500,label = "RSPsph")
            axs2[6,1].scatter(mode_v,rslts_Tamu[0]['sph'],color=cm_t[mode], marker = "$RH$",s=500, label = "RSPhex")
            axs2[6,1].scatter(mode_v,HSRL_sphrodT[0][0]['sph'],color=cm_sp2[mode],  marker = "$HO$",s=500, label = "HSRLsph")
            axs2[6,1].scatter(mode_v,HSRL_TamuT[0][0]['sph'], color=cm_t2[mode],marker = "$HH$",s=500, label = "HSRLhex")
            # axs2[6,1].legend(prop = { "size": 21 },ncol=1)
            axs2[6,1].set_xlabel("Aerosol type")
            axs2[6,1].set_ylabel('Spherical Frac')


            axs2[8,1].scatter('RSP SPh',rslts_Sph[0]['costVal'], color=cm_sp[mode],s=400, label = "RSPsph")
            axs2[8,1].scatter('RSP Hex',rslts_Tamu[0]['costVal'], color=cm_t[mode],s=400,label = "RSPhex")
            axs2[8,1].scatter('HSRL SPh',HSRL_sphrodT[0][0]['costVal'], color=cm_sp2[mode],s=400,label = "RSPsph")
            axs2[8,1].scatter('HSRL Hex',HSRL_TamuT[0][0]['costVal'], color=cm_t2[mode],s=400,label = "RSPhex")
            # axs2[8,1].legend()
            # axs2[8,1].set_xlabel("CostVal")
            axs2[8,1].set_ylabel("CostVal")
        
        if RetNo ==1:
            axs2[6,1].scatter(mode_v,LidarPolSph[0][0]['sph'], color=cm_sp2[mode], marker = "$RO$",s=500,label = "Jointsph")
            axs2[6,1].scatter(mode_v,LidarPolSph[0][0]['sph'],color=cm_t2[mode], marker = "$RH$",s=500, label = "Jointhex")
           # axs2[6,1].legend(prop = { "size": 21 },ncol=1)
            axs2[6,1].set_xlabel("Aerosol type")
            axs2[6,1].set_ylabel('Spherical Frac')


            axs2[8,1].scatter('Joint SPh',LidarPolSph[0][0]['costVal'], color='r',s=400, label = "RSPsph")
            axs2[8,1].scatter('Joint Hex',LidarPolSph[0][0]['costVal'], color='k',s=400,label = "RSPhex")
            # axs2[8,1].legend()
            # axs2[8,1].set_xlabel("CostVal")
            axs2[8,1].set_ylabel("CostVal")

        axs2[6,1].legend(prop = { "size": 18 },loc='upper center', bbox_to_anchor=(0.5, -0.4), shadow=True, ncol=4)
        axs2[8,1].legend(prop = { "size": 18 },loc='upper center', bbox_to_anchor=(0.5, -0.4), shadow=True, ncol=4)
        axs2[7,1].legend(prop = { "size": 18 },loc='upper center', bbox_to_anchor=(0.5, -0.4), shadow=True, ncol=4)
                 


        # lat_t = Hex['latitude']
        # lon_t = Hex['longitude']
        # dt_t = Hex['datetime']
        plt.tight_layout(rect=[0, 0, 1, 1])
        
        plt.savefig(f'/home/gregmi/ORACLES/HSRL_RSP/{RepMode}_RSPRetrieval{RepMode}.png', dpi = 400)
        
        # plt.suptitle(f'RSP Aerosol Retrieval \n  Lat:{lat_t} Lon :{lon_t}   Date: {dt_t}')
        if RepMode ==2:
            fig.savefig(f'/home/gregmi/ORACLES/HSRL_RSP/2+3_RSPRetrieval.png', dpi = 400)
        else:
            fig.savefig(f'/home/gregmi/ORACLES/HSRL_RSP/{NoMode}_RSPRetrieval.png', dpi = 400)


    



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

