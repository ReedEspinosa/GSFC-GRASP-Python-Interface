#!/usr/bin/env python3

# Author: Nirandi Jayasinnghe
# Last Updated: Aug 14 2024

""" This script is created to plot 2D/1D graphs of retrivals from LES and 1D/3D RT data."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

def FILE_PATH(RT):
    if RT == 1 : file_path = "/data/ESI/User/njayasinghe/LES_Retrievals/1D/"
    if RT == 3 : file_path = "/data/ESI/User/njayasinghe/LES_Retrievals/3D/"
    #else: print("RT dimensions were not specified")

    return file_path

def create_2D_Grid(RT,nX=168,nY=168):

    aod = np.zeros((nX,nY))
    ssa = np.zeros((nX,nY))

    aod[:,:] = np.nan
    ssa[:,:] = np.nan

    for xx in range(0,nX,1):
        for yy in range(0,nY,1):

            file_path = FILE_PATH(RT)+"LES_XP_"+str(xx)+"_YP_"+str(yy)
            try:
                data = np.load(file_path, allow_pickle=True)
                aod[xx,yy] = data[0]['aod'][0]
                ssa[xx,yy] = data[0]['ssa'][0]
                
            except:
                print("LES_XP_"+str(xx)+"_YP_"+str(yy)+" File not found !")

    return aod,ssa

def Plot2D_AOD_Map(var,RT,save_path=None):

    title ="Retrievd AOD at 0.669 um for "+str(RT)+"D RT"

    plt.figure(figsize=(10, 12), dpi = 200)
    ax = plt.axes()
    c = ax.pcolormesh(var, norm=colors.LogNorm(vmin = np.nanmin(var), vmax = np.nanmax(var)))
    ax.set_title('\n'+title+'\n',fontsize=15)
    ax.set_ylabel("nX",fontsize=15)
    ax.set_xlabel("nY",fontsize=15)
    ax.grid()
    plt.colorbar(c, cmap = 'jet', extend = 'both', orientation="horizontal", pad = 0.1, label='Log(AOD)')
    if save_path:
        plt.savefig(save_path+"AOD.png", dpi = 300)
        plt.close()

def Plot2D_SSA_Maps(var,RT,save_path=None):

    title ="Retrievd SSA at 0.669 um for "+str(RT)+"D RT"
    
    plt.figure(figsize=(12, 12), dpi = 300)
    ax = plt.axes()
    c = ax.pcolormesh(var, vmin = np.nanmin(var), vmax = np.nanmax(var))
    ax.set_title('\n'+title+'\n',fontsize=15)   
    ax.set_ylabel("nX",fontsize=15)
    ax.set_xlabel("nY",fontsize=15)
    ax.grid()
    plt.colorbar(c, cmap = 'hot', extend = 'both', orientation="horizontal", pad = 0.1, label='SSA')
    if save_path:
        plt.savefig(save_path+"SSA.png", dpi = 300)
        plt.close()


def PlotSinglePX(XP,YP,RT,save_path=None):

    #file_path = FILE_PATH(RT)+"LES_XP_"+str(XP)+"_YP_"+str(YP)
    #file_path = '/data/home/njayasinghe/LES/GSFC-GRASP-Python-Interface/tmp/Corrt_All_only_30_ang_XPX_'+str(XP)+'_YP_'+str(YP)
    #file_path = '/data/home/njayasinghe/LES/GSFC-GRASP-Python-Interface/tmp/Corrt_All_only_30_ang_XPX_'+str(XP)+'_YP_'+str(YP)+'FWD'
    file_path=f'/data/home/njayasinghe/LES/GSFC-GRASP-Python-Interface/tmp/Corrt_All_principal_30_ang_XPX_'+str(XP)+'_YP_'+str(YP)+'FWD'


    try:
        data = np.load(file_path, allow_pickle=True)
        #var_names = data[0].keys()
        meas_I = data[0]['meas_I'][:,0]
        meas_QoI = data[0]['meas_QoI'][:,0]
        meas_UoI = data[0]['meas_UoI'][:,0]
        meas_P_rel = data[0]['meas_P_rel'][:,0]

        fit_I= data[0]['fit_I'][:,0]
        fit_QoI= data[0]['fit_QoI'][:,0]
        fit_UoI = data[0]['fit_UoI'][:,0]
        fit_P_rel = data[0]['fit_P_rel'][:,0]
        
        sca_ang = data[0]['sca_ang'][:,0]
        aod = data[0]['aod'][0]
        ssa = data[0]['ssa'][0]
        #sph = data[0]['sph'][0]
        RRI = data[0]['n'][0]
        IRI = data[0]['k'][0]
        #brdf = data[0]['brdf']
        wl = data[0]['lambda'][0]


    except:
        print("File cannnot be found!")


    fig, axs = plt.subplots(4,1, figsize=(8,10),constrained_layout=True, dpi = 300)
    fig.suptitle("\n Xpx = "+ str(XP)+"; Ypx = "+str(YP)+"; wl = "+str(wl)+" um \n AOD = "+str(aod)+
        "; SSA = "+str(ssa)+"; RRI = "+str(RRI)+"; IRI = "+str(IRI)+"\n",fontsize=15)

    axs[0].plot(sca_ang,meas_I, '.', label= 'LES Data')
    axs[0].plot(sca_ang,fit_I, '-', label= 'GRASP fit')
    axs[0].grid()
    axs[0].set_ylabel("I",fontsize=15)
    axs[0].legend()

    axs[1].plot(sca_ang,meas_QoI, '.', label= 'LES Data')
    axs[1].plot(sca_ang,fit_QoI, '-', label= 'GRASP fit')
    axs[1].grid()
    axs[1].set_ylabel("Q/I",fontsize=15)

    axs[2].plot(sca_ang,meas_UoI, '.', label= 'LES Data')
    axs[2].plot(sca_ang,fit_UoI, '-', label= 'GRASP fit')
    axs[2].grid()
    axs[2].set_ylabel("U/I",fontsize=15)

    axs[3].plot(sca_ang,meas_P_rel, '.', label= 'LES Data')
    axs[3].plot(sca_ang,fit_P_rel, '-', label= 'GRASP fit')
    axs[3].grid()
    axs[3].set_ylabel("DoLP",fontsize=15)
    axs[3].set_xlabel("Scattering Angle")

    if save_path:
        plt.savefig(save_path+"LES_XP_"+str(XP)+"_YP_"+str(YP)+".png", dpi = 300)
        plt.close()

def Plot_I(XP,YP,RT,save_path=None):

    #file_path = '/data/home/njayasinghe/LES/GSFC-GRASP-Python-Interface/tmp/I_only_30ang'+str(XP)+'_YP_'+str(YP)
    file_path = f'/data/home/njayasinghe/LES/GSFC-GRASP-Python-Interface/tmp/Corrt_I_only_30_ang_XPX_'+str(XP)+'_YP_'+str(YP)
    #file_path = FILE_PATH(RT)+"LES_XP_"+str(XP)+"_YP_"+str(YP)
    #file_path = f'/data/home/njayasinghe/LES/GSFC-GRASP-Python-Interface/tmp/LESWithCorrection_XP_'+str(XPX)+'_YP_'+str(YPX)
    #file_path = f'/data/home/njayasinghe/LES/GSFC-GRASP-Python-Interface/tmp/LESNoCorrection_XP_'+str(XPX)+'_YP_'+str(YPX)


    try:
        data = np.load(file_path, allow_pickle=True)
        #var_names = data[0].keys()
        meas_I = data[0]['meas_I'][:,0]
        fit_I= data[0]['fit_I'][:,0]
        
        sca_ang = data[0]['sca_ang'][:,0]

        aod = data[0]['aod'][0]
        ssa = data[0]['ssa'][0]
        #sph = data[0]['sph'][0]
        RRI = data[0]['n'][0]
        IRI = data[0]['k'][0]
        #brdf = data[0]['brdf']
        wl = data[0]['lambda'][0]


    except:
        print("File cannnot be found!")


    fig, axs = plt.subplots(1,1, figsize=(4,5),constrained_layout=True, dpi = 300)
    fig.suptitle("\n Xpx = "+ str(XP)+"; Ypx = "+str(YP)+"; wl = "+str(wl)+" um \n AOD = "+str(aod)+
        "; SSA = "+str(ssa)+"\n RRI = "+str(RRI)+"; IRI = "+str(IRI)+"\n",fontsize=15)

    axs.plot(sca_ang,meas_I, '.', label= 'LES Data')
    axs.plot(sca_ang,fit_I, '-', label= 'GRASP fit')
    axs.grid()
    axs.set_ylabel("I",fontsize=15)
    axs.set_xlabel("Scattering Angle")
    axs.legend()

    if save_path:
        plt.savefig(save_path+"LES_XP_"+str(XP)+"_YP_"+str(YP)+".png", dpi = 300)
        plt.close()


def Plot_DOLP(XPX,YPX,RT,save_path=None):

    file_path = f'/data/home/njayasinghe/LES/GSFC-GRASP-Python-Interface/tmp/Corrt_DOLP_only_30_ang_XPX_'+str(XPX)+'_YP_'+str(YPX)
    #file_path = f'/data/home/njayasinghe/LES/GSFC-GRASP-Python-Interface/tmp/Corrt_DOLP_only_30_ang_XPX_'+str(XPX)+'_YP_'+str(YPX)
    #file_path = f'/data/home/njayasinghe/LES/GSFC-GRASP-Python-Interface/tmp/LESNoCorrection_XP_'+str(XPX)+'_YP_'+str(YPX)


    try:
        data = np.load(file_path, allow_pickle=True)
        #var_names = data[0].keys()
        meas_P_rel = data[0]['meas_P_rel'][:,0]
        fit_P_rel= data[0]['fit_P_rel'][:,0]
        
        sca_ang = data[0]['sca_ang'][:,0]

        aod = data[0]['aod'][0]
        ssa = data[0]['ssa'][0]
        #sph = data[0]['sph'][0]
        RRI = data[0]['n'][0]
        IRI = data[0]['k'][0]
        #brdf = data[0]['brdf']
        wl = data[0]['lambda'][0]


    except:
        print("File cannnot be found!")


    fig, axs = plt.subplots(1,1, figsize=(4,5),constrained_layout=True, dpi = 300)
    fig.suptitle("\n Xpx = "+ str(XPX)+"; Ypx = "+str(YPX)+"; wl = "+str(wl)+" um \n AOD = "+str(aod)+
        "; SSA = "+str(ssa)+"; RRI = "+str(RRI)+"; IRI = "+str(IRI)+"\n",fontsize=15)

    axs.plot(sca_ang,meas_P_rel, '.', label= 'LES Data')
    axs.plot(sca_ang,fit_P_rel, '-', label= 'GRASP fit')
    axs.grid()
    axs.set_ylabel("DOLP",fontsize=15)
    axs.set_xlabel("Scattering Angle")
    axs.legend()

    if save_path:
        plt.savefig(save_path+"LES_XP_"+str(XP)+"_YP_"+str(YP)+".png", dpi = 300)
        plt.close()














