# Greema Regmi, UMBC
# Date: Feb 24, 2023
"""
This code reads GRASP output dict from runGRASP.py and plots the data

"""

import numpy as np 
import matplotlib.pyplot as plt
import datetime as dt
import matplotlib.gridspec as gridspec

def calculate_rmse(y_true, y_pred):
        # calculate mean squared error
        mse = np.mean((y_true - y_pred)**2)
        # calculate root mean squared error
        rmse = np.sqrt(mse)
        return rmse




def PltGRASPoutput(RsltDict1, RsltDict2,file_name,PixNo,nkernel=1):
    plt.rcParams['font.size'] = '14.5'
    Spheriod = RsltDict1[0]
    Hex= RsltDict2[0]
    
    #Stokes Vectors Plot
    date_latlon = ['datetime', 'longitude', 'latitude']
    Xaxis = ['r','lambda','sca_ang','rv','height']
    Retrival = ['dVdlnr','aodMode','ssaMode','n', 'k']
    #['sigma', 'vol', 'aodMode','ssaMode', 'rEff', 'costVal']
    Angles =   ['sza', 'vis', 'fis','angle' ]
    Stokes =   ['meas_I', 'fit_I', 'meas_P_rel', 'fit_P_rel']
    Pij    = ['p11', 'p12', 'p22', 'p33'], 
    Lidar=  ['heightStd','g','LidarRatio','LidarDepol', 'gMode', 'LidarRatioMode', 'LidarDepolMode']


    
    # Plot the AOD data
    y = [0,1,2,0,1,2,]
    x = np.repeat((0,1),3)
    mode_v = ["fine", "coarse"]
    linestyle =[':', '-']

    cm_sp = ['#008080',"#C1E1C1" ]
    cm_t = ['#900C3F',"#FF5733" ]
    color_sph = '#0c7683'
    color_tamu = "#BC106F"

    #Retrivals:
    fig, axs = plt.subplots(nrows= 5, ncols=1, figsize=(8, 25))
    for i in range(len(Retrival)):
        for mode in range(Spheriod['r'].shape[0]): #for each modes
            if i ==0:
                axs[i].plot(Spheriod['r'][mode], Spheriod[Retrival[i]][mode], marker = "$O$",color = cm_sp[mode],ls = linestyle[mode], label=f"Sphrod_{mode_v[mode]}")
                axs[i].plot(Hex['r'][mode],Hex[Retrival[i]][mode], marker = "H", color = cm_t[mode] , ls = linestyle[mode],label=f"Hex_{mode_v[mode]}")
                axs[0].set_xlabel('Radius')
                axs[0].set_xscale("log")
            else:
                axs[i].plot(Spheriod['lambda'], Spheriod[Retrival[i]][mode], marker = "$O$",color = cm_sp[mode],ls = linestyle[mode], label=f"Sphrod_{mode_v[mode]}")
                axs[i].plot(Hex['lambda'],Hex[Retrival[i]][mode], marker = "H",color = cm_t[mode] , ls = linestyle[mode],label=f"Hex_{mode_v[mode]}")
                axs[i].set_xticks(Spheriod['lambda'])
                axs[i].set_xticklabels(['0.4103', '0.4691', '0.555' , '0.67'  , '0.8635'])
        axs[i].set_ylabel(f'{Retrival[i]}')
        axs[4].set_xlabel(r'$\lambda$')
        axs[0].legend()

    lat_t = Hex['latitude']
    lon_t = Hex['longitude']
    dt_t = Hex['datetime']
    plt.suptitle(f'RSP  Lat:{lat_t} Lon :{lon_t}\n Date: {dt_t}  Pixel:{PixNo}')

    fig.savefig(f'/home/gregmi/ORACLES/HSRL_RSP/{dt_t}_{PixNo}_RSPRetrieval.png', dpi = 400)

    #Stokes: 
    wl = RsltDict1[0]['lambda'] 
    fig, axs = plt.subplots(nrows= 4, ncols=len(wl), figsize=(20, 10),gridspec_kw={'height_ratios': [1, 0.3,1,0.3]}, sharex='col')
    # Plot the AOD data
    meas_P_rel = 'meas_P_rel'
    

    for nwav in range(len(wl)):
    # Plot the fit and measured I data
       
        axs[0, nwav].fill_between(Spheriod ['sca_ang'][:,nwav],Spheriod ['meas_I'][:,nwav], Spheriod ['meas_I'][:,nwav]*1.03, color = 'r',alpha=0.2, ls = "--",label="+3%")
        axs[0, nwav].fill_between(Spheriod ['sca_ang'][:,nwav],Spheriod ['meas_I'][:,nwav], Spheriod ['meas_I'][:,nwav]*0.97, color = "b",alpha=0.2, ls = "--",label="-3%")
        axs[0, nwav].plot(Spheriod['sca_ang'][:,nwav], Spheriod['meas_I'][:,nwav], color = "k", lw = 1, label="meas")

        axs[0, nwav].plot(Spheriod ['sca_ang'][:,nwav], Spheriod ['fit_I'][:,nwav],color =color_sph , lw = 2, ls = '--',label="fit sphrod")
        # axs[0, nwav].scatter(Spheriod ['sca_ang'][:,nwav][marker_indsp], Spheriod ['fit_I'][:,nwav][marker_indsp],color =color_sph , m = "o",label="fit sphrod")
        
        # axs[0, nwav].set_xlabel('Scattering angles (deg)')
        axs[0, 0].set_ylabel('I')
        # axs[0, nwav].legend()

        # Plot the fit and measured QoI data
        axs[2, nwav].plot(Spheriod ['sca_ang'][:,nwav], Spheriod ['meas_P_rel'][:,nwav],color = "k", lw = 1, label="meas")
        axs[2, nwav].plot(Spheriod ['sca_ang'][:,nwav], Spheriod ['fit_P_rel'][:,nwav], color =color_sph, ls = '--', label="fit sph")
      
        axs[2, nwav].fill_between(Spheriod ['sca_ang'][:,nwav],(Spheriod ['meas_P_rel'][:,nwav]), (Spheriod ['meas_P_rel'][:,nwav])*1.03,color = 'r', alpha=0.2,ls = "--", label="+3%")
        axs[2, nwav].fill_between(Spheriod ['sca_ang'][:,nwav],(Spheriod ['meas_P_rel'][:,nwav]), (Spheriod ['meas_P_rel'][:,nwav])*0.97,color = "b", alpha=0.2,ls = "--", label="-3%")
        # axs[2, nwav].set_xlabel('Scattering angles (deg)')
        axs[2, 0].set_ylabel('DOLP')
        axs[0, nwav].set_title(f"{wl[nwav]}", fontsize = 14)
        
        axs[0, nwav].plot(Hex['sca_ang'][:,nwav], Hex['fit_I'][:,nwav],color =color_tamu , lw = 2, ls = "dashdot",label="fit Hex")
        axs[2, nwav].plot(Hex['sca_ang'][:,nwav],Hex['fit_P_rel'][:,nwav],color = color_tamu , lw = 2,ls = "dashdot", label = "fit Hex") 

        


        sphErr = 100 * abs(Spheriod['meas_I'][:,nwav]-Spheriod ['fit_I'][:,nwav] )/Spheriod['meas_I'][:,nwav]
        HexErr = 100 * abs(Hex['meas_I'][:,nwav]-Hex['fit_I'][:,nwav] )/Hex['meas_I'][:,nwav]
        
        axs[1, nwav].plot(Spheriod ['sca_ang'][:,nwav], sphErr,color =color_sph ,label="Sphrod")
        axs[1, nwav].plot(Hex ['sca_ang'][:,nwav], HexErr,color = color_tamu ,label="Hex")
        axs[1, 0].set_ylabel('Err I %')
        

        sphErrP = 100 * abs(Spheriod['meas_P_rel'][:,nwav]-Spheriod ['fit_P_rel'][:,nwav])
        HexErrP = 100 * abs(Hex['meas_P_rel'][:,nwav]-Hex['fit_P_rel'][:,nwav] )
        
        axs[3, nwav].plot(Spheriod ['sca_ang'][:,nwav], sphErrP,color =color_sph ,label="Sphrod")
        axs[3, nwav].plot(Hex ['sca_ang'][:,nwav], HexErrP,color =color_tamu ,label="Hex")
       
        axs[3, nwav].set_xlabel('Scattering angles (deg)')
        # axs[3, nwav].set_ylabel('Err P')
        # axs[3, nwav].legend()
        axs[3, nwav].set_xlabel('Scattering angles (deg)')
        axs[3, 0].set_ylabel('|Meas-fit|')
        # axs[1, nwav].set_title(f"{wl[nwav]}", fontsize = 14)
    
        axs[0, 4].legend(bbox_to_anchor=(1.05, 0.5), loc='center left')
        axs[1, 4].legend(bbox_to_anchor=(1.05, 0.5), loc='center left')
        # plt.tight_layout()

        marker_ind = [80,120,130,140]

        plt.subplots_adjust(hspace=0.05)
    plt.suptitle(f'RSP  Lat:{lat_t} Lon :{lon_t}\n Date: {dt_t}  Pixel:{PixNo}')   
        
        

    fig.savefig(f'/home/gregmi/ORACLES/HSRL_RSP/{dt_t}_{PixNo}_I_P.png', dpi = 400)


    fig, axs = plt.subplots(nrows= 2, ncols=len(wl), figsize=(22, 10))
    # Plot the AOD data
    
    
    for nwav in range(len(wl)):
    # Plot the fit and measured I data
        axs[0, nwav].plot(Spheriod ['angle'][:,0,nwav], Spheriod ['p11'][:,0,nwav],color =color_sph,  label="Spheriod")
        axs[0, nwav].plot(Hex['angle'][:,0,nwav], Hex['p11'][:,0,nwav],color =color_tamu, label="Hexahedral")
        axs[0, nwav].set_xlabel('Scattering angles (deg)')
        axs[0, nwav].set_ylabel('P11')
        axs[0, nwav].legend()

        # Plot the fit and measured P data
        axs[1, nwav].plot(Spheriod ['angle'][:,0,nwav],Spheriod ['p12'][:,0,nwav]/Spheriod ['p11'][:,0,nwav],color =color_sph, label="Spheriod") 
        axs[1, nwav].plot(Hex['angle'][:,0,nwav],Hex['p12'][:,0,nwav]/Hex['p11'][:,0,nwav],color =color_tamu, label="Hexahedral") 
        axs[1, nwav].set_xlabel('Scattering angles (deg)')
        axs[1, nwav].set_ylabel('p12/p11')
        axs[1, nwav].set_title(f"{wl[nwav]}")
        axs[1, nwav].legend()
    plt.suptitle(f'RSP  Lat:{lat_t} Lon :{lon_t} Date: {dt_t}  Pixel:{PixNo}')

    # fig.savefig(f'/home/gregmi/ORACLES/HSRL_RSP/RSP_{date}_{PixNo}_Pval.png')






def PltGRASPoutput(RsltDict1, RsltDict2,file_name,PixNo,nkernel=1):
    

    plt.rcParams['font.size'] = '14.5'
    Spheriod = RsltDict1[0]
    Hex= RsltDict2[0]
    
    #Stokes Vectors Plot
    date_latlon = ['datetime', 'longitude', 'latitude']
    Xaxis = ['r','lambda','sca_ang','rv','height']
    Retrival = ['dVdlnr','aodMode','ssaMode','n', 'k']
    #['sigma', 'vol', 'aodMode','ssaMode', 'rEff', 'costVal']
    Angles =   ['sza', 'vis', 'fis','angle' ]
    Stokes =   ['meas_I', 'fit_I', 'meas_p_rel', 'fit_p_rel']
    Pij    = ['p11', 'p12', 'p22', 'p33'], 
    Lidar=  ['heightStd','g','LidarRatio','LidarDepol', 'gMode', 'LidarRatioMode', 'LidarDepolMode']


    
    # Plot the AOD data
    y = [0,1,2,0,1,2,]
    x = np.repeat((0,1),3)
    mode_v = ["fine", "coarse"]
    linestyle =[':', '-']

    cm_sp = ['#008080',"#C1E1C1" ]
    cm_t = ['#900C3F',"#FF5733" ]
    color_sph = '#0c7683'
    color_tamu = "#BC106F"

    #Retrivals:
    fig, axs = plt.subplots(nrows= 5, ncols=1, figsize=(8, 25))


    for i in range(len(Retrival)):
        for mode in range(Spheriod['r'].shape[0]): #for each modes
            if i ==0:
                axs[i].plot(Spheriod['r'][mode], Spheriod[Retrival[i]][mode], marker = "$O$",color = cm_sp[mode],ls = linestyle[mode], label=f"Sphrod_{mode_v[mode]}")
                axs[i].plot(Hex['r'][mode],Hex[Retrival[i]][mode], marker = "H", color = cm_t[mode] , ls = linestyle[mode],label=f"Hex_{mode_v[mode]}")
                axs[0].set_xlabel('Radius')
                axs[0].set_xscale("log")
            else:
                axs[i].plot(Spheriod['lambda'], Spheriod[Retrival[i]][mode], marker = "$O$",color = cm_sp[mode],ls = linestyle[mode], label=f"Sphrod_{mode_v[mode]}")
                axs[i].plot(Hex['lambda'],Hex[Retrival[i]][mode], marker = "H",color = cm_t[mode] , ls = linestyle[mode],label=f"Hex_{mode_v[mode]}")
                axs[i].set_xticks(Spheriod['lambda'])
                axs[i].set_xticklabels(['0.4103', '0.4691', '0.555' , '0.67'  , '0.8635'])
        axs[i].set_ylabel(f'{Retrival[i]}')
        axs[4].set_xlabel(r'$\lambda$')
        axs[0].legend()

    lat_t = Hex['latitude']
    lon_t = Hex['longitude']
    dt_t = Hex['datetime']
    plt.suptitle(f'RSP  Lat:{lat_t} Lon :{lon_t}\n Date: {dt_t}  Pixel:{PixNo}')

    fig.savefig(f'/home/gregmi/ORACLES/HSRL_RSP/{dt_t}_{PixNo}_RSPRetrieval.png', dpi = 400)

    #Stokes: 
    wl = RsltDict1[0]['lambda'] 
    fig, axs = plt.subplots(nrows= 4, ncols=len(wl), figsize=(20, 10),gridspec_kw={'height_ratios': [1, 0.3,1,0.3]}, sharex='col')
    # Plot the AOD data
    meas_P_rel = 'meas_P_rel'
    

    for nwav in range(len(wl)):
    # Plot the fit and measured I data
       
        axs[0, nwav].fill_between(Spheriod ['sca_ang'][:,nwav],Spheriod ['meas_I'][:,nwav], Spheriod ['meas_I'][:,nwav]*1.03, color = 'r',alpha=0.2, ls = "--",label="+3%")
        axs[0, nwav].fill_between(Spheriod ['sca_ang'][:,nwav],Spheriod ['meas_I'][:,nwav], Spheriod ['meas_I'][:,nwav]*0.97, color = "b",alpha=0.2, ls = "--",label="-3%")
        axs[0, nwav].plot(Spheriod['sca_ang'][:,nwav], Spheriod['meas_I'][:,nwav], color = "k", lw = 1, label="meas")

        axs[0, nwav].plot(Spheriod ['sca_ang'][:,nwav], Spheriod ['fit_I'][:,nwav],color =color_sph , lw = 2, ls = '--',label="fit sphrod")
        # axs[0, nwav].scatter(Spheriod ['sca_ang'][:,nwav][marker_indsp], Spheriod ['fit_I'][:,nwav][marker_indsp],color =color_sph , m = "o",label="fit sphrod")
        
        # axs[0, nwav].set_xlabel('Scattering angles (deg)')
        axs[0, 0].set_ylabel('I')
        # axs[0, nwav].legend()

        # Plot the fit and measured QoI data
        axs[2, nwav].plot(Spheriod ['sca_ang'][:,nwav], Spheriod ['meas_P_rel'][:,nwav],color = "k", lw = 1, label="meas")
        axs[2, nwav].plot(Spheriod ['sca_ang'][:,nwav], Spheriod ['fit_P_rel'][:,nwav], color =color_sph, ls = '--', label="fit sph")
      
        axs[2, nwav].fill_between(Spheriod ['sca_ang'][:,nwav],(Spheriod ['meas_P_rel'][:,nwav]), (Spheriod ['meas_P_rel'][:,nwav])*1.03,color = 'r', alpha=0.2,ls = "--", label="+3%")
        axs[2, nwav].fill_between(Spheriod ['sca_ang'][:,nwav],(Spheriod ['meas_P_rel'][:,nwav]), (Spheriod ['meas_P_rel'][:,nwav])*0.97,color = "b", alpha=0.2,ls = "--", label="-3%")
        # axs[2, nwav].set_xlabel('Scattering angles (deg)')
        axs[2, 0].set_ylabel('DOLP')
        axs[0, nwav].set_title(f"{wl[nwav]}", fontsize = 14)
        
        axs[0, nwav].plot(Hex['sca_ang'][:,nwav], Hex['fit_I'][:,nwav],color =color_tamu , lw = 2, ls = "dashdot",label="fit Hex")
        axs[2, nwav].plot(Hex['sca_ang'][:,nwav],Hex['fit_P_rel'][:,nwav],color = color_tamu , lw = 2,ls = "dashdot", label = "fit Hex") 

        


        sphErr = 100 * abs(Spheriod['meas_I'][:,nwav]-Spheriod ['fit_I'][:,nwav] )/Spheriod['meas_I'][:,nwav]
        HexErr = 100 * abs(Hex['meas_I'][:,nwav]-Hex['fit_I'][:,nwav] )/Hex['meas_I'][:,nwav]
        
        axs[1, nwav].plot(Spheriod ['sca_ang'][:,nwav], sphErr,color =color_sph ,label="Sphrod")
        axs[1, nwav].plot(Hex ['sca_ang'][:,nwav], HexErr,color = color_tamu ,label="Hex")
       
        # axs[0, nwav].set_xlabel('Scattering angles (deg)')
        axs[1, 0].set_ylabel('Err I %')
        

        sphErrP = 100 * abs(Spheriod['meas_P_rel'][:,nwav]-Spheriod ['fit_P_rel'][:,nwav])
        HexErrP = 100 * abs(Hex['meas_P_rel'][:,nwav]-Hex['fit_P_rel'][:,nwav] )
        
        axs[3, nwav].plot(Spheriod ['sca_ang'][:,nwav], sphErrP,color =color_sph ,label="Sphrod")
        axs[3, nwav].plot(Hex ['sca_ang'][:,nwav], HexErrP,color =color_tamu ,label="Hex")
       
        axs[3, nwav].set_xlabel('Scattering angles (deg)')
        # axs[3, nwav].set_ylabel('Err P')
        # axs[3, nwav].legend()
        axs[3, nwav].set_xlabel('Scattering angles (deg)')
        axs[3, 0].set_ylabel('|Meas-fit|')
        # axs[1, nwav].set_title(f"{wl[nwav]}", fontsize = 14)
    
        axs[0, 4].legend(bbox_to_anchor=(1.05, 0.5), loc='center left')
        axs[1, 4].legend(bbox_to_anchor=(1.05, 0.5), loc='center left')
        # plt.tight_layout()

        marker_ind = [80,120,130,140]

        plt.subplots_adjust(hspace=0.05)
    plt.suptitle(f'RSP  Lat:{lat_t} Lon :{lon_t}\n Date: {dt_t}  Pixel:{PixNo}')   
        
        

    fig.savefig(f'/home/gregmi/ORACLES/HSRL_RSP/{dt_t}_{PixNo}_I_P.png', dpi = 400)


    fig, axs = plt.subplots(nrows= 2, ncols=len(wl), figsize=(22, 10))
    # Plot the AOD data
    
    
    for nwav in range(len(wl)):
    # Plot the fit and measured I data
        axs[0, nwav].plot(Spheriod ['angle'][:,0,nwav], Spheriod ['p11'][:,0,nwav],color =color_sph,  label="Spheriod")
        axs[0, nwav].plot(Hex['angle'][:,0,nwav], Hex['p11'][:,0,nwav],color =color_tamu, label="Hexahedral")
        axs[0, nwav].set_xlabel('Scattering angles (deg)')
        axs[0, nwav].set_ylabel('P11')
        axs[0, nwav].legend()

        # Plot the fit and measured P data
        axs[1, nwav].plot(Spheriod ['angle'][:,0,nwav],Spheriod ['p12'][:,0,nwav]/Spheriod ['p11'][:,0,nwav],color =color_sph, label="Spheriod") 
        axs[1, nwav].plot(Hex['angle'][:,0,nwav],Hex['p12'][:,0,nwav]/Hex['p11'][:,0,nwav],color =color_tamu, label="Hexahedral") 
        axs[1, nwav].set_xlabel('Scattering angles (deg)')
        axs[1, nwav].set_ylabel('p12/p11')
        axs[1, nwav].set_title(f"{wl[nwav]}")
        axs[1, nwav].legend()
    plt.suptitle(f'RSP  Lat:{lat_t} Lon :{lon_t} Date: {dt_t}  Pixel:{PixNo}')

    # fig.savefig(f'/home/gregmi/ORACLES/HSRL_RSP/RSP_{date}_{PixNo}_Pval.png')




# Results = [rslts_Sph[0],rslts_Tamu[0], LidarPolTAMU[0]]

# for i in 

