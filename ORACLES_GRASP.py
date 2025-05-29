"""
# Greema Regmi, UMBC
# Date: Jan 31, 2023

This code reads Polarimetric data from the Campaigns and runs GRASP. This code was created to Validate the Aerosol retrivals performed using Non Spherical Kernels (Hexahedral from TAMU DUST 2020)

"""
# %load_ext autoreload
# %autoreload 2
# %reload_ext autoreload

# %run -d -b runGRASP.py:LINENUM scriptToRun.py
# %load_ext autoreload


import sys
import pandas as pd
import warnings
from CreateRsltsDict import Read_Data_RSP_Oracles
from CreateRsltsDict import Read_Data_HSRL_Oracles_Height
import netCDF4 as nc
from runGRASP import graspDB, graspRun, pixel, graspYAML
from matplotlib import pyplot as plt
import os
if os.uname()[1]=='uranus': plt.switch_backend('agg')
import numpy as np
import datetime as dt
from numpy import nanmean
import h5py 
# sys.path.append("/home/gregmi/git/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases")
# from architectureMap import returnPixel
from matplotlib.font_manager import FontProperties
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as mpatches
from matplotlib.ticker import ScalarFormatter
# from Plot_ORACLES import PltGRASPoutput, PlotRetrievals
import yaml
import pickle
import datetime
from netCDF4 import Dataset
from scipy.optimize import curve_fit
import re




"""Code contains the tools to read and run GRASP for Poalrimter (designed for RSP) and Lidar (HSRL2) from an aircraft measuments



"""

#Div




# # Path to the Polarimeter data (RSP, In this case)

# #Case1
# file_path = "/home/gregmi/ORACLES/RSP1-L1C_P3_20180922_R03/"  #Path to the ORACLE data file
# file_name =  "/RSP1-P3_L1C-RSPCOL-CollocatedRadiances_20180922T151106Z_V003-20210421T233946Z.h5" #Name of the ORACLES file
# #Paths to the Lidar Data


# file_path = '/home/gregmi/ORACLES/Case2'
# file_name ='/RSP1-P3_L1C-RSPCOL-CollocatedRadiances_20180921T180327Z_V003-20210421T232647Z.h5'

# #Case 1: 22nd sept 2018, ORACLES
# HSRLfile_path = "/home/gregmi/ORACLES/HSRL" #Path to the ORACLE data file 
# HSRLfile_name =  "/HSRL2_P3_20180922_R2.h5" #Name of the ORACLES file

# #Case2: 21st sept 2018, ORACLES

# HSRLfile_path = '/home/gregmi/ORACLES/Case2'
# HSRLfile_name =  "/HSRL2_P3_20180921_R2.h5"


# #Path to the gas absorption (tau) values for gas absorption correction
# GasAbsFn = '/home/gregmi/ORACLES/UNL_VRTM/shortwave_gas.unlvrtm.nc'
# SpecResFnPath = '/home/gregmi/ORACLES/RSP_Spectral_Response/'
# #This is required if we want to configure the HSRL yaml file based on the GRASP output for the RSP
# noMod =3  #number of aerosol mode, here 2 for fine+coarse mode configuration
# maxr=1.05  #set max and min value : here max = 5% incease, min 5% decrease : this is a very narrow distribution
# minr =0.95
# a=1 #no of char # this is a varible added to char and modes to avoid char[0]/ mod[0] which doesnt exist

# # #Saving state variables and noises from yaml file
# # def Normalize_prof():



class ScalarFormatterClass(ScalarFormatter):
   def _set_format(self):
      self.format = "%1.2f"

def VariableNoise(YamlFileName,nwl=3): #This will store the inforamtion of the characteristics and noises from the yaml file to a string for reference. 
    
    noMod =2  #number of aerosol mode, here 2 for fine+coarse mode configuration
    maxr=1.05  #set max and min value : here max = 5% incease, min 5% decrease : this is a very narrow distribution
    minr =0.95
    a=1 #no of char # this is a varible added to char and modes to avoid char[0]/ mod[0] which doesnt exist
    nwl=3
    Info = []
    with open(YamlFileName, 'r') as f:  
        data = yaml.safe_load(f)
    for i in range (1,nwl+1):
        Noise_yaml = data['retrieval']['inversion']['noises'][f'noise[{i}]']
        errType =  Noise_yaml['error_type']
        noiseName = Noise_yaml[f'measurement_type[1]']['type'] #type of measurement
        sd = Noise_yaml['standard_deviation'] #satndard divation for that meas typ
        NoiseInfo = f'The {errType} noise for {noiseName}  are {sd} '
        Info.append(NoiseInfo )

    #Characteristics'

    YamlChar =[]  #This list stores the name of the charater types in the yaml files
    noYmlChar = np.arange(1,7) #No of aerosol characters types in the yaml file (This can be adjusted based on the parameters we want to change)
    
    for i in noYmlChar:
        YamlChar.append(data['retrieval']['constraints'][f'characteristic[{i}]']['type'])
    print(len(YamlChar))
    
    #change the yaml intitial conditions using the RSP GRASP output
    for i in range(len(YamlChar)): #loop over the character types in the list
        for noMd in range(noMod): #loop over the aerosol modes (i.e 2 for fine and coarse)
            #State Varibles from yaml file: 
            typ= data['retrieval']['constraints'][f'characteristic[{i+1}]']['type']
            initCond = data['retrieval']['constraints'][f'characteristic[{i+a}]'][f'mode[{noMd+a}]']['initial_guess']
            # if YamlChar[i] == 'aerosol_concentration': #Update the in and max values from aerosol properties retrieved from the RSP measurements
            if YamlChar[i] == 'vertical_profile_normalized':
                val = [initCond['value'] [0],initCond['value'] [-1]]
                maxV = [initCond['max'][0],initCond['max'][-1] ]
                minV = [initCond['min'][0],initCond['min'][-1] ]
            else:
                
            #     initCond['value'] = float(RSP_rslt['vol'][noMd]) #value from the GRASP result for RSP
                val = initCond['value'] 
                maxV = initCond['max'] 
                minV = initCond['min'] 
            charInfo =  f'{typ} for mode {noMd+1} is {val} ,  ranging from  max: {maxV} to min : {minV} '
            Info.append(charInfo)
    
    #COmbining all the infroamtion into single sting
    combined_string = '-'

    for i in range (len(Info)):
        line = Info[i]
        combined_string = combined_string + "\n" + line

    return combined_string

def update_HSRLyaml(YamlFileName, noMod, maxr, minr, a, Kernel_type,ConsType,  RSP_rslt = None, LagrangianOnly = None, RSPUpdate= None, AprioriLagrange =None):
    '''HSRL provides information on the aerosol in each height,
      and RSP measures the total column intergrated values. We can use the total column information to bound the HSRL retrievals
    Set RSPUpdate= True to update the inital consdition from RSP retrievals
    Set LagrangianOnly = True to just cahnge the lagrangian for apriori in Vertical profile norm.
    
    
    '''
    #This function creates new yaml with initial conditions updated form microphysical properties of  Polarimeter retrievals

    # Load the YAML file for HSRL
    with open(YamlFileName, 'r') as f:  
        data = yaml.safe_load(f)

    YamlChar =[]  #This list stores the name of the charater types in the yaml files
    noYmlChar = np.arange(1,7) #No of aerosol characters types in the yaml file (This can be adjusted based on the parameters we want to change)
    
    for i in noYmlChar:
        YamlChar.append(data['retrieval']['constraints'][f'characteristic[{i}]']['type'])
    # RSP_rslt = np.load('RSP_sph.npy',allow_pickle= True).item()
    print(len(YamlChar))

    if LagrangianOnly == True:
        assert(AprioriLagrange != None)
        for noMd in range(noMod): #TODO assumed that vertical prof normalization is the first characteristics, needs to be more genratlized 
            initCond = data['retrieval']['constraints']['characteristic[1]'][f'mode[{noMd+a}]']['single_pixel']['a_priori_estimates']['lagrange_multiplier'] = float(AprioriLagrange),float(AprioriLagrange),float(AprioriLagrange),float(AprioriLagrange),float(AprioriLagrange),float(AprioriLagrange)
            print("Updating the lagrange_multiplier in a_priori_estimates")
                       
    if RSPUpdate== True:
        assert RSP_rslt != None
    #change the yaml intitial conditions using the RSP GRASP output
        for i in range(len(YamlChar)): #loop over the character types in the list
            for noMd in range(noMod): #loop over the aerosol modes (i.e 2 for fine and coarse)
                #State Varibles from yaml file: 
                # data['retrieval']['constraints'][f'characteristic[0]'][f'mode[{noMd+a}]']['value'][0] =  1e-8
                # data['retrieval']['constraints'][f'characteristic[0]'][f'mode[{noMd+a}]']['max'][0] =1e-7
                # data['retrieval']['constraints'][f'characteristic[0]'][f'mode[{noMd+a}]']['min'][0] =1e-9
                # #     # print("Updating",YamlChar[i])
                initCond = data['retrieval']['constraints'][f'characteristic[{i+a}]'][f'mode[{noMd+a}]']['initial_guess']
                # if YamlChar[i] == 'aerosol_concentration': #Update the in and max values from aerosol properties retrieved from the RSP measurements
                # #     # initCond['max'] = float(np.max(RSP_rslt['vol'][noMd])*maxr) #value from the GRASP result for RSP
                # #     # initCond['min'] = float(RSP_rslt['vol'][noMd]*minr)
                #     initCond['value'] = float(RSP_rslt['vol'][noMd]) #value from the GRASP result for RSP
                
                if YamlChar[i] == 'size_distribution_lognormal':
                    initCond['value'] = float(RSP_rslt['rv'][noMd]),float(RSP_rslt['sigma'][noMd])
                    initCond['max'] =float(RSP_rslt['rv'][noMd]*maxr),float(RSP_rslt['sigma'][noMd]*maxr)
                    initCond['min'] =float(RSP_rslt['rv'][noMd]*minr),float(RSP_rslt['sigma'][noMd]*minr)
                    # print("Updating",YamlChar[i])
                    if ConsType == 'strict': #This will set the retrieved parameteter to false. 
                        data['retrieval']['constraints'][f'characteristic[{i+a}]']['retrieved'] = 'false'
                
                if YamlChar[i] == 'real_part_of_refractive_index_spectral_dependent':
                    initCond['index_of_wavelength_involved'] = [1,2,3]
                    #Adding off sets because the HSRL abd RSP wavelengths dont match, this should be improved
                    if noMd==0: offs = 0  
                    if noMd==1: offs = 0

                    initCond['value'] =float(RSP_rslt['n'][noMd][0]+offs),float(RSP_rslt['n'][noMd][2]),float(RSP_rslt['n'][noMd][4])
                    initCond['max'] =float((RSP_rslt['n'][noMd][0]+offs)*maxr),float(RSP_rslt['n'][noMd][2]*maxr),float(RSP_rslt['n'][noMd][4]*maxr)
                    initCond['min'] =float((RSP_rslt['n'][noMd][0]+offs)*minr),float(RSP_rslt['n'][noMd][2]*minr),float(RSP_rslt['n'][noMd][4]*minr)
                    # print("Updating",YamlChar[i])
                    if ConsType == 'strict':  # tot set retrival in setting files to False
                        data['retrieval']['constraints'][f'characteristic[{i+a}]']['retrieved'] = 'false'
                
                if YamlChar[i] == 'imaginary_part_of_refractive_index_spectral_dependent':
                    initCond['index_of_wavelength_involved'] = [1,2,3]
                    initCond['value'] =float(RSP_rslt['k'][noMd][0]),float(RSP_rslt['k'][noMd][2]),float(RSP_rslt['k'][noMd][4])
                    initCond['max'] =float(RSP_rslt['k'][noMd][0]*maxr),float(RSP_rslt['k'][noMd][2]*maxr),float(RSP_rslt['k'][noMd][4]*maxr)
                    initCond['min'] = float(RSP_rslt['k'][noMd][0]*minr),float(RSP_rslt['k'][noMd][2]*minr),float(RSP_rslt['k'][noMd][4]*minr)
                    # print("Updating",YamlChar[i])
                    if ConsType == 'strict':
                        data['retrieval']['constraints'][f'characteristic[{i+a}]']['retrieved'] = 'false'
                
                if YamlChar[i] == 'sphere_fraction':
                
                    initCond['value'] = float(RSP_rslt['sph'][noMd]/100)
                    initCond['max'] =float(RSP_rslt['sph'][noMd] *maxr/100) #GARSP output is in %
                    initCond['min'] =float(RSP_rslt['sph'][noMd] *minr/100)
                    # print("Updating",YamlChar[i])
                    if ConsType == 'strict':
                        data['retrieval']['constraints'][f'characteristic[{i+a}]']['retrieved'] = 'false'
                


    if Kernel_type == "sphro":
        UpKerFile = 'Settings_Sphd_RSP_HSRL.yaml' #for spheroidal kernel
    if Kernel_type == "TAMU":
        UpKerFile = 'Settings_TAMU_RSP_HSRL.yaml'#for hexahedral kernel
    
    ymlPath = '/home/gregmi/git/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/'
    
    with open(ymlPath+UpKerFile, 'w') as f: #write the chnages to new yaml file
        yaml.safe_dump(data, f)
        
    return 

#Find the pixel index for nearest lat and lon for given LatH and LonH
def FindPix(LatH,LonH,Lat,Lon):

    # Assuming Lat, latH, Lon, and LonM are all NumPy arrays
    diffLat = np.abs(LatH - Lat) # Find the absolute difference between `Lat` and each element in `latH`
    indexLat = np.argwhere(diffLat == diffLat.min())[0] # Find the indices of all elements that minimize the difference

    diffLon = np.abs(LonH - Lon) # Find the absolute difference between `Lon` and each element in `LonM`
    indexLon = np.argwhere(diffLon == diffLon.min())[0] # Find the indices of all elements that minimize the difference
    return indexLat[0], indexLat[1]

def FMF(AEa): 
    
    '''caculating the fine mode AE from Supplement of 
    A global land aerosol fine-mode fraction dataset (2001–2020) 
    retrieved from MODIS using hybrid physical and deep learning approaches
    Xing Yan et al.
    '''
    AEwl = 0.532
    
    AEc =  -0.2  #-0.15  #coarse mode AE
    dAEc = 0     #spectral derivative of coarse mode AE, which is set to 0 under the assumption that coarse mode AOD is constant in visible

    aLow = -0.22 
    aUpp = -0.3
    a = (aLow - aUpp)/2


    bLow = 0.8
    bUpp = 10**-0.2388 * (AEwl)*1.0275
    b = (bLow +bUpp)/2
    bs = b+2*AEc*a


    cUpp = 10**-0.2633 * (AEwl)*-0.4683
    cLow = 0.63
    c = (cLow +cUpp)/2
    cs = c+(b+a*AEc)*AEc - dAEc
    
    dwl = 0.532-0.355

    dAE = np.diff(AEa,prepend=0)[:]
    
    dAEa =dAE/dwl
    

    t = AEa - AEc - ((dAEa- dAEc)/(AEa- AEc)) +bs
    
    AEf = 1/(2*(1-a)) * (t + np.sqrt(t**2-4 * cs*(1-a))) +AEc
    
    FMF = (AEa -AEc)/(AEf -AEc)
    
    
    #Bias and Correction:
    
    dAEerr = 0.65 *np.exp(-(FMF-0.78)**2 / (2*0.18**2))
    dAEa_corr = dAEa+dAEerr
    
    tCorr = AEa - AEc- ((dAEa_corr-dAEc)/(AEa-AEc))
    Dcorr = np.sqrt((tCorr+bs)**2 +4*(1-a)*cs)
    
    AEfCorr = 1/(2*(1-a)) *(tCorr+bs+Dcorr) +AEc
    
    FMFCorr = (AEa - AEc)/(AEfCorr-AEc)
    
    #More correction 

    
    return AEf, FMF, AEfCorr,FMFCorr

def find_dust(HSRLfile_path, HSRLfile_name, plot=None):
    # Open the HDF5 file in read mode
    f1 = h5py.File(HSRLfile_path + HSRLfile_name, 'r+')
    
    # Extract the Aerosol_ID data product
    Dust_pix = f1['DataProducts']['Aerosol_ID']
    
    # Create an empty list to store indices of dust pixels for each column
    dust_pixel = []
    
    # Loop over the columns in Dust_pix
    for i in range(Dust_pix.shape[1]):
        # Get the indices where the pixel value is 8 (dust)
        dust_pixel.append(np.where(Dust_pix[:, i] == 8)[0])

    # Concatenate the arrays along the first axis (rows)
    concatenated_array = np.concatenate(dust_pixel, axis=0)
    # Flatten the concatenated array to a 1D array
    all_dust_pixels = concatenated_array.flatten()
    
    # Find the unique values and their frequency counts in the flattened dust pixel array
    unique_values, counts = np.unique(all_dust_pixels, return_counts=True)
    
    # Filter out the dust pixel values where frequency count is less than 100
    dust_pix = unique_values[counts > 100]
    # Find the dust pixel value(s) with the highest frequency count
    max_dust = unique_values[counts == counts.max()]
    
    # If plot is True, create and display plots
    if plot == True:
        # Plot a bar diagram showing the frequency count of each dust pixel value
        plt.figure(figsize=(15,5))
        plt.bar(unique_values, counts)
        plt.xlabel('Values')
        plt.ylabel('Frequency')
        plt.show()
        
        # Create a contour plot of the Aerosol_ID data product and plot the dust pixel indices on it
        fig, ax = plt.subplots()
        c = ax.contourf(f1['DataProducts']['Aerosol_ID'][:].T, cmap='tab20b')
        ax.scatter(dust_pix, np.repeat((0), len(dust_pix)), c="k")
        plt.colorbar(c)
    
    # Close the HDF5 file
    f1.close()
    
    # Return the filtered dust pixel values and the dust pixel value(s) with the highest frequency count
    return dust_pix, max_dust

def AeroProfNorm_FMF(DictHsrl):
    '''Scheme 1: Modifided scheme April 2024: This function divides the total lidar extinction to contribution from dust, sea salt and fine mode
     This scheme is based on using Dust mixing ratio(from DP 532 nm) and FIne mode fraction from AE(355_532)  '''
# if VertProfConstrain == True: #True of we want to apply vertical profile normalization
    InstruNoise = 1e-6 #instrument's uncertainity
    
    rslt = DictHsrl[0] #Reading the rslt dictionary 
    hgt =  rslt['RangeLidar'][:,0][:] #Height
    Vext1 = rslt['meas_VExt'][:,0]  #m-1

 
    #Dust mixing ratio  using the Sugimoto and Lee, Appl Opt 2006"
    DP1064= rslt['meas_DP'][:,2][:]  #Depol at 1064 nm
    # aDP= DictHsrl[3] #Particle depol ratio at 532
    # maxDP =  np.nanmax(aDP)
    # DMR1 = ((1+maxDP) * aDP)/(maxDP*(1+aDP)) #dust mixing ratio from paper
    
    DMR1 = DictHsrl[2]
   
    #Some values of DMR are >1,  in such cases we recaculate the values. Otherwise, use the values reported in the HSRL products
    if np.any(DMR1 > 1): #If there is any pure dust event

        warnings.warn('DMR > 1, Recaculating', UserWarning)
        DP1064= rslt['meas_DP'][:,2][:]  #Depol at 1064 nm
        aDP= DictHsrl[3] #Particle depol ratio at 532
        maxDP =  np.nanmax(aDP)
        DMR1 = ((1+maxDP) * aDP)/(maxDP*(1+aDP)) #dust mixing ratio from paper
        
    else:
        # maxDP =  0.35 #From the paper
        # aDP= DictHsrl[3]
        DMR1 = DictHsrl[2]
        # maxDP =  np.nanmax(aDP)
        # print('No pure dustEvent') 
    
    
    #Separating the contribution of dust to total extinction
    VextDst = DMR1 * Vext1
    VextDst [np.where(VextDst <=0)[0]]= 1e-8
    Vextoth = (1-DMR1)* Vext1

    print(DMR1)


    #Separating the contribution of non-dust

    # AEt= DictHsrl[5] #total angstrom exponent
    # AEsph= DictHsrl[3] #spherical angstrom exp
    
    #Boundary layer height
    BLH_indx = np.where(np.gradient(DP1064,hgt) == np.max(np.gradient(DP1064,hgt)))[0]
    BLH = hgt[np.where(np.gradient(DP1064,hgt) == np.max(np.gradient(DP1064,hgt)))]

    #Caculating the fine mode fraction
    # FMF = AEt/AEsph #assumption: angstrom exponent for evry large particle is very small
    # AEt = DictHsrl[4]
    # FMF1 = FMF(AEt)[3]
    # FMFv[np.where(np.isnan(FMFv))[0]]= 0

    FMF1 = DictHsrl[4]

    
    #Filtering
    FMF1[np.where(FMF1<0)[0] ]= 0.1
    FMF1[np.where(FMF1>=1)[0]]= 0.9
    # print(FMF)
    
    #Spearating the contribution from fine mode 
    Vextfine = FMF1* Vextoth

    Vextfine[np.where(Vextfine <=0)[0]]= 1e-8
    Vextfine[np.where(hgt<BLH)[0]]= 1e-8

    # Vextfine[np.where(hgt>=BLH)[0]] = Vextoth[np.where(hgt>=BLH)[0]]
    
    # Vextfine[np.where(hgt<BLH)[0]]= Vextfine[np.where(hgt<BLH)[0]] 
    # if BLH>1000:
    #     FMFBelowBL = np.nanmean(FMF[np.where(hgt<BLH)[0]])
    #     Vextfine[np.where(hgt<BLH)[0]]= FMFBelowBL*Vextoth[np.where(hgt<BLH)[0]]  

    #Contribution from sea salt = ext from non dust - fine
    VextSea = Vextoth - Vextfine
    # VextSea[np.where(hgt<BLH)[0]] = Vextoth[np.where(hgt<BLH)[0]]
    VextSea[np.where(hgt>=BLH)[0]] = 1e-10
    VextSea[np.where(VextSea <=0)[0]]= 1e-8

    # VextDst[0] = 1e-9
    # Vextfine[0] = 1e-9
    # VextSea[0] = 1e-9

    #assuming that sea salt is limited within the boundary layer


    #Normalizing such that int(Vext.dh) = 1. This is equivalent of using vol conc
    
    DstProfext =VextDst/ np.trapz(VextDst[::-1],hgt[::-1])
    FineProfext = Vextfine/np.trapz(Vextfine[::-1],hgt[::-1])
    SeaProfext = VextSea/ np.trapz(VextSea[::-1],hgt[::-1])
    

    #Plotting the extinction from each aerosol
    fig,ax = plt.subplots(1,1, figsize=(6,6))
    ax.plot(VextDst,hgt, color = '#067084',label='Dust')
    ax.plot(VextSea,hgt,color ='#6e526b',label='Salt')
    ax.plot(Vextfine,hgt,color ='y',label='fine')
    ax.plot(Vext1,hgt,color='#8a9042',ls = '--',label='Total Ext')
    ax.plot(Vext1[BLH_indx],hgt[BLH_indx],color='#660000',marker = 'x',label='BLH')
    ax.set_xlabel('VExt m-1')
    ax.set_ylabel('Altitude m')
    plt.legend()
    plt.savefig('Vert.png', dpi = 300)
    

    fig = plt.figure()
    plt.scatter(FineProfext,hgt,color ='y',label='fine')
    plt.scatter(SeaProfext,hgt,color ='#6e526b',label='Salt')
    plt.scatter(DstProfext,hgt, color = '#067084',label='Dust')
    plt.legend()

    return FineProfext,DstProfext,SeaProfext


def AeroClassAOD(DictHsrl):

    "To provide the a priori constrain to GRASP. "

    "seperates the contribution of dust, fine and marine aerosols and caculates the AOD contribtuion form each aerosol type"
    rslt = DictHsrl[0] # Read the result dictionary with information on the DP at 1064, angstrong exponent
    Wl = [355,532] #Wavelengths 
    Aod_Classified = {}
    for rep in range(2):
        
        print(Wl[rep])

        Vext1 = rslt['meas_VExt'][:,rep]
        Vext1[np.where(Vext1<=0)] = 1e-10

        
        hgt =  rslt['RangeLidar'][:,0][:]
        DP1064= rslt['meas_DP'][:,2][:]

        # Calculating the boundary layer height
        BLH_indx = np.where(np.gradient(DP1064,hgt) == np.max(np.gradient(DP1064,hgt)))[0]
        BLH = hgt[np.where(np.gradient(DP1064,hgt) == np.max(np.gradient(DP1064,hgt)))]
        #altitude of the aircraft99
        AerID = DictHsrl[5]

        DMR1 = DictHsrl[2]
        DMR1[DMR1==0] = 0.2
        
        # DMR1[DMR1==0] = np.linspace(indxMaxDMR,0,lnDMR)

        #Some values of DMR are >1,  in such cases we recaculate the values. Otherwise, use the values reported in the HSRL products
        if np.any(DMR1 > 1) or np.any(AerID == 8 ) : #If there is any pure dust event

            warnings.warn('DMR > 1, Recaculating', UserWarning)
            DP1064= rslt['meas_DP'][:,2][:]  #Depol at 1064 nm
            aDP= DictHsrl[3] #Particle depol ratio at 532
            maxDP =  np.nanmax(aDP)
            DMR1 = ((1+maxDP) * aDP)/(maxDP*(1+aDP)) #dust mixing ratio from paper
            # DMR1[DMR1==0] = 0.0001
        else:
            # maxDP =  0.35 #From the paper
            # aDP= DictHsrl[3]
            DMR1 = DictHsrl[2]
            # maxDP =  np.nanmax(aDP)
            # print('No pure dustEvent')

        Vext1[np.where(Vext1<=0)] = 1e-10
        Vextoth = (1-DMR1)*Vext1
        VextDst = Vext1 - Vextoth 
        FMF1 = DictHsrl[4]

        
        #Filtering
        FMF1[np.where(FMF1<0)[0] ]= 0.00001
        FMF1[np.where(FMF1>=1)[0]]= 0.99
        # print(FMF)
        
        #Spearating the contribution from fine mode 
        VextFMF = FMF1* Vextoth
        aboveBL = Vextoth[:BLH_indx[0]]

        belowBL = Vextoth[BLH_indx[0]:]

        print(BLH_indx)

        if BLH< 1000: #if the boundary layer is low then assume the fine mode aerosol to be well mixed
            Vextfine = np.concatenate((aboveBL, np.ones(len(belowBL))*10**-6))
        else:
            belowBL = VextFMF[BLH_indx[0]:]
            Vextfine = np.concatenate((aboveBL,belowBL))

        # belowBL = VextFMF[BLH_indx[0]:] #Fine mode fractuon below BLH calchuateing using aeVoth/aesph
        # Vextfine = np.concatenate((aboveBL,belowBL))
        

        # Vextfine = np.concatenate((aboveBL, np.ones(len(belowBL))*10**-6))
        VextSea = Vextoth -Vextfine

        VextDst[np.where(VextDst<=0)[0]] = 1e-10
        Vextfine[np.where(Vextfine<=0)[0]] = 1e-10
        VextSea[np.where(VextSea<=0)[0]] = 1e-10
        VextSea[:BLH_indx[0]] = 1e-10

        print(Wl[rep])
          #stores aerosol contribution for each mode based on the classification. 
        # Plot the various line plots on the secondary x-axis
        Aod_Classified[f'Aod_dst_{Wl[rep]}'] = np.trapz(VextDst[::-1],hgt[::-1])
        Aod_Classified[f'Aod_sea_{Wl[rep]}'] = np.trapz(VextSea[::-1],hgt[::-1])
        Aod_Classified[f'Aod_fine_{Wl[rep]}'] = np.trapz(Vextfine[::-1],hgt[::-1])
        Aod_Classified[f'Aod_T_{Wl[rep]}'] = np.trapz(Vext1[::-1], hgt[::-1])
        Aod_Classified[f'Aod_oth_{Wl[rep]}'] =np.trapz(Vextoth [::-1], hgt[::-1])
        Aod_Classified[f'Aod_below_BLH_{Wl[rep]}'] = np.trapz(Vext1[BLH_indx[0]:][::-1], hgt[BLH_indx[0]:][::-1])
        Aod_Classified[f'Aod_above_BLH_{Wl[rep]}'] = np.trapz(Vext1[:BLH_indx[0]][::-1], hgt[:BLH_indx[0]][::-1])
        
        print(hgt[BLH_indx[0]:], 'below the boundary layer')

    print(Aod_Classified)

   
    return Aod_Classified


def AeroProfNorm_sc2(DictHsrl):

    """
    
    Calculates the vertical distribution of different aerosol types. 
    This sceme is more simple as it just based on DMR from HSRL data products
    
    
    """

    rslt = DictHsrl[0]
    Idx1064 = np.where(rslt['lambda'] == 1064/1000)[0][0]
    Idx532 = np.where(rslt['lambda'] == 532/1000)[0][0]
    # print(Idx1064,Idx532 )
    # Vext1 = (rslt['meas_VExt'][:,0]+rslt['meas_VExt'][:,1])/2
    Vext1 = rslt['meas_VExt'][:,0]   #Extinvtion at 355
    Vext1[np.where(Vext1<=0)] = 1e-10
 
    hgt =  rslt['RangeLidar'][:,0][:]
    DP1064= rslt['meas_DP'][:,Idx1064][:]

    # Calculating the boundary layer height
    BLH_indx = np.where(np.gradient(DP1064,hgt) == np.max(np.gradient(DP1064,hgt)))[0]
    BLH = hgt[np.where(np.gradient(DP1064,hgt) == np.max(np.gradient(DP1064,hgt)))]
    #altitude of the aircraft99
    AerID = DictHsrl[5]

    DMR1 = DictHsrl[2]


#Aerosols classification from burton et al 
    ExtToBckScaRatio = rslt['meas_VExt'][:,1]/rslt['meas_VBS'][:,1]
    BckColorRatio = rslt['meas_VBS'][:,1]/rslt['meas_VBS'][:,2]
    aDP532= DictHsrl[3]
    aDP1064 = DictHsrl[7]
    ratioDP = aDP1064/aDP532


  

    # Example data
    # ExtToBckScaRatio = [...]
    # BckColorRatio = [...]
    # aDP532 = [...]
    # ratioDP = [...]
    # hgt = [...]  # same length as the other arrays

    fig, axs5 = plt.subplots(2, 2, figsize=(12, 9))

    # Common scatter plot settings
    sc_kwargs = dict(c=hgt, cmap='vanimo', edgecolor='k')

    # First subplot
    sc0 = axs5[0, 0].scatter(ExtToBckScaRatio, BckColorRatio, **sc_kwargs)
    axs5[0, 0].set_xlabel('ExtToBckScaRatio')
    axs5[0, 0].set_ylabel('BckColorRatio')
    axs5[0, 0].set_title('ExtToBckScaRatio vs BckColorRatio')

    # Second subplot
    sc1 = axs5[0, 1].scatter(ExtToBckScaRatio, aDP532, **sc_kwargs)
    axs5[0, 1].set_xlabel('ExtToBckScaRatio')
    axs5[0, 1].set_ylabel('aDP532')
    axs5[0, 1].set_title('ExtToBckScaRatio vs aDP532')

    # Third subplot
    sc2 = axs5[1, 0].scatter(ExtToBckScaRatio, ratioDP, **sc_kwargs)
    axs5[1, 0].set_xlabel('ExtToBckScaRatio')
    axs5[1, 0].set_ylabel('ratioDP')
    axs5[1, 0].set_title('ExtToBckScaRatio vs ratioDP')

    # Fourth subplot
    sc3 = axs5[1, 1].scatter(ratioDP, BckColorRatio, **sc_kwargs)
    axs5[1, 1].set_xlabel('ratioDP')
    axs5[1, 1].set_ylabel('BckColorRatio')
    axs5[1, 1].set_title('ratioDP vs BckColorRatio')

    # Adjust layout to make space for colorbar
    fig.tight_layout(rect=[0, 0, 0.95, 1])

    # Add colorbar on the side (right)
    cbar = fig.colorbar(sc3, ax=axs5.ravel().tolist(), shrink=0.9, label='Height (hgt)', pad=0.02)

    plt.show()


    # indxMaxDMR = (np.max(np.where(DMR1==0)[0])) 

    #Some values of DMR are >1,  in such cases we recaculate the values. Otherwise, use the values reported in the HSRL products
    if np.any(DMR1 > 1): #If there is any pure dust event

        warnings.warn('DMR > 1, Recaculating', UserWarning)
        DP1064= rslt['meas_DP'][:,Idx1064][:]  #Depol at 1064 nm
        aDP= DictHsrl[3] #Particle depol ratio at 532
        maxDP =  np.nanmax(aDP[:])
        DMR1 = ((1+maxDP) * aDP)/(maxDP*(1+aDP)) #dust mixing ratio from paper

        



  
        # DMR1 = DictHsrl[2]
    else:
        # maxDP =  0.35 #From the paper
        # aDP= DictHsrl[3]
        DMR1 = DictHsrl[2]
        # maxDP =  np.nanmax(aDP)
        # print('No pure dustEvent')

    # TODO Hard coded-> This can be commented out for more general case. Used this as a test to see if the fine mode overestimation can be solved. This should be changed........
    # DMR1[DMR1>=1] = 1
    # DMR1[0] = 1
    # TODO This should be changed........
    print(DMR1)

    DMR1[np.where((aDP<0.2)&(ExtToBckScaRatio<35))[0]] = 0.01 #Seasalt/maritime from the paper

    print(DMR1)

    # DMR1[DMR1 ==0]= 0.01

    Vext1[np.where(Vext1<=0)] = 1e-12
    Vextoth = (1-DMR1)*Vext1
    VextDst = Vext1 - Vextoth 
    FMF1 = DictHsrl[4]    
    #Filtering
    FMF1[np.where(FMF1<0)[0] ]= 0.00001
    FMF1[np.where(FMF1>=1)[0]]= 0.99999
    
    #Spearating the contribution from fine mode 
    VextFMF = FMF1* Vextoth
    aboveBL = Vextoth[:BLH_indx[0]]
    belowBL = Vextoth[BLH_indx[0]:]
    print(BLH_indx)

    if BLH< 1000: #if the boundary layer is low then assume the fine mode aerosol to be well mixed
        Vextfine = np.concatenate((aboveBL, np.ones(len(belowBL))*10**-10))
    else:
        belowBL = VextFMF[BLH_indx[0]:]
        Vextfine = np.concatenate((aboveBL,belowBL))

    # Vextfine = np.concatenate((aboveBL, np.ones(len(belowBL))*10**-6))
    VextSea = Vextoth - Vextfine
    VextSea[:BLH_indx[0]] = 1e-20
    VextDst[np.where(VextDst<=0)[0]] = 1e-10
    Vextfine[np.where(Vextfine<=0)[0]] = 1e-10
    VextSea[np.where(VextSea<=0)[0]] = 1e-10
    VextSea[:BLH_indx[0]] = 1e-10


    Aod_Classified = {}  #stores aerosol contribution for each mode based on the classification. 
    # Plot the various line plots on the secondary x-axis
    Aod_Classified['Aod_dst'] = np.trapz(VextDst[::-1],hgt[::-1])
    Aod_Classified['Aod_sea'] = np.trapz(VextSea[::-1],hgt[::-1])
    Aod_Classified['Aod_fine'] = np.trapz(Vextfine[::-1],hgt[::-1])
    Aod_Classified['Aod_T'] = np.trapz(Vext1[::-1], hgt[::-1])

    Aod_Classified['Aod_below_BLH'] = np.trapz(Vext1[BLH_indx[0]:][::-1], hgt[BLH_indx[0]:][::-1])
    Aod_Classified['Aod_above_BLH'] = np.trapz(Vext1[BLH_indx[0]:][::-1], hgt[BLH_indx[0]:][::-1])

    DstProf =VextDst/ np.trapz(VextDst[::-1],hgt[::-1])
    FineProf = Vextfine/np.trapz(Vextfine[::-1],hgt[::-1])
    SeaProf = VextSea/ np.trapz(VextSea[::-1],hgt[::-1])
    # Temp = DictHsrl[5]

    fig = plt.figure()

    # Define the vmin and vmax for the color scale
    vmin = 290
    vmax = 300

    # Create subplots
    fig, ax = plt.subplots(1, 2, figsize=(12, 6), sharey = True)

    plt.plot(VextDst,hgt, color = '#067084',label='Dust')
    plt.plot(VextSea,hgt,color ='#6e526b',label='Salt')
    plt.plot(Vextfine,hgt,color ='y',label='fine')
    plt.plot(Vext1,hgt,color='#8a9042',ls = '--',label='Total Ext')
    plt.plot(Vext1[BLH_indx],hgt[BLH_indx],color='#660000',marker = 'x',label='BLH')
    plt.legend()

    fig = plt.figure()
    plt.plot(FineProf,hgt,color ='y',label='fine')
    plt.plot(SeaProf,hgt,color ='#6e526b',label='Salt')
    plt.plot(DstProf,hgt, color = '#067084',label='Dust')
    plt.legend()


    return FineProf,DstProf,SeaProf,Aod_Classified




def ModAeroProfNorm_sc2(DictHsrl):

    """

    Modified to make DMR 1 at top of the profile.
    This sceme is more simple as just based on DMR from HSRL data products

    """

    rslt = DictHsrl[0]
    Idx1064 = np.where(rslt['lambda'] == 1064/1000)[0][0]
    Idx532 = np.where(rslt['lambda'] == 532/1000)[0][0]

    max_alt = rslt['OBS_hght']
    Vext1 = rslt['meas_VExt'][:,0]
    Vext1[np.where(Vext1<=0)] = 1e-10
    
    hgt =  rslt['RangeLidar'][:,0][:]
    DP1064= rslt['meas_DP'][:,Idx1064][:]

    # Calculating the boundary layer height
    BLH_indx = np.where(np.gradient(DP1064,hgt) == np.max(np.gradient(DP1064,hgt)))[0]
    BLH = hgt[np.where(np.gradient(DP1064,hgt) == np.max(np.gradient(DP1064,hgt)))]
    #altitude of the aircraft99
    AerID = DictHsrl[5]

    DMR1 = DictHsrl[2]
    indxMaxDMR = (np.max(np.where(DMR1==0)[0])) 
    # DMR1[DMR1==0] = 0.05
    # DMR1[hgt[np.where(hgt>4000)]] = 1
   
    #Some values of DMR are >1,  in such cases we recaculate the values. Otherwise, use the values reported in the HSRL products
    
    if np.any(DMR1 > 1) or np.any(AerID == 8 ) : #If there is any pure dust event
        warnings.warn('DMR > 1, Recaculating', UserWarning)
        DP1064= rslt['meas_DP'][:,Idx1064][:]  # Depol at 1064 nm
        aDP= DictHsrl[3]                       # Particle depol ratio at 532
        maxDP =  np.nanmax(aDP)
        DMR1 = ((1+maxDP) * aDP)/(maxDP*(1+aDP)) #dust mixing ratio from paper
        
        
        DMR1 = DictHsrl[2]
    else:
        # maxDP =  0.35 #From the paper
        # aDP= DictHsrl[3]
        DMR1 = DictHsrl[2]
        # maxDP =  np.nanmax(aDP)
        # print('No pure dustEvent')

    Vext1[np.where(Vext1<=0)] = 1e-10
    Vextoth = (1-DMR1)*Vext1
    VextDst = Vext1 - Vextoth 
    FMF1 = DictHsrl[4]

    #Filtering
    FMF1[np.where(FMF1<0)[0] ]= 0.00001
    FMF1[np.where(FMF1>=1)[0]]= 0.99
    # print(FMF)
    
    #Spearating the contribution from fine mode 
    VextFMF = FMF1* Vextoth
    aboveBL = Vextoth[:BLH_indx[0]]
    belowBL = Vextoth[BLH_indx[0]:]
    belowBL = 10**-5

    print(BLH_indx)

    if BLH< 1000: #if the boundary layer is low then assume the fine mode aerosol to be well mixed
        Vextfine = np.concatenate((aboveBL, np.ones(len(belowBL))*10**-5))
    else:
        belowBL = VextFMF[BLH_indx[0]:]
        Vextfine = np.concatenate((aboveBL,belowBL))

    # belowBL = VextFMF[BLH_indx[0]:] #Fine mode fractuon below BLH calchuateing using aeVoth/aesph
    # Vextfine = np.concatenate((aboveBL,belowBL))
    

    # Vextfine = np.concatenate((aboveBL, np.ones(len(belowBL))*10**-6))
    VextSea = Vextoth - Vextfine
    VextSea[:BLH_indx[0]] = 1e-20


    VextDst[np.where(VextDst<=0)[0]] = 1e-10
    Vextfine[np.where(Vextfine<=0)[0]] = 1e-10
    VextSea[np.where(VextSea<=0)[0]] = 1e-10
    VextSea[:BLH_indx[0]] = 1e-10


    Aod_Classified = {}  #stores aerosol contribution for each mode based on the classification. 
    # Plot the various line plots on the secondary x-axis
    Aod_Classified['Aod_dst'] = np.trapz(VextDst[::-1],hgt[::-1])
    Aod_Classified['Aod_sea'] = np.trapz(VextSea[::-1],hgt[::-1])
    Aod_Classified['Aod_fine'] = np.trapz(Vextfine[::-1],hgt[::-1])
    Aod_Classified['Aod_T'] = np.trapz(Vext1[::-1], hgt[::-1])

    Aod_Classified['Aod_below_BLH'] = np.trapz(Vext1[BLH_indx[0]:][::-1], hgt[BLH_indx[0]:][::-1])
    Aod_Classified['Aod_above_BLH'] = np.trapz(Vext1[BLH_indx[0]:][::-1], hgt[BLH_indx[0]:][::-1])

    DstProf =VextDst/ np.trapz(VextDst[::-1],hgt[::-1])
    FineProf = Vextfine/np.trapz(Vextfine[::-1],hgt[::-1])
    SeaProf = VextSea/ np.trapz(VextSea[::-1],hgt[::-1])
    # Temp = DictHsrl[5]

    fig = plt.figure()

    # Define the vmin and vmax for the color scale
    vmin = 290
    vmax = 300

    # Create subplots
    fig, ax = plt.subplots(1, 2, figsize=(12, 6), sharey = True)
    plt.plot(VextDst,hgt, color = '#067084',label='Dust')
    plt.plot(VextSea,hgt,color ='#6e526b',label='Salt')
    plt.plot(Vextfine,hgt,color ='y',label='fine')
    plt.plot(Vext1,hgt,color='#8a9042',ls = '--',label='Total Ext')
    plt.plot(Vext1[BLH_indx],hgt[BLH_indx],color='#660000',marker = 'x',label='BLH')
    plt.legend()

    fig = plt.figure()
    plt.plot(FineProf,hgt,color ='y',label='fine')
    plt.plot(SeaProf,hgt,color ='#6e526b',label='Salt')
    plt.plot(DstProf,hgt, color = '#067084',label='Dust')
    plt.legend()


    return FineProf,DstProf,SeaProf,Aod_Classified





def ModAeroProfNorm_sc3(DictHsrl):

    """Modified to make DMR 1 at top of the profile. This sceme is more simple as just based on DMR from HSRL data products"""

    rslt = DictHsrl[0]
    Idx1064 = np.where(rslt['lambda'] == 1064/1000)[0][0]
    Idx532 = np.where(rslt['lambda'] == 532/1000)[0][0]
    # print(Idx1064,Idx532 )


    max_alt = rslt['OBS_hght']
    # Vext1 = (rslt['meas_VExt'][:,0]+rslt['meas_VExt'][:,1])/2
    Vext1 = rslt['meas_VExt'][:,0]
    Vext1[np.where(Vext1<=0)] = 1e-10

    
    hgt =  rslt['RangeLidar'][:,0][:]
    DP1064= rslt['meas_DP'][:,Idx1064][:]

    # Calculating the boundary layer height
    BLH_indx = np.where(np.gradient(DP1064,hgt) == np.max(np.gradient(DP1064,hgt)))[0]
    BLH = hgt[np.where(np.gradient(DP1064,hgt) == np.max(np.gradient(DP1064,hgt)))]
    #altitude of the aircraft99
    AerID = DictHsrl[5]
    DMR1 = DictHsrl[2]



    # indxMaxDMR = (np.max(np.where(DMR1==0)[0])) 
    DMR1[DMR1==0] = 0.05
    # DMR1[hgt[np.where(hgt>4000)]] = 1



    
    # DMR1[DMR1==0] = np.linspace(indxMaxDMR,0,lnDMR)


   
    #Some values of DMR are >1,  in such cases we recaculate the values. Otherwise, use the values reported in the HSRL products
    if np.any(DMR1 > 1) or np.any(AerID == 8 ) : #If there is any pure dust event

        warnings.warn('DMR > 1, Recaculating', UserWarning)
        DP1064= rslt['meas_DP'][:,Idx1064][:]  #Depol at 1064 nm
        aDP= DictHsrl[3] #Particle depol ratio at 532



        maxDP =  np.nanmax(aDP)
        DMR1 = ((1+maxDP) * aDP)/(maxDP*(1+aDP)) #dust mixing ratio from paper
        # DMR1[DMR1==0] = 0.0001
    else:
        # maxDP =  0.35 #From the paper
        # aDP= DictHsrl[3]
        DMR1 = DictHsrl[2]
        # maxDP =  np.nanmax(aDP)
        # print('No pure dustEvent')

    Vext1[np.where(Vext1<=0)] = 1e-10
    Vextoth = (1-DMR1)*Vext1
    VextDst = Vext1 - Vextoth 
    FMF1 = DictHsrl[4]

    
    #Filtering
    FMF1[np.where(FMF1<0)[0] ]= 0.00001
    FMF1[np.where(FMF1>=1)[0]]= 0.99
    # print(FMF)
    
    #Spearating the contribution from fine mode 
    VextFMF = FMF1* Vextoth
    aboveBL = Vextoth[:BLH_indx[0]]

    belowBL = Vextoth[BLH_indx[0]:]


    belowBL = 10**-5

    print(BLH_indx)

    if BLH< 1000: #if the boundary layer is low then assume the fine mode aerosol to be well mixed
        Vextfine = np.concatenate((aboveBL, np.ones(len(belowBL))*10**-5))
    else:
        belowBL = VextFMF[BLH_indx[0]:]
        Vextfine = np.concatenate((aboveBL,belowBL))

    # belowBL = VextFMF[BLH_indx[0]:] #Fine mode fractuon below BLH calchuateing using aeVoth/aesph
    # Vextfine = np.concatenate((aboveBL,belowBL))
    

    # Vextfine = np.concatenate((aboveBL, np.ones(len(belowBL))*10**-6))
    VextSea = Vextoth - Vextfine
    VextSea[:BLH_indx[0]] = 1e-20


    VextDst[np.where(VextDst<=0)[0]] = 1e-10
    Vextfine[np.where(Vextfine<=0)[0]] = 1e-10
    VextSea[np.where(VextSea<=0)[0]] = 1e-10
    VextSea[:BLH_indx[0]] = 1e-10


    Aod_Classified = {}  #stores aerosol contribution for each mode based on the classification. 
    # Plot the various line plots on the secondary x-axis
    Aod_Classified['Aod_dst'] = np.trapz(VextDst[::-1],hgt[::-1])
    Aod_Classified['Aod_sea'] = np.trapz(VextSea[::-1],hgt[::-1])
    Aod_Classified['Aod_fine'] = np.trapz(Vextfine[::-1],hgt[::-1])
    Aod_Classified['Aod_T'] = np.trapz(Vext1[::-1], hgt[::-1])

    Aod_Classified['Aod_below_BLH'] = np.trapz(Vext1[BLH_indx[0]:][::-1], hgt[BLH_indx[0]:][::-1])
    Aod_Classified['Aod_above_BLH'] = np.trapz(Vext1[BLH_indx[0]:][::-1], hgt[BLH_indx[0]:][::-1])
    
    #Aerosol classification for 355
    # Vext355 = rslt['meas_VExt'][:,0]




    # print(hgt[BLH_indx[0]:], 'below the boundary layer')


    # Added = Aod_Classified['Aod_dst']+Aod_Classified['Aod_sea']+Aod_Classified['Aod_fine']

    # print('Aod_dst %:', round(100*(Aod_Classified['Aod_dst']/Aod_Classified['Aod_T']),3),'Aod_sea %:',round(100*(Aod_Classified['Aod_sea']/Aod_Classified['Aod_T']),3),'Aod_fine%:',round(100*(Aod_Classified['Aod_fine']/Aod_Classified['Aod_T']),3) )
    
   
    # VextDst[0] = 1e-10
    # Vextfine[0] = 1e-10
    # VextSea[0] = 1e-10

    # VBack = 0.00002*Vextoth
    # Voth = 0.999998*Vextoth
    # VextSea = np.concatenate((VBack[:BLH_indx[0]],Voth[BLH_indx[0]:]))
    # Vextfine =np.concatenate((Voth[:BLH_indx[0]],VBack[BLH_indx[0]:]))

    DstProf =VextDst/ np.trapz(VextDst[::-1],hgt[::-1])
    FineProf = Vextfine/np.trapz(Vextfine[::-1],hgt[::-1])
    SeaProf = VextSea/ np.trapz(VextSea[::-1],hgt[::-1])
    # Temp = DictHsrl[5]

    fig = plt.figure()

    # Define the vmin and vmax for the color scale
    vmin = 290
    vmax = 300

    # Create subplots
    fig, ax = plt.subplots(1, 2, figsize=(12, 6), sharey = True)

    # Plot the first contour plot on the first subplot
   
    # Overlay the second contour plot on the second subplot
    # contour2 = ax[1].contourf(Temp[HSRLPixNo-300:HSRLPixNo+300, :].T, 
    #                         aspect='auto', origin='lower', 
    #                         cmap='rainbow', alpha=1, 
    #                         vmin=vmin, vmax=vmax, 
    #                         levels=np.linspace(vmin, vmax, 20))
    # fig.colorbar(contour2, ax=ax[1], orientation='vertical')
    



    plt.plot(VextDst,hgt, color = '#067084',label='Dust')
    plt.plot(VextSea,hgt,color ='#6e526b',label='Salt')
    plt.plot(Vextfine,hgt,color ='y',label='fine')
    plt.plot(Vext1,hgt,color='#8a9042',ls = '--',label='Total Ext')
    plt.plot(Vext1[BLH_indx],hgt[BLH_indx],color='#660000',marker = 'x',label='BLH')
    plt.legend()

    fig = plt.figure()
    plt.plot(FineProf,hgt,color ='y',label='fine')
    plt.plot(SeaProf,hgt,color ='#6e526b',label='Salt')
    plt.plot(DstProf,hgt, color = '#067084',label='Dust')
    plt.legend()


    return FineProf,DstProf,SeaProf,Aod_Classified


def RSP_Run_General(fwdModelYAMLpath, binPathGRASP, krnlPath,file_path,file_name,PixNo,ang1,ang2,TelNo, RSPwlIdx,GasAbsFn,SpecResFnPath): 
        #Runs GRASP for the given settings, bin and kernel
        #rslt is the GRASP rslt dictionary or contains GRASP Objects
        rslt = Read_Data_RSP_Oracles(file_path,file_name,PixNo,ang1,ang2,TelNo, RSPwlIdx,GasAbsFn,SpecResFnPath)
        print(rslt['OBS_hght'])
        # rslt['OBS_hght'] = 38000
        maxCPU = 10 #maximum CPU allocated to run GRASP on server
        gRuns = []
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





def RSP_Run(Kernel_type,file_path,file_name,PixNo,ang1,ang2,TelNo, RSPwlIdx,GasAbsFn,SpecResFnPath, ModeNo): 
        
        krnlPath = '/data/home/gregmi/GRASP_V2/grasp-dev-rtm-v120-new-inter/src/retrieval/internal_files'
# Kernel_type =  sphro is for the GRASP spheriod kernal, while TAMU is to run with Hexahedral Kernal
        if Kernel_type == "sphro":

            if ModeNo == None or ModeNo == 2:
                fwdModelYAMLpath = '/home/gregmi/git/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/settings_BCK_DoLP_POLAR_2modes_SphrodShape_ORACLE.yml'
            
            if ModeNo == 3:
                # fwdModelYAMLpath ='/home/gregmi/git/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/settings_BCK_DoLP_POLAR_3modes_SphrodShape_ORACLE.yml'
                #Case2 
                # fwdModelYAMLpath = '/data/home/gregmi/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/GRASPV2/settings_BCK_MAP_3modes_SphrodShape_ORACLES.yml'

                fwdModelYAMLpath ='/data/home/gregmi/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/GRASPV2/settings_BCK_MAP_3modes_Sphd_ORACLES.yml'
    
             # binPathGRASP ='/home/shared/GRASP_GSFC/build_RSP_v112/bin/grasp_app' 
            # binPathGRASP = '/data/home/gregmi/GRASP_V2/grasp-dev-rtm-v200-pgn/build/bin/grasp_app'  #GRASP Executable
            binPathGRASP = '/data/home/gregmi/GRASP_V2/grasp-dev-rtm-v120-new-inter/build/bin/grasp_app'
            # binPathGRASP ='/home/gregmi/GRASPV112/build_hexahedral/bin/grasp_app'
           
            savePath=f"/home/gregmi/ORACLES/RSP1-L1C_P3_20180922_R03_{Kernel_type}"
        
        if Kernel_type == "TAMU":
            if ModeNo == None or ModeNo == 2:
                fwdModelYAMLpath = '/home/gregmi/git/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/settings_BCK_DoLP_POLAR_2modes_HexShape_ORACLE.yml'
            if ModeNo == 3:
                # fwdModelYAMLpath = '/data/home/gregmi/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/GRASPV2/settings_BCK_MAP_3modes_SphrodShape_ORACLES.yml'

                fwdModelYAMLpath ='/data/home/gregmi/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/GRASPV2/settings_BCK_MAP_3modes_HexShape_ORACLES.yml'
            binPathGRASP = '/data/home/gregmi/GRASP_V2/grasp-dev-rtm-v120-new-inter/bin/grasp_app'
            
            # binPathGRASP = '/data/home/gregmi/GRASP_V2/grasp-dev-rtm-v200-pgn/build/bin/grasp_app'  #GRASP Executable
            # binPathGRASP ='/data/home/gregmi/GRASP_V2/grasp-dev-rtm-v120-new-inter/bin'
            # binPathGRASP ='/home/gregmi/GRASPV112/build_hexahedral/bin/grasp_app'
            # binPathGRASP ='/home/shared/GRASP_GSFC/build_HexV112_4Modes/bin/grasp_app'
            savePath=f"/home/gregmi/ORACLES/RSP1-L1C_P3_20180922_R03_{Kernel_type}"

        #rslt is the GRASP rslt dictionary or contains GRASP Objects
        rslt = Read_Data_RSP_Oracles(file_path,file_name,PixNo,ang1,ang2,TelNo, RSPwlIdx,GasAbsFn,SpecResFnPath)
        print(rslt['OBS_hght'])
        # rslt['OBS_hght'] = 38000
        maxCPU = 10 #maximum CPU allocated to run GRASP on server
        gRuns = []
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

#Running the GRASP for spherical or hexahedral shape model for HSRL data
def HSLR_run(Kernel_type,HSRLfile_path,HSRLfile_name,HSRLPixNo, nwl,updateYaml= None,ConsType = None,releaseYAML =True, ModeNo=None,VertProfConstrain =None,Simplestcase =None,rslts_Sph=None,RSP_rslt = None, LagrangianOnly = None, RSPUpdate= None, AprioriLagrange =None):
        """Run GRASP for HSRL data
        
        
        
        """
        
        
        #Path to the kernel files
        krnlPath = '/data/home/gregmi/GRASP_V2/grasp-dev-rtm-v120-new-inter/src/retrieval/internal_files'
        if Kernel_type == "sphro":  #If spheroid model

            #Path to the yaml file for sphreroid model
            # fwdModelYAMLpath ='/home/gregmi/git/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/settings_BCK_POLAR_4modes_Shape_ORACLE1.yml'
            if ModeNo == None or ModeNo == 2:
                fwdModelYAMLpath ='/home/gregmi/git/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/settings_BCK_POLARandLIDAR_10Vbins_2modes_ORACLES.yml'
            if ModeNo == 3:
                # fwdModelYAMLpath ='/data/home/gregmi/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/GRASPV2/settings_BCK_LIDAR_3modes_Shape_ORACLE.yml'
                # fwdModelYAMLpath ='/data/home/gregmi/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/GRASPV2/settings_BCK_LIDAR_3modes_Shape_ORACLE.yml'
                fwdModelYAMLpath ='/data/home/gregmi/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/GRASPV2/settings_BCK_LIDAR_3modes_Sphd_ORACLE.yml'

            
            if ModeNo ==4:
                fwdModelYAMLpath = '/data/home/gregmi/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/GRASPV2/settings_BCK_LIDAR_3modes_Shape_ORACLE.yml'
            
            if updateYaml == True:  # True if init conditions for Yaml file for HSRL is updated from the GRASP output from RSP
                maxr=1.05  #set max and min value : here max = 5% incease, min 5% decrease : this is a very narrow distribution
                minr =0.95
                a=1
                
                update_HSRLyaml(fwdModelYAMLpath, ModeNo, maxr, minr, a,Kernel_type,ConsType,RSP_rslt, LagrangianOnly, RSPUpdate, AprioriLagrange)
                fwdModelYAMLpath ='/data/home/gregmi/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/GRASPV2/Settings_Sphd_RSP_HSRL.yaml'
            
            
            # binPathGRASP = path toGRASP Executable for spheriod model
            binPathGRASP ='/data/home/gregmi/GRASP_V2/grasp-dev-rtm-v120-new-inter/build/bin/grasp_app'

            # binPathGRASP ='/data/home/gregmi/GRASP_V2/grasp-dev-rtm-v120-new-inter/build_hex/bin'
            # binPathGRASP ='/home/gregmi/GRASPV112/build/bin/grasp_app'
            # binPathGRASP ='/home/shared/GRASP_GSFC/build_t/bin/grasp_app'
            # binPathGRASP ='/home/shared/GRASP_GSFC/build-tmu/bin/grasp_app' #GRASP Executable
            # binPathGRASP ='/home/shared/GRASP_GSFC/build_HEX_v112/bin/grasp_app'
            info = VariableNoise(fwdModelYAMLpath,nwl)
            # binPathGRASP ='/home/shared/GRASP_GSFC/build_RSP_v112/bin/grasp_app' 
            savePath=f"/home/gregmi/ORACLES/HSRL1_P3_20180922_R03_{Kernel_type}"
        
        if Kernel_type == "TAMU":

            if ModeNo == None or ModeNo == 2:
                fwdModelYAMLpath ='/home/gregmi/git/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/settings_BCK_POLARandLIDAR_10Vbins_2modes_Hex.yml'
            if ModeNo == 3:
                # fwdModelYAMLpath ='/home/gregmi/git/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/settings_BCK_POLAR_3modes_Shape_Hex.yml'
                # fwdModelYAMLpath ='/data/home/gregmi/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/GRASPV2/settings_BCK_LIDAR_3modes_Hex_ORACLE.yml'

                fwdModelYAMLpath ='/data/home/gregmi/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/GRASPV2/settings_BCK_LIDAR_3modes_Hex_ORACLE.yml'

            
            
            if ModeNo ==4:
                fwdModelYAMLpath = '/home/gregmi/git/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/settings_BCK_POLAR_4modes_Shape_HEX.yml'
            # fwdModelYAMLpath ='/home/gregmi/git/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/settings_BCK_POLAR_3modes_Shape_Hex.yml'
            # fwdModelYAMLpath = '/home/gregmi/git/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/settings_BCK_POLAR_3modes_Shape_ORACLE.yml'
            if updateYaml == True:# True if init conditions for Yaml file for HSRL is updated from the GRASP output from RSP
                maxr=1.05  #set max and min value : here max = 5% incease, min 5% decrease : this is a very narrow distribution
                minr =0.95
                a=1
                
                update_HSRLyaml(fwdModelYAMLpath, ModeNo, maxr, minr, a,Kernel_type,ConsType,RSP_rslt, LagrangianOnly, RSPUpdate, AprioriLagrange)
                # update_HSRLyaml(fwdModelYAMLpath, rslts_Tamu[0], ModeNo, maxr, minr, a,Kernel_type,ConsType)
                fwdModelYAMLpath ='/data/home/gregmi/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/GRASPV2/Settings_TAMU_RSP_HSRL.yaml'
            #Path to the GRASP Executable for TAMU
            info = VariableNoise(fwdModelYAMLpath,nwl)
            # binPathGRASP ='/home/shared/GRASP_GSFC/build-tmu/bin/grasp_app' #GRASP Executable
            # binPathGRASP ='/home/shared/GRASP_GSFC/build_HEX_v112/bin/grasp_app'


            # binPathGRASP = '/data/home/gregmi/GRASP_V2/grasp-dev-rtm-v200-pgn/build/bin/grasp_app' 
            binPathGRASP = '/data/home/gregmi/GRASP_V2/grasp-dev-rtm-v120-new-inter/bin/grasp_app'



            # binPathGRASP ='/home/gregmi/GRASPV112/build_hexahedral/bin/grasp_app'
            # binPathGRASP ='/home/shared/GRASP_GSFC/build_HexV112_4Modes/bin/grasp_app'
            #Path to save output plot
            # savePath=f"/home/gregmi/ORACLES/RSP1-L1C_P3_20180922_R03_{Kernel_type}"

        #This section is for the normalization paramteter in the yaml settings file
        #rslt is the GRASP rslt dictionary or contains GRASP Objects
        # DictHsrl = Read_Data_HSRL_Oracles_Height(HSRLfile_path,HSRLfile_name,HSRLPixNo)

        if Simplestcase == True:  #Height grid is constant and no gas correction applied
            DictHsrl = Read_Data_HSRL_Oracles_Height(HSRLfile_path,HSRLfile_name,HSRLPixNo,gaspar =None,SimpleCase = True)
            # DictHsrl = Read_Data_HSRL_Oracles_Height_No355(HSRLfile_path,HSRLfile_name,HSRLPixNo,gaspar =None,SimpleCase = True)
        else: #Variable grid height and gas correction applied
            DictHsrl = Read_Data_HSRL_Oracles_Height(HSRLfile_path,HSRLfile_name,HSRLPixNo,gaspar =True)
            # DictHsrl = Read_Data_HSRL_Oracles_Height_No355(HSRLfile_path,HSRLfile_name,HSRLPixNo,gaspar =True)

        rslt = DictHsrl[0]
        max_alt = rslt['OBS_hght']
        
        #Updating the normalization values in the settings file. 
        with open(fwdModelYAMLpath, 'r') as f:  
            data = yaml.safe_load(f)
        
        


            AeroProf = AeroProfNorm_sc2(DictHsrl) #vertical profiles of Dust, marine and fine contribution to the total aod caculated based on the dust mixing ratio and fine mode fraction.
            AerClassAOD = AeroClassAOD(DictHsrl)
            # AeroProf =AeroProfNorm_FMF(DictHsrl)

            FineProfext,DstProfext,SeaProfext = AeroProf[0],AeroProf[1],AeroProf[2]

            for noMd in range(ModeNo+1): #loop over the aerosol modes (i.e 2 for fine and coarse)

                # MinVal = np.repeat(np.minimum(np.minimum(FineProf[FineProf>1.175494e-38],DstProf),SeaProf),10).tolist() # the size of the list will be adjusted later in the code
             # Updating the vertical profile norm values in yaml file: 
                if noMd ==1:  #Mode 1
                    # data['retrieval']['constraints'][f'characteristic[1]'][f'mode[{noMd}]']['initial_guess']['min'] =   MinVal  #Int his version of GRASP min and val value should be same for each modes
                    data['retrieval']['constraints'][f'characteristic[1]'][f'mode[{noMd}]']['initial_guess']['value'] =  FineProfext.tolist()
                if noMd ==2: #Mode 2 
                    # data['retrieval']['constraints'][f'characteristic[1]'][f'mode[{noMd}]']['initial_guess']['min'] =   MinVal
                    data['retrieval']['constraints'][f'characteristic[1]'][f'mode[{noMd}]']['initial_guess']['value'] =  DstProfext.tolist()
                if noMd ==3: 
                    # data['retrieval']['constraints'][f'characteristic[1]'][f'mode[{noMd}]']['initial_guess']['min'] =   MinVal
                    data['retrieval']['constraints'][f'characteristic[1]'][f'mode[{noMd}]']['initial_guess']['value'] =  SeaProfext.tolist()
        
            if Kernel_type == "sphro":
                UpKerFile = 'settings_BCK_POLAR_3modes_Shape_Sph_Update.yml' #for spheroidal kernel
            if Kernel_type == "TAMU":
                UpKerFile = 'settings_BCK_POLAR_3modes_Shape_HEX_Update.yml'#for hexahedral kernel
        
            ymlPath = '/data/home/gregmi/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/GRASPV2/'

            with open(ymlPath+UpKerFile, 'w') as f2: #write the chnages to new yaml file
                yaml.safe_dump(data, f2)
            #     # print("Updating",YamlChar[i])
                fwdModelYAMLpath = ymlPath+UpKerFile

            Vext = rslt['meas_VExt']
        
        #Running GRASP
        maxCPU = 10 #maximum CPU allocated to run GRASP on server
        gRuns = []
        yamlObj = graspYAML(baseYAMLpath= fwdModelYAMLpath)
        # ymlO = yamlObj._repeatElementsInField(fldName=fwdModelYAMLpath, Nrepeats =10, λonly=False)
        #eventually have to adjust code for height, this works only for one pixel (single height value)
        gRuns.append(graspRun(pathYAML=yamlObj, releaseYAML= releaseYAML)) # This should copy to new YAML object
        pix = pixel()
        pix.populateFromRslt(rslt, radianceNoiseFun=None, dataStage= 'meas', verbose=False)
        gRuns[-1].addPix(pix)
        gDB = graspDB(graspRunObjs=gRuns, maxCPU=maxCPU)
        #rslts contain all the results form the GRASP inverse run
        rslts, failPix = gDB.processData(binPathGRASP=binPathGRASP, savePath=None, krnlPathGRASP=krnlPath)
        
        return rslts, max_alt, info, AerClassAOD 

def PlotRandomGuess(filename_npy, NoItr): 
    #This function plots the values  from all the  random inital guesses. 
    noMod=3
    
    fig, ax = plt.subplots(nrows= 3, ncols=2, figsize=(20, 15))
# #     
    loaded_data = np.load(filename_npy, allow_pickle=True)
    
    costVal = []
    colors = cm.rainbow(costVal)
    c = ['#625460','#CB9129','#6495ED']
    ls = ['solid','solid','dashed' ]
    smoke = mpatches.Patch(color=c[0], label='fine')
    dust = mpatches.Patch(color=c[1], label='Non-sph dust')
    salt = mpatches.Patch(color=c[2], label='Sph sea salt')
    
    
#     for j in range(noMod):

    
#         for y, c in zip(costVal, colors):
#             plt.scatter(loaded_data[i][0]['lambda'], loaded_data[i][0]['n'][j], color=c)
  
    for i in range(NoItr):
        if len(loaded_data[i]) >0:
            costVal.append(loaded_data[i][0]['costVal'])
#             if loaded_data[i][0]['costVal'] < 4.5: 
#                 print(i), print(loaded_data[i][0]['costVal'])
#                 CombinedLidarPolPlot(loaded_data[i],loaded_data[i])

            for j in range(noMod):
                

                ax[0,1].plot(loaded_data[i][0]['lambda'],loaded_data[i][0]['n'][j], alpha = 0.5, color= c[j], ls = ls[j])
                ax[0,1].set_ylabel('n')
                ax[1,1].plot(loaded_data[i][0]['lambda'],loaded_data[i][0]['k'][j], alpha = 0.5, color= c[j], ls = ls[j])
                ax[1,1].set_ylabel('k')
                ax[1,0].plot(loaded_data[i][0]['lambda'],loaded_data[i][0]['aodMode'][j], alpha = 0.5, color= c[j], ls = ls[j])
                ax[1,0].set_ylabel('aodMode')
                ax[2,0].plot(loaded_data[i][0]['lambda'],loaded_data[i][0]['ssaMode'][j], alpha = 0.5, color= c[j], ls = ls[j])
                ax[2,0].set_ylabel('ssaMode')
                ax[2,1].plot(loaded_data[i][0]['r'][j], loaded_data[i][0]['dVdlnr'][j],color= c[j], ls = ls[j], alpha = 0.5,lw = 2)
               
                ax[2,1].set_xlabel(r'rv $ \mu m$')
                ax[2,1].set_ylabel('dVdlnr')
                ax[2,1].set_xscale("log")


    ax[0,0].hist(costVal, color ='#12106C') 
    ax[0,0].set_xlabel('CostVal')
    
    plt.suptitle(f'{filename_npy[:3]} RSP+HSRL Aerosol Retrieval')
    plt.legend(handles=[smoke,dust,salt]) 
    
    plt.savefig(f'{filename_npy[:5]}.png', dpi = 100)
    
    return costVal


def LidarAndMAP(Kernel_type,HSRLfile_path,HSRLfile_name,HSRLPixNo,file_path,file_name,RSP_PixNo,ang1,ang2,TelNo, RSPwlIdx,GasAbsFn,SpecResFnPath, ModeNo=None, updateYaml= None, RandinitGuess =None , NoItr=None,fnGuessRslt = None):
    
    failedmeas = 0 #Count of number of meas that failed

    #Path to the Kernel Files
    krnlPath = '/data/home/gregmi/GRASP_V2/grasp-dev-rtm-v120-new-inter/src/retrieval/internal_files'

    if Kernel_type == "sphro":  #If spheriod model
        #Path to the yaml file for sphriod model
        if ModeNo == None or ModeNo == 2:
            fwdModelYAMLpath = '/home/gregmi/git/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/settings_BCK_LidarAndMAP_V.1.2.yml'
        if ModeNo == 3:
            # fwdModelYAMLpath ='/data/home/gregmi/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/GRASPV2/settings_BCK_LidarAndMAP_3modes_V.1.2.yml'

            fwdModelYAMLpath ='/data/home/gregmi/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/GRASPV2/settings_BCK_LidarAndMAP_3modes_sphd_GV2.yml'

        if updateYaml == True:  # True if init conditions for Yaml file for HSRL is updated from the GRASP output from RSP
            update_HSRLyaml(fwdModelYAMLpath, rslts_Sph[0], noMod, maxr, minr, a,Kernel_type)
            
            fwdModelYAMLpath ='/data/home/gregmi/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/GRASPV2/settings_BCK_LidarAndMAP_3modes_Hex_GV2.yml'

        # binPathGRASP = path toGRASP Executable for spheriod model
        # binPathGRASP ='/home/shared/GRASP_GSFC/build_SphrdV112_Noise/bin/grasp_app'
        # binPathGRASP = '/data/home/gregmi/GRASP_V2/grasp-dev-rtm-v200-pgn/build/bin/grasp_app'   #This will work for both the kernels as it has more parameter wettings
        binPathGRASP = '/data/home/gregmi/GRASP_V2/grasp-dev-rtm-v120-new-inter/build/bin/grasp_app'
        
        # binPathGRASP ='/home/shared/GRASP_GSFC/build_RSP_v112/bin/grasp_app' 
        savePath=f"/home/gregmi/ORACLES/HSRL1_P3_20180922_R03_{Kernel_type}"
        fnGuessRslt ='sphRandnew.npy'
    
    if Kernel_type == "TAMU": #Using Hexhderal shape kernel.
        if ModeNo == None or ModeNo == 2:
            fwdModelYAMLpath = '/home/gregmi/git/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/settings_BCK_LidarAndMAP_V.1.2_TAMU.yml'
        if ModeNo == 3:
            fwdModelYAMLpath ='/home/gregmi/git/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/settings_BCK_LidarAndMAP_3modes_V.1.2_TAMU.yml'

            fwdModelYAMLpath ='/data/home/gregmi/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/GRASPV2/settings_BCK_LidarAndMAP_3modes_Hex_GV2.yml'
        if updateYaml == True:# True if init conditions for Yaml file for HSRL is updated from the GRASP output from RSP
            update_HSRLyaml(fwdModelYAMLpath, rslts_Tamu[0], noMod, maxr, minr, a,Kernel_type)
            fwdModelYAMLpath ='/home/gregmi/git/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/Settings_TAMU_RSP_HSRL.yaml'

        #Path to the GRASP Executable for TAMU
        # binPathGRASP ='/home/shared/GRASP_GSFC/build_HexV112_Noise/bin/grasp_app' #Recompiled to account for more noise parameter
        #GRASP Executable
        # binPathGRASP = '/data/home/gregmi/GRASP_V2/grasp-dev-rtm-v120-new-inter/bin/grasp_app'
        binPathGRASP = '/data/home/gregmi/GRASP_V2/grasp-dev-rtm-v120-new-inter/bin/grasp_app'
        # binPathGRASP ='/home/shared/GRASP_GSFC/build_HEX_v112/bin/grasp_app'
        #Path to save output plot
        savePath=f"/home/gregmi/ORACLES/RSP1-L1C_P3_20180922_R03_{Kernel_type}"
        fnGuessRslt ='HexRandnew.npy'

# /tmp/tmpn596k7u8$
    rslt_HSRL_1 = Read_Data_HSRL_Oracles_Height(HSRLfile_path,HSRLfile_name,HSRLPixNo)


    # rslt_HSRL_1 = Read_Data_HSRL_Oracles_Height_No355(HSRLfile_path,HSRLfile_name,HSRLPixNo)
    rslt_HSRL =  rslt_HSRL_1[0]
    
    rslt_RSP = Read_Data_RSP_Oracles(file_path,file_name,RSP_PixNo,ang1,ang2,TelNo, RSPwlIdx,GasAbsFn,SpecResFnPath)
    
    rslt= {}  # Teh order of the data is  First lidar(number of wl ) and then Polarimter data 
    rslt['lambda'] = np.concatenate((rslt_HSRL['lambda'],rslt_RSP['lambda']))

    #Sort the index of the wavelength if arranged in ascending order, this is required by GRASP
    sort = np.argsort(rslt['lambda']) 
    IndHSRL = rslt_HSRL['lambda'].shape[0]
    sort_Lidar, sort_MAP  = np.array([0,3,7]),np.array([1,2,4,5,6])
    
    # sort_Lidar, sort_MAP  = np.array([0,2,6]),np.array([1,3,4,5])
    
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
        print(rslt['gaspar'])

    if 'gaspar' in  rslt_RSP:
        gasparRSP = np.zeros(len(rslt['lambda']))
        gasparRSP[sort_MAP] = rslt_RSP['gaspar']

        rslt['gaspar'] = gasparRSP
        print(rslt['gaspar'])

    if 'gaspar' in  rslt_RSP and rslt_HSRL :
        gasparB = np.zeros(len(rslt['lambda']))
        gasparB[sort_MAP] = rslt_RSP['gaspar']
        gasparB[sort_Lidar] = rslt_HSRL['gaspar']
        rslt['gaspar'] = gasparB
        print(rslt['gaspar'])



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
    
    if Kernel_type == "sphro":
        UpKerFile = 'settings_LIDARandPOLAR_3modes_Shape_Sph_Update.yml' #for spheroidal kernel
    if Kernel_type == "TAMU":
        UpKerFile = 'settings_LIDARandPOLAR_3modes_Shape_HEX_Update.yml'#for hexahedral kernel

    ymlPath = '/data/home/gregmi/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/GRASPV2/'
    

    with open(ymlPath+UpKerFile, 'w') as f2: #write the chnages to new yaml file
        yaml.safe_dump(data, f2)
        #     # print("Updating",YamlChar[i])

    max_alt = rslt['OBS_hght'] #altitude of the aircraft
    # print(rslt['OBS_hght'])

    Vext = rslt['meas_VExt']
    Updatedyaml = ymlPath+UpKerFile
    # rslt['meas_DP'][1,-1] = np.nan
    # print(rslt['meas_DP'][1,2])
    Finalrslts =[]

    if RandinitGuess == True:
        
        maxCPU = 3
        if NoItr==None: NoItr=1
        gRuns = [[]]*NoItr
        pix = pixel()
        pix.populateFromRslt(rslt, radianceNoiseFun=None, dataStage= 'meas', verbose=False)
        dblist = []
        
        for i in range(NoItr):
            try:
                print(i)
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


def PlotRandomGuess(filename_npy, NoItr):
    fig, ax = plt.subplots(nrows= 3, ncols=2, figsize=(30, 10))
# #     
    loaded_data = np.load(filename_npy, allow_pickle=True)
    
    costVal = []

    for i in range(NoItr):
        if len(loaded_data[i]) >0:
            costVal.append(loaded_data[i][0]['costVal'])

            for j in range(noMod):
                ax[0,1].plot(loaded_data[i][0]['n'][i])
                ax[1,1].plot(loaded_data[i][0]['k'][i])
                ax[1,0].plot(loaded_data[i][0]['aodMode'][i])
                ax[2,0].plot(loaded_data[i][0]['ssaMode'][i])
    ax[0,0].hist(costVal)      
    return

def Netcdf(Finalrslts,NoItr):
    with nc.Dataset('/home/gregmi/git/GSFC-GRASP-Python-Interface/IntialGuessHSRLRSP.nc', 'a') as rootgrp:
        
        data = Finalrslts[1][0][0]
        # Create dimensions
        rootgrp.createDimension('r_dim', data['r'].shape[0])
        rootgrp.createDimension('range_dim', data['r'].shape[1])
        rootgrp.createDimension('d_range', data['meas_DP'].shape[0])
        rootgrp.createDimension('d_wl', data['range'].shape[1])
        rootgrp.createDimension('wlRSP', data['meas_I'].shape[0])
        rootgrp.createDimension('angle', data['meas_I'].shape[1])
        rootgrp.createDimension('allLambda', data['lambda'].shape[0])
        rootgrp.createDimension('iter_d', NoItr)

        rootgrp.createVariable('datetime', 'S19')
        rootgrp.createVariable('datetime', 'S19')
        rootgrp.createVariable('latitude', 'f4')
        rootgrp.createVariable('land_prct', 'f4')
        rootgrp.createVariable('r', 'f4', ('r_dim', 'range_dim','iter_d'))
        rootgrp.createVariable('dVdlnr', 'f4', ('r_dim', 'range_dim','iter_d'))
        rootgrp.createVariable('rv', 'f4', ('r_dim','iter_d'))
        rootgrp.createVariable('sigma', 'f4', ('r_dim','iter_d'))
        rootgrp.createVariable('vol', 'f4', ('r_dim','iter_d'))
        rootgrp.createVariable('sph', 'f4', ('r_dim','iter_d'))
        rootgrp.createVariable('range', 'f4', ('d_range','iter_d'))
        rootgrp.createVariable('beta_ext', 'f4', ('r_dim','d_range','iter_d'))
        rootgrp.createVariable('lambda', 'f4', 'allLambda')[:]
        rootgrp.createVariable('aod', 'f4',  ('allLambda','iter_d'))
        rootgrp.createVariable('aod_mode', 'f4', ('r_dim', 'allLambda','iter_d'))
        rootgrp.createVariable('ssa', 'f4',  ('allLambda','iter_d'))
        rootgrp.createVariable('ssa_mode', 'f4', ('r_dim', 'allLambda','iter_d'))
        rootgrp.createVariable('n', 'f4', ('r_dim','allLambda','iter_d'))
        rootgrp.createVariable('k', 'f4', ('r_dim','allLambda','iter_d'))
        rootgrp.createVariable('lidar_ratio', 'f4', ('allLambda','iter_d'))
        # rootgrp.createVariable('height', 'f4', ('d_range','d_wl'))[:] = data['height']
        # rootgrp.createVariable('r_eff', 'f4', ('r_dim', 'range_dim'))[:] = data['rEff']
        # rootgrp.createVariable('wtr_surf', 'f4', ('r_dim', 'range_dim'))[:] = data['wtrSurf']
        rootgrp.createVariable('cost_val', 'f4', ('r_dim','iter_d'))
        # rootgrp.createVariable('range_lidar', 'f4', ('d_range','d_wl'))[:] = data['RangeLidar']
        rootgrp.createVariable('meas_dp', 'f4', ('d_range','allLambda','iter_d'))
        rootgrp.createVariable('fit_dp', 'f4', ('d_range','allLambda','iter_d'))
        rootgrp.createVariable('meas_vext', 'f4', ('d_range','allLambda','iter_d'))
        rootgrp.createVariable('fit_vext', 'f4', ('d_range','allLambda','iter_d'))
        rootgrp.createVariable('meas_vbs', 'f4', ('d_range','allLambda','iter_d'))
        rootgrp.createVariable('fit_vbs', 'f4', ('d_range','allLambda','iter_d'))
        rootgrp.createVariable('sza', 'f4', ('angle','wlRSP','iter_d'))
        rootgrp.createVariable('vis', 'f4', ('angle','wlRSP','iter_d'))
        rootgrp.createVariable('fis', 'f4', ('angle','wlRSP','iter_d'))
        rootgrp.createVariable('sca_ang', 'f4', ('angle','wlRSP','iter_d'))
        rootgrp.createVariable('meas_i', 'f4', ('angle', 'wlRSP','iter_d'))
        rootgrp.createVariable('fit_i', 'f4', ('angle', 'wlRSP','iter_d'))
        rootgrp.createVariable('meas_p_rel', 'f4', ('angle', 'wlRSP','iter_d'))
        rootgrp.createVariable('fit_p_rel', 'f4', ('angle', 'wlRSP','iter_d'))


        for i in range(NoItr):
            data = Finalrslts[1][i][0]
        # Create variables and assign values
            rootgrp['datetime'] = np.array(str(data['datetime']), dtype='S19')
            rootgrp['longitude'][:] = data['longitude']
            rootgrp.createVariable('latitude', 'f4')[:] = data['latitude']
            rootgrp.createVariable('land_prct', 'f4')[:] = data['land_prct']
            rootgrp.createVariable('r', 'f4', ('r_dim', 'range_dim','iter_d'))[:,:,i] = data['r']
            rootgrp.createVariable('dVdlnr', 'f4', ('r_dim', 'range_dim','iter_d'))[:,:,i] = data['dVdlnr']
            rootgrp.createVariable('rv', 'f4', ('r_dim','iter_d'))[:,i] = data['rv']
            rootgrp.createVariable('sigma', 'f4', ('r_dim','iter_d'))[:,i] = data['sigma']
            rootgrp.createVariable('vol', 'f4', ('r_dim','iter_d'))[:,i] = data['vol']
            rootgrp.createVariable('sph', 'f4', ('r_dim','iter_d'))[:,i] = data['sph']
            rootgrp.createVariable('range', 'f4', ('d_range','iter_d'))[:,i] = data['range'][0]
            rootgrp.createVariable('beta_ext', 'f4', ('r_dim','d_range','iter_d'))[:,:,i] = data['βext']
            rootgrp.createVariable('lambda', 'f4', 'allLambda')[:] = data['lambda']
            rootgrp.createVariable('aod', 'f4',  ('allLambda','iter_d'))[:,i] = data['aod']
            rootgrp.createVariable('aod_mode', 'f4', ('r_dim', 'allLambda','iter_d'))[:,:,i] = data['aodMode']
            rootgrp.createVariable('ssa', 'f4',  ('allLambda','iter_d'))[:,i] = data['ssa']
            rootgrp.createVariable('ssa_mode', 'f4', ('r_dim', 'allLambda','iter_d'))[:,:,i] = data['ssaMode']
            rootgrp.createVariable('n', 'f4', ('r_dim','allLambda','iter_d'))[:,:,i] = data['n']
            rootgrp.createVariable('k', 'f4', ('r_dim','allLambda','iter_d'))[:,:,i] = data['k']
            rootgrp.createVariable('lidar_ratio', 'f4', ('allLambda','iter_d'))[:,i] = data['LidarRatio']
            # rootgrp.createVariable('height', 'f4', ('d_range','d_wl'))[:] = data['height']
            # rootgrp.createVariable('r_eff', 'f4', ('r_dim', 'range_dim'))[:] = data['rEff']
            # rootgrp.createVariable('wtr_surf', 'f4', ('r_dim', 'range_dim'))[:] = data['wtrSurf']
            rootgrp.createVariable('cost_val', 'f4', ('r_dim','iter_d'))[:,i] = data['costVal']
            # rootgrp.createVariable('range_lidar', 'f4', ('d_range','d_wl'))[:] = data['RangeLidar']
            rootgrp.createVariable('meas_dp', 'f4', ('d_range','allLambda','iter_d'))[:,:,i] = data['meas_DP']
            rootgrp.createVariable('fit_dp', 'f4', ('d_range','allLambda','iter_d'))[:,:,i] = data['fit_DP']
            rootgrp.createVariable('meas_vext', 'f4', ('d_range','allLambda','iter_d'))[:,:,i] = data['meas_VExt']
            rootgrp.createVariable('fit_vext', 'f4', ('d_range','allLambda','iter_d'))[:,:,i] = data['fit_VExt']
            rootgrp.createVariable('meas_vbs', 'f4', ('d_range','allLambda','iter_d'))[:,:,i] = data['meas_VBS']
            rootgrp.createVariable('fit_vbs', 'f4', ('d_range','allLambda','iter_d'))[:,:,i] = data['fit_VBS']
            rootgrp.createVariable('sza', 'f4', ('angle','wlRSP','iter_d'))[:,:,i] = data['sza']
            rootgrp.createVariable('vis', 'f4', ('angle','wlRSP','iter_d'))[:,:,i] = data['vis']
            rootgrp.createVariable('fis', 'f4', ('angle','wlRSP','iter_d'))[:,:,i] = data['fis']
            rootgrp.createVariable('sca_ang', 'f4', ('angle','wlRSP','iter_d'))[:,:,i] = data['sca_ang']
            rootgrp.createVariable('meas_i', 'f4', ('angle', 'wlRSP','iter_d'))[:,:,i] = data['meas_I']
            rootgrp.createVariable('fit_i', 'f4', ('angle', 'wlRSP','iter_d'))[:,:,i] = data['fit_I']
            rootgrp.createVariable('meas_p_rel', 'f4', ('angle', 'wlRSP','iter_d'))[:,:,i] = data['meas_P_rel']
            rootgrp.createVariable('fit_p_rel', 'f4', ('angle', 'wlRSP','iter_d'))[:,:,i] = data['fit_P_rel']
    
    # netcdf_file.close()    
    
    
    return


def randomInitGuess(Kernel_type,HSRLfile_path,HSRLfile_name,PixNo, nwl,updateYaml= None,ConsType = None,releaseYAML =True, ModeNo=None,NoItr=None):
    Finalrslts =[]
    krnlPath='/data/home/gregmi/GRASP_V2/grasp-dev-rtm-v120-new-inter/src/retrieval/internal_files'

    if Kernel_type == "sphro":  #If spheroid model

        #Path to the yaml file for sphreroid model
        # fwdModelYAMLpath ='/home/gregmi/git/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/settings_BCK_POLAR_4modes_Shape_ORACLE1.yml'
        if ModeNo == None or ModeNo == 2:
            fwdModelYAMLpath ='/home/gregmi/git/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/settings_BCK_POLARandLIDAR_10Vbins_2modes_ORACLES.yml'
        if ModeNo == 3:
            fwdModelYAMLpath ='/home/gregmi/git/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/settings_BCK_POLAR_3modes_Shape_ORACLE.yml'
        if ModeNo ==4:
            fwdModelYAMLpath = '/home/gregmi/git/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/settings_BCK_POLAR_4modes_Shape_ORACLE1.yml'
        
        if updateYaml == True:  # True if init conditions for Yaml file for HSRL is updated from the GRASP output from RSP
            update_HSRLyaml(fwdModelYAMLpath, rslts_Sph[0], noMod, maxr, minr, a,Kernel_type,ConsType)
            fwdModelYAMLpath ='/home/gregmi/git/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/Settings_Sphd_RSP_HSRL.yaml'
        
        
        # binPathGRASP = path toGRASP Executable for spheriod model
        # binPathGRASP ='/home/shared/GRASP_GSFC/build_HEX_v112/bin/grasp_app'
        # binPathGRASP ='/home/shared/GRASP_GSFC/build_t/bin/grasp_app'
        # binPathGRASP ='/home/shared/GRASP_GSFC/build-tmu/bin/grasp_app' #GRASP Executable
        binPathGRASP ='/home/shared/GRASP_GSFC/build_HEX_v112/bin/grasp_app'
        
        
        info = VariableNoise(fwdModelYAMLpath,nwl)
        
        
        
        # binPathGRASP ='/home/shared/GRASP_GSFC/build_RSP_v112/bin/grasp_app' 
        savePath=f"/home/gregmi/ORACLES/HSRL1_P3_20180922_R03_{Kernel_type}"
    
    if Kernel_type == "TAMU":

        if ModeNo == None or ModeNo == 2:
            fwdModelYAMLpath ='/home/gregmi/git/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/settings_BCK_POLARandLIDAR_10Vbins_2modes_Hex.yml'
        if ModeNo == 3:
            fwdModelYAMLpath ='/home/gregmi/git/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/settings_BCK_POLAR_3modes_Shape_Hex.yml'
        if ModeNo ==4:
            fwdModelYAMLpath = '/home/gregmi/git/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/settings_BCK_POLAR_4modes_Shape_HEX.yml'
        # fwdModelYAMLpath ='/home/gregmi/git/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/settings_BCK_POLAR_3modes_Shape_Hex.yml'
        # fwdModelYAMLpath = '/home/gregmi/git/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/settings_BCK_POLAR_3modes_Shape_ORACLE.yml'
        if updateYaml == True:# True if init conditions for Yaml file for HSRL is updated from the GRASP output from RSP
            update_HSRLyaml(fwdModelYAMLpath, rslts_Tamu[0], noMod, maxr, minr, a,Kernel_type,ConsType)
            fwdModelYAMLpath ='/home/gregmi/git/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/Settings_TAMU_RSP_HSRL.yaml'
        #Path to the GRASP Executable for TAMU
        info = VariableNoise(fwdModelYAMLpath,nwl)
        # binPathGRASP ='/home/shared/GRASP_GSFC/build-tmu/bin/grasp_app' #GRASP Executable
        binPathGRASP ='/home/shared/GRASP_GSFC/build_HEX_v112/bin/grasp_app'
        # binPathGRASP ='/home/shared/GRASP_GSFC/build_HexV112_4Modes/bin/grasp_app'
        #Path to save output plot
        savePath=f"/home/gregmi/ORACLES/RSP1-L1C_P3_20180922_R03_{Kernel_type}"

    rslt_HSRL_1 = Read_Data_HSRL_Oracles_Height(HSRLfile_path,HSRLfile_name,HSRLPixNo)
    rslt=  rslt_HSRL_1[0]

    maxCPU =5
    if NoItr==None: NoItr=1
    
    
    gRuns = [[]]*NoItr
    pix = pixel()
    pix.populateFromRslt(rslt, radianceNoiseFun=None, dataStage= 'meas', verbose=False)
    dblist = []
    
    for i in range(NoItr):
        gyaml = graspYAML(baseYAMLpath=fwdModelYAMLpath)
        yamlObjscramble = gyaml.scrambleInitialGuess(fracOfSpace=1, skipTypes=['vertical_profile_normalized','aerosol_concentration'])
        yamlObj = graspYAML(baseYAMLpath=fwdModelYAMLpath)
        # yamlObj=graspYAML(baseYAMLpath=fwdModelYAMLpath)
        #eventually have to adjust code for height, this works only for one pixel (single height value)
        gRuns = []
        gRuns.append(graspRun(pathYAML=yamlObj, releaseYAML= True )) # This should copy to new YAML object
        gRuns[-1].addPix(pix)

        gDB = graspDB(graspRunObjs=gRuns, maxCPU=maxCPU,maxT =1)
        dblist.append(gDB)#rslts contain all the results form the GRASP inverse run
        rslts, failPix = gDB.processData(binPathGRASP=binPathGRASP, savePath=None, krnlPathGRASP=krnlPath)

        Finalrslts.append(rslts)
    return Finalrslts,gRuns,dblist

def VrtGrad(HSRL_sphrod):

    '''This function calculates and plots the gradient of the vertical profiles to gauge for the noise of the measurement'''
    plt.rcParams['font.size'] = '25'
    fig, ax= plt.subplots(nrows = 3, ncols =3,  figsize= (24,24))
    for i in range (3):
        ax[0,i].plot(np.diff(HSRL_sphrod[0][0]['meas_VExt'][:,i]),HSRL_sphrod[0][0]['RangeLidar'][:,0][1:]/1000 , lw = 3, marker = '.')
        ax[0,i].set_xlabel("VExt, m-1")
        ax[1,i].plot(np.diff(HSRL_sphrod[0][0]['meas_VBS'][:,i]),HSRL_sphrod[0][0]['RangeLidar'][:,0][1:]/1000, lw = 3, marker = '.')
        ax[1,i].set_xlabel("VBS")
        ax[2,i].plot(np.diff(HSRL_sphrod[0][0]['meas_DP'][:,i]),HSRL_sphrod[0][0]['RangeLidar'][:,0][1:]/1000, lw = 3, marker = '.')
        ax[2,i].set_xlabel("DP")

def errPlots(rslts_Sph,rslts_Tamu):

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
    plt.savefig(f'{file_name[2:]}_{RSP_PixNo}_ErrorI.png', dpi = 300)
#Plotting the fits and the retrievals

def plot_HSRL(HSRL_sphrod,HSRL_Tamu, UNCERT, forward = None, retrieval = None, Createpdf = None,PdfName =None, combinedVal = None, key1= None, key2 = None ,RSP_plot =None):

    """ HSRL_sphrod = GRASP output array for spheroid 
        HSRL_Tamu = GRASP output array for hexahedral
        forward =  set this to True to plot the measuremnts and  fits
        retrieval = set thid to True yo plot retrieved properties 
        Createpdf = True if you want to create a pdf with plots 
        PdfName = Name/path of the pdf file
        combinedVal = combined string of noises and characteristics from yaml file, this is the output form VariableNoise'
    """

    Hsph,HTam = HSRL_sphrod,HSRL_Tamu
    font_name = "Times New Roman"
    plt.rcParams['font.size'] = '15'


    NoMode = HSRL_sphrod['r'].shape[0]

    if NoMode > 3: NoMode = 1 #If mode = 1 then r = 22 or the total no of bins

    # pdf_pages = PdfPages(PdfName) 

    if Createpdf == True:
        x = np.arange(10, 1000, 0.1)  #Adjusting the grid of the plot
        y = np.zeros(len(x))
        plt.figure(figsize=(30,15)) 
        #adding text inside the plot
        plt.text(-10, 0.000005,combinedVal , fontsize = 22)
        
        plt.plot(x, y, c='w')
        plt.xlabel("X-axis", fontsize = 15)
        plt.ylabel("Y-axis",fontsize = 15)
        
        
        # pdf_pages.savefig()

        Index1 = [0,1,2]
        Index2 = [0,1,2]
        IndexH = [0,3,7]
        IndexRSP = [1,2,4,5,6]

        cm_sp = ['k','#8B4000', '#87C1FF']
        cm_t = ["#BC106F",'#E35335', 'b']
        # color_sph = "#025043"
        color_sph = '#0c7683'
        color_tamu = "#d24787"
        
        if len(Hsph['lambda']) > 3: 
            Index1 = IndexH #Input is Lidar+polarimter

        if len(HTam['lambda']) > 3: 
            Index2 = IndexH 
        
        if RSP_plot !=None:
            Index3 = IndexRSP
            
        if len(HTam['lambda']) !=len(Hsph['lambda']) :
            cm_sp = ['#AA98A9','#CB9129', 'c']
            cm_t = ['#900C3F','#8663bd', '#053F5C']
            color_tamu = '#DE970B'
            color_sph = '#1175A8'

#Plotting the fits if forward is set to True
    if forward == True:
        #Converting range to altitude
        altd = (Hsph['RangeLidar'][:,0])/1000 #altitude for spheriod
        altT = (Hsph['RangeLidar'][:,0])/1000 #altitude for hexahedra
        fig, axs= plt.subplots(nrows = 1, ncols =3, figsize= (15,6))  #TODO make it more general which adjust the no of rows based in numebr of wl
        plt.subplots_adjust(top=0.78)
        for i in range(3):
            wave = str(Hsph['lambda'][i]) +"(μm)"
            axs[i].errorbar(Hsph['meas_VBS'][:,i],altd,xerr= UNCERT['VBS'],color = "#695E93", capsize=3,capthick =1,alpha =0.4, label =f"{UNCERT['VBS']}")
            
            axs[i].plot(Hsph['meas_VBS'][:,Index1[i]],altd, marker =">",color = "#281C2D", label ="Meas")
            axs[i].plot(Hsph['fit_VBS'][:,Index1[i]],altd,color =color_sph, marker = "$O$",label =f"{key1}",alpha =0.8)
            axs[i].plot(HTam['fit_VBS'][:,Index2[i]],altd,color =  color_tamu,ls = "--", label=f"{key2}", marker = "h")

            axs[i].xaxis.set_major_formatter(ScalarFormatter(useMathText=True))

            # print(UNCERT)
            # axs[i].errorbar(Hsph['fit_VBS'][:,i],altd,xerr= UNCERT['VBS'],color = "#025043", marker = "$O$",label ="Sphd")
            # axs[i].errorbar(HTam['fit_VBS'][:,i],altd,xerr= UNCERT['VBS'],color = "#d24787",ls = "--", label="Hex", marker = "h")

            axs[i].set_xlabel(f'$ VBS (m^{-1}Sr^{-1})$',fontproperties=font_name)
            axs[0].set_ylabel('Height above ground (km)',fontproperties=font_name)

            axs[i].set_title(wave)
            if i ==0:
                axs[0].legend()
            plt.suptitle(f"HSRL Vertical Backscatter profile  \nLat,Lon: {HTam['latitude']}, {HTam['longitude']} Date: {HTam['datetime']}\n ",fontproperties=font_name) #Initial condition strictly constrainted by RSP retrievals
        # pdf_pages.savefig()
        fig.savefig(f'/data/home/gregmi/LIDAR_MAP_dust/ORACLES_LiDAR_MAP_rslts/Case1/HSRLOnly/FIT_HSRL_Vertical_Backscatter_profile{NoMode}.png',dpi = 300)
            # plt.tight_layout()
        fig, axs= plt.subplots(nrows = 1, ncols =3, figsize= (15,6))
        plt.subplots_adjust(top=0.78)

        # AOD = np.trapz(Hsph['meas_VExt'],altd)
        for i in range(3):
            wave = str(Hsph['lambda'][i]) +"(μm)"

            axs[i].errorbar(Hsph['meas_DP'][:,Index1[i]],altd,xerr= UNCERT['DP'],color = '#695E93', capsize=4,capthick =1,alpha =0.6, label =f"{UNCERT['DP']}%")
            
            axs[i].plot(Hsph['meas_DP'][:,Index1[i]],altd, marker =">",color = "#281C2D", label ="Meas")
            axs[i].plot(Hsph['fit_DP'][:,Index1[i]],altd,color = color_sph, marker = "$O$",label =f"{key1}")
            axs[i].plot(HTam['fit_DP'][:,Index2[i]],altd,color = color_tamu, ls = "--",marker = "h",label=f"{key2}")
            
            # axs[i].errorbar(Hsph['fit_DP'][:,i],altd,xerr= UNCERT['DP'],fmt='-o',color = "#025043", marker = "$O$",label ="Sphd")
            # axs[i].errorbar(HTam['fit_DP'][:,i],altd,xerr= UNCERT['DP'],fmt='-o',color = "#d24787",ls = "--", label="Hex", marker = "h")


            axs[i].set_xlabel('DP (%)')
            axs[i].set_title(wave)
            if i ==0:
                axs[0].legend()
            axs[0].set_ylabel('Height above ground (km)',fontproperties=font_name)
            plt.suptitle(f"HSRL Depolarization Ratio \n Lat,Lon: {HTam['latitude']}, {HTam['longitude']}  Date: {HTam['datetime']}\n ",fontproperties=font_name) #Initial condition strictly constrainted by RSP retrievals
        # pdf_pages.savefig()
        fig.savefig(f'/data/home/gregmi/LIDAR_MAP_dust/ORACLES_LiDAR_MAP_rslts/Case1/HSRLOnly/FIT_HSRL_Depolarization_Ratio{NoMode}.png',dpi = 300)
        fig, axs= plt.subplots(nrows = 1, ncols =2, figsize= (11,6))
        plt.subplots_adjust(top=0.78)
        for i in range(2):
            wave = str(Hsph['lambda'][i]) +"(μm)"

            axs[i].errorbar(Hsph['meas_VExt'][:,Index1[i]],altd,xerr= UNCERT['VEXT']*Hsph['meas_VExt'][:,i],color = '#695E93',capsize=4,capthick =1,alpha =0.7, label =f"{UNCERT['VEXT']*100}%")
            
            axs[i].plot(Hsph['meas_VExt'][:,Index1[i]],altd, marker =">",color = "#281C2D", label ="Meas")
            axs[i].plot(Hsph['fit_VExt'][:,Index1[i]],altd,color = color_sph, marker = "$O$",label =f"{key1}")
            axs[i].plot(HTam['fit_VExt'][:,Index2[i]],altd,color = color_tamu,ls = "--", marker = "h",label=f"{key2}")
            
            axs[i].xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
            axs[i].ticklabel_format(style="sci", axis="x")
            # axs[i].errorbar(Hsph['fit_VExt'][:,i],altd,xerr= UNCERT['VEXT'],color = "#025043", marker = "$O$",label ="Sphd")
            # axs[i].errorbar(HTam['fit_VExt'][:,i],altd,xerr= UNCERT['VEXT'],color = "#d24787",ls = "--", label="Hex", marker = "h")

            
            
            axs[i].set_xlabel(f'$VExt (m^{-1})$',fontproperties=font_name)
            axs[0].set_ylabel('Height above ground (km)',fontproperties=font_name)
            axs[i].set_title(wave)
            if i ==0:
                axs[0].legend()
            plt.suptitle(f"HSRL Vertical Extinction profile\n Lat,Lon: {HTam['latitude']},{HTam['longitude']}  Date: {HTam['datetime']}\n ",fontproperties=font_name) #Initial condition strictly constrainted by RSP retrievals
        plt.tight_layout()
        # pdf_pages.savefig()
        fig.savefig(f'/data/home/gregmi/LIDAR_MAP_dust/ORACLES_LiDAR_MAP_rslts/Case1/HSRLOnly/FIT_HSRL_Vertical_Ext_profile{NoMode}.png',dpi = 300)

    if retrieval == True:
        plt.rcParams['font.size'] = '35'
        
        Spheriod,Hex = HSRL_sphrod,HSRL_Tamu
        # plt.rcParams['font.size'] = '17'
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
        

        # Plot the AOD data
        y = [0,1,2,0,1,2,]
        x = np.repeat((0,1),3)
        if Spheriod['r'].shape[0] ==2 :
            mode_v = ["fine", "dust","marine", 'NonSphMarine']
        if Spheriod['r'].shape[0] ==3 :
            mode_v = ["fine", "dust","marine", 'NonSphMarine']
        linestyle =[':', '-','-.','-']
        mode_v = ["fine", "dust","marine", 'NonSphMarine']

        


        if key1== None:
            key1 ='Sphroid'
        if key2 == None:
            key2 = 'Hex'

        #Retrivals:
        fig, axs2 = plt.subplots(nrows= 3, ncols=2, figsize=(38, 20))
        for i in range(len(Retrival)):
            a,b = i%3,i%2

            lambda_ticks1 = np.round(Spheriod['lambda'], decimals=2)
            lambda_ticks2 = np.round(Hex['lambda'], decimals=2)

            lambda_ticks_str1 = [str(x) for x in lambda_ticks1]
            lambda_ticks_str2 = [str(x) for x in lambda_ticks2]
            
            for mode in range(NoMode): #for each modes
                print ("Mode",mode)
                
                if i ==0:

                    # axs2[a,b].errorbar(Spheriod['r'][mode], Spheriod[Retrival[i]][mode],xerr=UNCERT['rv'],capsize=5,capthick =2, marker = "$O$",color = cm_sp[mode],lw = 3,ls = linestyle[mode], label=f"Sphrod_{mode_v[mode]}")
                    # axs2[a,b].errorbar(Hex['r'][mode],Hex[Retrival[i]][mode], marker = "H",xerr=UNCERT['rv'],capsize=5,capthick =2, color = cm_t[mode] ,lw = 3, ls = linestyle[mode],label=f"Hex_{mode_v[mode]}")
                    if NoMode == 1: 
                        axs2[a,b].plot(Spheriod['r'], Spheriod[Retrival[i]][mode], marker = "$O$",color = cm_sp[mode],lw = 5.5,ls = linestyle[mode], label=f"{key1}_{mode_v[mode]}")
                        axs2[a,b].plot(Hex['r'],Hex[Retrival[i]][mode], marker = "H", color = cm_t[mode] ,lw = 5.5, ls = linestyle[mode],label=f"{key2}_{mode_v[mode]}")
                    
                    else:
                        axs2[a,b].plot(Spheriod['r'][mode], Spheriod[Retrival[i]][mode], marker = "$O$",color = cm_sp[mode],lw = 5.5,ls = linestyle[mode], label=f"{key1}_{mode_v[mode]}")
                        axs2[a,b].plot(Hex['r'][mode],Hex[Retrival[i]][mode], marker = "H", color = cm_t[mode] ,lw = 5.5, ls = linestyle[mode],label=f"{key2}_{mode_v[mode]}")
                    # if RSP_plot != None:
                    #     axs2[a,b].plot(RSP_plot['r'][mode],RSP_plot[Retrival[i]][mode], marker = "H", color = cm_t[mode] ,lw = 5.5, ls = linestyle[mode],label=f"{key2}_{mode_v[mode]}")
                    

                    axs2[0,0].set_xlabel(r'rv  ($ \mu m$)')
                    axs2[0,0].set_xscale("log")
                else:

                    axs2[a,b].errorbar(lambda_ticks1, Spheriod[Retrival[i]][mode],yerr=UNCERT[Retrival[i]], marker = "$O$",markeredgecolor='k',capsize=7,capthick =2,markersize=25, color = cm_sp[mode],lw = 6,ls = linestyle[mode], label=f"{key1}_{mode_v[mode]}")
                    axs2[a,b].errorbar(lambda_ticks2,Hex[Retrival[i]][mode],yerr=UNCERT[Retrival[i]],capsize=7,capthick =2,  marker = "H",markeredgecolor='k',markersize=25, color = cm_t[mode] ,lw = 6, ls = linestyle[mode],label=f"{key2}_{mode_v[mode]}")
                    
                    # axs[a,b].plot(Spheriod['lambda'], Spheriod[Retrival[i]][mode], marker = "$O$",color = cm_sp[mode],lw = 2,ls = linestyle[mode],markersize=15, label=f"Sphrod_{mode_v[mode]}")
                    # axs[a,b].plot(Hex['lambda'],Hex[Retrival[i]][mode], marker = "H",color = cm_t[mode] ,lw = 2,  ls = linestyle[mode],markersize=15, label=f"Hex_{mode_v[mode]}")
                    # lambda_ticks = np.round(Spheriod['lambda'], decimals=2)

                    
                    # lambda_ticks = np.round(Spheriod['lambda'], decimals=2)
                    # axs2[a,b].set_xticks(lambda_ticks_str2)
                    
                    # axs2[a,b].set_xticks(Spheriod['lambda'])
                    # axs[a,b].set_xticklabels(['0.41', '0.46', '0.55' , '0.67'  , '0.86'])
                    axs2[a,b].set_xlabel(r'$\lambda (\mu m)$')
                    # axs2[a,b].set_ylim(bottom=0)
            
            axs2[a,b].set_ylabel(f'{Retrival[i]}')
            
            
        # axs[2,1].plot(Spheriod['lambda'], Spheriod['aod'], marker = "$O$",color = color_sph,markersize=15,lw = 2, label=f"Sphroid")
        # axs[2,1].plot(Hex['lambda'], Hex['aod'], marker = "H", color = color_tamu ,markersize=15,lw = 2, label=f"Hexahedral")
        
        axs2[2,1].errorbar(lambda_ticks1, Spheriod['aod'],yerr=0.03+UNCERT['aod']*Spheriod['aod'], marker = "$O$",color = color_sph,markeredgecolor='k',capsize=7,capthick =2,markersize=25,lw = 4, label=f"{key1}")
        axs2[2,1].errorbar(lambda_ticks2, Hex['aod'],yerr=0.03+UNCERT['aod']* Hex['aod'], marker = "H", color = color_tamu ,lw = 4,markeredgecolor='k',capsize=7,capthick =2,markersize=25, label=f"{key2}")
        # axs2[2,1].set_xticks(lambda_ticks_str)  
        # axs[2,1].set_xticklabels(Spheriod['lambda'],rotation=45)
        # axs[2,1].set_xticklabels(['0.41', '0.46', '0.55' , '0.67'  , '0.86'])
        meas_aod = []
        for i in range (NoMode):
            extrapolAOD = Hsph['meas_VExt'][:,i][-1]*Hsph['range'][i,:][-1]
            print("ExtraAOD", extrapolAOD)

            meas_aod.append(np.trapz(Hsph['meas_VExt'][:,i][::-1],Hsph['range'][i,:][::-1] )+ extrapolAOD ) 
        
        # meas_aodsph = []
        # for i in range (NoMode):
        #     meas_aodsph.append(np.trapz(Hsph['fit_VExt'][:,i][::-1],Hsph['range'][i,:][::-1] ))
        # meas_aodhex = []
        # for i in range (NoMode):
        #     meas_aodhex.append(np.trapz(HTam['fit_VExt'][:,i][::-1],HTam['range'][i,:][::-1] ))
        
        axs2[2,1].plot( Spheriod['lambda'],meas_aod,color = "k", ls = "--", marker = '*' ,markersize=20, label = " cal meas")    
        # axs[2,1].plot( Spheriod['lambda'],meas_aodsph,color = "r", ls = "-.", marker = '*' ,markersize=20, label = "cal sph")    
        # axs[2,1].plot( Spheriod['lambda'],meas_aod,color = "b", ls = "-.", marker = '*' ,markersize=20, label = "cal hex")    
       
        
        axs2[2,1].set_xlabel(r'$\lambda$')
        axs2[2,1].set_ylabel('Total AOD')
        axs2[0,0].legend(prop = { "size": 22 }, ncol=2)
        axs2[2,1].legend( ncol=2)
        lat_t = Hex['latitude']
        lon_t = Hex['longitude']
        dt_t = Hex['datetime']
        # chisph,chihex = Spheriod['costVal'] ,Hex['costVal']

        if len(lambda_ticks_str2) <4:
            plt.suptitle(f" HSRL-2 Aerosol Retrieval \n  {lat_t}N,{lon_t}E  {dt_t}\n")
        if len(lambda_ticks_str2) >3:
            plt.suptitle(f" HSRL-2+ RSP Aerosol Retrieval \n  {lat_t}N,{lon_t}E  {dt_t}\n")
        
                          
        plt.subplots_adjust(top=0.99)
        plt.tight_layout()
  
        
        # pdf_pages.savefig()
        # pdf_pages.close()
        fig.savefig(f'/data/home/gregmi/GRASP_V2/ORACLES_LiDAR_MAP_rslts/Case1/V2{dt_t}{NoMode}{key1}_{key2}_HSRL2Retrieval.png', dpi = 400)
        plt.rcParams['font.size'] = '15'
        fig, axs = plt.subplots(nrows= 1, ncols=2, figsize=(10, 5), sharey=True)
        
        for i in range(Spheriod['r'].shape[0]):
            #Hsph,HTam
            axs[0].plot(Hsph['βext'][i],Hsph['range'][i]/1000, label =i+1)
            axs[1].plot(HTam['βext'][i],HTam['range'][i]/1000, label =i+1)
            axs[1].set_xlabel('βext')
            axs[0].set_title('Spheriod')
            axs[0].set_xlabel('βext')
            axs[0].set_ylabel('Height (km)')
            axs[1].set_title('Hexahedral')
        plt.legend()
        plt.tight_layout()

    return

def CombinedLidarPolPlot(LidarPolSph,LidarPolTAMU,RSP_PixNo, UNCERT): #should be updated to 
    plt.rcParams['font.size'] = '14'
    fig, axs= plt.subplots(nrows = 1, ncols =3, figsize= (15,6))

    altd =LidarPolTAMU[0]['RangeLidar'][:,0]/1000
    HTam = LidarPolTAMU[0]
    plt.subplots_adjust(top=0.78)
    IndexH = [0,3,7]
    for i in range(3):

        wave = str(LidarPolSph[0]['lambda'][IndexH[i]]) +"μm"

        axs[i].plot(LidarPolSph[0]['meas_VBS'][:,IndexH[i]],(LidarPolSph[0]['RangeLidar'][:,0])/1000, marker =">",color = "#3B270C", label ="Meas")
        axs[i].errorbar(LidarPolSph[0]['meas_VBS'][:,IndexH[i]],(LidarPolSph[0]['RangeLidar'][:,0])/1000,xerr= UNCERT['VBS'],color = "#695E93", capsize=3,capthick =1,alpha =0.4, label =f"{UNCERT['VBS']}")
        axs[i].plot(LidarPolSph[0]['fit_VBS'][:,IndexH[i]],(LidarPolSph[0]['RangeLidar'][:,0])/1000, marker = "$O$",color = "#025043",label ="Sphd")
        axs[i].plot(LidarPolTAMU[0]['fit_VBS'][:,IndexH[i]],altd,color = "#d24787",ls = "--", label="Hex",marker = "h")

        axs[i].set_xlabel('VBS')
        axs[i].set_title(wave)
        if i ==0:
            axs[0].legend()
        plt.suptitle(f"Vertical Backscatter profile Fits(HSRL2+RSP)  \n {LidarPolTAMU[0]['latitude']}N,{LidarPolTAMU[0]['longitude']}E {HTam['datetime']} ") #Initial condition strictly constrainted by RSP retrievals
    fig.savefig(f'/data/home/gregmi/LIDAR_MAP_dust/ORACLES_LiDAR_MAP_rslts/Case1/FIT_Combined_DP.png',dpi = 300)

    fig, axs= plt.subplots(nrows = 1, ncols =3, figsize= (15,6))
    plt.subplots_adjust(top=0.78)
    for i in range(3):
        wave = str(LidarPolSph[0]['lambda'][IndexH[i]]) +"μm"

        axs[i].errorbar(LidarPolSph[0]['meas_DP'][:,IndexH[i]],(LidarPolSph[0]['RangeLidar'][:,0])/1000, marker =">",xerr= UNCERT['DP'],color = "#695E93", capsize=3,capthick =1,alpha =0.4, label =f"{UNCERT['DP']}%")
        

        axs[i].plot(LidarPolSph[0]['meas_DP'][:,IndexH[i]],(LidarPolSph[0]['RangeLidar'][:,0])/1000, marker =">",color = "#3B270C", label ="Meas")
        axs[i].plot(LidarPolSph[0]['fit_DP'][:,IndexH[i]],(LidarPolSph[0]['RangeLidar'][:,0])/1000,color = "#025043", marker = "$O$",label ="Sphd")
        axs[i].plot(LidarPolTAMU[0]['fit_DP'][:,IndexH[i]],altd,color = "#d24787", ls = "--",marker = "h", label="Hex")
        axs[i].set_xlabel('DP %')
        axs[i].set_title(wave)
        if i ==0:
            axs[0].legend()
        plt.suptitle(f" Depolarization Ratio Profile Fit(HSR2L+RSP) \n {LidarPolTAMU[0]['latitude']}N,{LidarPolTAMU[0]['longitude']}E {HTam['datetime']}") #Initial condition strictly constrainted by RSP retrievals
    fig.savefig(f'/data/home/gregmi/LIDAR_MAP_dust/ORACLES_LiDAR_MAP_rslts/Case1/FIT_Combined_VBS.png',dpi = 300)

    fig, axs= plt.subplots(nrows = 1, ncols =2, figsize= (9,6))
    plt.subplots_adjust(top=0.78)
    for i in range(2):
        wave = str(LidarPolSph[0]['lambda'][IndexH[i]]) +"μm"

        axs[i].errorbar(LidarPolSph[0]['meas_VExt'][:,IndexH[i]],(LidarPolSph[0]['RangeLidar'][:,0])/1000, marker =">",xerr= LidarPolSph[0]['meas_VExt'][:,IndexH[i]]*UNCERT['VEXT'],color = "#695E93", capsize=3,capthick =1,alpha =0.4, label =f"{UNCERT['VEXT']*100}%")
        
        axs[i].plot(LidarPolSph[0]['meas_VExt'][:,IndexH[i]],(LidarPolSph[0]['RangeLidar'][:,0])/1000, marker =">",color = "#3B270C", label ="Meas")
        axs[i].plot(LidarPolSph[0]['fit_VExt'][:,IndexH[i]],(LidarPolSph[0]['RangeLidar'][:,0])/1000,color = "#025043", marker = "$O$",label ="Sphd")
        axs[i].plot(LidarPolTAMU[0]['fit_VExt'][:,IndexH[i]],altd,color = "#d24787",ls = "--", marker = "h", label="Hex")
        axs[i].set_xlabel('VExt')
        axs[i].set_title(wave)
        if i ==0:
            axs[0].legend()
        plt.suptitle(f"Vertical Extinction Profile Fit (HSR2L+RSP) \n {LidarPolTAMU[0]['latitude']}N,{LidarPolTAMU[0]['longitude']}E {HTam['datetime']}\n") #Initial condition strictly constrainted by RSP retrievals
    
    
    fig.savefig(f'/data/home/gregmi/LIDAR_MAP_dust/ORACLES_LiDAR_MAP_rslts/Case1/FIT_Combined_Vertical_EXT_profile.png',dpi = 300)
    fig, axs= plt.subplots(nrows = 1, ncols =2, figsize= (10,5))
    for i in [1,2,4,5,6]:

        
        # axs[0].errorbar(LidarPolSph[0]['sca_ang'][:,i], LidarPolSph[0]['meas_I'][:,i],yerr=  LidarPolSph[0]['meas_I'][:,i]*UNCERT['I'],color = "#695E93", capsize=3,capthick =1,alpha =0.4, label =f"{UNCERT['I']}")
        axs[0].fill_between(LidarPolSph[0]['sca_ang'][:,i], LidarPolSph[0]['meas_I'][:,i], LidarPolSph[0]['meas_I'][:,i]*1.03, color = 'r',alpha=0.2, ls = "--",label="+3%")
        axs[0].fill_between(LidarPolSph[0]['sca_ang'][:,i], LidarPolSph[0]['meas_I'][:,i], LidarPolSph[0]['meas_I'][:,i]*0.97, color = 'b',alpha=0.2, ls = "--",label="-3%")
           
        axs[0].plot(LidarPolSph[0]['sca_ang'][:,i], LidarPolSph[0]['meas_I'][:,i],color = "#3B270C",lw= 2, label ="Meas")
        axs[0].plot(LidarPolSph[0]['sca_ang'][:,i],LidarPolSph[0]['fit_I'][:,i],color = "#025043",lw= 2, label ="Sphd")
        axs[0].plot(LidarPolSph[0]['sca_ang'][:,i],LidarPolTAMU[0]['fit_I'][:,i],color = "#d24787",lw= 2,ls = "--", label="Hex")
        axs[0].set_ylabel('I')
        axs[0].set_xlabel('Scattering angles')
        
        plt.suptitle(f"HSR2L+RSP Retrieval: I and DOLP fit at {LidarPolSph[0]['lambda'][i]}μm , \n {LidarPolTAMU[0]['latitude']}N,{LidarPolTAMU[0]['longitude']}E {HTam['datetime']} ") #Initial condition strictly constrainted by RSP retrievals
        axs[1].fill_between(LidarPolSph[0]['sca_ang'][:,i], LidarPolSph[0]['meas_P_rel'][:,i], LidarPolSph[0]['meas_P_rel'][:,i]+0.005, color = 'r',alpha=0.2, ls = "--",label="+3%")
        axs[1].fill_between(LidarPolSph[0]['sca_ang'][:,i], LidarPolSph[0]['meas_P_rel'][:,i], LidarPolSph[0]['meas_P_rel'][:,i]-0.005, color = 'b',alpha=0.2, ls = "--",label="-3%")
        
        axs[1].plot(LidarPolSph[0]['sca_ang'][:,i],LidarPolSph[0]['meas_P_rel'][:,i],color = "#3B270C",lw= 2, label ="Meas")
        axs[1].plot(LidarPolSph[0]['sca_ang'][:,i],LidarPolSph[0]['fit_P_rel'][:,i],color = "#025043",lw= 2,label ="Sphd")
        axs[1].plot(LidarPolSph[0]['sca_ang'][:,i],LidarPolTAMU[0]['fit_P_rel'][:,i],color = "#d24787",lw= 2,ls = "--", label="Hex")
        axs[1].set_ylabel('P/I')
        axs[1].set_xlabel('Scattering angles')
        axs[0].legend()
        # axs[1].set_title()
        fig.savefig(f'/data/home/gregmi/LIDAR_MAP_dust/ORACLES_LiDAR_MAP_rslts/Case1/FIT_{RSP_PixNo}_Combined_I_P_{i}.png',dpi = 300)

        fig, axs= plt.subplots(nrows = 1, ncols =2, figsize= (10,5))
       
def RSP_plot(rslts_Sph,rslts_Tamu,RSP_PixNo,UNCERT,rslts_Sph2=None,rslts_Tamu2=None, LIDARPOL= None, fn = "None",ModeComp = None): 
    
    if (rslts_Sph2!=None )and (rslts_Tamu2!=None):
        RepMode =2
    else:
        RepMode =1
    
    
    Spheriod,Hex = rslts_Sph[0],rslts_Tamu[0]

    cm_sp = ['k','#8B4000', '#87C1FF']
    cm_t = ["#BC106F",'#E35335', 'b']
    
    # color_sph = "#025043"
    color_sph = '#0c7683'
    color_tamu = "#d24787"

    fig, axs2 = plt.subplots(nrows= 3, ncols=2, figsize=(35, 20))
    fig, axsErr = plt.subplots(nrows= 3, ncols=2, figsize=(20, 15))
        


    for i in range(RepMode):
        if i ==1:
            Spheriod,Hex = rslts_Sph2[0],rslts_Tamu2[0]
            # cm_sp = ['#b54cb5','#14411b', '#87C1FF']
            # cm_t = ["#14411b",'#adbf4b', 'b']
            color_sph = '#adbf4b'
            color_tamu = "#936ecf"

        plt.rcParams['font.size'] = '26'
        plt.rcParams["figure.autolayout"] = True
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

                Err = Spheriod[Retrival[i]][mode] - Hex[Retrival[i]][mode]
                   
                if i ==0:

                    # axs2[a,b].errorbar(Spheriod['r'][mode], Spheriod[Retrival[i]][mode],xerr=UNCERT['rv'],capsize=5,capthick =2, marker = "$O$",color = cm_sp[mode],lw = 3,ls = linestyle[mode], label=f"Sphrod_{mode_v[mode]}")
                    # axs2[a,b].errorbar(Hex['r'][mode],Hex[Retrival[i]][mode], marker = "H",xerr=UNCERT['rv'],capsize=5,capthick =2, color = cm_t[mode] ,lw = 3, ls = linestyle[mode],label=f"Hex_{mode_v[mode]}")
                        
                    axs2[a,b].plot(Spheriod['r'][mode], Spheriod[Retrival[i]][mode], marker = "$O$",color = cm_sp[mode],lw = 2,ls = linestyle[mode], label=f"Sphrod_{mode_v[mode]}")
                    axs2[a,b].plot(Hex['r'][mode],Hex[Retrival[i]][mode], marker = "H", color = cm_t[mode] ,lw = 2, ls = linestyle[mode],label=f"Hex_{mode_v[mode]}")
                    
                    # DiffShape = Spheriod[Retrival[i]][mode] - Hex[Retrival[i]][mode]
                    # Err = DiffShape  # how much does hexhahderal vary  from spheriod

                    # if DiffShape>0: markerErr = "$S>$"
                    # if DiffShape<=0: markerErr =  "."

                    axsErr[a,b].plot(Spheriod['r'][mode], Err,  marker = "o",color = cm_sp[mode],lw = 5,ls = linestyle[mode], label=f"{mode_v[mode]}")
                    
                    
                    
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

                    axsErr[a,b].plot(lambda_ticks_str, Err,  marker = "o",color = cm_sp[mode],lw = 5,ls = linestyle[mode], label=f"{mode_v[mode]}")
                    axsErr[a,b].set_xticks(lambda_ticks_str) 
                    # axs[a,b].set_xticklabels(['0.41', '0.46', '0.55' , '0.67'  , '0.86'])
                    axsErr[a,b].set_xlabel(r'$\lambda \mu m$')
                    
                    
                    # axs[a,b].plot(Spheriod['lambda'], Spheriod[Retrival[i]][mode], marker = "$O$",color = cm_sp[mode],lw = 2,ls = linestyle[mode],markersize=15, label=f"Sphrod_{mode_v[mode]}")
                    # axs[a,b].plot(Hex['lambda'],Hex[Retrival[i]][mode], marker = "H",color = cm_t[mode] ,lw = 2,  ls = linestyle[mode],markersize=15, label=f"Hex_{mode_v[mode]}")
                    
                    axs2[a,b].set_xticks(lambda_ticks_str) 
                    # axs[a,b].set_xticklabels(['0.41', '0.46', '0.55' , '0.67'  , '0.86'])
                    axs2[a,b].set_xlabel(r'$\lambda \mu m$')
                    # axs2[a,b].set_xlim(0, np.maximum(Spheriod[Retrival[i]][mode],Hex[Retrival[i]][mode]))
            axs2[a,b].set_ylabel(f'{Retrival[i]}')
            
            
        # axs[2,1].plot(Spheriod['lambda'], Spheriod['aod'], marker = "$O$",color = color_sph,markersize=15,lw = 2, label=f"Sphroid")
        # axs[2,1].plot(Hex['lambda'], Hex['aod'], marker = "H", color = color_tamu ,markersize=15,lw = 2, label=f"Hexahedral")
        
        axs2[2,1].errorbar(lambda_ticks_str, Spheriod['aod'],yerr=0.03+UNCERT['aod']*Spheriod['aod'], marker = "$O$",color = color_sph,markeredgecolor='k',capsize=7,capthick =2,markersize=25,lw = 4, label=f"Sphroid")
        axs2[2,1].errorbar(lambda_ticks_str, Hex['aod'],yerr=0.03+UNCERT['aod']* Hex['aod'], marker = "H", color = color_tamu ,lw = 4,markeredgecolor='k',capsize=7,capthick =2,markersize=25, label=f"Hex")
        axsErr[2,1].errorbar(lambda_ticks_str, Err, marker = "o", color = color_tamu ,lw = 5,markeredgecolor='k',capsize=5,capthick =2,markersize=25, label=f"Hex")
          
        
        axs2[2,1].set_xticks(lambda_ticks_str)
        # axs[2,1].set_xticklabels(['0.41', '0.46', '0.55' , '0.67'  , '0.86'])
        axsErr[2,1].set_xlabel(r'$\lambda$')
        axsErr[2,1].set_ylabel('Total AOD')    
        axs2[2,1].set_xlabel(r'$\lambda$')
        axs2[2,1].set_ylabel('Total AOD')
        axs2[0,0].legend(prop = { "size": 21 }, ncol=2)
        
        lat_t = Hex['latitude']
        lon_t = Hex['longitude']
        dt_t = Hex['datetime']
        plt.tight_layout(rect=[0, 0, 1, 1])
        # axs2.savefig(f'/home/gregmi/ORACLES/HSRL_RSP/{dt_t}{RepMode}_RSPRetrieval{RepMode}.png', dpi = 400)
        
        plt.suptitle(f'RSP Aerosol Retrieval \n  Lat:{lat_t} Lon :{lon_t}   Date: {dt_t}')
        if RepMode ==2:
            fig.savefig(f'/data/home/gregmi/LIDAR_MAP_dust/ORACLES_LiDAR_MAP_rslts/{dt_t}2+3_RSPRetrieval.png', dpi = 400)
        else:
            fig.savefig(f'/data/home/gregmi/LIDAR_MAP_dust/ORACLES_LiDAR_MAP_rslts/{dt_t}{NoMode}_RSPRetrieval.png', dpi = 400)

        #Stokes: 
        wl = rslts_Sph[0]['lambda'] 
        fig, axs = plt.subplots(nrows= 4, ncols=len(wl), figsize=(len(wl)*7, 17),gridspec_kw={'height_ratios': [1, 0.3,1,0.3]}, sharex='col')
        # Plot the AOD data
        meas_P_rel = 'meas_P_rel'

        if LIDARPOL == True:
            wlIdx = [1,2,4,5,6]  #Lidar wavelngth indices
        else:
            wlIdx = np.arange(len(wl))


        for nwav in range(len(wlIdx)):
        # Plot the fit and measured I data
            
            axs[0, nwav].fill_between(Spheriod ['sca_ang'][:,wlIdx[nwav]],Spheriod ['meas_I'][:,wlIdx[nwav]], Spheriod ['meas_I'][:,wlIdx[nwav]]*1.03, color = 'r',alpha=0.2, ls = "--",label="+3%")
            axs[0, nwav].fill_between(Spheriod ['sca_ang'][:,wlIdx[nwav]],Spheriod ['meas_I'][:,wlIdx[nwav]], Spheriod ['meas_I'][:,wlIdx[nwav]]*0.97, color = "b",alpha=0.2, ls = "--",label="-3%")
            axs[0, nwav].plot(Spheriod['sca_ang'][:,wlIdx[nwav]], Spheriod['meas_I'][:,wlIdx[nwav]], color = "k", lw = 3, label="meas")

            axs[0, nwav].plot(Spheriod ['sca_ang'][:,wlIdx[nwav]], Spheriod ['fit_I'][:,wlIdx[nwav]],color =color_sph , lw = 3, ls = '--',label="fit sphrod")
            # axs[0, nwav].scatter(Spheriod ['sca_ang'][:,nwav][marker_indsp], Spheriod ['fit_I'][:,nwav][marker_indsp],color =color_sph , m = "o",label="fit sphrod")
            
            # axs[0, nwav].set_xlabel('Scattering angles (deg)')
            axs[0, 0].set_ylabel('I')
     

            # Plot the fit and measured QoI data
            axs[2, nwav].plot(Spheriod ['sca_ang'][:,wlIdx[nwav]], Spheriod ['meas_P_rel'][:,wlIdx[nwav]],color = "k", lw = 3, label="meas")
            axs[2, nwav].plot(Spheriod ['sca_ang'][:,wlIdx[nwav]], Spheriod ['fit_P_rel'][:,wlIdx[nwav]], color =color_sph, lw = 3, ls = '--', label="fit sph")
            
            axs[2, nwav].fill_between(Spheriod ['sca_ang'][:,wlIdx[nwav]],(Spheriod ['meas_P_rel'][:,wlIdx[nwav]]), (Spheriod ['meas_P_rel'][:,wlIdx[nwav]])*1.03,color = 'r', alpha=0.2,ls = "--", label="+3%")
            axs[2, nwav].fill_between(Spheriod ['sca_ang'][:,wlIdx[nwav]],(Spheriod ['meas_P_rel'][:,wlIdx[nwav]]), (Spheriod ['meas_P_rel'][:,wlIdx[nwav]])*0.97,color = "b", alpha=0.2,ls = "--", label="-3%")
            # axs[2, nwav].set_xlabel('Scattering angles (deg)')
            axs[2, 0].set_ylabel('DOLP')
            axs[0, nwav].set_title(f"{wl[wlIdx[nwav]]} $\mu m$", fontsize = 22)
            
            axs[0, nwav].plot(Hex['sca_ang'][:,wlIdx[nwav]], Hex['fit_I'][:,wlIdx[nwav]],color =color_tamu , lw = 3, ls = "dashdot",label="fit Hex")
            axs[2, nwav].plot(Hex['sca_ang'][:,wlIdx[nwav]],Hex['fit_P_rel'][:,wlIdx[nwav]],color = color_tamu , lw = 3,ls = "dashdot", label = "fit Hex") 


            sphErr = 100 * abs(Spheriod['meas_I'][:,wlIdx[nwav]]-Spheriod ['fit_I'][:,wlIdx[nwav]] )/Spheriod['meas_I'][:,wlIdx[nwav]]
            HexErr = 100 * abs(Hex['meas_I'][:,nwav]-Hex['fit_I'][:,nwav] )/Hex['meas_I'][:,wlIdx[nwav]]
            
            axs[1, nwav].plot(Spheriod ['sca_ang'][:,wlIdx[nwav]], sphErr,color =color_sph , lw = 3,label="Sphrod")
            axs[1, nwav].plot(Hex ['sca_ang'][:,wlIdx[nwav]], HexErr,color = color_tamu, lw = 3 ,label="Hex")
            axs[1, 0].set_ylabel('Err I %')
            
    #Absolute error
            sphErrP =  abs(Spheriod['meas_P_rel'][:,wlIdx[nwav]]-Spheriod ['fit_P_rel'][:,wlIdx[nwav]])
            HexErrP =  abs(Hex['meas_P_rel'][:,wlIdx[nwav]]-Hex['fit_P_rel'][:,wlIdx[nwav]] )
            
            axs[3, nwav].plot(Spheriod ['sca_ang'][:,wlIdx[nwav]], sphErrP,color =color_sph , lw = 3,label="Sphrod")
            axs[3, nwav].plot(Hex ['sca_ang'][:,wlIdx[nwav]], HexErrP,color =color_tamu, lw = 3 ,label="Hex")
            
            axs[3, nwav].set_xlabel(r'$\theta$s(deg)')
            
            axs[3, nwav].set_xlabel(r'$\theta$s(deg)')
            axs[3, 0].set_ylabel('|Meas-fit|')
        
            # axs[1, nwav].set_title(f"{wl[nwav]}", fontsize = 14)

            if i == 0:

                axs[0, 4].legend(bbox_to_anchor=(1.05, 0.5), loc='center left')
     
            plt.suptitle(f'RSP Aerosol Retrieval \n  Lat:{lat_t} Lon :{lon_t}   Date: {dt_t}')
            if fn ==None : fn = "1"
            fig.savefig(f'/data/home/gregmi/LIDAR_MAP_dust/ORACLES_LiDAR_MAP_rslts/{dt_t}_RSPFits_{fn}.png', dpi = 400)

            plt.tight_layout(rect=[0, 0, 1, 1])

    plt.show()

def Ext2Vconc(botLayer,topLayer,Nlayers,wl, Shape, AeroType = None, ):


    "Run GRASP forward model"

    dict_of_dicts = {}
    
    krnlPath='/data/home/gregmi/GRASP_V2/grasp-dev-rtm-v120-new-inter/src/retrieval/internal_files'
    binPathGRASP ='/home/shared/GRASP_GSFC/build_HEX_v112/bin/grasp_app'
    ymlPath = '/home/gregmi/git/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/'
    UpKerFile =  'settings_dust_Vext_conc_dump.yml'  #Yaml file after updating the state vectors

    fwdModelYAMLpath ='/home/gregmi/git/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/settings_dust_Vext_conc.yml'
    
    #
    #all values are set for 0.532 nm values in the ymal file.
    if 'dust' in AeroType.lower():
        rv,sigma, n, k, sf = 2, 0.5, 1.55, 0.004, 1e-6
    if 'seasalt' in AeroType.lower():
        rv,sigma, n, k, sf = 1.5 , 0.6, 1.33, 0.0001, 0.999          
    if 'fine' in AeroType.lower():
        rv,sigma, n, k, sf = 0.13, 0.49, 1.45, 0.003, 1e-6
    
    #Setting the shape model Sphroids and Hexhedral
    if 'sph' in Shape.lower():
        Kernel = 'KERNELS_BASE'
        print(Kernel)
    if 'hex' in Shape.lower():
        Kernel = 'Ver_sph'
        print(Kernel)

    Volume = np.linspace(1e-5,2,20)
    VConcandExt = np.zeros((len(Volume),2))*np.nan

#Geometry
    singProf = np.linspace(botLayer, topLayer, Nlayers)[::-1]
    #Inputs for simulation
    wvls = [0.532] # wavelengths in μm
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

    return dict_of_dicts, VConcandExt
   


# botLayer = 2000
# topLayer = 4000
# Nlayers = 3

# v_sph_dust= Ext2Vconc(botLayer,topLayer,Nlayers,wl=0.532, AeroType ='dust',Shape='sph')
# v_hex_dust= Ext2Vconc(botLayer,topLayer,Nlayers,wl=0.532, AeroType ='dust',Shape='hex')

# v_sph_sea= Ext2Vconc(botLayer,topLayer,Nlayers,wl=0.532, Shape='sph', AeroType ='seasalt',)
# v_hex_sea= Ext2Vconc(botLayer,topLayer,Nlayers,wl=0.532, AeroType ='seasalt', Shape='hex')

# v_sph_fine= Ext2Vconc(botLayer,topLayer,Nlayers,wl=0.532, AeroType ='fine',Shape='sph')
# v_hex_fine= Ext2Vconc(botLayer,topLayer,Nlayers,wl=0.532, AeroType ='fine',Shape='hex')

def PlotE2C(v,name = None):

    m, b = np.polyfit( v[1][:,1],v[1][:,0], 1) #slope and y intercept
    

    ax = plt.gca()
    yScalarFormatter = ScalarFormatterClass(useMathText=True)
    yScalarFormatter.set_powerlimits((0,0))
    ax.xaxis.set_major_formatter(yScalarFormatter)

    ax.plot(v[1][:,1],v[1][:,0], 'yo', v[1][:,1], m*v[1][:,1]+b, '--k', label ="fir")
    ax.scatter(v[1][:,1],v[1][:,0], label = 'GRASP Values')
    ax.set_xlabel('Vext m-1')
    ax.set_ylabel('Volume Conc')
    ax.set_title(f'Slope: {m},\n Yintercept = {b}')

    plt.savefig(f'{name}_VConc2Ext.png' , dpi = 200)


    return m,b
# #Plot Vext VS volume Conc

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
    colorwl = ['#70369d','#4682B4','#01452c','#FF7F7F','#d4af37','#4c0000','c','k']
    
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

def PlotcombEachMode(rslts_Sph2,rslts_Tamu2,HSRL_sphrodT,HSRL_TamuT,LidarPolSph=None,LidarPolTAMU=None, HiGEAR=None):



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

    fig, axs2 = plt.subplots(nrows= 6, ncols=3, figsize=(65, 80))
    # fig, axsErr = plt.subplots(nrows= 6, ncols=3, figsize=(10, 15))
        
    fig.patch.set_alpha(0.0)
    # Plot the AOD data
    y = [0,1,2,0,1,2,]
    x = np.repeat((0,1),3)

    

    lambda_ticks = np.round(Spheriod['lambda'], decimals=2)
    lambda_ticks_str = [str(x) for x in lambda_ticks]

    NoMode =Spheriod['MapR'].shape[0]
    if Spheriod['MapR'].shape[0] ==2 :
            mode_v = ["Fine", "Dust","Marine"]
    if Spheriod['MapR'].shape[0] ==3 :
        mode_v = ["Fine", "Dust","Marine"]
    linestyle =['-', '-','-.']

    cm_spj = ['#4459AA','#568FA0', 'r']
    cm_tj = ["#FB6807",'#C18BB7', 'y']

    
    for RetNo in range(RepMode):

        if  RetNo ==1: 
            Spheriod,Hex = LidarPolSph[0][0],LidarPolTAMU[0][0]


            # cm_sp = ['#757565','#5F381A', '#4BCBE2']
            # cm_t =  ["#882255",'#D44B15', '#1E346D']


            # cm_sp2 = ['b','#BFBF2A', '#844772']
            # cm_t2 = ["#14411b",'#936ecf', '#FFA500']

           


        if  RetNo ==0:

            cm_t = ['#14411b','#94673E','#00B2FF']
            cm_sp = ['#8A898E','#DCD98D','#23969A']


            cm_t2 = ["#882255",'#FE5200','#8F64B3']
            cm_sp2 = ["#DEA520",'#44440F','#212869']


            


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

                  
                    axs2[a,b].plot(Spheriod['MapR'][mode], Spheriod['MapVol'][mode], marker = "$O$",color = cm_sp[mode],lw = 15,ls = linestyle[mode], label=f"RSP_Sphd")
                    axs2[a,b].plot(Hex['MapR'][mode],Hex['MapVol'][mode], marker = "H", color=cm_t[mode] ,lw = 10, ls = linestyle[mode],label=f"RSP_Hex")
                    

                    axs2[a,b].plot(Spheriod['lidarR'][mode], Spheriod['lidarVol'][mode], marker = "$O$",color = cm_sp2[mode],lw = 15,ls = linestyle[mode], label=f"HSRL_Sphd")
                    axs2[a,b].plot(Hex['lidarR'][mode],Hex['lidarVol'][mode], marker = "H", color = cm_t2[mode]  ,lw = 15, ls = linestyle[mode],label=f"HSRL_Hex")
                    
                    # axsErr[a,b].plot(Spheriod['MapR'][mode], ErrMap,  marker = "o",color = cm_sp[mode],lw = 15,ls = linestyle[mode], label=f"{mode_v[mode]}")
                    # axsErr[a,b].plot(Spheriod['MapR'][mode], ErrLidar,  marker = "o",color = cm_t2[mode],lw = 15,ls = linestyle[mode], label=f"{mode_v[mode]}")
                    
                    axs2[a,b].set_xlabel(r'rv $ \mu m$')
                    axs2[a,b].set_ylabel(r'dVdlnr')
                    axs2[a,b].set_xscale("log")
                    axs2[a,b].set_title(f'{mode_v[mode]}', weight='bold')

                    # axsErr[a,b].set_xlabel(r'rv $ \mu m$')
                    # axsErr[a,b].set_xscale("log")
                    axs2[a,b].legend()

                if i ==0 and RetNo ==1: #Plotting single sensor retrievals

                    cm_sp= cm_sp2
                    cm_t = cm_t2


                    color_sph2 = '#adbf4b'
                    color_tamu2 = "#936ecf"


                    if HiGEAR !=None:

                        dV_HiGEAR, r_HiGEAR = HiGEAR
                        axs2[a,b].plot(r_HiGEAR, dV_HiGEAR, marker = "$*$",color = cm_spj[mode],lw = 15,ls = linestyle[mode], label=f"HIGEAR")

        


                  
                    axs2[a,b].plot(Spheriod['r'][mode], Spheriod[Retrival[i]][mode], marker = "$O$",color = cm_spj[mode],lw = 15,ls = linestyle[mode], label=f"Joint_Sphd")
                    axs2[a,b].plot(Hex['r'][mode],Hex[Retrival[i]][mode], marker = "H", color=cm_tj[mode] ,lw = 15, ls = linestyle[mode],label=f"Joint_Hex")
                    

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
                    axs2[a,b].legend()
                    # axs2[a,b].legend(prop = { "size": 22 },loc='upper center', bbox_to_anchor=(0.5, -0.4), shadow=True, ncol=3)

                if i>0:
                    
                    axs2[a,b].errorbar(lambda_ticks_str,Hex[Retrival[i]][mode],yerr=UNCERT[Retrival[i]], color = cm_t[mode] ,lw = 1, ls = linestyle[mode])
                    axs2[a,b].errorbar(lambda_ticks_str, Spheriod[Retrival[i]][mode],yerr=UNCERT[Retrival[i]], color = cm_sp[mode],lw = 1,ls = linestyle[mode])
                    

                    for scp in range(len(lambda_ticks_str)):
                        axs2[a,b].errorbar(lambda_ticks_str[scp],Hex[Retrival[i]][mode][scp],capsize=7,capthick =2,  marker =  markerHex[scp],markersize=50, color = cm_t[mode] ,lw = 1,alpha = 0.8, ls = linestyle[mode])
                        axs2[a,b].errorbar(lambda_ticks_str[scp], Spheriod[Retrival[i]][mode][scp], marker = markerSph[scp],markeredgecolor='k',capsize=7,capthick =2,markersize=55,alpha = 0.8,color = cm_sp[mode],lw = 1,ls = linestyle[mode])
                        
                    for scp2 in range(1):
                        axs2[a,b].errorbar(lambda_ticks_str[scp2],Hex[Retrival[i]][mode][scp2],capsize=7,capthick =2,  marker =  markerHex[scp],markersize=50, color = cm_t[mode] ,lw = 1,alpha = 0.8, ls = linestyle[mode], label=f"{keyInd}Hex")
                        axs2[a,b].errorbar(lambda_ticks_str[scp2], Spheriod[Retrival[i]][mode][scp2], marker = markerSph[scp],markeredgecolor='k',capsize=7,capthick =2,markersize=55,alpha = 0.8,color = cm_sp[mode],lw = 1,ls = linestyle[mode], label=f"{keyInd}Sphd")
                      
                        if scp2 ==0:
                            if a ==2:
                                axs2[a,b].legend(prop = { "size": 45 },ncol=2)
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
            axs2[5,1].errorbar(lambda_ticks_str[scp], Spheriod['aod'][scp], marker = markerSph[scp],markeredgecolor='k',capsize=7,capthick =2,markersize=50, color = cm_sp[mode])
            axs2[5,1].errorbar(lambda_ticks_str[scp],Hex['aod'][scp],capsize=7,capthick =2,  marker =  markerHex[scp],markeredgecolor='k',markersize=55, color = cm_t[mode])
            # if scp ==1:
            #     axs2[7,1].legend()
        for scp in range(1):
            axs2[5,1].errorbar(lambda_ticks_str[scp], Spheriod['aod'][scp], marker = markerSph[scp],markeredgecolor='k',capsize=7,capthick =2,markersize=50, color = cm_sp[mode], label=f"{keyInd}Sphd")
            axs2[5,1].errorbar(lambda_ticks_str[scp],Hex['aod'][scp],capsize=7,capthick =2,  marker =  markerHex[scp],markeredgecolor='k',markersize=55, color = cm_t[mode],label=f"{keyInd}Hex")
            axs2[5,1].legend(prop = { "size": 18 })
                
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

        if RetNo ==0:
            axs2[5,2].scatter(mode_v,rslts_Sph[0]['sph'], color=cm_sp[mode], marker = "$RO$",s=1500,label = "RSPsph")
            axs2[5,2].scatter(mode_v,rslts_Tamu[0]['sph'],color=cm_t[mode], marker = "$RH$",s=1500, label = "RSPhex")
            axs2[5,2].scatter(mode_v,HSRL_sphrodT[0][0]['sph'],color=cm_sp2[mode],  marker = "$HO$",s=1500, label = "HSRLsph")
            axs2[5,2].scatter(mode_v,HSRL_TamuT[0][0]['sph'], color=cm_t2[mode],marker = "$HH$",s=1500, label = "HSRLhex")
            # axs2[6,1].legend(prop = { "size": 21 },ncol=1)
            axs2[5,2].set_xlabel("Aerosol type", weight='bold')
            axs2[5,2].set_ylabel('Spherical Frac', weight='bold')


            axs2[5,0].scatter('RSP SPh',rslts_Sph[0]['costVal'], color=cm_sp[mode],s=1400, label = "RSPsph")
            axs2[5,0].scatter('RSP Hex',rslts_Tamu[0]['costVal'], color=cm_t[mode],s=1400,label = "RSPhex")
            axs2[5,0].scatter('HSRL SPh',HSRL_sphrodT[0][0]['costVal'], color=cm_sp2[mode],s=1400,label = "RSPsph")
            axs2[5,0].scatter('HSRL Hex',HSRL_TamuT[0][0]['costVal'], color=cm_t2[mode],s=1400,label = "RSPhex")
            # axs2[8,1].legend()
            # axs2[8,1].set_xlabel("CostVal")
            axs2[5,0].set_ylabel("CostVal", weight='bold')
        
        if RetNo ==1:

            # for scp in range(3):
            axs2[5,2].errorbar(mode_v[scp],LidarPolSph[0][0]['sph'][scp], marker = '+',markeredgecolor='k',capsize=7,capthick =2,markersize=50, color = cm_sp[mode],label = "Jointsph")
            axs2[5,2].errorbar(mode_v[scp],LidarPolSph[0][0]['sph'][scp],capsize=7,capthick =2,  marker =  '*',markeredgecolor='k',markersize=55, color = cm_t[mode],label = "Jointsph")
        
            
            axs2[5,2].legend(prop = { "size": 18 },ncol=1)
            axs2[5,2].set_xlabel("Aerosol type", weight='bold')
            axs2[5,2].set_ylabel('Spherical Frac', weight='bold')


            axs2[5,0].scatter('Joint SPh',LidarPolSph[0][0]['costVal'], color='r',s=400, label = "RSPsph")
            axs2[5,0].scatter('Joint Hex',LidarPolSph[0][0]['costVal'], color='k',s=400,label = "RSPhex")
            # axs2[8,1].legend()
            # axs2[8,1].set_xlabel("CostVal")
            axs2[5,0].set_ylabel("CostVal", weight='bold')

        # axs2[6,1].legend(prop = { "size": 18 },loc='upper center', bbox_to_anchor=(0.5, -0.4), shadow=True, ncol=4)
        # axs2[8,1].legend(prop = { "size": 18 },loc='upper center', bbox_to_anchor=(0.5, -0.4), shadow=True, ncol=4)
        # axs2[7,1].legend(prop = { "size": 18 },loc='upper center', bbox_to_anchor=(0.5, -0.4), shadow=True, ncol=4)
                 


        # lat_t = Hex['latitude']
        # lon_t = Hex['longitude']
        # dt_t = Hex['datetime']
        # plt.tight_layout(rect=[0, 0, 1, 1])
        
        plt.savefig(f'/home/gregmi/ORACLES/HSRL_RSP/{RepMode}_RSPRetrieval{RepMode}.png', dpi = 400 ,transparent=True)
        
        # plt.suptitle(f'RSP Aerosol Retrieval \n  Lat:{lat_t} Lon :{lon_t}   Date: {dt_t}')
        if RepMode ==2:
            fig.savefig(f'/home/gregmi/ORACLES/HSRL_RSP/2+3_RSPRetrieval.png', dpi = 400)
        else:
            fig.savefig(f'/home/gregmi/ORACLES/HSRL_RSP/{NoMode}_RSPRetrieval.png', dpi = 400)

    
def VolEqSph_to_VolEqHex(rv_sph ,psi_sph, psi_hex ):
    
    """
        rv_sph: volume equivalent radius of spheroid
        
        psi_sph : ensemble weighted degree of sphericity of  spheroids
        
        psi_hex : ensemble weighted degree of sphericity of hexahderals
               
    """
    
    
    #This is assuming that Reff radius for hexahedrals and spheriods are the same. 
    
    rv_hex = (psi_sph/psi_hex)*rv_sph
    
    return rv_hex


def ComparePhaseFunct(rv,sigma):


    """
    rv: lsit of volume equivalent radius

    """

    "Run GRASP forward model for different shapes"

    Kernel =['Ver_sph' , 'KERNELS_BASE']
    sf =[1e-6, 1]


    Shape_rsltdict = {}

    
    krnlPath='/data/home/gregmi/GRASP_V2/grasp-dev-rtm-v120-new-inter/src/retrieval/internal_files'
    binPathGRASP ='/home/shared/GRASP_GSFC/build_HEX_v112/bin/grasp_app'
    ymlPath = '/home/gregmi/git/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/'
    UpKerFile =  'settings_FWD_dust_RSP.yml'  #Yaml file after updating the state vectors

    UpKerFile_dump = 'settings_FWD_dust_RSP_dump.yml'



    fwdModelYAMLpath ='/home/gregmi/git/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/settings_dust_Vext_conc.yml'
    
    #

    #Geometry
    #Inputs for simulation
    sza = 40 # solar zenith angle
    wvls = [0.355,0.410,0.470,0.532,0.55,0.670,0.865,1.064] # wavelengths in μm
    msTyp = [41, 46 ] # grasp measurements types (I, Q, U) [must be in ascending order]
    azmthΑng = np.r_[0:180:10] # azimuth angles to simulate (0,10,...,175)
    vza = np.r_[0:75:5] # viewing zenith angles to simulate [in GRASP cord. sys.]
    upwardLooking = False # False -> downward looking imagers, True -> upward looking


    Nvza = len(vza)
    Nazimth = len(azmthΑng)
    thtv_g = np.tile(vza, len(msTyp)*len(azmthΑng)) # create list of all viewing zenith angles in (Nvza X Nazimth X 3) grid
    phi_g = np.tile(np.concatenate([np.repeat(φ, len(vza)) for φ in azmthΑng]), len(msTyp)) # create list of all relative azimuth angles in (Nvza X Nazimth X 3) grid
    nbvm = len(thtv_g)/len(msTyp)*np.ones(len(msTyp), int) # number of view angles per Stokes element (Nvza X Nazimth)
    meas = np.r_[np.repeat(0.1, nbvm[0]), np.repeat(0.01, nbvm[1])] # dummy measurement values for I, Q and U, respectively 

    nowPix = pixel(dt.datetime.now(), 1, 1, 0, 0, masl=0, land_prct=0)
    for wvl in wvls: nowPix.addMeas(wvl, msTyp, nbvm, sza, thtv_g, phi_g, meas)


    if isinstance(rv, (list, tuple, np.ndarray)):
        loop_count = len(rv)
        
    else:
        loop_count = 1
        

            
    for i in range(loop_count):
        with open(ymlPath+UpKerFile, 'r') as f:  
            data = yaml.safe_load(f) 
            # print(data)
            # data['retrieval']['constraints']['characteristic[2]']['mode[1]']['initial_guess']['value'][0] = float(Volume[i])
            if loop_count >1: #If rv is list or array
                data['retrieval']['constraints']['characteristic[2]']['mode[1]']['initial_guess']['value'][0],data['retrieval']['constraints']['characteristic[2]']['mode[1]']['initial_guess']['value'][1] = float(rv[i]),float(sigma)
            if loop_count == 1: #If rv is list or array
                data['retrieval']['constraints']['characteristic[2]']['mode[1]']['initial_guess']['value'][0],data['retrieval']['constraints']['characteristic[2]']['mode[1]']['initial_guess']['value'][1] = float(rv),float(sigma)
    #             
    # 
    #           # data['retrieval']['constraints']['characteristic[4]']['mode[1]']['initial_guess']['value'][0] = float(n)
    #                 # data['retrieval']['constraints']['characteristic[5]']['mode[1]']['initial_guess']['value'][0] = float(k)
    #                 data['retrieval']['constraints']['characteristic[6]']['mode[1]']['initial_guess']['value'][0] = float(sf[j])
            KernelName = data['retrieval']['forward_model']['phase_matrix']['kernels_folder']

            f.close()
            # UpKerFile = 'settings_dust_Vext_conc_dump.yml'
            with open(ymlPath+UpKerFile_dump, 'w') as f: #write the chnages to new yaml file
                yaml.safe_dump(data, f)
            f.close()

    #New yaml file after updating state vectors/variables
        Newyaml = ymlPath+UpKerFile_dump
        #Running GRASP
        print('Using settings file at %s' % Newyaml)
        gr = graspRun(pathYAML= Newyaml, releaseYAML=True, verbose=True) # setup instance of graspRun class, add the above pixel, run grasp and read the output to gr.invRslt[0] (dict)
        gr.addPix(nowPix) # add the pixel we created above
        gr.runGRASP(binPathGRASP=binPathGRASP, krnlPathGRASP=krnlPath) # run grasp binary (output read to gr.invRslt[0] [dict])
        print('AOD at %5.3f μm was %6.4f.' % (gr.invRslt[0]['lambda'][-1],gr.invRslt[0]['aod'][-1])) # 

        if loop_count>1:
            Shape_rsltdict[f'{KernelName}_{rv[i]}'] = gr.invRslt[0]
        if loop_count == 1:
            Shape_rsltdict[f'{KernelName}_{rv}'] = gr.invRslt[0]



    return  Shape_rsltdict
   

# def PlotShapeDict(Dict_shape):

#     "Plots the phase function for different shape models for comparision"

#     plt.rcParams['font.size'] = '18'
    

#     color = ['#614C0E','#E09428','#B1DBFF','#B1DBFF'] #Color blind friendly scheme for different retrieval techniques< :rsponlt, hsrl only and combined
#     WlIdx =  [0,4,7] #Index of the wavelength 

#     # color = ['']

#     markerIdx= np.arange(0,181,30)
#     marker = ["$0$", 'D', 'o','.']
#     label = ['Spheroid', "Hex", "sph",'']

    
#     "Plots comparing the phase function and dolp for different shape models from Dict_shape['sphCoarseKernel']= ComparePhaseFunct(Shape='sph')"
#     fig,axs = plt.subplots(4,3, figsize =(16,8), sharex=True)
#     shapeNamekeys = Dict_shape.keys()
#     for idx, shapeName in enumerate(Dict_shape.keys()):
    
#         for i in range(len(WlIdx)):
#             if idx<3:
#                 axs[0,i].plot(np.arange(181),Dict_shape[shapeName]['p11'][:,0,i], lw =3, color = color[idx])
#                 axs[0,i].scatter(np.arange(181)[markerIdx],Dict_shape[shapeName]['p11'][:,0,i][markerIdx],s = 100,edgecolors='black', linewidths=1, color = color[idx], marker =marker[idx], label = label[idx])
#                 axs[0,i].set_yscale('log')
#                 axs[1,i].plot(np.arange(181),Dict_shape['Spheroid']['p11'][:,0,i] - Dict_shape['hex']['p11'][:,0,i], lw =3, color = color[idx])
                

#                 axs[2,i].plot(np.arange(181),-Dict_shape[shapeName]['p12'][:,0,i]/Dict_shape[shapeName]['p11'][:,0,i],lw =3, color = color[idx])
#                 axs[2,i].scatter(np.arange(181)[markerIdx],-Dict_shape[shapeName]['p12'][:,0,i][markerIdx]/Dict_shape[shapeName]['p11'][:,0,i][markerIdx],s = 100,edgecolors='black', linewidths=1, color = color[idx], marker =marker[idx],)
#                 axs[3,i].plot(np.arange(181),(-Dict_shape['Spheroid']['p12'][:,0,i]/Dict_shape['Spheroid']['p11'][:,0,i])-(-Dict_shape['hex']['p12'][:,0,i]/Dict_shape['hex']['p11'][:,0,i]),lw =3, color = color[idx])
               
#                 # axs[0,i].set_xlabel(r'$\theta_{s}$')
#                 axs[3,i].set_xlabel(r'$\theta_{s}$')

#                 axs[2,i].set_ylabel(r'-P12/P11')

#                 # txtylabel = 'P11 \n' + str(Dict_shape[shapeName]['lambda'][WlIdx[i]])
#                 if i ==0:
#                     axs[0,i].set_ylabel('P11')

#                 if i ==0:
#                     axs[0,i].legend(fontsize =18)
#                 axs[0,i].set_title(str(Dict_shape[shapeName]['lambda'][WlIdx[i]])+r"$\mu$m")
#                 # axs[i,1].set_title(Dict_shape[shapeName]['lambda'][WlIdx[i]])

#     titlelabel = '(rv,σ) : ' + str(Dict_shape[shapeName]['rv'])+',' + str(Dict_shape[shapeName]['sigma'])+ " n: " + str(Dict_shape[shapeName]['n'][WlIdx]) + " , k : " + str(Dict_shape[shapeName]['k'][WlIdx]) 



#     plt.suptitle(titlelabel )
#     fig.tight_layout()

#     plt.savefig('Phasefunction.png', dpi = 200)



def PlotShapeDict(Dict_shape, name =None):


    """ PlotShapeDict(DictAll['0'], name =None)"""

    "Plots the phase function for different shape models for comparision"

    plt.rcParams['font.size'] = '18'

    color_instrument = ['#800080','#CDEDFF','#404790','#CE357D','#FBD381','#A78236','#711E1E','k']
          
    wl = [0.35, 0.41, 0.46, 0.53, 0.55 , 0.67  , 0.86, 1.06]

    color = ['#614C0E','#E09428','#B1DBFF','#B1DBFF'] #Color blind friendly scheme for different retrieval techniques< :rsponlt, hsrl only and combined
    WlIdx =  [0,4] #Index of the wavelength 
    # WlIdx =  [4] #Index of the wavelength 
    # color = ['']

    markerIdx= np.arange(0,181,30)
    marker = ["$0$", 'D', 'o','.']
    # label = ['Spheroid', "Hex", "sph",'']


    "Plots comparing the phase function and dolp for different shape models from Dict_shape['sphCoarseKernel']= ComparePhaseFunct(Shape='sph')"
    fig,axs = plt.subplots(2,len(WlIdx)+3, figsize =(40,10), sharex=True)
    shapeNamekeys = list(Dict_shape.keys())
    
    label = shapeNamekeys

    # if shapeNamekeys

    a = len(WlIdx)

    for j in range(len(shapeNamekeys)-1):
        if j ==1: a = len(WlIdx)+1

        for ii in range(len(wl)):


            axs[0,a].plot(np.arange(181),(100*(Dict_shape[f'{shapeNamekeys[0]}']['p11'][:,0,ii] - Dict_shape[f'{shapeNamekeys[j+1]}']['p11'][:,0,ii])/Dict_shape[f'{shapeNamekeys[0]}']['p11'][:,0,ii]), lw =3, color = color_instrument[ii], label =wl[ii])
                # axs[1,i].set_yscale('log')
            axs[0,a].set_ylabel(r'Diff P11 %')
            axs[1,a].plot(np.arange(181),((-Dict_shape[f'{shapeNamekeys[0]}']['p12'][:,0,ii]/Dict_shape[f'{shapeNamekeys[0]}']['p11'][:,0,ii])-(-Dict_shape[f'{shapeNamekeys[j+1]}']['p12'][:,0,ii]/Dict_shape[f'{shapeNamekeys[j+1]}']['p11'][:,0,ii])),lw =3, color = color_instrument[ii], label = wl[ii])
            axs[1,a].set_ylabel(r'Abs Diff -P12/P11')
            
            axs[1,a].set_xlabel(r'$\theta_{s}$')
            axs[0,a].set_title(f'{shapeNamekeys[0]} - {shapeNamekeys[j+1]}')
    for ii in range(len(wl)):
        axs[0,4].plot(np.arange(181),(100*abs(Dict_shape[f'{shapeNamekeys[1]}']['p11'][:,0,ii] - Dict_shape[f'{shapeNamekeys[2]}']['p11'][:,0,ii])/Dict_shape[f'{shapeNamekeys[1]}']['p11'][:,0,ii]), lw =3, color = color_instrument[ii], label =wl[ii])
            # axs[1,i].set_yscale('log')
        # axs[0,4].set_ylabel(r'Diff P11 %')
        axs[1,4].plot(np.arange(181),abs((-Dict_shape[f'{shapeNamekeys[1]}']['p12'][:,0,ii]/Dict_shape[f'{shapeNamekeys[1]}']['p11'][:,0,ii])-(-Dict_shape[f'{shapeNamekeys[2]}']['p12'][:,0,ii]/Dict_shape[f'{shapeNamekeys[2]}']['p11'][:,0,ii])),lw =3, color = color_instrument[ii], label = wl[ii])
        axs[0,4].set_title(f'{shapeNamekeys[1]} - {shapeNamekeys[2]}')
        axs[1,4].set_xlabel(r'$\theta_{s}$')
    axs[1,a].legend(ncol =1, prop = '10')
    for idx, shapeName in enumerate(Dict_shape.keys()):
        
    
        for i in range(len(WlIdx)):

                
                if idx<3:
                    axs[0,i].plot(np.arange(181),Dict_shape[shapeName]['p11'][:,0,i], lw =3, color = color[idx])
                    axs[0,i].scatter(np.arange(181)[markerIdx],Dict_shape[shapeName]['p11'][:,0,i][markerIdx],s = 100,edgecolors='black', linewidths=1, color = color[idx], marker =marker[idx], label = label[idx])
                    axs[0,i].set_yscale('log')
                    
                    axs[1,i].plot(np.arange(181),-Dict_shape[shapeName]['p12'][:,0,i]/Dict_shape[shapeName]['p11'][:,0,i],lw =3, color = color[idx])
                    axs[1,i].scatter(np.arange(181)[markerIdx],-Dict_shape[shapeName]['p12'][:,0,i][markerIdx]/Dict_shape[shapeName]['p11'][:,0,i][markerIdx],s = 100,edgecolors='black', linewidths=1, color = color[idx], marker =marker[idx],)
                    
                    # axs[0,i].set_xlabel(r'$\theta_{s}$')
                    axs[1,i].set_xlabel(r'$\theta_{s}$')

                    

                    # txtylabel = 'P11 \n' + str(Dict_shape[shapeName]['lambda'][WlIdx[i]])
                    if i ==0:
                        axs[0,i].set_ylabel('P11')
                        axs[1,i].set_ylabel(r'-P12/P11')

                    if i ==0:
                        axs[0,i].legend(fontsize =18)
                    axs[0,i].set_title(str(Dict_shape[shapeName]['lambda'][WlIdx[i]])+r"$\mu$m")
                    # axs[i,1].set_title(Dict_shape[shapeName]['lambda'][WlIdx[i]])

    titlelabel = '(rv,σ) : ' + str(Dict_shape[shapeName]['rv'])+',' + str(Dict_shape[shapeName]['sigma'])+ " n: " + str(Dict_shape[shapeName]['n'][WlIdx]) + " , k : " + str(Dict_shape[shapeName]['k'][WlIdx]) 
    


    plt.suptitle(titlelabel )

    plt.savefig(f'Phasefunction_{name}.png', dpi = 200)
    fig.tight_layout()

def Values(rv_sph,sigma,psi_sph,psi_hex):

    """T
    his function converts the give volume equivant radius of spheroid to hexahedra (assuming that they are same in the equivanet radius space)and runs GRASP's forward model
    The output is saved in a dictionary
    
    """

    # rv_sph = np.linspace(0.5,5,10)
    Dict_sph_rv =ComparePhaseFunct(rv_sph,sigma) 

    #DONT USE THIS!!


    #TODO have to manually change the kernel name in the settings file.  so cant run this as a function. 

    Dict_hex_rv =  ComparePhaseFunct(rv_sph,sigma) 

    rv_hex = VolEqSph_to_VolEqHex(rv_sph ,psi_sph, psi_hex )   #[psi = 0.8521, 0.7]
    Dict_hex_rv_convert =  ComparePhaseFunct(rv_hex,sigma)     




    DictAll ={}
    for i in range(len(Dict_hex_rv_convert.keys())):
        DictAll[f'{i}'] = {}

        if isinstance(rv, (list, tuple, np.ndarray)):
            DictAll[f'{i}']['hex_rv_eff'] = Dict_hex_rv_convert[f'{list(Dict_hex_rv_convert.keys())[i]}']
            DictAll[f'{i}']['sph_rv'] = Dict_sph_rv[f'{list(Dict_sph_rv.keys())[i]}']
            DictAll[f'{i}']['hex_rv'] = Dict_hex_rv [f'{list(Dict_hex_rv.keys())[i]}']

        else:
            DictAll[f'{i}']['hex_rv_eff'] = Dict_hex_rv_convert[f'{list(Dict_hex_rv_convert.keys())[0]}']
            DictAll[f'{i}']['sph_rv'] = Dict_sph_rv[f'{list(Dict_sph_rv.keys())[0]}']
            DictAll[f'{i}']['hex_rv'] = Dict_hex_rv [f'{list(Dict_hex_rv.keys())[0]}']


    DictAll['rv'] = rv_sph
    DictAll['Mod_rv_reff'] =rv_hex 
    DictAll['sphericity_sph'] = psi_sph
    DictAll['sphericity_hex'] = psi_hex


    with open(f'dict_{rv_sph}.txt', 'wb') as handle:
        pickle.dump(DictAll, handle, protocol=pickle.HIGHEST_PROTOCOL)


    handle.close()


    return DictAll 
def Read_HiGear(file_name, Idx):

    plt.rcParams['font.size'] = '12'

    #Plot the size distribution from HiGEAR Oracles 2018

    # Open a .nc file ("file_name")

    file_id = Dataset(file_name)
    d = file_id.variables['diameter'][:]  #micrometers
    
    r = d/2   #Radius
    lnr = np.log(r) #loge r

    dNdlogd = file_id.variables["dNdlogd"][:,Idx]  #log10 number distribution

    dNdlnr = dNdlogd / np.log(10)
    dNdlnr [np.where(dNdlnr <=0)[0]] = np.nan #set the bins with 0 concetration to nan, for plotting.

    V = (4/3)* np.pi* r**3  # Volume of an equivalent sphere. 
    dVdlnr = V * dNdlnr  #dV / dlnr 

    # Vtotal = np.trapz(dVdlnr, lnr)

    DvDlnr = dVdlnr

    # timeStart=  file_id.variables["time_start"][Idx ]  #time
    # timeEnd = file_id.variables["time_end"][Idx ]
    # # Convert Unix time to UTC datetime
    # utc_datetime_start = datetime.datetime.utcfromtimestamp(float(timeStart))
    # utc_datetime_End = datetime.datetime.utcfromtimestamp(float(timeEnd))

    # print("UTC Date and Time:", utc_datetime_start,utc_datetime_End,'time window:',utc_datetime_End-utc_datetime_start )


    return DvDlnr, r 



def CheatPlotShapeDict(Dict_shape, name =None):


    """ PlotShapeDict(DictAll['0'], name =None)"""

    "Plots the phase function for different shape models for comparision"

    plt.rcParams['font.size'] = '25'

    color_instrument = ['#800080','#CDEDFF','#404790','#CE357D','#FBD381','#A78236','#711E1E','k']
          
    wl = [0.35, 0.41, 0.46, 0.53, 0.55 , 0.67  , 0.86, 1.06]

    color = ['#614C0E','#E09428','#B1DBFF','#B1DBFF'] #Color blind friendly scheme for different retrieval techniques< :rsponlt, hsrl only and combined
    WlIdx =  [4] #Index of the wavelength 
    # WlIdx =  [4] #Index of the wavelength 
    # color = ['']

    markerIdx= np.arange(0,181,30)
    marker = ["$0$", 'D', 'o','.']
    # label = ['Spheroid', "Hex", "sph",'']


    "Plots comparing the phase function and dolp for different shape models from Dict_shape['sphCoarseKernel']= ComparePhaseFunct(Shape='sph')"
    fig,axs = plt.subplots(2,3, figsize =(25,15), sharex=True)
    shapeNamekeys = list(Dict_shape.keys())
    
    label = shapeNamekeys

    # if shapeNamekeys

    a = len(WlIdx)

    

    for ii in range(len(wl)):

        axs[0,1].plot(np.arange(181),(100*(Dict_shape[f'{shapeNamekeys[0]}']['p11'][:,0,ii] - Dict_shape[f'{shapeNamekeys[1]}']['p11'][:,0,ii])/Dict_shape[f'{shapeNamekeys[0]}']['p11'][:,0,ii]), lw =3, color = color_instrument[ii], label =wl[ii])
            # axs[1,i].set_yscale('log')
        axs[0,1].set_ylabel(r'Diff P11 %')
        axs[1,1].plot(np.arange(181),((-Dict_shape[f'{shapeNamekeys[0]}']['p12'][:,0,ii]/Dict_shape[f'{shapeNamekeys[0]}']['p11'][:,0,ii])-(-Dict_shape[f'{shapeNamekeys[1]}']['p12'][:,0,ii]/Dict_shape[f'{shapeNamekeys[1]}']['p11'][:,0,ii])),lw =3, color = color_instrument[ii], label = wl[ii])
        axs[1,1].set_ylabel(r'Diff -P12/P11')
        
        axs[1,1].set_xlabel(r'$\theta_{s}$')
        axs[0,1].set_title(f'{shapeNamekeys[0]} - {shapeNamekeys[1]}')
    axs[0,1].plot(np.arange(181),np.zeros(181),lw =0.85, color= 'k' , ls ='--')
    axs[1,1].plot(np.arange(181),np.zeros(181),lw =0.85, color= 'k' , ls ='--')
        
    for ii in range(len(wl)):
        axs[0,2].plot(np.arange(181),(100*(Dict_shape[f'{shapeNamekeys[2]}']['p11'][:,0,ii] - Dict_shape[f'{shapeNamekeys[1]}']['p11'][:,0,ii])/Dict_shape[f'{shapeNamekeys[2]}']['p11'][:,0,ii]), lw =3, color = color_instrument[ii], label =wl[ii])
            # axs[1,i].set_yscale('log')
        # axs[0,4].set_ylabel(r'Diff P11 %')
        axs[1,2].plot(np.arange(181),((-Dict_shape[f'{shapeNamekeys[2]}']['p12'][:,0,ii]/Dict_shape[f'{shapeNamekeys[2]}']['p11'][:,0,ii])-(-Dict_shape[f'{shapeNamekeys[1]}']['p12'][:,0,ii]/Dict_shape[f'{shapeNamekeys[1]}']['p11'][:,0,ii])),lw =3, color = color_instrument[ii], label = wl[ii])
        axs[0,2].set_title(f'{shapeNamekeys[2]} - {shapeNamekeys[1]}')
        axs[1,2].set_xlabel(r'$\theta_{s}$')


        axs[1,2].plot(np.arange(181),np.zeros(181),lw =0.85, color= 'k' , ls ='--')
        axs[0,2].plot(np.arange(181),np.zeros(181),lw =0.85, color= 'k' , ls ='--')
        
    axs[0,1].legend(ncol =2, loc='best',prop = '6')
    for idx, shapeName in enumerate(Dict_shape.keys()):
        
    
        for i in range(len(WlIdx)):

                
                if idx<3:
                    axs[0,i].plot(np.arange(181),Dict_shape[shapeName]['p11'][:,0,i], lw =3, color = color[idx])
                    axs[0,i].scatter(np.arange(181)[markerIdx],Dict_shape[shapeName]['p11'][:,0,i][markerIdx],s = 100,edgecolors='black', linewidths=1, color = color[idx], marker =marker[idx], label = label[idx])
                    axs[0,i].set_yscale('log')
                    
                    axs[1,i].plot(np.arange(181),-Dict_shape[shapeName]['p12'][:,0,i]/Dict_shape[shapeName]['p11'][:,0,i],lw =3, color = color[idx])
                    axs[1,i].scatter(np.arange(181)[markerIdx],-Dict_shape[shapeName]['p12'][:,0,i][markerIdx]/Dict_shape[shapeName]['p11'][:,0,i][markerIdx],s = 100,edgecolors='black', linewidths=1, color = color[idx], marker =marker[idx],)
                    
                    # axs[0,i].set_xlabel(r'$\theta_{s}$')
                    axs[1,i].set_xlabel(r'$\theta_{s}$')

                    

                    # txtylabel = 'P11 \n' + str(Dict_shape[shapeName]['lambda'][WlIdx[i]])
                    if i ==0:
                        axs[0,i].set_ylabel('P11')
                        axs[1,i].set_ylabel(r'-P12/P11')

                    if i ==0:
                        axs[0,i].legend()
                    axs[0,i].set_title(str(Dict_shape[shapeName]['lambda'][WlIdx[i]])+r"$\mu$m")
                    # axs[i,1].set_title(Dict_shape[shapeName]['lambda'][WlIdx[i]])

    titlelabel = '(rv,σ) : ' + str(Dict_shape[shapeName]['rv'])+',' + str(Dict_shape[shapeName]['sigma'])+ " n: " + str(Dict_shape[shapeName]['n'][WlIdx]) + " , k : " + str(Dict_shape[shapeName]['k'][WlIdx]) 
    


    plt.suptitle(titlelabel )

    plt.savefig(f'Phasefunction_{name}.png', dpi = 200)
    # fig.tight_layout()


# PlotE2C(v_hex_sea,name ='v_hex_sea')
# PlotE2C(v_sph_sea,name ='v_sph_sea')

# PlotE2C(v_sph_fine,name ='v_sph_fine')
# PlotE2C(v_hex_fine,name ='v_hex_fine')

# PlotE2C(v_sph_dust,name ='v_sph_dust')
# PlotE2C(v_hex_dust,name ='v_hex_dust')


# def ComparisionPlots(LidarOnlyShape1,LidarOnlyShape2, CombShape1, CombShape2,key1 = None,key2= None,UNCERT= None): #Comapre the fits and retreivals between independendt and combined retreivals
    
#     IndexH = [0,3,7] #INdex for Lidar values in COmbined retrievals, there are nan values for values in RSP wavelength
#         #Converting range to altitude
#     altd = (LidarOnlyShape1['RangeLidar'][:,0])/1000 #altitude for spheriod
    
#     fig, axs= plt.subplots(nrows = 2, ncols =3, figsize= (15,12))  #TODO make it more general which adjust the no of rows based in numebr of wl
#     plt.subplots_adjust(top=0.78)
#     for i in range(3):
            
#         wave = str(LidarOnlyShape1['lambda'][i]) +"μm"
#         axs[i].errorbar(LidarOnlyShape1['meas_VBS'][:,i],altd,xerr= UNCERT['VBS'],color = "#695E93", capsize=3,capthick =1,alpha =0.4, label =f"{UNCERT['VBS']}")
#         axs[i].plot(LidarOnlyShape1['meas_VBS'][:,i],altd, marker =">",color = "#281C2D", label ="Meas")
#         axs[i].plot(LidarOnlyShape1['fit_VBS'][:,i],altd,color ="#025043", marker = "$O$",label =f"{key1}",alpha =0.8) #HSRL only
#         axs[i].plot(LidarOnlyShape1['fit_VBS'][:,IndexH[i]],altd,color ="#025043", marker = "$O$",label =f"{key2}",alpha =0.8) #Combined
        

        
        
#         axs[i].plot(HTam['fit_VBS'][:,i],altd,color =  "#d24787",ls = "--", label=f"{key2}", marker = "h")
#             axs[i].xaxis.set_major_formatter(ScalarFormatter(useMathText=True))

#             # print(UNCERT)
#             # axs[i].errorbar(Hsph['fit_VBS'][:,i],altd,xerr= UNCERT['VBS'],color = "#025043", marker = "$O$",label ="Sphd")
#             # axs[i].errorbar(HTam['fit_VBS'][:,i],altd,xerr= UNCERT['VBS'],color = "#d24787",ls = "--", label="Hex", marker = "h")

#             axs[i].set_xlabel(f'$ VBS (m^{-1}Sr^{-1})$',fontproperties=font_name)
#             axs[0].set_ylabel('Height above ground (km)',fontproperties=font_name)

#             axs[i].set_title(wave)
#             if i ==0:
#                 axs[0].legend()
#             plt.suptitle(f"HSRL Vertical Backscatter profile  \nLat,Lon: {HTam['latitude']}, {HTam['longitude']} Date: {HTam['datetime']}\n ",fontproperties=font_name) #Initial condition strictly constrainted by RSP retrievals
#         pdf_pages.savefig()
#         fig.savefig(f'/home/gregmi/ORACLES/HSRL_RSP/FIT_HSRL Vertical Backscatter profile{NoMode} .png',dpi = 300)
#             # plt.tight_layout()
#         fig, axs= plt.subplots(nrows = 1, ncols =3, figsize= (15,6))
#         plt.subplots_adjust(top=0.78)
#         for i in range(3):
#             wave = str(Hsph['lambda'][i]) +"μm"

#             axs[i].errorbar(Hsph['meas_DP'][:,i],altd,xerr= UNCERT['DP'],color = '#695E93', capsize=4,capthick =1,alpha =0.6, label =f"{UNCERT['DP']}%")
            
#             axs[i].plot(Hsph['meas_DP'][:,i],altd, marker =">",color = "#281C2D", label ="Meas")
#             axs[i].plot(Hsph['fit_DP'][:,i],altd,color = "#025043", marker = "$O$",label =f"{key1}")
#             axs[i].plot(HTam['fit_DP'][:,i],altd,color = "#d24787", ls = "--",marker = "h",label=f"{key2}")
            
#             # axs[i].errorbar(Hsph['fit_DP'][:,i],altd,xerr= UNCERT['DP'],fmt='-o',color = "#025043", marker = "$O$",label ="Sphd")
#             # axs[i].errorbar(HTam['fit_DP'][:,i],altd,xerr= UNCERT['DP'],fmt='-o',color = "#d24787",ls = "--", label="Hex", marker = "h")


#             axs[i].set_xlabel('DP %')
#             axs[i].set_title(wave)
#             if i ==0:
#                 axs[0].legend()
#             axs[0].set_ylabel('Height above ground (km)',fontproperties=font_name)
#             plt.suptitle(f"HSRL Depolarization Ratio \n Lat,Lon: {HTam['latitude']}, {HTam['longitude']}  Date: {HTam['datetime']}\n ",fontproperties=font_name) #Initial condition strictly constrainted by RSP retrievals
#         pdf_pages.savefig()
#         fig.savefig(f'/home/gregmi/ORACLES/HSRL_RSP/FIT_HSRL Depolarization Ratio{NoMode}.png',dpi = 300)
#         fig, axs= plt.subplots(nrows = 1, ncols =2, figsize= (11,6))
#         plt.subplots_adjust(top=0.78)
#         for i in range(2):
#             wave = str(Hsph['lambda'][i]) +"μm"

#             axs[i].errorbar(Hsph['meas_VExt'][:,i],altd,xerr= UNCERT['VEXT']*Hsph['meas_VExt'][:,i],color = '#695E93',capsize=4,capthick =1,alpha =0.7, label =f"{UNCERT['VEXT']*100}%")
            
#             axs[i].plot(Hsph['meas_VExt'][:,i],altd, marker =">",color = "#281C2D", label ="Meas")
#             axs[i].plot(Hsph['fit_VExt'][:,i],altd,color = "#025043", marker = "$O$",label =f"{key1}")
#             axs[i].plot(HTam['fit_VExt'][:,i],altd,color = "#d24787",ls = "--", marker = "h",label=f"{key2}")
#             axs[i].xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
#             # axs[i].errorbar(Hsph['fit_VExt'][:,i],altd,xerr= UNCERT['VEXT'],color = "#025043", marker = "$O$",label ="Sphd")
#             # axs[i].errorbar(HTam['fit_VExt'][:,i],altd,xerr= UNCERT['VEXT'],color = "#d24787",ls = "--", label="Hex", marker = "h")

            
            
#             axs[i].set_xlabel(f'$VExt (m^{-1})$',fontproperties=font_name)
#             axs[0].set_ylabel('Height above ground (km)',fontproperties=font_name)
#             axs[i].set_title(wave)
#             if i ==0:
#                 axs[0].legend()
#             plt.suptitle(f"HSRL Vertical Extinction profile\n Lat,Lon: {HTam['latitude']},{HTam['longitude']}  Date: {HTam['datetime']}\n ",fontproperties=font_name) #Initial condition strictly constrainted by RSP retrievals
#         plt.tight_layout()
#         pdf_pages.savefig()
#         fig.savefig(f'/home/gregmi/ORACLES/HSRL_RSP/FIT_HSRL_Vertical_Ext_profile{NoMode}.png',dpi = 300)





        
# for i in range(1):

#     #Reading the ORACLE data for given pixel no, Tel_no = aggregated altitude
#     #working pixels: 16800,16813 ,16814
#     # RSP_PixNo = 2776  #4368  #Clear pixel 8/01 Pixel no of Lat,Lon that we are interested

#     # RSP_PixNo = 13240
#     RSP_PixNo = 13200
#     #Case2

#     RSP_PixNo = 2687


#      #Dusty pixel on 9/22 
#     # PixNo = find_dust(file_path,file_name)[1][0]
#     TelNo = 0 # aggregated altitude. To obtain geometries corresponding to data from the 1880 nm channel, aggregation altitude should be set to 1, while aggregation altitude =0 should be used for all other channels.
#     nwl = 5 # first  nwl wavelengths
#     ang1 = 5
#     ang2 = 140 # :ang angles  #Remove

#     f1_MAP = h5py.File(file_path+file_name,'r+')   
#     Data = f1_MAP['Data']
#     LatRSP = f1_MAP['Geometry']['Collocated_Latitude'][TelNo,RSP_PixNo]
#     LonRSP = f1_MAP['Geometry']['Collocated_Longitude'][TelNo,RSP_PixNo]
#     f1_MAP.close()
#     f1= h5py.File(HSRLfile_path + HSRLfile_name,'r+')  #reading hdf5 file  
    
#     #Lat and Lon values for that pixel
#     LatH = f1['Nav_Data']['gps_lat'][:]
#     LonH = f1['Nav_Data']['gps_lon'][:]
#     f1.close()
#     # HSRLPixNo is the index of HSRL pixel taht corresponds to the RSP Lat Lon
#     HSRLPixNo = FindPix(LatH,LonH,LatRSP,LonRSP)[0]  # Or can manually give the index of the pixel that you are intrested in
 
# # #  Kernel_type = Run(Kernel_type) for spheriod, Kernel_type = 'TAMU' for hexahedral
#     rslts_Sph = RSP_Run("sphro",RSP_PixNo,ang1,ang2,TelNo,nwl)
#     rslts_Tamu = RSP_Run("TAMU",RSP_PixNo,ang1,ang2,TelNo,nwl)
#     RSP_plot(rslts_Sph,rslts_Tamu,RSP_PixNo)
  

#     HSRL_sphrod = HSLR_run("sphro",HSRLfile_path,HSRLfile_name,HSRLPixNo,nwl,ModeNo=3, updateYaml= False,releaseYAML= True)
#     plot_HSRL(HSRL_sphrod[0][0],HSRL_sphrod[0][0], forward = True, retrieval = True, Createpdf = True,PdfName ="/home/gregmi/ORACLES/rsltPdf/HSRL_Only_Plots_444.pdf", combinedVal =HSRL_sphrod[2]) 

#     HSRL_Tamu = HSLR_run("TAMU",HSRLfile_path,HSRLfile_name, HSRLPixNo,nwl,ModeNo=3,updateYaml= False,releaseYAML= True)
#     plot_HSRL(HSRL_Tamu[0][0],HSRL_Tamu[0][0], forward = True, retrieval = True, Createpdf = True,PdfName ="/home/gregmi/ORACLES/rsltPdf/HSRL_Only_Plots_444.pdf", combinedVal =HSRL_Tamu[2])

    
#     plot_HSRL(HSRL_sphrod[0][0],HSRL_Tamu[0][0], forward = True, retrieval = True, Createpdf = True,PdfName ="/home/gregmi/ORACLES/rsltPdf/HSRL_Only_Plots_444.pdf", combinedVal =HSRL_sphrod[2])

#     # # print('Cost Value Sph, tamu: ',  rslts_Sph[0]['costVal'], rslts_Tamu[0]['costVal'])
#     # RSP_plot(rslts_Sph,rslts_Tamu,RSP_PixNo)
#     # plot_HSRL(rslts_Sph[0],rslts_Tamu[0], forward = False, retrieval = True, Createpdf = True,PdfName =f"/home/gregmi/ORACLES/rsltPdf/RSP_only_{RSP_PixNo}.pdf")
    
# #     # Retrieval_type = 'NosaltStrictConst_final'
# #     # #Running GRASP for HSRL, HSRL_sphrod = for spheriod kernels,HSRL_Tamu = Hexahedral kernels
    
# #     print('Cost Value Sph, tamu: ',  HSRL_sphrod[0][0]['costVal'],HSRL_Tamu[0][0]['costVal'])
   
# # # #     # # #Constrining HSRL retrievals by 5% 
# #     HSRL_sphro_5 = HSLR_run("sphro",HSRLfile_path,HSRLfile_name,HSRLPixNo,nwl,ModeNo=3,updateYaml= True,releaseYAML= True) 
# #     HSRL_Tamu_5 = HSLR_run("TAMU",HSRLfile_path,HSRLfile_name,HSRLPixNo,nwl,ModeNo=3,updateYaml= True,releaseYAML= True)

# # #     print('Cost Value Sph, tamu: ',  HSRL_sphro_5[0][0]['costVal'],HSRL_Tamu_5[0][0]['costVal'])

# # # #     #Strictly Constrining HSRL retrievals to values from RSP retrievals
# #     HSRL_sphrod_strict = HSLR_run("sphro",HSRLfile_path,HSRLfile_name,HSRLPixNo,nwl,ModeNo=3,updateYaml= True, ConsType = 'strict',releaseYAML= True) 
# #     HSRL_Tamu_strict = HSLR_run("TAMU",HSRLfile_path,HSRLfile_name,HSRLPixNo,nwl,ModeNo=3,updateYaml= True, ConsType = 'strict',releaseYAML= True)

# # #     print('Cost Value Sph, tamu: ',  HSRL_sphrod_strict[0][0]['costVal'],HSRL_Tamu_strict[0][0]['costVal'])
    
#     # RIG = randomInitGuess('sphro',HSRLfile_path,HSRLfile_name,HSRLPixNo, nwl,releaseYAML =True, ModeNo=3)
#      #Lidar+pol combined retrieval
#     # LidarPolSph = LidarAndMAP('sphro',HSRLfile_path,HSRLfile_name,HSRLPixNo,file_path,file_name,RSP_PixNo,ang1,ang2,TelNo, nwl,GasAbsFn,ModeNo=3, updateYaml= None,RandinitGuess=True, NoItr =100 )
#     # # PlotRandomGuess('gregmi/git/GSFC-GRASP-Python-Interface/try.npy', 2,0)
#     LidarPolSph = LidarAndMAP('sphro',HSRLfile_path,HSRLfile_name,HSRLPixNo,file_path,file_name,RSP_PixNo,ang1,ang2,TelNo, nwl,GasAbsFn,ModeNo=3, updateYaml= None)
#     LidarPolTAMU = LidarAndMAP('TAMU',HSRLfile_path,HSRLfile_name,HSRLPixNo,file_path,file_name,RSP_PixNo,ang1,ang2,TelNo, nwl,GasAbsFn,ModeNo=3, updateYaml= None)
    
#     LidarPolTAMU = LidarAndMAP('TAMU',HSRLfile_path,HSRLfile_name,HSRLPixNo,file_path,file_name,RSP_PixNo,ang1,ang2,TelNo, nwl,GasAbsFn,ModeNo=3, updateYaml= None,RandinitGuess=True, NoItr =150)
#     LidarPolSph = LidarAndMAP('sphro',HSRLfile_path,HSRLfile_name,HSRLPixNo,file_path,file_name,RSP_PixNo,ang1,ang2,TelNo, nwl,GasAbsFn,ModeNo=3, updateYaml= None,RandinitGuess=True, NoItr =150)

    
    
    # # LidarPolSph = LidarAndMAP('sphro',HSRLfile_path,HSRLfile_name,HSRLPixNo,file_path,file_name,RSP_PixNo,ang1,ang2,TelNo, nwl,GasAbsFn,ModeNo=3, updateYaml= None,RandinitGuess=True, NoItr =150)
    # LidarPolTAMU = LidarAndMAP('TAMU',HSRLfile_path,HSRLfile_name,HSRLPixNo,file_path,file_name,RSP_PixNo,ang1,ang2,TelNo, nwl,GasAbsFn,ModeNo=3, updateYaml= None,RandinitGuess=True, NoItr =150)


    # RandinitGuess=True, NoItr =2 
    # print('Cost Value Sph, tamu: ',  LidarPolSph[0]['costVal'],LidarPolTAMU[0]['costVal'])
    # # print('SPH',"tam" )


    # %matplotlib inline
    # plot_HSRL(HSRL_sphrod[0][0],HSRL_Tamu[0][0], forward = True, retrieval = True, Createpdf = True,PdfName ="/home/gregmi/ORACLES/rsltPdf/HSRL_Only_Plots_444.pdf", combinedVal =HSRL_sphrod[2])
    # plot_HSRL(HSRL_sphro_5[0][0],HSRL_Tamu_5[0][0], forward = True, retrieval = True, Createpdf = True,PdfName ="/home/gregmi/ORACLES/rsltPdf/5%_HSRL_Plots_444.pdf", combinedVal =HSRL_sphrod[2])
    # plot_HSRL(HSRL_sphrod_strict[0][0],HSRL_Tamu_strict[0][0], forward = True, retrieval = True, Createpdf = True,PdfName ="/home/gregmi/ORACLES/rsltPdf/STR_HSRL_Plots_444.pdf", combinedVal =HSRL_sphrod[2])
    # plot_HSRL(LidarPolSph[0],LidarPolTAMU[0], forward = False, retrieval = True, Createpdf = True,PdfName ="/home/gregmi/ORACLES/rsltPdf/LIDARPOL_Plots_444.pdf")
    # plot_HSRL(LidarPolSph[0][0],LidarPolTAMU[0][0], forward = False, retrieval = True, Createpdf = True,PdfName ="/home/gregmi/ORACLES/rsltPdf/LIDARPOL_Plots_444.pdf")
    # CombinedLidarPolPlot(LidarPolSph[0],LidarPolTAMU[0])

#     plot_HSRL(LidarPolTAMU[0][0],LidarPolTAMU[0][0], forward = False, retrieval = True, Createpdf = True,PdfName ="/home/gregmi/ORACLES/rsltPdf/LIDARPOL_Plots_444.pdf")
# CombinedLidarPolPlot(LidarPolTAMU[0],LidarPolTAMU[0])
def plot_phase_err():

    #Calculate the difference between pahse functions of two shapes and 

    Dict_shape = DictAll2['0']

    fig, axs = plt.subplots(nrows= 1, ncols=2, figsize=(16, 9), sharey =True)

    ErrSph2 = 100*((rslts_Sph2[0]['meas_I'] - rslts_Sph2[0]['fit_I']))/rslts_Sph2[0]['meas_I']
    ErrHex2 = 100*((rslts_Tamu2[0]['meas_I'] - rslts_Tamu2[0]['fit_I']))/rslts_Tamu2[0]['meas_I']


    wl = rslts_Sph2[0]['lambda']
    colorwl = ['#70369d','#4682B4','#01452c','#FF7F7F','#d4af37','#4c0000']

    for i in range(len(wl)):
        axs[0].plot(rslts_Sph2[0]['sca_ang'][:,i], ErrSph2[:,i] ,color = colorwl[i],marker = "$O$",markersize= 10, lw = 0.2,label=f"{np.round(wl[i],2)}")
        axs[1].plot(rslts_Tamu2[0]['sca_ang'][:,i], ErrHex2[:,i],color = colorwl[i], marker = 'H',markersize= 12, lw = 0.2,label=f"{np.round(wl[i],2)}" )
    axs[0].set_title('Spheriod')
    axs[1].set_title('Hexahedral')

    # axs[0].plot(rslts_Sph[0]['sca_ang'][:,0], 100*UNCERT['I']*np.ones(len(ErrSph[:,i])) ,color = 'k',ls = '--', lw = 1,label=f"Meas Uncert")
    # axs[1].plot(rslts_Sph[0]['sca_ang'][:,0], 100*UNCERT['I']*np.ones(len(ErrSph[:,i])) ,color = 'k',ls = '--', lw = 1,label=f"Meas Uncert")


    axs[0].set_ylabel('Error I %')
    # axs[1].set_ylabel('Error %')
    plt.legend( ncol=2)

    axs[0].set_xlabel(r'${\theta_s}$')
    axs[1].set_xlabel(r'${\theta_s}$')
    plt.suptitle("RSP-Only Case 1")
    # plt.savefig(f'{file_name[2:]}_{RSP_PixNo}_ErrorI.png', dpi = 300)

    #Absolute err because DOLP are in %
    ErrSphP2 = (-(rslts_Sph2[0]['meas_P_rel'] - rslts_Sph2[0]['fit_P_rel']))
    ErrHexP2 = (-(rslts_Tamu2[0]['meas_P_rel'] - rslts_Tamu2[0]['fit_P_rel']))


    ax0 =  axs[0].twinx()
    ax01 =  axs[1].twinx()
    ax1 =  axs[1].twinx()

        
    for ii in range(len(wl)):

        # Ang = np.arange(181)[np.where((np.arange(181)>= np.min(rslts_Sph[0]['sca_ang'][:,0])& np.arange(181)<= np.max(rslts_Sph[0]['sca_ang'][:,0]) ))][0]

        # print(Ang)
        P11_c = (100*(Dict_shape[f'{shapeNamekeys[2]}']['p11'][:,0,ii] - Dict_shape[f'{shapeNamekeys[1]}']['p11'][:,0,ii])/Dict_shape[f'{shapeNamekeys[2]}']['p11'][:,0,ii])

        Dolp_c = ((-Dict_shape[f'{shapeNamekeys[2]}']['p12'][:,0,ii]/Dict_shape[f'{shapeNamekeys[2]}']['p11'][:,0,ii])-(-Dict_shape[f'{shapeNamekeys[1]}']['p12'][:,0,ii]/Dict_shape[f'{shapeNamekeys[1]}']['p11'][:,0,ii]))


        

        ax0.plot(np.arange(181)[75:],P11_c[75:], lw =3, color = color_instrument[ii], label =wl[ii])
    
        ax01.plot(np.arange(181)[75:],P11_c[75:],lw =3, color = color_instrument[ii], label = wl[ii])
        # axs[0,2].set_title(f'{shapeNamekeys[2]} - {shapeNamekeys[1]}')
        # axs[1,2].set_xlabel(r'$\theta_{s}$')

        



    fig, axs = plt.subplots(nrows= 1, ncols=2, figsize=(16, 8), sharey = True)



    for i in range(len(wl)):
        axs[0].plot(rslts_Sph2[0]['sca_ang'][:,i], ErrSphP2[:,i] ,color = colorwl[i],marker = "$O$",markersize= 10, lw = 0.2,label=f"{np.round(wl[i],2)}")
        axs[1].plot(rslts_Tamu2[0]['sca_ang'][:,i], ErrHexP2[:,i],color = colorwl[i], marker = 'H',markersize= 12, lw = 0.2,label=f"{np.round(wl[i],2)}" )


    axs[0].set_title('Spheriod')
    axs[1].set_title('Hexahedral')

    # axs[0].plot(rslts_Sph[0]['sca_ang'][:,0], 100*UNCERT['DoLP']*np.ones(len(ErrSph[:,i])) ,color = 'k',ls = '--', lw = 1,label=f"Meas Uncert")
    # axs[1].plot(rslts_Sph[0]['sca_ang'][:,0], 100*UNCERT['DoLP']*np.ones(len(ErrSph[:,i])) ,color = 'k',ls = '--', lw = 1,label=f"Meas Uncert")


    axs[0].set_ylabel('Error DoLP %')
    # plt.legend( ncol=2)

    axs[0].set_xlabel(r'${\theta_s}$')
    axs[1].set_xlabel(r'${\theta_s}$')
    plt.suptitle("RSP-Only Case 1")


    plt.rcParams['font.size'] = '25'

    color_instrument = ['#800080','#CDEDFF','#404790','#CE357D','#FBD381','#A78236','#711E1E','k']
            
    wl = [0.35, 0.41, 0.46, 0.53, 0.55 , 0.67  , 0.86, 1.06]

    color = ['#614C0E','#E09428','#B1DBFF','#B1DBFF'] #Color blind friendly scheme for different retrieval techniques< :rsponlt, hsrl only and combined
    WlIdx =  [4] #Index of the wavelength 
    # WlIdx =  [4] #Index of the wavelength 
    # color = ['']

    markerIdx= np.arange(0,181,30)
    marker = ["$0$", 'D', 'o','.']
    # label = ['Spheroid', "Hex", "sph",'']


    "Plots comparing the phase function and dolp for different shape models from Dict_shape['sphCoarseKernel']= ComparePhaseFunct(Shape='sph')"
    # fig,axs = plt.subplots(2,3, figsize =(25,15), sharex=True)
    shapeNamekeys = list(Dict_shape.keys())

    label = shapeNamekeys


    a = len(WlIdx)
    # fig, axs = plt.subplots(nrows= 1, ncols=2, figsize=(10, 5))

    ax0 =  axs[0].twinx()

    ax01 =  axs[1].twinx()
        
    for ii in range(len(wl)):

        # print(Ang)
        # P11_c = (100*(Dict_shape[f'{shapeNamekeys[2]}']['p11'][:,0,ii] - Dict_shape[f'{shapeNamekeys[1]}']['p11'][:,0,ii])/Dict_shape[f'{shapeNamekeys[2]}']['p11'][:,0,ii])

        Dolp_c1 = (-Dict_shape[f'{shapeNamekeys[2]}']['p12'][:,0,ii]/Dict_shape[f'{shapeNamekeys[2]}']['p11'][:,0,ii])-(-Dict_shape[f'{shapeNamekeys[1]}']['p12'][:,0,ii]/Dict_shape[f'{shapeNamekeys[1]}']['p11'][:,0,ii])

        # ax0.plot(np.arange(181)[30:],P11_c[30:], lw =3, color = color_instrument[ii], label =wl[ii])
    
        ax0.plot(np.arange(181)[75:],-Dolp_c1[75:],lw =3, color = color_instrument[ii], label = wl[ii])
        ax01.plot(np.arange(181)[75:],-Dolp_c1[75:],lw =3, color = color_instrument[ii], label = wl[ii])
        # # axs[0,2].set_title(f'{shapeNamekeys[2]} - {shapeNamekeys[1]}')
        # axs[1,2].set_xlabel(r'$\theta_{s}$')

        return
    



def InterpolateRIdxfromHSRLforRSP(HSRLrslt, wlRSP):

    #HSRLrslt is the result from GRASP using HSRL2 data
    #wlRSP is RSP wavelengths
    wlHSRL = HSRLrslt['lambda']
    wlRSP



def RunFwdModel_LIDARandMAP():


    dict_of_dicts = {}
    # sphericalFract = np.arange(1e-5, 0.99,0.01)

    "Designed to characterize the fine mode overestimation in Lidar and MAP"


    #Path to GRASP executables and Kernels

    krnlPath='/data/home/gregmi/GRASP_V2/grasp-dev-rtm-v120-new-inter/src/retrieval/internal_files'
    binPathGRASP ='/home/shared/GRASP_GSFC/build_HEX_v112/bin/grasp_app'
    ymlPath = '/home/gregmi/git/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/'
    UpKerFile =  'settings_dust_Vext_conc_dump.yml'  #Yaml file after updating the state vectors
    fwdModelYAMLpath ='/home/gregmi/git/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/settings_dust_Vext_conc.yml'
    
    #
    #all values are set for 0.532 nm values in the ymal file.
    if 'dust' in AeroType.lower():
        rv,sigma, n, k, sf = 2, 0.5, 1.55, 0.004, 1e-6
    if 'seasalt' in AeroType.lower():
        rv,sigma, n, k, sf = 1.5 , 0.6, 1.33, 0.0001, 0.999          
    if 'fine' in AeroType.lower():
        rv,sigma, n, k, sf = 0.13, 0.49, 1.45, 0.003, 1e-6
    
    #Setting the shape model Sphroids and Hexhedral
    if 'sph' in Shape.lower():
        Kernel = 'KERNELS_BASE'
        print(Kernel)
    if 'hex' in Shape.lower():
        Kernel = 'Ver_sph'
        print(Kernel)

    Volume = np.linspace(1e-5,2,20)
    VConcandExt = np.zeros((len(Volume),2))*np.nan

#Geometry
    singProf = np.linspace(botLayer, topLayer, Nlayers)[::-1]
    #Inputs for simulation
    wvls = [0.532] # wavelengths in μm
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





# Codes for foward model run: 



def update_HSRLyaml(UpdatedymlPath, YamlFileName: str, noMod: int, Kernel_type: str,  
                    GRASPModel = None, AeroProf =None, ConsType= None, YamlChar=None, maxr=None, minr=None, 
                    NewVarDict: dict = None, DataIdxtoUpdate=None): 
    """
    Update the YAML file for HSRL with initial conditions from polarimeter retrievals.
    
    Arguments:
    ymlPath: str              -- Path to which new updated setting file will be saved
    UpdatedymlPath: str            -- path of the new settings file
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
    GRASPModel: str           -- 'Fwd' for forward model or 'Bck'for inverse
    AeroProf:list             -- (Apriori) Normalized vertical profile for each aerosol mode (For LIDARS), size = (Vertical grid X no of aerosol modes)

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

#.......................................
#Set the shape model
#.......................................
    
    if Kernel_type =='spheroid':
        data['retrieval']['forward_model']['phase_matrix']['kernels_folder'] = 'KERNELS_BASE'
    if Kernel_type =='hex':
        data['retrieval']['forward_model']['phase_matrix']['kernels_folder'] = 'Ver_sph'

#.......................................
#Set GRASP to forward or inverse mode
#.......................................


    if GRASPModel != None:
        if 'fwd' in GRASPModel.lower(): #If set to forward mode
            data['retrieval']['mode'] = 'forward'
            data['output']['segment']['stream'] = 'bench_FWD_IQUandLIDAR_rslts.txt'
            data['input']['file'] = 'bench.sdat'
         
        if 'bck' in GRASPModel.lower(): #If set to inverse mode
            data['retrieval']['mode'] = 'inversion'
            data['output']['segment']['stream'] = 'bench_inversionRslts.txt'
            data['input']['file'] = 'inversionBACK.sdat'

#.......................................
#Set the characteristics
#.......................................


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
           
            initCond = data['retrieval']['constraints'][f'characteristic[{i + 1}]'][f'mode[{noMd + 1}]']['initial_guess']

            if char_type == 'vertical_profile_normalized':
                try:
                    if AeroProf is not None: 
                        initCond['value'] =  AeroProf[noMd].tolist()
                        print(f"{char_type} Updated")
                except Exception as e:
                    print(char_type)
                    print(f"An error occurred: {e} for {char_type} ")
                    continue
            if char_type == 'aerosol_concentration':
                try: 
                    if len(NewVarDict['vol']) != 0: 
                        initCond['value'] = (float(NewVarDict['vol'][noMd]))
                        initCond['max'] = (float(NewVarDict['vol'][noMd] * maxr))
                        initCond['min'] = (float(NewVarDict['vol'][noMd] * minr))

                        print(f"{char_type} Updated")
                except Exception as e:
                    print(f"An error occurred for {char_type}: {e}")
                    continue


               
            
            if char_type == 'size_distribution_lognormal':
                try:
                    if len(NewVarDict['rv']) != 0 and len(NewVarDict['sigma']) != 0:
                        # Check if noMd is within range
                        if noMd < len(NewVarDict['rv']) and noMd < len(NewVarDict['sigma']):
                            initCond['value'] = (float(NewVarDict['rv'][noMd]), float(NewVarDict['sigma'][noMd]))
                            initCond['max'] = (float(NewVarDict['rv'][noMd] * maxr), float(NewVarDict['sigma'][noMd] * maxr))
                            initCond['min'] = (float(NewVarDict['rv'][noMd] * minr), float(NewVarDict['sigma'][noMd] * minr))

                            print(f"{char_type} Updated")
                        else:
                            print("Error: noMd index out of range for rv or sigma.")
                    else:
                        print("Warning: rv or sigma lists are empty.")
                except Exception as e:
                    print(f"An error occurred for {char_type}: {e}")
                    continue

                if ConsType == 'strict':
                    data['retrieval']['constraints'][f'characteristic[{i + a}]']['retrieved'] = 'false'
            
            elif char_type == 'real_part_of_refractive_index_spectral_dependent':
                
                try:
                    if len(NewVarDict['n'])!=0:

                        if DataIdxtoUpdate is None:
                            DataIdxtoUpdate = [i for i in range(len(NewVarDict['lambda']))]

                        
                        InitVal = [float(NewVarDict['n'][noMd][i]) for i in DataIdxtoUpdate]
                        Init = np.array(InitVal)  # Convert to array for numpy operations


                        # Init[Init > 1.68] = 1.67  # Apply limit for Max
                        # Init[Init < 1.33] = 1.34  # Apply limit for Max


                        initCond['value'] = Init.tolist()

                        # initCond['index_of_wavelength_involved']= np.array(1,len(NewVarDict['lambda']))

                        # Setting Max with a cap of 1.7
                        Max = [float(NewVarDict['n'][noMd][i] * maxr) for i in DataIdxtoUpdate]
                        Max = np.array(Max)  # Convert to array for numpy operations
                        # Max[Max > 1.68] = 1.68  # Apply limit for Max

                        # Setting Min with a floor of 1.33
                        Min = [float(NewVarDict['n'][noMd][i] * minr) for i in DataIdxtoUpdate]
                        Min = np.array(Min)  # Convert to array for numpy operations
                        # Min[Min < 1.33] = 1.33  # Apply limit for Min   

                        # Assigning the limits and index values to initCond
                        initCond['max'] = Max.tolist()  # Convert back to list if needed
                        initCond['min'] = Min.tolist()  # Convert back to list if needed
                        initCond['index_of_wavelength_involved'] = [i for i in range(1, len(NewVarDict['lambda']+1))]

                        print(f"{char_type} Updated")

                    if ConsType == 'strict':
                        data['retrieval']['constraints'][f'characteristic[{i + a}]']['retrieved'] = 'false'
                except Exception as e:
                    print(f"An error occurred for {char_type}: {e}")
                    continue
            elif char_type == 'imaginary_part_of_refractive_index_spectral_dependent':
                try:
                    if len(NewVarDict['k'])!=0:

                        if DataIdxtoUpdate is None:
                            DataIdxtoUpdate = [i for i in range(len(NewVarDict['lambda']))]

                        initCond['value'] = [float(NewVarDict['k'][noMd][i]) for i in DataIdxtoUpdate]
                        initCond['max'] = [float(NewVarDict['k'][noMd][i] * maxr) for i in DataIdxtoUpdate]
                        initCond['min'] = [float(NewVarDict['k'][noMd][i] * minr) for i in DataIdxtoUpdate]
                        initCond['index_of_wavelength_involved'] = [i for i in range(1, len(NewVarDict['lambda']+1))]

                        print(f"{char_type} Updated")

                    if ConsType == 'strict':
                        data['retrieval']['constraints'][f'characteristic[{i + a}]']['retrieved'] = 'false'
                except Exception as e:
                    print(f"An error occurred for {char_type}: {e}")
                    continue

            elif char_type == 'sphere_fraction':

                try:
                    if len(NewVarDict['sph'])!=0:

                        initCond['value'] = float(NewVarDict['sph'][noMd] / 100)
                        initCond['max'] = float(NewVarDict['sph'][noMd] * maxr / 100)
                        initCond['min'] = float(NewVarDict['sph'][noMd] * minr / 100)
                        print(f"{char_type} Updated")

                        if ConsType == 'strict':
                            data['retrieval']['constraints'][f'characteristic[{i + a}]']['retrieved'] = 'false'
                except Exception as e:
                    print(f"An error occurred for {char_type}: {e}")
                    continue

            elif char_type =='vertical_profile_parameter_standard_deviation':

                try:
                    if len(NewVarDict['heightStd'])!=0:
                        initCond['value'] = float(NewVarDict['heightStd'][noMd])
                        initCond['max'] = float(NewVarDict['heightStd'][noMd] * maxr)
                        initCond['min'] = float(NewVarDict['heightStd'][noMd] * minr)

                        print(f"{char_type} Updated")
                        
                except Exception as e:
                    print(f"An error occurred for {char_type}: {e}")
                    continue
            
            elif char_type =='vertical_profile_parameter_height':

                try:
                    if len(NewVarDict['height'])!=0:
                        initCond['value'] = float(NewVarDict['height'][noMd])
                        initCond['max'] = float(NewVarDict['height'][noMd] * maxr)
                        initCond['min'] = float(NewVarDict['height'][noMd] * minr)

                        print(f"{char_type} Updated")
                        
                except Exception as e:
                    print(f"An error occurred for {char_type}: {e}")
                    continue

            elif char_type == 'surface_water_cox_munk_iso':
                try:
                    if len(NewVarDict['height'])!=0:
                        initCond['value'] = float(NewVarDict['height'][noMd])
                        initCond['max'] = float(NewVarDict['height'][noMd] * maxr)
                        initCond['min'] = float(NewVarDict['height'][noMd] * minr)

                        initCond['index_of_wavelength_involved'] = [i for i in range(1, len(NewVarDict['lambda']+1))]

                except Exception as e:
                    print(f"An error occurred for {char_type}: {e}")
                    continue


            





        #.......................................
            # Save the updated YAML file
        #.......................................
    with open(UpdatedymlPath, 'w') as f:
        yaml.safe_dump(data, f)

        print(f"YAML file updated and saved as {UpdatedymlPath}")



        #..............HOW TO RUN..........................................

        # fwdModelYAMLpath = '/home/gregmi/git/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/settings_BCK_POLAR_3modes_Shape_Hex_Case2.yml'
        # HSRL_data = Read_Data_HSRL_Oracles_Height(HSRLfile_path,HSRLfile_name,HSRLPixNo)

        # # Kernel_type = 'hex'
        # # rsltDict = HSRL_data[0]
        # UpdateDict = HSRL_TamuT[0][0]
        # UpKerFile = 'settings_LIDARandPOLAR_3modes_Shape_Sph_Update.yml'
        # ymlPath = '/home/gregmi/git/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/'

        # AeroProf= AeroProfNorm_sc2(HSRL_data)

        #Up = update_HSRLyaml(ymlPath = ymlPath, UpKerFile=UpKerFile , YamlFileName = fwdModelYAMLpath , noMod = noMod , Kernel_type = Kernel_type, AeroProf=AeroProf, NewVarDict = UpdateDict, GRASPModel = 'fwd')




        #From the combined retrieval" 

        # fwdModelYAMLpath ='/home/gregmi/git/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/settings_BCK_LidarAndMAP_3modes_V.1.2_Simulate.yml'
        # UpKerFile = 'settings_LIDARandPOLAR_3modes_Shape_Sph_Update.yml'
        # ymlPath = '/home/gregmi/git/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/'

        # UpdateDict = LidarPolTAMU[0][0]
        # UpKerFile = 'settings_LIDARandPOLAR_3modes_Shape_Sph_Update.yml'
        # ymlPath = '/home/gregmi/git/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/'
        # Kernel_type = 'hex'

        # Up = update_HSRLyaml(ymlPath = ymlPath, UpKerFile=UpKerFile , YamlFileName = fwdModelYAMLpath , noMod = noMod , Kernel_type = Kernel_type, NewVarDict = UpdateDict, GRASPModel = 'fwd')

        #.....................................................................

        

    return UpdatedymlPath







# def update_HSRLyaml(UpdatedymlPath, YamlFileName: str, noMod: int, Kernel_type: str,  
#                     GRASPModel = None, AeroProf =None, ConsType= None, YamlChar=None, maxr=None, minr=None, 
#                     NewVarDict: dict = None, DataIdxtoUpdate=None): 
#     """
#     Update the YAML file for HSRL with initial conditions from polarimeter retrievals.
    
#     Arguments:
#     ymlPath: str              -- Path to which new updated setting file will be saved
#     UpdatedymlPath: str            -- path of the new settings file
#     YamlFileName: str         -- Path to the YAML file to be updated.

#     noMod: int                -- Number of aerosol modes to iterate over.
#     Kernel_type: str          -- Type of kernel ('spheroid' or 'hex').
#     ConsType: str             -- Constraint type ('strict' to fix retrieval).
#     YamlChar: list or None    -- Characteristics in the YAML file (optional).
#     maxr: float               -- Factor for maximum values.
#     minr: float               -- Factor for minimum values.
#     NewVarDict: dict          -- New variables to update the YAML with.
#     RSP_rslt: dict            -- RSP results containing aerosol properties.
#     LagrangianOnly: bool      -- Whether to only update Lagrangian multipliers.
#     RSPUpdate: bool           -- Whether to update initial conditions from RSP retrievals.
#     DataIdxtoUpdate: list     -- Index of data to update (optional).
#     AprioriLagrange: float    -- Lagrange multiplier for a priori estimates.
#     a: int                    -- Offset for characteristic indices (default: 0).
#     ymlPath: str              -- Path to save the updated YAML file (default: '').
#     UpKerFile: str            -- Filename for the updated YAML file (default: '').
#     GRASPModel: str           -- 'Fwd' for forward model or 'Bck'for inverse
#     AeroProf:list             -- (Apriori) Normalized vertical profile for each aerosol mode (For LIDARS), size = (Vertical grid X no of aerosol modes)

#     Returns:
#     None -- Writes the updated YAML data to a new file.
#     """
    




#     if maxr is None: 
#         maxr = 1

#     if minr is None: 
#         minr = 1

#     # Load the YAML file
#     with open(YamlFileName, 'r') as f:  
#         data = yaml.safe_load(f)

# #.......................................
# #Set the shape model
# #.......................................
    
#     if Kernel_type =='spheroid':
#         data['retrieval']['forward_model']['phase_matrix']['kernels_folder'] = 'KERNELS_BASE'
#     if Kernel_type =='hex':
#         data['retrieval']['forward_model']['phase_matrix']['kernels_folder'] = 'Ver_sph'

# #.......................................
# #Set GRASP to forward or inverse mode
# #.......................................


#     if GRASPModel != None:
#         if 'fwd' in GRASPModel.lower(): #If set to forward mode
#             data['retrieval']['mode'] = 'forward'
#             data['output']['segment']['stream'] = 'bench_FWD_IQUandLIDAR_rslts.txt'
#             data['input']['file'] = 'bench.sdat'
         
#         if 'bck' in GRASPModel.lower(): #If set to inverse mode
#             data['retrieval']['mode'] = 'inversion'
#             data['output']['segment']['stream'] = 'bench_inversionRslts.txt'
#             data['input']['file'] = 'inversionBACK.sdat'

# #.......................................
# #Set the characteristics
# #.......................................


#     if YamlChar is None:
#         # Find the number of characteristics in the settings (Yaml) file
#         NoChar = []  # No of characteristics in the YAML file
#         for key, value in data['retrieval']['constraints'].items():
#             # Match the strings with "characteristics"
#             match = re.match(r'characteristic\[\d+\]', key)
#             if match:
#                 # Extract the number 
#                 numbers = re.findall(r'\d+', match.group())
#                 NoChar.append(int(numbers[0]))

#         # All the characteristics in the settings file
#         YamlChar = [data['retrieval']['constraints'][f'characteristic[{i}]']['type'] for i in NoChar]



#     assert NewVarDict is not None, "NewVarDict must not be None"

#     for i, char_type in enumerate(YamlChar):
#         for noMd in range(noMod):
           
#             initCond = data['retrieval']['constraints'][f'characteristic[{i + 1}]'][f'mode[{noMd + 1}]']['initial_guess']

#             if char_type == 'vertical_profile_normalized':
#                 try:
#                     if AeroProf is not None: 
#                         initCond['value'] =  AeroProf[noMd].tolist()
#                         print(f"{char_type} Updated")
#                 except Exception as e:
#                     print(char_type)
#                     print(f"An error occurred: {e} for {char_type} ")
#                     continue
#             if char_type == 'aerosol_concentration':
#                 try: 
#                     if len(NewVarDict['vol']) != 0: 
#                         initCond['value'] = (float(NewVarDict['vol'][noMd]))
#                         initCond['max'] = (float(NewVarDict['vol'][noMd] * maxr))
#                         initCond['min'] = (float(NewVarDict['vol'][noMd] * minr))

#                         print(f"{char_type} Updated")
#                 except Exception as e:
#                     print(f"An error occurred for {char_type}: {e}")
#                     continue


               
            
#             if char_type == 'size_distribution_lognormal':
#                 try:
#                     if len(NewVarDict['rv']) != 0 and len(NewVarDict['sigma']) != 0:
#                         # Check if noMd is within range
#                         if noMd < len(NewVarDict['rv']) and noMd < len(NewVarDict['sigma']):
#                             initCond['value'] = (float(NewVarDict['rv'][noMd]), float(NewVarDict['sigma'][noMd]))
#                             initCond['max'] = (float(NewVarDict['rv'][noMd] * maxr), float(NewVarDict['sigma'][noMd] * maxr))
#                             initCond['min'] = (float(NewVarDict['rv'][noMd] * minr), float(NewVarDict['sigma'][noMd] * minr))

#                             print(f"{char_type} Updated")
#                         else:
#                             print("Error: noMd index out of range for rv or sigma.")
#                     else:
#                         print("Warning: rv or sigma lists are empty.")
#                 except Exception as e:
#                     print(f"An error occurred for {char_type}: {e}")
#                     continue

#                 if ConsType == 'strict':
#                     data['retrieval']['constraints'][f'characteristic[{i + a}]']['retrieved'] = 'false'
            
#             elif char_type == 'real_part_of_refractive_index_spectral_dependent':
                
#                 try:
#                     if len(NewVarDict['n'])!=0:

#                         if DataIdxtoUpdate is None:
#                             DataIdxtoUpdate = [i for i in range(len(NewVarDict['lambda']))]

                        
#                         InitVal = [float(NewVarDict['n'][noMd][i]) for i in DataIdxtoUpdate]
#                         Init = np.array(InitVal)  # Convert to array for numpy operations
#                         # Init[Init > 1.68] = 1.67  # Apply limit for Max
#                         # Init[Init < 1.33] = 1.34  # Apply limit for Max


#                         initCond['value'] = Init.tolist()

#                         # Setting Max with a cap of 1.7
#                         Max = [float(NewVarDict['n'][noMd][i] * maxr) for i in DataIdxtoUpdate]
#                         Max = np.array(Max)  # Convert to array for numpy operations
#                         # Max[Max > 1.68] = 1.68  # Apply limit for Max

#                         # Setting Min with a floor of 1.33
#                         Min = [float(NewVarDict['n'][noMd][i] * minr) for i in DataIdxtoUpdate]
#                         Min = np.array(Min)  # Convert to array for numpy operations
#                         # Min[Min < 1.33] = 1.33  # Apply limit for Min   

#                         # Assigning the limits and index values to initCond
#                         initCond['max'] = Max.tolist()  # Convert back to list if needed
#                         initCond['min'] = Min.tolist()  # Convert back to list if needed
#                         initCond['index_of_wavelength_involved'] = [i for i in range(len(NewVarDict['lambda']))]

#                         print(f"{char_type} Updated")

#                     if ConsType == 'strict':
#                         data['retrieval']['constraints'][f'characteristic[{i + a}]']['retrieved'] = 'false'
#                 except Exception as e:
#                     print(f"An error occurred for {char_type}: {e}")
#                     continue
#             elif char_type == 'imaginary_part_of_refractive_index_spectral_dependent':
#                 try:
#                     if len(NewVarDict['k'])!=0:

#                         if DataIdxtoUpdate is None:
#                             DataIdxtoUpdate = [i for i in range(len(NewVarDict['lambda']))]

#                         initCond['value'] = [float(NewVarDict['k'][noMd][i]) for i in DataIdxtoUpdate]
#                         initCond['max'] = [float(NewVarDict['k'][noMd][i] * maxr) for i in DataIdxtoUpdate]
#                         initCond['min'] = [float(NewVarDict['k'][noMd][i] * minr) for i in DataIdxtoUpdate]
#                         initCond['index_of_wavelength_involved'] = [i for i in range(len(NewVarDict['lambda']))]

#                         print(f"{char_type} Updated")

#                     if ConsType == 'strict':
#                         data['retrieval']['constraints'][f'characteristic[{i + a}]']['retrieved'] = 'false'
#                 except Exception as e:
#                     print(f"An error occurred for {char_type}: {e}")
#                     continue

#             elif char_type == 'sphere_fraction':

#                 try:
#                     if len(NewVarDict['sph'])!=0:

#                         initCond['value'] = float(NewVarDict['sph'][noMd] / 100)
#                         initCond['max'] = float(NewVarDict['sph'][noMd] * maxr / 100)
#                         initCond['min'] = float(NewVarDict['sph'][noMd] * minr / 100)
#                         print(f"{char_type} Updated")

#                         if ConsType == 'strict':
#                             data['retrieval']['constraints'][f'characteristic[{i + a}]']['retrieved'] = 'false'
#                 except Exception as e:
#                     print(f"An error occurred for {char_type}: {e}")
#                     continue

#             elif char_type =='vertical_profile_parameter_standard_deviation':

#                 try:
#                     if len(NewVarDict['heightStd'])!=0:
#                         initCond['value'] = float(NewVarDict['heightStd'][noMd])
#                         initCond['max'] = float(NewVarDict['heightStd'][noMd] * maxr)
#                         initCond['min'] = float(NewVarDict['heightStd'][noMd] * minr)

#                         print(f"{char_type} Updated")
                        
#                 except Exception as e:
#                     print(f"An error occurred for {char_type}: {e}")
#                     continue
            
#             elif char_type =='vertical_profile_parameter_height':

#                 try:
#                     if len(NewVarDict['height'])!=0:
#                         initCond['value'] = float(NewVarDict['height'][noMd])
#                         initCond['max'] = float(NewVarDict['height'][noMd] * maxr)
#                         initCond['min'] = float(NewVarDict['height'][noMd] * minr)

#                         print(f"{char_type} Updated")
                        
#                 except Exception as e:
#                     print(f"An error occurred for {char_type}: {e}")
#                     continue

#             elif char_type == 'surface_water_cox_munk_iso':
#                 try:
#                     if len(NewVarDict['height'])!=0:
#                         initCond['value'] = float(NewVarDict['height'][noMd])
#                         initCond['max'] = float(NewVarDict['height'][noMd] * maxr)
#                         initCond['min'] = float(NewVarDict['height'][noMd] * minr)
#                 except Exception as e:
#                     print(f"An error occurred for {char_type}: {e}")
#                     continue





#         #.......................................
#             # Save the updated YAML file
#         #.......................................
#     with open(UpdatedymlPath, 'w') as f:
#         yaml.safe_dump(data, f)

#         print(f"YAML file updated and saved as {UpdatedymlPath}")



#         #..............HOW TO RUN..........................................

#         # fwdModelYAMLpath = '/home/gregmi/git/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/settings_BCK_POLAR_3modes_Shape_Hex_Case2.yml'
#         # HSRL_data = Read_Data_HSRL_Oracles_Height(HSRLfile_path,HSRLfile_name,HSRLPixNo)

#         # # Kernel_type = 'hex'
#         # # rsltDict = HSRL_data[0]
#         # UpdateDict = HSRL_TamuT[0][0]
#         # UpKerFile = 'settings_LIDARandPOLAR_3modes_Shape_Sph_Update.yml'
#         # ymlPath = '/home/gregmi/git/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/'

#         # AeroProf= AeroProfNorm_sc2(HSRL_data)

#         #Up = update_HSRLyaml(ymlPath = ymlPath, UpKerFile=UpKerFile , YamlFileName = fwdModelYAMLpath , noMod = noMod , Kernel_type = Kernel_type, AeroProf=AeroProf, NewVarDict = UpdateDict, GRASPModel = 'fwd')




#         #From the combined retrieval" 

#         # fwdModelYAMLpath ='/home/gregmi/git/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/settings_BCK_LidarAndMAP_3modes_V.1.2_Simulate.yml'
#         # UpKerFile = 'settings_LIDARandPOLAR_3modes_Shape_Sph_Update.yml'
#         # ymlPath = '/home/gregmi/git/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/'

#         # UpdateDict = LidarPolTAMU[0][0]
#         # UpKerFile = 'settings_LIDARandPOLAR_3modes_Shape_Sph_Update.yml'
#         # ymlPath = '/home/gregmi/git/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/'
#         # Kernel_type = 'hex'

#         # Up = update_HSRLyaml(ymlPath = ymlPath, UpKerFile=UpKerFile , YamlFileName = fwdModelYAMLpath , noMod = noMod , Kernel_type = Kernel_type, NewVarDict = UpdateDict, GRASPModel = 'fwd')

#         #.....................................................................

        

#     return UpdatedymlPath





def Gaussian_fits(heights, aerosol_concentration):


    "Plot the Gaussian vertical profile from heignt and std output from Polarimeter"
    
    # Define the Gaussian function
    def gaussian(h, A, h0, sigma):
        return A * np.exp(-((h - h0) ** 2) / (2 * sigma ** 2))

    # Initial guesses for A, h0, sigma
    initial_guess = [1, np.mean(heights), np.std(heights)]

    # Fit the Gaussian curve to the data
    popt, pcov = curve_fit(gaussian, heights, aerosol_concentration, p0=initial_guess)

    # Extract the fitting parameters
    A_fit, h0_fit, sigma_fit = popt
    print(f"Fitted mean height (h0): {h0_fit}")
    print(f"Fitted standard deviation (sigma): {sigma_fit}")

    # Generate data points from the fitted Gaussian for plotting
    height_fit = np.linspace(min(heights), max(heights), 100)
    aerosol_fit = gaussian(height_fit, A_fit, h0_fit, sigma_fit)

    # Plot the original data and the Gaussian fit
    plt.figure(figsize=(8, 6))
    plt.plot(aerosol_concentration, heights, 'o', label='Original Data')
    plt.plot(aerosol_fit, height_fit, '-', label='Gaussian Fit')
    plt.xlabel('Aerosol Concentration')
    plt.ylabel('Height (km)')
    plt.title('Gaussian Fit to Aerosol Layer Height Profile')
    plt.legend()
    plt.grid(True)
    plt.show()

    return



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




    
def FactorforConstAOD(fineRRI,FixAOD,DictRslt,mode):

    #Mode: The aerosol mode to change value: For mode 1 - enter 1


    #Calculate the volume conc that keeps the AOD constant

    cpDictRslt = DictRslt
    constVolFactor = np.zeros(len(fineRRI))
    #New volume concetration to maintain a constant AOD
    NewVolConc = np.zeros(len(fineRRI))

    for i in range(len(fineRRI)):
        volFac= FixAOD/cpDictRslt[i]['aodMode'][mode-1][0]  #taking the first wavelength, mode-1 because 0 index
        constVolFactor[i] = volFac
        NewVolConc[i] = cpDictRslt[i]['vol'][0]*volFac



    #...................How to run..................................
    # cpDictRslt = DictRslt
    # constVolFactor = np.zeros(len(fineRRI))
    # FixAOD = 0.19895
    # NewVol = FactorforConstAOD(fineRRI,FixAOD,DictRslt)[1]
    #...............................................................
    
    return NewVolConc, constVolFactor



def CXmunk(V):

    '''V is the wind speed in m/s'''
    CxMunk = (0.003+0.00512*V)/2
    return CxMunk 




def Combine_MAPandLidar_Rsltdict(rslt_RSP,rslt_HSRL1, HSRLPixNo):

    """This funtion combines the rslt dict from Lidar and Polarimeter togetehr into a single rslt dictionary
        
        The rslt dict for RSP and HSRL are colocated.
        rslt_RSP = result dict for polarimeter
        rslt_HSRL = result dict fot Lidar
        TelNo  =  (Specific to Research Scanning Polarimeter) RSP has two telescopes, we will average the I and DoLP
    
    """
    rslt_HSRL= rslt_HSRL1[0]
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










def CXmunk(V):

    '''V is the wind speed in m/s'''
    CxMunk = np.sqrt(0.003+0.00512*V)
    return CxMunk 


def Combine_MAPandLidar_Rsltdict(rslt_RSP,HSRL_data, HSRLPixNo):


    rslt_HSRL  = HSRL_data[0]

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


    AeroProf = AeroProfNorm_sc2(HSRL_data)
    
    # AeroProf =AeroProfNorm_FMF(rslt_HSRL_1)
    FineProfext,DstProfext,SeaProfext = AeroProf[0],AeroProf[1],AeroProf[2]



    rslt['VertProf_Mode1'] = AeroProf[0]  #Fine mode 
    rslt['VertProf_Mode2'] = AeroProf[1]
    rslt['VertProf_Mode3'] = AeroProf[2]


    return rslt


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


def update_HSRLyaml(ymlPath, UpKerFile, YamlFileName: str, noMod: int, Kernel_type: str,  
                    GRASPModel = None, AeroProf =None, ConsType= None, YamlChar=None, maxr=None, minr=None, 
                    NewVarDict: dict = None, DataIdxtoUpdate=None): 
    """
    Update the YAML file for HSRL with initial conditions from polarimeter retrievals.
    
    Arguments:
    ymlPath: str              -- Path to which new updated setting file will be saved
    UpKerFile: str            -- Name of the new settings file
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
    GRASPModel: str           -- 'Fwd' for forward model or 'Bck'for inverse
    AeroProf:list             -- (Apriori) Normalized vertical profile for each aerosol mode (For LIDARS), size = (Vertical grid X no of aerosol modes)

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

#.......................................
#Set the shape model
#.......................................
    
    if Kernel_type =='spheroid':
        data['retrieval']['forward_model']['phase_matrix']['kernels_folder'] = 'KERNELS_BASE'
    if Kernel_type =='hex':
        data['retrieval']['forward_model']['phase_matrix']['kernels_folder'] = 'Ver_sph'

#.......................................
#Set GRASP to forward or inverse mode
#.......................................


    if GRASPModel != None:
        if 'fwd' in GRASPModel.lower(): #If set to forward mode
            data['retrieval']['mode'] = 'forward'
            data['output']['segment']['stream'] = 'bench_FWD_IQUandLIDAR_rslts.txt'
            data['input']['file'] = 'bench.sdat'
         
        if 'bck' in GRASPModel.lower(): #If set to inverse mode
            data['retrieval']['mode'] = 'inversion'
            data['output']['segment']['stream'] = 'bench_inversionRslts.txt'
            data['input']['file'] = 'inversionBACK.sdat'

#.......................................
#Set the characteristics
#.......................................


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
           
            initCond = data['retrieval']['constraints'][f'characteristic[{i + 1}]'][f'mode[{noMd + 1}]']['initial_guess']

            if char_type == 'vertical_profile_normalized':
                try:
                    if AeroProf is not None: 
                        initCond['value'] =  AeroProf[noMd].tolist()
                        print(f"{char_type} Updated")
                except Exception as e:
                    print(char_type)
                    print(f"An error occurred: {e} for {char_type} ")
                    continue
            if char_type == 'aerosol_concentration':
                try: 
                    if len(NewVarDict['vol']) != 0: 
                        initCond['value'] = (float(NewVarDict['vol'][noMd]))
                        initCond['max'] = (float(NewVarDict['vol'][noMd] * maxr))
                        initCond['min'] = (float(NewVarDict['vol'][noMd] * minr))

                        print(f"{char_type} Updated")
                except Exception as e:
                    print(f"An error occurred for {char_type}: {e}")
                    continue


               
            
            if char_type == 'size_distribution_lognormal':
                try:
                    if len(NewVarDict['rv']) != 0 and len(NewVarDict['sigma']) != 0:
                        # Check if noMd is within range
                        if noMd < len(NewVarDict['rv']) and noMd < len(NewVarDict['sigma']):
                            initCond['value'] = (float(NewVarDict['rv'][noMd]), float(NewVarDict['sigma'][noMd]))
                            initCond['max'] = (float(NewVarDict['rv'][noMd] * maxr), float(NewVarDict['sigma'][noMd] * maxr))
                            initCond['min'] = (float(NewVarDict['rv'][noMd] * minr), float(NewVarDict['sigma'][noMd] * minr))

                            print(f"{char_type} Updated")
                        else:
                            print("Error: noMd index out of range for rv or sigma.")
                    else:
                        print("Warning: rv or sigma lists are empty.")
                except Exception as e:
                    print(f"An error occurred for {char_type}: {e}")
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

                        print(f"{char_type} Updated")

                    if ConsType == 'strict':
                        data['retrieval']['constraints'][f'characteristic[{i + a}]']['retrieved'] = 'false'
                except Exception as e:
                    print(f"An error occurred for {char_type}: {e}")
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

                        print(f"{char_type} Updated")

                    if ConsType == 'strict':
                        data['retrieval']['constraints'][f'characteristic[{i + a}]']['retrieved'] = 'false'
                except Exception as e:
                    print(f"An error occurred for {char_type}: {e}")
                    continue

            elif char_type == 'sphere_fraction':

                try:
                    if len(NewVarDict['sph'])!=0:

                        initCond['value'] = float(NewVarDict['sph'][noMd] / 100)
                        initCond['max'] = float(NewVarDict['sph'][noMd] * maxr / 100)
                        initCond['min'] = float(NewVarDict['sph'][noMd] * minr / 100)
                        print(f"{char_type} Updated")

                        if ConsType == 'strict':
                            data['retrieval']['constraints'][f'characteristic[{i + a}]']['retrieved'] = 'false'
                except Exception as e:
                    print(f"An error occurred for {char_type}: {e}")
                    continue

            elif char_type =='vertical_profile_parameter_standard_deviation':

                try:
                    if len(NewVarDict['heightStd'])!=0:
                        initCond['value'] = float(NewVarDict['heightStd'][noMd])
                        initCond['max'] = float(NewVarDict['heightStd'][noMd] * maxr)
                        initCond['min'] = float(NewVarDict['heightStd'][noMd] * minr)

                        print(f"{char_type} Updated")
                        
                except Exception as e:
                    print(f"An error occurred for {char_type}: {e}")
                    continue
            
            elif char_type =='vertical_profile_parameter_height':

                try:
                    if len(NewVarDict['height'])!=0:
                        initCond['value'] = float(NewVarDict['height'][noMd])
                        initCond['max'] = float(NewVarDict['height'][noMd] * maxr)
                        initCond['min'] = float(NewVarDict['height'][noMd] * minr)

                        print(f"{char_type} Updated")
                        
                except Exception as e:
                    print(f"An error occurred for {char_type}: {e}")
                    continue



#.......................................
    # Save the updated YAML file
#.......................................
    with open(ymlPath + UpKerFile, 'w') as f:
        yaml.safe_dump(data, f)

    print(f"YAML file updated and saved as {ymlPath + UpKerFile}")



#..............HOW TO RUN..........................................

# fwdModelYAMLpath = '/home/gregmi/git/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/settings_BCK_POLAR_3modes_Shape_Hex_Case2.yml'
# HSRL_data = Read_Data_HSRL_Oracles_Height(HSRLfile_path,HSRLfile_name,HSRLPixNo)

# # Kernel_type = 'hex'
# # rsltDict = HSRL_data[0]
# UpdateDict = HSRL_TamuT[0][0]
# UpKerFile = 'settings_LIDARandPOLAR_3modes_Shape_Sph_Update.yml'
# ymlPath = '/home/gregmi/git/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/'

# AeroProf= AeroProfNorm_sc2(HSRL_data)

#Up = update_HSRLyaml(ymlPath = ymlPath, UpKerFile=UpKerFile , YamlFileName = fwdModelYAMLpath , noMod = noMod , Kernel_type = Kernel_type, AeroProf=AeroProf, NewVarDict = UpdateDict, GRASPModel = 'fwd')




#From the combined retrieval" 

# fwdModelYAMLpath ='/home/gregmi/git/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/settings_BCK_LidarAndMAP_3modes_V.1.2_Simulate.yml'
# UpKerFile = 'settings_LIDARandPOLAR_3modes_Shape_Sph_Update.yml'
# ymlPath = '/home/gregmi/git/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/'

# UpdateDict = LidarPolTAMU[0][0]
# UpKerFile = 'settings_LIDARandPOLAR_3modes_Shape_Sph_Update.yml'
# ymlPath = '/home/gregmi/git/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/'
# Kernel_type = 'hex'

# Up = update_HSRLyaml(ymlPath = ymlPath, UpKerFile=UpKerFile , YamlFileName = fwdModelYAMLpath , noMod = noMod , Kernel_type = Kernel_type, NewVarDict = UpdateDict, GRASPModel = 'fwd')

#.....................................................................

   

    return ymlPath + UpKerFile


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
                    minr=1, maxr=1, noMod=3, Kernel_type = Shape, 
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


#...........................how to run ..................
# UpdateDict['n'][0][0] = 1.45

# Up = update_HSRLyaml(ymlPath = ymlPath, UpKerFile=UpKerFile , YamlFileName = fwdModelYAMLpath , noMod = noMod , Kernel_type = Kernel_type, AeroProf=AeroProf, NewVarDict = UpdateDict, GRASPModel = 'fwd')
# Updatedyaml = Up

# # Create and populate a pixel from HSRL result
# pix = pixel()
# pix.populateFromRslt(rsltDict, radianceNoiseFun=None, dataStage='meas', verbose=False)

# # Set up graspRun class, add pixel, and run GRASP
# gr = graspRun(pathYAML=Updatedyaml, releaseYAML=True, verbose=True)
# gr.addPix(pix)  # Add pixel to the graspRun instance
# gr.runGRASP(binPathGRASP=binPathGRASP, krnlPathGRASP=krnlPath)  # Run GRASP

# # Print AOD at last wavelength
# print('AOD at %5.3f μm was %6.4f.' % (gr.invRslt[0]['lambda'][-1], gr.invRslt[0]['aod'][-1]))


# plot_HSRL(gr.invRslt[0],gr.invRslt[0],UNCERT, forward = True, retrieval = True, Createpdf = True,PdfName ="/home/gregmi/ORACLES/rsltPdf/HSRL_Only_Plots_444.pdf", combinedVal =HSRL_TamuT[2])
#...................................................


def Interpolate(HSRLRslt, RSPwl, NoMode = None, Plot=False):

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
    
        if Plot == True:

            plt.plot(HSRLwl, HSRLn[mode],'-',marker = 'o', label='HSRL')
            plt.plot(RSPwl,RSPn[mode], '-',marker = 'o', label='RSP')
            plt.legend()
            plt.show()

    IntpDict['n'] =  RSPn
    IntpDict['k'] =  RSPk

    return IntpDict



fineRRI = np.linspace(1.35, 1.6, 8)

def RunSensitivitySingleVal(constAOD = True, UpdateforconstAOD =None ):

    

    Fine_RRI =[]
    for i in range(len(fineRRI)):

        print(i)

        UpdateDict['n'][0][0] = fineRRI[i]

        if UpdateforconstAOD == True:


            Up = update_HSRLyaml(ymlPath = ymlPath, UpKerFile=UpKerFile , YamlFileName = fwdModelYAMLpath , noMod = noMod , Kernel_type = Kernel_type, AeroProf=AeroProf, NewVarDict = UpdateDict, GRASPModel = 'fwd')

        Updatedyaml = Up

        # Create and populate a pixel from HSRL result
        pix = pixel()
        pix.populateFromRslt(rsltDict, radianceNoiseFun=None, dataStage='meas', verbose=False)

        # Set up graspRun class, add pixel, and run GRASP
        gr = graspRun(pathYAML=Updatedyaml, releaseYAML=True, verbose=True)
        gr.addPix(pix)  # Add pixel to the graspRun instance
        gr.runGRASP(binPathGRASP=binPathGRASP, krnlPathGRASP=krnlPath)  # Run GRASP

        # Print AOD at last wavelength
        print('AOD at %5.3f μm was %6.4f.' % (gr.invRslt[0]['lambda'][-1], gr.invRslt[0]['aod'][-1]))

        Fine_RRI.append(gr.invRslt[0])



        if constAOD == True:

            #If AOD is fixed to a constant value
             NewVol =FactorforConstAOD(fineRRI,FixAOD,DictRslt)[0]

            

        plot_HSRL(gr.invRslt[0],gr.invRslt[0],UNCERT, forward = True, retrieval = True, Createpdf = True,PdfName ="/home/gregmi/ORACLES/rsltPdf/HSRL_Only_Plots_444.pdf", combinedVal =HSRL_TamuT[2])




def Plot_sensitivity( Idx, DictRslt, ChangeVariable):
    # sort_Lidar = np.array([0, 3, 7])

    colors = [
        'y',
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

    sort_Lidar = Idx
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


    plt.savefig("Lidar_sensitivity_rv.png", dpi = 120)



    fig,ax = plt.subplots(5,3, figsize = (12,12))
    for Itr in range(len(ChangeVariable)):
        for i in range(3):
            # ax[i].plot(DictRslt[Itr]['sca_ang'][:,i],DictRslt[0]['meas_I'][:,i])
            ax[0,i].plot(DictRslt[Itr]['r'][i],DictRslt[Itr][ 'dVdlnr'][i], color = colors[Itr])
            ax[0,i].set_xscale('log')


            ax[1,i].plot(DictRslt[Itr]['lambda'],DictRslt[Itr][ 'n'][i], color = colors[Itr])
            ax[2,i].plot(DictRslt[Itr]['lambda'],DictRslt[Itr][ 'k'][i], color = colors[Itr])
            ax[3,i].plot(DictRslt[Itr]['lambda'],DictRslt[Itr][ 'ssaMode'][i], color = colors[Itr])
            ax[4,i].plot(DictRslt[Itr]['lambda'],DictRslt[Itr][ 'aodMode'][i], color = colors[Itr])

            
    plt.tight_layout()
    plt.savefig("Lidar_sensitivity_rv_data2.png", dpi = 120)



def FactorforConstAOD(fineRRI,FixAOD,DictRslt):
    cpDictRslt = DictRslt
    constVolFactor = np.zeros(len(fineRRI))
    #New volume concetration to maintain a constant AOD
    NewVolConc = np.zeros(len(fineRRI))

    for i in range(len(fineRRI)):
        volFac= FixAOD/cpDictRslt[i]['aodMode'][0][0]  #taking the first wavelength 
        constVolFactor[i] = volFac
        NewVolConc[i] = cpDictRslt[i]['vol'][0]*volFac



    #...................How to run..................................
    # cpDictRslt = DictRslt
    # constVolFactor = np.zeros(len(fineRRI))
    # FixAOD = 0.19895
    # NewVol = FactorforConstAOD(fineRRI,FixAOD,DictRslt)[1]
    #...............................................................
    
    return NewVolConc, constVolFactor

    



# All_cases ={}
# fwdModelYAMLpath = '/home/gregmi/git/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/settings_BCK_DoLP_POLAR_3modes_HexShape_ORACLE_case2.yml'

# Shape = 'spheroid'
# rsltDict = rslt_RSP
# UpdateDict = {'rv': HSRL_sphrodT[0][0]['rv'],'sigma': HSRL_sphrodT[0][0]['sigma'] }
# UpKerFile = 'settings_LIDARandPOLAR_3modes_Shape_Sph_Update.yml'
# ymlPath = '/home/gregmi/git/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/'
# Sph_HSRLsizetoRSP= RunGRASPwUpdateYaml(Shape, ymlPath, fwdModelYAMLpath, UpKerFile, binPathGRASP, krnlPath, UpdateDict, rsltDict,  Plot = False)
# All_cases['Sph_HSRLsizetoRSP'].append(Sph_HSRLsizetoRSP)

# update_HSRLyaml(ymlPath = ymlPath, UpKerFile=UpKerFile , YamlFileName = fwdModelYAMLpath , noMod = noMod , Kernel_type = Kernel_type, AeroProf=AeroProf, NewVarDict = UpdateDict, GRASPModel = 'fwd')
#    Up = update_HSRLyaml(ymlPath = ymlPath, UpKerFile=UpKerFile , YamlFileName = fwdModelYAMLpath , noMod = noMod , Kernel_type = Kernel_type, AeroProf=AeroProf, NewVarDict = UpdateDict, GRASPModel = 'fwd')

# Updatedyaml = Up

# # Create and populate a pixel from HSRL result
# pix = pixel()
# pix.populateFromRslt(rsltDict, radianceNoiseFun=None, dataStage='meas', verbose=False)

# # Set up graspRun class, add pixel, and run GRASP
# gr = graspRun(pathYAML=Updatedyaml, releaseYAML=True, verbose=True)
# gr.addPix(pix)  # Add pixel to the graspRun instance
# gr.runGRASP(binPathGRASP=binPathGRASP, krnlPathGRASP=krnlPath)  # Run GRASP

# # Print AOD at last wavelength
# print('AOD at %5.3f μm was %6.4f.' % (gr.invRslt[0]['lambda'][-1], gr.invRslt[0]['aod'][-1]))
    

# plot_HSRL(gr.invRslt[0],gr.invRslt[0],UNCERT, forward = True, retrieval = True, Createpdf = True,PdfName ="/home/gregmi/ORACLES/rsltPdf/HSRL_Only_Plots_444.pdf", combinedVal =HSRL_TamuT[2])









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
        
    fig.savefig(f'/data/home/gregmi/Data/RSP_HSRL_oneStep/AllRetrieval{RepMode}_case1.png', dpi = 200,  transparent=True)
    
    # plt.suptitle(f'RSP Aerosol Retrieval \n  Lat:{lat_t} Lon :{lon_t}   Date: {dt_t}')
    # if RepMode ==2:
    #     fig.savefig(f'/home/gregmi/ORACLES/HSRL_RSP/2+3_RSPRetrieval.png', dpi = 300)
    # else:
    #     fig.savefig(f'/home/gregmi/ORACLES/HSRL_RSP/{NoMode}_RSPRetrieval.png', dpi = 300)


# PlotcombEachMode(rslts_Sph,rslts_Tamu,HSRL_sphrodT,HSRL_TamuT,LidarPolSph,LidarPolTAMU,costValCal)



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

# PlotDiffLidarCases(HSRL_sphrodT[0][0], HSRL_TamuT[0][0] ,LidarPolSph[0][0], LidarPolTAMU[0][0],HSRL_sphrodT[0][0], HSRL_TamuT[0][0],LidarPolSph[0][0], LidarPolTAMU[0][0], NoOfCases = 1)

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

# df=ErrHeatMap(rslts_Sph[0],rslts_Tamu[0])

#Caculating the chi-square



def chiSquare(meas,fit,measErr):
    chisquare = np.sum(((meas-fit)/measErr)**2) 

    return chisquare


