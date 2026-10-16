

#This code creates the GRASP hexahderdal Kernel using TAMUdust2020 database
#Greema Regmi


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from scipy import stats
import os
import fortranformat as ff
import pickle


def update_spher(nml_file_path, new_spher_value):
    # Open the .nml file in read mode to load all lines
    with open(nml_file_path, 'r') as f:
        lines = f.readlines()  # Read all lines into a list

    # Open the same file in write mode to overwrite its contents
    with open(nml_file_path, 'w') as f:
        for line in lines:
            # Check if the line contains the 'spher' parameter (ignoring leading whitespace)
            if line.strip().startswith('spher'):
                # Write the updated spher line using scientific notation with 7 decimal digits
                # Retain the original inline comment
                f.write(f' spher = {new_spher_value:.7E}  ! Sphericity of paricles\n')
            else:
                # If the line is not the one we want to change, write it back unchanged
                f.write(line)




def volume_kernels(Qext1, Qabs1, Area1, Volume1, size_descriptor=None, degree_of_sphericity=None):
    """GRASP extinction and absorption kernels from TAMUdust2020 output.

    GRASP kernels are cross sections per unit particle volume times the radius-grid spacing,
    b = C/V * dlnr [1/um]. TAMUdust2020's extinction efficiency Qe is relative to the
    orientation-averaged projected area A (Qe = C_ext/A, Saito et al. 2021), so C/V = Qe*A/V for
    every size descriptor. (Before 2026-10 the VER kernels used Qe*3/(4*r_v) = Qe*pi*r_v^2/V, which
    is low by the sphericity, about 0.70.)

    Qext1, Qabs1, Area1, Volume1: arrays (n_real_RI, n_imag_RI, n_size) from isca.dat (Qe, Qe*(1-SSA),
    projected area, volume). size_descriptor:
      'VER' (default) or 'Allsaitoang': radius grid = volume-equivalent radius cbrt(3V/(4 pi))
      'TAMUratioA2V_VER': same (kept for backward compatibility)
      'ER', 'TAMUratioA2V_ER': radius grid = effective radius 3V/(4A); the kernel values are the same C/V
    Returns b_ext, b_abs, radius grid, dlnr, and a default folder name.
    """
    if size_descriptor in (None, 'VER'):
        folder_name = f"Hex_VER_{degree_of_sphericity}"
        Radius = np.cbrt(3*Volume1[1,1,:]/(np.pi*4))
    elif size_descriptor == 'Allsaitoang':
        folder_name = "Saito_GEOS_VarAngRes"  # volume-equivalent radius, variable angular grid for GEOS
        Radius = np.cbrt(3*Volume1[1,1,:]/(np.pi*4))
    elif size_descriptor == 'TAMUratioA2V_VER':
        folder_name = "Ver_TAMU"
        Radius = np.cbrt(3*Volume1[1,1,:]/(np.pi*4))
    elif size_descriptor == 'ER':
        folder_name = "Eff_sph"
        Radius = 3*Volume1[1,1,:]/(4*Area1[1,1,:])
    elif size_descriptor == 'TAMUratioA2V_ER':
        folder_name = "Eff_TAMU"
        Radius = 3*Volume1[1,1,:]/(4*Area1[1,1,:])
    else:
        raise ValueError(f'unknown size_descriptor {size_descriptor!r}')
    dlnr = np.log(Radius[1]/Radius[0])
    assert np.allclose(np.diff(np.log(Radius)), dlnr, rtol=1e-3), 'radius grid must be log-uniform (GRASP uses a single dlnr); TAMUdust2020 v1.0 does not give this, use master'
    b_ext = Qext1*Area1*dlnr/Volume1   # C_ext/V * dlnr = Qe*A/V * dlnr
    b_abs = Qabs1*Area1*dlnr/Volume1
    # Build-time check: in the volume-equivalent sphere convention, C_ext/(pi r_v^2) -> 2*A/(pi r_v^2) at large x
    Rv = np.cbrt(3*Volume1[1,1,-1]/(np.pi*4))
    alpha = Area1[1,1,-1]/(np.pi*Rv**2)
    q_v = b_ext[:,0,-1]/dlnr*4*Rv/3
    assert np.all(np.abs(q_v/(2*alpha) - 1) < 0.05), f'C_ext/(pi r_v^2) at the largest size ({q_v}) is not within 5% of 2*A/(pi r_v^2) = {2*alpha}'
    return b_ext, b_abs, Radius, dlnr, folder_name


def ensemble_sphericity(Area1, Volume1):
    """Sphericity psi = pi*r_v^2/A of the ensemble TAMUdust2020 delivered (mean over sizes) and alpha = 1/psi.
    At a nominal sphericity between the database bins it differs slightly from the nominal value
    (e.g. 0.70064 at 0.7), because A and V are interpolated separately between the bins."""
    Rv = np.cbrt(3*Volume1[1,1,:]/(np.pi*4))
    psi = np.pi*Rv**2/Area1[1,1,:]
    return float(psi.mean()), float((1/psi).mean())


def RunTAMUdust2020(df, nml_file_path,SizeParam, RealRi,Imag1,AngVal,Allangle,size_descriptor=None, degree_of_sphericity = None,  params_path=None,output_path=None):


    wl=0.34 # in um
    Diameter =(SizeParam*wl)/(2*np.pi)    #from the definition of the size parameter in the TAMU DATA base which is 2*pi*D/wl
    Radius =Diameter/2
    len(Diameter)

    #Diameter parameter file for TAMU

    # params_path = '/home/gregmi/TAMU_project/TAMUdust2020-v1.0/examples/params/TAMUdust2020_Size'
    params_path_size = '/home/gregmi/TAMU_project/TAMUdust2020-master/examples/params/TAMUdust2020_Size'
    with open(params_path_size, "w") as file_object:
        for i in range(len(Diameter) ):
            file_object.write(f'{"{:.7e}".format(Diameter[i])}')

            file_object.write("\n")

    #Wavelength parameter file for TAMU
    # Imag1 = df['Imaginary Refractive Index:'][:31].values

    params_path_wl = "/home/gregmi/TAMU_project/TAMUdust2020-master/examples/params/TAMUdust2020_Wavelength_exp1"
    with open(params_path_wl, "w") as file_object:
        for i in range(len(Imag1)):

            file_object.write(f'{"{:.7e}".format(wl)}')
            file_object.write("\n")



    if params_path==None:
    #Path to parameter files for refractive index
        params_path="/home/gregmi/TAMU_project/TAMUdust2020-master/examples/params/TAMUdust2020_RefractiveIndex_exp1"

    if output_path==None:
        output_path="/home/gregmi/TAMU_project/TAMUdust2020-master/examples/output/TAMUdust2020create_exp1"


    #UPDATING THE nml file with the given degree of sphericity
    update_spher(nml_file_path, degree_of_sphericity)
    

    #Storing the TAMU results for all the optical properites in the following lists #Scattering properties
    Qext_max=[]
    Qsca_max=[]
    Area_max=[]
    Volume_max=[]
    Dimension_max=[]
    SSA_max=[]
    g =[]

    #Phase function: the following list stores all the output values from the TAMU run
    P11=[]
    P12=[]
    P22=[]
    P33=[]
    P43=[]
    P44=[]

    #We cannot change the TAMUdust angle range : (0,180,0.1). Extrating the values  Tamu has more finer angular resolution 

    Angles = AngVal
    # Angles =np.linspace(0,180,181)

    Ang_format=[]
    for i in range(len(Angles)):
        Ang_format.append("{:.7E}".format(Angles[i]))

        
    #TAMUdust2020create calculates optical properties at each wavelength.
    #reapeating the values of wavelength and real refractive indices for different v
    #updating the RI paramter files for each values of real imaginary refractive ind Imag1 - dfl'Imaginary Refractive Index:'][:31].values
    for j in range (len(RealRi)):
            with open(params_path, "w") as file_object:
                for i in range(len(Imag1)):
                    file_object.write("{:.5E}".format(RealRi[j])+' '+"\t"+"{: .5E}".format(Imag1[i]))
                    file_object.write("\n")
            print(f'Running {output_path}')


            os.system('/home/gregmi/TAMU_project/TAMUdust2020-master/bin/tamudust2020create /home/gregmi/TAMU_project/TAMUdust2020-master/examples/TAMUdust2020create_exp1.nml')
            
            
            #Reading the output files
                                    
            fn_Scattering = output_path+'/isca.dat'
            df1 = pd.read_csv(fn_Scattering, delim_whitespace=True, usecols=[0, 1, 2, 3,4,5,6], names =["Wl","Dimension","Volume","Projected_area","Qe", "SSA","g"])
                                                                            
            Qext_max.append(df1["Qe"])
            Qsca_max.append(df1["Qe"]*df1["SSA"])#scattering efficiency
            Area_max.append(df1[ 'Projected_area'])
            Volume_max.append(df1['Volume'])
            Dimension_max.append(df1['Dimension'])
            SSA_max.append(df1["SSA"])
            g.append(df1['g'])
                                                                            
            fn_Scattering11 = output_path+'/P11.dat'
            df1 = pd.read_csv(fn_Scattering11,delim_whitespace=True)
        
            if Allangle == True:
                P11.append(df1)
            else:
                P11.append(df1[Ang_format][:])
                
                                                           
            fn_Scattering12 = output_path+'/P12.dat'
            df2 = pd.read_csv(fn_Scattering12,delim_whitespace=True)

            if Allangle == True:
                P12.append(df2)
            else:
                P12.append(df2[Ang_format][:])
                                                                            
            fn_Scattering22 = output_path+'/P22.dat'
            df3 = pd.read_csv(fn_Scattering22, delim_whitespace=True)
            if Allangle == True:
                P22.append(df3)
            else:    
                P22.append(df3[Ang_format][:])
                                                                            
            fn_Scattering33 = output_path+'/P33.dat'
            df4 = pd.read_csv(fn_Scattering33, delim_whitespace=True)
            
            if Allangle == True:
                P33.append(df4)
            else:
                P33.append(df4[Ang_format][:])
                                                                            
            fn_Scattering44 = output_path+'/P44.dat'
            df5 = pd.read_csv(fn_Scattering44, delim_whitespace=True)

            if Allangle == True: 
                P44.append(df5[:])
            else:
                P44.append(df5[Ang_format])
                                                                            
            fn_Scattering43 = output_path+'/P43.dat'
            df6 = pd.read_csv(fn_Scattering43, delim_whitespace=True) 
            if Allangle == True:
                P43.append(df6[:])

            else:
                P43.append(df6[Ang_format])
            print("Done")


    #reshaping all the TAMU output as (n_real_RI, n_imag_RI, n_size)

    Dimension1 = np.array(Dimension_max).reshape(8,31,273)
    Area1 =  np.array(Area_max).reshape(8,31,273)
    Volume1 = np.array(Volume_max).reshape(8,31,273)
    Qext1 = np.array(Qext_max).reshape(8,31,273)
    Qsca1= np.array(Qsca_max).reshape(8,31,273)
    Asy_parameter_g = np.array(g).reshape(8,31,273)
    Qabs1=Qext1-Qsca1
    wl = "0.34000"

    #UNcomment the folder name that you want to create

    # Volume kernels (C/V * dlnr) and radius grid for the chosen size descriptor (see volume_kernels)
    b_ext, b_abs, Radius_v, dlnr, folder_name = volume_kernels(Qext1, Qabs1, Area1, Volume1, size_descriptor, degree_of_sphericity)
    psi_e, alpha = ensemble_sphericity(Area1, Volume1)

    b_sca = b_ext - b_abs

    #Calculation of radius
    plt.figure(figsize =(10,6))
    plt.plot(Dimension1[0,24,:]/2, label ="maximum dimension")
    plt.plot(np.sqrt(Area1[0,24,:]/np.pi),label ="projected-area equivalent sphere")
    plt.plot(np.cbrt(3*Volume1[0,24,:]/(np.pi*4)),label ="volume-equivalent sphere radius")
    plt.plot(3*Volume1[0,24,:]/(4*Area1[0,24,:]), label= "Effective radius")
    plt.ylabel("Radius", fontsize=15)
    plt.legend( fontsize =13)

    Asy_parameter_g = np.array(g).reshape(8,31,273)


    HexDict= {}

    #Storing the TAMU results for all the optical properites in the following lists #Scattering properties
    HexDict['Qext'] =b_ext
    HexDict['Qsca'] =b_sca
    HexDict['Area']=Area_max
    HexDict['Volume']=Volume_max
    HexDict['Dimension']=Dimension_max
    HexDict['SSA_max']= SSA_max
    HexDict['g'] = g

    #Phase function: the following list stores all the output values from the TAMU run
    HexDict['P11']=P11
    HexDict['P12']=P12
    HexDict['P22']=P22
    HexDict['P33']=P33
    HexDict['P43'] = P43
    HexDict['P44'] = P44

    HexDict['rv'] = Radius_v
    HexDict['rri'] = RealRi
    HexDict['iri'] = Imag1 
    HexDict['theta'] =AngVal
    HexDict['sph'] = degree_of_sphericity
    HexDict['psi_e'] = psi_e      # sphericity of the delivered ensemble, from TAMUdust2020 A and V
    HexDict['alpha'] = alpha      # A/(pi*r_v^2) = 1/psi_e
    HexDict['size_descriptor'] = size_descriptor if size_descriptor is not None else 'VER'
    HexDict['dlnr'] = dlnr



    with open(f'/home/gregmi/git/GSFC-GRASP-Python-Interface/TAMU_GRASP_Kernel_Generator/TAMUSimulationDiffSph/TAMUsim{degree_of_sphericity}.pickle', 'wb') as f:
        pickle.dump(HexDict, f)
        f.close()

    return HexDict




def Phase_Function_299(Dict1,a,folder_name):

    Angles = Dict1['theta']
    Radius_v = Dict1['rv']
    RealRi = Dict1['rri']  
    Imag1 = Dict1['iri']
    P11 = Dict1['P11']
    b_sca = Dict1['Qsca']

# def Phase_Function_299(P11,b_sca,b_ext, b_abs, a,folder_name,AngVal,Radius_v,RealRi, Imag1):

     
    wl='0.34000'#used wavelength


    if a =='grid':
    
            
        with open(f'{folder_name}/grid1.dat', 'w') as f:
            f.write(f' {len(Radius_v)} 0.340\n')
            for i in range(len(Radius_v) ) :
                notation = str(0)+ff.FortranRecordWriter('(E12.7)').write([Radius_v[i]])
                f.write(f'{str(notation )}\n')
            f.write(f'\t{len(Angles)}\n')
            for i in range(len(Angles)):
                val ="{:.2f}".format(Angles[i])
                f.write(f'{str(val)}\n')

    #Writes the phase funtions into GRASP Kernel format

    elif a == 5:

        b_ext = Dict1['Qext']
        b_abs = b_ext-b_sca

        # Provenance: what the kernel values are and the geometry behind them
        info = {'contents': 'GRASP kernels: C/V * dlnr [1/um] at the reference wavelength (EXTINCTION, ABSORPTION); '
                            'absolute scattering-matrix elements F_ij * C_sca/V * dlnr with 0.5*int(F11 sin) = 1. '
                            'Element 34 = -P43 (TAMUdust2020) * P11 * C_sca/V * dlnr.',
                'reference_wavelength_um': float(wl),
                'nominal_sphericity': Dict1.get('sph'), 'ensemble_sphericity_psi': Dict1.get('psi_e'),
                'alpha_A_over_pi_rv2': Dict1.get('alpha'), 'size_descriptor': Dict1.get('size_descriptor'),
                'dlnr': Dict1.get('dlnr'), 'n_radii': int(len(Radius_v)),
                'note': 'Built as Qe*A/V (corrected 2026-10; earlier kernels used Qe*pi*r_v^2/V and were low by psi).'}
        import json
        with open(f'{folder_name}/kernel_info.json', 'w') as fj:
            json.dump(info, fj, indent=4)

        with open(f'{folder_name}/kernels_299_00.txt','w')as f:
        
            f.write(f'{ "%s %s"%("{:.6E}".format(min(Radius_v)),"{:.6E}".format(max(Radius_v)))} 2.98600 \n') 
            f.write(f'{-len(Radius_v)}   number of grid radii\n') 
            f.write(f'{"%s %s"%("{:.8}".format(min(RealRi)),"{:.8}".format(max(RealRi)))}  range of real refr. index\n')
            f.write(f'{"%s %s"%("{:.7E}".format(-min(Imag1)), "{:.7E}".format(-max(Imag1)))}  range of imag refr. index\n') #range of imaginary refractive index
            f.write(f'{"%s %s"%(len(RealRi),-len(Imag1))}   number of nodes for real and imag parts of refr. index')


            for i in range (len(RealRi)):           #loop over real RI
                for j in range(len(Imag1)):          #loop over imag RI
                    f.write("\n")
                    f.write(f' 0 2.98600 element, ratio \n')
                    f.write(f'{"%s %s %s" %(wl,"{:.7}".format(RealRi[i]),"{:.7E}".format(-Imag1[j]))}   wavel, rreal, rimag\n')

                    f.write("EXTINCTION (1/um):\n")
                    for k in range (len(b_ext[0][1])): #loop over size parameters or dia 
                        f.write("{:.7E}".format(float(b_ext[i][j][k]))+'\t') 
                    f.write("\n")

                    f.write("ABSORPTION (1/um):\n")
                    for k in range (len(b_abs[0][0])): #loop over size parameters or di
                        f.write("{:.7E}".format(float(b_abs[i][j][k]))+'\t')
                print(f"Done{i}")
            f.write('\n') 
            f.write('Shape distribution:\n')

            f.write("1 nr \n")
            f.write("#       r    rd \n")
            f.write("1      2.98600      0.100000E+01")




    elif a in [11, 12, 22, 33, 34, 43, 44]:
        if a == 11:
            # Reshape P11 into a 4D array based on parameter lengths
            F = np.array(P11).reshape(len(RealRi), len(Imag1), len(Radius_v), len(Angles))
        
        # For Mueller matrix elements that require scaling
        if a in [12, 22, 33, 44]:
            # Multiply P11 by a scaling factor from Dict1 and reshape
            F = (np.array(P11) * Dict1[f'P{a}']).reshape(len(RealRi), len(Imag1), len(Radius_v), len(Angles))

        # GRASP element 34 = -P43 of TAMUdust2020 (TAMU provides P43/P11); always written as kernels_299_34.txt
        if a in [34, 43]:
            F = -(np.array(P11) * Dict1['P43']).reshape(len(RealRi), len(Imag1), len(Radius_v), len(Angles))
            a = 34

        # df= pd.read_csv("/home/gregmi/TAMU_project/DATABASE.csv")
        # Imag1 = df['Imaginary Refractive Index:'][:31].values #Read imaginary refrac 
        
        with open(f'{folder_name}/kernels_299_{a}.txt', 'w') as f:
            print(f'kernels_299_{a}.txt')
            f.write(f'{"%s %s"%("{:.6}".format(min(Radius_v)),"{:.6}".format(max(Radius_v)))} \t 2.98600\n') 
            f.write(f'{-len(Radius_v)}   number of grid radii\n') 
            f.write(f'\t {len(Angles)}  number of scattering angles\n')
                        
            for ang in range(len(Angles)):
                f.write(f'{("{:.3f}".format(Angles[ang]))}\t')
                if (ang%10==9):
                    f.write(f'\n')
            f.write(f'\n')
            f.write(f'{"%s %s"%("{:.8}".format(min(RealRi)),"{:.8}".format(max(RealRi)))}  range of real refr. index\n')
            f.write(f'{"%s %s"%("{:.7E}".format(-min(Imag1)), "{:.7E}".format(-max(Imag1)))}  range of imag refr. index\n') #range of imaginary refractive index
            f.write(f'{"%s %s"%(len(RealRi),-len(Imag1))}   number of nodes for real and imag parts of refr. index \n')


            for i in range (len(RealRi)):
                for j in range(len(Imag1)):

                    #1o0p over real RI
                    #loop over imag RI

                    f.write(f' {a}    2.98600  element, ratio \n')
                    f.write(f'{"%s %s %s" %(wl,"{:.7}".format(RealRi[i]),"{:.7E}".format(-Imag1[j]))}   wavel, rreal, rimag\n')

                    for k in range (273): #loop over size parameters or diameters
                        for ang in range(len(Angles)):
                            P11_sca = F[i][j][k][ang]*b_sca[i][j][k]
                            # f.write(Dec_7(P11_sca)+'\t')
                            f.write(f'{"%s"%("{:.7E}".format(P11_sca))}\t')
                        f.write("\n")
                print(f"Done{i}")
            f.write('\n') 
            f.write('Shape distribution:\n')

            f.write("1 nr \n")
            f.write("#       r    rd \n")
            f.write("1     2.98600      0.100000E+01")
    else:
        print(f'Phase_Function_299: unknown element {a!r}')

        return
    




    # Phase_Function_299(P11,11,folder_name,AngVal,Radius_v,RealRi)
    # Phase_Function_299(np.array(P11)*P12,12,folder_name,AngVal,Radius_v,RealRi)
    # Phase_Function_299(np.array(P11)*P22,22,folder_name,AngVal,Radius_v,RealRi)
    # Phase_Function_299(np.array(P11)*P33,33,folder_name,AngVal,Radius_v,RealRi)
    # Phase_Function_299(-np.array(P11)*P43,34,folder_name,AngVal,Radius_v,RealRi)
    # Phase_Function_299(np.array(P11)*P44,44,folder_name,AngVal,Radius_v,RealRi)



