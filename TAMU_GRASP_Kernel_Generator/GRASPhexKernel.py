

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

    if size_descriptor == 'VER' or size_descriptor == None : #Volume Equivant Radius

        folder_name = f"Hex_VER_{degree_of_sphericity}"  #Kernels with volume equivalent radius and using volume and area of the spheres
        # calcuating radius from the volume
        Radius_v = np.cbrt(3*Volume1[1,1,:]/(np.pi*4))
        Radius = Radius_v
        #The GRASP Kernels inputs bext *lnr
        dlnr = np.log(Radius_v[1]/Radius_v[0])
        #Uncomment this if you want to use the volume and area of sphere
        b_ext = Qext1*3*dlnr/(4*Radius_v)  #Qext = (4/3)*Qext *(dlnr/r)
        b_abs = Qabs1*3*dlnr/(4*Radius_v)


    if size_descriptor == 'ER': #Effective radius
        folder_name = "Eff_sph"  #Kernels with effiective radius and using volume and area of the spheres
        Radius_v =3*Volume1[1,1,:]/(4*Area1[1,1,:])
        Radius = Radius_v
        dlnr = np.log(Radius_v[1]/Radius_v[0])
        b_ext = Qext1*3*dlnr/(4*Radius_v)
        b_abs = Qabs1*3*dlnr/(4*Radius_v)

    if size_descriptor == 'Allsaitoang' or size_descriptor == None : #Volume Equivant Radius
        folder_name = "Saito_GEOS_VarAngRes"  #Kernels with volume equivalent radius and using volume and area of the spheres for GEOS model for variable angular grids
        ### calcuating radius from the volume
        Radius_v = np.cbrt(3*Volume1[1,1,:]/(np.pi*4))
        Radius = Radius_v
        #The GARSP Kernels inputs bext *lnr
        dlnr = np.log(Radius_v[1]/Radius_v[0])
        #Uncomment this if you want to use the volume and area of sphere
        b_ext = Qext1*3*dlnr/(4*Radius_v)
        b_abs = Qabs1*3*dlnr/(4*Radius_v)
        # b_sca = b_ext - b_abs


    if size_descriptor == 'TAMUratioA2V_VER':
        folder_name = "Ver_TAMU" #kernels with volume equivalent radius and using volume and area definition of TAMU
        Radius_v = np.cbrt(3*Volume1[1,1,:]/(np.pi*4))
        dlnr = np.log(Radius_v[1]/Radius_v[0])
        b_ext = (Qext1 *Area1 *dlnr)/Volume1
        b_abs = (Qabs1 *Area1 *dlnr)/Volume1

    
    if size_descriptor == 'TAMUratioA2V_ER':
        folder_name = "Eff_TAMU" #kernels with effective radius and using volume and area definition of TAMU
        Radius_v =3*Volume1[1,1,:]/(4*Area1[1,1,:])
        Radius = Radius_v
        dlnr = np.log(Radius_v[1]/Radius_v[0])
        b_ext = (Qext1 *Area1 *dlnr)/(4*Volume1)
        b_abs = (Qabs1 *Area1 *dlnr)/(4*Volume1)


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



    with open(f'/home/gregmi/git/GSFC-GRASP-Python-Interface/TAMU_GRASP_Kernel_Generator/TAMUSimulationDiffSph/TAMUsim{degree_of_sphericity}.pickle', 'wb') as f:
        pickle.dump(dict, f)
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

    if a == 5:

        b_ext = Dict1['Qext']
        b_abs = b_ext-b_sca

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




    if a in [11, 12, 22, 33, 43, 44]:
        if a == 11:
            # Reshape P11 into a 4D array based on parameter lengths
            F = np.array(P11).reshape(len(RealRi), len(Imag1), len(Radius_v), len(Angles))
        
        # For Mueller matrix elements that require scaling
        if a in [12, 22, 33, 43, 44]:
            # Multiply P11 by a scaling factor from Dict1 and reshape
            F = (np.array(P11) * Dict1[f'P{a}']).reshape(len(RealRi), len(Imag1), len(Radius_v), len(Angles))

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
        print('Error in Line 419')

        return
    




    # Phase_Function_299(P11,11,folder_name,AngVal,Radius_v,RealRi)
    # Phase_Function_299(np.array(P11)*P12,12,folder_name,AngVal,Radius_v,RealRi)
    # Phase_Function_299(np.array(P11)*P22,22,folder_name,AngVal,Radius_v,RealRi)
    # Phase_Function_299(np.array(P11)*P33,33,folder_name,AngVal,Radius_v,RealRi)
    # Phase_Function_299(-np.array(P11)*P43,34,folder_name,AngVal,Radius_v,RealRi)
    # Phase_Function_299(np.array(P11)*P44,44,folder_name,AngVal,Radius_v,RealRi)



