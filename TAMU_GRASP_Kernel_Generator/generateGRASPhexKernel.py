

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
from GRASPhexKernel import RunTAMUdust2020,Phase_Function_299




df= pd.read_csv("/home/gregmi/TAMU_project/DATABASE.csv") # Reading the Database file from TAMU 

ys = df['Size Parameter:']
xs = np.arange(0,169,1)[:len(ys) ]

#Lognormal fitting of the original data
Pp = np.polyfit(xs, np.log(ys), 1)
fit = np.exp(Pp[1]) * np.exp(Pp[0] * xs)


#Calculating the ratio of normal distribution 
for i in range(len(fit)-1):
    ratio = fit[i]/fit[i+1]
#     print(ratio)

#Adding more values in the fit to cover the both limits of the data
a = fit[0]
add_value=[]
for i in range(31):
    new = a *ratio
    add_value.append(new)
    a = new

b = fit[-1]
back_value=[]
for i in range(73):
    new = b /ratio
    back_value.append(new)
    b = new

data1 = np.concatenate((np.array(add_value[::-1]), np.array(fit)), axis=0)
data2 = np.concatenate((data1, np.array(back_value)), axis=0)



fig, ax = plt.subplots(1,2, figsize=(15, 7))
ax[0].plot(xs, ys, '.', label='Original')
ax[0].plot(xs, fit, '*', label = "Fitted")
ax[0].set_yscale('log')
ax[0].legend(fontsize=15)
ax[0].set_ylabel("Size Parameter", fontsize=15 )
ax[0].set_xlabel("Number of points",fontsize=15 )

diff = (ys-fit) #Difference between the fit and the original 

ax[1].plot(diff, marker ='o')
ax[1].set_ylabel("Error (Original-fitted)",fontsize=15 )
ax[1].set_xlabel("Number of points",fontsize=15 )

print(np.min(fit),np.max(fit), np.min(ys),np.max(ys) )

ys = df['Size Parameter:']
xs = np.arange(0,169,1)[:len(ys)]+50

fig =plt.figure(figsize=(10, 7))
plt.plot(xs, ys, '.', label='Original')
plt.plot(data2, label = "Fitted")
plt.yscale('log')
print(len(data2))
print(np.min(data2),np.max(data2)), np.min(df['Size Parameter:'])
plt.legend(fontsize=15)
plt.ylabel("Size Parameter",fontsize=15 )
plt.xlabel("Number of points",fontsize=15 )

plt.title(f"Number of points: {len(data2)}, Min:{round(np.min(data2),7)} , Max:{round(np.max(data2),7)} ",fontsize=15)



fig, ax = plt.subplots(1,2, figsize=(15,7))


ys = df['Real Refractive Index:'][:8]
xs=np.arange(0,len(ys),1)[:8]
ax[0].plot(xs, ys, '.', label='original')

ax[0].set_yscale('log')
ax[0].set_ylabel("Real Refractive Index",fontsize=15)
ax[0].set_xlabel("Number of points",fontsize=15)
#linear fit
linear = stats.linregress((xs,ys))
RI_fit = linear.intercept+0.012+(linear.slope-0.005)*xs  #values are adjusted  to get the closer fit
ax[0].plot(xs,RI_fit, 'k',marker = "*", label='linear fit')

ax[0].set_yscale('log')
ax[0].legend(fontsize=15)
 

diff= abs(ys-RI_fit)
ax[1].plot(100*diff/ys)
ax[1].set_ylabel("Error &",fontsize=15)
ax[1].set_xlabel("Number of points",fontsize=15)
#fixing the bounds
RI= RI_fit
TAMU_min=1.3701
constant= TAMU_min - RI[0]
new_RI=RI+constant
print(new_RI)
ri=pd.DataFrame()
ri['RI']= new_RI



fig =plt.figure(figsize=(10, 7))
plt.plot(df['Imaginary Refractive Index:'],marker="o")
plt.yscale('log')
plt.ylabel("Imaginary Refractive Index",fontsize=15 )
plt.xlabel("Number of points",fontsize=15 )
plt.title("TAMUDUST2020 Database Imaginary Refractive Index")




degree_of_sphericity = 0.74
size_descriptor = 'VER'
nml_file_path = '/home/gregmi/TAMU_project/TAMUdust2020-master/examples/TAMUdust2020create_exp1.nml'
AngVal = np.linspace(0,180,181)
RealRi =  new_RI
Imag1 = df['Imaginary Refractive Index:'][:31].values
Allangle = False
params_path="/home/gregmi/TAMU_project/TAMUdust2020-master/examples/params/TAMUdust2020_RefractiveIndex_exp1"

output_path="/home/gregmi/TAMU_project/TAMUdust2020-master/examples/output/TAMUdust2020create_exp1"

SizeParam = data2



# wavelengths = [0.695, 0.704, 0.713, 0.722, 0.731, 0.74, 0.749, 0.758, 0.767, 0.776, 0.785]

# for wl in wavelengths:
#     folder_name = f"{wl:.3f}"
#     os.makedirs(folder_name, exist_ok=True)
DOS = np.linspace(0.695,0.785,11)

for degree_of_sphericity in DOS:

    Dict1 = RunTAMUdust2020(df, nml_file_path,SizeParam, RealRi,Imag1,AngVal,Allangle,size_descriptor, degree_of_sphericity,  params_path,output_path)


    folder_name = f'/home/gregmi/git/GSFC-GRASP-Python-Interface/TAMU_GRASP_Kernel_Generator/TAMUSimulationDiffSph/hex_kernels/{degree_of_sphericity}'
    Phase_Function_299(Dict1,5,folder_name)
    Phase_Function_299(Dict1,00,folder_name)
    Phase_Function_299(Dict1,11,folder_name)
    Phase_Function_299(Dict1,12,folder_name)
    Phase_Function_299(Dict1,22,folder_name)
    Phase_Function_299(Dict1,33,folder_name)
    Phase_Function_299(Dict1,43,folder_name)
    Phase_Function_299(Dict1,44,folder_name)
