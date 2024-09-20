# %%
# %matplotlib inline
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import rc
from lumicks.pyoptics import trapping as trp
import numpy as np
import sympy as sp
from sympy import symbols, Sum, factorial, oo, IndexedBase, Function, Integer, lambdify
from scipy.constants import speed_of_light as C, epsilon_0 as epsilon_0, mu_0 as mu_0
#from lumicks.pyoptics.farfield_data import FarfieldData
from lumicks.pyoptics.objective import BackFocalPlaneCoordinates

# Properties of the bead, laser, and medium---------------
bead_diameter = 1.0e-6  # [m]
lambda_vac = 1.064e-6   # [m]
n_bead =  1.33#1.57 #PS  1.43 #Silica      # [-]
n_medium = 1.33         # [-]
bead = trp.Bead(bead_diameter=bead_diameter, n_bead=n_bead, n_medium=n_medium, lambda_vac=lambda_vac)

# Properties of the objective-----------------------------
focal_length = 4.43e-3  # [m]
NA = 1.2                # [-]
n_bfp = 1.0             # [-]
objective = trp.Objective(NA=NA, focal_length=focal_length, n_bfp=n_bfp, n_medium=n_medium)

##Develop the in plane coordinates the calculations will be 
#performed for. Dependig on which axes are chosen as arrays,
#you will plot a different plane
#np.linspace can return asymmetric values for negative and 
#positive sides of an axis. The code below ensures that the
#negative values of x are the same as the positive values 
#of x, except for the sign. This helps to reduce the number 
#of points to calculate by taking advantage of symmetry.
num_pts = 36
half_extent = bead_diameter
s = np.zeros(num_pts * 2 - 1)
_s = np.linspace(0, 3*half_extent, num_pts)
s[0:num_pts] = -_s[::-1]
s[num_pts:] = _s[1:]
x = s
y = s
z = 0
bead_center = (0,0,0)  # [m]

#Initial definitions of the field------------------------
filling_factor = 0.9                                # [-]
w0 = filling_factor * focal_length * NA / n_medium  # [m]
Pmax = 1.5                                          # [W]
power_percentage = 100                              # [%]
P = Pmax * power_percentage / 100                   # [W]
I0 = 2 * P / (np.pi * w0**2)                        # [W/m^2]
E0 = (I0 * 2/(epsilon_0 * C * n_bfp))**0.5          # [V/m]

#sampling------------
bfp_sampling_n = 9

def Choose_Aberration(ns,ms):
    #Outputs a symbolic formula to be used in creating aberrations
    #in a gaussian field
    #Uses a generator function to generate Zernike polynomials
    #Zernike polynomials describe aberrations to EM fields
    #Takes list of n and m values as input, where each pair of
    #n and m describe a different aberration
    #Inputting multiple n and m values at once will lead to a
    #combined aberration
    ##single aberration Ex:  n = 1, m = -1
    ##combined aberration Ex: ns = [1, 4],  ms = [-1, 0]

    #Initiate the polynomial
    Z_nm = 0
    #define symbols used when generating symbolic formula of aberrations
    r, s, t = symbols('r s t')
    #index used to sort through m asscoiated with n
    idx = 0
    #Cycle through all n and their associate m values
    for n in ns:     
        #Error prevention if statement. Ensures that if you forgot to
        #include enough m values in ms, then the missing values will
        #just be set equal to n
        if len(ms)>idx:
            m = ms[idx]
        else:
            m = n
            print('Fewer m values than n values. Missing m value set equal to n.')
        #Part of normalization factor. 1 if m is 0, 0 if m is not 0
        if m==0:
            delta_m0 = 1
        else:
            delta_m0 = 0
        #Calucate the normalization factor
        #In final output the square root is in decimal form
        N_nm = sp.sqrt((2*(n+1))/(1+delta_m0))
        #Caluate the polynomial. Output in the form of "2 * r**2 - 1"
        R_nm = Sum((r**(n-2*s))*((-1)**s * factorial(n-s))/(factorial(s) * factorial((1/2)*(n+abs(m)) - s) * factorial((1/2)*(n-abs(m)) - s)),(s,0,Integer((n-abs(m))/2))).doit()
        #Determine if azimuthal angle will be evaluated by sin or cos function
        if m > 0:
            Angle = sp.sin(m * t)
        elif m < 0:
            Angle = sp.cos(m * t)
        else:
            Angle = 1        
        #Combine all componennts and store in Z
        #If multiple n values were input they will be 
        #concatenated into the one formula before being returned
        Z_nm += N_nm*R_nm*Angle
        #Increment index for reading m values
        idx+=1
    print(Z_nm)
    return Z_nm

def gaussian_beam_aberr(_, x_bfp, y_bfp, r_bfp, r_max, Degrees, *args):
    #Functioning defining an aberrated gaussian beam
    #n and m are parameters defining zernike polynomials, which
    #describe common aberrations   ## m = n, n-2, ..., -n
    #Should include as many m values as n values. Code automatically
    #defines m=n for any values of m that do not have an m value
    #symbols used in 'Choose_Aberration' function. Needed for lambdify function
    r, t = symbols('r t')
    #Input the desired aberration pararmeters of the form [n1, n2, ...., ni]
    ns = [4]
    ms = [0]
    #Have the formula output from 'Choose_Aberration' converted into a
    #usable function instead of a string
    W = lambdify([r, t], Choose_Aberration(ns,ms))
    #Generate the gaussian and apply the aberration (W)
    Ex = np.exp(-((x_bfp - 0*1e-3)**2  + y_bfp**2) / w0**2) * E0 * (1 + 0j)
    Ex *= np.exp(2j * np.pi * W(r_bfp/r_max,np.arctan2(y_bfp,x_bfp)) * 72e-3) #number of millilambdas
    return (Ex, None)

def gaussian_beam(_, x_bfp, y_bfp, *args):
    #Function defining a conventional gaussian beam profile
    #Will use this to compare with the aberrated formula to see
    #how the beam profile changed with aberration
    Ex = np.exp(-(x_bfp**2 + y_bfp**2) / w0**2) * E0 * np.exp(2j * np.pi * 72e-3)
    return (Ex, None)

##Input parameters used to calculate the electric fields
#Abberrated field
Ex, Ey, Ez, X, Y, Z = trp.fields_focus(
        gaussian_beam_aberr, #gaussian_beam, #
        objective,
        bead,
        x=x,
        y=y,
        z=z,
        bead_center=bead_center,
        bfp_sampling_n=bfp_sampling_n,
        num_orders=None,
        return_grid=True,
        total_field=True,
        magnetic_field=False,
        verbose=False,
        grid=True,
        )

#Normal Gaussian field
Ex_gauss, Ey_gauss, Ez_gauss = trp.fields_focus(
        gaussian_beam, #
        objective,
        bead,
        x=x,
        y=y,
        z=z,
        bead_center=bead_center,
        bfp_sampling_n=bfp_sampling_n,
        num_orders=None,
        return_grid=False,
        total_field=True,
        magnetic_field=False,
        verbose=False,
        grid=True,
        )

##Determine the field for each plane
E_xz_gauss = np.sqrt(np.abs(Ex_gauss)**2 + np.abs(Ez_gauss)**2)
E_xz_aberr = np.sqrt(np.abs(Ex)**2 + np.abs(Ez)**2)
E_yz_gauss = np.sqrt(np.abs(Ey_gauss)**2 + np.abs(Ez_gauss)**2)
E_yz_aberr = np.sqrt(np.abs(Ey)**2 + np.abs(Ez)**2)
E_xy_gauss = np.sqrt(np.abs(Ex_gauss)**2 + np.abs(Ey_gauss)**2)
E_xy_aberr = np.sqrt(np.abs(Ex)**2 + np.abs(Ey)**2)

##Determine the maximum values of the intensity for all fields 
#so that the heatmap in the plots can be set to the same max 
#value. Difference plot does not receive this value
Max_gauss_XZ = np.max(E_xz_gauss)
Max_aberr_XZ = np.max(E_xz_aberr)
Max_gauss_XY = np.max(E_xy_gauss)
Max_aberr_XY = np.max(E_xy_gauss)
Max_gauss_YZ = np.max(E_yz_gauss)
Max_aberr_YZ = np.max(E_yz_aberr)

#Set the minimum value of the colorbars for the aberration
#and normal gaussian plot. Difference plot does not receive
#this value
Minimum_Int = 0

##Determine difference between gaussian and aberrated fields
##When a=1 subtract the aberration from the gaussian, and 
##subtract the gaussian from the aberration when a=-1
a=-1
E_xz_diff = a*(E_xz_gauss - E_xz_aberr)
E_yz_diff = a*(E_yz_gauss - E_yz_aberr)
E_xy_diff = a*(E_xy_gauss - E_xy_aberr)

##In initial conditions, one axis is set to 0 while the other two are arrays
##This if statement makes it so that we automatically plot the two axes that
##are arrays instead of having to check
if isinstance(y, int) or isinstance(y, float):
    Maximum_Int = max(Max_gauss_XZ,Max_aberr_XZ)
    # X-Z
    print('With Aberrations')
    plt.figure(figsize=(10, 8))
    plt.pcolor(X * 1e6, Z * 1e6, E_xz_aberr, cmap='plasma', shading='auto', vmin=Minimum_Int, vmax=Maximum_Int)
    plt.xlabel('X [µm]')
    plt.ylabel('Z [µm]')
    plt.title('|E| [V/m]')
    plt.colorbar()
    plt.show()
    # X-Z (Gauss)
    print('Normal Gaussian')
    plt.figure(figsize=(10, 8))
    plt.pcolor(X * 1e6, Z * 1e6, E_xz_gauss, cmap='plasma', shading='auto', vmin=Minimum_Int, vmax=Maximum_Int)
    plt.xlabel('X [µm]')
    plt.ylabel('Z [µm]')
    plt.title('|E| [V/m]')
    plt.colorbar()
    plt.show()
    # X-Z (diff)
    print('Difference')
    plt.figure(figsize=(10, 8))
    plt.pcolor(X * 1e6, Z * 1e6, E_xz_diff, cmap='inferno', shading='auto')#, vmin=Minimum_Int, vmax=Maximum_Int)
    plt.xlabel('X [µm]')
    plt.ylabel('Z [µm]')
    plt.title('|E| [V/m]')
    plt.colorbar()
    plt.show()
elif isinstance(x, int) or isinstance(x, float):
    Maximum_Int = max(Max_gauss_YZ,Max_aberr_YZ)
    # Y-Z
    print('With Aberrations')
    plt.figure(figsize=(10, 8))
    plt.pcolor(Y * 1e6, Z * 1e6, E_yz_aberr, cmap='plasma', shading='auto', vmin=Minimum_Int, vmax=Maximum_Int)
    plt.xlabel('Y [µm]')
    plt.ylabel('Z [µm]')
    plt.title('|E| [V/m]')
    plt.colorbar()
    plt.show()
    # Y-Z (Gauss)
    print('Normal Gaussian')
    plt.figure(figsize=(10, 8))
    plt.pcolor(Y * 1e6, Z * 1e6, E_yz_gauss, cmap='plasma', shading='auto', vmin=Minimum_Int, vmax=Maximum_Int)
    plt.xlabel('Y [µm]')
    plt.ylabel('Z [µm]')
    plt.title('|E| [V/m]')
    plt.colorbar()
    plt.show()
    # Y-Z (diff)
    print('Difference')
    plt.figure(figsize=(10, 8))
    plt.pcolor(Y * 1e6, Z * 1e6, E_yz_diff, cmap='inferno', shading='auto')#, vmin=Minimum_Int, vmax=Maximum_Int)
    plt.xlabel('Y [µm]')
    plt.ylabel('Z [µm]')
    plt.title('|E| [V/m]')
    plt.colorbar()
    plt.show()
elif isinstance(z, int) or isinstance(z, float):
    Maximum_Int = max(Max_gauss_XY,Max_aberr_XY)
    # X-Y
    print('With Aberrations')
    plt.figure(figsize=(10, 8))
    plt.pcolor(X * 1e6, Y * 1e6, E_xy_aberr, cmap='plasma', shading='auto', vmin=Minimum_Int, vmax=Maximum_Int)
    plt.xlabel('X [µm]')
    plt.ylabel('Y [µm]')
    plt.title('|E| [V/m]')
    plt.colorbar()
    plt.show()
    # X-Y (Gauss)
    print('Normal Gaussian')
    plt.figure(figsize=(10, 8))
    plt.pcolor(X * 1e6, Y * 1e6, E_xy_gauss, cmap='plasma', shading='auto', vmin=Minimum_Int, vmax=Maximum_Int)
    plt.xlabel('X [µm]')
    plt.ylabel('Y [µm]')
    plt.title('|E| [V/m]')
    plt.colorbar()
    plt.show(
    # X-Y
    print('Difference')
    plt.figure(figsize=(10, 8))
    plt.pcolor(X * 1e6, Y * 1e6, E_xy_diff, cmap='inferno', shading='auto')#, vmin=Minimum_Int, vmax=Maximum_Int)
    plt.xlabel('X [µm]')
    plt.ylabel('Y [µm]')
    plt.title('|E| [V/m]')
    plt.colorbar()
    #plt.savefig('16_10_4.png')
    plt.show()