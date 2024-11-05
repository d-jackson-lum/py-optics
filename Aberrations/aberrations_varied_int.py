# %%
# %matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
from lumicks.pyoptics import trapping as trp
from define_field import Initial_Field, Choose_Aberration

#%%
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
num_pts = 16
half_extent = bead_diameter
s = np.zeros(num_pts * 2 - 1)
_s = np.linspace(0, 3*half_extent, num_pts)
s[0:num_pts] = -_s[::-1]
s[num_pts:] = _s[1:]
x = s
y = s
z = 219.8e-9  # [m]
bead_center = (0,0,0)  # [m]

#Initial definitions of the field------------------------
#E0, w0 = Initial_Field(filling_factor, focal_length[m], NA, n_medium, n_bfp, P_max[W], P_percent[%])
E0, w0 = Initial_Field(0.9, focal_length, NA, n_medium, n_bfp, 1.5, 100)  

#sampling------------
bfp_sampling_n = 16

#%%
#Functioning defining an aberrated gaussian beam
#n and m are parameters defining zernike polynomials, which
#describe common aberrations   ## m = n, n-2, ..., -n
#Should include as many m values as n values. Code automatically
#defines m=n for any values of m that do not have an m value
#Input the desired aberration pararmeters of the form [n1, n2, ...., ni]
ns = [3]
ms = [3]
millis = [72e-3] #number of millilambdas
W = Choose_Aberration(ns,ms,millis)

def gaussian_beam_aberr(_, x_bfp, y_bfp, r_bfp, r_max, *args):
    #Generate the gaussian
    Ex = np.exp(-(x_bfp**2  + y_bfp**2) / w0**2) * E0 * (1 + 0j)
    #and apply the aberration (W)
    Ex *= np.exp(2j * np.pi * W(r_bfp/r_max,np.arctan2(y_bfp,x_bfp))) #Millilambda values included in the expression for aberration
    return (Ex, None)


def gaussian_beam(_, x_bfp, y_bfp, *args):
    #Function defining a conventional gaussian beam profile
    #Will use this to compare with the aberrated formula to see
    #how the beam profile changed with aberration
    Ex = np.exp(-(x_bfp**2 + y_bfp**2) / w0**2) * E0 * np.exp(2j * np.pi * 72e-3)
    return (Ex, None)
#%%
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
#%%
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
#%%
##In initial conditions, one axis is set to 0 while the other two are arrays
##This if statement makes it so that we automatically plot the two axes that
##are arrays instead of having to check
if isinstance(y, int) or isinstance(y, float):
    Maximum_Int = max(Max_gauss_XZ,Max_aberr_XZ)
    # X-Z
    plt.figure(figsize=(10, 8))
    plt.pcolor(X * 1e6, Z * 1e6, E_xz_aberr, cmap='plasma', shading='auto', vmin=Minimum_Int, vmax=Maximum_Int)
    plt.xlabel('X [µm]')
    plt.ylabel('Z [µm]')
    plt.title('ABerrations |E| [V/m]')
    plt.colorbar()
    plt.show()
    # X-Z (Gauss)
    plt.figure(figsize=(10, 8))
    plt.pcolor(X * 1e6, Z * 1e6, E_xz_gauss, cmap='plasma', shading='auto', vmin=Minimum_Int, vmax=Maximum_Int)
    plt.xlabel('X [µm]')
    plt.ylabel('Z [µm]')
    plt.title('Gaussian |E| [V/m]')
    plt.colorbar()
    plt.show()
    # X-Z (diff)
    plt.figure(figsize=(10, 8))
    plt.pcolor(X * 1e6, Z * 1e6, E_xz_diff, cmap='inferno', shading='auto')#, vmin=Minimum_Int, vmax=Maximum_Int)
    plt.xlabel('X [µm]')
    plt.ylabel('Z [µm]')
    plt.title('Difference |E| [V/m]')
    plt.colorbar()
    plt.show()
elif isinstance(x, int) or isinstance(x, float):
    Maximum_Int = max(Max_gauss_YZ,Max_aberr_YZ)
    # Y-Z
    plt.figure(figsize=(10, 8))
    plt.pcolor(Y * 1e6, Z * 1e6, E_yz_aberr, cmap='plasma', shading='auto', vmin=Minimum_Int, vmax=Maximum_Int)
    plt.xlabel('Y [µm]')
    plt.ylabel('Z [µm]')
    plt.title('Aberrations |E| [V/m]')
    plt.colorbar()
    plt.show()
    # Y-Z (Gauss)
    plt.figure(figsize=(10, 8))
    plt.pcolor(Y * 1e6, Z * 1e6, E_yz_gauss, cmap='plasma', shading='auto', vmin=Minimum_Int, vmax=Maximum_Int)
    plt.xlabel('Y [µm]')
    plt.ylabel('Z [µm]')
    plt.title('Gaussian |E| [V/m]')
    plt.colorbar()
    plt.show()
    # Y-Z (diff)
    plt.figure(figsize=(10, 8))
    plt.pcolor(Y * 1e6, Z * 1e6, E_yz_diff, cmap='inferno', shading='auto')#, vmin=Minimum_Int, vmax=Maximum_Int)
    plt.xlabel('Y [µm]')
    plt.ylabel('Z [µm]')
    plt.title('Difference |E| [V/m]')
    plt.colorbar()
    plt.show()
elif isinstance(z, int) or isinstance(z, float):
    Maximum_Int = max(Max_gauss_XY,Max_aberr_XY)
    # X-Y
    plt.figure(figsize=(10, 8))
    plt.pcolor(X * 1e6, Y * 1e6, E_xy_aberr, cmap='plasma', shading='auto', vmin=Minimum_Int, vmax=Maximum_Int)
    plt.xlabel('X [µm]')
    plt.ylabel('Y [µm]')
    plt.title('Aberrations |E| [V/m]')
    plt.colorbar()
    plt.show()
    # X-Y (Gauss)
    plt.figure(figsize=(10, 8))
    plt.pcolor(X * 1e6, Y * 1e6, E_xy_gauss, cmap='plasma', shading='auto', vmin=Minimum_Int, vmax=Maximum_Int)
    plt.xlabel('X [µm]')
    plt.ylabel('Y [µm]')
    plt.title('Gaussian |E| [V/m]')
    plt.colorbar()
    plt.show()
    # X-Y
    plt.figure(figsize=(10, 8))
    plt.pcolor(X * 1e6, Y * 1e6, E_xy_diff, cmap='inferno', shading='auto')#, vmin=Minimum_Int, vmax=Maximum_Int)
    plt.xlabel('X [µm]')
    plt.ylabel('Y [µm]')
    plt.title('Difference |E| [V/m]')
    plt.colorbar()
    #plt.savefig('16_10_4.png')
    plt.show()