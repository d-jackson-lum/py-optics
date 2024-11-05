# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Calculating forces on a trapped bead

# %%
# %matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import lumicks.pyoptics.trapping as trp
from scipy.interpolate import interp1d
from tqdm.auto import tqdm
from define_field import Initial_Field, Choose_Aberration

# %% [markdown]
# ## Definition of coordinate system
# The optical axis (direction of the light to travel into) is the $+z$ axis. For an aberration-free system, the focus of the laser beam ends up at $(x, y, z) = (0, 0, 0)$ See also below:
#
# <figure>
#     <img src="images/axes.png" width=400>
#     <figcaption>Fig. 1: Definition of the coordinate system</figcaption>
# </figure>
#
# ## Properties of the bead, the medium and the laser
# The bead is described by a refractive index $n_{bead}$, a diameter $D$ and a location in space $(x_b, y_b, z_b)$, the latter two in meters. In the code, the diameter is given by `bead_diameter`. The refractive index is given by `n_bead` and the location is passed to the code as a tuple `bead_center` containing three floating point numbers. These numbers represent the $x$-, $y$- and $z$-location of the bead, respectively, in meters. The wavelength of the trapping light is given in meters as well, by the parameter `lambda_vac`. The wavelength is given as it occurs in vacuum ('air'), not in the medium. The refractive index of the medium $n_{medium}$ is given by the parameter `n_medium`.

# %%
# instantiate a Bead object
bead_diameter = 4e-6  # [m]
lambda_vac = 1064e-9    # [m]
n_bead =  1.57          # [-]
n_medium = 1.33         # [-]
bead = trp.Bead(bead_diameter=bead_diameter, n_bead=n_bead, n_medium=n_medium, lambda_vac=lambda_vac)
# Tell use how many scattering orders are used according to the formula in literature:
print(f'Number of scattering orders used by default: {bead.number_of_orders}')                                                                       

# %%
# objective properties, for water immersion
NA = 1.2                # [-]
focal_length = 4.43e-3  # [m]
n_bfp = 1.0             # [-] Other side of the water immersion objective is air
# Instantiate an Objective. Note that n_medium has to be defined here as well
objective = trp.Objective(NA=NA, focal_length=focal_length, n_bfp=n_bfp, n_medium=n_medium)

# %%
# approximation of the focus, higher is better and slower (scales with N**2)
# best to check the correct range by plotting fields!
# Note that the number here is relatively low for demonstration purposes
# In general, a convergence check is required to verify the computed results
bfp_sampling_n = 16

# 100% is 1.75W into a single trapping beam before the objective, at trap split = 50%
Pmax = 1.75  # [W]
power_percentage = 25  # [%]

# %%
#E0, w0 = Initial_Field(filling_factor, focal_length[m], NA, n_medium, n_bfp, P_max[W], P_percent[%])
E0, w0 = Initial_Field(0.9, focal_length, NA, n_medium, n_bfp, 1.5, 100) 



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

# %% [markdown]
# ## Forces in the $z$ direction - find the equilibrium
# Set the range in z to calculate the forces at. We expect the force on the bead in the z-direction to be zero for the interval of $[0, \inf)$
# Start with a range of $0\ldots 1 \mu m$

# %%
# Resolution of force calucaltion
resolution_z = 21
start_z = 10e-9
finish_z = 1000e-9
resolution_xy = 21
start_xy = 50e-9
finish_xy = 1.0e-6
z = np.linspace(start_z, finish_z, resolution_z)  
Fz = np.empty(z.shape)
kz = np.empty(z.shape)
x = np.linspace(start_xy, finish_xy, resolution_xy)
Fx = np.empty(x.shape)
kx = np.empty(x.shape)
y = np.linspace(start_xy, finish_xy, resolution_xy)
Fy = np.empty(y.shape)
ky = np.empty(y.shape)

# %%
for idx, zz in enumerate(tqdm(z)):
    F = trp.forces_focus(gaussian_beam_aberr, objective, bead, bfp_sampling_n=bfp_sampling_n, bead_center=(0, 0, zz), 
                                  num_orders=None, integration_orders=None, verbose=False)
    Fz[idx] = F[2]

# %%
plt.figure(figsize=(8, 6))
plt.plot(z * 1e9, Fz * 1e12)
plt.xlabel('$z$ [nm]')
plt.ylabel('$F$ [pN]')
plt.title(f'{bead_diameter * 1e6} µm bead, $F_z$ at $x_b = y_b = 0$')
plt.show()

# %%
# linearly interpolate. For better accuracy, you might consider scipy.optimize.brentq
z_eval = interp1d(Fz, z)(0)
print(f'Force in z zero near z = {(z_eval*1e9):.1f} nm')

# %% Force when dispalced in X and Y
# ## Forces in $x$ and $y$ directions
# Calculate the forces in the $x$ and $y$ direction at the location where the force in the $z$ direction is (nearly) zero


for idx, xx in enumerate(tqdm(x)):
    F = trp.forces_focus(gaussian_beam_aberr, objective, bead, bfp_sampling_n=bfp_sampling_n, 
                                  bead_center=(xx, 0, z_eval), num_orders=None, integration_orders=None)
    Fx[idx] = F[0]

for idx, yy in enumerate(tqdm(y)):
    F = trp.forces_focus(gaussian_beam_aberr, objective, bead, bfp_sampling_n=bfp_sampling_n, 
                                  bead_center=(0, yy, z_eval), num_orders=None, integration_orders=None)
    Fy[idx] = F[1]

plt.figure(figsize=(8, 6))
plt.plot(x * 1e9,Fx * 1e12, label='x')
plt.plot(y * 1e9,Fy * 1e12, label='y')
plt.xlabel('Displacement [nm]')
plt.ylabel('F [pN]')
plt.legend()
plt.title(f'{bead_diameter * 1e6} µm bead, Magnitude of $F_x$ and $F_y$ at equilibrium z height')
#plt.title(f'{bead_diameter * 1e6} µm bead, $F_x$ at $(x, 0, {z_eval * 1e9:.1f})$ nm, $F_y$ at $(0, y, {z_eval * 1e9:.1f})$ nm')
plt.show()

#%% Stiffness as function of displacement

kx = -Fx/x
ky = -Fy/y
kz = -Fz/z

plt.figure(figsize=(8, 6))
#plt.plot(x * 1e9,kx, label='x')
#plt.plot(y * 1e9,ky, label='y')
plt.semilogy(x * 1e9,kx, label='x')
plt.semilogy(y * 1e9,ky, label='y')
#plt.plot(z * 1e9,kz, label='z')
plt.xlabel('Displacement [nm]')
plt.ylabel('$\kappa$ [pN/nm]')
plt.legend()
plt.title('Stiffness from force and displacement. F = \u2212$\kappa$x') #\u2212 unicode for hyphen minus
plt.show()

#%% Percent change

perx = np.empty(x.shape)
pery = np.empty(y.shape)

for idx, kkx in enumerate(kx):
    perx[idx] = 100*(kx[idx]-kx[0])/kx[0]
    pery[idx] = 100*(ky[idx]-ky[0])/ky[0]
    
plt.figure(figsize=(8, 6))
plt.plot(x * 1e9, perx, label='x')
plt.plot(y * 1e9, pery, label='y')
#plt.plot(z * 1e9,kz, label='z')
plt.xlabel('Displacement [nm]')
plt.ylabel('% Change')
plt.legend()
plt.title('Percent change in the stiffness with displacement')
plt.show()

# %% [markdown]
# 1. *Decrease* the number of spherical harmonics `num_orders` and check the difference between the old and newly calculated forces.
#     1. It may help to plot the absolute values of the Mie scattering coefficients on a logarithmic y axis to decide the initial cutoff.
# 1. *Decrease* the number of plane waves in the back focal plane, and check the difference between the old and newly calculated forces.

# %%
an, bn = bead.ab_coeffs()
plt.figure(figsize=(8, 6))
plt.semilogy(range(1, an.size + 1), np.abs(an), label='$a_n$')
plt.semilogy(range(1, bn.size + 1), np.abs(bn), label='$b_n$')
plt.xlabel('Order')
plt.ylabel('|$a_n$|, $|b_n|$ [-]')
plt.title(f'Magnitude of scattering coefficients for {bead.bead_diameter * 1e6:.2f} µm bead')
plt.legend()
plt.grid()
plt.show()
