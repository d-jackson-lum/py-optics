# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 11:46:41 2024

@author: DanielJackson
"""

#%%
import numpy as np
from scipy.constants import speed_of_light as C, epsilon_0 as epsilon_0
import sympy as sp
from sympy import symbols, Sum, factorial, Integer, lambdify


#%%

def Initial_Field(filling_factor, focal_length, NA, n_medium, n_bfp, P_max, P_percent):
    #Initial definitions of the field------------------------
    w0 = filling_factor * focal_length * NA / n_medium  # [m]                           # [%]
    P = P_max * P_percent / 100                   # [W]
    I0 = 2 * P / (np.pi * w0**2)                        # [W/m^2]
    return (I0 * 2/(epsilon_0 * C * n_bfp))**0.5, w0
    #E0 = (I0 * 2/(epsilon_0 * C * n_bfp))**0.5          # [V/m]
    
 #%%   
def Choose_Aberration(ns,ms,millis):
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
            print('Fewer m values than n values. Missing m value set equal to n as default.')
        #Error prevention if statement. Ensures that if you forgot to
        #include enough millilambda values in millis, then the missing values will
        #just be set equal to 72e-3 as that is the default
        if len(millis)>idx:
            milli = millis[idx]
        else:
            milli = 72e-3
            print('Fewer milli values than n values. Missing milli values set equal to 72e-3 as default')
        
        #Calucate the normalization factor
        #In final output the square root is in decimal form
        #Part of normalization factor. 1 if m is 0, 0 if m is not 0
        if m==0:
            delta_m0 = 1
        else:
            delta_m0 = 0
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
        Z_nm += N_nm*R_nm*Angle*milli
        #Increment index for reading m values
        idx+=1
    print(Z_nm)
    #Have the formula output from 'Choose_Aberration' converted into a
    #usable function instead of a string
    return lambdify([r, t], Z_nm)
