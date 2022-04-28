#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 15:55:05 2021

@author: cwilkinson
"""
import numpy as np
import matplotlib.pyplot as plt
import time
import numpy as np 
import matplotlib.pyplot as plt 
import h5py 
from scipy.interpolate import interp1d

def make_profile(T_int, g, P_th,idx):

    T_int = np.round(T_int,1)
    P_th = np.round(P_th,1)
    
    file = './exorem/inputs/atmospheres/temperature_profiles/temperature_profile_example_ref.dat'
    data = load_init(file)
    P = np.logspace(-3,3,num=200) * (1e5) # en Pa
    P = data[:,0] # from example T_P exorem file
    g = g * np.ones(np.shape(P))
    met = 1
    
    for i in range (0,1) : 
        T, tau_ =  T_profile(P,T_int,met,g)
             
    data[:,1] = T
    file = './exorem/inputs/atmospheres/temperature_profiles/temperature_profile_example'+'-'+str(idx) + '.dat'
    save_profile(file,data)
    
    # plt.plot(data[:,1],data[:,0]*1e-5)
    # plt.yscale('log')
    # plt.gca().invert_yaxis()
    # plt.show()
    
    inter = interp1d(P, T, kind='nearest')
    T_th = inter(P_th)
    return T_th
    
    
def T_profile(P,T_int,met,g) :
    T = [T_0(T_int)]
    c = np.zeros([11,len(P)])
    tau_ = []
    flag = False
    for ii in range(1, len(P)):
        c[:,ii] = coefficients(T[-1])
        if ii > 5 :
            adiabat = (np.log(T[-1])-np.log(T[-2]))/(np.log(P[ii-1])-np.log(P[ii-2]))
            if adiabat > (2/7) :
                flag = True
        if flag :
            T.append(profile_adiabatique (T,P,met,c,g[ii]))
        else :
            T.append(((3/4) * T_int**4*((2/3) + tau(T,P,met,c,g[ii])))**(1/4)) 
        tau_.append(tau(T,P,met,c,g[ii]))
    return T, tau_

def profile_adiabatique (T,P,met,c,g) :
    T = T[-1] * (P[-1]/P[-2])**(2/7)
    #T = T*(2/7)*(((P[-1]-P[0]))/P[0]) + T
    return T
    
def T_0(T_int) :
    T = ((3/4)*T_int**4*(2/3))**(1/4)
    return T

def coefficients(T) :
    c = np.zeros(11) 
    c[0] = -37.50
    c[1] = 0.00105
    c[2] = 3.2610
    c[3] = 0.84315
    c[4] = -2.339   
    if T<=800 :
        c[5] = -14.051
        c[6] = 3.055
        c[7] = 0.024
        c[8] = 1.877
        c[9] = -0.445
        c[10] = 0.8321       
    if T>800 :
        c[5] = 82.241
        c[6] = -55.456
        c[7] = 8.754
        c[8] = 0.7048
        c[9] = -0.0414
        c[10] = 0.8321      
    return c
        
def tau(T,P,met,coeff,g) :
    tau = 0
    for jj in range(0,len(T)-1) :
        c = coeff[:,jj]
        kappa_low = kappa_low_p(T[jj],P[jj],met,c)
        kappa_high = kappa_high_p(T[jj],P[jj],met,c)
        kappa = (kappa_low + kappa_high)#**(-1)
        m = (P[jj+1]-P[jj])/g 
        tau += kappa * m
    
    return tau
    
def kappa_low_p(T,P,met,c) :
    
    kappa = 10**(c[0]*(np.log10(T) - c[1]*np.log10(P)-c[2])**2 \
                 + (c[3]*met + c[4])) # in cm2/g

    return kappa*1e-4/(1e-3) # converting to kg/m**2
    
def kappa_high_p(T,P,met,c) :
    P = P*10 # Converting to dyn/cm-2 from pa
    
    kappa = 10**((c[5] + c[6]*np.log10(T) + c[7]*np.log10(T**2)) \
                 + np.log10(P) * (c[8] + c[9]*np.log10(T)) \
                     + met*c[10]*(0.5 + (1/np.pi)*np.arctan((np.log10(T)-2.5)/0.2))) # in cm2/g
    return kappa*1e-4/(1e-3) # converting to kg/m**2

def load_init(file):
    data = np.loadtxt(file,skiprows=2)
    return data 

def save_profile(file,data):
    
    header = """pressure temperature temparature_adiabatic temperature_uncertainty temperature_uncertainty_b delta_temperature_convection radiosity radiosity_convective is_convective grada altitude
Pa K K K K K W.m-2 W.m-2 None None m m2.s-1 None None m None"""
    np.savetxt(file,data,header=header)

#make_profile(10,2,1e7,0)
