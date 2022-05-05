# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 13:54:35 2021

@author: cwilkinson
"""

import os
import glob
import pandas as pd
import numpy as np
import multiprocess
import multiprocessing
from itertools import product
import f90nml
import time
from scipy.interpolate import interp1d
from joblib import Parallel, delayed
import re
import sys


def main():
    idx = int(eval(sys.argv[-1]))
    file_prefix = 'Large_shifted_sweep'
    g = np.arange(26.25,200,1)
    g = np.round(g,3)
    launcher(g[idx],file_prefix)

def launcher(g,file_prefix) :
    os.system('cp -r exorem ./running_exorem/exorem_'+file_prefix+'_'+str(g))

    g = [np.round(g,2)]
    T_int = np.arange(25,3025,50)
    T_irr = np.arange(50,3050,50)-3
    M = [1]
    file_prefix = [file_prefix]

    parameters = list(product(M, g, T_int, T_irr, file_prefix))
    
    df = pd.DataFrame()
    for (var) in parameters:
        df = exorem(var,df)

    os.system('rm -r ./running_exorem/exorem_'+file_prefix+'_'+str(g))

def exorem(var, df) :
    M, g, T_int, T_irr, file_prefix = var
    print()
    print('M, g, T_int, T_irr, prefix')
    print(M, g, T_int, T_irr, file_prefix)
    max_index = 0
    status = False
    print('In computation\n')
    T_int = float(np.round(T_int, 2))
    T_irr = np.round(T_irr, 2)
    g = np.round(g, 2)
    P = 1e8
    T_th = make_profile(T_int, g , P, file_prefix)

    weight_apriori = 0.1
    retrieval_flux_error = 1e-4
    chemistry = 0
    
    files = glob.glob('./profiles/temperature_profile_*.dat')
    temperature_profiles_available = [x for x in files if 'temperature' in x]
    if (len(temperature_profiles_available)>0) :
        gs = [x.split('_')[-3] for x in temperature_profiles_available]
        gs_float = [float(x.split('_')[-3]) for x in temperature_profiles_available]
        closest_g_float = list(np.abs(np.array(gs_float) - g))
        closest_g = gs[closest_g_float.index(min(closest_g_float))]
        profiles = glob.glob('./profiles/temperature_profile_'+str(closest_g)+'_'+'*.dat')

        T_ints = [x.split('_')[-2] for x in profiles]
        T_ints_float = [float(x.split('_')[-2]) for x in profiles]
        closest_T_int_float = list(np.abs(np.array(T_ints_float) - T_int))
        closest_T_int = T_ints[closest_T_int_float.index(min(closest_T_int_float))]
        profiles = glob.glob('./profiles/temperature_profile_'+str(closest_g)+'_'+str(closest_T_int)+'*.dat')
        
        T_irrs = [x.split('_')[-1].replace('.dat','') for x in profiles]
        T_irrs_float = [float(x.split('_')[-1].replace('.dat','')) for x in profiles]
        closest_T_irr_float = list(np.abs(np.array(T_irrs_float) - T_irr))
        closest_T_irr = T_irrs[closest_T_irr_float.index(min(closest_T_irr_float))]
        
        print('Looking for following temperature profile :')
        print('./profiles/temperature_profile_'+str(g)+'_'+str(closest_T_int)+'_'+str(closest_T_irr)+'.dat')
        profile = glob.glob('./profiles/temperature_profile_'+str(closest_g)+'_'+str(closest_T_int)+'_'+str(closest_T_irr)+'.dat')[0]
        os.system('cp ' + profile + ' ./running_exorem/exorem_'+file_prefix+'_'+str(g)+'/inputs/atmospheres/temperature_profiles')
        profile = profile.split('/')[-1]
        print('\n\nUsing profile - T_int: {}K - T_irr: {}K\n\n'.format(closest_T_int,closest_T_irr))
    else :
        print('\n\nUsing theoretical profile\n\n')
        profile = 'temperature_profile_example-'+file_prefix+'_'+str(g)+'.dat'

    fail = 0
    while status == False :      
        input_file = f90nml.read('./running_exorem/exorem_'+file_prefix+'_'+str(g)+'/inputs/example.nml')
        earth_mass = 5.97*1e24
        earth_radius = 6371*1e3 
        sigma = 5.67*1e-8
        
        input_file['output_files']['output_files_suffix'] =  str(g)+'_'+str(T_int)+'_'+str(T_irr)
        input_file['target_parameters']['use_gravity'] = True
        input_file['target_parameters']['target_equatorial_gravity'] = g
        input_file['target_parameters']['target_internal_temperature'] = T_int
        input_file['target_parameters']['target_equatorial_radius'] = 70000*1e3
        
        input_file['species_parameters']['elements_names'] = ['He', 'Ne', 'Ar', 'Kr', 'Xe']
        input_file['species_parameters']['elements_metallicity'] = [1.0, 1.0, 1.0, 1.0, 1.0]
        input_file['species_parameters']['species_names'] = ['CH4', 'CO', 'CO2', 'FeH', 'H2O', 'H2S', 'HCN', 'K', 
                                                            'Na', 'NH3', 'PH3', 'TiO', 'VO']
        input_file['species_parameters']['species_at_equilibrium'] = [False, False, False, False, False, 
                                                                    False, False, False, False, False, 
                                                                    False, False, False]

        input_file['retrieval_parameters']['temperature_profile_file'] = profile
        #input_file['retrieval_parameters']['temperature_profile_file'] = 'temperature_profile_example_ref.dat'
        input_file['retrieval_parameters']['retrieval_level_bottom'] = 2
        input_file['retrieval_parameters']['retrieval_level_top'] = 71
        input_file['retrieval_parameters']['retrieval_flux_error_bottom'] = retrieval_flux_error
        input_file['retrieval_parameters']['retrieval_flux_error_top'] = retrieval_flux_error
        input_file['retrieval_parameters']['n_iterations'] = 75
        input_file['retrieval_parameters']['n_non_adiabatic_iterations'] = 0
        input_file['retrieval_parameters']['chemistry_iteration_interval'] = chemistry
        input_file['retrieval_parameters']['cloud_iteration_interval'] = 0
        input_file['retrieval_parameters']['n_burn_iterations'] = 99
        input_file['retrieval_parameters']['smoothing_bottom'] = 0.5
        input_file['retrieval_parameters']['smoothing_top'] = 0.5
        input_file['retrieval_parameters']['weight_apriori'] = weight_apriori
        
        input_file['spectrum_parameters']['wavenumber_min'] = 130
        input_file['spectrum_parameters']['wavenumber_max'] = 30130
        input_file['spectrum_parameters']['wavenumber_step'] = 200 
        
        input_file['atmosphere_parameters']['metallicity'] = M

        input_file['paths']['path_data'] = './running_exorem/exorem_'+file_prefix+'_'+str(g)+'/data/'
        input_file['paths']['path_cia'] = './running_exorem/exorem_'+file_prefix+'_'+str(g)+'/data/cia/'
        input_file['paths']['path_clouds'] = './running_exorem/exorem_'+file_prefix+'_'+str(g)+'/data/cloud_optical_constants/'
        input_file['paths']['path_k_coefficients'] = './running_exorem/exorem_'+file_prefix+'_'+str(g)+'/data/k_coefficients_tables/R50/'
        input_file['paths']['path_temperature_profile'] = './running_exorem/exorem_'+file_prefix+'_'+str(g)+'/inputs/atmospheres/temperature_profiles/'
        input_file['paths']['path_thermochemical_tables'] = './running_exorem/exorem_'+file_prefix+'_'+str(g)+'/data/thermochemical_tables/'
        input_file['paths']['path_light_source_spectra'] = './running_exorem/exorem_'+file_prefix+'_'+str(g)+'/data/stellar_spectra/'
        input_file['paths']['path_outputs'] = './running_exorem/exorem_'+file_prefix+'_'+str(g)+'/outputs/exorem/'
        
        if T_irr == 0 :
            input_file['light_source_parameters']['add_light_source'] = False
            input_file['light_source_parameters']['use_irradiation'] = False
            input_file['light_source_parameters']['use_light_source_spectrum'] = False
            input_file['light_source_parameters']['light_source_irradiation'] = 0

        else :
            input_file['light_source_parameters']['add_light_source'] = True
            input_file['light_source_parameters']['use_irradiation'] = True
            input_file['light_source_parameters']['use_light_source_spectrum'] = False
            input_file['light_source_parameters']['light_source_radius'] = 774263682.6811
            input_file['light_source_parameters']['light_source_range'] = 77426368268112.78
            input_file['light_source_parameters']['light_source_effective_temperature'] = 3555.5556
            input_file['light_source_parameters']['light_source_irradiation'] = sigma*T_irr**4

        input_file['light_source_parameters']['light_source_spectrum_file'] = 'spectrum_BTSettl_3500K_logg5_met0.dat'
        input_file['light_source_parameters']['incidence_angle'] = 0.0

        input_file['atmosphere_parameters']['pressure_max'] = 2000*1e5
        
        if len(glob.glob('./running_exorem/exorem_'+file_prefix+'_'+str(g)+'/inputs/input-'+str(g)+'_'+str(T_int)+'_'+str(T_irr)+'.nml')) : 
            os.remove('./running_exorem/exorem_'+file_prefix+'_'+str(g)+'/inputs/input-'+str(g)+'_'+str(T_int)+'_'+str(T_irr)+'.nml')
                
        input_file.write('./running_exorem/exorem_'+file_prefix+'_'+str(g)+'/inputs/input-'+str(g)+'_'+str(T_int)+'_'+str(T_irr)+'.nml')

        os.system('./running_exorem/exorem_'+file_prefix+'_'+str(g)+'/bin/exorem.exe ./running_exorem/exorem_'+file_prefix+'_'+str(g)+'/inputs/input-'+str(g)+'_'+str(T_int)+'_'+str(T_irr)+'.nml')
        os.remove('./running_exorem/exorem_'+file_prefix+'_'+str(g)+'/inputs/input-'+str(g)+'_'+str(T_int)+'_'+str(T_irr)+'.nml')
        
        storage_files = '/travail/cwilkinson/Travail/VMR_spectra'
        spectra_files = glob.glob('./running_exorem/exorem_'+file_prefix+'_'+str(g)+'/outputs/exorem/*'+str(g)+'_'+str(T_int)+'_'+str(T_irr)+'.dat')
        spectra_files = [x for x in spectra_files if 'temperature' not in x]

        for file in spectra_files :
            os.system('cp {} {}'.format(file,storage_files))
            os.remove(file)

        temperature_profile = glob.glob('./running_exorem/exorem_'+file_prefix+'_'+str(g)+'/outputs/exorem/temperature_profile_'+str(g)+'_'+str(T_int)+'_'+str(T_irr)+'.dat')
        fail += 1

        if (len(temperature_profile) == 0) or (len(spectra_files) == 0): 
            if fail == 1 :
                print('\nfail %a\n'%1)
                weight_apriori = 0.01
                retrieval_flux_error = 1e-4
            elif fail == 2 :
                print('\nfail %a\n'%2)
                weight_apriori = 0.001
                retrieval_flux_error = 1e-4
            elif fail == 3 :
                print('\nfail %a\n'%3)
                weight_apriori = 0.1
                retrieval_flux_error = 1e-3
            elif fail == 4 :
                print('\nfail %a\n'%4)
                weight_apriori = 0.01
                retrieval_flux_error = 1e-3
            elif fail == 5 :
                print('\nfail %a\n'%5)
                weight_apriori = 0.001
                retrieval_flux_error = 1e-3
            elif fail == 5 :
                print('\nfail %a\n'%5)
                weight_apriori = 0.0001
                retrieval_flux_error = 1e-3
            elif fail == 6 :
                print('\nfail %a\n'%6)
                weight_apriori = 0.01
                retrieval_flux_error = 1e-3
                chemistry = 3
            elif fail == 7 :
                print('\nfail %a\n'%7)
                weight_apriori = 0.01
                retrieval_flux_error = 1e-3
                chemistry = 6
            elif fail == 8 :
                print('\nfail %a\n'%8)
                weight_apriori = 0.01
                retrieval_flux_error = 1e-3
                chemistry = 9
            elif fail == 0 :
                print('\nfail %a\n'%9)
                weight_apriori = 0.001
                retrieval_flux_error = 1e-3
                chemistry = 9
            elif fail == 10 :
                print('\nfail %a\n'%10)
                weight_apriori = 1
                retrieval_flux_error = 1e-3
                chemistry = 9
            elif fail == 11 :
                print('\nfail %a\n'%11)
                weight_apriori = 0.1
                retrieval_flux_error = 1e-3
                chemistry = 0
                profile = 'temperature_profile_example_ref.dat'
            elif fail == 12 :
                print('\nfail %a\n'%12)
                weight_apriori = 0.01
                retrieval_flux_error = 1e-3
                chemistry = 0
                profile = 'temperature_profile_example_ref.dat'
            elif fail == 13 :
                print('\nfail %a\n'%13)
                weight_apriori = 0.1
                retrieval_flux_error = 1e-3
                chemistry = 6
                profile = 'temperature_profile_example_ref.dat'
            elif fail == 14 :
                print('\nfail %a\n'%14)
                weight_apriori = 1
                retrieval_flux_error = 1e-3
                chemistry = 0
                profile = 'temperature_profile_example_ref.dat'
            elif fail > 14 :
                print('\nExorem failed\n')
                time.sleep(2)
                status = True
                T_rem = np.nan
                exo = 0
                uncertainty = 100
            
        else :
            status = True
            print('\nLooking for following spectra :')
            print('./VMR_spectra/spectra_'+str(g)+'_'+str(T_int)+'_'+str(T_irr)+'.dat')
            spectra = glob.glob('./VMR_spectra/spectra_'+str(g)+'_'+str(T_int)+'_'+str(T_irr)+'.dat')[0]
            data_dict = load_dat(temperature_profile[0])
            pressure = np.asarray(data_dict['pressure'])
            temperature = np.asarray(data_dict['temperature'])
            inter = interp1d(pressure, temperature, kind = 'nearest')
            T_rem = inter(P)
            exo = 1
            uncertainty = 100*max((data_dict['temperature_uncertainty'] + data_dict['temperature_uncertainty_b'])/data_dict['temperature'])

            df_new = pd.DataFrame({'g' : [g], 'T_int' : [T_int], 'T_irr' : [T_irr], 'atm_M': [M], 'T_rem' : [np.round(T_rem, 2)], 'T_eff' : [T_eff(spectra)] ,'T_p' : [temperature], 'P_p': [pressure], 'uncertainty' : uncertainty})
            df = df.append(df_new, ignore_index='True')
            df.to_csv('./output_exorem/'+file_prefix+'_'+str(g)+'.csv', index = False)

            print (df)

            os.system('cp '+temperature_profile[0]+' ./profiles')

    return df

def T_eff(file) :
    sigma = 5.67*1e-8
    data = load_dat(file)
    del data['units']
    data = pd.DataFrame(data)
    data = data[data['spectral_flux']>0]
    return (((data['wavenumber'].diff()*data['spectral_flux']).sum(skipna=True))/sigma)**(1/4)
    
def load_dat(file, **kwargs):
    """
    Load an Exo-REM data file.
    :param file: data file
    :param kwargs: keyword arguments for loadtxt
    :return: the data
    """
    with open(file, 'r') as f:
        header = f.readline()
        unit_line = f.readline()

    header_keys = header.rsplit('!')[0].split('#')[-1].split()
    units = unit_line.split('#')[-1].split()

    data = np.loadtxt(file, **kwargs)
    data_dict = {}

    for i, key in enumerate(header_keys):
        data_dict[key] = data[:, i]

    data_dict['units'] = units

    return data_dict

def make_profile(T_int, g, P_th, file_prefix):

    T_int = np.round(T_int, 1)
    P_th = np.round(P_th, 1)
    
    file = './exorem/inputs/atmospheres/temperature_profiles/temperature_profile_example_ref.dat'
    data = load_init(file)
    P = np.logspace(-3, 3, num = 200) * (1e5) # en Pa
    P = data[:, 0] # from example T_P exorem file
    g = g * np.ones(np.shape(P))
    met = 1
    
    for i in range (0, 1) : 
        T, tau_ =  t_profile(P, T_int, met, g)
            
    data[:, 1] = T
    g = g[0]
    file = './running_exorem/exorem_'+file_prefix+'_'+str(g)+'/inputs/atmospheres/temperature_profiles/temperature_profile_example-'+file_prefix+'_'+str(g)+'.dat'
    save_profile(file, data)
    
    # plt.plot(data[:, 1], data[:, 0]*1e-5)
    # plt.yscale('log')
    # plt.gca().invert_yaxis()
    # plt.show()
    
    inter = interp1d(P, T, kind = 'nearest')
    T_th = inter(P_th)
    return T_th
    
    
def t_profile(P, T_int, met, g) :
    T = [t_0(T_int)]
    c = np.zeros([11, len(P)])
    tau_ = []
    flag = False
    for ii in range(1, len(P)):
        c[:, ii] = coefficients(T[-1])
        if ii > 5 :
            adiabat = (np.log(T[-1])-np.log(T[-2]))/(np.log(P[ii-1])-np.log(P[ii-2]))
            if adiabat > (2/7) :
                flag = True
        if flag :
            T.append(profile_adiabatique (T, P, met, c, g[ii]))
        else :
            T.append(((3/4) * T_int**4*((2/3) + tau(T, P, met, c, g[ii])))**(1/4)) 
        tau_.append(tau(T, P, met, c, g[ii]))
    return T, tau_

def profile_adiabatique (T, P, met, c, g) :
    T = T[-1] * (P[-1]/P[-2])**(2/7)
    #T = T*(2/7)*(((P[-1]-P[0]))/P[0]) + T
    return T
    
def t_0(T_int) :
    T = ((3/4)*T_int**4*(2/3))**(1/4)
    return T

def coefficients(T) :
    c = np.zeros(11) 
    c[0] = -37.50
    c[1] = 0.00105
    c[2] = 3.2610
    c[3] = 0.84315
    c[4] = -2.339   
    if T <= 800 :
        c[5] = -14.051
        c[6] = 3.055
        c[7] = 0.024
        c[8] = 1.877
        c[9] = -0.445
        c[10] = 0.8321       
    if T > 800 :
        c[5] = 82.241
        c[6] = -55.456
        c[7] = 8.754
        c[8] = 0.7048
        c[9] = -0.0414
        c[10] = 0.8321      
    return c
        
def tau(T, P, met, coeff, g) :
    tau = 0
    for jj in range(0, len(T)-1) :
        c = coeff[:, jj]
        kappa_low = kappa_low_p(T[jj], P[jj], met, c)
        kappa_high = kappa_high_p(T[jj], P[jj], met, c)
        kappa = (kappa_low + kappa_high)#**(-1)
        m = (P[jj+1]-P[jj])/g 
        tau += kappa * m
    
    return tau
    
def kappa_low_p(T, P, met, c) :
    
    kappa = 10**(c[0]*(np.log10(T) - c[1]*np.log10(P)-c[2])**2 \
                 + (c[3]*met + c[4])) # in cm2/g

    return kappa*1e-4/(1e-3) # converting to kg/m**2
    
def kappa_high_p(T, P, met, c) :
    P = P*10 # Converting to dyn/cm-2 from pa
    
    kappa = 10**((c[5] + c[6]*np.log10(T) + c[7]*np.log10(T**2)) \
                 + np.log10(P) * (c[8] + c[9]*np.log10(T)) \
                     + met*c[10]*(0.5 + (1/np.pi)*np.arctan((np.log10(T)-2.5)/0.2))) # in cm2/g
    return kappa*1e-4/(1e-3) # converting to kg/m**2

def load_init(file):
    data = np.loadtxt(file, skiprows = 2)
    return data 

def save_profile(file, data):
    
    header = """pressure temperature temparature_adiabatic temperature_uncertainty temperature_uncertainty_b delta_temperature_convection radiosity radiosity_convective is_convective grada altitude
Pa K K K K K W.m-2 W.m-2 None None m m2.s-1 None None m None"""
    np.savetxt(file, data, header = header)


if __name__  ==  "__main__":
    start_time = time.time()
    main()
    print("\n -- - %s minutes -- -" % np.round((time.time() - start_time)/60, 2))