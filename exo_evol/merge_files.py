#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 23:56:54 2022

@author: cwilkinson
"""

import pandas as pd
import numpy as np
import glob
import csv
import os
from scipy.interpolate import interp1d

def main() :
    merge_exoris()
    merge_exorem()
    
def merge_exoris():
    files = glob.glob('../exoris_output/*.csv')
    files = [x for x in files if 'grid' not in x]
    if len(files)>1:
        frames = []
        for file in files :
            frames.append(pd.read_csv(file,error_bad_lines=False, engine="python"))
            
        exoris = pd.concat(frames,ignore_index=True)
        exoris = exoris.loc[:, ~exoris.columns.str.contains('^Unnamed')]
        exoris = exoris.dropna().drop_duplicates(keep='last')
        exoris.to_csv('./exoris_grid.csv',index=False)
        exoris.to_csv('../exoris_output/exoris_grid.csv',index=False)

        Delete_All = [os.remove(file) for file in files]

def merge_exorem():
    files = glob.glob('./../output_exorem/*.csv')
    files = [x for x in files if 'grid' not in x]
    if len(files)>1:
        frames = []
        for file in files :
            frames.append(pd.read_csv(file,error_bad_lines=False, engine="python"))

        exorem = pd.concat(frames,ignore_index=True)
        exorem = exorem.loc[:, ~exorem.columns.str.contains('^Unnamed')]
        exorem = exorem.dropna().drop_duplicates(keep='last')
        exorem = correct_T_eff_R(exorem)
        exorem.to_csv('./exorem_grid.csv',index=False)
        exorem.to_csv('../output_exorem/exorem_grid.csv',index=False)

        Delete_All = [os.remove(file) for file in files]

def correct_T_eff_R(exorem):
    T_eff_corrected = []
    delta_r = []
    for ii in range(0,len(exorem)):
        g = exorem['g'].iloc[ii]
        T_int = exorem['T_int'].iloc[ii]
        T_irr = exorem['T_irr'].iloc[ii]
        spectra = '../VMR_spectra/spectra_{}_{}_{}.dat'.format(g,T_int,T_irr)
        T_profile = '../profiles/temperature_profile_{}_{}_{}.dat'.format(g,T_int,T_irr)
        T_eff_corrected.append(T_eff(spectra))
        delta_r.append(Radius_1000_1(T_profile))

    exorem['T_eff'] = T_eff_corrected
    exorem['delta_r'] = delta_r

    return exorem

def Radius_1000_1(file) :
    data = load_dat(file)
    del data['units']
    data = pd.DataFrame(data)
    P_p = np.array(data['pressure'])*1e-5
    R = np.array(data['altitude'])
    R_1 = float(interp1d(P_p,R,kind='cubic')(1))
    R_1000 = float(interp1d(P_p,R,kind='cubic')(100)) #Defined negatively 
    return R_1-R_1000

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
    
    
if __name__  ==  "__main__":
    main()