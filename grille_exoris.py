#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 14 16:01:05 2021

@author: cwilkinson
"""

import os
import glob
import json
from multiprocessing import Process
import numpy as np
import pandas as pd
import multiprocess
import multiprocessing
from itertools import product
import sys
from profile_T import *
import time
import random
from joblib import Parallel, delayed
import sys
from datetime import datetime, timedelta


def main() :
    file_prefix = 'Large_mass_range_low_high_T'

    M_E = 5.97 * 1e24
    idx = int(eval(sys.argv[-1]))

    print('input ID is {}'.format(idx))
    os.system('cp -r ./exoris ./running_exoris/exoris_' + file_prefix + '_' + str(idx))
    os.chdir('./running_exoris/exoris_' + file_prefix + '_' + str(idx) + '/obj')
    os.system('make clean')
    os.system('make')
    os.system('make install')
    os.chdir("./../../../")
    time.sleep(1)

    file_prefix = [file_prefix]
    M = np.logspace(26,28.7,24)
    M = [M[idx]]
    T = np.round(np.linspace(100,5000, num=20),2)
    core = [10,20]
    rock = [0.01]
    yhe = [0.001,0.24]
    core_type = ['mass'] #Options frac/mass ; mass in earths
    parameters = list(product(M, T, core, rock, yhe, core_type, file_prefix))

    start = 0
    for ii in range(start, len(parameters)) :
        parameters[ii] = parameters[ii] + (idx, )

    df = pd.DataFrame()

    for (var) in parameters:
        df = exoris (var,df)

    os.system('rm -rf ./running_exoris/exoris_' + file_prefix[0] + '_' + str(idx))
        
def exoris (var,df) :
    M, T, core, rock, yhe, core_type, file_prefix, idx = var

    print('\n idx is: {}'.format(idx))
    print('\n')

    eos_env = 'ker'
    eos_ice = 'hm'
    eos_core = 'hm'

    M_E = 5.97 * 1e24
    M = np.round(M,1)
    T = np.round(T,1)

    if (core_type=='mass'):
        core_value = core*M_E/M
    else: 
        core_value = core
    P = 100.0

    #os.system('cp -r ./exoris ./exoris_' + str(input))
    with open('./running_exoris/exoris_'+file_prefix+'_'+str(idx)+'/bin/parameters.txt',"w") as f:
        L = [" ! Parameter file \n"]
        L = L + ["&horedt_nml \n\n"]
        L = L + ["computation = 'model' \n"]
        L = L + ["Nint = 300 \n\n"] 
        L = L + ["prot = 9.925\n"]
        L = L + ["measured_r = 7.1492d9 \n"] # In cm
        L = L + ["mass = ", str(np.round(M*1e3,1)), " \n"] # In grams
        L = L + ["surface_T = ", str(np.round(T,1))," \n"] # In K
        L = L + ["surface_P = ", str(P)," \n"] # In bars
        L = L + ["core = ", str(np.round(core_value,6))," \n"]
        L = L + ["rock = ", str(np.round(rock,6))," \n"]
        L = L + ["yhe = ", str(np.round(yhe,6))," \n"]
        L = L + ["eos_env = '", eos_env,"' \n"]
        L = L + ["eos_ice = '", eos_ice,"' \n"]
        L = L + ["eos_core = '", eos_core,"' \n"]
        L = L + ["verbose = false\n"]
        L = L + ["figure = false\n"]
        L = L + ["giant = true\n"]
        L = L + ["/\n"]
        L = L + [" "]

        f.writelines(L)
        f.close()

    os.chdir('./running_exoris/exoris_'+file_prefix+'_'+str(idx)+'/bin/')
    os.system("./horedt5")
    os.chdir("./../../../")
    
    try : 
        with open('./running_exoris/exoris_'+file_prefix+'_'+str(idx)+'/bin/model.json') as f:
            data = json.load(f) 

        T_p = np.array(data['T']['val']) 
        P_p = np.array(data['P']['val'])
        idx_p = np.where(P_p<=1000)[0][0] # Index of pressure linkage
        coef = np.mean(np.diff(np.log(T_p[idx_p-3:idx_p+3]))/np.diff(np.log(P_p[idx_p-3:idx_p+3]))) # Sloap at linkage
        y = 1/(1-coef) # Adiabatic index

        df_S = pd.read_table('./running_exoris/exoris_'+file_prefix+'_'+str(idx) +'/bin/fort.99', delim_whitespace=True, names=('S', 'P', 'T','rho')).dropna()

        df_new = pd.DataFrame({'Time': [datetime.now()],
                                'M': [data['parameters']['physical']['mass']['val']], 
                                'T': [data['parameters']['physical']['Ts']['val']], 
                                'core': [data['parameters']['model']['core']['val']], 
                                'core_type': [core_type],
                                'rock': [data['parameters']['model']['rock']['val']], 
                                'yhe' : [data['parameters']['model']['yhe']['val']], 
                                'S' : [df_S['S'].iloc[0]], 
                                'Req' : [data['Req']['val']*1e-5],
                                'R_p' : [np.array(data['R']['val'])*1e-2],
                                'P_s' : [P],
                                'P_p':[P_p],
                                'rho_p':[np.array(data['rho']['val'])*1e-3/1e-6],
                                'T_p':[T_p],
                                'gamma':[y]})
    except :
        df_new = pd.DataFrame({'Time': [datetime.now()],
                       'M': [M], 
                       'T': [T], 
                       'core': [core], 
                       'core_type': [core_type],
                       'rock': [rock], 
                       'yhe' : [yhe], 
                       'S' : [np.nan], 
                       'Req' : [np.nan],
                       'R_p' : [np.nan],
                       'P_s' : [P],
                        'P_p':[np.nan],
                        'rho_p':[np.nan],
                        'T_p':[np.nan],
                        'gamma':[np.nan]})
    
    df = df.append(df_new, ignore_index='True')
    print('\n\n\n Name to be used is : ./exoris_output/'+file_prefix+'_'+str(idx)+'.csv')
    print('\n\n\n')
    df.to_csv('./exoris_output/'+file_prefix+'_'+str(idx)+'.csv',index=False)
    print(df)

    return df

if __name__  ==  "__main__":
    main()