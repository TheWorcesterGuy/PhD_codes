import os
import glob
import numpy as np
import pandas as pd
import sys
from scipy.interpolate import interp1d, interp2d
from scipy.signal import medfilt
from datetime import datetime, timedelta
import re
import matplotlib.pyplot as plt 
import warnings
import random
warnings.filterwarnings("ignore")

R_J = 69911000
M_J = 1.898*1e27

def main() :
    os.system('python3 merge_files.py')
    parameters = load_parameters()
    exoris = pd.read_csv('exoris_grid.csv')
    exorem = pd.read_csv('exorem_grid.csv')
    exoris, exorem = formating(exoris,exorem)
    exorem, exoris = cleaning(exorem,exoris)
    plot_exorem_profile(exorem)
    exoris, exorem = initial_grid_reduction(parameters,exoris,exorem)
    plotting_exoris(exoris)
    plotting_exorem(exorem)
    plotting_merged()


def plotting_exoris(exoris=pd.read_csv('exoris_grid.csv')) :
    cm = plt.cm.get_cmap('RdYlBu').reversed()
    plt.scatter(exoris['S'], exoris['T'], c=exoris['y'], cmap=cm)
    plt.xlabel('Entropy (Ev/K/atm)')
    plt.ylabel('$T_{1000}$ (K)')
    plt.title('Adiabatic index as a function of Entropy and Temperature')
    plt.colorbar()
    plt.show()  
    
    plt.scatter(exoris['g'], exoris['T_1000'], c=exoris['y'], cmap=cm)
    plt.xlabel('Entropy (Ev/K/atm)')
    plt.ylabel('$T_{1000}$ (K)')
    plt.title('Adiabatic index as a function of Entropy and Temperature')
    plt.colorbar()
    plt.show() 

    plt.scatter(exoris['M'], exoris['Req'], c=exoris['S'], cmap=cm)
    plt.xlabel('Mass (Kg)')
    plt.ylabel('Radius (m)')
    plt.xscale('log')
    plt.title('Entropy as a function of Radius and Mass')
    plt.colorbar()
    plt.show() 


def plotting_exorem(exorem=pd.read_csv('exorem_grid.csv')) :
    cm = plt.cm.get_cmap('RdYlBu').reversed()
    exorem = exorem[exorem['y']>1] 
    frames = []
    for g in list(set(exorem['g'])):
        exorem_ = exorem[exorem['g']==g]
        exorem_ = exorem_.sort_values(by=['T_1000'])
        exorem_[exorem_['y'].isin(medfilt(exorem_['y'],11))]
        frames.append(exorem_)

    exorem = pd.concat(frames, ignore_index=True)

    exorem_ = exorem[exorem['g']==random.choice(list(exorem['g']))]
    plt.scatter(exorem_['T_int'], exorem_['T_irr'], c=exorem_['T_eff'], cmap=cm)
    plt.xlabel('$T_{int}$ (K)')
    plt.ylabel('$T_{irr}$ (k)')
    plt.title('Exorem : T_eff=f(T_int,T_irr) at g {} $m/s^2$'.format(exorem_['g'].iloc[0]))
    plt.colorbar()
    plt.show()  

    exorem_ = exorem[exorem['T_irr']==random.choice(list(exorem['T_irr']))]
    plt.scatter(exorem_['g'], exorem_['T_int'], c=exorem_['T_eff'], cmap=cm)
    plt.xlabel('g ($m.s^{-2}$)')
    plt.ylabel('$T_{int}$ (k)')
    plt.title('Exorem : T_eff=f(g,T_int)')
    plt.colorbar()
    plt.show() 

    exorem_ = exorem[exorem['T_int']==random.choice(list(exorem['T_int']))]
    plt.scatter(exorem_['g'], exorem_['T_irr'], c=exorem_['T_eff'], cmap=cm)
    plt.xlabel('g ($m.s^{-2}$)')
    plt.ylabel('$T_{irr}$ (k)')
    plt.title('Exorem : T_eff=f(g,T_irr)')
    plt.colorbar()
    plt.show()

    exorem_ = exorem[exorem['T_int']==random.choice(list(exorem['T_int']))]
    plt.scatter(exorem_['g'], exorem_['T_irr'], c=exorem_['T_1000'], cmap=cm)
    plt.xlabel('g ($m.s^{-2}$)')
    plt.ylabel('$T_{irr}$ (k)')
    plt.title('Exorem : T_1000=f(g,T_irr)')
    plt.colorbar()
    plt.show()

    exorem_ = exorem[exorem['T_int']==random.choice(list(exorem['T_int']))]
    exorem_ = exorem_[exorem_['g']==random.choice(list(exorem['g']))]
    plt.plot(exorem_['T_irr'],exorem_['T_eff'])
    plt.xlabel('$T_{irr}$ (K)')
    plt.ylabel('$T_{eff}$ (k)')
    plt.title('Exorem : T_eff=f(T_irr)')
    plt.show()

def plot_exorem_profile(exorem):
    exorem_t = exorem[exorem['T_int']==random.choice(list(exorem['T_int']))]
    exorem_t = exorem_t[exorem_t['T_irr']==random.choice(list(exorem_t['T_irr']))]
    gs = np.sort(list(set(exorem_t['g'])))

    viridis = plt.cm.get_cmap('viridis', len(gs))
    i=0
    for g in gs:
        exorem_ = exorem_t[exorem_t['g']==g]
        plt.plot(np.array(exorem_['T_p'].iloc[0]),np.array(exorem_['P_p'].iloc[0])*1e-5,color=viridis(i))
        i += 1
            
    plt.yscale('log')
    plt.xscale('log')
    plt.gca().invert_yaxis()
    plt.xlabel('$T$ (K)')
    plt.ylabel('$P$ (bar)')
    plt.title('Exorem profile at T_int : {}K, T_irr : {}K'.format(exorem_['T_int'].iloc[0],exorem_['T_irr'].iloc[0]))
    plt.show()


    exorem_t = exorem[exorem['T_int']==random.choice(list(exorem['T_int']))]
    exorem_t = exorem_t[exorem_t['g']==random.choice(list(exorem_t['g']))]
    T_irrs = np.sort(list(set(exorem_t['T_irr'])))

    viridis = plt.cm.get_cmap('viridis', len(T_irrs))
    i=0
    for T in T_irrs:
        exorem_ = exorem_t[exorem_t['T_irr']==T]
        plt.plot(np.array(exorem_['T_p'].iloc[0]),np.array(exorem_['P_p'].iloc[0])*1e-5,color=viridis(i))
        i += 1
            
    plt.yscale('log')
    plt.xscale('log')
    plt.gca().invert_yaxis()
    plt.xlabel('$T$ (K)')
    plt.ylabel('$P$ (bar)')
    plt.title('Exorem profile at T_int : {}K, g : {}m.s-2'.format(exorem_['T_int'].iloc[0],exorem_['g'].iloc[0]))
    plt.show()

def plotting_merged(merged_grids=pd.read_csv('evolution_grid.csv')) :
    cm = plt.cm.get_cmap('RdYlBu').reversed()
    merge_ = merged_grids[merged_grids['M']==random.choice(list(merged_grids['M']))]
    #merge_ = merged_grids
    plt.scatter(merge_['T_eff'], merge_['Req']/(R_J), c=merge_['S'], cmap=cm)
    plt.xlabel('T_eff ($K$)')
    plt.ylabel('$R_{eq}$ (J)')
    plt.title('S=f(T_eff,R)')
    plt.colorbar()
    plt.show() 

    merge_ = merged_grids
    plt.scatter(merge_['T_eff'], merge_['Req']/(R_J), c=merge_['M']/(M_J), cmap=cm)
    plt.xlabel('T_eff ($K$)')
    plt.ylabel('$R_{eq}$ (J)')
    plt.title('M=f(T_eff,R)')
    plt.colorbar()
    plt.show() 


def initial_grid_reduction(parameters,exoris,exorem):
    M_E = 5.97 * 1e24
    
    #Exoris on Core fraction
    exoris['core_earths'] = (exoris['core']*exoris['M']/M_E).round(0)
    exoris = exoris[exoris['core_type']=='mass']
    exoris = exoris[exoris['core_earths'] == exoris.iloc[(exoris['core_earths']-parameters['core'].iloc[0]).abs().argsort()[:1]]['core_earths'].iloc[0]]
    #Exoris on rock fraction
    exoris = exoris[exoris['rock'] == exoris.iloc[(exoris['rock']-parameters['rock'].iloc[0]).abs().argsort()[:1]]['rock'].iloc[0]]
    #Exoris on yhe fraction
    exoris = exoris[exoris['yhe'] == exoris.iloc[(exoris['yhe']-0.24*parameters['Met'].iloc[0]).abs().argsort()[:1]]['yhe'].iloc[0]]
    
    #Exorem on T_irr
    if parameters['use_Tirr'].iloc[0]:
        exorem = exorem[(exorem['T_irr']-parameters['T_irr'].iloc[0]).abs()<parameters['dT_irr'].iloc[0]]
    #Exorem on T_eff
    if parameters['use_Teff'].iloc[0]:
        exorem = exorem[(exorem['T_eff']-parameters['T_eff'].iloc[0]).abs()<parameters['dT_eff'].iloc[0]]
    
    return exoris, exorem

def load_parameters() :
    parameters = pd.read_csv("parameters.txt",delimiter='=',skipinitialspace=True).T
    parameters.columns = parameters.columns.str.strip()
    parameters.apply(pd.to_numeric, errors='coerce')
    return parameters

def str_to_list(x):
    if ',' in x:
        return eval(re.sub("\s+", "",x.replace('\n','')))
    else :
        return eval(re.sub("\s+", ",",x.replace('\n','')).replace('[,','['))

def extract_T_at_P_exoris(df):
    df_temp = df
    df_temp['T_p'] = df_temp['T_p'].apply(str_to_list)
    df_temp['P_p'] = df_temp['P_p'].apply(str_to_list)
    
    T_1000 = []
    y = []
    for ii in range(0,len(df_temp)) :
        P_p = np.array(df_temp.iloc[ii]['P_p'])*1e5
        T_p = np.array(df_temp.iloc[ii]['T_p'])
        T_1000.append(float(interp1d(df_temp.iloc[ii]['P_p'],df_temp.iloc[ii]['T_p'])(1000)))
        
        idx_p = np.where(P_p<=1000*1e5)[0][0] # Index of pressure linkage
        coef = np.mean(np.diff(np.log(T_p[idx_p-3:idx_p+3]))/np.diff(np.log(P_p[idx_p-3:idx_p+3]))) # Sloap at linkage
        y.append(1/(1-coef)) # Adiabatic index
    
    df['T_1000'] = T_1000
    df['y'] = y
    return df

def extract_T_at_P_exorem(df):
    df_temp = df
    df_temp['T_p'] = df_temp['T_p'].apply(str_to_list)
    df_temp['P_p'] = df_temp['P_p'].apply(str_to_list)
    
    T_1000 = []
    T_1 = []
    y = []
    for ii in range(0,len(df_temp)) :
        P_p = np.array(df_temp.iloc[ii]['P_p'])
        T_p = np.array(df_temp.iloc[ii]['T_p'])
        T_1000.append(float(interp1d(np.array(df_temp.iloc[ii]['P_p'])*1e-5,df_temp.iloc[ii]['T_p'])(1)))
        T_1.append(float(interp1d(np.array(df_temp.iloc[ii]['P_p'])*1e-5,df_temp.iloc[ii]['T_p'])(1000)))

        idx_p = np.where(P_p>=1000*1e5)[0][0] # Index of pressure linkage
        coef = np.mean(np.diff(np.log(T_p[idx_p-1:idx_p+1]))/np.diff(np.log(P_p[idx_p-1:idx_p+1]))) # Sloap at linkage
        y.append(1/(1-coef)) # Adiabatic index
    
    df['T_1000'] = T_1000
    df['y'] = y
    df['T_1'] = T_1
    return df

def formating(exoris,exorem):
    G_u = 6.674*1e-11
    try :
        exorem = exorem.drop(['uncertainty'], axis=1)
    except :
        pass
    exoris = exoris.rename(columns={"gamma": "y"})
    exorem = extract_T_at_P_exorem(exorem)
    exoris = extract_T_at_P_exoris(exoris)

    exoris['M'] = exoris['M']*1e-3
    exoris['Req'] = exoris['Req']*1e3

    if 'g' not in exoris.columns :
        exoris['g'] = G_u*exoris['M']/exoris['Req']**2
    
    return exoris, exorem

def cleaning(exoris,exorem):
    exorem = exorem[exorem['y']>1] 
    frames = []
    for g in list(set(exorem['g'])):
        exorem_ = exorem[exorem['g']==g]
        exorem_ = exorem_.sort_values(by=['T_1000'])
        exorem_[exorem_['y'].isin(medfilt(exorem_['y'],11))]
        frames.append(exorem_)

    exorem = pd.concat(frames, ignore_index=True)
    return exoris, exorem


if __name__  ==  "__main__":
    main()