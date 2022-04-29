import os
import glob
import numpy as np
import pandas as pd
import sys
from scipy.interpolate import interp1d, interp2d
from datetime import datetime, timedelta
import re
import matplotlib.pyplot as plt 

def main() :
    parameters = load_parameters()
    exoris = pd.read_csv('exoris_grid.csv')
    exorem = pd.read_csv('exorem_grid.csv')
    exoris, exorem = formating(exoris,exorem)
    exoris, exorem = initial_grid_reduction(parameters,exoris,exorem)
    exoris = interpolation_exoris_at_M(exoris,parameters)
    merged_grids = interpolation_exoris_at_exorem(exoris,exorem,parameters)
    merged_grids['error'] = (merged_grids['y']-merged_grids['y_exorem']).abs()
    merged_grids = merged_grids.sort_values(by=['error'])
    final = merged_grids[(merged_grids['y']-merged_grids['y_exorem']).abs()<0.2]
    final = final.drop_duplicates(['g'], keep='first')
    plotting(final)
    
def load_parameters() :
    parameters = pd.read_csv("parameters.txt",delimiter='=',skipinitialspace=True).T
    parameters.columns = parameters.columns.str.strip()
    parameters.apply(pd.to_numeric, errors='coerce')
    return parameters

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
       
def interpolation_exoris_at_exorem(exoris,exorem,parameters) :    
    G_u = 6.674*1e-11
    M_p = parameters['mass'].iloc[0]
    frames = []
    for g, T, T_int,T_irr,T_eff,y in zip(list(exorem['g']),list(exorem['T_1000']),list(exorem['T_int']),list(exorem['T_irr']),list(exorem['T_eff']),list(exorem['y'])) :
        exoris_new = pd.DataFrame()
        if (g > exoris['g'].min()) & (g < exoris['g'].max()) & (T > exoris['T_1000'].min()) & (T < exoris['T_1000'].max()) :
            exoris_new['y'] = interp2d(list(exoris['g']),list(exoris['T_1000']),list(exoris['y']))(g,T)
            exoris_new['S'] = interp2d(list(exoris['g']),list(exoris['T_1000']),list(exoris['S']))(g,T)
            exoris_new['g'] = g
            exoris_new['T_1000'] = T
            exoris_new['T_int'] = T_int
            exoris_new['T_irr'] = T_irr
            exoris_new['T_eff'] = T_eff
            exoris_new['y_exorem'] = y
            frames.append(exoris_new)
            
    exoris = pd.concat(frames,ignore_index=True)
    
    exoris['M'] = M_p
    exoris['R'] = (G_u*exoris['M']/exoris['g'])**(1/2)
    
    return exoris
    
def interpolation_exoris_at_M(exoris,parameters):
    G_u = 6.674*1e-11
    M_p = parameters['mass'].iloc[0]
    frames = []
    for T in list(set(exoris['T'])) :
        exoris_new = pd.DataFrame()
        exoris_ = exoris[exoris['T']==T]
        if (M_p > exoris_['M'].min()) & (M_p < exoris_['M'].max()) :
            exoris_new['T_1000'] = [interp1d(exoris_['M'],exoris_['T_1000'])(M_p)]
            exoris_new['y'] = [interp1d(exoris_['M'],exoris_['y'],kind='nearest')(M_p)]
            exoris_new['g'] = [interp1d(exoris_['M'],exoris_['g'],kind='nearest')(M_p)]
            exoris_new['S'] = [interp1d(exoris_['M'],exoris_['S'],kind='nearest')(M_p)]
            frames.append(exoris_new)
        
    exoris = pd.concat(frames,ignore_index=True)
    exoris['M'] = M_p
    exoris['R'] = (G_u*exoris['M']/exoris['g'])**(1/2)
    return exoris

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
    for ii in range(0,len(df_temp)) :
        T_1000.append(float(interp1d(df_temp.iloc[ii]['P_p'],df_temp.iloc[ii]['T_p'])(1000)))
    
    df['T_1000'] = T_1000
    
    return df

def extract_T_at_P_exorem(df):
    df_temp = df
    df_temp['T_p'] = df_temp['T_p'].apply(str_to_list)
    df_temp['P_p'] = df_temp['P_p'].apply(str_to_list)
    
    T_1000 = []
    for ii in range(0,len(df_temp)) :
        T_1000.append(float(interp1d(np.array(df_temp.iloc[ii]['P_p'])*1e-5,df_temp.iloc[ii]['T_p'])(1000)))
    
    df['T_1000'] = T_1000
    
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

def plotting(final):
    R_J = 69911000
    M_J = 1.898*1e27
    plt.figure(figsize=(8, 6))
    plt.plot(final['T_eff'],final['R']/R_J,'*')
    plt.xlabel('$T_{eff}$ (K)')
    plt.ylabel('Radius ($R_{J}$)')
    plt.title('Radius as a function of planets effective temperature\nMass : {:.2f} Jupters'.format(final['M'].iloc[0]/M_J))
    plt.show()
    
    
if __name__  ==  "__main__":
    main()