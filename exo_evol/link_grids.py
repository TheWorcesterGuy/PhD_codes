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
warnings.filterwarnings("ignore")

def main() :
    parameters = load_parameters()
    exoris = pd.read_csv('exoris_grid.csv')
    exorem = pd.read_csv('exorem_grid.csv')
    exoris, exorem = formating(exoris,exorem)
    exoris, exorem = initial_grid_reduction(parameters,exoris,exorem)
    exorem, exoris = cleaning(exorem,exoris)
    exorem = interpolate_exorem(exorem)
    exoris = interpolate_exoris_at_M(exoris,parameters)
    exoris = interpolation_exoris_at_exorem(exoris,exorem,parameters)
    merged_grids = merge_grids(exoris,exorem)
    #merged_grids = interpolate_final_at_M(merged_grids,parameters)
    plotting(merged_grids)
    
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
    axis_to_exoris = ['T_1000','g','S','y',]
    axis_to_exorem = ['y','g','T_1000']
    fixed_axis = 'M'

    frames = []
    for M in np.sort(list(set(exoris['M']))) :
        exoris_ = exoris[exoris['M']==M]
        exoris_new = pd.DataFrame()
        if len(exoris_)>3:
            exoris_new['y'] = interp1d(exoris_['T_1000'],exoris_['y'], bounds_error=False ,fill_value=np.nan)(np.sort(list(set(exorem['T_1000']))))
            exoris_new['S'] = interp1d(exoris_['T_1000'],exoris_['S'], bounds_error=False ,fill_value=np.nan)(np.sort(list(set(exorem['T_1000']))))
            exoris_new['g'] = interp1d(exoris_['T_1000'],exoris_['g'], bounds_error=False ,fill_value=np.nan)(np.sort(list(set(exorem['T_1000']))))
            exoris_new['T_1000'] = np.sort(list(set(exorem['T_1000'])))
            exoris_new['M'] = M
            frames.append(exoris_new)
    exoris = pd.concat(frames,ignore_index=True).dropna()

    frames = []
    for T in np.sort(list(set(exoris['T_1000']))) :
        exoris_ = exoris[exoris['T_1000']==T]
        exoris_new = pd.DataFrame()
        if len(exoris_)>5:
            exoris_new['y'] = interp1d(exoris_['g'],exoris_['y'], bounds_error=False ,fill_value=np.nan)(np.sort(list(set(exorem['g']))))
            exoris_new['S'] = interp1d(exoris_['g'],exoris_['S'], bounds_error=False ,fill_value=np.nan)(np.sort(list(set(exorem['g']))))
            exoris_new['M'] = interp1d(exoris_['g'],exoris_['M'], bounds_error=False ,fill_value=np.nan)(np.sort(list(set(exorem['g']))))
            exoris_new['g'] = np.sort(list(set(exorem['g'])))
            exoris_new['T_1000'] = T
            frames.append(exoris_new)
    exoris = pd.concat(frames,ignore_index=True).dropna()

    exoris['Req'] = (G_u*exoris['M']/exoris['g'])**(1/2)
    return exoris

def merge_grids(exoris,exorem):

    df = exoris.merge(exorem, how='inner', on=['g','T_1000'])
    df_ =  df[(df['y_x']-df['y_y']).abs()<0.025]
    return df_
    
def interpolate_exoris_at_M(exoris,parameters):
    G_u = 6.674*1e-11
    M_p = parameters['mass'].iloc[0]

    frames = []
    M_grid = list(np.logspace(26,29,4000))
    M_grid.append(M_p)
    M_grid = np.sort(M_grid)
    for T in np.sort(list(set(exoris['T']))):
        exoris_ = exoris[exoris['T']==T]
        exoris_new = pd.DataFrame()
        if len(exoris_)>5:
            exoris_new['g'] = interp1d(exoris_['M'],exoris_['g'], bounds_error=False ,fill_value=np.nan)(M_grid)
            exoris_new['S'] = interp1d(exoris_['M'],exoris_['S'], bounds_error=False ,fill_value=np.nan)(M_grid)
            exoris_new['y'] = interp1d(exoris_['M'],exoris_['y'], bounds_error=False ,fill_value=np.nan)(M_grid)
            exoris_new['T_1000'] = interp1d(exoris_['M'],exoris_['T_1000'], bounds_error=False ,fill_value=np.nan)(M_grid)
            exoris_new['M'] = M_grid
            frames.append(exoris_new)
    exoris = pd.concat(frames,ignore_index=True).dropna()

    exoris['Req'] = (G_u*exoris['M']/exoris['g'])**(1/2)
    return exoris

def interpolate_exorem(exorem):
    frames = []
    T_int = np.linspace(0,2000,4000)
    for g in np.sort(list(set(exorem['g']))):
        exorem_temp = exorem[exorem['g']==g]
        for T in np.sort(list(set(exorem['T_irr']))):
            exorem_ = exorem_temp[exorem_temp['T_irr']==T]
            exorem_new = pd.DataFrame()
            if len(exorem_)>10:
                exorem_new['y'] = interp1d(exorem_['T_int'],exorem_['y'], bounds_error=False ,fill_value=np.nan)(T_int)
                exorem_new['T_1000'] = interp1d(exorem_['T_int'],exorem_['T_1000'], bounds_error=False ,fill_value=np.nan)(T_int)
                exorem_new['T_1'] = interp1d(exorem_['T_int'],exorem_['T_1'], bounds_error=False ,fill_value=np.nan)(T_int)
                exorem_new['T_eff'] = interp1d(exorem_['T_int'],exorem_['T_eff'], bounds_error=False ,fill_value=np.nan)(T_int)
                exorem_new['T_int'] = T_int
                exorem_new['g'] = g
                exorem_new['T_irr'] = T
                frames.append(exorem_new)
    exorem = pd.concat(frames,ignore_index=True).dropna()

    # frames = []
    # g = np.linspace(0,100,300)
    # for T_int in np.sort(list(set(exorem['T_int']))):
    #     exorem_temp = exorem[exorem['T_int']==T_int]
    #     for T_irr in np.sort(list(set(exorem['T_irr']))):
    #         exorem_ = exorem_temp[exorem_temp['T_irr']==T_irr]
    #         exorem_new = pd.DataFrame()
    #         if len(exorem_)>10:
    #             exorem_new['y'] = interp1d(exorem_['g'],exorem_['y'], bounds_error=False ,fill_value=np.nan)(g)
    #             exorem_new['T_1000'] = interp1d(exorem_['g'],exorem_['T_1000'], bounds_error=False ,fill_value=np.nan)(g)
    #             exorem_new['T_1'] = interp1d(exorem_['g'],exorem_['T_1'], bounds_error=False ,fill_value=np.nan)(g)
    #             exorem_new['T_eff'] = interp1d(exorem_['g'],exorem_['T_eff'], bounds_error=False ,fill_value=np.nan)(g)
    #             exorem_new['T_irr'] = T_irr
    #             exorem_new['T_int'] = T_int
    #             exorem_new['g'] = g
    #             frames.append(exorem_new)
    # exorem = pd.concat(frames,ignore_index=True).dropna()

    return exorem

def interpolate_final_at_M(final,parameters):
    G_u = 6.674*1e-11
    M_p = parameters['mass'].iloc[0]

    frames = []
    R_grid = list(np.linspace(final['Req'].min(),final['Req'].max(),2000))
    R_grid = np.sort(R_grid)
    for g in np.sort(list(set(final['g']))):
        final_ = final[final['g']==g]
        final_new = pd.DataFrame()
        if len(final_)>5:
            final_new['S'] = interp1d(final_['Req'],final_['S'], bounds_error=False ,fill_value=np.nan)(R_grid)
            final_new['T_int'] = interp1d(final_['Req'],final_['T_int'], bounds_error=False ,fill_value=np.nan)(R_grid)
            final_new['T_eff'] = interp1d(final_['Req'],final_['T_eff'], bounds_error=False ,fill_value=np.nan)(R_grid)
            final_new['T_1000'] = interp1d(final_['Req'],final_['T_1000'], bounds_error=False ,fill_value=np.nan)(R_grid)
            final_new['M'] = interp1d(final_['Req'],final_['M'], bounds_error=False ,fill_value=np.nan)(R_grid)
            final_new['Req'] = R_grid
            final_new['g'] = g
            frames.append(final_new)
    final = pd.concat(frames,ignore_index=True).dropna()

    frames = []
    M_grid = list(np.linspace(final['M'].min(),final['M'].max(),2000))
    M_grid = np.sort(M_grid)
    for R in np.sort(list(set(final['Req']))):
        final_ = final[final['Req']==R]
        final_new = pd.DataFrame()
        if len(final_)>3:
            final_new['S'] = interp1d(final_['M'],final_['S'], bounds_error=False ,fill_value=np.nan)(M_grid)
            final_new['T_int'] = interp1d(final_['M'],final_['T_int'], bounds_error=False ,fill_value=np.nan)(M_grid)
            final_new['T_eff'] = interp1d(final_['M'],final_['T_eff'], bounds_error=False ,fill_value=np.nan)(M_grid)
            final_new['T_1000'] = interp1d(final_['M'],final_['T_1000'], bounds_error=False ,fill_value=np.nan)(M_grid)
            final_new['g'] = interp1d(final_['M'],final_['g'], bounds_error=False ,fill_value=np.nan)(M_grid)
            final_new['Req'] = R
            final_new['M'] = M_grid
            frames.append(final_new)
    final = pd.concat(frames,ignore_index=True).dropna()

    return final

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

def scatter_plot(exoris) :
    cm = plt.cm.get_cmap('RdYlBu')
    #1
    plt.scatter(exoris['M'], exoris['T_1000'], c=exoris['S'], cmap=cm)
    plt.colorbar()
    plt.show()  

    plt.scatter(exoris['M'], exoris['T_1000'], c=exoris['g'], cmap=cm)
    plt.colorbar()
    plt.show()  

    plt.scatter(exoris['M'], exoris['T_1000'], c=exoris['y'], cmap=cm)
    plt.colorbar()
    plt.show()   

    #2
    plt.scatter(exoris['M'], exoris['y'], c=exoris['T_1000'], cmap=cm)
    plt.colorbar()
    plt.show()  

    plt.scatter(exoris['M'], exoris['y'], c=exoris['g'], cmap=cm)
    plt.colorbar()
    plt.show()  

    plt.scatter(exoris['M'], exoris['y'], c=exoris['S'], cmap=cm)
    plt.colorbar()
    plt.show()  

    #3
    plt.scatter(exoris['M'], exoris['g'], c=exoris['T_1000'], cmap=cm)
    plt.colorbar()
    plt.show()  

    plt.scatter(exoris['M'], exoris['g'], c=exoris['y'], cmap=cm)
    plt.colorbar()
    plt.show()  

    plt.scatter(exoris['M'], exoris['g'], c=exoris['S'], cmap=cm)
    plt.colorbar()
    plt.show()  

    plt.scatter(exoris['S'], exoris['T_1000'], c=exoris['y'], cmap=cm)
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

    exorem = exorem[exorem['y']>1] 
    frames = []
    for g in list(set(exorem['g'])):
        exorem_ = exorem[exorem['g']==g]
        exorem_ = exorem_.sort_values(by=['T_1000'])
        exorem_[exorem_['y'].isin(medfilt(exorem_['y'],11))]
        frames.append(exorem_)

    exorem = pd.concat(frames, ignore_index=True)

    plt.scatter(exorem['T_1000'], exorem['g'], c=exorem['y'], cmap=cm)
    plt.ylabel('g (m/s2)')
    plt.xlabel('T_1000 (k)')
    plt.colorbar()
    plt.show() 

def plotting(final):
    final.to_csv('evolution_grid.csv',index=False)
    final = pd.read_csv('evolution_grid.csv')
    R_J = 69911000
    M_J = 1.898*1e27

    cm = plt.cm.get_cmap('RdYlBu')
    plt.figure()
    plt.scatter(final['M']/M_J, final['Req']/(R_J), c=final['T_eff'], cmap=cm)
    plt.xlabel('M (J)')
    plt.xscale('log')
    plt.ylabel('Radius (m)')
    plt.title('T_eff = Radius and M')
    plt.colorbar()
    plt.savefig('./images/T_f(R_M).png')
    #plt.show() 

    plt.figure()
    plt.scatter(final['T_eff'], final['Req']/(R_J), c=final['M']/M_J, cmap=cm)
    plt.xlabel('T_eff (K)')
    plt.ylabel('Radius (m)')
    plt.title('Mass (J) = Radius and T_eff')
    #plt.xscale('log')
    plt.colorbar()
    plt.savefig('./images/M_f(T_R).png')
    #plt.show() 

    
if __name__  ==  "__main__":
    main()