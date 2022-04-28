#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 02:45:33 2022

@author: cwilkinson
"""

import matplotlib.pyplot as plt 
import matplotlib.pylab as pl
import matplotlib as mpl
import matplotlib
import pandas as pd
import glob
import numpy as np
import random
import os
import json
import time
import sys


def exoris (var) :
    #T, core, rock, yhe, M, idx = var
    #T, M, idx = var
    T, core, M, idx = var
    P = 100
    #core = 0.01
    rock = 0.01
    yhe = 0.24

    print('M,T, core, rock, yhe, idx')
    print(var)
    
    R_J = 7.1492 * 1e7
    M_J = 1.898 * 1e27
    
    eos_core = 'hm'
    eos_ice = 'hm'
    eos_env = 'ker'
    
    M = np.round(M,1)
    T = np.round(T,1)
    
    os.system('cp -r ./exoris ./exoris_' + str(idx))
    time.sleep(1)
    os.chdir("./exoris_" + str(idx) + "/obj")
    os.system('make clean')
    os.system('make')
    os.system('make install')
    os.chdir("./../../")

    with open("./exoris_"+ str(idx) +"/bin/parameters.txt","w")  as f:
        L = [" ! Parameter file \n"]
        L = L + ["&horedt_nml \n\n"]
        L = L + ["computation = 'model' \n"]
        L = L + ["Nint = 300 \n\n"] 
        L = L + ["prot = 9.925\n"]
        L = L + ["measured_r = 7.1492d9 \n"] # In cm
        L = L + ["mass = ", str(np.round(M*1e3,1)), " \n"] # In grams
        L = L + ["surface_T = ", str(np.round(T,1))," \n"] # In K
        L = L + ["surface_P = ", str(P)," \n"] # In bars
        L = L + ["core = ", str(np.round(core,6))," \n"]
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
        
    # os.chdir('./exoris/obj')
    # os.system('make clean')
    # os.system('make')
    # os.system('make install')
    
    # os.chdir('./../..')
    os.chdir("./exoris_"+ str(idx) +"/bin")
    os.system("./horedt5")
    os.chdir("./../../") 
    
    try:
        with open("./exoris_"+ str(idx) +"/bin/model.json") as f:
            data = json.load(f)
            
        r = (data['Req']['val']*1e-2)/R_J
        M = data['parameters']['physical']['mass']['val']*1e-3/M_J
    except :
        r = np.nan
        M = np.nan

    Na = 6.022 * 1e23
    ev = 1.60218e-19
    df = pd.read_table("./exoris_"+str(idx)+"/bin/fort.99", delim_whitespace=True, names=('S', 'P', 'T','rho')).dropna()
    S = df['S'].iloc[0]
    S = ((S*6.242e+11*1000)/((1-yhe)*Na+yhe*2*Na))

    pd.DataFrame({'S': [S], 'R' : [r]}).to_csv('z.csv')
    
    os.system("rm -rf " + "./exoris_"+ str(idx))

    print('\n\nResults here: ',r, S,'\n\n')
    return r, S

def rand_tuple (tpl) :
    if type(tpl[0]) == type(np.linspace(0.1,0.2,1)[0]) :
        return np.round(random.uniform(tpl[0], tpl[1]),3)
    else :
        return random.randint(tpl[0], tpl[1])
    
def delta (tpl,i,dim) :
    sign = 1 if random.random() < 0.5 else -1
    if type(tpl[0]) == type(np.linspace(0.1,0.2,1)[0]) :
        return sign*(tpl[1] - tpl[0])/(3+int(i/(3*dim)))
    else :
        return sign*random.randint(1,2)

def mcmc (bounds,steps,input,target,idx) :
    step = []
    S = []
    gain_step = []
    R = []
    init = ()
    initial_r = np.nan
    file_name = './temp/'+str(idx)+'_'+str(input)+'_'+str(target)+'.csv'
    while glob.glob(file_name) :
        idx += 24
        file_name = './temp/'+str(idx)+'_'+str(input)+'_'+str(target)+'.csv'

    while np.isnan(initial_r) :
        init = ()
        for tpl in bounds :
            init += (rand_tuple(tpl),)

        init += (input,)    
        init += (idx,)
        
        initial_r, initial_S = exoris(init)
    
    step.append(init)
    S.append(initial_S)
    R.append(initial_r)
    gain_step.append(np.abs(initial_r-target))
    df = pd.DataFrame({'steps' : step, 'S': S,'r' : R, 'gain' : gain_step})
    df.to_csv('./temp/'+str(idx)+'_'+str(input)+'_'+str(target)+'.csv',index=False)
    
    for i in range(0,steps) :
        r = np.nan
        while np.isnan(r):
            if not (i == 0) :
                idx_best = np.where(np.abs(np.array(gain_step))==min(np.abs(gain_step)))[0][0]
            else :
                idx_best = 0
            idx_bounds = random.randint(0,len(bounds)-1)
            correct_next_step = False
            while not correct_next_step :
                d = delta (bounds[idx_bounds],i,len(bounds))
                next_step = list(step[idx_best])
                next_step[idx_bounds] = next_step[idx_bounds]+d
            
                if (next_step[idx_bounds] >= bounds[idx_bounds][0]) & (next_step[idx_bounds] <= bounds[idx_bounds][1]) :
                    correct_next_step = True
            
            next_step = tuple(next_step)
            r, entropy = exoris (next_step)

        step.append(next_step)
        if np.isnan(r) :
            gain_step.append(np.nan)
            R.append(np.nan)
        else :
            gain_step.append(np.abs(r-target))
            R.append(r)

        S.append(entropy)

        df = pd.DataFrame({'steps' : step, 'S': S, 'r' : R, 'gain' : gain_step})
        df.to_csv('./temp/'+str(idx)+'_'+str(input)+'_'+str(target)+'.csv',index=False)
        
    return df


M = np.round(np.logspace(26,28, num=20),2)
T = np.round(np.linspace(500, 8000, num=10),2)
core = np.round(np.logspace(-2,-0.01,10),3)
rock = np.round(np.logspace(-1,-0.01,10),3)
yhe = np.round(np.logspace(-2,-0.3,10),3)
P = np.round(np.logspace(0,2,10),2)


gain = []

bounds = ((min(T),max(T)),(min(core),max(core)),(min(rock),max(rock)),
          (min(yhe),max(yhe)))
        
bounds = ((min(T),max(T)),(min(core),max(core)))

R_E = 6371 * 1e3
R_J = 7.1492 * 1e7
M_E = 5.97 * 1e24
M_J = 1.898 * 1e27

file = './exo_evol/data/M_R/zeng_0.7.csv'
lit = pd.read_csv(file)
lit['M'] = lit['M']*M_E
lit['R'] = lit['R']*R_E/R_J
lit['S'] = float(file.split('_')[-1].replace('.csv',''))

idx = int(eval(sys.argv[-1]))

M = lit.iloc[-idx]['M']
R = lit.iloc[-idx]['R']

mcmc (bounds,150,M,R,idx)


files = glob.glob('./temp/*')
for file in files :
   plt.plot(pd.read_csv(file)['gain'])
plt.yscale('log')
plt.show()

def plot_result():
    files = glob.glob('./temp/*')
    frames = []
    for file in files :
        df = pd.read_csv(file)
        df['M'] = float(file.split('_')[-2])
        frames.append(df[df['gain']==min(df['gain'])])

    df = pd.concat(frames)

    # plt.plot(df['M']/M_E,df['r']*R_J/R_E,'*',color='k')
    # plt.plot(lit['M']/M_E,lit['R']*R_J/R_E,'*',color='g')
    # plt.xscale('log')
    # plt.show()

    files = ['./exo_evol/data/M_R/zeng_0.5.csv','./exo_evol/data/M_R/zeng_0.6.csv','./exo_evol/data/M_R/zeng_0.7.csv','./exo_evol/data/M_R/zeng_0.8.csv','./exo_evol/data/M_R/zeng_0.9.csv']
    frames = []
    for file in files :
        lit = pd.read_csv(file)
        lit['M'] = lit['M']*M_E
        lit['R'] = lit['R']*R_E/R_J
        lit['S'] = float(file.split('_')[-1].replace('.csv',''))
        frames.append(lit)

    lit = pd.concat(frames)


    plt.scatter(df['M']/M_E, df['r']*R_J/R_E, c=df['S'],
            cmap='plasma', marker='o',
            norm=mpl.colors.Normalize(vmin=min(df['S']), vmax=max(df['S'])))

    plt.scatter(lit['M']/M_E, lit['R']*R_J/R_E, c=lit['S'],
            cmap='plasma', marker='*',
            norm=mpl.colors.Normalize(vmin=min(df['S']), vmax=max(df['S'])))

    plt.colorbar()

    plt.xscale('log')
    plt.show()


#plot_result()

# files = glob.glob('./temp/*')

# T = []
# S = []
# core = []
# for file in files :
#     df = pd.read_csv(file)
#     for tpl, s in zip(df['steps'], df['S']) :
#         T.append(eval(tpl)[0])
#         core.append(eval(tpl)[1])
#         S.append(s)
        
# plt.plot(T,S,'*')
# plt.show()
# plt.scatter(T, core, c=S,
#         cmap='plasma', marker='*',
#         norm=mpl.colors.Normalize(vmin=min(S), vmax=max(S)))


plt.figure(figsize=(10, 10))
df = pd.read_csv('./exo_evol/exoris_grid.csv')
df = df[df['P_l']==100]
colors = pl.cm.jet(np.linspace(0,1,8))
for y, c in zip(np.sort(random.sample(list(set(df['yhe'])),8)),colors):
    df_ = df[df['yhe']==y].sort_values(['S'])
    plt.plot(df_['T'],df_['S'],'o-',label = 'Yhe : {}'.format(np.round(y,4)),color=c)

plt.ylabel('S (erg/K/g)')
plt.xlabel('T (K)')
plt.title('S = f(T,y=K)')
plt.legend()
plt.show()

plt.figure(figsize=(10, 10))
df = pd.read_csv('./exo_evol/exoris_grid.csv')
df = df[df['P_l']==100]
for T,c in zip(np.sort(random.sample(list(set(df['T'])),8)),colors):
    df_ = df[df['T']==T].sort_values(['S'])
    plt.plot(df_['yhe'],df_['S'],'o-',label = 'T : {}'.format(np.round(T,1)),color=c)

plt.xlabel('yhe ($M_{frac}$)')
plt.ylabel('S (erg/K/g)')
plt.title('S = f(y,T=K)')
plt.legend()
plt.show()

