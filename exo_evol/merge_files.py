#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 23:56:54 2022

@author: cwilkinson
"""

import pandas as pd
import glob
import csv
import os

def main() :
    merge_exoris()
    
def merge_exoris():
    files = glob.glob('../exoris_output/*.csv')
    if len(files)>1:
        frames = []
        for file in files :
            frames.append(pd.read_csv(file,error_bad_lines=False, engine="python"))
            
        exoris = pd.concat(frames,ignore_index=True)
        exoris = exoris.loc[:, ~exoris.columns.str.contains('^Unnamed')]
        exoris = exoris.dropna()
        exoris.to_csv('./exoris_grid.csv',index=False)
        exoris.to_csv('../exoris_output/exoris_grid.csv',index=False)

        Delete_All = [os.remove(file) for file in files]

def merge_exorem():
    files = glob.glob('./../output_exorem/*.csv')
    if len(files)>1:
        frames = []
        for file in files :
            frames.append(pd.read_csv(file,error_bad_lines=False, engine="python"))

        exorem = pd.concat(frames,ignore_index=True)
        exorem = exorem.loc[:, ~exorem.columns.str.contains('^Unnamed')]
        exorem = exorem.dropna()
        exorem.to_csv('./exorem_grid.csv',index=False)
        exorem.to_csv('../output_exorem/exorem_grid.csv',index=False)

        Delete_All = [os.remove(file) for file in files]
    
    
if __name__  ==  "__main__":
    main()