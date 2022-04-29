#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 23:56:54 2022

@author: cwilkinson
"""

import pandas as pd
import glob
import csv

def main() :
    merge_exoris()
    
def merge_exoris():
    files = glob.glob('../exoris_output/*.csv')
    frames = []
    for file in files :
        frames.append(pd.read_csv(file,error_bad_lines=False, engine="python"))
        
    exoris = pd.concat(frames,ignore_index=True)
    exoris = exoris.loc[:, ~exoris.columns.str.contains('^Unnamed')]
    exoris = exoris.dropna()
    exoris.to_csv('./exoris_grid.csv',index=False)
    
    
if __name__  ==  "__main__":
    main()