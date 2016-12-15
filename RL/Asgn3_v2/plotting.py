#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 17:57:19 2016

@author: labmember
"""

from numpy import *
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.signal import medfilt

if __name__ == '__main__':
    alphaSet = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    
    fig = plt.figure(num=None, figsize=(18.5, 10.5), dpi=1000, facecolor='w', edgecolor='k')
    x_vec2 = arange(0,50,5)
    clrs = ['b','g','r','k','c','m']
    handle = []
    clr_idx = 0    

    for i in range(len(alphaSet)):
        f_name = 'rewardData/t_n_eps_data' + str(alphaSet[i]) + '.npy'
        f1 = open(f_name,'r')
        t_n_eps = load(f1)
        
        clr = clrs[clr_idx]
        clr_idx += 1
        filter_t_n_eps = medfilt(t_n_eps, 3)
        line1, = plt.plot(x_vec2,filter_t_n_eps,clr, linewidth=2.0)
        handle = handle + [line1,]

    axes = plt.gca()
    axes.set_xlim([0,45])
    axes.set_ylim([30,200])
    
    plt.xlabel('Reward at Goal')
    plt.ylabel('Number of Steps taken')
#    plt.text(35., 1500.,'alpha = ' + str(alpha), fontsize=18)
    rcParams.update({'font.size': 18})
    plt.legend(handle,['alpha = 0.05', 'alpha = 0.1','alpha = 0.2',\
                       'alpha = 0.3', 'alpha = 0.4','alpha = 0.5'],loc=1)
    
    filename = 'rewardData/Combined_Steps_vs_goalReward_alphas.png'
    plt.savefig(filename, bbox_inches='tight')
    plt.close('all')
    