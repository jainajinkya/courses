#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 14:55:14 2016

@author: labmember
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.signal import medfilt


if __name__ == '__main__':
    n_eps = 1000
    
    step_sz = 50
    filt_steps = np.zeros(n_eps)
   
    for i in np.arange(0,n_eps):
        filt_steps[i] = np.mean(t_n_steps[i:i+step_sz])

    # Plot reward function
    fig = plt.figure()
#    plt.hold(True)
    x_vec = np.arange(0,n_eps)

    m1 = np.mean(filt_steps)*np.ones(len(x_vec))
    line1, = plt.plot(x_vec,filt_steps, 'r', linewidth=3.0)    
    plt.plot(x_vec,m1,'r--', linewidth=3.0)

    axes = plt.gca()
    axes.set_xlim([0,n_eps])
#    axes.set_ylim([-400, 0])
#    plt.legend([line1],['SARSA', 'Q-Learning'],loc=4)
    plt.xlabel('No. of Episodes')
    plt.ylabel('Number of Steps in each episode')
#    plt.text(5., -25.,'\eps = 0.2', fontsize=18)
    rcParams.update({'font.size': 18})
    plt.show()