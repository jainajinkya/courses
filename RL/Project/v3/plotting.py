#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 12:40:29 2016

@author: Ajinkya
"""
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import rcParams
#from scipy.signal import savgol_filter
from scipy.signal import medfilt
import numpy as np


if __name__ == '__main__':
    ex_number = 60
    filename = 'data_hybrid/data' + str(ex_number) + '.txt'
    f = open(filename,'r')
    fContents = f.read()
    
    # Parsing the file
    p1 = fContents.split('\n')
    
    dummy_ep = p1[0].split('\t')
    ep = dummy_ep[0].split()
    n_eps = ep[4]
    n_eps = [int(i) for i in n_eps]
    n_eps = n_eps[0]
    
    # Parsing the Value function Update
    p2 = p1[2].split('[')
    p2 = p2[0].split(']')
    valueUpdate = p2[0].split()
    valueUpdate = [float(i) for i in valueUpdate]
                   
    # Parsing the Value function Update
    p2 = p1[1].split('[')
    p2 = p2[1].split(']')
    totalReward = p2[0].split()
    totalReward = [float(i) for i in totalReward];
    
    totalReward = [-1.19617173e+00,   4.06896818e+02,   6.45701345e+01,   3.17456370e+01,
   9.80022124e+02,   6.99696077e+02,   6.46936640e+02,   6.47813228e+02,  8.50295015e+02,   1.52341800e+03]
        # Plot reward function
    valueUpdate = [-3.56214088,  232.96031641,   36.72854367,   26.93514793,  182.55894002,
  129.7505381,    24.87859067,   66.24123479,   63.06859072,  109.62783111]
    n_eps = 10
    fig = plt.figure()
    x_vec = np.arange(0,n_eps)
#    filtered_rew = savgol_filter(valueUpdate, 5, 1)
    # filtered_rew_q_learn = savgol_filter(total_reward_q_learn, 5, 1)

    filtered_rew = medfilt(valueUpdate, 3)
    filtered_totalReward = medfilt(totalReward, 3)

    m1 = np.mean(filtered_rew)*np.ones(len(x_vec))
    m2 = np.mean(filtered_totalReward)*np.ones(len(x_vec))
    # filtered_rew_q_learn = medfilt(total_reward_q_learn, 5)
    line1, = plt.plot(x_vec,filtered_rew, 'r', linewidth=3.0)
    # plt.plot(x_vec,filtered_rew_q_learn, 'b')
    # plt.plot(x_vec,filt_r_s, 'r')
    line2, = plt.plot(x_vec,filtered_totalReward, 'b', linewidth=3.0)
    
    plt.plot(x_vec,m1,'r--', linewidth=3.0)
    plt.plot(x_vec,m2,'b--', linewidth=3.0)

    axes = plt.gca()
    axes.set_xlim([0,n_eps])
    axes.set_ylim([0,1000])
    plt.legend([line1, line2],['Update in Value Functions', 'Total Returns obtained'],loc=2)
    plt.xlabel('No. of Episodes')
    plt.ylabel('Reward in each episode')
    plt.text(5., -25.,'\eps = 0.2', fontsize=18)
    rcParams.update({'font.size': 18})
    plt.show()
