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
    ex_number = 61
    filename = 'data_hybrid/data' + str(ex_number) + '.txt'
    f = open(filename,'r')
    fContents = f.read()
    
    # Parsing the file
    p1 = fContents.split('\n')
    
    dummy_ep = p1[0].split('\t')
    ep = dummy_ep[0].split()
    n_eps = ep[4]
    n_eps = int(n_eps)

    # Parsing the Value function Update
#    p2 = p1[2].split('[')
#    p2 = p2[0].split(']')
#    valueUpdate = p2[0].split()
#    valueUpdate = [float(i) for i in valueUpdate]
#                   
#    # Parsing the Value function Update
#    p2 = p1[1].split('[')
#    p2 = p2[1].split(']')
#    totalReward = p2[0].split()
#    totalReward = [float(i) for i in totalReward];
#    
    totalReward = [ -1.49049913e+00,  -1.69724600e+00,  4.46504984e+02,  -2.44095231e+00,
   4.18707660e+01,   8.64606160e+02,   7.82022520e+02,   1.13158976e+03,
   4.83136426e+02,   1.25252530e+03,   1.37901027e+03,  1.46900476e+03,
   1.50341462e+03,   1.85622429e+03,   1.61342466e+03,   1.32477802e+03,
   1.62701978e+03,   1.89007319e+03,   1.33729458e+03,   1.62816499e+03]
        # Plot reward function
    valueUpdate = [ -5.20686439e+00,  -4.01762068e+00,   2.72095363e+02,  -4.90637860e+00,
   3.78319663e+01,   2.34028692e+02,   9.78068220e+01,   3.45757466e+01,
   5.27144460e+01,   7.23817905e+01,   1.73609238e+01,   6.59383745e-01,
   1.77137050e+01,   3.98598152e+01,   9.65943975e+00,   6.74888089e+00,
   3.29599470e-01,   6.12406832e-02,   2.78893419e-01,   1.92217153e-02]
    
#    n_eps = 20
    fig = plt.figure()
    x_vec = np.arange(0,n_eps)
#    filtered_rew = savgol_filter(valueUpdate, 5, 1)
    # filtered_rew_q_learn = savgol_filter(total_reward_q_learn, 5, 1)

    filtered_rew = medfilt(valueUpdate, 5)
    filtered_totalReward = medfilt(totalReward, 5)

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
    axes.set_ylim([0,2500])
    plt.legend([line1, line2],['Update in Value Functions', 'Total Returns obtained'],loc=2)
    plt.xlabel('No. of Episodes')
    plt.ylabel('Reward in each episode')
    plt.text(5., -25.,'\eps = 0.2', fontsize=18)
    rcParams.update({'font.size': 18})
    plt.show()
