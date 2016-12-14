#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 22:18:29 2016

@author: doc
"""

import numpy as np
import numpy.random as rand
import matplotlib.pyplot as plt
# from scipy.siself.gnal import medfilt
#from tiles import *
import itertools
from matplotlib import rcParams
from scipy.signal import medfilt

## Simulator
class env():
    def __init__(self):
        self.m1 = 1
        self.m2 = 1
        self.I1 = 1
        self.I2 = 1
        self.l1 = 1
        self.l2 = 1
        self.lc1 = 0.5
        self.lc2 = 0.5
        self.g = 9.8
        self.dt = 0.05
        
        self.nActions = 3


    def simulator(self,cur_State,action):
        tau = action*1.0
        [th1,th1_Dot, th2, th2_Dot] = cur_State


        d2 = self.m2*(self.lc2**2 + self.l1*self.lc2*np.cos(th2)) + self.I2
        d1 = self.m1*self.lc1**2 + self.m2*(self.l1**2 + self.lc2**2 + 2*self.l1*self.lc2*np.cos(th2)) + self.I1 + self.I2

        phi2 = self.m2*self.lc2*self.g*np.cos(th1 + th2 - np.pi/2)
        phi1 = -self.m2*self.l1*self.lc2*th2_Dot**2*np.sin(th2) - 2*self.m2*self.l1*self.lc2*th2_Dot*th1_Dot*np.sin(th2) + (self.m1*self.lc1 + self.m2*self.l1)*self.g*np.cos(th1-np.pi/2) + phi2

        th2_Ddot = (tau + (d2/d1)*phi1 - self.m2*self.l1*self.lc2*th1_Dot**2*np.sin(th2) - phi2)/(self.m2*(self.lc2**2) + self.I2 - d2**2/d1)
        th1_Ddot = -(d2*th2_Ddot + phi1)/d1

        ## New state
        th1_Dot_new = th1_Dot + th1_Ddot*self.dt
        th2_Dot_new = th2_Dot + th2_Ddot*self.dt
        th1_new     = th1 + th1_Dot*self.dt
        th2_new     = th2 + th2_Dot*self.dt

        return [th1_new, th1_Dot_new, th2_new, th2_Dot_new]


    def r_function(self):
        return -1.


## Function Approximation
class basis_fn():
    def fourier(self,cur_state,f_order):
        n_dim = len(cur_state)
        x = np.arange(0,f_order+1)
        c = [p for p in itertools.product(x, repeat=n_dim)]
        phi = np.zeros(len(c))

        scaled_state = [cur_state[0]/(2*np.pi), cur_state[1]/(4*np.pi), cur_state[2]/(2*np.pi), cur_state[3]/(4*np.pi)]

        for i in range(len(c)):
             phi[i] = np.cos(np.pi*np.sum([c[i][j]*scaled_state[j] for j in range(n_dim)]))
#            phi[i] = np.cos(np.pi*np.sum([c[i][j]*cur_state[j] for j in range(n_dim)]))


        return phi

    def alpha_weights(self,f_order,alpha):
        x = np.arange(0,f_order+1)
        c = [p for p in itertools.product(x, repeat=4)]

        alpha2 = alpha*np.ones(len(c))

        for i in range(len(c)-1):
            # print "i =", np.sum(c[i+1])
            alpha2[i+1] = alpha/(np.sqrt(np.sum(np.square(c[i+1]))))

        return alpha2
        

### SARSA
class algorithm():
    def __init__(self,f_order,init_state,world,basis):
        self.gamma = 1.
        self.lamda = 0.9
        self.f_order = f_order
        self.alpha = 2e-4
        self.alpha2 = basis.alpha_weights(self.f_order,self.alpha)      
        self.weights = np.zeros((world.nActions,len(self.alpha2)))
    
    def e_greedy(self, phi,n_eps):
        p = rand.rand()
        if n_eps<0:
            eps = 0.0
        else:
            eps= 0.0
            
        if p<eps:
            action = rand.randint(-1,2,1)
        else:
            action = np.array([np.argmax(np.dot(self.weights,phi)) - 1])

#            if idx == 0:
#                action = -1
#            elif idx == 1:
#                action = 0
#            else: action = 1

        return action

        
    def sarsa(self,n_eps,init_state,world,basis,nSkipSteps):
        t_n_steps = np.zeros(n_eps)
        
        cur_state = init_state
        phi = basis.fourier(cur_state,self.f_order)
        
        Q = np.zeros(world.nActions)
        
        ## Running the episodes
        for ep in range (n_eps):
            print "Starting Episode Number :", ep
            cur_state = init_state
            phi = basis.fourier(cur_state,self.f_order)
            action = self.e_greedy(phi,ep)
        
            tip = [world.l1*np.cos(cur_state[0]-np.pi/2) + world.l2*np.cos(cur_state[0]\
               - np.pi/2 + cur_state[2]), world.l1*np.sin(cur_state[0]- np.pi/2) \
               + world.l2*np.sin(cur_state[0] - np.pi/2 + cur_state[2])]

            n_steps = 0
            count_action = 0
            
            Q_old = 0
            eligibility = np.zeros(len(phi))
        
            while(tip[1] < world.l1):
                if count_action == nSkipSteps:
                    action = self.e_greedy(phi,ep)
                    count_action = 0
                else:
                    action = 0
                    count_action = count_action + 1

                new_state = world.simulator(cur_state,action)
                new_phi = basis.fourier(new_state,self.f_order)
                
                reward = world.r_function()
                new_action = self.e_greedy(new_phi,ep)
    
                Q = np.dot(self.weights[action+1,:],phi)
                new_Q = np.dot(self.weights[new_action+1,:],new_phi)
    
                eligibility  = self.gamma*self.lamda*eligibility\
                            + (1 - self.gamma*self.lamda*np.dot(eligibility,np.multiply(self.alpha2,phi)))*phi
                
                delta = reward + self.gamma*new_Q - Q
               
                self.weights[action+1,:] = self.weights[action+1,:]\
                         + (delta + Q - Q_old)*np.multiply(self.alpha2,eligibility)\
                         - (Q - Q_old)*np.multiply(self.alpha2,phi) 
    
                cur_state = new_state
                phi = new_phi
                action = new_action
                Q_old = new_Q
    
                tip = [world.l1*np.cos(cur_state[0]-np.pi/2) + world.l2*np.cos(cur_state[0]\
                       - np.pi/2 + cur_state[2]), world.l1*np.sin(cur_state[0]- np.pi/2)\
                       + world.l2*np.sin(cur_state[0] - np.pi/2 + cur_state[2])]
    
                n_steps = n_steps + 1
#                print "n_steps =", n_steps

            print "n_steps =", n_steps
#            print "\n"
#            print "tip = ", tip
            # print "update in weights = ", weights - old_weights
            t_n_steps[ep] = n_steps
        
        return t_n_steps
            
    
    def finalPlan(self,initState,world,basis):
        cur_state = initState
        tip = [world.l1*np.cos(cur_state[0]-np.pi/2) + world.l2*np.cos(cur_state[0] - np.pi/2 + cur_state[2]),\
           world.l1*np.sin(cur_state[0]- np.pi/2) + world.l2*np.sin(cur_state[0] - np.pi/2 + cur_state[2])]

        
        f = open('acrobot_data.txt', 'w')
        
        counter2 = 0
    
        while(tip[1] < world.l1 and counter2<1000):
            phi = basis.fourier(cur_state,self.f_order)
            action = np.array([np.argmax(np.dot(self.weights,phi))-1])
            new_state = world.simulator(cur_state,action)
            cur_state = new_state
    
            pt1 = [world.l1*np.cos(cur_state[0]-np.pi/2) , world.l1*np.sin(cur_state[0]-np.pi/2)]
            tip = [world.l1*np.cos(cur_state[0]-np.pi/2) + world.l2*np.cos(cur_state[0] \
                   - np.pi/2 + cur_state[2]), world.l1*np.sin(cur_state[0]- np.pi/2) \
                   + world.l2*np.sin(cur_state[0] - np.pi/2 + cur_state[2])]
    
#            print "tip[1] - world.l1 = ", tip[1] - world.l1
            counter2 = counter2 + 1
    #        print "counter2 = ", counter2
            f.write(str(pt1[0]))
            f.write('\t')
            f.write(str(pt1[1]))
            f.write('\t')
            f.write(str(tip[0]))
            f.write('\t')
            f.write(str(tip[1]))
            f.write('\n') 
        
        f.close()
        return
        
        
if __name__ == '__main__':
    world = env()
    basis = basis_fn()

    f_order = 3
    initState = np.array([0., 0., 0., 0.])
    n_eps = 500
    nSkipSteps = 4
    
#    fig = plt.figure()
    x_vec = np.arange(0,n_eps)
    clrs = ['b','g','r','k','c','m']
    handle = []
    for i in range(3,nSkipSteps):
    	algo = algorithm(f_order,initState,world,basis)
    
    	t_n_steps = np.zeros((n_eps,nSkipSteps))
        print "#############################"
        print "Effort Skip Steps =", i
        t_n_steps[:,i]  = algo.sarsa(n_eps,initState,world,basis,i)
        algo.finalPlan(initState,world,basis)
        
        f_name = 'acrobot_step_' + str(i) + '.npy'
        f_steps = open(f_name, 'w')
        np.save(f_steps,t_n_steps)
        f_steps.close()
    
    ## Plot reward function #################################################
        step_sz = 10
        filt_steps = np.zeros(n_eps)
   
        for j in np.arange(0,n_eps):
            filt_steps[j] = np.mean(t_n_steps[j:j+step_sz,i])

        m1 = np.mean(filt_steps)*np.ones(len(x_vec))
        clr = clrs[i]
        line1, = plt.plot(x_vec,filt_steps,clr, linewidth=3.0)
        handle = handle + [line1,]
        plt.plot(x_vec,m1,clr+'--', linewidth=3.0)

    axes = plt.gca()
    axes.set_xlim([0,n_eps])
    axes.set_ylim([0,2500])
    plt.legend(handle,['0 Steps', '1 Step','2 Steps','3 Steps', '4 Steps'],loc=4)
    plt.xlabel('No. of Episodes')
    plt.ylabel('Number of Steps in each episode')
#    plt.text(5., -25.,'\eps = 0.2', fontsize=18)
    rcParams.update({'font.size': 18})
    plt.show()
    


###############################################################################
# Previous Work
###############################################################################
#    world = env()
#    ctrl = policy()
#    basis = basis_fn()
#
#    n_eps = 200
#    
#    f_steps = open('acrobot_steps.npy', 'w')
##    gamma = 1.
##    labda = 0.9
##    n_actions = 3
##    t_n_steps = np.zeros(n_eps)
#
#
#
#    f_order = 4
#    cur_state = [0., 0., 0., 0.]
#
#    algo = algorithm(f_order,cur_state,basis)
#    t_n_steps  = algo.sarsa(n_eps,cur_state,world,basis)
##    
##
##    phi = basis.fourier(cur_state,f_order)
##    weights = np.zeros((n_actions,len(phi)))
##    
##    alpha = 0.001
##    alpha2 = basis.alpha_weights(f_order,alpha)
###    t_state = [0,3]
##
###    Q_old = 100.
##    Q = np.zeros(n_actions)
###    new_Q = np.zeros(n_actions)
##
##    for ep in range (n_eps):
##        cur_state = [0., 0., 0., 0.]
##        phi = basis.fourier(cur_state,f_order)
##        
##        action = ctrl.e_greedy(weights,phi,ep)
##        n_steps = 0
##        
##        tip = [world.l1*np.cos(cur_state[0]-np.pi/2) + world.l2*np.cos(cur_state[0]\
##               - np.pi/2 + cur_state[2]), world.l1*np.sin(cur_state[0]- np.pi/2) \
##               + world.l2*np.sin(cur_state[0] - np.pi/2 + cur_state[2])]
##
##        count_action = 0
##
###        e_vector = np.zeros(len(phi))
###        Q_old = 0
##        
##        while(tip[1] < world.l1):
##           if count_action == 3:
##               action = ctrl.e_greedy(weights,phi,ep)
##               count_action = 0
##           else:
##               action = 0
##               count_action = count_action + 1
##
##           new_state = world.simulator(cur_state,action)
##           reward = world.r_function(new_state)
##           new_phi = basis.fourier(new_state,f_order)
##           new_action = ctrl.e_greedy(weights,new_phi,ep)
##
##           Q = np.dot(weights[action+1,:],phi)
##           new_Q = np.dot(weights[new_action+1,:],new_phi)
##
##           delta = reward + gamma*new_Q - Q
##           
###           e_vector = gamma*labda*e_vector + (1 - gamma*labda*np.dot(e_vector,np.multiply(alpha2,phi)))*phi
##
##           weights[action+1,:] = weights[action+1,:] + alpha*delta*phi
###           weights[action+1,:] = weights[action+1,:] + (delta + Q - Q_old)*np.multiply(alpha2,e_vector)\
###                                 - (Q-Q_old)*np.multiply(alpha2,phi) 
##           
##           phi = new_phi
##           cur_state = new_state
##           action = new_action
##           Q_old = new_Q
##
##           tip = [world.l1*np.cos(cur_state[0]-np.pi/2) + world.l2*np.cos(cur_state[0] - np.pi/2 + cur_state[2]), world.l1*np.sin(cur_state[0]- np.pi/2) + world.l2*np.sin(cur_state[0] - np.pi/2 + cur_state[2])]
##
##           n_steps = n_steps + 1
###           print "n_steps =", n_steps
##
##        print "ep=", ep, "n_steps =", n_steps
##        print "tip = ", tip
##       # print "update in weights = ", weights - old_weights
##        t_n_steps[ep] = n_steps
#    
#    np.save(f_steps,t_n_steps)
#    f_steps.close()
#
# ## Final Policy
#    cur_state = initState
#    tip = [world.l1*np.cos(cur_state[0]-np.pi/2) + world.l2*np.cos(cur_state[0] - np.pi/2 + cur_state[2]),\
#           world.l1*np.sin(cur_state[0]- np.pi/2) + world.l2*np.sin(cur_state[0] - np.pi/2 + cur_state[2])]
#
#    f = open('acrobot_data.txt', 'w')
#    f2 = open('data_n_steps.txt','w')
#    f2.write(str(t_n_steps))
#    f2.write('\n')
#    f2.close()
#    
#    counter2 = 0
#
#    while(tip[1] < world.l1 and counter2<1000):
#        phi = basis.fourier(cur_state,f_order)
#        action = np.argmax(np.dot(weights,phi))-1
#        new_state = world.simulator(cur_state,action)
#        cur_state = new_state
#
#        pt1 = [world.l1*np.cos(cur_state[0]-np.pi/2) , world.l1*np.sin(cur_state[0]-np.pi/2)]
#        tip = [world.l1*np.cos(cur_state[0]-np.pi/2) + world.l2*np.cos(cur_state[0] - np.pi/2 + cur_state[2]), world.l1*np.sin(cur_state[0]- np.pi/2) + world.l2*np.sin(cur_state[0] - np.pi/2 + cur_state[2])]
#
#        print "tip[1] - world.l1 = ", tip[1] - world.l1
#        counter2 = counter2 + 1
##        print "counter2 = ", counter2
#        f.write(str(pt1[0]))
#        f.write('\t')
#        f.write(str(pt1[1]))
#        f.write('\t')
#        f.write(str(tip[0]))
#        f.write('\t')
#        f.write(str(tip[1]))
#        f.write('\n') 
  