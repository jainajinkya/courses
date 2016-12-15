#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 14:28:02 2016

@author: labmember
"""

from numpy import *
import matplotlib.pyplot as plt
#import matplotlib
#from matplotlib import cm
from matplotlib import rcParams
from scipy.signal import medfilt
#from tiles import *
import itertools

class env():
    def __init__(self,setReward):
        self.nCols = 14
        self.nRows = 12
        self.nStates = self.nCols*self.nRows
#        self.nObservations = self.nStates
        self.nActions = 4

        self.domain,self.walls = self.defineDomain()
        self.action = ['right', 'left', 'up', 'down']
#        self.observation_success_prob = 0.85
        self.transistion_success_prob = 1.0
        self.goal = [self.nRows-2,self.nCols-2]
        self.goalReward = setReward
#        self.transistionMatrix = array([self.trans_matrix(action,0) for action in range(self.nActions)])

        
    def defineDomain(self):
        bottom = [[0, i] for i in range(self.nCols)]
        top = [[self.nRows-1,i] for i in range(self.nCols)]
        right = [[i,self.nCols-1] for i in range(self.nRows)]
        left = [[i,0] for i in range(self.nRows)]
        middle = [[i,int(round(self.nCols/2))] for i in range(self.nRows)]
#        middle = []
        opening = [[int(round(self.nRows/2))-1,int(round(self.nCols/2))],[int(round(self.nRows/2)),int(round(self.nCols/2))]]

        walls = bottom + top + right + left + middle
        walls = [x for x in walls if x not in opening]

        domain = [[i,j] for i in range(self.nRows) for j in range(self.nCols) \
                  if [i,j] not in walls]

        return domain, walls
      
        
    def stateUpdate(self,state,action):
        # State here should come in [i,j] and not in just the state number
        new_state = [state[0],state[1]]

        trans_prob = self.transistion_success_prob
       
        p_idx = random.rand()
        if p_idx > trans_prob:
            action = random.randint(0,self.nActions)

        if action == 0:
            new_state[1] = state[1] + 1
            if new_state in self.walls:
                new_state[1] = state[1]

        if action == 1:
            new_state[1] = state[1] - 1
            if new_state in self.walls:
                new_state[1] = state[1]

        if action == 2:
            new_state[0] = state[0] + 1
            if new_state in self.walls:
                new_state[0] = state[0]

        if action == 3:
            new_state[0] = state[0] - 1
            if new_state in self.walls:
                new_state[0] = state[0]       

        return new_state
        

    def rewardFunction(self,state):
        if state == self.goal:
#            print "goal reached"
            return self.goalReward
        else:
#            print "state =", [state/self.nCols,state%self.nCols]
            return -1

            
class basis_fn():
    def __init__(self,nStates):
        self.lengthPhi = nStates
        
    def stateExcitation(self,state,world):
        ## Pass index of the state
        state_idx = state[0]*world.nCols + state[1]
        phi = zeros(self.lengthPhi)
        phi[state_idx] = 1.
        
        return phi        


class policy():
    def e_greedy(self, weights,phi,n_eps):
        # eps = 0.1*exp(-n_eps/2)
        if n_eps>0:
            eps = 0.
        else: eps = 0.1

        p = random.rand()
        if p <= 1-eps: 
            action = argmax(dot(weights,phi))     
        else:
            action = random.randint(0,4)
            
        return action


if __name__ == '__main__':
    
    alpha = 0.5
    gamma = 1.0
    labda =  0.9 
    
    ctrl = policy() 
    t_n_eps = []
    fname2 = 'rewardData/t_n_eps_data' + str(alpha) + '.npy'
    f2 = open(fname2,'w')
    
    
    for re in range(0,50,5):
        setReward = re
        world = env(setReward)
        basis = basis_fn(world.nStates)
        run = True
        
        nEps = 0
        n_eps = 50000
        
        n_actions = world.nActions
        
        f_name = 'episodicData/goalReward_' + str(re) + '.txt'
        f = open(f_name, 'w') 
    
#        f.write("n_eps = ")
#        f.write(str(n_eps))
        f.write("\t alpha = ")
        f.write(str(alpha))
        f.write("\t gamma = ")
        f.write(str(gamma))
        f.write("\t labda = ")
        f.write(str(labda))
#       f.write("\t f_order = ")
#       f.write(str(f_order))
        f.write("\t world size = ")
        f.write(str(world.nRows))
        f.write(",")
        f.write(str(world.nCols))
        f.write("\n")
   
        cur_state = [1,1]
        t_state = world.goal
    
        phi = basis.stateExcitation(cur_state,world)
        weights = zeros((n_actions,basis.lengthPhi))
    
        t_n_steps = []
        total_reward = []
        aggregate_steps = 0.0
   
        while(run):
            cur_state = [1,1]
            phi = basis.stateExcitation(cur_state,world)
            action = ctrl.e_greedy(weights,phi,nEps+1)
            
            e = zeros(basis.lengthPhi)
            Q_old = 0
        
            n_steps = 0
            counter = 0
            reward_sarsa = 0
            
            nEps += 1
  
            while(cur_state != t_state):
                new_state = world.stateUpdate(cur_state,action)
                reward = world.rewardFunction(new_state)
                new_phi = basis.stateExcitation(new_state,world)
                new_action = ctrl.e_greedy(weights,new_phi,nEps+1)

                ## Hack to avoid getting stuck in an infinite loop
                if(n_steps%2 == 0):
                    state_older = cur_state

                if(new_state == state_older):
                        counter = counter + 1

                if(counter > 20):
                    action = ctrl.e_greedy(weights,phi,0)
                    new_state = world.stateUpdate(cur_state,action)
                    reward = world.rewardFunction(new_state)
                    new_phi = basis.stateExcitation(new_state,world)
                    new_action = ctrl.e_greedy(weights,new_phi,nEps+1)
                    counter = 0
                #############################################

                Q = dot(weights[action,:],phi)
                new_Q = dot(weights[new_action,:],new_phi)

                delta = reward + gamma*new_Q - Q
            
                # Elgibility trace update
                # Thresholding the feature vector to            
                e = gamma*labda*e + (1 - alpha*gamma*labda*dot(e,phi))*phi
                weights[action,:] = weights[action,:] + alpha*(delta + Q - Q_old)*e - alpha*(Q-Q_old)*phi

                phi = new_phi
                cur_state = new_state
                action = new_action
                Q_old = new_Q
            
                n_steps = n_steps + 1
                reward_sarsa = reward_sarsa + reward

#            print "n_steps =", n_steps
            t_n_steps.extend([n_steps])
            total_reward.extend([reward_sarsa])
            
            if nEps > 50:
                aggregate_steps = mean(t_n_steps[nEps-51:nEps-1])
            else:
                aggregate_steps = mean(t_n_steps[0:nEps])
            
            if aggregate_steps in [20.0, 22.0, 24.0]:
                run = False
                
            
         ### Plots
        fig = plt.figure()
        x_vec = arange(0,nEps)
        filtered_rew = medfilt(total_reward, 9)
        m1 = mean(filtered_rew)*ones(len(x_vec))
        line1, = plt.plot(x_vec,filtered_rew, 'r', linewidth=3.0)
        plt.plot(x_vec,m1,'r--', linewidth=3.0)
        axes = plt.gca()
        axes.set_xlim([0,nEps])
        plt.xlabel('No. of Episodes')
        plt.ylabel('Total reward in each episode')
        rcParams.update({'font.size': 18})
        
        filename = 'episodicData/Total_reward_plot_goalReward_' + str(re) + '.png'
        plt.savefig(filename, bbox_inches='tight')
        
        t_n_eps.extend([nEps])
        
    print "t_n_steps = ", t_n_eps
    
  
    ### Plots
#    fig = plt.figure()
#    x_vec2 = arange(0,50,5)
#    plt.plot(x_vec2,t_n_eps,'b', linewidth=2.0)
#    axes = plt.gca()
#    axes.set_xlim([0,50])
#    plt.xlabel('Reward at Goal')
#    plt.ylabel('Number of Steps taken')
#    plt.text(35., 1500.,'alpha = ' + str(alpha), fontsize=18)
#    rcParams.update({'font.size': 18})
#    filename2 = 'rewardData/Steps_vs_goalReward_alpha_'+ str(alpha) + '.png'
#    plt.savefig(filename2, bbox_inches='tight')
    plt.close('all')
    save(f2,t_n_eps)
    f2.close()
#    plt.show()
