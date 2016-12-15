#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 15:54:05 2016

@author: labmember
"""

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
        # eps = 0.1*np.exp(-n_eps/2)
        if n_eps>0:
            eps = 0.
        else: eps = 0.1

        p = rand.rand()
        if p <= 1-eps: 
            action = np.argmax(np.dot(weights,phi))     
        else:
            action = rand.randint(0,4)
            
        return action


if __name__ == '__main__':
    
    alpha = 0.5
    gamma = 1.0
    labda =  0.9
    n_eps = 100
    
    ctrl = policy() 
    t_n_eps = []
#    fname2 = 'rewardData/t_n_eps_data' + str(alpha) + '.npy'
#    f2 = open(fname2,'w')
    
    ## Plot
    fig = plt.figure(num=None, figsize=(18.5, 10.5), dpi=1000, facecolor='w', edgecolor='k')
    x_vec = np.arange(0,n_eps)
    clrs = ['b','g','r','k','c','m']
    handle = []
    clr_idx = 0

    for re in range(0,51,5):
        setReward = re
        world = env(setReward)
        basis = basis_fn(world.nStates)
        run = True
        
        nEps = 0

        n_actions = world.nActions
        cur_state = [1,1]
        t_state = world.goal
    
        phi = basis.stateExcitation(cur_state,world)
        weights = zeros((n_actions,basis.lengthPhi))
    
        t_n_steps = []
        total_reward = []
   
        while(run):
            cur_state = [1,1]
            phi = basis.stateExcitation(cur_state,world)
            action = ctrl.e_greedy(weights,phi,ep+1)
            
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
                new_action = ctrl.e_greedy(weights,new_phi,ep+1)

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
                    new_action = ctrl.e_greedy(weights,new_phi,ep+1)
                    counter = 0
                #############################################

                Q = dot(weights[action,:],phi)
                new_Q = dot(weights[new_action,:],new_phi)

                delta = reward + gamma*new_Q - Q    
                e = gamma*labda*e + (1 - alpha*gamma*labda*np.dot(e,phi))*phi
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
            
            if nEps > n_eps-1:
                run = False
                
            
#         ### Plots
        if re in [0,5,10,20,30,50]:
            filtered_rew = medfilt(total_reward, 5)
            clr = clrs[clr_idx]
            clr_idx += 1
            line1, = plt.plot(x_vec,filtered_rew, clr, linewidth=1.5)
            handle = handle + [line1,]
            
    axes = plt.gca()
    axes.set_xlim([0,n_eps])
    
    plt.xlabel('No. of Episodes')
    plt.ylabel('Total reward in each episode')
    plt.legend(handle,['Goal Reward = 0', 'Goal Reward = 5','Goal Reward = 10',\
                       'Goal Reward = 20', 'Goal Reward = 30','Goal Reward = 50'],loc=4)
    rcParams.update({'font.size': 18})
    
    filename = 'rewardData/Combined_Total_reward_plot_goalReward_alpha' + str(alpha) + '.png'
    plt.savefig(filename, bbox_inches='tight')
    plt.close('all')
    
    t_n_eps.extend([nEps])