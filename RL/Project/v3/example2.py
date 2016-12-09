#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 15:03:18 2016

@author: Ajinkya
"""

from numpy import *
from pypr.clustering import *
from sklearn import mixture
import random as randomize
#from TigerProblem import TigerProblem

class  POMDP:
    def __init__(self):
        self.nCols = 5
        self.nRows = 4
        self.nStates = self.nCols*self.nRows
        self.nObservations = self.nStates
        self.nActions = 4
        
        self.domain,self.walls = self.defineDomain()
#        self.state = ['tiger_left', 'tiger_right']
        self.action = ['right', 'left', 'up', 'down']
#        self.obs = ['tiger_left', 'tiger_right']
        self.observation_success_prob = 1.0
        self.transistion_success_prob = 1.0
        self.goal = [3,2]
        self.transistionMatrix = array([self.trans_matrix(action) for action in range(self.nActions)])
    
    def defineDomain(self):
        bottom = [[0,i] for i in range(self.nCols)]
        top = [[self.nRows-1,i] for i in range(self.nCols)]
        right = [[i,self.nCols-1] for i in range(self.nRows)]
        left = [[i,0] for i in range(self.nRows)]
#        middle = [[i,round(self.nCols/2)] for i in range(self.nRows)]
        middle = []
#        opening = [3,round(self.nCols/2)]
        
        walls = bottom + top + right + left + middle
#        walls.remove(opening)
        
        domain = [[i,j] for i in range(self.nRows) for j in range(self.nCols) \
                  if [i,j] not in walls] 

        return domain, walls
        
    def trans_matrix(self,action):
        mat = zeros([self.nStates,self.nStates])
        if action == 0:
            for i in range(self.nStates):
                r = i/self.nCols
                c = i%self.nCols
                
                if [r,c] in self.domain:
                    if [r,c+1] in self.domain:
                        mat[i,i+1] = self.transistion_success_prob # Right
                    else:
                        mat[i,i] = 1.0
                        continue
                        
                    if([r+1,c] in self.domain and [r-1,c] in self.domain): # Top
                        mat[i,i + self.nCols] = (1.-self.transistion_success_prob)/2. #0.1
                        mat[i,i - self.nCols] = (1.-self.transistion_success_prob)/2. #0.1               
                    elif([r+1,c] in self.domain):
                        mat[i,i + self.nCols] = 1.-self.transistion_success_prob #0.2
                    else:
                        mat[i,i - self.nCols] = 1.-self.transistion_success_prob #0.2
 
       
        elif action == 1:
            for i in range(self.nStates):
                r = i/self.nCols
                c = i%self.nCols
                
                if [r,c] in self.domain:
                    if [r,c-1] in self.domain:
                        mat[i,i-1] = self.transistion_success_prob #0.8 # Left
                    else:
                        mat[i,i] = 1.0
                        continue
                        
                    if([r+1,c] in self.domain and [r-1,c] in self.domain): # Top
                        mat[i,i + self.nCols] = (1.-self.transistion_success_prob)/2. #0.1
                        mat[i,i - self.nCols] = (1.-self.transistion_success_prob)/2. #0.1               
                    elif([r+1,c] in self.domain):
                        mat[i,i + self.nCols] = 1.-self.transistion_success_prob #0.2
                    else:
                        mat[i,i - self.nCols] = 1.-self.transistion_success_prob #0.2
        
        elif action == 2:
            for i in range(self.nStates):
                r = i/self.nCols
                c = i%self.nCols
                
                if [r,c] in self.domain:
                    if [r+1,c] in self.domain:
                        mat[i,i+self.nCols] = self.transistion_success_prob #0.8 # Up
                    else:
                        mat[i,i] = 1.0
                        continue
                        
                    if([r,c+1] in self.domain and [r,c-1] in self.domain): # Top
                        mat[i,i -1] = (1.-self.transistion_success_prob)/2. #0.1
                        mat[i,i +1] = (1.-self.transistion_success_prob)/2. #0.1               
                    elif([r,c+1] in self.domain):
                        mat[i,i + 1] = 1.-self.transistion_success_prob #0.2
                    else:
                        mat[i,i - 1] = 1.-self.transistion_success_prob #0.2
    
        elif action == 3:
            for i in range(self.nStates):
                r = i/self.nCols
                c = i%self.nCols
                
                if [r,c] in self.domain:
                    if [r-1,c] in self.domain:
                        mat[i,i-self.nCols] = self.transistion_success_prob #0.8 # Down
                    else:
                        mat[i,i] = 1.0
                        continue
                        
                    if([r,c+1] in self.domain and [r,c-1] in self.domain): 
                        mat[i,i -1] = (1.-self.transistion_success_prob)/2. #0.1
                        mat[i,i +1] = (1.-self.transistion_success_prob)/2. #0.1               
                    elif([r,c+1] in self.domain):
                        mat[i,i + 1] = 1.-self.transistion_success_prob #0.2
                    else:
                        mat[i,i - 1] = 1.-self.transistion_success_prob #0.2

        return mat
        
                   
    def stateUpdate(self,state,action):
        new_state = [state[0],state[1]]
        
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
    
    
    def stateTransitionProb(self,new_state,belief,action):
#        self.transistionMatrix = self.trans_matrix(action)
        new_state_idx = new_state[0]*self.nCols + new_state[1]
        return dot(self.transistionMatrix[action,:,new_state_idx],belief)
        
        
    def stateTransitionProbState(self,new_state,state,action):
#        self.transistionMatrix = self.trans_matrix(action)
        new_state_idx = new_state[0]*self.nCols + new_state[1]
        b = zeros([self.nStates])
        b[state[0]*self.nCols + state[1]] = 1.0
        return dot(self.transistionMatrix[action,:,new_state_idx],b)
        

    def rewardFunction(self,state):
        if [state/self.nCols,state%self.nCols] == self.goal:
            print "goal reached"
            return 100
        else:
            print "state =", [state/self.nCols,state%self.nCols]
            return -1
            
        
    def observations(self,state,action):
        rand_num = random.rand()
        if rand_num < self.observation_success_prob:
            return state  
        else:
            return self.get_other_state(state)
    
    def observationProb(self,state,obs):
        if obs == state:
            return self.observation_success_prob
        else: 
            return 1 - self.observation_success_prob
            
            
    def get_other_state(self,state):
        """
        A useful util that returns the state other than self.cur_state
        """
        s = randomize.choice(self.domain)
        while(s == state):
            s = randomize.choice(self.domain)                      
        return s       
            
            
class algorithm():
    def __init__(self,nTraj,nT,pomdp,init_alpha,init_actions_alpha):
        self.nTraj = nTraj
        self.nT = nT       
        self.K = 2
        self.gamma = 1.0
        self.pomdp = pomdp
        self.B = zeros([nTraj,nT,self.pomdp.nStates])
        self.valueFn = init_alpha
        self.actions_alpha = init_actions_alpha
        self.nAlphas = 1
        self.run = True
        
#        gmix = mixture.GMM(n_components=self.K, covariance_type='full')
#        gmix.fit(self.valueFn)
        self.valueFn_gmm = zeros(self.K)

    def generateBeliefSet(self,b_init):
        for i in range(self.nTraj):
            b = b_init # This can be a unimodal Guassian; Let's try with discrete set of Probabilities
#            samples = 1
            
            ############ Remember to change the inputs to the function so as to meet the actual inputs to the GMM
#            state = gmm.sample_gaussian_mixture(b[0],b[1], samples=samples)
            state = randomize.choice(self.pomdp.domain)  # randomly choose a state out of nStates
            b = b_init = zeros(pomdp.nStates)
            b[state[0]*self.pomdp.nCols + state[1]] = 1.0
            
            for j in range(self.nT):
                # Take a random action
                action = random.randint(0,self.pomdp.nActions) 

                # Draw a sample from the new probability distribution
                new_state = self.pomdp.stateUpdate(state,action)

                # Draw a sample from the observation probabilities
                obs = self.pomdp.observations(new_state,action)

                # update the belief
                b_a_z = self.beliefUpdate(b,action,obs)

                # Add belief to the belief Set
                self.B[i,j,:] = b_a_z ## You need to store the belief State, action and observations as well

                # Update for the new loop
                b = b_a_z
                state = new_state

    def beliefUpdate(self,b,action,obs):
        new_b = zeros(self.pomdp.nStates)
        for i in range(self.pomdp.nStates):
            r = i/self.pomdp.nCols
            c = i%self.pomdp.nCols
            state_prob = self.pomdp.stateTransitionProb([r,c],b,action)
            obs_prob = self.pomdp.observationProb([r,c],obs)
#            print "r,c =", r,c
#            print "state_prob = ", state_prob
#            print "obs_prob = ", obs_prob
            new_b[i] = state_prob*obs_prob
    
        new_b = new_b/sum(new_b)       

        return new_b      

        
    def valueIteration(self):
        k = 1
#        r = 1
        while(k<10): # Check if this condition for checking is correct or not
            # Randomly choose a trajectory
            r = random.randint(0,self.nTraj)
            
            k = k+1
            for t in range(self.nT-1,-1,-1):
                old_value = self.beliefValue(self.B[r,t])

                # Value function backup
                new_alpha, action_alpha, new_value = self.valueFnBackup(self.B[r,t])

                # If increase in the projected value, add the new alpha vector
#                new_valueFn = self.beliefValueWithNewAlpha(new_alpha,(pomdp.B_set[r,t]))

                if(new_value > old_value):
                    self.valueFn = vstack((self.valueFn, new_alpha)) # Check how this addition should be done
                    self.actions_alpha = vstack((self.actions_alpha,action_alpha))
                    self.nAlphas = self.nAlphas + 1
                    
                    # projecting to smaller representation: fit a K parameter GMM
                    # Check if direct implemetation like this works or do we need to generate samples before we can fit
                    # K-means gaussians
#                    gmix = mixture.GMM(n_components=self.K, covariance_type='full')
#                    gmix.fit(self.valueFn)
#                    self.valueFn_gmm = gmix.means_


    def beliefValue(self,b):
        if self.nAlphas == 1:
            return dot(self.valueFn[:],b)
        else:
            return max([dot(self.valueFn[j,:],b) for j in range(self.nAlphas)])
    
    def valueFnBackup(self,b):
        new_alpha = zeros(self.pomdp.nStates)
        new_V = max([dot(self.max_a_R(act),b) + dot(self.max_alpha_azj(act),b) for act in range(self.pomdp.nActions)])
        opt_action = argmax([dot(self.max_a_R(act),b) + dot(self.max_alpha_azj(act),b) for act in range(self.pomdp.nActions)])
        new_alpha = add(self.max_a_R(opt_action),self.max_alpha_azj(opt_action))
        return new_alpha, opt_action, new_V
        
    def max_a_R(self,action):
        return [self.pomdp.rewardFunction(state) for state in range(self.pomdp.nStates)]
        
    def max_alpha_azj(self,action):
        if self.nAlphas == 1:
            return [self.gamma*sum([max([sum([self.valueFn[s_dash]\
            *self.pomdp.observationProb(s_dash,obs)\
            *self.pomdp.stateTransitionProbState([s_dash/self.pomdp.nCols, s_dash%self.pomdp.nCols],[state/self.pomdp.nCols, state%self.pomdp.nCols],action) \
            for s_dash in range(self.pomdp.nStates)]) for j in range(self.nAlphas)]) \
            for obs in range(self.pomdp.nObservations)]) for state in range(self.pomdp.nStates)]
        else:
            return [self.gamma*sum([max([sum([self.valueFn[j][s_dash]\
            *self.pomdp.observationProb(s_dash,obs)\
            *self.pomdp.stateTransitionProbState([s_dash/self.pomdp.nCols, s_dash%self.pomdp.nCols],[state/self.pomdp.nCols, state%self.pomdp.nCols],action) \
            for s_dash in range(self.pomdp.nStates)]) for j in range(self.nAlphas)]) \
            for obs in range(self.pomdp.nObservations)]) for state in range(self.pomdp.nStates)]


    def plan(self,b_init,n_eps):        
        for i in range(n_eps):
            print "Starting New Episode #########################"
            state = random.randint(self.pomdp.nStates)
            print "state =", self.pomdp.state[state]
            belief = b_init
            self.run = True
          
            while(self.run):
                print "Current belief = ", belief          
                alpha_max = argmax([dot(self.valueFn[j],belief) for j in range(self.nAlphas)])
                action = self.actions_alpha[alpha_max]
                print "action = ", self.pomdp.action[int(action)]
                obs = self.pomdp.observations(state,action)
                belief = self.beliefUpdate(belief,action,obs)
                print "observation = ", self.pomdp.obs[int(obs)]                
                print "\n"
                
                if int(action) != 0:
                    self.run = False         
            
        return


if __name__ == "__main__":
    nEps = 5
    nTraj = 10
    nT = 5
    pomdp = POMDP()
#    b_init = (1./pomdp.nStates)*ones(pomdp.nStates)
    b_init = zeros(pomdp.nStates)
    b_init[pomdp.nCols] = 1.0
#    b_init[pomdp.nCols+ 1] = 0.2
#    b_init[pomdp.nCols + pomdp.nCols] = 0.2
#    b_init[pomdp.nCols + pomdp.nCols + 1] = 0.1
    alpha_init = 0.*ones(pomdp.nStates)
    init_actions_alpha = array([0])
#    print pomdp.stateTransitionProbState([4,4],[4,5],0)
    pomdp_solve = algorithm(nTraj,nT,pomdp, alpha_init,init_actions_alpha)
    
    pomdp_solve.generateBeliefSet(b_init)
    pomdp_solve.valueIteration()
#    pomdp_solve.plan(b_init,nEps)
#    state = 0
#    pomdp_solve.run = True
#    new_state, beleif = pomdp_solve.plan(state,b_init)
#    print "Beleif Set = ", pomdp_solve.B
    print "Value function =", pomdp_solve.valueFn
#    print traj
#        environment.reset()
        





