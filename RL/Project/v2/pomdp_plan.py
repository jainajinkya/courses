#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 15:03:18 2016

@author: Ajinkya
"""

from numpy import *
from pypr.clustering import *
from sklearn import mixture
#from TigerProblem import TigerProblem

class  POMDP:
    def __init__(self):
        self.nStates = 2
        self.nObservations = 2
        self.nActions = 3
        
#        self.states = ['tiger_left', 'tiger_right']
#        self.actions = ['listen', 'open_left', 'open_right']
#        self.observations = ['tiger_left', 'tiger_right']
        self.listen_success_prob = 0.85
        
        self.cur_state = 0
    
    def stateUpdate(self,state,action):
        if action == 1:            
            if state == 0 :
                self.cur_state = None
                return round(random.uniform())   # None indicates end of episode
                                  # -100 because agent opened tiger door
            else:
                self.cur_state = None
                return round(random.uniform())     # None indicates end of episode
                                 # +10 because agent opened treasure door
        elif action == 2:
            if state == 1 :
                self.cur_state = None
                return round(random.uniform())  # None indicates end of episode
                                    # -100 because agent opened tiger door
            else:
                self.cur_state = None
                return round(random.uniform())    # None indicates end of episode
                                    # +10 because agent opened treasure door
        else:
            return state
    
    def stateTransitionProb(self,new_state,belief,action):
        if new_state == 0:
            if action == 0:
                p1 = 1.
                p2 = 0.
            else: 
                p1 = 0.5
                p2 = 0.5
        
        else: 
            if action == 0:
                p1 = 0.
                p2 = 1.0
                    
            else: 
                p1 = 0.5
                p2 = 0.5
        
        return dot(belief,array([p1,p2]))
        
    def stateTransitionProbState(self,new_state,state,action):
        if new_state == 0:
            if state == 0:
                if action == 0:
                    return  1.0
                else: 
                    return 0.5
            else: 
                if action == 0:
                    return  0.0
                else: 
                    return 0.5
        
        else: 
            if state == 0:
                if action == 0:
                    return  0.0
                else: 
                    return 0.5
            else: 
                if action == 0:
                    return  1.0
                else: 
                    return 0.5
        
 
    def get_other_state(self,state):
        """
        A useful util that returns the state other than self.cur_state
        """
        if state == 0:
            return 0
        else:
            return [st for st in range(1,3)
                    if st != state][0]        


    def rewardFunction(self,state,action):
        if action == 1:            
            if state == 0 :
                return -100   # None indicates end of episode
                                  # -100 because agent opened tiger door
            else:
                return 10     # None indicates end of episode
                                 # +10 because agent opened treasure door
        elif action == 2:
            if state == 1 :
                return -100   # None indicates end of episode
                                    # -100 because agent opened tiger door
            else :
                return 10     # None indicates end of episode
                                    # +10 because agent opened treasure door
        else:
            return -1
            
        
    def observations(self,state,action):
        if action == 0:
            rand_num = random.uniform()
            if rand_num < self.listen_success_prob:
                return state  
            else:
                return self.get_other_state(state)
        else:
            return round(random.uniform())
    
    def observationProb(self,state,obs):
        if state == 0:
            if obs == 0.:
                return 0.85
            else: 
                return 0.15
        else: 
            if obs == 0:
                return 0.15
            else: 
                return 0.85
                  

    def policy(self):
        idx = random.randint(pomdp.nActions)
        return idx
        
    
        

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
            state = random.randint(self.pomdp.nStates)  # randomly choose a state out of nStates
            
            for j in range(self.nT):
                action = self.pomdp.policy()

                # Draw a sample from the new probability distribution
                new_state = self.pomdp.stateUpdate(state,action)

                # Draw a sample from the observation probabilities
                obs = self.pomdp.observations(state,action)

                # update the belief
                b_a_z = self.beliefUpdate(b,action,obs)

                # Add belief to the belief Set
                self.B[i,j,:] = b_a_z

                # Update for the new loop
                b = b_a_z
                state = new_state

    def beliefUpdate(self,b,action,obs):
        new_b = zeros(self.pomdp.nStates)
        for i in range(self.pomdp.nStates):
            state_prob = self.pomdp.stateTransitionProb(i,b,action)
            obs_prob = self.pomdp.observationProb(i,obs)
            new_b[i] = state_prob*obs_prob
    
        new_b = new_b/sum(new_b)       

        return new_b      

        
    def valueIteration(self):
        k = 0
        while(k<6): # Check if this condition for checking is correct or not
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
                    gmix = mixture.GMM(n_components=self.K, covariance_type='full')
                    gmix.fit(self.valueFn)
                    self.valueFn_gmm = gmix.means_

    def beliefValue(self,b):
        if self.nAlphas == 1:
            return dot(self.valueFn[:],b)
        else:
            return max([dot(self.valueFn[j,:],b) for j in range(self.nAlphas)])
        
#    def beliefValueWithNewAlpha(self,new_alpha,belief):
#        return dot(new_alpha,belief)

    
    def valueFnBackup(self,b):
        new_alpha = zeros(self.pomdp.nStates)
        action_alpha = zeros(self.pomdp.nStates)
        for i in range(self.pomdp.nStates):
            new_alpha[i] = max([self.max_a_R(i,j) + self.gamma*self.max_alpha_azj(i,j) for j in range(self.pomdp.nActions)])
            action_alpha[i] = argmax([self.max_a_R(i,j) + self.gamma*self.max_alpha_azj(i,j) for j in range(self.pomdp.nActions)])
        return new_alpha, action_alpha, dot(new_alpha,b)
        
    def max_a_R(self,state,action):
        return self.pomdp.rewardFunction(state,action)
        
    def max_alpha_azj(self,state,action):
        if self.nAlphas ==1:
            return sum([max([sum([dot(self.valueFn[:],s_dash)*self.pomdp.observationProb(s_dash,z)\
                 *self.pomdp.stateTransitionProbState(s_dash,state,action) for s_dash in range(self.pomdp.nStates)]) \
                 for j in range(self.nAlphas)]) for z in range(self.pomdp.nObservations)])

        else:
            return sum([max([sum([dot(self.valueFn[j],s_dash)*self.pomdp.observationProb(s_dash,z)\
                 *self.pomdp.stateTransitionProbState(s_dash,state,action) for s_dash in range(self.pomdp.nStates)]) \
                 for j in range(self.nAlphas)]) for z in range(self.pomdp.nObservations)])
    
    def plan(self,state,belief):
        while(self.run):
            alpha_max = argmax([dot(self.valueFn[j,:],belief) for j in range(self.nAlphas)])
            action_plan = self.actions_alpha[alpha_max,:]
            new_state = self.pomdp.stateUpdate(state,action_plan[state])
            if action_plan[state] != 0:
                self.run = False
            state = new_state
            return new_state
            
        

if __name__ == "__main__":
    nEps = 2
    nTraj = 20
    nT = 3
    b_init = array([0.5, 0.5])
    alpha_init = array([0.0, 0.0])
    init_actions_alpha = array([0, 0])
#    B_set = [b_init]
       
#    environment = TigerProblem()
    pomdp = POMDP()
    pomdp_solve = algorithm(nTraj,nT,pomdp, alpha_init,init_actions_alpha)
#    vi = VI(alpha,nTraj,nT)
    
    pomdp_solve.generateBeliefSet(b_init)
    pomdp_solve.valueIteration()
#    
#    state = 0
#    pomdp_solve.run = True
#    new_state, beleif = pomdp_solve.plan(state,b_init)
    print "Beleif Set = ", pomdp_solve.B
    print "Value function =", pomdp_solve.valueFn
#        environment.reset()
        





