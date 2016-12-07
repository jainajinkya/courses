#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 15:03:18 2016

@author: Ajinkya
"""

from numpy import *
from pypr.clustering import *
from sklearn import mixture
from TigerProblem import TigerProblem

class  POMDP:
    def __init__(self):
        self.nStates = 2
        self.nObservations = 2
        self.nActions = 3
        
#        self.states = ['tiger_left', 'tiger_right']
#        self.actions = ['listen', 'open_left', 'open_right']
#        self.observations = ['tiger_left', 'tiger_right']
        self.listen_success_prob = 0.85
        
        self.new_state = 0

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
               
    def stateUpdate(self,state,action):
        if action == 1:            
            if state == 0 :
                return round(random.uniform())   # None indicates end of episode
                                  # -100 because agent opened tiger door
            else:
                return round(random.uniform())     # None indicates end of episode
                                 # +10 because agent opened treasure door
        elif action == 2:
            if state == 1 :
                return round(random.uniform())  # None indicates end of episode
                                    # -100 because agent opened tiger door
            else :
                return round(random.uniform())    # None indicates end of episode
                                    # +10 because agent opened treasure door
        else:
            return state
    

    def policy(self):
        idx = random.randint(pomdp.nActions)
        return idx
        
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
        

class beliefSet():
    def __init__(self,nTraj,nT):
        self.nTraj = nTraj
        self.nT = nT
        self.B = zeros([nT,nTraj,2])
#        self.Rewards = zeros([nT, nTraj])

    def generateBeliefSet(self,b_init,pomdp):
        for i in range(self.nTraj):
            b = b_init # This can be a unimodal Guassian; Let's try with discrete set of Probabilities
            samples = 1
            
            ############ Remember to change the inputs to the function so as to meet the actual inputs to the GMM
#            state = gmm.sample_gaussian_mixture(b[0],b[1], samples=samples)
            state = random.randint(pomdp.nStates)  # randomly choose a state out of nStates
            
            for j in range(self.nT):
                action = pomdp.policy()

                # Draw a sample from the new probability distribution
                new_state = pomdp.stateUpdate(state,action)

                # Draw a sample from the observation probabilities
                obs = pomdp.observations(state,action)

                # update the belief
                b_a_z = self.beliefUpdate(b,action,obs,pomdp)

                # Add belief to the belief Set
                self.B[j,i,:] = b_a_z

                # Update for the new loop
                b = b_a_z
                state = new_state

    def beliefUpdate(self,b,action,obs,pomdp):
        new_b = zeros(pomdp.nStates)
        for i in range(pomdp.nStates):
            state_prob = pomdp.stateTransitionProb(i,b,action)
            obs_prob = pomdp.observationProb(i,obs)
            new_b[i] = state_prob*obs_prob
    
        new_b = new_b/sum(new_b)       

        return new_b

class VI():
    def __init__(self,init_alpha,nTraj, nT,pomdp):
        # Initialize the value function
        self.valueFn = init_alpha
        self.nTraj = nTraj
        self.nT = nT
        self.K = 5
        self.gamma = 1.0
        self.pomdp = pomdp

    def valueIteration(self,goal):
        k = 0
        while(k <= 100): # Check if this condition for checking is correct or not
            # Randomly choose a trajectory
            r = random.randint(0,self.nTraj)
            k = k+1
            for t in range(self.nT,0,-1):
                old_valueFn = self.beliefValue(self.pomdp.B_set[r,t])

                # Value function backup
                new_alpha = self.valueFnBackup(self.pomdp.B_set[r,t])

#                # projecting to smaller representation: fit a K parameter GMM
#                # Check if direct implemetation like this works or do we need to generate samples before we can fit
#                # K-means gaussians
#                gmix = mixture.GMM(n_components=self.K, covariance_type='full')
#                gmix.fit(new_alpha)
#                new_alpha = gmix.means_

                # If increase in the projected value, add the new alpha vector
                new_valueFn = self.beliefValueWithNewAlpha(new_alpha,(pomdp.B_set[r,t]))

                if(new_valueFn > old_valueFn):
                    self.valueFn = [self.valueFn, new_alpha] # Check how this addition should be done

    def beliefValue(self,b):
        return max([dot(self.valueFn[j],b) for j in range(shape(self.valueFn,2))])
    
    def valueFnBackup(self,b):
        for i in range(self.pomdp.nStates):
            alpha_dummy[i] = self.max_a_R(i) + self.gamma*max_alpha_azj(i)
    
        return dot(alpha_dummy,b)
        
    def max_a_R(self,state):
        return max([self.pomdp.rewardFunction(state,i) for i in range(self.pomdp.nActions)])
        
    def max_alpha_azj(self,state):
      return sum([max([max([sum([dot(self.valueFn[j],s_dash)*self.pomdp.observationProb(s_dash,z)\
                 *self.pomdp.stateTransitionProbState(s_dash,state,i) for s_dash in range(self.pomdp.nStates)]) \
                 for i in range(self.pomdp.nActions)]) for j in range(shape(self.valueFn,2))])])
            ## Ehaan par jara dekh lijiyega ki kahi dimension ka punga to nhi bhaya

    def beliefValueWithNewAlpha(self,new_alpha,belief):
        return dot(new_alpha,belief)


if __name__ == "__main__":
    nEps = 100
    nTraj = 20
    nT = 3
    b_init = array([0.5, 0.5])
#    B_set = [b_init]
       
#    environment = TigerProblem()
    beliefs = beliefSet(nTraj,nT)
#    vi = VI(alpha,nTraj,nT)
    pomdp = POMDP()
    
    for i in range(nEps):
        beliefs.generateBeliefSet(b_init,pomdp)
        print "Beleif Set = ", beliefs.B
#        environment.reset()
        





