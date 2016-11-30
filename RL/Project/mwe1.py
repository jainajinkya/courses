import numpy as np
from scipy.stats import mvn
import itertools
from numpy.linalg import inv
from numpy import matmul
#import random


class system():
    def __init__(self):
        self.nState = 4
        self.nInput = 4
        self.nOutput = 4
        self.nModel = 3
        self.nGauss = 1
        self.mA = np.zeros([self.nState,self.nState,self.nModel])
        self.mB = np.zeros([self.nInput,self.nInput,self.nModel])
        self.mC = np.zeros([self.nOutput,self.nOutput,self.nModel])
        self.wts = (1./self.nModel)*np.ones(self.nModel)
        self.pitch = 0.1
        
        # Comibining all of the system matrices to generate the combines hybrid dynamics matrices 
        for i in range(self.nModel):
            [self.mA[:,:,i],self.mB[:,:,i],self.mC[:,:,i]] = self.dynamics_templates(i)
        
    def dynamics_templates(self,Idx):
        if(Idx==0):
            A = np.eye(self.nState)
            B = np.eye(self.nInput)
            C = np.eye(self.nOutput)
            
        if(Idx==1):
            A = np.eye(self.nState)
            B = np.zeros([self.nState,self.nInput])
            C = np.eye(self.nOutput)
            
            B[2,3] = (self.pitch/(2*np.pi))
            B[3,3] = 1.
            
        if(Idx==2):
            A = np.eye(self.nState)
            B = np.zeros([self.nState,self.nInput])
            C = np.eye(self.nOutput)
        
        return [A,B,C]
               
    def beliefUpdate(self,x,mu,cov,u):
        muSet = np.zeros([self.nState,self.nGauss,self.nModel])
        covSet = np.zeros([self.nState,self.nState,self.nGauss,self.nModel]);
        muNew = np.zeros([self.nState,self.nGauss])
        covNew = np.zeros([self.nState,self.nState,self.nGauss])
        
        W = 0.5*np.power((5.0-x[1]),2)*np.eye(self.nState)
        for k in range(self.nGauss):
            for i in range(self.nModel):
                muSet[:,k,i] = np.transpose(self.mA[:,:,i].dot(mu)) + self.mB[:,:,i].dot(u)
                gamma = matmul(matmul(self.mA[:,:,i],cov[:,:,k]),np.transpose(self.mA[:,:,i]));
                covSet[:,:,k,i] = gamma - matmul(matmul(gamma,np.transpose(self.mC[:,:,i])),matmul((inv(matmul(matmul(self.mC[:,:,i],gamma),np.transpose(self.mC[:,:,i])) + W)),np.matmul(self.mC[:,:,i],gamma)));

        for i in range(self.nState):
            for j in range(self.nGauss):
                for k in range(self.nModel):
                    muNew[i,j] = muNew[i,j] + self.wts[k]*muSet[i,j,k]
                    covNew[i,:,j] = covNew[i,:,j] + self.wts[k]*covSet[i,:,j,k]
        
        self.wts_update(muNew,covNew)
        return [muNew,covNew]
    

    def systemUpdate(self,x,u):
        x_dummy = np.zeros([np.size(x),self.nModel])
        
        for i in range(self.nModel):
            x_dummy[:,i] = self.mA[:,:,i].dot(x) + self.mB[:,:,i].dot(u)
            
        x_new = x_dummy.dot(self.wts)
        return x_new

    
    def wts_update(self,mu,cov): # Remember here the input x is the mean of the Gaussians 
    #representing distribution over states
        chpts = [[1,1,1,0.],[0.5,0.5,0.5,np.pi]]
        
        new_wts = np.zeros(self.nModel)

        # Bounds for calculating cdf
        low = -100*np.ones(self.nState)
        for i in range(np.shape(chpts)[0]):
            new_wts[i],i = mvn.mvnun(low,chpts[i],mu,cov)
        
        #Normalize
        new_wts = np.concatenate([[new_wts[0]],np.diff((new_wts))[0:-1],[1-new_wts[-1]]],axis=0)
        new_wts = new_wts/np.linalg.norm(new_wts)
        self.wts = new_wts
        return
    
    def reward(self):
        reward = -1.
        return reward
        
        
class basis_fn():
    def __init__(self,sim,f_order):
         self.n_dim = sim.nState
         self.f_order = f_order
         x = np.arange(0,self.f_order+1)
         self.c = [p for p in itertools.product(x, repeat=self.n_dim)]
         
    def fourier(self,cur_state):
        phi = np.zeros(len(self.c))
#        scaled_state = [cur_state[0]/10.,cur_state[1]/10.,cur_state[2]/10., cur_state[3]/(6*np.pi),cur_state[4]/(6*np.pi),cur_state[5]/(6*np.pi)]
        scaled_state = [cur_state[0]/10.,cur_state[1]/10.,cur_state[2]/10., cur_state[3]/(6*np.pi)]
        for i in range(len(self.c)):
            # print "dim =", self.n_dim
            # print "term =", self.c
            phi[i] = np.cos(np.pi*np.sum([self.c[i][j]*scaled_state[j] for j in range(self.n_dim)]))
            # phi[i] = np.cos(np.pi*np.sum([self.c[i][j]*cur_state[j] for j in range(self.n_dim)]))

        return phi

    def alpha_weights(self):
        alpha = 0.5
        alpha2 = 0.5*np.ones(len(self.c))

        for i in range(len(self.c)-1):
            # print "i =", np.sum(self.c[i+1])
            alpha2[i+1] = alpha/(np.sqrt(np.sum(np.square(self.c[i+1]))))

        return alpha2


class policy():
    def __init__(self,n_actions,world):
        self.n_actions = n_actions
        x = [-0.2, 0., 0.2]
        self.c = [p for p in itertools.product(x, repeat=world.nInput)]
        
    def e_greedy(self, weights,phi,n_eps):
        if n_eps < 15:
            eps = 0.1
        else: eps = 0.0
        p = np.random.rand()
        if p <= 1-eps: 
            idx = np.argmax(np.dot(weights,phi))
            action = self.c[idx]
        else:
            idx = np.random.randint(len(self.c))
            action = self.c[idx]
            
        return [action,idx]
    

if __name__ == '__main__':
   
    
#    x = np.transpose(np.array([2.,0.]))
#    u = np.zeros([world.nInput,1])
#    mu = np.zeros([world.nState,world.nGauss])
#    cov = np.zeros([world.nState,world.nState,world.nGauss])
#   
#    for i in range(world.nGauss):
#        mu[:,i] = np.transpose(np.array([2.,2.]))  
#        cov[:,:,i] = 5*np.eye(world.nState)
#
#    for j in range(50):
#        [mu,cov] = world.beliefUpdate(x,mu,cov,u)
#        print "mu =", mu
#    print "cov = ", cov
#    
    world = system()
    
    n_eps = 500
    alpha = 0.5
    gamma = 0.9
    labda =  0.9 
    n_actions = 3**world.nState
    f_order = 5
        
    basis = basis_fn(world,f_order)
    ctrl = policy(n_actions,world)
    f = open('hybrid dynamics_data.txt', 'w') 
    
    f.write("n_eps = ")
    f.write(str(n_eps))
    f.write("\t alpha = ")
    f.write(str(alpha))
    f.write("\t gamma = ")
    f.write(str(gamma))
    f.write("\t labda = ")
    f.write(str(labda))
    f.write("\t f_order = ")
    f.write(str(f_order))
    f.write("\n")
   
    cur_mu = np.zeros([world.nState,world.nGauss])
    cur_cov = np.zeros([world.nState,world.nState,world.nGauss])
    
    for i in range(world.nGauss):
        cur_mu[:,i] = 2*np.random.rand(world.nState)
        cur_cov[:,:,i] = 5*np.eye(world.nState)
    
    cur_state = (1/world.nGauss)*np.sum(cur_mu,axis=1)   
    cur_init = cur_state    
    t_state = np.array([0., 0., 0.,np.pi*2])
    phi = basis.fourier(cur_state)
    weights = np.zeros((n_actions,len(phi)))
#    alpha2 = basis.alpha_weights(f_order)
    
    t_n_steps = np.zeros(n_eps)
   

    for ep in range (n_eps):
        cur_mu = np.zeros([world.nState,world.nGauss])
        cur_cov = np.zeros([world.nState,world.nState,world.nGauss])
    
        for i in range(world.nGauss):
            cur_mu[:,i] = 2*np.random.rand(world.nState)
            cur_cov[:,:,i] = 5*np.eye(world.nState)
    
        cur_state = (1/world.nGauss)*np.sum(cur_mu,axis=1) 
        phi = basis.fourier(cur_state)
        [action,idx] = ctrl.e_greedy(weights,phi,ep+1)
        e = np.zeros(len(phi))
        Q_old = 0
        
        n_steps = 0
        counter = 0
        
#        p=1
        while(max(abs(t_state - cur_state)) > 0.5 and np.amax(cur_cov) > 0.5):
#        while(p==1):
            new_state = world.systemUpdate(cur_state,action)
            [new_mu,new_cov] = world.beliefUpdate(cur_state,cur_mu,cur_cov,action)
            
            reward = world.reward()
            new_phi = basis.fourier(new_state)
            [new_action,new_idx] = ctrl.e_greedy(weights,new_phi,ep+1)

            ## Hack to avoid getting stuck in an infinite loop
            if(n_steps%2 == 0):
                state_older = cur_state

            if((new_state == state_older).all()):
                counter = counter + 1

            if(counter > 20):
                [action,idx] = ctrl.e_greedy(weights,phi,10)
                new_state = world.systemUpdate(cur_state,action)
                new_phi = basis.fourier(new_state)
                [new_action,new_idx] = ctrl.e_greedy(weights,new_phi,ep+1)
                counter = 0
            #############################################

            Q = np.dot(weights[idx,:],phi)
            new_Q = np.dot(weights[new_idx,:],new_phi)

            delta = reward + gamma*new_Q - Q

            weights[idx,:] = weights[idx,:] + delta*alpha*phi/len(phi)
            
            # Elgibility trace update
            # Thresholding the feature vector to           
#            e = gamma*labda*e + (1 - alpha*gamma*labda*np.dot(e,phi))*phi/len(phi)
##            e = gamma*labda*e + (1 - alpha*gamma*labda*np.dot(e,phi))*phi
#            weights[action,:] = weights[action,:] + alpha*(delta + Q - Q_old)*e - alpha*(Q-Q_old)*phi/len(phi)
#            weights[action,:] = weights[action,:] + alpha*(delta + Q - Q_old)*e - alpha*(Q-Q_old)*phi

            phi = new_phi
            cur_state = new_state
            cur_mu = new_mu
            cur_cov = new_cov
            action = new_action
            idx = new_idx
            Q_old = new_Q
            
            n_steps = n_steps + 1
            p=0
#            print "n_steps =", n_steps
            # print "cur_State = ", cur_state

        print "n_steps =", n_steps
        t_n_steps[ep] = n_steps
        # print "weights = ", weights
    print "t_n_steps = ", t_n_steps

    s_st = str(t_n_steps)
    f.write("No. of Steps \n")
    f.write(s_st)

   ## Plots for final paths
#    cur_state = [0,0]
    cur_state = cur_init
    route_sarsa = []
    route_sarsa = route_sarsa + [cur_state[:]]
    
    k = 0
    while((cur_state != t_state).all() and k<1000):
        phi = basis.fourier(cur_state)
        idx = np.argmax(np.dot(weights,phi))
        action = ctrl.c[idx]
        cur_state = world.systemUpdate(cur_state,action)        
        route_sarsa = route_sarsa + [cur_state[:]]
        k = k+1
        # print "final_path", cur_state
        
    print "route_sarsa = ", route_sarsa
    
    s_route = str(route_sarsa)
    f.write("\n \n")
    f.write("route_sarsa \n")
    f.write(s_route)

#    q_function = np.zeros((world.n_row,world.n_col))
#
#    for i in range(world.nState):
#        for j in range(i):
#            cur_state = [i,j]
#            phi = basis.fourier(cur_state)
#            q_function[i,j] = np.argmax(np.dot(weights,phi))
#            # print "(i,j) = ", i,j
#            # print "phi = ", phi

    
#    print "q_function =", q_function
#
#    s_q = str(q_function)
#    f.write("\n \n")
#    f.write("Q_function \n")
#    f.write(s_q)
    f.close()