import numpy as np
from scipy.stats import mvn
import itertools
from numpy.linalg import inv
from numpy import matmul


class system():
    def __init__(self):
        self.nState = 2
        self.nInput = 2
        self.nOutput = 2
        self.nModel = 3
        self.nGauss = 1
        self.mA = np.zeros([self.nState,self.nState,self.nModel])
        self.mB = np.zeros([self.nInput,self.nInput,self.nModel])
        self.mC = np.zeros([self.nOutput,self.nOutput,self.nModel])
        self.wts = (1./self.nModel)*np.ones(self.nModel)
        
        # Comibining all of the system matrices to generate the combines hybrid dynamics matrices 
        for i in range(self.nModel):
            [self.mA[:,:,i],self.mB[:,:,i],self.mC[:,:,i]] = self.dynamics_templates(i)
        
    def dynamics_templates(self,Idx):
        if(Idx==0):
            A = np.eye(self.nState)
            B = np.eye(self.nInput)
            C = np.eye(self.nOutput)
            
        if(Idx==1):
            A = 2*np.eye(self.nState)
            B = 2*np.eye(self.nInput)
            C = 2*np.eye(self.nOutput)
            
        if(Idx==2):
            A = 0.5*np.eye(self.nState)
            B = 0.5*np.eye(self.nInput)
            C = 0.5*np.eye(self.nOutput)
        
        return [A,B,C]
               
    def beliefUpdate(self,x,mu,cov,u):
        muSet = np.zeros([self.nState,self.nGauss,self.nModel])
        covSet = np.zeros([self.nState,self.nState,self.nGauss,self.nModel]);
        muNew = np.zeros([self.nState,self.nGauss])
        covNew = np.zeros([self.nState,self.nState,self.nGauss])
        
        W = 0.5*np.power((5.0-x[1]),2)*np.eye(self.nState)
        for k in range(self.nGauss):
            for i in range(self.nModel):
                muSet[:,k,i] = np.transpose(self.mA[:,:,i].dot(mu) + self.mB[:,:,i].dot(u))
                gamma = matmul(matmul(self.mA[:,:,i],cov[:,:,k]),np.transpose(self.mA[:,:,i]));
                covSet[:,:,k,i] = gamma - matmul(matmul(gamma,np.transpose(self.mC[:,:,i])),matmul((inv(matmul(matmul(self.mC[:,:,i],gamma),np.transpose(self.mC[:,:,i])) + W)),np.matmul(self.mC[:,:,i],gamma)));

        for i in range(self.nState):
            for j in range(self.nGauss):
                for k in range(self.nModel):
                    muNew[i,j] = muNew[i,j] + self.wts[k]*muSet[i,j,k]
                    covNew[i,:,j] = covNew[i,:,j] + self.wts[k]*covSet[i,:,j,k]
        
        return [muNew,covNew]
    

    def systemUpdate(self,x,u):
        x_dummy = np.zeros([np.size(x),self.nModel])
        
        for i in range(self.nModel):
            x_dummy[:,i] = self.mA[:,:,i].dot(x) + self.mB[:,:,i].dot(u)
            
        x_new = x_dummy.dot(self.wts)
        return x_new

    
    def wts_update(self,mu,cov): # Remember here the input x is the mean of the Gaussians 
    #representing distribution over states
        chpts = [[1,0],[2,0]]
        
        new_wts = np.zeros(self.nModel)

        # Bounds for calculating cdf
        low = -100*np.ones(self.nState)
        for i in range(np.shape(chpts)[0]):
            new_wts[i],i = mvn.mvnun(low,chpts[i],mu,cov)
        
        #Normalize
        new_wts = np.concatenate([[new_wts[0]],np.diff((new_wts))[0:-1],[1-new_wts[-1]]],axis=0)
        new_wts = new_wts/np.linalg.norm(new_wts)
        return new_wts
    
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
        scaled_state = [cur_state[i]/15. for i in range(self.n_dim)]

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
    def e_greedy(self, weights,phi,nInputs):
        eps = 0.1
        p = np.rand.rand()
        if p <= 1-eps: 
            action = np.argmax(np.dot(weights,phi))     
        else:
            action = np.rand.rand()*np.ones(nInputs)
            
        return action
    

if __name__ == '__main__':
    world = system()
    basis = basis_fn(world,15)
    ctrl = policy()
    
    x = np.transpose(np.array([2.,0.]))
    u = np.zeros([world.nInput,1])
    mu = np.zeros([world.nState,world.nGauss])
    cov = np.zeros([world.nState,world.nState,world.nGauss])
   
    for i in range(world.nGauss):
        mu[:,i] = np.transpose(np.array([2.,2.]))  
        cov[:,:,i] = 5*np.eye(world.nState)

    for j in range(50):
        [mu,cov] = world.beliefUpdate(x,mu,cov,u)
        print "mu =", mu
    print "cov = ", cov
    
    
    
    
#    n_eps = 200
#    alpha = 0.5
#    gamma = 1.0
#    labda =  0.9 
#    n_actions = 4
#    t_n_steps = np.zeros(n_eps)
#    
#    f = open('data.txt', 'w')
#    f_order = 25
#    cur_state = [0,0,0]
#    t_state = [0,world.n_col-1,1]
#    phi = basis.fourier(cur_state,f_order,world)
#    weights = np.zeros((n_actions,len(phi)))
#    alpha2 = basis.alpha_weights(f_order)
#   
#
#    for ep in range (n_eps):
#        cur_state = [0,0,0]
#        phi = basis.fourier(cur_state,f_order,world)
#        action = ctrl.e_greedy2(weights,phi,ep+1)
#        
#        n_steps = 0
#        counter = 0
#        t_n_steps = np.zeros(n_eps)
#  
#        while(cur_state != t_state):
#            new_state = world.grid_world(cur_state,action)
#            reward = world.r_fn2(new_state)
#            new_phi = basis.fourier(new_state,f_order,world)
#            new_action = ctrl.e_greedy2(weights,new_phi,ep+1)
#
#            ## Hack to avoid getting stuck in an infinite loop
#            if(n_steps%2 == 0):
#                state_older = cur_state
#
#            if(new_state == state_older):
#                counter = counter + 1
#
#            if(counter > 20):
#                action = ctrl.e_greedy2(weights,phi,10)
#                new_state = world.grid_world(cur_state,action)
#                ew_phi = basis.fourier(new_state,f_order,world)
#                new_action = ctrl.e_greedy2(weights,new_phi,ep+1)
#                counter = 0
#            #############################################
#
#            Q = np.dot(weights[action,:],phi)
#            new_Q = np.dot(weights[new_action,:],new_phi)
#
#            delta = reward + gamma*new_Q - Q
#            # print "delta = ", delta
#            
#            weights[action,:] = weights[action,:] + delta*alpha*phi/len(phi)
#
#            phi = new_phi
#            cur_state = new_state
#            action = new_action
#
#            n_steps = n_steps + 1
#            # print "n_steps =", n_steps
#            # print "cur_State = ", cur_state
#
#        print "n_steps =", n_steps
#        t_n_steps[ep] = n_steps
#        # print "weights = ", weights
#    # print "weights = ", weights
#
#    s_st = str(t_n_steps)
#    f.write("No. of Steps \n")
#    f.write(s_st)
#    