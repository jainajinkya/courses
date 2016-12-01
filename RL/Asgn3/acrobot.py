import numpy as np
import numpy.random as rand
#import matplotlib.pyplot as plt
# from scipy.siself.gnal import medfilt
#from tiles import *
import itertools

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


    def simulator(self,cur_State,action):
        tau = action
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
  
     
#    def grid_world(self,cur_State,action):
#        n_row = 3
#        n_col = 4
#        new_state = [cur_state[0],cur_state[1]]
#
#        if action == 0:
#            if cur_state[1]< (n_col-1):
#                new_state[1] = cur_state[1] + 1
#            else:
#                new_state[1] = cur_state[1]
#
#        if action == 1:
#            if cur_state[1] > 0:
#                new_state[1] = cur_state[1] - 1
#            else:
#                new_state[1] = cur_state[1]
#
#        if action == 2:
#            if cur_state[0] < (n_row-1):
#                new_state[0] = cur_state[0] + 1
#            else:
#                new_state[0] = cur_state[0]
#
#        if action == 3:
#            if cur_state[0] > 0:
#                new_state[0] = cur_state[0] - 1
#            else:
#                new_state[0] = cur_state[0]
#
#        return new_state


    def r_function(self,cur_state):
        return -1.
      
#    def r_fn2(self,cur_state):
#        cliff = [[0,i] for i in range(1,3)]
#        reward = -1     
#        # if cur_state == self.t_state:
#        #   reward = 50.
#
#        for i in range(len(cliff)):
#            if cur_state == cliff[i]:
#                reward = -100
#                break
#
#        return reward
#  
  

class basis_fn():
    def fourier(self,cur_state,f_order):
        n_dim = len(cur_state)
        x = np.arange(0,f_order+1)
        c = [p for p in itertools.product(x, repeat=2)]
        phi = np.zeros(len(c))

        scaled_state = [cur_state[0]/(2*np.pi), cur_state[1]/(8*np.pi), cur_state[2]/(2*np.pi), cur_state[3]/(18*np.pi)]

        for i in range(len(c)):
            # phi[i] = np.cos(np.pi*np.sum([c[i][j]*scaled_state[j] for j in range(n_dim)]))
            phi[i] = np.cos(np.pi*np.sum([c[i][j]*cur_state[j] for j in range(n_dim)]))


        return phi

    def alpha_weights(self,f_order):
        x = np.arange(0,f_order+1)
        c = [p for p in itertools.product(x, repeat=4)]

        alpha = 0.5
        alpha2 = 0.5*np.ones(len(c))

        for i in range(len(c)-1):
            # print "i =", np.sum(c[i+1])
            alpha2[i+1] = alpha/(np.sqrt(np.sum(np.square(c[i+1]))))

        return alpha2


class policy():
    def e_greedy(self, weights,phi):
        eps = 0.
        p = rand.rand()
        if p<eps: 
            action = rand.randint(-1,2,1)
        else:
            idx = np.argmax(np.dot(weights,phi))

            if idx == 0:
                action = -1
            elif idx == 1:
                action = 0
            else: action = 1
            
        return action
  
    def e_greedy2(self, weights,phi):
        eps = 0.0
        p = rand.rand()
        if p <= 1-eps: 
            action = np.argmax(np.dot(weights,phi))     
        else:
            action = rand.randint(0,4)
            
        return action


if __name__ == '__main__':
    world = env()
    ctrl = policy()
    basis = basis_fn()

    n_eps = 500
    alpha = 0.5
    gamma = 0.95
    labda = 0.9
    n_actions = 4
    t_n_steps = np.zeros(n_eps)
    
    f_order = 3
    cur_state = [0., 0., 0., 0.]
#    cur_state = [0,0]
    phi = basis.fourier(cur_state,f_order)
    weights = np.zeros((n_actions,len(phi)))
    alpha2 = basis.alpha_weights(f_order)
    t_state = [0,3]

    # Q_old = 100.
    # Q = np.zeros(n_actions)
    # new_Q = np.zeros(n_actions)

    for ep in range (n_eps):
        cur_state = [0,0]
        phi = basis.fourier(cur_state,f_order)
        action = ctrl.e_greedy2(weights,phi)
        n_steps = 0
         tip = [world.l1*np.cos(cur_state[0]-np.pi/2) + world.l2*np.cos(cur_state[0] - np.pi/2 + cur_state[2]), world.l1*np.sin(cur_state[0]- np.pi/2) + world.l2*np.sin(cur_state[0] - np.pi/2 + cur_state[2])]
         count_action = 0
         old_weights = np.copy(weights)

  
#        while(cur_state != t_state):
#            
#            new_state = world.grid_world(cur_state,action)
#            reward = world.r_fn2(new_state)
#            new_phi = basis.fourier(new_state,f_order)
#            new_action = ctrl.e_greedy2(weights,new_phi)
#
#
#            Q = np.dot(weights[action,:],phi)
#            new_Q = np.dot(weights[new_action,:],new_phi)
#
#            delta = reward + gamma*new_Q - Q
#            
#            # weights[action,:] = weights[action,:] + delta*np.multiply(alpha2,phi)
#            # weights[action,:] = weights[action,:] + delta*alpha*phi/100
#            weights[action,:] = weights[action,:] + delta*alpha
#
#            phi = new_phi
#            cur_state = new_state
#            action = new_action
#
#            # tip = [world.l1*np.cos(cur_state[0]-np.pi/2) + world.l2*np.cos(cur_state[0] - np.pi/2 + cur_state[2]), world.l1*np.sin(cur_state[0]- np.pi/2) + world.l2*np.sin(cur_state[0] - np.pi/2 + cur_state[2])]
#
#            n_steps = n_steps + 1
#            #print "n_steps =", n_steps
#
#        print "n_steps =", n_steps
#        # print "update in weights = ", weights - old_weights
#        t_n_steps[ep] = n_steps
#    print "weights = ", weights

#==============================================================================
       while(tip[1] < world.l1):
           if count_action == 3:
               action = ctrl.e_greedy(weights,phi)
               count_action = 0
           else:
               action = 0
               count_action = count_action + 1
 
           new_state = world.simulator(cur_state,action)
           reward = world.r_function(new_state)
           new_phi = basis.fourier(new_state,f_order)
 
           Q = np.dot(weights[action+1,:],phi)
           new_Q = np.dot(weights[action+1,:],new_phi)
 
           delta = reward + gamma*new_Q - Q
           
           weights[action+1,:] = weights[action+1,:] + delta*np.multiply(alpha2,phi)
 
           Q_old = Q
           phi = new_phi
           cur_state = new_state
 
           tip = [world.l1*np.cos(cur_state[0]-np.pi/2) + world.l2*np.cos(cur_state[0] - np.pi/2 + cur_state[2]), world.l1*np.sin(cur_state[0]- np.pi/2) + world.l2*np.sin(cur_state[0] - np.pi/2 + cur_state[2])]
 
           n_steps = n_steps + 1
           # print "n_steps =", n_steps
 
       print "n_steps =", n_steps
       # print "update in weights = ", weights - old_weights
       t_n_steps[ep] = n_steps
#==============================================================================
        # print "######################################################################"
#==============================================================================
# 
#   ## Final Policy
#   cur_state = [0., 0., 0., 0.]
#   tip = [world.l1*np.cos(cur_state[0]-np.pi/2) + world.l2*np.cos(cur_state[0] - np.pi/2 + cur_state[2]), world.l1*np.sin(cur_state[0]- np.pi/2) + world.l2*np.sin(cur_state[0] - np.pi/2 + cur_state[2])]
# 
#   f = open('data.txt', 'w')
#   f2 = open('data_n_steps.txt','w')
#   f2.write(str(t_n_steps))
#   f2.write('\n')
#   f2.close()
#   counter2 = 0
# 
#   while(tip[1] < world.l1  + 0.6):
#       phi = basis.fourier(cur_state,f_order)
#       action = np.argmax(np.dot(weights,phi))
#       new_state = world.simulator(cur_state,action)
#       cur_state = new_state
# 
#       pt1 = [world.l1*np.cos(cur_state[0]-np.pi/2) , world.l1*np.sin(cur_state[0]-np.pi/2)]
#       tip = [world.l1*np.cos(cur_state[0]-np.pi/2) + world.l2*np.cos(cur_state[0] - np.pi/2 + cur_state[2]), world.l1*np.sin(cur_state[0]- np.pi/2) + world.l2*np.sin(cur_state[0] - np.pi/2 + cur_state[2])]
# 
#       print "tip[1] - world.l1 = ", tip[1] - world.l1
#       counter2 = counter2 + 1
#       print "counter2 = ", counter2
#       f.write(str(pt1[0]))
#       f.write('\t')
#       f.write(str(pt1[1]))
#       f.write('\t')
#       f.write(str(tip[0]))
#       f.write('\t')
#       f.write(str(tip[1]))
#       f.write('\n')
# 
#   f.close()
#==============================================================================
    # max_q_value_q_learn = np.argmax(np.dot(weights,phi))
    # cur_state = [0,0]
    # route_q_learn = []
    # route_q_learn = route_q_learn + [cur_state[:]]
    # k = 0

    # while(cur_state != t_state and k<1000):
    #     cur_state = world.grid_world(cur_state,max_q_value_q_learn)
    #     route_q_learn = route_q_learn + [cur_state[:]]
    #     k = k+1
        
    # print "route_q_learn = ", route_q_learn 
    print "max_q_value_q_learn = ", np.dot(weights,phi) 
   
   
   
   

    # for ep in range   (n_eps):
    #   e_vector = np.zeros(np.shape(weights)[1])
    #   cur_state = [0., 0., 0., 0.]
    #   Q_old = 0.
    #   phi = basis.fourier(cur_state,f_order)
    #   n_steps = 0
    #   tip = [world.l1*np.cos(cur_state[0]) + world.l2*np.cos(cur_state[0] + cur_state[1]), world.l1*np.sin(cur_state[0]) + world.l2*np.sin(cur_state[0] + cur_state[1])]

    #   # print "intial condition, tip[1] =", tip[1]
    #   # print "alpha2 =", alpha2

    #   while(tip[1] < world.l1):
    #       action = ctrl.e_greedy(Q_old)
    #       new_state = world.simulator(cur_state,action)
    #       reward = world.r_function(new_state)
    #       new_phi = basis.fourier(new_state,f_order)

    #       Q = np.dot(weights[action,:],phi)
    #       new_Q = np.dot(weights[action,:],new_phi)

    #       # e_vector = gamma*labda*e_vector + (1.*np.ones(len(phi)) - gamma*labda*np.dot(alpha2, np.dot(e_vector,phi))/np.linalg.norm(gamma*labda*np.dot(alpha2, np.dot(e_vector,phi))))*phi
    #       # e_vector = gamma*labda*e_vector + (1.*np.ones(len(phi)) - gamma*labda*np.dot(e_vector,np.multiply(alpha2,phi)))*phi
            
    #       delta = reward + gamma*new_Q - Q
            
    #       print "Q =", Q
    #       # print "Q_new =", new_Q

    #       # print "phi = ", phi
    #       # print "e_vector = ", e_vector
            
    #       # weights[action,:] = weights[action,:] + (delta + Q - Q_old)*np.dot(alpha2,e_vector) - (Q-Q_old)*np.dot(alpha2,phi)
    #       weights[action,:] = weights[action,:] + delta*np.multiply(alpha2,phi)

    #       Q_old = Q
    #       phi = new_phi
    #       cur_state = new_state

    #       tip = [world.l1*np.cos(cur_state[0]) + world.l2*np.cos(cur_state[0] + cur_state[1]), world.l1*np.sin(cur_state[0]) + world.l2*np.sin(cur_state[0] + cur_state[1])]

    #       n_steps = n_steps + 1

    #   print "n_steps =", n_steps
    # ## Final Policy
    # cur_state = [0., 0., 0., 0.]
    # tip = [world.l1*np.cos(cur_state[0]) + world.l2*np.cos(cur_state[0] + cur_state[1]), world.l1*np.sin(cur_state[0]) + world.l2*np.sin(cur_state[0] + cur_state[1])]

    # while(tip[1] < world.l1):
    #   phi = basis.fourier(cur_state,f_order)
    #   action = np.argmax(np.dot(weights,phi))
    #   new_state = world.simulator(cur_state,action)
    #   cur_state = new_state
    #   tip = [world.l1*np.cos(cur_state[0]) + world.l2*np.cos(cur_state[0] + cur_state[1]), world.l1*np.sin(cur_state[0]) + world.l2*np.sin(cur_state[0] + cur_state[1])]
        # print "tip trajctory = ", tip
