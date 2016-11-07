import numpy as np
import numpy.random as rand
from mpl_toolkits.mplot3d.axes3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm
# from scipy.siself.gnal import medfilt
#from tiles import *
import itertools

class env():
    def __init__(self):
        self.dt = 0.05
        self.n_row = 7
        self.n_col = 20
     
    def grid_world(self,cur_State,action):
        n_row = self.n_row
        n_col = self.n_col
        new_state = [cur_state[0],cur_state[1]]

        if action == 0:
            if cur_state[1]< (n_col-1):
                new_state[1] = cur_state[1] + 1
            else:
                new_state[1] = cur_state[1]

        if action == 1:
            if cur_state[1] > 0:
                new_state[1] = cur_state[1] - 1
            else:
                new_state[1] = cur_state[1]

        if action == 2:
            if cur_state[0] < (n_row-1):
                new_state[0] = cur_state[0] + 1
            else:
                new_state[0] = cur_state[0]

        if action == 3:
            if cur_state[0] > 0:
                new_state[0] = cur_state[0] - 1
            else:
                new_state[0] = cur_state[0]

        return new_state
      
    def r_fn2(self,cur_state):
        cliff = [[0,i] for i in range(1,self.n_col-1)]
        reward = -1 
        # if cur_state == self.t_state:
        #   reward = 50.

        for i in range(len(cliff)):
            if cur_state == cliff[i]:
                reward = -100
                break

        return reward
    

class basis_fn():
    def fourier(self,cur_state,f_order,world):
        n_dim = len(cur_state)
        x = np.arange(0,f_order+1)
        c = [p for p in itertools.product(x, repeat=2)]
        phi = np.zeros(len(c))
        n_row = world.n_row
        n_col = world.n_col

        scaled_state = [cur_state[0]/(2*np.pi*n_row), cur_state[1]/(2*np.pi*n_col)]

        for i in range(len(c)):
            phi[i] = np.cos(np.pi*np.sum([c[i][j]*scaled_state[j] for j in range(n_dim)]))
            # phi[i] = np.cos(np.pi*np.sum([c[i][j]*cur_state[j] for j in range(n_dim)]))


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
    def e_greedy2(self, weights,phi,n_eps):
        eps = 0.1*np.exp(-n_eps/2)
        # eps = 0.1
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

    n_eps = 200
    alpha = 0.5
    gamma = 1.0
    labda =  0.9 
    n_actions = 4
    t_n_steps = np.zeros(n_eps)
    
    f_order = 10
    cur_state = [0,0]
    t_state = [0,world.n_col-1]
    phi = basis.fourier(cur_state,f_order,world)
    weights = np.zeros((n_actions,len(phi)))
    alpha2 = basis.alpha_weights(f_order)
   

    for ep in range (n_eps):
        cur_state = [0,0]
        phi = basis.fourier(cur_state,f_order,world)
        action = ctrl.e_greedy2(weights,phi,ep+1)
        
        n_steps = 0
        counter = 0
  
        while(cur_state != t_state):
            new_state = world.grid_world(cur_state,action)
            reward = world.r_fn2(new_state)
            new_phi = basis.fourier(new_state,f_order,world)
            new_action = ctrl.e_greedy2(weights,new_phi,ep+1)

            ## Hack to avoid getting stuck in an infinite loop
            if(n_steps%2 == 0):
                state_older = cur_state

            if(new_state == state_older):
                counter = counter + 1

            if(counter > 20):
                action = ctrl.e_greedy2(weights,phi,10)
                new_state = world.grid_world(cur_state,action)
                ew_phi = basis.fourier(new_state,f_order,world)
                new_action = ctrl.e_greedy2(weights,new_phi,ep+1)
                counter = 0
            #############################################

            Q = np.dot(weights[action,:],phi)
            new_Q = np.dot(weights[new_action,:],new_phi)

            delta = reward + gamma*new_Q - Q
            # print "delta = ", delta
            
            weights[action,:] = weights[action,:] + delta*alpha*phi/len(phi)

            phi = new_phi
            cur_state = new_state
            action = new_action

            n_steps = n_steps + 1
            # print "n_steps =", n_steps
            # print "cur_State = ", cur_state

        print "n_steps =", n_steps
        # t_n_steps[ep] = n_steps
        # print "weights = ", weights
    # print "weights = ", weights


    # for ep in range (n_eps):
    #     cur_state = [0,0]
    #     phi = basis.fourier(cur_state,f_order,world)
    #     action = ctrl.e_greedy2(weights,phi,ep+1)
    #     e = np.zeros(len(phi))

    #     n_steps = 0
    #     counter = 0
    #     Q_old = 0
  
    #     while(cur_state != t_state):
    #         new_state = world.grid_world(cur_state,action)
    #         reward = world.r_fn2(new_state)
    #         new_phi = basis.fourier(new_state,f_order,world)
    #         new_action = ctrl.e_greedy2(weights,new_phi,ep+1)

    #         ## Hack to avoid getting stuck in an infinite loop
    #         if(n_steps%2 == 0):
    #             state_older = cur_state

    #         if(new_state == state_older):
    #             counter = counter + 1

    #         if(counter > 20):
    #             action = ctrl.e_greedy2(weights,phi,10)
    #             new_state = world.grid_world(cur_state,action)
    #             ew_phi = basis.fourier(new_state,f_order,world)
    #             new_action = ctrl.e_greedy2(weights,new_phi,ep+1)
    #             counter = 0
    #         #############################################

    #         Q = np.dot(weights[action,:],phi)
    #         new_Q = np.dot(weights[new_action,:],new_phi)

    #         e = gamma*labda*e + (1 - alpha*gamma*labda*np.dot(e,phi))*phi/len(phi)

    #         delta = reward + gamma*new_Q - Q

    #         weights[action,:] = weights[action,:] + alpha*(delta + Q - Q_old)*e - alpha*(Q-Q_old)*phi/len(phi)

    #         phi = new_phi
    #         cur_state = new_state
    #         action = new_action
    #         Q_old = new_Q

    #         n_steps = n_steps + 1
    #         # print "n_steps =", n_steps
    #         # print "cur_State = ", cur_state
    #         # print "e = ", e

    #     print "n_steps =", n_steps
    #     # t_n_steps[ep] = n_steps
    #     # print "weights = ", weights
    # # print "weights = ", weights

    cur_state = [0,0]
    route_sarsa = []
    route_sarsa = route_sarsa + [cur_state[:]]
    
    k = 0
    while(cur_state != t_state and k<100):
        phi = basis.fourier(cur_state,f_order,world)
        cur_state = world.grid_world(cur_state,np.argmax(np.dot(weights,phi)))
        route_sarsa = route_sarsa + [cur_state[:]]
        k = k+1
        # print "final_path", cur_state
        
    print "route_sarsa = ", route_sarsa

    q_function = np.zeros((world.n_row,world.n_col))

    for i in range(world.n_row):
        for j in range(world.n_col):
            cur_state = [i,j]
            phi = basis.fourier(cur_state,f_order,world)
            q_function[i,j] = np.argmax(np.dot(weights,phi))
            # print "(i,j) = ", i,j
            # print "phi = ", phi

    print "q_function =", q_function
    
    # print "reward_fxn =", reward_fxn
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # X = np.arange(0, world.n_row, 1)
    # Y = np.arange(0, world.n_col, 1)
    # q_function = np.transpose(q_function)

    # X, Y = np.meshgrid(X, Y)
    # print "len(X), len(Y)", np.shape(X), np.shape(Y)
    # print "len(q_function)=", np.shape(q_function)
    # surf = ax.plot_surface(X, Y, q_function, rstride=1, cstride=1, cmap=cm.coolwarm,
    #             linewidth=0, antialiased=False)

    # plt.show()
