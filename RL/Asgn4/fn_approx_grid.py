import numpy as np
import numpy.random as rand
#from mpl_toolkits.mplot3d.axes3d import Axes3D
import matplotlib.pyplot as plt
#import matplotlib
#from matplotlib import cm
from matplotlib import rcParams
from scipy.signal import medfilt
#from tiles import *
import itertools

class env():
    def __init__(self):
        self.dt = 0.05
        self.n_row = 50
        self.n_col = 50
     
    def grid_world(self,cur_state,action):
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
        c = [p for p in itertools.product(x, repeat=n_dim)]
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
        # eps = 0.1*np.exp(-n_eps/2)
        if n_eps>15:
            eps = 0.
        else: eps = 0.1
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

    n_eps = 1
    alpha = 0.5
    gamma = 0.9
    labda =  0.9 
    n_actions = 4
    f_order = 10
    
    f = open('data1.txt', 'w') 
    
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
    f.write("\t world size = ")
    f.write(str(world.n_row))
    f.write(",")
    f.write(str(world.n_col))
    f.write("\n")
   
    cur_state = [0,0]
    t_state = [0,world.n_col-1]
    phi = basis.fourier(cur_state,f_order,world)
    weights = np.zeros((n_actions,len(phi)))
    alpha2 = basis.alpha_weights(f_order)
    
    t_n_steps = np.zeros(n_eps)
    total_reward_sarsa = np.zeros(n_eps)
   

    for ep in range (n_eps):
        cur_state = [0,0]
        phi = basis.fourier(cur_state,f_order,world)
        action = ctrl.e_greedy2(weights,phi,ep+1)
        e = np.zeros(len(phi))
        Q_old = 0
        
        n_steps = 0
        counter = 0
        reward_sarsa = 0
        
  
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
                new_phi = basis.fourier(new_state,f_order,world)
                new_action = ctrl.e_greedy2(weights,new_phi,ep+1)
                counter = 0
            #############################################

            Q = np.dot(weights[action,:],phi)
            new_Q = np.dot(weights[new_action,:],new_phi)

            delta = reward + gamma*new_Q - Q
            # print "delta = ", delta
            
            weights[action,:] = weights[action,:] + delta*alpha*phi/len(phi)
            
            # Elgibility trace update
            # Thresholding the feature vector to            
#            e = gamma*labda*e + (1 - alpha*gamma*labda*np.dot(e,phi))*phi/len(phi)
#            e = gamma*labda*e + (1 - alpha*gamma*labda*np.dot(e,phi))*phi
#            weights[action,:] = weights[action,:] + alpha*(delta + Q - Q_old)*e - alpha*(Q-Q_old)*phi/len(phi)
#            weights[action,:] = weights[action,:] + alpha*(delta + Q - Q_old)*e - alpha*(Q-Q_old)*phi

            phi = new_phi
            cur_state = new_state
            action = new_action
            Q_old = new_Q
            
            n_steps = n_steps + 1
            reward_sarsa = reward_sarsa + reward
#            print "n_steps =", n_steps
            # print "cur_State = ", cur_state

        print "n_steps =", n_steps
        t_n_steps[ep] = n_steps
        total_reward_sarsa[ep] = reward_sarsa
        # print "weights = ", weights
    print "t_n_steps = ", t_n_steps
    

    s_st = str(t_n_steps)
    f.write("No. of Steps \n")
    f.write(s_st)


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
    
    s_route = str(route_sarsa)
    f.write("\n \n")
    f.write("route_sarsa \n")
    f.write(s_route)

    q_function = np.zeros((world.n_row,world.n_col))

    for i in range(world.n_row):
        for j in range(world.n_col):
            cur_state = [i,j]
            phi = basis.fourier(cur_state,f_order,world)
            q_function[i,j] = np.argmax(np.dot(weights,phi))
            # print "(i,j) = ", i,j
            # print "phi = ", phi

    print "q_function =", q_function

    s_q = str(q_function)
    f.write("\n \n")
    f.write("Q_function \n")
    f.write(s_q)
    f.close()
   
    
    ### Plots
    fig = plt.figure()
    x_vec = np.arange(0,n_eps)
    # filtered_rew_sarsa = savgol_filter(total_reward_sarsa, 5, 1)
    # filtered_rew_q_learn = savgol_filter(total_reward_q_learn, 5, 1)

    filtered_rew_sarsa = medfilt(total_reward_sarsa, 15)

    m1 = np.mean(filtered_rew_sarsa)*np.ones(len(x_vec))
#    m2 = np.mean(filt_r_q)*np.ones(len(x_vec))
    # filtered_rew_q_learn = medfilt(total_reward_q_learn, 5)
    line1, = plt.plot(x_vec,filtered_rew_sarsa, 'r', linewidth=3.0)
    # plt.plot(x_vec,filtered_rew_q_learn, 'b')
    # plt.plot(x_vec,filt_r_s, 'r')
#    line2, = plt.plot(x_vec,filt_r_q, 'b', linewidth=3.0)
    
    plt.plot(x_vec,m1,'r--', linewidth=3.0)
#    plt.plot(x_vec,m2,'b--', linewidth=3.0)

    axes = plt.gca()
    axes.set_xlim([0,n_eps])
    axes.set_ylim([-200, 0])
#    plt.legend([line1],['SARSA'],loc=2)
    plt.xlabel('No. of Episodes')
    plt.ylabel('Total reward in each episode')
    plt.text(5., -25.,'\eps = 0.2', fontsize=18)
    rcParams.update({'font.size': 18})
    
    
    fig2 = plt.figure()
    line2, = plt.plot(x_vec,t_n_steps, 'r', linewidth=3.0)
    plt.plot(x_vec,t_n_steps,'r--', linewidth=3.0)
#    plt.plot(x_vec,m2,'b--', linewidth=3.0)

    axes = plt.gca()
    axes.set_xlim([0,n_eps])
    axes.set_ylim([0,500000])
#    plt.legend([line1],['SARSA'],loc=2)
    plt.xlabel('No. of Episodes')
    plt.ylabel('Number of steps taken in each episode')
    plt.text(5., -25.,'\eps = 0.2', fontsize=18)
    rcParams.update({'font.size': 18})
    
    plt.show()
    # p_value = np.zeros((size_world[0],size_world[1]))
    # # dummy_state = world.state_update([1,11],3)
    # # p_value(dummy_state) = 1.0
    # print dummy_state

    # ax = fig.add_subplot(111, projection='3d')
    # X = np.arange(0, size_world[1], 1)
    # Y = np.arange(0, size_world[0], 1)
    # X, Y = np.meshgrid(X, Y)
    # surf = ax.plot_surface(X, Y, p_value, rstride=1, cstride=1, cmap=cm.coolwarm,
    #         linewidth=0, antialiased=False)
    # plt.show()
    
    
