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
    
#    for re in range(0,100,5):
    setReward = 50
    world = env(setReward)
    ctrl = policy()
    basis = basis_fn(world.nStates)

    n_eps = 1000
    alpha = 0.001
    gamma = 1.0
    labda =  0.9 
    n_actions = world.nActions
    
    f = open('data1.txt', 'w') 
    
    f.write("n_eps = ")
    f.write(str(n_eps))
    f.write("\t alpha = ")
    f.write(str(alpha))
    f.write("\t gamma = ")
    f.write(str(gamma))
    f.write("\t labda = ")
    f.write(str(labda))
#    f.write("\t f_order = ")
#    f.write(str(f_order))
    f.write("\t world size = ")
    f.write(str(world.nRows))
    f.write(",")
    f.write(str(world.nCols))
    f.write("\n")
   
    cur_state = [1,1]
    t_state = world.goal
    
    phi = basis.stateExcitation(cur_state,world)
    weights = zeros((n_actions,basis.lengthPhi))
    
    t_n_steps = zeros(n_eps)
    total_reward = zeros(n_eps)
   
    for ep in range (n_eps):
        cur_state = [1,1]
        phi = basis.stateExcitation(cur_state,world)
        action = ctrl.e_greedy(weights,phi,ep+1)
        
        e = zeros(basis.lengthPhi)
        Q_old = 0
        
        n_steps = 0
        counter = 0
        reward_sarsa = 0
  
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
            
            # Elgibility trace update
            # Thresholding the feature vector to            
            e = gamma*labda*e + (1 - alpha*gamma*labda*np.dot(e,phi))*phi
            weights[action,:] = weights[action,:] + alpha*(delta + Q - Q_old)*e - alpha*(Q-Q_old)*phi

            phi = new_phi
            cur_state = new_state
            action = new_action
            Q_old = new_Q
            
            n_steps = n_steps + 1
            reward_sarsa = reward_sarsa + reward

        print "n_steps =", n_steps
        t_n_steps[ep] = n_steps
        total_reward[ep] = reward_sarsa

    print "t_n_steps = ", t_n_steps
    

    s_st = str(t_n_steps)
    f.write("No. of Steps \n")
    f.write(s_st)
    f.close
    
    ### Plots
    fig = plt.figure()
    x_vec = np.arange(0,n_eps)
    filtered_rew = medfilt(total_reward, 9)
    m1 = np.mean(filtered_rew)*np.ones(len(x_vec))
    line1, = plt.plot(x_vec,filtered_rew, 'r', linewidth=3.0)
    plt.plot(x_vec,m1,'r--', linewidth=3.0)
    axes = plt.gca()
    axes.set_xlim([0,n_eps])
#    axes.set_ylim([-300, 0])
    plt.xlabel('No. of Episodes')
    plt.ylabel('Total reward in each episode')
    rcParams.update({'font.size': 18})
    plt.show()
    

#    ### Final Policy #######################################
#    cur_state = [1,1]
#    route_sarsa = []
#    route_sarsa = route_sarsa + [cur_state[:]]
#    
#    k = 0
#    while(cur_state != t_state and k<100):
#        phi = basis.stateExcitation(cur_state,world)
#        act  = argmax(dot(weights,phi))
#        print "current State = ", cur_state ,"Optimal action taken = ", world.action[act], "\n"
#        
#        cur_state = world.stateUpdate(cur_state,act)
#        route_sarsa = route_sarsa + [cur_state[:]]
#        k = k+1
#        # print "final_path", cur_state
#        
##    print "route_sarsa = ", route_sarsa
#    print "current State = ", cur_state, "\n"
#    if cur_state == t_state:
#        print "Goal Reached", "\n"
#    else:
#        print "Goal NOT reached", "\n"
#    
#    s_route = str(route_sarsa)
#    f.write("\n \n")
#    f.write("route_sarsa \n")
#    f.write(s_route)

#    q_function = np.zeros((world.nRows,world.nCols))
#
#    for i in range(world.nRows):
#        for j in range(world.nCols):
#            cur_state = [i,j]
#            phi = basis.stateExcitation(cur_state,world)
#            q_function[i,j] = argmax(dot(weights,phi))
#            # print "(i,j) = ", i,j
#            # print "phi = ", phi
#
#    print "q_function =", q_function
#
#    s_q = str(q_function)
#    f.write("\n \n")
#    f.write("Q_function \n")
#    f.write(s_q)
#    f.close()
#   
#    
    ### Plots
    fig = plt.figure()
    x_vec = np.arange(0,n_eps)
#    # filtered_rew_sarsa = savgol_filter(total_reward_sarsa, 5, 1)
#    # filtered_rew_q_learn = savgol_filter(total_reward_q_learn, 5, 1)
#
    filtered_rew = medfilt(total_reward, 9)
#
    m1 = np.mean(filtered_rew)*np.ones(len(x_vec))
##    m2 = np.mean(filt_r_q)*np.ones(len(x_vec))
#    # filtered_rew_q_learn = medfilt(total_reward_q_learn, 5)
    line1, = plt.plot(x_vec,filtered_rew, 'r', linewidth=3.0)
#    # plt.plot(x_vec,filtered_rew_q_learn, 'b')
#    # plt.plot(x_vec,filt_r_s, 'r')
##    line2, = plt.plot(x_vec,filt_r_q, 'b', linewidth=3.0)
#    
    plt.plot(x_vec,m1,'r--', linewidth=3.0)
##    plt.plot(x_vec,m2,'b--', linewidth=3.0)
#
    axes = plt.gca()
    axes.set_xlim([0,n_eps])
    axes.set_ylim([-300, 0])
#    plt.legend([line1],['SARSA'],loc=2)
    plt.xlabel('No. of Episodes')
    plt.ylabel('Total reward in each episode')
#    plt.text(5., -25.,'\eps = 0.2', fontsize=18)
    rcParams.update({'font.size': 18})
    plt.show()
#    
#    
#    fig2 = plt.figure()
#    line2, = plt.plot(x_vec,t_n_steps, 'r', linewidth=3.0)
#    plt.plot(x_vec,t_n_steps,'r--', linewidth=3.0)
##    plt.plot(x_vec,m2,'b--', linewidth=3.0)
#
#    axes = plt.gca()
#    axes.set_xlim([0,n_eps])
#    axes.set_ylim([0,500000])
##    plt.legend([line1],['SARSA'],loc=2)
#    plt.xlabel('No. of Episodes')
#    plt.ylabel('Number of steps taken in each episode')
#    plt.text(5., -25.,'\eps = 0.2', fontsize=18)
#    rcParams.update({'font.size': 18})
#    
#    plt.show()
#    # p_value = np.zeros((size_world[0],size_world[1]))
#    # # dummy_state = world.state_update([1,11],3)
#    # # p_value(dummy_state) = 1.0
#    # print dummy_state
#
#    # ax = fig.add_subplot(111, projection='3d')
#    # X = np.arange(0, size_world[1], 1)
#    # Y = np.arange(0, size_world[0], 1)
#    # X, Y = np.meshgrid(X, Y)
#    # surf = ax.plot_surface(X, Y, p_value, rstride=1, cstride=1, cmap=cm.coolwarm,
#    #         linewidth=0, antialiased=False)
#    # plt.show()
#    
    
