import numpy as np
import numpy.random as rand
from mpl_toolkits.mplot3d.axes3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm

        
def build_trans_mat_gridworld():
  # 5x5 gridworld laid out like:
  # 0  1  2  3  4
  # 5  6  7  8  9 
  # ...
  # 20 21 22 23 24
  # where 24 is a goal state that always transitions to a 
  # special zero-reward terminal state (25) with no available actions
  trans_mat = np.zeros((26,4,26))
  
  # NOTE: the following iterations only happen for states 0-23.
  # This means terminal state 25 has zero probability to transition to any state, 
  # even itself, making it terminal, and state 24 is handled specially below.
  
  # Action 0 = down
  for s in range(24):
    if s < 20:
      trans_mat[s,0,s+5] = 1
    else:
      trans_mat[s,0,s] = 1
      
  # Action 1 = up
  for s in range(24):
    if s >= 5:
      trans_mat[s,1,s-5] = 1
    else:
      trans_mat[s,1,s] = 1
      
  # Action 2 = left
  for s in range(24):
    if s%5 > 0:
      trans_mat[s,2,s-1] = 1
    else:
      trans_mat[s,2,s] = 1
      
 # Action 3 = right
  for s in range(24):
    if s%5 < 4:
      trans_mat[s,3,s+1] = 1
    else:
      trans_mat[s,3,s] = 1

  # Finally, goal state always goes to zero reward terminal state
  for a in range(4):
    trans_mat[24,a,25] = 1  
      
  return trans_mat

           
def calcMaxEntPolicy(trans_mat, horizon, r_weights, state_features):
  """
  For a given reward function and horizon, calculate the MaxEnt policy that gives equal weight to equal reward trajectories
  
  trans_mat: an S x A x S' array of transition probabilites from state s to s' if action a is taken
  horizon: the finite time horizon (int) of the problem for calculating state frequencies
  r_weights: a size F array of the weights of the current reward function to evaluate
  state_features: an S x F array that lists F feature values for each state in S
  
  return: an S x A policy in which each entry is the probability of taking action a in state s
  """
  n_states = np.shape(trans_mat)[0]
  n_actions = np.shape(trans_mat)[1]
  policy = np.zeros((n_states,n_actions))

  z_s = np.zeros((n_states,1))
  z_a = np.zeros((n_states,n_actions))

  z_s[-1] = 1.
  reward = state_features.dot(r_weights)

  for t in range(0,horizon):
    for i in range(0,n_states):
      a_sum = 0.
      for j in range(0,n_actions):
        p_update = 0.
        for k in range(0,n_states):
          p_update = p_update + trans_mat[i][j][k]*z_s[k]

        z_a[i][j] = p_update*np.exp(reward[i])
        a_sum = a_sum + z_a[i][j]

      if i == n_states-1:
        z_s[i] = 1. + a_sum
      else: z_s[i] = a_sum

  for i in range(0,n_states):
    for j in range(0,n_actions):
      policy[i][j] = z_a[i][j]/z_s[i]
            
  # print "policy_sum =", np.sum(policy,axis=1)
  # print "policy =", policy
  return policy
  
def calcExpectedStateFreq(trans_mat, horizon, start_dist, policy):
  """0 finite time horizon (int) of the problem for calculating state frequencies
  start_dist: a size S array of starting start probabilities - must sum to 1
  policy: an S x A array array of probabilities of taking action a when in state s
  
  return: a size S array of expected state visitation frequencies
  """
  state_freq = np.zeros(len(start_dist))
  n_states = np.shape(trans_mat)[0]
  n_actions = np.shape(trans_mat)[1]
  timed_state_freq = np.zeros((n_states,horizon+1))
  timed_state_freq[:,0] = start_dist
  # print "timed_state_freq =", timed_state_freq

  for k in range(0,n_states):
    for t in range(0,horizon): 
      update = 0.
      for i in range(0,n_states):
        for j in range(0,n_actions):
          update = update + timed_state_freq[i][t]*policy[i][j]*trans_mat[i][j][k]
          # if k==n_states-2:
          #   print "timed_state_freq[i][t]*policy[i][j]= ", timed_state_freq[i][t]*policy[i][j]*trans_mat[i][j][k]
                        
      timed_state_freq[k][t+1] = update

      # print "np.dot(policy.transpose(),trans_mat[:,:,k]) = ", np.shape(np.dot(timed_state_freq.transpose(),np.dot(policy,trans_mat[:,:,k].transpose())))

    state_freq[k] = np.sum(timed_state_freq[k])  
  
  return state_freq


def maxEntIRL(trans_mat, state_features, demos, seed_weights, n_epochs, horizon, learning_rate):
  """
  Compute a MaxEnt reward function from demonstration trajectories
  
  trans_mat: an S x A x S' array that describes transition probabilites from state s to s' if action a is taken
  state_features: an S x F array that lists F feature values for each state in S
  demos: a list of lists containing D demos of varying lengths, where each demo is series of states (ints)
  seed_weights: a size F array of starting reward weights
  n_epochs: how many times (int) to perform gradient descent steps
  horizon: the finite time horizon (int) of the problem for calculating state frequencies
  learning_rate: a multiplicative factor (float) that determines gradient step size
  
  return: a size F array of reward weights
  """
  
  n_states = np.shape(state_features)[0]
  n_features = np.shape(state_features)[1]
  r_weights = np.zeros(n_features)
  m = len(demos)
  f_tilde = np.zeros(n_features)

  r_weights = seed_weights
  start_dist = np.zeros(n_states)
  start_dist[0] = 1.

  for i in range(m):
  	for j in range(len(demos[i])):
  		f_tilde = f_tilde + state_features[demos[i][j]]

  f_tilde = f_tilde/m
  # print "f_tilde = ", f_tilde
  
  for i in range(n_epochs):
    f_count = np.zeros(n_features)
    policy = calcMaxEntPolicy(trans_mat,horizon,r_weights,state_features)
    state_freq = calcExpectedStateFreq(trans_mat, horizon, start_dist, policy)
    
    f_count = np.dot(state_freq,state_features)
    # print "fcount = ", f_count

    delta_L = f_tilde - f_count
    delta_L[-1] = 0.
    # print "delta_L =", delta_L
    # print "delta_L =", np.linalg.norm(delta_L)
    r_weights = r_weights + learning_rate*delta_L
    l_norm = np.linalg.norm(delta_L)

  # print "state_freq =", state_freq
  return r_weights, l_norm
 
 
if __name__ == '__main__':
  
  # Build domain, features, and demos
  trans_mat = build_trans_mat_gridworld()
  state_features = np.eye(26,25)  # Terminal state has no features, forcing zero reward 
  demos = [[0,1,2,3,4,9,14,19,24,25],[0,5,10,15,20,21,22,23,24,25],[0,5,6,11,12,17,18,23,24,25],[0,1,6,7,12,13,18,19,24,25]]
  seed_weights = np.zeros(25)
  # seed_weights = -1*np.ones(25)
  
  # # Parameters
  n_epochs = 100
  horizon = 10
  learning_rate = 1.5
  
  # Main algorithm call
  # r_weights = maxEntIRL(trans_mat, state_features, demos, seed_weights, n_epochs, horizon, learning_rate)
  # # print "error =", np.linalg.norm(l_grad)
  # # Construct reward function from weights and state features
  # reward_fxn = []
  # for s_i in range(25):
  #   reward_fxn.append( np.dot(r_weights, state_features[s_i]) )
  # reward_fxn = np.reshape(reward_fxn, (5,5))

  # reward_fxn[4,4] = 40.0
  # ##Plot reward function
  # fig = plt.figure()
  # ax = fig.add_subplot(111, projection='3d')
  # X = np.arange(0, 5, 1)
  # Y = np.arange(0, 5, 1)
  # X, Y = np.meshgrid(X, Y)
  # surf = ax.plot_surface(X, Y, reward_fxn, rstride=1, cstride=1, cmap=cm.coolwarm,
		# 	linewidth=0, antialiased=False)

  # plt.show()

  
  ############# Q1 #############################
  # alpha = np.arange(0.0,3.0,0.05)
  # # # alpha = np.arange(0.0,1.0,0.5)
  # # av_r = np.zeros(len(alpha))
  # # av_r2 = np.zeros(len(alpha))
  # # av_r3 = np.zeros(len(alpha))
  # L_grad = np.zeros(len(alpha))

  # for k in range(len(alpha)):
  #   learning_rate = alpha[k]
  #   r_weights,l_grad = maxEntIRL(trans_mat, state_features, demos, seed_weights, n_epochs, horizon, learning_rate)
  #   L_grad[k] = np.linalg.norm(l_grad)
  #   # r_dummy = np.zeros(4)
  #   # for i in range(len(demos)):
  #   #   av_reward = 0.0
  #   #   for j in range(len(demos[i])):
  #   #     av_reward = av_reward + (np.dot(r_weights, state_features[demos[i][j]]))/len(demos[i])

  #   #   r_dummy[i] = av_reward
    
  #   # av_r[k] = np.sum(r_dummy)/4
  #   # av_r2[k] = max(r_dummy)
  #   # av_r3[k] = min(r_dummy)


  # # plot1, = plt.plot(alpha,av_r, linewidth=5.0, label = 'Average value of rewards')
  # # plot2, = plt.plot(alpha,av_r2,'r--', linewidth=5.0, label = 'Maximum value of rewards')
  # # plot3, = plt.plot(alpha,av_r3,'k--', linewidth=5.0, label = 'Minimum value of rewards')

  # # plt.ylabel('Average Reward for the Demos')
  # # plt.xlabel('Learning rate')
  # # plt.legend([plot1, plot2,plot3], ['Average value of rewards', 'Maximum value of rewards', 'Minimum value of rewards'], loc=2)
  # # # plt.legend()
  # # matplotlib.rcParams.update({'font.size': 18})
  # plt.plot(alpha,L_grad)
  # plt.ylabel('Gradient Value')
  # plt.xlabel('Learning rate')
  # matplotlib.rcParams.update({'font.size': 18})
  # plt.show()
  # print "av_reward = ", av_reward
 #########################################

 ######Q2################
  # av_r = np.zeros(4)
  # # av_r2 = np.zeros(len(alpha))
  # # av_r3 = np.zeros(len(alpha))
  # # for k in range(len(alpha)):
  # # r_weights = maxEntIRL(trans_mat, state_features, demos, seed_weights, n_epochs, horizon, learning_rate)
  # r_dummy = np.zeros(4)
  # r_fxn = np.reshape(reward_fxn, (25,1))
  # for i in range(len(demos)):
  # 	av_reward = 0.0
  # 	for j in range(len(demos[i])):
  # 		av_reward = av_reward + (np.dot(r_weights, state_features[demos[i][j]]))/len(demos[i])

  # 	r_dummy[i] = av_reward
 	# av_r[i] = ((max(r_fxn[0:-1]) - (np.sum(r_dummy[i]))/4)/max(r_fxn[0:-1]))*100.

  # print "av_rewards = ", r_dummy
  # print "max_reward = ", max(r_fxn[0:-1])
  # X_1 = np.arange(1,5,1)
  # plot1, = plt.plot(X_1,av_r, linewidth=5.0)
  # # # plot2, = plt.plot(alpha,av_r2,'r--', linewidth=5.0, label = 'Maximum value of rewards')
  # # # plot3, = plt.plot(alpha,av_r3,'k--', linewidth=5.0, label = 'Minimum value of rewards')
  # plt.ylabel('% Difference in Rewards of Demos and Maximum possible reward')
  # plt.xlabel('Demo')
  # # # plt.legend([plot1, plot2,plot3], ['Average value of rewards', 'Maximum value of rewards', 'Minimum value of rewards'], loc=2)
  # # # plt.legend()
  # matplotlib.rcParams.update({'font.size': 18})
  # plt.show()

  # ############## Q3 #############
  seed_weights = -1*np.ones(25)
  seed_weights[-1] = 10

  # Parameters
  n_epochs = 100
  horizon = 100
  learning_rate = 1.9

  new_policy = calcMaxEntPolicy(trans_mat,horizon,seed_weights,state_features)
  
  for i in range(len(new_policy)):
    print "i =", i, "dir =", np.argmax(new_policy[i,:])


  # r_weights = maxEntIRL(trans_mat, state_features, demos, seed_weights, n_epochs, horizon, learning_rate)
  # optimal_policy = calcMaxEntPolicy(trans_mat,horizon,r_weights,state_features)
  # print "optimal_policy = " , optimal_policy

  r_weights = seed_weights
  reward_fxn = []
  for s_i in range(25):
    reward_fxn.append( np.dot(r_weights, state_features[s_i]) )
  reward_fxn = np.reshape(reward_fxn, (5,5))

  # reward_fxn[4,4] = 40.0
  ##Plot reward function
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  X = np.arange(0, 5, 1)
  Y = np.arange(0, 5, 1)
  X, Y = np.meshgrid(X, Y)
  surf = ax.plot_surface(X, Y, reward_fxn, rstride=1, cstride=1, cmap=cm.coolwarm,
			linewidth=0, antialiased=False)

  plt.show()
