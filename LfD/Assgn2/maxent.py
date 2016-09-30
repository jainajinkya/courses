import numpy as np
import numpy.random as rand
from mpl_toolkits.mplot3d.axes3d import Axes3D
import matplotlib.pyplot as plt
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

  # print "r_weights = ", r_weights

  for t in range(0,horizon):
    for i in range(0,n_states):
      a_sum = 0.
      for j in range(0,n_actions):
        p_update = 0.
        for k in range(0,n_states):
          p_update = p_update + trans_mat[i][j][k]*z_s[k]

        z_a[i][j] = p_update*np.exp(reward[i])
        a_sum = a_sum + z_a[i][j]

      # z_s[i] = np.sum(z_a,axis=1)
      z_s[i] = a_sum

    z_s[-1] = z_s[-1] + 1.

  for i in range(0,n_states):
    for j in range(0,n_actions):
      policy[i][j] = z_a[i][j]/z_s[i]
            
  # print "policy_sum =", np.sum(policy,axis=1)
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

  for k in range(0,n_states):
    for t in range(0,horizon): 
      update = 0.

      for i in range(0,n_states):
        for j in range(0,n_actions):
          update = update + timed_state_freq[i][t]*policy[i][j]*trans_mat[i][j][k]
     
      timed_state_freq[k][t+1] = timed_state_freq[k][t+1] + update

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
  	policy = calcMaxEntPolicy(trans_mat,horizon,r_weights,state_features)
  	state_freq = calcExpectedStateFreq(trans_mat, horizon, start_dist, policy)
  	f_count = np.dot(state_freq,state_features)

  	delta_L = f_tilde - f_count
  	# print "f_tilde =", f_tilde
  	r_weights = r_weights + learning_rate*delta_L

  return r_weights
  
 
 
if __name__ == '__main__':
  
  # Build domain, features, and demos
  trans_mat = build_trans_mat_gridworld()
  state_features = np.eye(26,25)  # Terminal state has no features, forcing zero reward 
  demos = [[0,1,2,3,4,9,14,19,24,25],[0,5,10,15,20,21,22,23,24,25],[0,5,6,11,12,17,18,23,24,25],[0,1,6,7,12,13,18,19,24,25]]
  seed_weights = np.zeros(25)
  
  # Parameters
  n_epochs = 100
  horizon = 10
  learning_rate = 1.0
  
  # Main algorithm call
  r_weights = maxEntIRL(trans_mat, state_features, demos, seed_weights, n_epochs, horizon, learning_rate)
  
  # Construct reward function from weights and state features
  reward_fxn = []
  for s_i in range(25):
    reward_fxn.append( np.dot(r_weights, state_features[s_i]) )
  reward_fxn = np.reshape(reward_fxn, (5,5))
  
  # Plot reward function
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  X = np.arange(0, 5, 1)
  Y = np.arange(0, 5, 1)
  X, Y = np.meshgrid(X, Y)
  surf = ax.plot_surface(X, Y, reward_fxn, rstride=1, cstride=1, cmap=cm.coolwarm,
			linewidth=0, antialiased=False)
  plt.show()