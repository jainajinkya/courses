import numpy as np
import numpy.random as rand
from mpl_toolkits.mplot3d.axes3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
# from scipy.signal import savgol_filter
from scipy.signal import medfilt



class grid_world:
	def __init__(self):
		self.state = np.zeros((5,12))
		self.t_state = [0, np.shape(self.state)[1]-1]

	def state_update(self, cur_state, action):
		n_row = np.shape(self.state)[0]
		n_col = np.shape(self.state)[1]

		# new_state = cur_state
		new_state = [cur_state[0],cur_state[1]]
		## action_set = ['r', 'l', 'u', 'd']

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


	def r_function(self,cur_state):
		cliff = [[0,i] for i in range(1,np.shape(self.state)[1]-1)]
		reward = -1.
		
		# if cur_state == self.t_state:
		# 	reward = 50.

		for i in range(len(cliff)):
			if cur_state == cliff[i]:
				reward = -100.
				break

		return reward


class control():
	def e_greedy(self, cur_state,q_value_sarsa):
		eps = 0.1
		p = rand.rand()
		if p<eps: 
			action = rand.randint(0,4)
		else:
			action = np.argmax(q_value_sarsa[cur_state[0]][cur_state[1]])
			
		return action


if __name__ == '__main__':
	world = grid_world()
	ctrl = control()
	size_world = np.shape(world.state)
	n_actions = 4
	n_eps = 500
	alpha = 0.5
	disc = 1.0

	# q_value_sarsa = 100*rand.rand(size_world[0],size_world[1],n_actions) # action_set = ['r','l','u','d']
	q_value_sarsa = np.zeros((size_world[0],size_world[1],n_actions)) 
	q_value_q_learn = np.zeros((size_world[0],size_world[1],n_actions)) 
	# q_value_sarsa[world.t_state[0]][world.t_state[1]] = np.zeros(n_actions)
	
	max_q_value_sarsa = []
	max_q_value_q_learn = []
	total_reward_sarsa = np.zeros(n_eps)
	total_reward_q_learn = np.zeros(n_eps)



	for ep in range(n_eps):
		cur_state = [0,0]
		counter = 0
		reward_sarsa = 0.0

		###### SARSA###########################################
		action = ctrl.e_greedy(cur_state,q_value_sarsa)

		while(cur_state != world.t_state):
			new_state = world.state_update(cur_state,action)
			reward = world.r_function(new_state)
			new_action = ctrl.e_greedy(new_state,q_value_sarsa)

			q_value_sarsa[cur_state[0],cur_state[1],action] = q_value_sarsa[cur_state[0],cur_state[1],action] + alpha*(reward + disc*q_value_sarsa[new_state[0],new_state[1],new_action] - q_value_sarsa[cur_state[0],cur_state[1],action])

			if reward == -100.:
				new_state = [0,0] ## Restarting

			cur_state = new_state
			action = new_action
			counter = counter + 1
			reward_sarsa = reward_sarsa + reward

		print "counter = ", counter
		# print "total_reward_sarsa =", total_reward_sarsa
		total_reward_sarsa[ep] = reward_sarsa


		############################################################
		###### Q-Learning###########################################
		cur_state = [0,0]
		counter = 0
		reward_q_learn = 0.0

		while(cur_state != world.t_state):
			action = ctrl.e_greedy(cur_state,q_value_q_learn)
			new_state = world.state_update(cur_state,action)
			reward = world.r_function(new_state)
			best_action = np.argmax(q_value_q_learn[new_state[0],new_state[1]])
			
			q_value_q_learn[cur_state[0],cur_state[1],action] = q_value_q_learn[cur_state[0],cur_state[1],action] + alpha*(reward + disc*q_value_q_learn[new_state[0],new_state[1],best_action] - q_value_q_learn[cur_state[0],cur_state[1],action])

			if reward == -100.:
				new_state = [0,0] ## Restarting

			cur_state = new_state
			counter = counter + 1
			reward_q_learn = reward_q_learn + reward

		print "counter = ", counter
		# print "total_reward_q_learn =", total_reward_q_learn
		total_reward_q_learn[ep] = reward_q_learn


	for i in range(np.shape(q_value_sarsa)[0]):
		for j in range(np.shape(q_value_sarsa)[1]):
			max_q_value_sarsa = max_q_value_sarsa + [np.argmax(q_value_sarsa[i][j])]
			max_q_value_q_learn = max_q_value_q_learn + [np.argmax(q_value_q_learn[i][j])]

	########## Final Trajectory

	###### SARSA #########
	cur_state = [0,0]
	route_sarsa = []
	route_sarsa = route_sarsa + [cur_state[:]]
	k = 0
	print "max_q_value_sarsa", max_q_value_sarsa
	while(cur_state != world.t_state and k<len(max_q_value_sarsa)):
		cur_state = world.state_update(cur_state,max_q_value_sarsa[k])
		route_sarsa = route_sarsa + [cur_state[:]]
		k = k+1
		
	print "route_sarsa = ", route_sarsa

	########## Q-Learning ########
	cur_state = [0,0]
	route_q_learn = []
	route_q_learn = route_q_learn + [cur_state[:]]
	k = 0

	while(cur_state != world.t_state and k<len(max_q_value_q_learn)):
		cur_state = world.state_update(cur_state,max_q_value_q_learn[k])
		route_q_learn = route_q_learn + [cur_state[:]]
		k = k+1
		
	print "route_q_learn = ", route_q_learn
	

	step_sz = 50
	filt_r_s = np.zeros(len(total_reward_q_learn))
	filt_r_q = np.zeros(len(total_reward_q_learn))
	for i in np.arange(0,len(total_reward_q_learn)):
		filt_r_q[i] = np.mean(total_reward_q_learn[i:i+step_sz])
		filt_r_s[i] = np.mean(total_reward_sarsa[i:i+step_sz])



	# Plot reward function
	fig = plt.figure()
	x_vec = np.arange(0,n_eps)
	# filtered_rew_sarsa = savgol_filter(total_reward_sarsa, 5, 1)
	# filtered_rew_q_learn = savgol_filter(total_reward_q_learn, 5, 1)
	filtered_rew_sarsa = medfilt(total_reward_sarsa, 9)
	# filtered_rew_q_learn = medfilt(total_reward_q_learn, 5)
	plt.plot(x_vec,filtered_rew_sarsa, 'r')
	# plt.plot(x_vec,filtered_rew_q_learn, 'b')
	# plt.plot(x_vec,filt_r_s, 'r')
	plt.plot(x_vec,filt_r_q, 'b')
	axes = plt.gca()
	axes.set_xlim([0,n_eps])
	axes.set_ylim([-100, 0])
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
	# 		linewidth=0, antialiased=False)
	# plt.show()





