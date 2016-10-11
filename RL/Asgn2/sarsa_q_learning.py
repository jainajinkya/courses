import numpy as np
import numpy.random as rand
from mpl_toolkits.mplot3d.axes3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm


class grid_world:
	def __init__(self):
		self.state = np.zeros((5,12))
		self.t_state = [0,11]
		# self.actions = ['r','l','u','d']

	def state_update(self, cur_state, action):
		n_row = np.shape(self.state)[0]
		n_col = np.shape(self.state)[1]

		new_state = cur_state
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
		if cur_state == self.t_state:
			reward = 50.

		for i in range(len(cliff)):
			if cur_state == cliff[i]:
				reward = -100.
				break

		return reward


class control():
	def e_greedy(self, cur_state,q_value):
		eps = 0.2

		p = rand.rand()
		if p<eps: 
			action = rand.randint(0,4)
		else:
			action = np.argmax(q_value[cur_state[0]][cur_state[1]])

		# if action == 0:
		# 	action = 'r'
		# if action == 1:
		# 	action = 'l'
		# if action == 2:
		# 	action = 'u'
		# if action == 3:
		# 	action = 'd'

		return action

if __name__ == '__main__':
	world = grid_world()
	ctrl = control()
	size_world = np.shape(world.state)
	n_actions = 4
	n_eps = 500
	alpha = 0.2
	disc = 1.0
	
	q_value = rand.rand(size_world[0],size_world[1],n_actions) # action_set = ['r','l','u','d']
	# max_q_value = np.zeros((size_world[0],size_world[1]))
	final_policy = []

	for ep in range(n_eps):
		cur_state = [0,0]
		counter = 0

		while(cur_state != world.t_state and counter<1000):
			action = ctrl.e_greedy(cur_state,q_value)
			new_state = world.state_update(cur_state,action)
			reward = world.r_function(new_state)
			new_action = ctrl.e_greedy(new_state,q_value)

			## SARSA update
			q_value[cur_state[0],cur_state[1],action] = q_value[cur_state[0],cur_state[1],action] + alpha*(reward + disc*q_value[new_state[0],new_state[1],action] - q_value[cur_state[0],cur_state[1],action])

			if reward == -100.:
				new_state = [0,0] ## Restarting

			cur_state = new_state
			counter = counter + 1
		# print "counter = ", counter
	
	# print q_value

	for i in range(np.shape(q_value)[0]):
		for j in range(np.shape(q_value)[1]):
			# max_q_value[i][j] = max(q_value[i][j])
			# print "q_value[i][j]", np.argmax(q_value[i][j])
			final_policy = final_policy + [np.argmax(q_value[i][j])]

	######## Final Trajectory
	print "q_value =", q_value

	cur_state = [0, 0]
	for k in range(len(final_policy)):
		cur_state = world.state_update(cur_state,final_policy[k])
		print "cur_state = ", cur_state
		# print "route =", r
		

	

	# Plot reward function
	# fig = plt.figure()

	# ax = fig.add_subplot(111, projection='3d')
	# X = np.arange(0, size_world[1], 1)
	# Y = np.arange(0, size_world[0], 1)
	# X, Y = np.meshgrid(X, Y)
	# surf = ax.plot_surface(X, Y, max_q_value, rstride=1, cstride=1, cmap=cm.coolwarm,
	# 		linewidth=0, antialiased=False)
	# plt.show()





