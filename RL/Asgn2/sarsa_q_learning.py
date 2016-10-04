import numpy as np
import numpy.random as rand
from mpl_toolkits.mplot3d.axes3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm


class grid_world:
	def __init__(self):
		self.state = np.zeros((4,12))
		# self.actions = ['r','l','u','d']

	def state_update(self, cur_state, action):
		n_row = np.shape(self.state)[0]
		n_col = np.shape(self.state)[1]

		new_state = cur_state

		if action == 'r':
			if cur_state[1]< (n_col-1):
				new_state[1] = cur_state[1] + 1
			else:
				new_state[1] = cur_state[1]

		if action == 'l':
			if cur_state[1] > 0:
				new_state[1] = cur_state[1] - 1
			else:
				new_state[1] = cur_state[1]

		if action == 'u':
			if cur_state[0] < (n_row-1):
				new_state[0] = cur_state[0] + 1
			else:
				new_state[0] = cur_state[0]

		if action == 'd':
			if cur_state[0] > 0:
				new_state[0] = cur_state[0] - 1
			else:
				new_state[0] = cur_state[0]

		return new_state


	def r_function(self,cur_state):
		cliff = [[0,i] for i in range(1,np.shape(self.state)[0]-2)]

		reward = -1

		for i in range(len(cliff)):
			if cur_state == cliff[i]:
				reward = -100
				break

		return reward


class control():
	def e_greedy(self, cur_state,q_value):
		eps = 0.1

		p = rand.rand()
		if p<0.1: 
			action = rand.randint(0,4)
		else:
			action = np.argmax(q_value[cur_state[0]][cur_state[1]])

		if action == 0:
			action = 'r'
		if action == 1:
			action = 'l'
		if action == 2:
			action = 'u'
		if action == 3:
			action = 'd'

		return action

	# def policy(self):

if __name__ == '__main__':
	world = grid_world()
	ctrl = control()
	size_world = np.shape(world.state)
	n_actions = 4

	cur_state = [3,0]
	q_value = np.zeros((size_world[0],size_world[1],n_actions)) # action_set = ['r','l','u','d']
	max_q_value = np.zeros((size_world[0],size_world[1]))

	for i in range(10):
		# action = 'd'
		action = ctrl.e_greedy(cur_state,q_value)
		print "(state, action) = ", cur_state, action
		
		new_state = world.state_update(cur_state,action)
		reward = world.r_function(new_state)

		q_value[new_state[0],new_state[1],0] = reward
		if reward == -100:
			new_state = [0,0]

		cur_state = new_state

	for i in range(np.shape(q_value)[0]):
		for j in range(np.shape(q_value)[1]):
			max_q_value[i][j] = min(q_value[i][j])


	# Plot reward function
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	X = np.arange(0, size_world[1], 1)
	Y = np.arange(0, size_world[0], 1)
	X, Y = np.meshgrid(X, Y)
	surf = ax.plot_surface(X, Y, max_q_value, rstride=1, cstride=1, cmap=cm.coolwarm,
			linewidth=0, antialiased=False)
	plt.show()





