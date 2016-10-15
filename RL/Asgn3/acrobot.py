import numpy as np
import numpy.random as rand
import matplotlib.pyplot as plt
# from scipy.siself.gnal import medfilt
from tiles import *
import itertools

class env():
	"""docstring for env"""
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


	def r_function(self,cur_state):
		return -1.

class basis_fn():
	def fourier(self,cur_state,f_order):
		n_dim = len(cur_state)
		x = np.arange(0,f_order+1)
		c =	[p for p in itertools.product(x, repeat=4)]
		phi = np.zeros(len(c))

		for i in range(len(c)):
			phi[i] = np.cos(np.pi*np.sum([c[i][j]*cur_state[j] for j in range(n_dim)]))

		return phi


class policy():
	def e_greedy(self, q_value):
		eps = 0.1
		p = rand.rand()
		if p<eps: 
			action = rand.randint(-1,2,1)
		else:
			idx = np.argmax(q_value)

			if idx == 0:
				action = -1
			elif idx == 1:
				action = 0
			else: action = 1
			
		return action


if __name__ == '__main__':
	world = env()
	ctrl = policy()
	basis = basis_fn()

	n_eps = 1
	alpha = 0.5
	gamma = 1.0
	labda = 0.9
	n_actions = 3
	
	f_order = 4
	cur_state = [0., 0., 0., 0.]
	phi = basis.fourier(cur_state,f_order)
	weights = np.zeros(len(phi))
	tip = [world.l1*np.cos(cur_state[0]) + world.l2*np.cos(cur_state[0] + cur_state[1]), world.l1*np.sin(cur_state[0]) + world.l2*np.sin(cur_state[0] + cur_state[1])]

	Q_old = np.zeros(n_actions)
	Q = np.zeros(n_actions)
	new_Q = np.zeros(n_actions)
	  

	for ep in range	(n_eps):
		e_vector = np.zeros(len(weights))
		cur_state = [0., 0., 0., 0.]
		Q_old = np.zeros(n_actions)
		phi = basis.fourier(cur_state,f_order)

		while(tip[1] < world.l1):
			print "yo"
			action = ctrl.e_greedy(Q_old)
			new_state = world.simulator(cur_state,action)
			reward = world.r_function(new_state)

			new_phi = basis.fourier(new_state,f_order)

			Q = np.dot(weights,phi)
			new_Q = np.dot(weights,new_phi)

			e_vector = gamma*labda*e_vector + (1. - alpha*gamma*labda*np.dot(e_vector,phi))*phi
			delta = reward + gamma*new_Q - Q

			weights = weights + alpha*(delta + Q - Q_old)*e_vector - alpha*(Q-Q_old)*phi

			Q_old = Q
			phi = new_phi
			cur_state = new_state

			tip = [world.l1*np.cos(cur_state[0]) + world.l2*np.cos(cur_state[0] + cur_state[1]), world.l1*np.sin(cur_state[0]) + world.l2*np.sin(cur_state[0] + cur_state[1])]


		print "terminal_phi = ", phi
