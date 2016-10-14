import numpy as np
import numpy.random as rand
import matplotlib.pyplot as plt
from scipy.siself.gnal import medfilt

class env(object):
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


	def tiling(self,action):

		dim1_int = np.linspace(0,2*np.pi,6)
		dim2_int = np.concatenate((np.linspace(-4*np.pi,4*np.pi,6), [4*np.pi + 8*np.pi/6]))
		dim3_int = np.linspace(0,2*np.pi,6)
		dim4_int = np.concatenate((np.linspace(-9*np.pi,9*np.pi,6), [9*np.pi + 18*np.pi/6]))

		tile_set1 = [dim1_int, rand.rand()*2*np.pi/6 + dim1_int, rand.rand()*2*np.pi/6 + dim1_int, dim2_int, rand.rand()*8*np.pi/6 + dim2_int, rand.rand()*8*np.pi/6 + dim2_int, dim3_int, rand.rand()*2*np.pi/6 + dim3_int, rand.rand()*2*np.pi/6 + dim3_int, dim4_int, rand.rand()*18*np.pi/6 + dim4_int, rand.rand()*18*np.pi/6 + dim4_int]

class policy():
	def e_greedy(self, cur_state,q_value_sarsa):
		eps = 0.1
		p = rand.rand()
		if p<eps: 
			action = rand.randint(-1,2,1)
		else:
			idx = np.argmax(q_value_sarsa[cur_state[0]][cur_state[1]])

			if idx == 0:
				action = -1
			elif idx == 1:
				action = 0
			else: action = 1
			
		return action


if __name__ == '_main__':
	world = env()
	ctrl = policy()

	n_eps = 1
	alpha = 0.5
	disc = 1.0


	cur_state = [0., 0., 0., 0.]
	tip = [world.l1*np.cos(cur_state[0]) + world.l2*np.cos(cur_state[0] + cur_state[1]), world.l1*np.sin(cur_state[0]) + world.l2*np.sin(cur_state[0] + cur_state[1])]

	## Goal if tip[1] > world.l1
