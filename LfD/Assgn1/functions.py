import numpy as np
import collections
import matplotlib.pyplot as plt


class demonstration:
	def __init__(self):
		self.t_res = 0.1

	def single_demo(self,x_start,x_end,time):
		t = [i*0.1 for i in np.arange(0,time*(1/self.t_res))]
		x = np.arange(x_start,x_end,(x_end-x_start)/len(t))
		y = np.sin(x)

		traj = collections.namedtuple('Demo', ['t', 'x', 'y'])
		demo = traj(t,x=x,y=y)
		return demo

	def noisy_demo(self,x_start,x_end,time):
		t = [i*0.1 for i in np.arange(0,time*(1/self.t_res))]
		x = np.arange(x_start,x_end,(x_end-x_start)/len(t))
		y = np.sin(x)

		sigma = 0.05

		x = [x[i] + sigma*np.random.randn() for i in range(len(x))]
		y = [y[i] + sigma*np.random.randn() for i in range(len(y))]

		traj = collections.namedtuple('Demo', ['t', 'x', 'y'])
		demo = traj(t,x=x,y=y)

		return demo


class DMP:
	def __init__(self,trajectory_time):
		self.alpha = np.log(0.01)
		self.tau = trajectory_time
	
	def phase_variable(self,t):
		s = np.exp(self.alpha*(t/self.tau))
		return s

	def basis_function(self,c,h,s):
		p = np.exp(-h*(s-c)**2)
		return p

	def obs_coupling(self, obs, cur): # Both vectors
		gamma = 100.
		beta = 5.
		cur = np.array(cur)
		obs = np.array(obs)
		r_vec = cur-obs
		mag_r_vec = np.sqrt((r_vec[0])**2 + (r_vec[1])**2)
		p = gamma*np.exp(-beta*r_vec)*(r_vec/mag_r_vec)
		return p

	def radial_fn_approx(self,s,f,n_gauss): # n_gauss = len(t)/10
		c = [s[int(i*len(s)/n_gauss)] for i in range(n_gauss)] ## Kinda equally spaced in time
		h = [n_gauss/c[i] for i in range(len(c))]

		# c = [1 - i*(1.0/n_gauss) for i in range(n_gauss)] ## Equally Spaced in space
		# h = (1./n_gauss)*np.ones([len(c),1])


		# x = np.linspace(-0.5,1.5,100)

		# for i in range(len(c)):
		# 	y = [self.basis_function(h[i],c[i],x[j]) for j in range(len(x))]
		# 	plt.plot(x,y)

		# plt.show()

		psi = np.zeros((len(s),n_gauss))
	
		for i in range(len(s)):
			for j in range(n_gauss):
				psi[i][j] = self.basis_function(c[j],h[j],s[i])

		A = [s[i]*psi[i][:] for i in range(len(s))]
		A_inv = np.linalg.pinv(A)
		w = np.dot(A_inv,f)
		return [w,c,h]



	def learning(self,demo, K, D, single_demo): 
		[t,x] = demo
	
		x_0 = x[0]
		g_x = x[-1]
	
		x_dot = np.zeros([len(x),1])
		x_ddot = np.zeros([len(x),1])
		
		for i in range(1, len(t)):
			x_dot[i-1] = (x[i] - x[i-1])/(t[i] - t[i-1])
			
			if i>1:
				x_ddot[i-1] = (x_dot[i-1] - x_dot[i-2])/(t[i] - t[i-1])
			
		v_x = [self.tau*x_dot[i] for i in range(len(x_dot))]
		v_dot_x = [self.tau*x_ddot[i] for i in range(len(x_ddot))]
		
		s = np.zeros([len(t),1])
		f_x = np.zeros([len(t),1])
		
		for i in range(len(t)):
			s[i] = self.phase_variable(t[i])
			f_x[i] = (self.tau*v_dot_x[i] + D*v_x[i])/K - (g_x-x[i]) + (g_x-x_0)*s[i]
		
		if(single_demo):
			params = [K,D,s,f_x]
		else:
			[w,c,h] = self.radial_fn_approx(s,f_x,int(round(len(t)/2)))
			params = [K,D,w,c,h]

		return params


	def planning(self,params,start,vel_0,goal,traj_time,dt,single_demo):
		t = [i*dt for i in np.arange(0,traj_time*(1/dt))]
		
		self.tau = traj_time
		s = [self.phase_variable(t[i]) for i in range(len(t))]

		x = [start]
		v = [vel_0]
		K = params[0]
		D = params[1]
		f = [0.]

		if(single_demo):
			s_learnt = params[2]
			f_learnt = params[3]

			for i in range(len(s)):
				if( [j for j in range(len(s_learnt)) if s[i]>=s_learnt[j]] and [j for j in range(len(s_learnt)) if s[i]<s_learnt[j]]):
					l_s =  min([j for j in range(len(s_learnt)) if s[i]>=s_learnt[j]])
					h_s =  max([j for j in range(len(s_learnt)) if s[i]<s_learnt[j]])
					s1 = s_learnt[l_s]
					f1 = f_learnt[l_s]
					s2 = s_learnt[h_s]
					f2 = f_learnt[h_s]

					f = f + [f1 + ((f2-f1)/(s2-s1))*(s[i]-s1)]
				else:
					f = f + [f[-1]]
						
				x = x + [x[i]+(v[i]*dt)/self.tau]
				v = v + [(dt/self.tau)*(K*(goal-x[i]) - D*v[i] - K*(goal-x[0])*s[i] + K*f[i+1]) + v[i]]

		
		else:
			w = params[2]
			c = params[3]
			h = params[4]

			for i in range(len(s)):
				w_s = 0.
				for j in range(len(w)):
					w_s = w_s + w[j]*self.basis_function(c[j],h[j],s[i])

				f = f + [w_s*s[i]]

				x = x + [x[i]+(v[i]*dt)/self.tau]
				v = v + [(dt/self.tau)*(K*(goal-x[i]) - D*v[i] - K*(goal-x[0])*s[i] + K*f[i+1]) + v[i]]
			 
			# x_s = np.linspace(-0.5,1.5,100)

			# for i in range(len(c)):
			# 	y_s = [self.basis_function(h[i],c[i],t[j]) for j in range(len(t))]
			# 	plt.plot(t,y_s)

			# plt.show()


		return [t,x[0:-1]]

	def planning_w_obs(self,params,start,vel_0,goal,traj_time,dt):
		t = [i*dt for i in np.arange(0,traj_time*(1/dt))]

		self.tau = traj_time
		s = [self.phase_variable(t[i]) for i in range(len(t))]

		x = np.array([start])
		v = np.array([vel_0])
		K = np.matrix([[params[0] 0],[0 params[0]]])
		D = np.matrix([[params[1] 0],[0 params[1]]])
		f = np.array([0. , 0.])

		s_learnt = params[2]
		f_learnt_x = params[3]
		f_learnt_y = params[4]

		for i in range(len(s)):
			if( [j for j in range(len(s_learnt)) if s[i]>=s_learnt[j]] and [j for j in range(len(s_learnt)) if s[i]<s_learnt[j]]):
				l_s =  min([j for j in range(len(s_learnt)) if s[i]>=s_learnt[j]])
				h_s =  max([j for j in range(len(s_learnt)) if s[i]<s_learnt[j]])
				s1 = s_learnt[l_s]
				f1_x = f_learnt_x[l_s]
				f1_y = f_learnt_y[l_s]
				s2 = s_learnt[h_s]
				f2_x = f_learnt_x[h_s]
				f2_y = f_learnt_y[h_s]

				f = f + [f1 + ((f2-f1)/(s2-s1))*(s[i]-s1)]
			else:
				f = f + [f[-1]]
					
			x = x + [x[i]+(v[i]*dt)/self.tau]
			v = v + [(dt/self.tau)*(K*(goal-x[i]) - D*v[i] - K*(goal-x[0])*s[i] + K*f[i+1]) + v[i]]

				 
			# x_s = np.linspace(-0.5,1.5,100)

			# for i in range(len(c)):
			# 	y_s = [self.basis_function(h[i],c[i],t[j]) for j in range(len(t))]
			# 	plt.plot(t,y_s)

			# plt.show()


		return [t,x[0:-1]]





				
			








