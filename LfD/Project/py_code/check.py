# Generate data for control problem.
import numpy as np
from cvxpy import *
import matplotlib.pyplot as plt

if __name__ == '__main__':
	np.random.seed(1)
	nState = 2
	nInput = 2
	T = 50
	
	A = np.eye(nState)
	B = np.eye(nState,nInput)
	C = np.eye(nState)

	Q = 0.5*np.eye(nState)
	R = 0.5*np.eye(nInput)
	Q_f = 20*np.eye(nState)
	labda = 200
	

	# Initialize
	x_0 = np.array([2.,0.])
	m_0 = np.array([2.5,2.5])
	sig = 5*np.eye(nState)
	# Form and solve control problem.

	x = Variable(nState, T+1)
	m = Variable(nState, T+1)
	s = Variable(1,T+1)
	u = Variable(nInput, T)

	belief_state = []
	for t in range(T):
		# cost = sum_squares(x[:,t+1]) + sum_squares(u[:,t])
		cost = m[:,t+1].T*Q*m[:,t+1] + u[:,t].T*R*u[:,t]

		if(t==T-1):
			cost = m[:,t+1].T*Q*m[:,t+1] + u[:,t].T*R*u[:,t] + s[:,T].T*labda*s[:,T]

		gamma = A*sig*A.T
		W = (0.5*(5.0-power(m[1,t+1],2))*np.eye(nState))
		# print "C*gamma*C.T + W = ", power(C*gamma*C.T + W,-1)
		sig = gamma - gamma*C.T*(power(C*gamma*C.T + W,-1))*C*gamma
		# sig = gamma - quad_form(gamma*C.T,power(C*gamma*C.T + W,-1))
		
		constr = [x[:,t+1] == A*x[:,t] + B*u[:,t],
				  m[:,t+1] == A*m[:,t] + B*u[:,t],
				  s[:,t+1] == sig[1,1],
				  norm(u[:,t], 'inf') <= 1]
		belief_state.append( Problem(Minimize(cost), constr) )
	# sums problem objectives and concatenates constraints.


	prob = sum(belief_state)
	prob.constraints += [x[:,T] == 0, x[:,0] == x_0]
	prob.constraints += [m[:,T] == 0, m[:,0] == m_0]
	prob.solve()



	# Plot results.

	# %matplotlib inline
	# %config InlineBackend.figure_format = 'svg'

	f = plt.figure()

	# Plot (u_t)_1.
	ax = f.add_subplot(411)
	plt.plot(u[0,:].value.A.flatten())
	plt.ylabel(r"$(u_t)_1$", fontsize=16)
	plt.yticks(np.linspace(-1.0, 1.0, 3))
	plt.xticks([])

	# Plot (u_t)_2.
	plt.subplot(4,1,2)
	plt.plot(u[1,:].value.A.flatten())
	plt.ylabel(r"$(u_t)_2$", fontsize=16)
	plt.yticks(np.linspace(-1, 1, 3))
	plt.xticks([])

	# Plot (x_t)_1.
	plt.subplot(4,1,3)
	x1 = x[0,:].value.A.flatten()
	plt.plot(x1)
	plt.ylabel(r"$(x_t)_1$", fontsize=16)
	plt.yticks([-10, 0, 10])
	plt.ylim([-10, 10])
	plt.xticks([])

	# Plot (x_t)_2.
	plt.subplot(4,1,4)
	x2 = x[1,:].value.A.flatten()
	plt.plot(range(51), x2)
	plt.yticks([-25, 0, 25])
	plt.ylim([-25, 25])
	plt.ylabel(r"$(x_t)_2$", fontsize=16)
	plt.xlabel(r"$t$", fontsize=16)
	plt.tight_layout()
	plt.show()