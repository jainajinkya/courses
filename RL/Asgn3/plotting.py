import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib import animation


def double_pendulum_plot(data):
	## Plot ------------------
	# fig1 = plt.figure
	# for i in range(0,np.size(data,axis=0)):
	# 	if i< np.size(data,axis=0)-1:
	# 		direct = data[i+1,2]
	# 	pt1 = [0.,0.]
	# 	pt2 = [math.cos(data[i,1]), math.sin(data[i,1])]
	# 	u = prep_action(pt1,pt2,direct)
	# 	l_y = np.linspace(pt1[1],pt2[1],num=10)
	# 	l_x = np.linspace(pt1[0], pt2[0], num=10)
	# 	u_y = np.linspace(pt2[1], 0.1*u[1]+pt2[1],num=10)
	# 	u_x = np.linspace(pt2[0], 0.1*u[0]+pt2[0],num=10)
	# 	plt.plot(l_x,l_y,'b')
	# 	plt.plot(u_x,u_y,'r')
	
	# plt.axis([-2, 2, -2, 2])
	# plt.show()

	####Animation -----------------------------------
	fig = plt.figure()
	ax = fig.add_subplot(111, autoscale_on=False, xlim=(-2.5, 2.5), ylim=(-2.5, 2.5))
	ax.grid()

	line, = ax.plot([], [], '-', lw=4)
	line1, = ax.plot([], [], '-r', lw=4)
	time_template = 'time = %.1fs'
	time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)


	def init():
	    line.set_data([], [])
	    line1.set_data([], [])
	    time_text.set_text('')
	    return line, line1, time_text

	def animate(i):
		res = 0.05
		pt1 = [data[0,i], data[1,i]]
		pt2 = [data[2,i],data[3,i]]
		l_y = np.linspace(0.,pt1[1],num=10)
		l_x = np.linspace(0., pt1[0], num=10)
		u_y = np.linspace(pt1[1], pt2[1],num=10)
		u_x = np.linspace(pt1[0], pt2[0],num=10)
		j1_x = np.zeros(10)
		j1_y = np.zeros(10)
		j2_x = pt1[0]*np.ones(10)
		j2_y = pt1[1]*np.ones(10)

		thisx = [l_x, u_x,j1_x,j2_x]
		thisy = [l_y, u_y,j1_y, j2_y]
		
		line.set_data(thisx[0], thisy[0])
		line1.set_data(thisx[1], thisy[1])
		time_text.set_text(time_template % (i))
		return line, line1,time_text

	ani = animation.FuncAnimation(fig, animate, range(0,np.size(data,axis=1)),
	                              interval=500, blit=True, init_func=init)

	ani.save('double_pendulum1.mp4', writer='mencoder', fps=5)



if __name__ == "__main__":
	dat = np.loadtxt('data.txt', delimiter = '\t',unpack=True)
	print len(dat[0])
	# double_pendulum_plot(dat[:,3400:-1])
	double_pendulum_plot(dat)

