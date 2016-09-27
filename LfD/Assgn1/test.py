import functions as fn
import numpy as np
import matplotlib.pyplot as plt

def main():
	x_start = 0.
	x_goal = 6*np.pi
	traj_time = 12.
	K = 2500.0
	D = 2*np.sqrt(K)
	singleDemo = False

	demo = fn.demonstration()
	dmp = fn.DMP(traj_time)

	[t,x,y] = demo.single_demo(x_start,x_goal,traj_time)
	[t_n,x_n,y_n] = demo.noisy_demo(x_start,x_goal,traj_time)

	# plt.plot(x,y,'b-',linewidth=3.0)
	# plt.plot(x_n,y_n, 'ro')
	# plt.show()

	demo_x = [t,x]
	demo_y = [t,y]
	param_x = dmp.learning(demo_x,K,D, singleDemo)
	param_y = dmp.learning(demo_y,K,D, singleDemo)

	# DMP Planning
	start = [x[0]+ 0.0, y[0]+ 0.0]
	goal = [x[-1]*2, y[-1] + 0.0]
	[t_new,x_new] = dmp.planning(param_x, start[0], 0.0, goal[0],traj_time*0.5,0.1,singleDemo)
	[t_new,y_new] = dmp.planning(param_y, start[1], 0.0, goal[1],traj_time*0.5,0.1,singleDemo)

	
	## Data
	data_old = [[t[i],x[i],y[i]] for i in range(len(t))]
	data_new = [[t_new[i],x_new[i],y_new[i]] for i in range(len(t_new))]

	f_old = open('data_old.txt', 'w')
	for item in data_old:
		f_old.write("%s\n" % item)

	f_new = open('data_new.txt', 'w')
	for item in data_new:
		f_new.write("%s\n" % item)

	f_new.close()
	f_old.close()

	ref1_x = np.linspace(x[0],x[-1], 100)
	ref1_y = np.linspace(y[0],y[-1], 100)
	ref2_x = np.linspace(start[0],goal[0], 100)
	ref2_y = np.linspace(start[1],goal[1], 100)

	plt.plot(x,y, linewidth=5.0)
	# plt.plot(ref1_x,ref1_y, 'b.')
	# plt.plot(x_new,y_new,'r--', linewidth=3.0)
	plt.plot(x_new,y_new,'ro', linewidth=5.0)
	# plt.plot(ref2_x,ref2_y, 'r.')
	plt.show()
	

if __name__ == "__main__":
    main()
