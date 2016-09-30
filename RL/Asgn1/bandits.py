import numpy as np
import matplotlib.pyplot as plt
import time

start_time = time.time()

class banditTestbed:
    def __init__(self,n_arms,epsilon):
        self.n = n_arms
        self.counts = [0] * n_arms
        if epsilon==0:
            self.values = [5] * n_arms #try 3, 5 ,10 and 20
        else: self.values = [0.] * n_arms # Initialization of Q_values
        self.values_mean = np.random.randn(n_arms)

    def reward(self,arm):
        rew = np.random.randn() + self.values_mean[arm]
        # return np.random.randn() + self.values_mean[arm]
        return rew

    def update(self,arm,reward):
        """Update an arm with some reward value""" # Example: click = 1; no click = 0
        self.counts[arm] = self.counts[arm] + 1
        value = self.values[arm]
        alpha = 0.1
        new_value = (1-alpha)*value + alpha*reward
        self.values[arm] = new_value
        # print "arm = ", arm,"  New Value = ", self.values[arm]

    def optimal_arm(self,arm):
        actual_arm = np.argmax(self.values_mean)
        if arm==actual_arm:
            return 1.
        else: 
            # print "actual_arm =", actual_arm, "arm = ", arm
            return 0.


class actionSelection:

    def EpsilonGreedy(self,testbed,epsilon):
        """Choose an arm for testing"""
        # epsilon = 0.01 # defininf constant epsilon
        if np.random.random() > epsilon:
            # Exploit (use best arm)
            return np.argmax(testbed.values)
        else:
            # Explore (test all arms)
            return np.random.randint(testbed.n)

    def Greedy(self,testbed):
        return np.argmax(testbed.values)

def main():
    n_arms = 10
    n_testbeds = 2000
    n_steps = 50
    fig1 = plt.figure(1)
    x = range(0,n_steps)

    for epsilon in [0,0.1]:
        avg_reward = [0.]*n_steps # calculating the average reward
        mean_value = [0.]*n_steps
        n_optimal_action = [0.]*n_testbeds
        percent_optimal = [0.]*n_steps
        r_dummy = 0.
        arm_dummy = [0.]*n_testbeds

        bandit = [banditTestbed(n_arms,epsilon) for i in range(n_testbeds)]
        action = actionSelection()

        for i in range(0,n_steps):
            for j in range(0,n_testbeds):
                arm = action.EpsilonGreedy(bandit[j],epsilon)
                arm_dummy[j] = arm
                reward = bandit[j].reward(arm)
                # print "reward = ", reward
                bandit[j].update(arm,reward)
                r_dummy = r_dummy + reward
                n_optimal_action[j] = bandit[j].optimal_arm(arm)

            avg_reward[i] = r_dummy/n_testbeds
            # print np.sum(n_optimal_action)
            percent_optimal[i] = (np.sum(n_optimal_action)/n_testbeds)*100
            n_optimal_action = [0.]*n_testbeds
            r_dummy = 0.
            # print "arm =", arm_dummy

        if epsilon == 0:
            color = 'g'
        elif epsilon == 0.1:
            color = 'k'
        else: color = 'r'
        # plt.plot(x,avg_reward, color)
        plt.plot(x,percent_optimal,color)
        plt.xlegend("Steps")
        plt.ylegend("% Optimal Values")

    fig1.show()
    raw_input()

    # bandit = banditTestbed(n_arms)
    # action = actionSelection()
    
    # for i in range(0,1000):
    #     arm = action.EpsilonGreedy(bandit)
    #     reward = bandit.reward(arm)
    #     bandit.update(arm,reward)
    #     print "reward = ", reward

if __name__ == "__main__":
    main()
    print("--- %s seconds ---" % (time.time() - start_time))

