"""
Variant of the tiger problem intorduced in Cassandra et al, 1994 where
the observation probability varies with time
"""
import numpy as np

from Environment import Environment


class TigerProblem(Environment):
    def __init__(self, obs_indicator_multiplier=1,
                 init_listen_success_prob=0.85):
        self.states = ['tiger_left', 'tiger_right']
        self.actions = ['listen', 'open_left', 'open_right']
        self.observations = ['tiger_left', 'tiger_right']

        self.cur_time_step = 0
        # Set cur_state to None to ensure that reset() must be called to
        # start an episode
        self.cur_state = None

        # The following parameters are to calculate the probability that
        # listening indicates that the tiger is behind the true door
        # This probability for the current time step is held by
        # self.listen_success_prob
        # self.obs_indicator is available to the agent as an indicator
        # of possible change in the observation function
        # self.obs_indicator_multiplier is a parameter specific to this
        # problem and can be interpreted as a factor by which
        # self.obs_indicator is off in indicating the true probability
        # of getting the right observation on listening
        # For this problem,
        #   self.listen_success_prob = self.obs_indicator *
        #                              self.obs_indicator_multiplier
        self.listen_success_prob = init_listen_success_prob
        self.obs_indicator_multiplier = obs_indicator_multiplier
        self.obs_indicator = \
            self.listen_success_prob / self.obs_indicator_multiplier

    def set_listen_success_prob(self, listen_success_prob):
        """
        A generic method that lets you set the listen_success_prob.
        This is kept this way so that we can easily see the effect of
        various types of change in this
        """
        self.listen_success_prob = listen_success_prob
        self.obs_indicator = \
            self.listen_success_prob / self.obs_indicator_multiplier

    def get_obs_indicator(self):
        return self.obs_indicator

    def reset(self):
        self.cur_time_step = 0
        self.cur_state = np.random.choice(self.states)
        return None # None is treated as uniform belief

    def get_other_state(self):
        """
        A useful util that returns the state other than self.cur_state
        """
        if self.cur_state is None:
            return None
        else:
            return [state for state in self.states
                    if state != self.cur_state][0]

    def take_action(self, action):
        if action not in self.actions:
            raise RuntimeError('Invalid action ' + str(action) + '! Permissible' +
                               'actions are ' + str(self.actions))
        if self.cur_state is None:
            raise RuntimeError('Current state is None. Should have started' +
                               ' an episode by calling reset()')

        if action == 'open_left':
            last_state = self.cur_state
            self.cur_state = None   # Forces a need to call reset() for
                                    # next episode
            if last_state == 'tiger_left' :
                return None, -100   # None indicates end of episode
                                    # -100 because agent opened tiger door
            else :
                return None, 10     # None indicates end of episode
                                    # +10 because agent opened treasure door
        elif action == 'open_right':
            last_state = self.cur_state
            self.cur_state = None   # Forces a need to call reset() for
                                    # next episode
            if last_state == 'tiger_right' :
                return None, -100   # None indicates end of episode
                                    # -100 because agent opened tiger door
            else :
                return None, 10     # None indicates end of episode
                                    # +10 because agent opened treasure door
        else:
            # Agent chose to listen
            rand_num = np.random.uniform()
            if rand_num < self.listen_success_prob:
                return self.cur_state, -1   
                    # Listen was successful => Obs = True state
                    # -1 because listen always costs -1
            else:
                return self.get_other_state(), -1 
                    # Listen failed => Obs = Other state
                    # -1 because listen always costs -1
