"""
A general interface for environments. These are POMDP environments so we
need the class doing some tracking of the current state because ideally
this should not be presented to an agent.
"""


class Environment:
    def reset(self):
        """
        Resets the environment to an initial state.
        """
        raise NotImplementedError('Subclass must implement function ' +
                                  'reset(self)')

    def take_action(self, action):
        """
        Transitions to the next state based on action.
        Returns observation, reward from this next state, action pair.
        observation=None indicates end of episode in an episodic task
        """
        raise NotImplementedError('Subclass must implement ' +
                                  'take_action(self, action)')

    def get_obs_indicator(self):
        """
        To allow an agent to get the indicator of change in the
        observation functiom
        """
        raise NotImplementedError('Subclass must implement ' +
                                  'get_obs_indicator(self)')
