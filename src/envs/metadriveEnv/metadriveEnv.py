import pdb

from envs.multiagentenv import MultiAgentEnv
import numpy as np

from metadrive import (
    MultiAgentMetaDrive,
    MultiAgentTollgateEnv,
    MultiAgentBottleneckEnv,
    MultiAgentIntersectionEnv,
    MultiAgentRoundaboutEnv,
    MultiAgentParkingLotEnv
)

envs_classes = dict(
    roundabout=MultiAgentRoundaboutEnv,
    intersection=MultiAgentIntersectionEnv,
    tollgate=MultiAgentTollgateEnv,
    bottleneck=MultiAgentBottleneckEnv,
    parkinglot=MultiAgentParkingLotEnv,
    pgma=MultiAgentMetaDrive
)

class MetaDriveEnv(MultiAgentEnv):
    def __init__(
            self,
                 discrete_steering_dim,
                 discrete_throttle_dim,
                 discrete_action=True,
                 env_name="roundabout",
                 episode_limit=200
                 ):
        config = dict(discrete_steering_dim=discrete_steering_dim,
                      discrete_throttle_dim=discrete_throttle_dim,
                      discrete_action=discrete_action)

        self.env = envs_classes[env_name](config)

        self.n_agents = len(self.env.observations)
        self.n_actions = discrete_steering_dim * discrete_throttle_dim
        self.episode_limit = episode_limit

        self._episode_steps = 0
        self._last_obs = None
        self._cur_obs = None

    def step(self, actions):
        #assume actions is a list
        actions = actions.tolist()
        actions_multidrive = {'agent%d'%i : actions[i] for i in range(len(actions))}
        o, r, d, i = self.env.step(actions_multidrive)
        self._last_obs = self._cur_obs

        self._episode_steps += 1

        env_reward = 0
        for agent, agent_reward in r.items():
            env_reward += agent_reward

        env_done = False
        for agent, agent_done in d.items():
            env_done = env_done or agent_done
        env_done = env_done or self._episode_steps >= self.episode_limit

        return env_reward, env_done, {}

    def get_obs_agent(self, agent_id):
        if 'agent%d'%agent_id in self.env.vehicles:
            return self.env.observations['agent%d'%agent_id].observe(self.env.vehicles['agent%d'%agent_id])
        return self._last_obs[agent_id]

    def get_obs(self):
        self._cur_obs = [self.get_obs_agent(agent_id) for agent_id in range(self.n_agents)]
        return self._cur_obs

    def get_obs_size(self):
        return self.env.observations['agent0'].observation_space.shape[0]

    def get_state(self):
        obs_concat = np.concatenate(self.get_obs(), axis=0).astype(
            np.float32
        )
        return obs_concat

    def get_state_size(self):
        return self.get_obs_size() * len(self.env.observations)

    def get_avail_actions(self):
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_agent = self.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_agent)
        return avail_actions

    def get_avail_agent_actions(self, agent_id):
        """ Returns the available actions for agent_id """
        return [1 for _ in range(self.n_actions)]

    def get_total_actions(self):
        """ Returns the total number of actions an agent could ever take """
        # TODO: This is only suitable for a discrete 1 dimensional action space for each agent
        return self.n_actions

    def reset(self):
        """ Returns initial observations and states"""
        self.env.reset()
        self._episode_steps = 0

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

    def seed(self):
        return None

    def save_replay(self):
        raise NotImplementedError


