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

# config = dict(discrete_action=True, use_render=True)
class MetaDriveEnv(MultiAgentEnv):
    def __init__(self, config, env_name="roundabout"):
        self.env = envs_classes[env_name](config)
        self.n_actions = config['discrete_steering_dim'] * config['discrete_throttle_dim']

    def step(self, actions):
        #assume actions is a list
        actions_multidrive = {'agent%d'%i : actions[i] for i in range(len(actions))}
        o, r, d, info = self.env.step(actions_multidrive)

        env_reward = 0
        for agent, agent_reward in r.items():
            env_reward += agent_reward

        env_done = False
        for agent, agent_done in d.items():
            env_done = env_done or agent_done

        return env_reward, env_done, info

    def get_obs_agent(self, agent_id):
        return self.env.observations['agent%d'%agent_id].observe(self.env.vehicles['agent%d'%agent_id])

    def get_obs(self):
        return [self.get_obs_agent(agent_id) for agent_id in range(len(self.env.vehicles))]

    def get_obs_size(self):
        return len(self.env.observations['agent0'].observe(self.env.vehicles['agent0']))

    def get_state(self):
        obs_concat = np.concatenate(self.get_obs(), axis=0).astype(
            np.float32
        )
        return obs_concat

    def get_state_size(self):
        return self.get_obs_size() * len(self.env.vehicles)

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

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

    def seed(self):
        return None

    def save_replay(self):
        raise NotImplementedError


