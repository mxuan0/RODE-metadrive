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

    # def


