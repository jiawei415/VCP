import numpy as np
from gym.core import Wrapper
from gym.spaces import Dict, Box


class ReacherGoalWrapper(Wrapper):
    def __init__(self, env, threshold=0.02):
        Wrapper.__init__(self, env)
        self.env = env
        self.action_space = env.action_space
        # observation_box = env.observation_space
        observation_box = Box(np.array([-np.inf] * 8), np.array([np.inf] * 8))
        desired_goal_box = Box(np.array([-np.inf, -np.inf]), np.array([np.inf, np.inf]))
        achieved_goal_box = Box(np.array([-np.inf, -np.inf]), np.array([np.inf, np.inf]))
        self.observation_space = Dict([
                                ('observation', observation_box),
                                ('desired_goal', desired_goal_box),
                                ('achieved_goal', achieved_goal_box),
                                ])

        self.threshold = threshold

    def reset(self):
        obs = self.env.reset()
        obs_dict = self.obs_to_dict(obs)
        return obs_dict

    def compute_rewards(self, achieved_goal, desired_goal, info=None):
        dist = np.linalg.norm(achieved_goal - desired_goal, axis=1)
        reward = np.zeros((achieved_goal.shape[0], 1))
        reward[np.where(dist > self.threshold)] = -1
        return reward.reshape(-1)

    def compute_reward(self, achieved_goal, desired_goal, info=None):
        if len(achieved_goal.shape) == 2 and achieved_goal.shape[0] > 1:
            return self.compute_rewards(achieved_goal, desired_goal)

        dist = np.linalg.norm(achieved_goal - desired_goal)
        reward = -1 if dist > self.threshold else 0
        return reward

    def obs_to_dict(self, obs):
        obs_ag = self.env.get_body_com("fingertip")[:-1]
        obs_g = self.env.get_body_com("target")[:-1]
        obs_ = np.concatenate([obs[:4], obs[6:8], obs_ag])
        obs_dict = {
            'observation': obs_,
            'desired_goal': obs_g,
            'achieved_goal': obs_ag
        }
        return obs_dict

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs_dict = self.obs_to_dict(obs)
        reward = self.compute_reward(obs_dict['desired_goal'], obs_dict['achieved_goal'])
        if reward == 0:
            info['is_success'] = True
        else:
            info['is_success'] = False
        return obs_dict, reward, done, info

    def render(self, mode='human'):
        return self.env.render()


# for point env 
class PointGoalWrapper(Wrapper):
    observation_keys = ['observation', 'desired_goal', 'achieved_goal']
    def __init__(self, env):
        Wrapper.__init__(self, env=env)
        self.env = env
        self.action_space = env.action_space
        # observation
        for key in list(env.observation_space.spaces.keys()):
            if key not in self.observation_keys:
                del env.observation_space.spaces[key]

        self.observation_space = env.observation_space

        if hasattr(self.env, 'reward_type'):
            self.env.reward_type = 'sparse'
        if hasattr(self.env, 'env') and hasattr(self.env.env, 'reward_type'):
            self.env.env.reward_type = 'sparse'
    
    def reset(self):
        return self.env.reset()
    
    def step(self, action):
        obs_dict, reward, done, info = self.env.step(action)
        obs = {
            'observation':obs_dict['observation'],
            'desired_goal':obs_dict['desired_goal'],
            'achieved_goal':obs_dict['achieved_goal']
        }
        return obs, reward, done, info
    
    def render(self, mode='human'):
        return self.env.render()
    
    def compute_reward(self, achieved_goal, desired_goal, info=None):
        obs = {
            'state_achieved_goal': achieved_goal,
            'state_desired_goal':desired_goal
        }
        action = np.array([])
        return self.env.compute_rewards(action, obs)

    def sample_goal(self):
        goal_dict = self.env.sample_goal()
        return goal_dict['desired_goal']

# for sawyer env
class SawyerGoalWrapper(Wrapper):
    reward_type_dict = {
        'dense':'hand_distance',
        'sparse': 'hand_success',
    }
    observation_keys = ['observation', 'desired_goal', 'achieved_goal']
        
    def __init__(self, env, reward_type='sparse'):
        Wrapper.__init__(self, env=env)
        self.env = env
        self.action_space = env.action_space
        # observation
        for key in list(env.observation_space.spaces.keys()):
            if key not in self.observation_keys:
                del env.observation_space.spaces[key]

        self.observation_space = env.observation_space
        self.reward_type = reward_type

        if hasattr(self.env, 'reward_type'):
            self.env.reward_type = self.reward_type_dict[self.reward_type]
        if hasattr(self.env, 'env') and hasattr(self.env.env, 'reward_type'):
            self.env.env.reward_type = self.reward_type_dict[self.reward_type]
        
    
    def reset(self):
        return self.env.reset()
    
    def step(self, action):
        obs_dict, reward, done, info = self.env.step(action)
        obs = {
            'observation':obs_dict['observation'],
            'desired_goal':obs_dict['desired_goal'],
            'achieved_goal':obs_dict['achieved_goal']
        }
        if 'hand_success' in info.keys():
            info['is_success'] = info['hand_success']
        if 'success' in info.keys():
            info['is_success'] = info['success']
        return obs, reward, done, info
    
    def render(self, mode='human'):
        return self.env.render()
    
    def compute_reward(self, achieved_goal, desired_goal, info):
        obs = {
            'state_achieved_goal': achieved_goal,
            'state_desired_goal':desired_goal
        }
        action = np.array([])
        return self.env.compute_rewards(action, obs)

    def sample_goal(self):
        goal_dict = self.env.sample_goal()
        return goal_dict['desired_goal']
