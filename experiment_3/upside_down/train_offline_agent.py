import os
import math
import time
import gym
import random
import utils
import keras
import numpy as np

from collections import deque
from matplotlib import pyplot as plt

class ReplayBuffer():
    """
        Thank you: https://github.com/BY571/
    """

    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = np.load('./buffers/CartPole-v0/1/memory_buffer.npy')
      
    def sort(self):
        #sort buffer
        self.buffer = sorted(self.buffer, key = lambda i: i["summed_rewards"],reverse=True)
        # keep the max buffer size
        self.buffer = self.buffer[:self.max_size]

    def get_random_samples(self, batch_size):
        self.sort()
        
        idxs = np.random.randint(0, len(self.buffer), batch_size)
        batch = [self.buffer[idx] for idx in idxs]
       
        return batch
    
    def get_n_best(self, n):
        self.sort()
        return self.buffer[:n]
    
    def __len__(self):
        return len(self.buffer)


class UpsideDownAgent():
    def __init__(self, environment):
        self.environment = gym.make(environment)
        self.state_size = self.environment.observation_space.shape[0]
        self.action_size = self.environment.action_space.n
        self.memory = ReplayBuffer(700)
        self.last_few = 75 
        self.batch_size = 32
        self.command_size = 2 # desired return + desired horizon
        self.desired_return = 1 
        self.desired_horizon = 1
        self.horizon_scale = 0.02
        self.return_scale = 0.02
        self.testing_state = 0

        self.behaviour_function = utils.get_functional_behaviour_function(self.state_size, self.command_size, self.action_size, False)
        
        self.testing_rewards = []

   
    def get_action(self, observation, command):
        """
            We will sample from the action distribution modeled by the Behavior Function 
        """
        
        action_probs = self.behaviour_function.predict([observation, command])
        action = np.random.choice(np.arange(0, self.action_size), p=action_probs[0])

        return action
 
    def get_greedy_action(self, observation, command):

        action_probs = self.behaviour_function.predict([observation, command])
        action = np.argmax(action_probs)

        return action

    def train_behaviour_function(self):

        random_episodes = self.memory.get_random_samples(self.batch_size)
       
        training_observations = np.zeros((self.batch_size, self.state_size))
        training_commands = np.zeros((self.batch_size, 2))

        y = []
        
        for idx, episode in enumerate(random_episodes):
            T = len(episode['states'])
            t1 = np.random.randint(0, T-1)
            t2 = np.random.randint(t1+1, T)
            
            state = episode['states'][t1]
            desired_return = sum(episode["rewards"][t1:t2]) 
            desired_horizon = t2 -t1
 
            target = episode['actions'][t1]
            
            training_observations[idx] = state[0]
            training_commands[idx] = np.asarray([desired_return*self.return_scale, desired_horizon*self.horizon_scale])
            y.append(target) 
         
        _y = keras.utils.to_categorical(y) 

        self.behaviour_function.fit([training_observations, training_commands], _y, verbose=0)
        

    def sample_exploratory_commands(self):
        best_episodes = self.memory.get_n_best(self.last_few)
        exploratory_desired_horizon = np.mean([len(i["states"]) for i in best_episodes])
        
        returns = [i["summed_rewards"] for i in best_episodes]
        exploratory_desired_returns = np.random.uniform(np.mean(returns), np.mean(returns)+np.std(returns))

        return [exploratory_desired_returns, exploratory_desired_horizon]

    def generate_episode(self, environment, e, desired_return, desired_horizon, testing):
       
        env = gym.make(environment)
        tot_rewards = []
        done = False
        
        score = 0
        state = env.reset()
         
        scores = []
        states = []
        actions = []
        rewards = []

        while not done:            
            state = np.reshape(state, [1, self.state_size])
            states.append(state)

            observation = state
            
            command = np.asarray([desired_return * self.return_scale, desired_horizon * self.horizon_scale])
            command = np.reshape(command, [1, len(command)])

            if not testing:
                action = self.get_action(observation, command)
                actions.append(action)
            else:
                action = self.get_greedy_action(observation, command)

            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, self.state_size])
            
            rewards.append(reward)
            score += reward

            state = next_state
           
            desired_return -= reward  # Line 8 Algorithm 2
            desired_horizon -= 1 # Line 9 Algorithm 2
            desired_horizon = np.maximum(desired_horizon, 1)
             
        self.testing_rewards.append(score)

        if testing:
            print('Querying the model ...')
            print('Testing score: {}'.format(score))

        return score

def run_experiment():

    environment = 'CartPole-v0'
    seed = 1

    episodes = 500 
    returns = []

    agent = UpsideDownAgent(environment)

    for e in range(episodes):
        for i in range(100):
            agent.train_behaviour_function()

        for i in range(15):            
            tmp_r = []
            r = agent.generate_episode(environment, e, 200, 200, False)
            tmp_r.append(r)

        print(np.mean(tmp_r))
        returns.append(np.mean(tmp_r))
    
    agent.generate_episode(environment, 1, 200, 200, True)

    utils.save_offline_results(environment, approximator, seed, returns)


if __name__ == "__main__":
    run_experiment()
