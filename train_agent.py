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
from sklearn.preprocessing import OneHotEncoder


class ReplayBuffer():
    """
        Thank you: https://github.com/BY571/
    """

    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = []
     
    def add_sample(self, states, actions, rewards):
        episode = {"states": states, "actions":actions, "rewards": rewards, "summed_rewards":sum(rewards)}
        self.buffer.append(episode)
    
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
    def __init__(self, environment, state_size, action_size):
        self.render = False
        self.memory = ReplayBuffer(700)
        self.train_start = 50
        self.last_few = 50
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = 256
        self.command_size = 2 # desired return + desired horizon
        self.desired_return = 1 
        self.desired_horizon = 1
        self.horizon_scale = 0.02
        self.return_scale = 0.02

        self.behaviour_function = utils.get_functional_behaviour_function(self.state_size, self.command_size, self.action_size)
        self.testing_rewards = []
    

    def get_action(self, observation, command):
        """
            We will sample from the action distribution modeled by the Behavior Function 
        """
        
        command[0][0] = command[0][0]*self.return_scale
        command[0][1] = command[0][1]*self.horizon_scale

        action_probs = self.behaviour_function.predict([observation, command])
        action = np.random.choice(np.arange(0, self.action_size), p=action_probs[0])
        
        return action
 
    def train_behaviour_function(self):
        if len(self.memory) < self.train_start:
            return

        for i in range(10):
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
                training_commands[idx] = np.asarray([desired_return, desired_horizon])
                y.append(target) 
         
            y = keras.utils.to_categorical(y) 

            self.behaviour_function.fit([training_observations, training_commands], y, verbose=0)

    def sample_exploratory_commands(self):
        best_episodes = self.memory.get_n_best(self.last_few)
        exploratory_desired_horizon = np.mean([len(i["states"]) for i in best_episodes])
        
        returns = [i["summed_rewards"] for i in best_episodes]
        exploratory_desired_returns = np.random.uniform(np.mean(returns), np.mean(returns)+np.std(returns))

        #print(exploratory_desired_horizon)
        #print(exploratory_desired_returns)

        return [exploratory_desired_returns, exploratory_desired_horizon]

    def generate_new_behaviour(self, exploratory_commands):
        
        env = gym.make('CartPole-v0')
        state = env.reset()
        states = []
        actions = []
        rewards = []
        done = False

        while not done:
            state = np.reshape(state, [1, len(state)])
            #print('Generating random trajectories')
            states.append(state)

            observation = state
            command = np.asarray([exploratory_commands[0]*self.return_scale, exploratory_commands[1]*self.horizon_scale])
            command = np.reshape(command, [1, len(command)])

            action_probs = self.behaviour_function.predict([observation, command])
            action = np.random.choice(np.arange(0, self.action_size), p=action_probs[0])
            actions.append(action)

            next_state, reward, done, info = env.step(action)
            
            actions.append(action)
            rewards.append(reward)
        
            state = next_state
            
            exploratory_commands[0] -= reward 
            exploratory_commands[1] -= 1 
            exploratory_commands[1] = np.maximum(self.desired_horizon, 1)
         
        self.memory.add_sample(states, actions, rewards)

def run_experiment():

    episodes = 200
    
    env = gym.make('CartPole-v0')
    
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = UpsideDownAgent(env, state_size, action_size)

    tot_rewards = []

    for e in range(episodes):
        done = False
        score = 0
        state = env.reset()
         
        scores = []
        states = []
        actions = []
        rewards = []
        
        desired_return = 1
        desired_horizon = 1 
 
        while not done:           
            agent.train_behaviour_function()
        
            state = np.reshape(state, [1, state_size])
            states.append(state)

            observation = state
            command = np.asarray([desired_return, desired_horizon])
            command = np.reshape(command, [1, len(command)])

            action = agent.get_action(observation, command)
            actions.append(action)

            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            
            rewards.append(reward)
            score += reward

            state = next_state
            
            desired_return -= reward 
            desired_horizon -= 1 
            desired_horizon = np.maximum(desired_horizon, 1)
        
            #print(desired_return)

        agent.memory.add_sample(states, actions, rewards)
          
        tot_rewards.append(score)
        print(score)
        
        exploratory_commands = agent.sample_exploratory_commands()
        agent.generate_new_behaviour(exploratory_commands)
       
    plt.plot(tot_rewards)
    plt.show()

if __name__ == "__main__":
    run_experiment()
