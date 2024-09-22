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
    def __init__(self, environment, approximator):
        self.environment = gym.make(environment)
        self.approximator = approximator
        self.state_size = (84, 84, 4)
        self.action_size = 3
        self.warm_up_episodes = 1 #50 
        self.render = False
        self.memory = ReplayBuffer(700)
        self.last_few = 50
        self.batch_size = 256
        self.command_size = 2 # desired return + desired horizon
        self.desired_return = 1 
        self.desired_horizon = 1
        self.horizon_scale = 0.02
        self.return_scale = 0.02
     
        self.behaviour_function = utils.get_atari_behaviour_function(self.action_size)
       
        self.testing_rewards = []
        self.warm_up_buffer()

    def warm_up_buffer(self):
        print('Warming up')

        for i in range(self.warm_up_episodes):
            
            states = []
            rewards = []
            actions = []
            
            dead = False
            done = False
            desired_return = 1
            desired_horizon = 1

            step, score, start_life = 0, 0, 5
            observe = self.environment.reset()

            for _ in range(random.randint(1, 30)):
                observe, _, _, _ = self.environment.step(1)

            state = utils.pre_processing(observe)
            history = np.stack((state, state, state, state), axis=2)
            history = np.reshape([history], (1, 84, 84, 4))


            while not done:
                
                states.append(history)
                command = np.asarray([desired_return * self.return_scale, desired_horizon * self.horizon_scale])
                command = np.reshape(command, [1, len(command)])

                action = self.get_action(history, command)
                actions.append(action)
               
                if action == 0:
                    real_action = 1
                elif action == 1:
                    real_action = 2
                else:
                    real_action = 3 

                next_state, reward, done, info = self.environment.step(real_action)
                next_state = utils.pre_processing(observe)
                next_state = np.reshape([next_state], (1, 84, 84, 1))
                next_history = np.append(next_state, history[:, :, :, :3], axis = 3)
 
                rewards.append(reward)

                state = next_state
        
                if start_life > info['ale.lives']:
                    dead = True 
                    start_lide = info['ale.lives']

                if dead:
                    dead = False
                else:
                    history = next_history

                desired_return -= reward  # Line 8 Algorithm 2
                desired_horizon -= 1 # Line 9 Algorithm 2
                desired_horizon = np.maximum(desired_horizon, 1)
                          
            self.memory.add_sample(states, actions, rewards)


    def get_action(self, observation, command):
        """
            We will sample from the action distribution modeled by the Behavior Function 
        """
        
        observation = np.float32(observation / 255.0)

        action_probs = self.behaviour_function.predict([observation, command])
        action = np.random.choice(np.arange(0, self.action_size), p=action_probs[0])

        return action
 
    def get_greedy_action(self, observation, command):

        action_probs = self.behaviour_function.predict([observation, command])
        action = np.argmax(action_probs)

        return action
 
    def train_behaviour_function(self):

        random_episodes = self.memory.get_random_samples(self.batch_size)
       
        training_observations = np.zeros((self.batch_size, self.state_size[0], self.state_size[1], self.state_size[2]))
        training_commands = np.zeros((self.batch_size, 2))

        y = []
        
        for idx, episode in enumerate(random_episodes):
            T = len(episode['states'])
            t1 = np.random.randint(0, T-1)
            t2 = np.random.randint(t1+1, T)
            
            state = np.float32(episode['states'][t1] / 255.)
            desired_return = sum(episode["rewards"][t1:t2]) 
            desired_horizon = t2 -t1
 
            target = episode['actions'][t1]
    
            training_observations[idx] = state[0]
            training_commands[idx] = np.asarray([desired_return*self.return_scale, desired_horizon*self.horizon_scale])
            y.append(target) 
         
        _y = keras.utils.to_categorical(y, num_classes=self.action_size) 
      
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
        dead = False
  
        scores = []
        states = []
        actions = []
        rewards = []

        step, score, start_life = 0, 0, 5 

        observe = env.reset()
        for _ in range(random.randint(1, 30)):
            observe, _, _, _ = env.step(1)

        state = utils.pre_processing(observe)
        history = np.stack((state, state, state, state), axis=2)
        history = np.reshape([history], (1, 84, 84, 4))

        while not done:            
            states.append(history)
            
            command = np.asarray([desired_return * self.return_scale, desired_horizon * self.horizon_scale])
            command = np.reshape(command, [1, len(command)])

            if not testing:
                action = self.get_action(history, command)
                actions.append(action)
            else:
                action = self.get_greedy_action(history, command)

            if action == 0:
                real_action = 1
            elif action == 1:
                real_action = 2
            else:
                real_action = 3

            next_state, reward, done, info = env.step(real_action)
            next_state = utils.pre_processing(observe)
            next_state = np.reshape([next_state], (1, 84, 84, 1))
            next_history = np.append(next_state, history[:, :, :, :3], axis = 3)

            clipped_reward = np.clip(reward, -1, 1)
            rewards.append(clipped_reward)
            
            score += reward

            if start_life > info['ale.lives']:
                dead = True 
                start_life = info['ale.lives']

            if dead:
                dead = False
            else:
                history = next_history
           
            desired_return -= reward  # Line 8 Algorithm 2
            desired_horizon -= 1 # Line 9 Algorithm 2
            desired_horizon = np.maximum(desired_horizon, 1)
            
        self.memory.add_sample(states, actions, rewards)
        
        self.testing_rewards.append(score)

        if testing:
            print('Querying the model ...')
            print('Testing score: {}'.format(score))

        return score

def run_experiment():

    import argparse

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--approximator', type=str, default='neural_network')
    parser.add_argument('--environment', type=str, default='PongDeterministic-v4')
    parser.add_argument('--seed', type=int, default=1)

    args = parser.parse_args()

    approximator = args.approximator
    environment = args.environment
    seed = args.seed

    episodes =  1500  
    returns = []

    agent = UpsideDownAgent(environment, approximator)

    for e in range(episodes):

        print("Episode {}".format(e))

        for i in range(100):
            agent.train_behaviour_function()

        print("Finished training B!")
        
        for i in range(15):            
            tmp_r = []
            exploratory_commands = agent.sample_exploratory_commands() # Line 5 Algorithm 1
            desired_return = exploratory_commands[0]
            desired_horizon = exploratory_commands[1]
            r = agent.generate_episode(environment, e, desired_return, desired_horizon, False)
            tmp_r.append(r)

        print(np.mean(tmp_r))
        returns.append(np.mean(tmp_r))

        exploratory_commands = agent.sample_exploratory_commands()
        
    #agent.generate_episode(environment, 1, 200, 200, True)

        utils.save_results(environment, approximator, seed, returns)
    
    if approximator == 'neural_network':
        utils.save_trained_model(environment, seed, agent.behaviour_function)

    plt.plot(returns)
    plt.show()

if __name__ == "__main__":
    run_experiment()
