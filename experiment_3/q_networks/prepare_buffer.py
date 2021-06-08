import os
import sys
import gym
import random
import numpy as np
import pickle 

from collections import deque

from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential
from matplotlib import pyplot as plt

WEIGHTS_PATH = './trained_models/CartPole-v0/1/'
BUFFER_PATH = './buffers/CartPole-v0/1/'

class Agent:
    def __init__(self, algorithm, state_size, action_size):
        self.algorithm = algorithm
        self.render = False
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)

        if self.algorithm in ['DQN', 'DDQN', 'DQV']:
            self.model = self.build_model()
            self.model.load_weights(os.path.join(WEIGHTS_PATH, self.algorithm, 'trained_model.h5'))
        else:
            self.model = self.build_actor()
            self.model.load_weights(os.path.join(WEIGHTS_PATH, self.algorithm, 'trained_model.h5'))


    def build_actor(self):
        actor = Sequential()
        actor.add(Dense(24, input_dim=self.state_size, activation='relu', kernel_initializer='he_uniform'))
        actor.add(Dense(self.action_size, activation='softmax', kernel_initializer='he_uniform'))

        return actor

    def build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu',
						kernel_initializer='he_uniform'))
        model.add(Dense(24, activation='relu',
						kernel_initializer='he_uniform'))
        model.add(Dense(self.action_size, activation='linear',
						kernel_initializer='he_uniform'))

        return model

    def get_action(self, state):
        if self.algorithm == 'A2C':
            policy = self.model.predict(state, batch_size=1).flatten()

            return np.random.choice(self.action_size, 1, p=policy)[0]

        else:
            q_value = self.model.predict(state)
            return np.argmax(q_value[0])

    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def save_buffer(self):
        if not os.path.exists(os.path.join(BUFFER_PATH, self.algorithm)):
            os.makedirs(os.path.join(BUFFER_PATH, self.algorithm))

        with open(os.path.join(BUFFER_PATH, self.algorithm, 'memory_buffer.p'), 'wb') as filehandler:
            pickle.dump(self.memory, filehandler)

def fill_buffer(algorithm):
    max_len = 10000 
    results = []
    game = 'CartPole-v0'

    env = gym.make(game)
    
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = Agent(algorithm, state_size, action_size)

    while True:
        done = False
        score = 0
        state = env.reset()
        state = np.reshape(state, [1, state_size])

        while not done:
            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])

            agent.append_sample(state, action, reward, next_state, done)

            score += reward
            state = next_state 

        if len(agent.memory) > max_len:
            agent.save_buffer()
            break

fill_buffer('DQN')
