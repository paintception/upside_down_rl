import os
import argparse
import pickle
import keras
import numpy as np

STORING_PATH = '../offline_rl_results/'
MODELS_PATH = './trained_models/'

def save_results(environment, approximator, seed, rewards):
    storing_path = os.path.join(STORING_PATH, environment, approximator, str(seed))
    if not os.path.exists(storing_path):
        os.makedirs(storing_path)
    
    np.save(storing_path + '/' + 'upside_down_rewards.npy', rewards)

def save_trained_model(environment, seed, algorithm, model):
    storing_path = os.path.join(MODELS_PATH, environment, str(seed), algorithm)
    if not os.path.exists(storing_path):
        os.makedirs(storing_path)
    
    model.save_weights(storing_path + '/' + 'trained_model.h5')

def save_offline_results(environment, algorithm, seed, returns):
    storing_path = os.path.join(STORING_PATH, algorithm, str(seed))
    if not os.path.exists(storing_path):
        os.makedirs(storing_path)

    np.save(storing_path + '/rewards.npy', returns)
