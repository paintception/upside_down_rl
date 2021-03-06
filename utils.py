import os
import argparse
import pickle
import keras
import numpy as np

from keras.layers import Dense, Multiply
from keras.models import Sequential, Model
from keras.optimizers import Adam, RMSprop, SGD


def get_functional_behaviour_function(state_size, command_size, action_size):
    observation_input = keras.Input(shape=(state_size,))
    linear_layer = Dense(64, activation='sigmoid')(observation_input)

    command_input = keras.Input(shape=(command_size,))
    sigmoidal_layer = Dense(64, activation='sigmoid')(command_input)

    multiplied_layer = Multiply()([linear_layer, sigmoidal_layer])

    layer_1 = Dense(64, activation='relu')(multiplied_layer)
    layer_2 = Dense(64, activation='relu')(layer_1)
    layer_3 = Dense(64, activation='relu')(layer_2)
    layer_4 = Dense(64, activation='relu')(layer_3)
    final_layer = Dense(action_size, activation='softmax')(layer_4)

    model = Model(inputs=[observation_input, command_input], outputs=final_layer)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001))

    return model
