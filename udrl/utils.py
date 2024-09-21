import os
import keras
import numpy as np

from keras.layers import Dense, Multiply, Input, Conv2D, Flatten
from keras.models import Model
from keras.optimizers import Adam, RMSprop

from skimage.transform import resize
from skimage.color import rgb2gray
from pathlib import Path

STORING_PATH = "./results/"
MODELS_PATH = "./trained_models/"


def generate_exp_name(env: str, approx: str, seed: int):
    name = f"{env}_{approx}_{seed}"
    return name


def save_results(exp_name, rewards, base_path: str = "results"):
    path = Path(base_path) / exp_name
    path.mkdir(parents=True, exist_ok=True)
    np.save(str(path / "upside_down_rewards.npy"), rewards)


def save_trained_model(environment, seed, model):
    storing_path = os.path.join(MODELS_PATH, environment, str(seed))
    if not os.path.exists(storing_path):
        os.makedirs(storing_path)

    model.save_weights(storing_path + "/" + "trained_model.h5")


def get_functional_behaviour_function(state_size, command_size, action_size):
    observation_input = keras.Input(shape=(state_size,))
    linear_layer = Dense(64, activation="sigmoid")(observation_input)

    command_input = keras.Input(shape=(command_size,))
    sigmoidal_layer = Dense(64, activation="sigmoid")(command_input)

    multiplied_layer = Multiply()([linear_layer, sigmoidal_layer])

    layer_1 = Dense(64, activation="relu")(multiplied_layer)
    layer_2 = Dense(64, activation="relu")(layer_1)
    layer_3 = Dense(64, activation="relu")(layer_2)
    layer_4 = Dense(64, activation="relu")(layer_3)
    final_layer = Dense(action_size, activation="softmax")(layer_4)

    model = Model(
        inputs=[observation_input, command_input],
        outputs=final_layer,
    )
    model.compile(loss="categorical_crossentropy", optimizer=Adam(lr=0.001))

    return model


def get_atari_behaviour_function(action_size):

    print("Getting the model")

    input_state = Input(shape=(84, 84, 4))

    first_conv = Conv2D(
        32,
        (8, 8),
        strides=(4, 4),
        activation="relu",
    )(input_state)
    second_conv = Conv2D(
        64,
        (4, 4),
        strides=(2, 2),
        activation="relu",
    )(first_conv)
    third_conv = Conv2D(
        64,
        (3, 3),
        strides=(1, 1),
        activation="relu",
    )(second_conv)

    flattened = Flatten()(third_conv)
    dense_layer = Dense(512, activation="relu")(flattened)

    command_input = keras.Input(shape=(2,))
    sigmoidal_layer = Dense(512, activation="sigmoid")(command_input)

    multiplied_layer = Multiply()([dense_layer, sigmoidal_layer])
    final_layer = Dense(256, activation="relu")(multiplied_layer)

    action_layer = Dense(action_size, activation="softmax")(final_layer)

    model = Model(inputs=[input_state, command_input], outputs=action_layer)
    model.compile(
        loss="categorical_crossentropy",
        optimizer=RMSprop(lr=0.001, rho=0.95, epsilon=0.01),
    )

    print(model.summary())

    return model


def get_catch_behaviour_function(action_size):

    print("Getting the Catch-model")

    input_state = Input(shape=(84, 84, 4))

    first_conv = Conv2D(
        32,
        (8, 8),
        strides=(4, 4),
        activation="relu",
    )(input_state)
    second_conv = Conv2D(
        64,
        (4, 4),
        strides=(2, 2),
        activation="relu",
    )(first_conv)
    third_conv = Conv2D(
        64,
        (3, 3),
        strides=(1, 1),
        activation="relu",
    )(second_conv)

    flattened = Flatten()(third_conv)
    dense_layer = Dense(512, activation="relu")(flattened)

    command_input = keras.Input(shape=(2,))
    sigmoidal_layer = Dense(512, activation="sigmoid")(command_input)

    multiplied_layer = Multiply()([dense_layer, sigmoidal_layer])
    final_layer = Dense(256, activation="relu")(multiplied_layer)

    action_layer = Dense(action_size, activation="softmax")(final_layer)

    model = Model(inputs=[input_state, command_input], outputs=action_layer)
    model.compile(
        loss="categorical_crossentropy",
        optimizer=RMSprop(lr=0.001, rho=0.95, epsilon=0.01),
    )

    print(model.summary())

    return model


def pre_processing(state):
    return np.uint8(resize(rgb2gray(state), (84, 84), mode="constant") * 255)
