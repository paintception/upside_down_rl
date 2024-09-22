from dataclasses import dataclass, field
from typing import Dict, Any, Union
from abc import ABC
import importlib
from pickle import dump, load


from sklearn.exceptions import NotFittedError
from sklearn.base import BaseEstimator
import numpy as np

import keras
from keras.layers import Dense, Multiply
from keras.models import Model
from keras.optimizers import Adam


class ABCPolicy(ABC):
    """An abstract base class for defining agent policies.

    Methods
    -------
    __call__(state, command, test)
        Selects an action based on the given state and command.

        Parameters
        ----------
        state : np.array
            The current state of the environment.
        command : np.array
            The command or goal provided to the policy.
        test : bool
            Whether the policy is being used in a testing scenario.

        Returns
        -------
        int or np.array
            The selected action.

    train(states, commands, actions)
        Trains the policy using the provided experiences.

        Parameters
        ----------
        states : np.array
            A batch of states.
        commands : np.array
            A batch of corresponding commands.
        actions : np.array
            A batch of corresponding actions taken.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing training metrics or other  information.

    save(path)
        Saves the policy to the specified path.

        Parameters
        ----------
        path : str
            The path to save the policy to.

    load(path)
        Loads the policy from the specified path.

        Parameters
        ----------
        path : str
            The path to load the policy from.
    """

    def __call__(
        self,
        state: np.array,
        command: np.array,
        test: bool,
    ) -> Union[int, np.array]: ...

    def train(
        self,
        states: np.array,
        commands: np.array,
        actions: np.array,
    ) -> Dict[str, Any]: ...

    def save(self, path: str): ...
    def load(path: str): ...


@dataclass
class SklearnPolicy(ABCPolicy):
    """A policy using a scikit-learn estimator for action selection.

    Parameters
    ----------
    epsilon : float
        Exploration rate for epsilon-greedy action selection.
    action_size : int
        The number of possible actions in the environment.
    estimator_name : str
        The fully qualified name of the scikit-learn estimator class
        (e.g., 'ensemble.RandomForestClassifier').
    estimator_kwargs : Dict[str, Any], optional
        Keyword arguments to pass to the estimator constructor (default: {}).

    Attributes
    ----------
    estimator : BaseEstimator
        The initialized scikit-learn estimator.

    Methods
    -------
    __call__(state, command, test)
        Selects an action based on the given state and command,
        using the estimator or epsilon-greedy exploration.

    train(states, commands, actions)
        Trains the estimator using the provided experiences.

    save(path)
        Saves the policy (including the estimator) to a pickle file.

    load(path)
        Loads the policy (including the estimator) from a pickle file.
    """

    epsilon: float
    action_size: int
    estimator_name: str
    estimator_kwargs: Dict[str, Any] = field(default_factory=dict)
    estimator: BaseEstimator = field(init=False)

    def __post_init__(self):
        module, clf_name = self.estimator_name.split(".")
        module = importlib.import_module("sklearn." + module)
        self.estimator = getattr(module, clf_name)(
            **self.estimator_kwargs,
        )

    def __call__(
        self,
        state: np.array,
        command: np.array,
        test: bool,
    ):
        input_state = np.concatenate((state, command), axis=1)
        actions = None
        try:
            actions = self.estimator.predict(input_state)
        except NotFittedError:
            ...

        if not test and (actions is None or np.random.rand() <= self.epsilon):
            return np.random.choice(self.action_size)

        return np.argmax(actions) if len(actions.shape) == 2 else actions[0]

    def train(
        self,
        states: np.array,
        commands: np.array,
        actions: np.array,
    ):
        input_classifier = np.concatenate((states, commands), axis=1)

        try:
            self.estimator.fit(input_classifier, actions)
        except ValueError:
            self.estimator.fit(input_classifier, np.argmax(actions, axis=1))

    def save(self, path: str):
        with open(path + ".pkl", "wb") as f:
            dump(self, f)

    def load(path: str):
        with open(path + ".pkl", "rb") as f:
            policy = load(f)
        return policy


@dataclass
class NeuralPolicy(ABCPolicy):
    """A policy implemented using a neural network for action selection.

    Parameters
    ----------
    state_size : int
        The size of the state space in the environment.
    command_size : int, optional
        The size of the command or goal vector (default: 2).
    action_size : int, optional
        The number of possible actions in the environment (default: 2).

    Attributes
    ----------
    estimator : keras.Model
        The compiled Keras neural network model.

    Methods
    -------
    __call__(state, command, test)
        Selects an action based on the given state and command using the
        neural network. During testing, the action with the highest
        probability is chosen; otherwise, an action is sampled according
        to the predicted probabilities.

    train(states, commands, actions)
        Trains the neural network using the provided experiences.
    """

    state_size: int
    command_size: int = 2
    action_size: int = 2
    estimator: BaseEstimator = field(init=False)

    def __post_init__(self):

        observation_input = keras.Input(shape=(self.state_size,))
        linear_layer = Dense(64, activation="sigmoid")(observation_input)

        command_input = keras.Input(shape=(self.command_size,))
        sigmoidal_layer = Dense(64, activation="sigmoid")(command_input)

        multiplied_layer = Multiply()([linear_layer, sigmoidal_layer])

        layer_1 = Dense(64, activation="relu")(multiplied_layer)
        layer_2 = Dense(64, activation="relu")(layer_1)
        layer_3 = Dense(64, activation="relu")(layer_2)
        layer_4 = Dense(64, activation="relu")(layer_3)
        final_layer = Dense(self.action_size, activation="softmax")(layer_4)

        model = Model(
            inputs=[observation_input, command_input],
            outputs=final_layer,
        )
        model.compile(
            loss="categorical_crossentropy",
            optimizer=Adam(lr=0.001),
        )
        self.estimator = model

    def __call__(
        self,
        state: np.array,
        command: np.array,
        test: bool,
    ):
        action_probs = self.estimator.predict([state, command])
        if test:
            return np.argmax(action_probs)
        return np.random.choice(
            np.arange(0, self.action_size),
            p=action_probs[0],
        )

    def train(
        self,
        states: np.array,
        commands: np.array,
        actions: np.array,
    ):
        self.estimator([states, commands], actions, verbose=0)
