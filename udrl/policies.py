from dataclasses import dataclass, field
from typing import Dict, Any, Union
from abc import ABC
import importlib
from pickle import dump, load

import numpy as np
from sklearn.exceptions import NotFittedError
from sklearn.base import BaseEstimator

import udrl.utils as ut


class ABCPolicy(ABC):
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
    epsilon: float
    estimator_name: str
    action_size: int = 2
    estimator: BaseEstimator = field(init=False)
    estimator_kwargs: Dict[str, Any] = field(default_factory=dict)

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

        return np.argmax(actions)

    def train(
        self,
        states: np.array,
        commands: np.array,
        actions: np.array,
    ):
        input_classifier = np.concatenate((states, commands), axis=1)
        self.estimator.fit(input_classifier, actions)

    def save(self, path: str):
        with open(path + ".pkl", "wb") as f:
            dump(self, f)

    def load(path: str):
        with open(path + ".pkl", "rb") as f:
            policy = load(f)
        return policy


@dataclass
class NeuralPolicy(ABCPolicy):
    state_size: int
    command_size: int = 2
    action_size: int = 2
    approximator: BaseEstimator = field(init=False)

    def __post_init__(self):
        self.approximator = ut.get_functional_behaviour_function(
            self.state_size, self.command_size, self.action_size
        )

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
