from dataclasses import dataclass, field
from typing import Dict, Any, Union
from abc import ABC
import importlib
from pickle import dump, load


from sklearn.exceptions import NotFittedError
from sklearn.base import BaseEstimator
from sklearn.metrics import classification_report
import numpy as np

import torch
from torch import nn
from torch.distributions import Categorical


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
            It MUST contain the key "metric"

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
        return actions[0]

    def train(
        self,
        states: np.array,
        commands: np.array,
        actions: np.array,
    ):
        input_state = np.concatenate((states, commands), axis=1)
        self.estimator.fit(input_state, actions)
        pred = self.estimator.predict(input_state)
        report = classification_report(actions, pred, output_dict=True)
        report["metric"] = report["accuracy"]
        return report

    def save(self, path: str):
        with open(path + ".pkl", "wb") as f:
            dump(self, f)

    def load(path: str):
        with open(path + ".pkl", "rb") as f:
            policy = load(f)
        return policy


class BehaviorNet(nn.Module):
    """
    A neural network module designed to model agent behavior based on state
    and command inputs.

    Parameters
    ----------
    state_size : int
        Dimensionality of the state input.
    action_size : int
        Dimensionality of the action output.
    command_size : int
        Dimensionality of the command input.
    hidden_size : int, optional
        Number of neurons in the hidden layers. Defaults to 64.

    Returns
    -------
    torch.Tensor
        A probability distribution over actions,
        shaped (batch_size, action_size).
    """

    def __init__(
        self,
        state_size: int,
        action_size: int,
        command_size: int,
        hidden_size: int = 64,
    ):
        super().__init__()
        self.state_entry = nn.Sequential(
            nn.Linear(state_size, hidden_size), nn.Sigmoid()
        )
        self.command_entry = nn.Sequential(
            nn.Linear(command_size, hidden_size), nn.Sigmoid()
        )
        self.model = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
            nn.Softmax(dim=-1),
        )

    def forward(self, state, command):
        state_out = self.state_entry(state)
        command_out = self.command_entry(command)
        out = state_out * command_out
        return self.model(out)


@dataclass
class NeuralPolicy(ABCPolicy):
    """
    A policy that uses a neural network to map states and commands to actions.

    Parameters
    ----------
    state_size : int
        The dimensionality of the state input.
    action_size : int
        The dimensionality of the action output.
    command_size : int, optional
        The dimensionality of the command input. Defaults to 2.
    hidden_size : int, optional
        The number of neurons in the hidden layers of the neural network.
        Defaults to 64.
    device : str, optional
        The device on which to run the neural network.
        Can be "auto" (to automatically select CUDA if available, else CPU),
        or a valid torch device string. Defaults to "auto".
    loss : nn.Module, optional
        The loss function class used for training.
        Defaults to `nn.CrossEntropyLoss`.

    Attributes
    ----------
    estimator : nn.Module
        The neural network used to estimate the action probabilities.
    loss : nn.Module
        The instantiated loss function used for training.
    optim : torch.optim.Adam
        The optimizer used for training.

    Methods
    -------
    __call__(state, command, test)
        Selects an action based on the given state and command

    train(states, commands, actions)
        Trains the estimator using the provided experiences.

    save(path)
        Saves the policy.

    load(path)
        Loads the policy.
    """

    state_size: int
    action_size: int
    command_size: int = 2
    hidden_size: int = 64
    # NOTE GPU maybe be drastically slower for small batch_size
    device: str = "cpu"
    loss: nn.Module = nn.CrossEntropyLoss
    estimator: nn.Module = field(init=False)

    def __post_init__(self):
        self.estimator = BehaviorNet(
            self.state_size,
            self.action_size,
            self.command_size,
            self.hidden_size,
        )
        if self.device == "auto":
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        self.estimator.to(self.device)

        self.loss = self.loss()
        self.optim = torch.optim.Adam(self.estimator.parameters())

    def __call__(
        self,
        state: np.array,
        command: np.array,
        test: bool,
    ):
        state = torch.FloatTensor(state).to(self.device)
        command = torch.FloatTensor(command).to(self.device)
        action_probs = self.estimator(state, command)
        if test:
            return torch.argmax(action_probs).item()
        return Categorical(action_probs).sample().item()

    def train(
        self,
        states: np.array,
        commands: np.array,
        actions: np.array,
    ):
        states = torch.FloatTensor(states).to(self.device)
        commands = torch.FloatTensor(commands).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)

        pred = self.estimator(states, commands)
        self.optim.zero_grad()
        loss = self.loss(pred, actions)
        loss.backward()
        self.optim.step()
        return {"metric": loss.item()}

    def save(self, path: str):
        torch.save(
            {
                "model": self.estimator.state_dict(),
                "optim": self.optim.state_dict(),
                "state_size": self.state_size,
                "action_size": self.action_size,
                "command_size": self.command_size,
                "hidden_size": self.hidden_size,
            },
            path + ".pth",
        )

    def load(path: str):
        saved_dict = torch.load(path + ".pth")
        policy = NeuralPolicy(
            saved_dict["state_size"],
            saved_dict["action_size"],
            saved_dict["command_size"],
            saved_dict["hidden_size"],
        )
        policy.estimator.load_state_dict(saved_dict["model"])
        policy.optim.load_state_dict(saved_dict["optim"])
        return policy
