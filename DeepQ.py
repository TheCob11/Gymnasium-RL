import tensorflow as tf
import matplotlib.pyplot as plt
from collections import deque, defaultdict
import gymnasium as gym
from gymnasium.wrappers.record_episode_statistics import RecordEpisodeStatistics
from gymnasium.wrappers.record_video import RecordVideo
from typing import Any
from matplotlib.figure import Figure, SubFigure
import numpy as np
from gymnasium.core import Env
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
import time


class Blake:
    def __init__(
        self,
        learn_rate: float = 0.001,
        episodes: int = 500,
        epsilon_init: float = 1,
        epsilon_final: float = 0.01,
        model_training_frequency: int = 4,
        model_copy_frequency: int = 100,
        batch_size: int = 128,
        env: Env = gym.make("CartPole-v1", render_mode="rgb_array")
    ) -> None:
        '''Creates a new Agent

        Parameters:
            learn_rate (float): Alpha value that determines how fast/slow Blake learns
            episodes (int): How many episodes to play
            epsilon_init (float): Initial value for epsilon(probablity that Blake will explore new options instead of exploiting his current knowledge)
            epsilon_final (float): Final value for epsilon
            model_training_frequency (int): How often(in steps) the model trains
            model_copy_frequency (int): How often(in steps) the weights from the main model are copied over to the target model
            batch_size (int): Number of past steps to train on per training run 
            env (Env): The environment for Blake to play on

        Returns:
            None
        '''
        self.lr: float = learn_rate
        self.episodes: int = episodes
        self.eps: float = epsilon_init
        self.eps_fin: float = epsilon_final
        self.train_freq: int = model_training_frequency
        self.copy_freq: int = model_copy_frequency
        self.env: Any = RecordEpisodeStatistics(
            env, episodes)
        self.init: str = "HeUniform"
        self.main_qnet: tf.keras.Sequential = self.init_qnet()
        self.target_qnet: tf.keras.Sequential = self.init_qnet()
        self.memory: deque[tuple[tuple[int, int, int], int, float,
                                 tuple[int, int, int], bool]] = deque(maxlen=50000)
        self.batch_size: int = batch_size
        self.qalues: list[float] = []
        self.temporal_differences: list[float] = []

    def init_qnet(self) -> tf.keras.Sequential:
        qnet: tf.keras.Sequential = tf.keras.Sequential()
        qnet.add(tf.keras.layers.Dense(24, input_shape=np.shape(self.env.observation_space.sample()),
                                       activation='relu', kernel_initializer=self.init))
        qnet.add(tf.keras.layers.Dense(
            12, activation='relu', kernel_initializer=self.init))
        qnet.add(tf.keras.layers.Dense(self.env.action_space.n,
                                       activation='linear', kernel_initializer=self.init))
        qnet.compile(loss=tf.keras.losses.Huber(), optimizer=tf.keras.optimizers.Adam(
            learning_rate=self.lr), metrics=["accuracy"])
        return qnet

    def model_single_input(self, model: tf.keras.Sequential, obs: tuple[int, int, int]) -> np.ndarray:
        return np.asarray(model(np.reshape(obs, (1, -1))))[0]

    def choose_action(self, obs: tuple[int, int, int], eps: float | None=None) -> tuple[int, float]:
        '''Uses Epsilon-Greedy Strategy to pick an action based on the observation

        Blake has a likelihood of epsilon to explore, choosing a random action;
        but otherwise he will exploit his current knowledge, taking the best known outcome.

        Parameters:
            obs (tuple[int, int, int]): Observation of the current state from the environment
            eps (float | None): Epsilon value, defaults to Blake's epsilon

        Returns:
            tuple[int, float]: Action to perform and it's Q-Value
        '''
        if (np.random.rand() > (eps or self.eps)):
            options: np.ndarray = self.model_single_input(self.main_qnet, obs)
            return (int(options.argmax()), options.max())
        else:
            return (self.env.action_space.sample(), 0)

    def main(self) -> None:
        '''Runs episodes, intermittently training Blake on his past experiences'''
        obs: tuple[int, int, int]
        next_obs: tuple[int, int, int]
        action: int 
        reward: float
        term: bool
        trunc: bool
        step: int = 0
        for ep in range(self.episodes):
            print(f"Episode {ep}")
            obs = self.env.reset()[0]
            episode_qalue: float = 0
            curr_qalue: float = 0
            term = False
            trunc = False
            while not (term or trunc):
                step += 1
                action,  curr_qalue = self.choose_action(obs)
                episode_qalue += curr_qalue
                next_obs, reward, term, trunc = self.env.step(action)[:-1]
                self.memory.append((obs, action, reward, next_obs, term))
                obs = next_obs
                if ((not step % self.train_freq or term or trunc) and len(self.memory) >= self.batch_size*8):
                    self.train()
            if (step >= self.copy_freq):
                self.target_qnet.set_weights(self.main_qnet.get_weights())
                step = 0
            # Decaying epsilon so blake gets more confident over time
            self.eps = np.exp(ep*np.log(self.eps_fin)/self.episodes)
            self.qalues.append(episode_qalue)

    def play(self, episodes: int=1, env: Env | None=None) -> None:
        '''Runs episodes without training Blake

        Parameters:
            episodes (int): Number of episodes to play
            env (Env | None): Environment to play on, defaults to Blake's current environment

        '''
        env = env or self.env
        for ep in range(episodes):
            print(f"Episode {ep}")
            obs = env.reset()[0]
            episode_qalue: float = 0
            curr_qalue: float = 0
            term: bool = False
            trunc: bool = False
            step: int = 0
            while not (term or trunc):
                step += 1
                action,  curr_qalue = self.choose_action(obs, -1)
                episode_qalue += curr_qalue
                next_obs, reward, term, trunc = env.step(action)[:-1]
                print(f"Step {step}: {(obs, action, reward, next_obs, term)}")
                obs = next_obs

    def train(self) -> None:
        '''Trains Blake'''
        discount: float = .618
        alpha: float = .7
        mem_arr: np.ndarray = np.asarray(self.memory, dtype=object)[np.random.choice(len(self.memory), self.batch_size, False)]
        all_obs: np.ndarray = np.asarray([i[0] for i in mem_arr])
        all_nexts: np.ndarray = np.asarray([i[3] for i in mem_arr])
        current_qals: np.ndarray = self.main_qnet.predict(all_obs)
        next_qals: np.ndarray = self.target_qnet.predict(all_nexts)
        rewards: np.ndarray = mem_arr[:, 2].astype(float)
        terminations: np.ndarray = np.asarray(mem_arr[:, 4], dtype=float)
        best_nexts: np.ndarray = next_qals.max(1)*np.logical_not(terminations)
        temp_diffs: np.ndarray = rewards + discount * best_nexts
        self.temporal_differences.append(temp_diffs.mean())
        expected_qals: np.ndarray = current_qals.copy()
        changing_qals: np.ndarray = expected_qals[np.arange(self.batch_size), mem_arr[:, 1].astype(int)]
        changing_qals = (1-alpha)*changing_qals + alpha*temp_diffs.reshape(-1, 1)
        self.main_qnet.fit(all_obs, expected_qals, batch_size=self.batch_size)
        return


# functions are pretty much copy pasted from gymnasium website
def create_training_fig(agent, env, fig=None, rolling_length=500):
    if fig is None:
        fig, axs = plt.subplots(ncols=3, figsize=(12, 5))
    else:
        axs = fig.subplots(ncols=3)
    axs[0].set_title("Episode rewards")
    reward_moving_average = (
        np.convolve(
            np.array(env.return_queue).flatten(), np.ones(rolling_length), mode="valid"
        )
        / rolling_length
    )
    axs[0].plot(range(len(reward_moving_average)), reward_moving_average)
    axs[1].set_title("Episode lengths")
    length_moving_average = (
        np.convolve(
            np.array(env.length_queue).flatten(), np.ones(rolling_length), mode="same"
        )
        / rolling_length
    )
    axs[1].plot(range(len(length_moving_average)), length_moving_average)
    axs[2].set_title("Training Error")
    training_error_moving_average = (
        np.convolve(np.array(agent.temporal_differences),
                    np.ones(rolling_length), mode="same")
        / rolling_length
    )
    axs[2].plot(range(len(training_error_moving_average)),
                training_error_moving_average)
    return fig


if __name__ == "__main__":
    SAVE_PATH: str = f"training_vids/{time.time()}"
    blake: Blake = Blake(
        episodes=500, model_training_frequency=4, model_copy_frequency=100,
        batch_size=128, env=RecordVideo(gym.make("CartPole-v1", render_mode="rgb_array"), SAVE_PATH, lambda i: bool(np.any((np.arange(-1,1)+i)%50==0))))
    blake.main()
    fig: Figure = plt.figure()
    sfigs: list[SubFigure] = fig.subfigures(2)
    sfigs[0].add_subplot().plot(blake.qalues)
    sfigs[0].suptitle("Total Q-Values Per Episode")
    # plt.show()
    create_training_fig(blake, blake.env, sfigs[1], 1)
    plt.show(block=True)