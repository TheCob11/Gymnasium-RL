from collections import defaultdict
from typing import Any
import gymnasium as gym
from matplotlib.figure import Figure, SubFigure
import numpy as np
from numpy.typing import NDArray
from gymnasium import Env
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns


class BlackjackJake:
    def __init__(
        self,
        learn_rate: float = 0.01,
        episodes: int = 2000,
        epsilon_init: float = 1,
        epsilon_min: float = 0.01,
        epsilon_decay: float | None = None,
        discount: float = (1+5**.5)/2,
        env: Env[Any, Any] = gym.make("Blackjack-v1")
    ) -> None:
        '''Creates a new Blackjack Agent

        Parameters:
            learn_rate (float): Alpha value that determines how fast/slow Jake learns
            episodes (int): How many "episodes"(rounds of Blackjack) to play
            epsilon_init (float): Initial value for epsilon
            epsilon_min (float): Minimum value for epsilon
            epsilon_decay (float): Value that determines how quickly epsilon decays, making Jake more likely to do what he knows as time goes on
            discount (float): Gamma value that determines how significant the quantity of the next reward is in training(default is golden ratio)
            env (Env[Any, Any]): The Blackjack environment for Jake to play on

        Returns:
            None
        '''
        self.alpha: float = learn_rate
        self.episodes: int = episodes
        self.eps: float = epsilon_init
        self.eps_min: float = epsilon_min
        self.eps_dec: float = self.eps / \
            (episodes/2) if epsilon_decay is None else epsilon_decay
        self.gamma: float = discount
        self.env: Env[Any, Any] = gym.wrappers.RecordEpisodeStatistics(
            env, episodes)
        self.qable: defaultdict[tuple[int, int, bool],
                                NDArray] = defaultdict(lambda: np.zeros(self.env.action_space.n))
        self.qalues: list[float] = []
        self.errors: list[float] = []

    def choose_action(self, obs: tuple[int, int, bool]) -> tuple[int, float]:
        '''Uses Epsilon-Greedy Strategy to pick an action based on the observation

        Jake has a likelihood of epsilon to explore, choosing a random action,
        but otherwise he will exploit his current knowledge, taking the best known outcome.

        Parameters:
            obs (tuple[int, int, bool]): Observation of the current state from the environment

        Returns:
            tuple[int, float]: Action to perform(0 for stick, 1 for hit) and it's Q-Value
        '''
        if (np.random.rand() > self.eps):
            action: int = int(np.argmax(self.qable[obs]))
            return (action, self.qable[obs][action])
        else:
            return (self.env.action_space.sample(), 0)

    def train(self) -> None:
        '''Trains Jake's Q-Table so he can learn how to play

        Paramaters:
            None

        Returns:
            None
        '''
        obs: tuple[int, int, bool]  # Observation: sum of current hand, dealer's face up card, has a usable ace
        next_obs: tuple[int, int, bool]  # State after action is taken
        action: int  # 0 for stick, 1 for hit
        reward: float  # +1 for win, -1 for loss, 0 for draw
        term: bool  # Whether episode has finished
        trunc: bool  # Whether episode has been prematurely ended
        for i in range(self.episodes):
            obs = self.env.reset()[0]
            episode_qalue: float = 0
            curr_qalue: float = 0
            term = False
            trunc = False
            # all_eps.append(self.eps)
            while not (term or trunc):
                action,  curr_qalue = self.choose_action(obs)
                episode_qalue += curr_qalue
                next_obs, reward, term, trunc = self.env.step(action)[:-1]
                # rewards.append(reward)
                self.update(obs, action, next_obs, reward, term)
                obs = next_obs
            # Decaying epsilon so Jake gets more confident over time
            self.eps = max(self.eps_min, self.eps-self.eps_dec)
            self.qalues.append(episode_qalue)

    def update(self, state: tuple[int, int, bool], action: int, next_state: tuple[int, int, bool], reward: float, term: bool) -> None:
        '''Updates Jake's Q-Table according to the action and reward

        Parameters:
            state (tuple[int, int, bool]): The state/observation when the action was taken
            action (int): The action taken in the aforementioned state
            next_state (tuple[int, int, bool]): The new state/observation resulting from the action
            reward (float): The reward recieved from taking the action
            term (bool): Whether or not the episode was terminated after the action was taken

        Returns:
            None
        '''
        current: float = self.qable[state][action]
        error: float = (reward + self.gamma *
                        self.qable[next_state].max()*(not term) - current)
        self.qable[state][action] = current + self.alpha*error
        self.errors.append(error)


jake: BlackjackJake = BlackjackJake(episodes=100000)
jake.train()
fig: Figure = plt.figure()
sfigs: list[SubFigure] = fig.subfigures(4)
sfigs[0].add_subplot().plot(jake.qalues)
sfigs[0].suptitle("Total Q-Values Per Episode")
# plt.show()


# functions are pretty much copy pasted from gymnasium website
def create_training_fig(agent, env, fig=None):
    rolling_length = 500
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
        np.convolve(np.array(agent.errors),
                    np.ones(rolling_length), mode="same")
        / rolling_length
    )
    axs[2].plot(range(len(training_error_moving_average)),
                training_error_moving_average)
    return fig

def create_grids(agent, usable_ace=False):
    """Create value and policy grid given an agent."""
    # convert our state-action values to state values
    # and build a policy dictionary that maps observations to actions
    state_value = defaultdict(float)
    policy = defaultdict(int)
    for obs, action_values in agent.qable.items():
        state_value[obs] = float(np.max(action_values))
        policy[obs] = int(np.argmax(action_values))

    player_count, dealer_count = np.meshgrid(
        # players count, dealers face-up card
        np.arange(12, 22),
        np.arange(1, 11),
    )

    # create the value grid for plotting
    value = np.apply_along_axis(
        lambda obs: state_value[(obs[0], obs[1], usable_ace)],
        axis=2,
        arr=np.dstack([player_count, dealer_count]),
    )
    value_grid = player_count, dealer_count, value

    # create the policy grid for plotting
    policy_grid = np.apply_along_axis(
        lambda obs: policy[(obs[0], obs[1], usable_ace)],
        axis=2,
        arr=np.dstack([player_count, dealer_count]),
    )
    return value_grid, policy_grid

def create_plots(value_grid, policy_grid, title: str, fig=None):
    """Creates a plot using a value and policy grid."""
    # create a new figure with 2 subplots (left: state values, right: policy)
    player_count, dealer_count, value = value_grid
    if fig is None:
        fig = plt.figure(figsize=plt.figaspect(0.4))
        fig.suptitle(title, fontsize=16)
    # plot the state values
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.plot_surface(
        player_count,
        dealer_count,
        value,
        rstride=1,
        cstride=1,
        cmap="viridis",
        edgecolor="none",
    )
    ax1.set_xticks(range(12, 22), range(12, 22))
    ax1.set_yticks(range(1, 11), ["A"] + list(range(2, 11)))
    ax1.set_title(f"State values: {title}")
    ax1.set_xlabel("Player sum")
    ax1.set_ylabel("Dealer showing")
    ax1.zaxis.set_rotate_label(False)
    ax1.set_zlabel("Value", fontsize=14, rotation=90)
    ax1.view_init(20, 220)

    # plot the policy
    ax2 = fig.add_subplot(1,2,2)
    sns.heatmap(policy_grid, annot=True, cmap="Accent_r", cbar=False, ax=ax2)
    ax2.set_title(f"Policy: {title}")
    ax2.set_xlabel("Player sum")
    ax2.set_ylabel("Dealer showing")
    ax2.set_xticklabels(range(12, 22))
    ax2.set_yticks(range(10))
    ax2.set_yticklabels(["A"] + list(range(2, 11)), fontsize=12)

    # add a legend
    legend_elements = [
        Patch(facecolor="lightgreen", edgecolor="black", label="Hit"),
        Patch(facecolor="grey", edgecolor="black", label="Stick"),
    ]
    ax2.legend(handles=legend_elements, bbox_to_anchor=(1.3, 1))
    return fig


# state values & policy with usable ace (ace counts as 11)
# value_grid, policy_grid = create_grids(jake, usable_ace=True)
# figAce = create_plots(value_grid, policy_grid, title="With usable ace")
# # plt.show()
# figNAce = create_plots(*create_grids(jake, usable_ace=False),
#                     title="Without usable ace")
create_training_fig(jake, jake.env, sfigs[1])
jake.env.close()
create_plots(*create_grids(jake, usable_ace=False),
             title="Without usable ace", fig=sfigs[2])
create_plots(*create_grids(jake, True),
             title="With usable ace", fig=sfigs[3])
plt.show()
