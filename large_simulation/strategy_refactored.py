"""
Multi-Agent DDoS Defense Strategy Comparison

This module implements and compares different cooperative strategies for DDoS defense:
- Q-Learning: Reinforcement learning approach
- Tit-for-Tat: Reciprocal cooperation
- Accumulate Credits: Always cooperate
- Detective: Copycat with cheat-back punishment
- Grudger: Permanent punishment for cheating

Each strategy runs for multiple episodes and rewards are tracked for comparison.
"""

import gym
import numpy as np
import random
from ddos_gym.envs.defense import Defense
import pandas as pd
import matplotlib.pyplot as plt


# ==================== Configuration ====================
ALPHA = 0.7  # Q-learning weight factor for value updates
DISCOUNT_FACTOR = 0.618  # Q-learning discount factor for future rewards
INITIAL_EPSILON = 1.0  # Starting exploration rate
MAX_EPSILON = 1.0  # Maximum exploration rate
MIN_EPSILON = 0.01  # Minimum exploration rate
EPSILON_DECAY = 0.01  # Rate of exploration decay
TRAIN_EPISODES = 1500  # Number of training episodes per strategy
MAX_ROUNDS_PER_EPISODE = 10  # Maximum rounds in each episode
ACCOUNT_LIMIT = 4  # Maximum credits an agent can hold
MODE = 'cedric'  # Credit allocation mode ('cedric' or 'shared')
RANDOM_SEED = 42

random.seed(RANDOM_SEED)


# ==================== Helper Functions ====================
def state_tuple_to_index(agent_states, base, max_index):
    """
    Convert a tuple of agent states to a unique index for Q-table.

    Args:
        agent_states: Tuple/list of state values for all agents
        base: Base value for conversion (typically account_limit)
        max_index: Maximum allowed index value

    Returns:
        Integer index for Q-table lookup
    """
    index = 0
    for state_value in agent_states:
        index = index * base + state_value

    # Clamp to valid range
    if index >= max_index:
        index = max_index - 1
    if index < 0:
        index = 0

    return int(index)


def calculate_investment(agent_id, agent_states, agent_actions, strategy_name):
    """
    Calculate how much an agent should invest based on their strategy.

    Args:
        agent_id: ID of the agent
        agent_states: Current state (credits) for all agents
        agent_actions: Current actions for all agents
        strategy_name: Name of strategy being used

    Returns:
        Investment amount (integer)
    """
    current_credits = agent_states[agent_id]

    # If cooperating and has credits, invest all available
    if agent_actions[agent_id] == 1 and current_credits > 0:
        return current_credits

    return 0


# ==================== Strategy Base Class ====================
class DefenseStrategy:
    """Base class for DDoS defense strategies."""

    def __init__(self, env, graph, name="Base Strategy"):
        """
        Initialize a defense strategy.

        Args:
            env: Gym environment for DDoS simulation
            graph: Defense graph containing agent information
            name: Human-readable name for this strategy
        """
        self.env = env
        self.graph = graph
        self.name = name
        self.num_agents = len(graph.agents)

    def select_action(self, agent_id, agent_states, episode, step):
        """
        Select an action for a given agent.

        Args:
            agent_id: ID of the agent making decision
            agent_states: Current states of all agents
            episode: Current episode number
            step: Current step within episode

        Returns:
            Action (0=don't cooperate, 1=cooperate)
        """
        raise NotImplementedError("Subclasses must implement select_action()")

    def update(self, agent_id, agent_states, agent_actions, rewards, new_agent_states):
        """
        Update strategy's internal state based on outcomes.

        Args:
            agent_id: ID of the agent
            agent_states: Previous states
            agent_actions: Actions taken
            rewards: Rewards received
            new_agent_states: New states after action
        """
        pass  # Default: no update needed


# ==================== Q-Learning Strategy ====================
class QLearningStrategy(DefenseStrategy):
    """Q-Learning reinforcement learning strategy for DDoS defense."""

    def __init__(self, env, graph):
        super().__init__(env, graph, "Q-Learning")

        # Initialize Q-tables for action selection and investment
        self.max_q_index = ACCOUNT_LIMIT ** len(env.observation_space.spaces)
        self.q_table_actions = {}  # Q-values for cooperation decisions
        self.q_table_investments = {}  # Q-values for investment decisions

        for agent_id in graph.agents:
            self.q_table_actions[agent_id] = np.zeros(
                (self.max_q_index, env.action_space.n)
            )
            self.q_table_investments[agent_id] = np.zeros(
                (self.max_q_index, ACCOUNT_LIMIT)
            )

        self.epsilon = INITIAL_EPSILON

    def select_action(self, agent_id, agent_states, episode, step):
        """Select action using epsilon-greedy Q-learning."""
        state_index = state_tuple_to_index(agent_states, ACCOUNT_LIMIT, self.max_q_index)

        if random.uniform(0, 1) > self.epsilon:
            # Exploit: choose best known action
            action = np.argmax(self.q_table_actions[agent_id][state_index, :])
        else:
            # Explore: choose random action
            action = self.env.action_space.sample()

        return action

    def select_investment(self, agent_id, agent_states):
        """Select investment amount using Q-table."""
        state_index = state_tuple_to_index(agent_states, ACCOUNT_LIMIT, self.max_q_index)
        current_credits = agent_states[agent_id]

        if current_credits > 0:
            return np.argmax(self.q_table_investments[agent_id][state_index, :])
        return 0

    def update(self, agent_id, agent_states, agent_actions, rewards, new_agent_states):
        """Update Q-table using Q-learning update rule."""
        state_index = state_tuple_to_index(agent_states, ACCOUNT_LIMIT, self.max_q_index)
        new_state_index = state_tuple_to_index(new_agent_states, ACCOUNT_LIMIT, self.max_q_index)
        action = agent_actions[agent_id]
        reward = rewards[agent_id]

        # Q-learning update: Q(s,a) += α * (r + γ * max(Q(s',a')) - Q(s,a))
        current_q = self.q_table_actions[agent_id][state_index, action]
        max_future_q = np.max(self.q_table_actions[agent_id][new_state_index, :])
        new_q = current_q + ALPHA * (reward + DISCOUNT_FACTOR * max_future_q - current_q)
        self.q_table_actions[agent_id][state_index, action] = new_q

    def decay_epsilon(self, episode):
        """Decay exploration rate over time."""
        self.epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * np.exp(-EPSILON_DECAY * episode)


# ==================== Tit-for-Tat Strategy ====================
class TitForTatStrategy(DefenseStrategy):
    """Tit-for-Tat: Mirror the previous action of the victim agent."""

    def __init__(self, env, graph):
        super().__init__(env, graph, "Tit-for-Tat")
        # Track whether each agent cooperated in previous round
        self.previous_cooperation = {agent_id: False for agent_id in graph.agents}

    def select_action(self, agent_id, agent_states, episode, step):
        """Cooperate if the victim cooperated last round, otherwise defect."""
        # Get current victim from environment
        current_event = self.env.ddos[self.env.time]
        victim_agent_id = current_event[1][0]

        # Mirror victim's previous cooperation
        return 1 if self.previous_cooperation[victim_agent_id] else 0

    def update_cooperation_history(self, agent_actions):
        """Track which agents cooperated this round."""
        for agent_id in self.graph.agents:
            self.previous_cooperation[agent_id] = (agent_actions[agent_id] == 1)


# ==================== Accumulate Credits Strategy ====================
class AccumulateCreditsStrategy(DefenseStrategy):
    """Always Cooperate: Accumulate credits through consistent cooperation."""

    def __init__(self, env, graph):
        super().__init__(env, graph, "Accumulate Credits")

    def select_action(self, agent_id, agent_states, episode, step):
        """Always cooperate to accumulate credits."""
        return 1


# ==================== Detective Strategy ====================
class DetectiveStrategy(DefenseStrategy):
    """
    Detective: Start by not cooperating (cheating).
    If victim retaliates, switch to copycat (Tit-for-Tat).
    Otherwise, continue cheating to exploit.
    """

    def __init__(self, env, graph):
        super().__init__(env, graph, "Detective")
        self.cheat_back_mode = False  # Whether we've been punished
        self.previous_actions = {}

    def select_action(self, agent_id, agent_states, episode, step):
        """Select action based on detective logic."""
        current_event = self.env.ddos[self.env.time]
        victim_agent_id = current_event[1][0]

        # Check if victim retaliated in previous round
        if self.previous_actions.get(victim_agent_id) == 0:
            self.cheat_back_mode = True

        if self.cheat_back_mode:
            # Copycat: mirror victim's previous action
            return self.previous_actions.get(victim_agent_id, 0)
        else:
            # Exploit: don't cooperate
            return 0

    def update(self, agent_id, agent_states, agent_actions, rewards, new_agent_states):
        """Track previous actions for all agents."""
        self.previous_actions = dict(agent_actions)


# ==================== Grudger Strategy ====================
class GrudgerStrategy(DefenseStrategy):
    """
    Grudger: Start cooperating. If any agent cheats, permanently stop cooperating.
    """

    def __init__(self, env, graph):
        super().__init__(env, graph, "Grudger")
        self.has_been_cheated = False

    def select_action(self, agent_id, agent_states, episode, step):
        """Cooperate unless someone has cheated."""
        current_event = self.env.ddos[self.env.time]
        victim_agent_id = current_event[1][0]

        if self.has_been_cheated:
            return 0  # Never cooperate again
        else:
            return 1  # Cooperate

    def update(self, agent_id, agent_states, agent_actions, rewards, new_agent_states):
        """Check if any agent cheated (didn't cooperate)."""
        current_event = self.env.ddos[self.env.time]
        victim_agent_id = current_event[1][0]

        if agent_actions.get(victim_agent_id) == 0:
            self.has_been_cheated = True


# ==================== Strategy Runner ====================
def run_strategy(strategy, num_episodes, max_rounds):
    """
    Run a strategy for multiple episodes and collect rewards.

    Args:
        strategy: Strategy object to run
        num_episodes: Number of episodes to run
        max_rounds: Maximum rounds per episode

    Returns:
        List of reward arrays (one per episode)
    """
    episode_rewards = []

    for episode in range(num_episodes):
        # Reset environment
        agent_states = strategy.env.reset()
        total_rewards = [0] * strategy.num_agents

        # Run episode
        for step in range(max_rounds):
            agent_actions = {}
            agent_investments = np.zeros(strategy.num_agents)

            # Each agent selects action and investment
            for agent_id in strategy.graph.agents:
                action = strategy.select_action(agent_id, agent_states, episode, step)
                agent_actions[agent_id] = action

                # Determine investment based on strategy
                if hasattr(strategy, 'select_investment'):
                    agent_investments[agent_id] = strategy.select_investment(agent_id, agent_states)
                else:
                    agent_investments[agent_id] = calculate_investment(
                        agent_id, agent_states, agent_actions, strategy.name
                    )

            # Execute actions in environment
            new_agent_states, rewards = strategy.env.step(agent_investments, agent_actions)

            # Update strategy and track rewards
            for agent_id in strategy.graph.agents:
                strategy.update(agent_id, agent_states, agent_actions, rewards, new_agent_states)
                total_rewards[agent_id] += rewards[agent_id]
                agent_states[agent_id] = new_agent_states[agent_id]

            # Strategy-specific updates
            if isinstance(strategy, TitForTatStrategy):
                strategy.update_cooperation_history(agent_actions)

            # Decay epsilon for Q-learning
            if isinstance(strategy, QLearningStrategy):
                strategy.decay_epsilon(episode)

            strategy.env.time += 1

        episode_rewards.append(total_rewards)

    return episode_rewards


# ==================== Main Execution ====================
def main():
    """Main execution: train all strategies and compare results."""

    # Initialize environment
    env = gym.make('ddos-v0', mode=MODE)
    graph = Defense()

    # Define all strategies
    strategies = [
        QLearningStrategy(env, graph),
        TitForTatStrategy(env, graph),
        AccumulateCreditsStrategy(env, graph),
        DetectiveStrategy(env, graph),
        GrudgerStrategy(env, graph)
    ]

    # Run all strategies and collect rewards
    all_rewards = []
    strategy_names = []

    print("Training strategies...")
    for strategy in strategies:
        print(f"  Running {strategy.name}...")
        rewards = run_strategy(strategy, TRAIN_EPISODES, MAX_ROUNDS_PER_EPISODE)
        all_rewards.extend(rewards)
        strategy_names.append(strategy.name)

    # Save all rewards to CSV
    df_all_rewards = pd.DataFrame(all_rewards)
    df_all_rewards.to_csv('output/strategy.csv', index=False)

    # Calculate average rewards per strategy per episode
    df_rewards = pd.read_csv('output/strategy.csv')
    strategy_avg_rewards = {
        'q_learning': df_rewards.iloc[0:TRAIN_EPISODES].reset_index(drop=True).mean(axis=1),
        'tit_for_tat': df_rewards.iloc[TRAIN_EPISODES:2*TRAIN_EPISODES].reset_index(drop=True).mean(axis=1),
        'accumulate_credits': df_rewards.iloc[2*TRAIN_EPISODES:3*TRAIN_EPISODES].reset_index(drop=True).mean(axis=1),
        'detective': df_rewards.iloc[3*TRAIN_EPISODES:4*TRAIN_EPISODES].reset_index(drop=True).mean(axis=1),
        'grudger': df_rewards.iloc[4*TRAIN_EPISODES:].reset_index(drop=True).mean(axis=1)
    }

    # Save strategy average rewards
    df_strategy_rewards = pd.DataFrame(strategy_avg_rewards)
    df_strategy_rewards.to_csv('output/strategy_rewards.csv', index=False)

    # Calculate rolling mean for smoother visualization
    window_size = 100
    df_rolling_rewards = df_strategy_rewards.rolling(window=window_size).mean()

    # Plot results
    print("Generating plots...")
    plt.figure(figsize=(8, 6))
    plt.plot(df_rolling_rewards['q_learning'], label='Q-Learning')
    plt.plot(df_rolling_rewards['tit_for_tat'], label='Tit-for-Tat')
    plt.plot(df_rolling_rewards['accumulate_credits'], label='Accumulate Credits')
    plt.plot(df_rolling_rewards['detective'], label='Detective')
    plt.plot(df_rolling_rewards['grudger'], label='Grudger')

    plt.title('Average Rewards per Strategy Group per Episode', fontsize=14)
    plt.xlabel('Episodes', fontsize=14)
    plt.ylabel('Average Rewards', fontsize=14)
    plt.legend(loc='upper left')
    plt.grid(True)

    plt.savefig('output/average_rewards_per_strategy.png')
    print("Results saved to output/ directory")


if __name__ == '__main__':
    main()
