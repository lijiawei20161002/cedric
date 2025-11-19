"""
Deep Q-Network (DQN) for Multi-Agent DDoS Defense

This module implements a Deep Q-Network approach for learning optimal defense
strategies in the multi-agent DDoS environment. Each agent uses DQN to learn:
1. When to cooperate (action)
2. How much to invest (investment amount)
"""

from ddos_gym.envs.defense import Defense
import gym
import numpy as np
import tensorflow as tf
from collections import deque
import random

random.seed(42)


# ==================== Configuration ====================
STATE_SIZE = None  # Determined at runtime based on number of agents
ACTION_SIZE = 2  # 0=don't cooperate, 1=cooperate
MEMORY_SIZE = 2000  # Experience replay buffer size
GAMMA = 0.95  # Discount factor for future rewards
INITIAL_EPSILON = 1.0  # Starting exploration rate
MIN_EPSILON = 0.01  # Minimum exploration rate
EPSILON_DECAY = 0.995  # Epsilon decay per episode
BATCH_SIZE = 32  # Minibatch size for training
TRAIN_EPISODES = 100  # Number of training episodes
MAX_STEPS_PER_EPISODE = 500  # Maximum steps per episode


# ==================== DQN Model ====================
class DQNModel(tf.keras.Model):
    """
    Deep Q-Network neural network model.

    Architecture:
    - Input: State vector (agent credit balances)
    - Hidden Layer 1: 24 neurons with ReLU activation
    - Hidden Layer 2: 24 neurons with ReLU activation
    - Output: Q-values for each action (linear activation)
    """

    def __init__(self, output_size):
        """
        Initialize the DQN model.

        Args:
            output_size: Number of possible actions (output Q-values)
        """
        super(DQNModel, self).__init__()
        self.dense1 = tf.keras.layers.Dense(24, activation='relu')
        self.dense2 = tf.keras.layers.Dense(24, activation='relu')
        self.output_layer = tf.keras.layers.Dense(output_size, activation='linear')

    def call(self, inputs):
        """
        Forward pass through the network.

        Args:
            inputs: State tensor

        Returns:
            Q-values for each action
        """
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.output_layer(x)


# ==================== DQN Agent ====================
class DQNAgent:
    """
    DQN agent that learns to select actions and investments.

    The agent maintains two separate Q-networks:
    1. Action Q-network: Learns when to cooperate
    2. Investment Q-network: Learns how much to invest
    """

    def __init__(self, state_size, action_size, investment_size):
        """
        Initialize the DQN agent.

        Args:
            state_size: Dimension of state space
            action_size: Number of possible actions
            investment_size: Number of possible investment amounts
        """
        self.state_size = state_size
        self.num_agents = state_size
        self.action_size = action_size
        self.investment_size = investment_size

        # Experience replay memory
        self.memory = deque(maxlen=MEMORY_SIZE)

        # Q-learning parameters
        self.gamma = GAMMA
        self.epsilon = INITIAL_EPSILON
        self.epsilon_min = MIN_EPSILON
        self.epsilon_decay = EPSILON_DECAY

        # Create Q-networks for actions and investments
        self.action_network = DQNModel(action_size)
        self.investment_network = DQNModel(investment_size)

        # Compile networks
        self.action_network.compile(
            loss='mse',
            optimizer=tf.keras.optimizers.Adam()
        )
        self.investment_network.compile(
            loss='mse',
            optimizer=tf.keras.optimizers.Adam()
        )

    def remember(self, state, investments, actions, reward, next_state, done=False):
        """
        Store experience in replay memory.

        Args:
            state: Current state
            investments: Investment amounts chosen
            actions: Actions chosen
            reward: Reward received
            next_state: Next state
            done: Whether episode is finished
        """
        self.memory.append((state, investments, actions, reward, next_state, done))

    def act(self, state):
        """
        Select actions and investments for all agents using epsilon-greedy policy.

        Args:
            state: Current state observation

        Returns:
            tuple: (investments_dict, actions_dict)
        """
        actions = {}
        investments = {}

        for agent_id in range(self.num_agents):
            if np.random.rand() <= self.epsilon:
                # Explore: choose random action and investment
                actions[agent_id] = random.randrange(self.action_size)
                investments[agent_id] = random.randrange(self.investment_size)
            else:
                # Exploit: choose best known action and investment
                investment_q_values = self.investment_network.predict(state, verbose=0)
                investments[agent_id] = np.argmax(investment_q_values[0])

                action_q_values = self.action_network.predict(state, verbose=0)
                actions[agent_id] = np.argmax(action_q_values[0])

        return investments, actions

    def replay(self, batch_size):
        """
        Train networks using experience replay.

        Samples a minibatch from memory and performs Q-learning updates:
        Q(s,a) = r + γ * max(Q(s',a'))

        Args:
            batch_size: Size of minibatch to sample
        """
        if len(self.memory) < batch_size:
            return

        # Sample random minibatch
        minibatch = random.sample(self.memory, batch_size)

        for state, investments, actions, reward, next_state, done in minibatch:
            # Calculate target Q-values for action network
            target_action_q = reward
            if not done:
                next_action_q = self.action_network.predict(next_state, verbose=0)
                target_action_q = reward + self.gamma * np.amax(next_action_q[0])

            # Update action Q-network
            current_action_q = self.action_network.predict(state, verbose=0)
            current_action_q[0][actions] = target_action_q
            self.action_network.fit(state, current_action_q, epochs=1, verbose=0)

            # Calculate target Q-values for investment network
            target_investment_q = reward
            if not done:
                next_investment_q = self.investment_network.predict(next_state, verbose=0)
                target_investment_q = reward + self.gamma * np.amax(next_investment_q[0])

            # Update investment Q-network
            current_investment_q = self.investment_network.predict(state, verbose=0)
            current_investment_q[0][investments] = target_investment_q
            self.investment_network.fit(state, current_investment_q, epochs=1, verbose=0)

        # Decay exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_models(self, action_path, investment_path):
        """
        Save trained models to disk.

        Args:
            action_path: Path to save action network
            investment_path: Path to save investment network
        """
        self.action_network.save(action_path)
        self.investment_network.save(investment_path)

    def load_models(self, action_path, investment_path):
        """
        Load trained models from disk.

        Args:
            action_path: Path to load action network
            investment_path: Path to load investment network
        """
        self.action_network = tf.keras.models.load_model(action_path)
        self.investment_network = tf.keras.models.load_model(investment_path)


# ==================== Training Loop ====================
def train_dqn_agent():
    """
    Train a DQN agent on the DDoS defense environment.
    """
    # Initialize environment
    env = gym.make('ddos-v0')
    state_size = len(env.account)
    action_size = env.action_space.n
    investment_size = env.account_limit + 1  # 0 to account_limit

    # Create agent
    agent = DQNAgent(state_size, action_size, investment_size)

    print(f"Training DQN agent...")
    print(f"State size: {state_size}")
    print(f"Action size: {action_size}")
    print(f"Investment size: {investment_size}")
    print()

    # Training loop
    for episode in range(TRAIN_EPISODES):
        # Reset environment
        state = env.reset()

        # Validate state size
        if len(state) != state_size:
            print(f"Warning: State size mismatch. Expected {state_size}, got {len(state)}")

        state = np.reshape(state, [1, state_size])
        total_reward = 0

        # Episode loop
        for step in range(MAX_STEPS_PER_EPISODE):
            # Select actions and investments
            investments, actions = agent.act(state)

            # Execute actions in environment
            next_state, rewards = env.step(investments, actions)
            next_state = np.reshape(next_state, [1, state_size])

            # Calculate average reward across all agents
            avg_reward = np.mean(list(rewards.values())) if len(rewards) > 0 else 0
            total_reward += avg_reward

            # Store experience (simplified: using avg reward for all agents)
            done = (step == MAX_STEPS_PER_EPISODE - 1)
            agent.remember(state, investments, actions, avg_reward, next_state, done)

            # Update state
            state = next_state

            # Train networks with experience replay
            if len(agent.memory) > BATCH_SIZE:
                agent.replay(BATCH_SIZE)

            if done:
                break

        # Print progress
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}/{TRAIN_EPISODES}, "
                  f"Total Reward: {total_reward:.2f}, "
                  f"Epsilon: {agent.epsilon:.4f}")

    print("\nTraining completed!")

    # Save trained models
    agent.save_models('models/action_network.h5', 'models/investment_network.h5')
    print("Models saved to models/ directory")

    return agent


# ==================== Main Execution ====================
if __name__ == '__main__':
    trained_agent = train_dqn_agent()
