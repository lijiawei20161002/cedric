from ddos_gym.envs.defense import Defense
import gym
import numpy as np
import tensorflow as tf
from collections import deque
import random

random.seed(42)

# Define the DQN Model
class DQNModel(tf.keras.Model):
    def __init__(self, output_size):
        super(DQNModel, self).__init__()
        self.dense1 = tf.keras.layers.Dense(24, activation='relu')
        self.dense2 = tf.keras.layers.Dense(24, activation='relu')
        self.dense3 = tf.keras.layers.Dense(output_size, activation='linear')

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.dense3(x)

# Define the DQN Agent
class DQNAgent:
    def __init__(self, state_size, action_size, invest_size):
        self.state_size = state_size
        self.num_agents = state_size
        self.action_size = action_size
        self.invest_size = invest_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model_action = DQNModel(action_size)
        self.model_invest = DQNModel(invest_size)
        self.model_action.compile(loss='mse', optimizer=tf.keras.optimizers.Adam())
        self.model_invest.compile(loss='mse', optimizer=tf.keras.optimizers.Adam())

    def remember(self, state, invest_n, action_n, reward, next_state, done=False):
        self.memory.append((state, invest_n, action_n, reward, next_state, done))

    def act(self, state):
        action_n = {}
        invest_n = {}

        for agent_id in range(self.num_agents):
            if np.random.rand() <= self.epsilon:
                action_n[agent_id] = random.randrange(self.action_size)
                invest_n[agent_id] = random.randrange(self.invest_size)
            else:
                # Predict invest_n for each agent
                invest_values = self.model_invest.predict(state)
                invest_n[agent_id] = np.argmax(invest_values[0])

                # Predict action_n for each agent
                action_values = self.model_action.predict(state)
                action_n[agent_id] = np.argmax(action_values[0])

        return invest_n, action_n
    
    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, invest_n, action_n, reward, next_state, done in minibatch:
            # Update for action model
            target_action = reward
            if not done:
                target_action = reward + self.gamma * np.amax(self.model_action.predict(next_state)[0])
            target_f_action = self.model_action.predict(state)
            target_f_action[0][action_n] = target_action

            # Update for invest model
            target_invest = reward
            if not done:
                target_invest = reward + self.gamma * np.amax(self.model_invest.predict(next_state)[0])
            target_f_invest = self.model_invest.predict(state)
            target_f_invest[0][invest_n] = target_invest

            # Fit both models
            self.model_action.fit(state, target_f_action, epochs=1, verbose=0)
            self.model_invest.fit(state, target_f_invest, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Initialize the DDoS gym environment
env = gym.make('ddos-v0')
state_size = len(env.account)  # Adjust based on your environment
action_size = env.action_space.n
invest_size = env.account_limit + 1  # Including 0 to account_limit
agent = DQNAgent(state_size, action_size, invest_size)
batch_size = 32

# Training loop
train_episodes = 100
max_steps_per_episode = 500
for e in range(train_episodes):
    state = env.reset()
    if len(state) != state_size:
        print(f"Warning: State size mismatch. Expected {state_size}, got {len(state)}.")
    state = np.reshape(state, [1, state_size])

    for step in range(max_steps_per_episode):
        invest_n, action_n = agent.act(state)
        next_state, reward = env.step(invest_n, action_n)
        print(next_state)
        next_state = np.reshape(next_state, [1, state_size])

        agent.remember(state, invest_n, action_n, reward, next_state)
        state = next_state

        if len(agent.memory) > batch_size:
            agent.replay(batch_size)

