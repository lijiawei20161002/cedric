"""
DDoS Multi-Agent Simulation Environment

This OpenAI Gym environment simulates a multi-agent DDoS defense scenario where
agents (countries) can cooperate to defend against attacks. Agents earn rewards
and credits based on their contributions.

Two credit allocation modes are supported:
- 'cedric': Uses Shapley value-based credit assignment (CEDRIC algorithm)
- 'shared': Equally distributes social gain among coalition members
"""

import gym
from gym import spaces
import numpy as np
import csv
from ddos_gym.envs.defense import Defense
import random


# ==================== Configuration ====================
INITIAL_BALANCE = 4  # Starting credits for each agent
SHAPLEY_SAMPLES = 5  # Number of Monte Carlo samples for Shapley value estimation
ACCOUNT_LIMIT = 4  # Maximum credits an agent can hold
MAX_AGENT_NUM = 200  # Maximum number of agents in simulation
RANDOM_SEED = 42

random.seed(RANDOM_SEED)


class DDoS(gym.Env):
    """
    Multi-agent DDoS defense simulation environment.

    In each timestep:
    1. A DDoS attack event occurs (from attack.csv)
    2. Agents decide whether to cooperate (action) and how much to invest
    3. Defense is executed based on coalition
    4. Rewards and credits are distributed based on outcomes
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, mode='cedric', size=5):
        """
        Initialize the DDoS environment.

        Args:
            render_mode: Rendering mode (not currently used)
            mode: Credit allocation mode ('cedric' or 'shared')
            size: Size parameter (not currently used)
        """
        # Attack event data
        self.ddos_events = []  # List of (attackers, victims, bandwidth) tuples
        self.current_time_step = 0

        # Configuration
        self.credit_mode = mode
        self.account_limit = ACCOUNT_LIMIT
        self.max_agent_num = MAX_AGENT_NUM

        # Network and agents
        self.network_graph = Defense()
        self.ddos_events = self._load_attack_data("ddos_gym/ddos_gym/envs/data/attack.csv")
        self.agent_credits = {agent_id: INITIAL_BALANCE for agent_id in self.network_graph.agents}

        # Define action and observation spaces
        self.action_space = spaces.Discrete(2)  # 0=don't cooperate, 1=cooperate
        credit_spaces = [spaces.Discrete(self.account_limit)] * len(self.network_graph.agents)
        self.observation_space = spaces.Tuple([spaces.Discrete(MAX_AGENT_NUM)] + credit_spaces)

    def _load_attack_data(self, filepath):
        """
        Load DDoS attack events from CSV file.

        CSV format: Each row contains attack information including:
        - Source countries (column 4)
        - Destination countries (column 1)
        - Bandwidth (column 3)

        Args:
            filepath: Path to attack CSV file

        Returns:
            List of attack event tuples (attackers, victims, bandwidth)
        """
        attack_events = []

        with open(filepath, 'r') as f:
            csvreader = csv.reader(f)
            next(csvreader)  # Skip header

            for row in csvreader:
                attackers = self._parse_country_list(row[4])
                victims = self._parse_country_list(row[1])
                bandwidth = float(row[3])
                attack_events.append((attackers, victims, bandwidth))

        return attack_events

    def _parse_country_list(self, country_str):
        """
        Parse a string representation of country list into agent IDs.

        Input format: "['US', 'CN', ...]"

        Args:
            country_str: String containing list of country names

        Returns:
            List of agent IDs
        """
        agent_ids = []

        # Extract country names from string format
        raw_countries = country_str.split('[')[1].split(']')[0].split(",")

        for country_part in raw_countries:
            if len(country_part.split("\'")) < 2:
                continue

            country_name = country_part.split("\'")[1].split("\'")[0]

            if country_name in self.network_graph.country_to_id:
                agent_id = self.network_graph.country_to_id[country_name]
                agent_ids.append(agent_id)

        return agent_ids

    def reset(self, seed=None, options=None):
        """
        Reset the environment to initial state.

        Returns:
            Initial observation (credit balances for all agents)
        """
        self.current_time_step = 0
        self.agent_credits = {agent_id: INITIAL_BALANCE for agent_id in self.network_graph.agents}
        observation = np.array(list(self.agent_credits.values()))
        return observation

    def step(self, agent_investments, agent_actions):
        """
        Execute one timestep of the environment.

        Args:
            agent_investments: Dict mapping agent_id -> investment amount
            agent_actions: Dict mapping agent_id -> action (0 or 1)

        Returns:
            tuple: (observation, rewards)
                - observation: numpy array of agent credit balances
                - rewards: dict mapping agent_id -> reward value
        """
        rewards = {}

        # Get current attack event
        event = self.ddos_events[self.current_time_step]
        attackers, victims, bandwidth = event[0], event[1][0], event[2]

        # Form coalition from agents who chose to cooperate
        coalition = {agent_id for agent_id in self.network_graph.agents if agent_actions[agent_id] == 1}

        # Execute defense and calculate social gain
        defense_successful, social_gain = Defense(attackers, victims, coalition, bandwidth).social_gain()

        # Calculate payoffs and update credits for all coalition members
        for agent_id in coalition:
            payoff, credit_change = self._calculate_agent_outcome(
                agent_id=agent_id,
                victim_id=victims,
                social_gain=social_gain,
                defense_successful=defense_successful,
                agent_actions=agent_actions,
                agent_investments=agent_investments,
                coalition=coalition,
                attackers=attackers,
                bandwidth=bandwidth
            )

            # Update agent's credit balance (averaged over Shapley samples)
            self.agent_credits[agent_id] = int(
                self.agent_credits[agent_id] + credit_change / SHAPLEY_SAMPLES
            )
            rewards[agent_id] = payoff

        # In 'shared' mode, equally distribute social gain among coalition
        if self.credit_mode == 'shared':
            gain_per_agent = social_gain / len(coalition) if len(coalition) > 0 else 0
            for agent_id in coalition:
                self.agent_credits[agent_id] += gain_per_agent

        # Return new observation and rewards
        observation = np.array(list(self.agent_credits.values()))
        return observation, rewards

    def _calculate_agent_outcome(self, agent_id, victim_id, social_gain,
                                   defense_successful, agent_actions, agent_investments,
                                   coalition, attackers, bandwidth):
        """
        Calculate payoff and credit change for an agent.

        Args:
            agent_id: ID of the agent
            victim_id: ID of the victim being defended
            social_gain: Social gain from defense
            defense_successful: Whether defense was successful
            agent_actions: Actions taken by all agents
            agent_investments: Investments made by all agents
            coalition: Set of cooperating agents
            attackers: List of attacker IDs
            bandwidth: Attack bandwidth

        Returns:
            tuple: (payoff, credit_change)
        """
        payoff = 0
        credit_change = 0

        # Victim gains application value if defense successful
        if victim_id == agent_id and defense_successful:
            payoff += self.network_graph.app[agent_id]

        # Cooperating agents pay defense cost
        if agent_actions[agent_id] == 1:
            payoff -= self.network_graph.cost[agent_id]

        # All coalition members share in social gain
        payoff += social_gain

        # CEDRIC mode: Calculate Shapley value-based credits
        if self.credit_mode == 'cedric':
            credit_change = self._calculate_cedric_credits(
                agent_id=agent_id,
                victim_id=victim_id,
                coalition=coalition,
                attackers=attackers,
                bandwidth=bandwidth,
                agent_investments=agent_investments,
                social_gain=social_gain
            )
            # Credits also contribute to immediate payoff
            payoff += credit_change / SHAPLEY_SAMPLES

        return payoff, credit_change

    def _calculate_cedric_credits(self, agent_id, victim_id, coalition,
                                    attackers, bandwidth, agent_investments, social_gain):
        """
        Calculate CEDRIC credits using Monte Carlo Shapley value estimation.

        CEDRIC (Credit-based Defense Incentive Coordination) uses Shapley values
        to fairly distribute credits based on each agent's marginal contribution
        to the defense coalition.

        The Shapley value is estimated by:
        1. Sampling random subsets of the coalition
        2. Comparing defense outcomes with/without the agent
        3. Crediting proportional to marginal contribution

        Args:
            agent_id: Agent whose credits to calculate
            victim_id: Victim being defended
            coalition: Full coalition of defenders
            attackers: List of attacker IDs
            bandwidth: Attack bandwidth
            agent_investments: Investments by all agents
            social_gain: Total social gain from defense

        Returns:
            float: Credit change for the agent
        """
        total_credit = 0
        agent_singleton = {agent_id}

        # Monte Carlo sampling to estimate Shapley value
        for _ in range(SHAPLEY_SAMPLES):
            # Sample random subset of coalition (excluding this agent)
            other_agents = coalition - agent_singleton
            if len(other_agents) > 0:
                subset_size = random.randint(1, len(other_agents))
                random_subset = set(random.sample(list(other_agents), subset_size))
            else:
                random_subset = set()

            # Calculate social gain without this agent
            _, gain_without_agent = Defense(
                attackers, victim_id, random_subset, bandwidth
            ).social_gain()

            # Calculate social gain with this agent added
            subset_with_agent = random_subset.copy()
            subset_with_agent.add(agent_id)
            _, gain_with_agent = Defense(
                attackers, victim_id, subset_with_agent, bandwidth
            ).social_gain()

            # Agent pays investment if they are the victim
            if victim_id == agent_id:
                total_credit -= agent_investments[agent_id]

            # Agent receives credit proportional to marginal contribution
            if gain_with_agent > 0:
                marginal_contribution = gain_with_agent - gain_without_agent
                contribution_ratio = marginal_contribution / gain_with_agent
                total_credit += contribution_ratio * agent_investments[victim_id]

        return total_credit

    def render(self):
        """Render the environment (currently just prints timestep)."""
        if self.render_mode == "human":
            print(f'Step: {self.current_time_step}')

    def close(self):
        """Clean up environment resources."""
        pass
