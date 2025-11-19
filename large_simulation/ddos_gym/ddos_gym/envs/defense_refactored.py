"""
Network Defense Simulation

This module simulates a network topology where agents (countries) can form coalitions
to defend against DDoS attacks by blocking links in the network.
"""

import numpy as np
import csv


class Defense:
    """
    Represents a network topology and defense mechanism against DDoS attacks.

    The network consists of agents (countries) connected by links. Agents can form
    coalitions to defend by removing their connections, attempting to isolate attackers
    from victims.
    """

    # Default values for agent benefits and costs
    DEFAULT_APP_VALUE = 10  # Application value (benefit of successful defense)
    DEFAULT_COST_VALUE = 2  # Cost of participating in defense

    def __init__(self, src='US', dst='JP', coalition=None, bandwidth=1):
        """
        Initialize the defense network.

        Args:
            src: Source of the attack (attacker country/countries)
            dst: Destination of the attack (victim country)
            coalition: Set of agent IDs participating in defense
            bandwidth: Attack bandwidth (volume)
        """
        # Network structure
        self.links = set()  # Set of network links (edges between agents)
        self.agents = set()  # Set of all agent IDs

        # Mapping between country names and agent IDs
        self.country_to_id = {}  # country_name -> agent_id
        self.id_to_country = {}  # agent_id -> country_name

        # Agent properties
        self.app = {}  # Application value per agent (benefit)
        self.cost = {}  # Defense cost per agent

        # Load network topology from file
        self._load_network_topology('ddos_gym/ddos_gym/envs/data/link.txt')

        # Attack/defense parameters
        self.src = src  # Attack source(s)
        self.dst = dst  # Attack destination
        self.coalition = coalition if coalition is not None else []
        self.bandwidth = bandwidth  # Attack volume
        self.max_hop_distance = 2  # Maximum hops to search for paths

    def _load_network_topology(self, filepath):
        """
        Load network topology from file.

        File format: Each line contains two comma-separated country names
        representing a bidirectional link.

        Args:
            filepath: Path to the link topology file
        """
        with open(filepath, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) < 2:
                    continue

                country_a = parts[0].strip()
                country_b = parts[1].strip()

                if not country_a or not country_b:
                    continue

                # Register countries and assign IDs
                agent_a = self._register_country(country_a)
                agent_b = self._register_country(country_b)

                # Add bidirectional link
                self.links.add(frozenset([agent_a, agent_b]))
                self.agents.add(agent_a)
                self.agents.add(agent_b)

    def _register_country(self, country_name):
        """
        Register a country and assign it an agent ID.

        Args:
            country_name: Name of the country

        Returns:
            agent_id: Integer ID for this country
        """
        if country_name not in self.country_to_id:
            agent_id = len(self.country_to_id)
            self.country_to_id[country_name] = agent_id
            self.id_to_country[agent_id] = country_name
            self.app[agent_id] = self.DEFAULT_APP_VALUE
            self.cost[agent_id] = self.DEFAULT_COST_VALUE
            return agent_id
        else:
            return self.country_to_id[country_name]

    # ==================== Setters ====================
    def set_src(self, src):
        """Set attack source."""
        self.src = src

    def set_dst(self, dst):
        """Set attack destination."""
        self.dst = dst

    def set_coalition(self, coalition):
        """Set defending coalition."""
        self.coalition = coalition

    def set_bandwidth(self, bandwidth):
        """Set attack bandwidth."""
        self.bandwidth = bandwidth

    # ==================== Defense Mechanism ====================
    def apply_defense(self, defender_id):
        """
        Apply defense by removing all links connected to a defender.

        This simulates a defender blocking traffic at their location.

        Args:
            defender_id: ID of the agent defending
        """
        links_to_remove = set()
        for link in self.links:
            if defender_id in link:
                links_to_remove.add(link)

        self.links = self.links - links_to_remove

    def check_path_exists(self, source_id, destination_id):
        """
        Check if a path exists between source and destination in current network.

        Uses a breadth-first search approach with hop limit to find connectivity.

        Args:
            source_id: Starting agent ID
            destination_id: Target agent ID

        Returns:
            bool: True if path exists within max_hop_distance, False otherwise
        """
        # Sets of agents reachable from source and destination
        reachable_from_source = {source_id}
        reachable_from_dest = {destination_id}

        hop_count = 0
        searching = True

        while searching:
            searching = False
            links_to_process = set()

            # Expand reachability sets by following links
            for link in self.links:
                # Expand from source side
                if link & reachable_from_source:
                    reachable_from_source = reachable_from_source.union(link)
                    links_to_process.add(link)
                    searching = True

                # Expand from destination side
                if link & reachable_from_dest:
                    reachable_from_dest = reachable_from_dest.union(link)
                    links_to_process.add(link)
                    searching = True

                # Check if source and destination are connected
                if reachable_from_source & reachable_from_dest:
                    return True

            # Remove processed links to avoid reprocessing
            self.links = self.links - links_to_process

            hop_count += 1
            if hop_count > self.max_hop_distance:
                searching = False

        return False

    def execute_defense(self):
        """
        Execute defense by having coalition members block their links,
        then check if any attacker can still reach the victim.

        Returns:
            bool: True if defense successfully blocks all attack paths, False otherwise
        """
        if self.coalition is not None:
            # Apply defense at each coalition member's location
            for defender_id in self.coalition:
                self.apply_defense(defender_id)

            # Check if any attacker can still reach victim
            for attacker_id in self.src:
                if self.check_path_exists(attacker_id, self.dst):
                    return True  # Attack path still exists

        return False  # All paths blocked

    # ==================== Damage Calculation ====================
    def calculate_network_damage(self, coalition=None):
        """
        Calculate bandwidth damage to the network from the attack.

        Damage is measured as the bandwidth reaching agents who are:
        1. Attackers themselves (if not in coalition)
        2. On the path between attackers and victim

        Args:
            coalition: Coalition of defenders (None means no defense)

        Returns:
            float: Total bandwidth damage to the network
        """
        total_damage = 0

        for agent_id in self.agents:
            for attacker_id in self.src:
                # Agent is damaged if:
                # 1. Agent is attacker and not defending
                # 2. Agent is on path between attacker and victim
                is_attacker_not_defending = (
                    agent_id == attacker_id and
                    (coalition is None or attacker_id not in coalition)
                )
                is_on_attack_path = (
                    self.check_path_exists(attacker_id, agent_id) and
                    self.check_path_exists(agent_id, self.dst)
                )

                if is_attacker_not_defending or is_on_attack_path:
                    total_damage += self.bandwidth

        return total_damage

    def social_gain(self):
        """
        Calculate the social gain from defense.

        Social gain is the reduction in network damage achieved by the coalition.

        Returns:
            tuple: (defense_success, damage_reduction)
                - defense_success: bool, whether defense blocked all attack paths
                - damage_reduction: float, bandwidth damage prevented by defense
        """
        # Calculate damage without defense
        damage_before_defense = self.calculate_network_damage(coalition=None)

        # Execute defense and check success
        defense_successful = self.execute_defense()

        # Calculate damage after defense
        damage_after_defense = self.calculate_network_damage(coalition=self.coalition)

        # Social gain is the reduction in damage
        damage_reduction = damage_before_defense - damage_after_defense

        return (defense_successful, damage_reduction)
