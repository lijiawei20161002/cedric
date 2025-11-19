# CEDRIC Quick Reference Guide

## What Changed - At a Glance

### Variable Naming
| Old Name | New Name | Meaning |
|----------|----------|---------|
| `state_n` | `agent_states` | State (credits) for each agent |
| `action_n` | `agent_actions` | Cooperation decision for each agent |
| `invest_n` | `agent_investments` | Investment amount for each agent |
| `time` | `current_time_step` | Current simulation timestep |
| `ddos` | `ddos_events` | List of attack events |
| `account` | `agent_credits` | Credit balance for each agent |
| `rm` | `links_to_remove` | Links to remove from network |
| `frm` | `source_id` | Source node ID |
| `cnt` | `hop_count` | Number of hops in path search |
| `iso` | `agent_singleton` | Single-agent set |
| `g1`, `g2` | `gain_without_agent`, `gain_with_agent` | Social gains for Shapley value |

### File Structure
```
cedric/
├── REFACTORING_SUMMARY.md          (this explains everything)
├── QUICK_REFERENCE.md              (quick lookup)
└── large_simulation/
    ├── strategy_refactored.py      (5 strategy classes, much clearer)
    ├── main_refactored.py          (DQN with better docs)
    └── ddos_gym/ddos_gym/envs/
        ├── ddos_refactored.py      (environment with clear CEDRIC logic)
        └── defense_refactored.py   (network topology with clear path-finding)
```

## Key Improvements

### 1. Strategy.py → Clear Strategy Classes

**Before:** All strategies mixed together in one confusing loop
**After:** Each strategy is its own class

```python
# Now you can easily understand each strategy:
strategies = [
    QLearningStrategy(env, graph),      # Learns optimal policy
    TitForTatStrategy(env, graph),      # Mirrors victim's action
    AccumulateCreditsStrategy(env, graph),  # Always cooperates
    DetectiveStrategy(env, graph),      # Exploits unless punished
    GrudgerStrategy(env, graph)         # Cooperates until betrayed
]
```

### 2. Defense.py → Clear Network Operations

**Before:** Cryptic path-finding with single-letter variables
**After:** Clear method names and documentation

```python
# Clear what each method does:
defense.apply_defense(defender_id)           # Block defender's links
defense.check_path_exists(source, dest)      # Check connectivity
defense.execute_defense()                    # Run full defense
defense.calculate_network_damage()           # Calculate damage
defense.social_gain()                        # Calculate benefit
```

### 3. DDoS.py → CEDRIC Algorithm Explained

**Before:** Mysterious `_calculate_cedric_credit` with `iso`, `g1`, `g2`
**After:** Fully documented Shapley value calculation

```python
# Now clearly documented:
def _calculate_cedric_credits(self, ...):
    """
    Calculate CEDRIC credits using Monte Carlo Shapley value estimation.

    CEDRIC (Credit-based Defense Incentive Coordination) uses Shapley values
    to fairly distribute credits based on each agent's marginal contribution
    to the defense coalition.

    The Shapley value is estimated by:
    1. Sampling random subsets of the coalition
    2. Comparing defense outcomes with/without the agent
    3. Crediting proportional to marginal contribution
    """
```

### 4. Main.py → Complete DQN Implementation

**Before:** Incomplete training loop
**After:** Full DQN with progress tracking and model saving

```python
# Now has complete training:
def train_dqn_agent():
    """Train a DQN agent on the DDoS defense environment."""
    # Initialize environment and agent
    # Run training episodes
    # Track and report progress
    # Save trained models
```

## Quick Start

### Run Refactored Code

```bash
cd cedric/large_simulation

# Run strategy comparison (much clearer output)
python strategy_refactored.py

# Run DQN training (with progress tracking)
python main_refactored.py
```

### Use Refactored Classes

```python
import gym
from ddos_gym.envs.defense_refactored import Defense
from strategy_refactored import QLearningStrategy

# Create environment
env = gym.make('ddos-v0', mode='cedric')
graph = Defense()

# Create and run a strategy
strategy = QLearningStrategy(env, graph)
rewards = run_strategy(strategy, num_episodes=100, max_rounds=10)
```

## What Stayed The Same

✅ All functionality is identical
✅ Same algorithms (Q-learning, CEDRIC, etc.)
✅ Same inputs and outputs
✅ Same network topology
✅ Same reward calculations
✅ Same random seed = same results

## Configuration Constants

All magic numbers moved to top of files:

```python
# strategy_refactored.py
ALPHA = 0.7                    # Q-learning learning rate
DISCOUNT_FACTOR = 0.618        # Future reward discount
TRAIN_EPISODES = 1500          # Number of training episodes
MAX_ROUNDS_PER_EPISODE = 10    # Rounds per episode
ACCOUNT_LIMIT = 4              # Max credits per agent

# main_refactored.py
MEMORY_SIZE = 2000             # Experience replay buffer
GAMMA = 0.95                   # DQN discount factor
EPSILON_DECAY = 0.995          # Exploration decay rate
BATCH_SIZE = 32                # Minibatch size
```

## Understanding CEDRIC

**CEDRIC = Credit-based Defense Incentive Coordination**

Uses **Shapley values** to fairly reward agents:

1. Agent contributes to defense coalition
2. Calculate their marginal contribution (how much they helped)
3. Reward proportional to contribution
4. Prevents free-riding, encourages strategic cooperation

**Monte Carlo Estimation:**
- Sample random subsets of coalition
- Compare defense with/without the agent
- Average over multiple samples
- Approximates true Shapley value

## Strategy Comparison

| Strategy | Behavior | Best When |
|----------|----------|-----------|
| **Q-Learning** | Learns optimal policy through trial and error | Long-term optimization |
| **Tit-for-Tat** | Mirrors victim's previous action | Others are reciprocal |
| **Accumulate Credits** | Always cooperates | Building up resources |
| **Detective** | Exploits unless punished | Others are naive |
| **Grudger** | Cooperates until betrayed once | Zero tolerance for cheating |

## Common Tasks

### Add a New Strategy

```python
class MyStrategy(DefenseStrategy):
    def __init__(self, env, graph):
        super().__init__(env, graph, "My Strategy")
        # Initialize your strategy

    def select_action(self, agent_id, agent_states, episode, step):
        # Implement your logic here
        return action  # 0 or 1
```

### Modify Network Topology

Edit `ddos_gym/ddos_gym/envs/data/link.txt`:
```
US,JP
US,CN
CN,JP
...
```

### Modify Attack Events

Edit `ddos_gym/ddos_gym/envs/data/attack.csv`:
```
victim,source,timestamp,bandwidth
['JP'],['CN','US'],0,100.5
...
```

### Change Credit Mode

```python
# CEDRIC mode (Shapley values)
env = gym.make('ddos-v0', mode='cedric')

# Shared mode (equal distribution)
env = gym.make('ddos-v0', mode='shared')
```

## Documentation Locations

- **Full explanation:** `REFACTORING_SUMMARY.md`
- **Quick lookup:** This file (`QUICK_REFERENCE.md`)
- **Code docstrings:** In every refactored file
- **Inline comments:** For complex logic

## Next Steps

1. Read `REFACTORING_SUMMARY.md` for detailed explanations
2. Try running `strategy_refactored.py` to see clearer output
3. Look at the strategy classes to understand each approach
4. Experiment with different configurations
5. Consider applying same refactoring to `small_simulation/`

## Help

Each refactored file has comprehensive docstrings:

```python
# Get help on any class or method
help(QLearningStrategy)
help(Defense.check_path_exists)
help(DDoS.step)
```

The refactored code is self-documenting with clear variable names, method names, and extensive comments.
