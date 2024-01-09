# cedric_simulation

## Installation
cd ddos_gym

pip install -e .

## Data
In ddos_gym/envs/data, you should provide 2 files: attack.csv and link.txt.

attack.csv should provide the DDoS attack events, with the format as current example.

link.txt should provide the Internet topology, with the format as current example.

## Parameters
- Environment Param

  cost for each country: cost_value in ddos_gym/envs/defense.py

  account_limit: maximum credits in each agent's account

  mode: different credit allocation mode

- Algorithm Param

  Q-learning:

  alpha: weight factor in Q-value estimation updating

  discount factor: discount factor

  epsilon: exploration probability, decaying from max_epsilon to min_episilon by decay rate

  max_rounds: number of interaction rounds per episode
