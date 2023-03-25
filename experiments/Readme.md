## Usage:

### Training
```

python train_agent.py --config-file 5bunkers_2presses.json --budget 2000000 --n-steps 6144 --RL-agent DQN --n-seeds 9

```

```

python train_agent.py --config-file 5bunkers_2presses.json --budget 2000000 --n-steps 6144 --RL-agent PPO --n-seeds 5

```

```

python train_agent.py --config-file 5bunkers_2presses.json --budget 2000000 --n-steps 6144 --RL-agent TRPO --n-seeds 9

```
```

python train_agent.py --config-file 5bunkers_2presses.json --budget 2000000 --n-steps 6144 --RL-agent A2C --n-seeds 5

```

### Evaluation

```

python evaluate_agent.py --config-file 5bunkers_2presses.json --budget 2000000 --n-steps 6144 --RL-agent DQN --n-seeds 9 --inf-eplen 600

```
```

python evaluate_agent.py --config-file 5bunkers_2presses.json --budget 2000000 --n-steps 6144 --RL-agent PPO --n-seeds 5 --inf-eplen 600

```

```

python evaluate_agent.py --config-file 5bunkers_2presses.json --budget 2000000 --n-steps 6144 --RL-agent TRPO --n-seeds 9 --inf-eplen 600

```
```

python evaluate_agent.py --config-file 5bunkers_2presses.json --budget 2000000 --n-steps 6144 --RL-agent A2C --n-seeds 5 --inf-eplen 600

```

### Plot reward curves (training) for an Agent for each seed in different plots

```

python plot_smoothened_reward.py --config-file 5bunkers_2presses.json --budget 2000000 --n-steps 6144 --RL-agent DQN --n-seeds 9

```

### Plot reward curve (training) for an Agent for all seeds in a single plot

```

python plot_smoothened_reward_all_seeds.py --config-file 5bunkers_2presses.json --budget 2000000 --n-steps 6144 --RL-agent DQN --n-seeds 9

```

