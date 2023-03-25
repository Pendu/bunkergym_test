import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib.patches as mpatches
import numpy as np
import argparse
import os
import sys
#sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))

def parse_args():
    parser = argparse.ArgumentParser()

    # Experiment specific arguments
    parser.add_argument("--config-file", type=str, default='1bunker_1press.json',
                        help="The name of the config file for the env")
    parser.add_argument("--render-episode", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                        help="if toggled, render the episodes during evaluation")
    parser.add_argument("--budget", type=int, default=10000,
                        help="total number of timesteps of the experiments")
    parser.add_argument("--inf-eplen", type=int, default=600,
                        help="total number of timesteps of the experiments")
    parser.add_argument("--n-steps", type=int, default=2048,
                        help="total number of timesteps of the experiments")
    parser.add_argument("--n-seeds", type=int, default=4,
                        help="total number of seeds to run the experiment")
    parser.add_argument("--RL-agent", type=str, default='PPO',
                        help="The name of the agent to train the env")
    args = parser.parse_args()

    return args

dataframe_list = []
def utility_function(seed, args):


    run_name = f"{args.RL_agent}_{args.config_file.replace('.json', '')}_seed_{seed}_budget_{args.budget}_n_steps_{args.n_steps}"

    fig_name = 'run_trained_agent_deterministic_policy_' + run_name

    log_dir = os.path.dirname(os.path.abspath(__file__)) + '/logs/' + run_name + '/'

    budget = int(run_name.split("budget_")[-1].split("_")[0])  # Parse budget from run name

    file_name = log_dir + 'train_reward_' + run_name + '.csv'

    dataframe = pd.read_csv(file_name)
    print(dataframe.head())



    return dataframe

def plot_graph(data, args):
    sns.set_style("whitegrid", {'axes.grid': True,
                                'axes.edgecolor': 'black'

                                })

    fig_format = 'svg'
    fig = plt.figure()
    plt.clf()
    ax = fig.gca()
    color_patch = []
    sns.lineplot(data=data, x="x", y="y", hue="seed")
    color_patch.append(mpatches.Patch(color="green", label=args.RL_agent))
    # print(min_len)
    plt.xlabel('Training timesteps', fontsize=14)
    plt.ylabel('Average return', fontsize=14)
    lgd = plt.legend(frameon=True, fancybox=True, prop={'weight': 'bold', 'size': 10}, handles=color_patch, loc="best")
    plt.title('Training reward', fontsize=14)

    plt.setp(ax.get_xticklabels(), fontsize=12)
    plt.setp(ax.get_yticklabels(), fontsize=12)
    sns.despine()
    plt.tight_layout()


    run_name = f"{args.RL_agent}_{args.config_file.replace('.json', '')}_allseeds__budget_{args.budget}_n_steps_{args.n_steps}"

    log_dir_results = os.path.dirname(os.path.abspath(__file__)) + '/results/'
    os.makedirs(log_dir_results, exist_ok=True)

    plt.savefig(log_dir_results + 'train_reward_' + run_name + '.%s' % fig_format)
    plt.show()


if __name__ == "__main__":

    # get the arguments
    args = parse_args()

    seeds = range(1, args.n_seeds)
    for seed in seeds:
        dataframe = utility_function(seed, args)
        dataframe_list.append(dataframe)
    # print(dataframe_list)
    append_dataframe_list = pd.concat(dataframe_list)
    run_name = f"{args.RL_agent}_{args.config_file.replace('.json', '')}_allseeds_budget_{args.budget}_n_steps_{args.n_steps}"

    log_dir_results = os.path.dirname(os.path.abspath(__file__)) + '/results/'
    os.makedirs(log_dir_results, exist_ok=True)

    file_name = log_dir_results + 'train_reward_' + run_name + '.csv'

    append_dataframe_list.to_csv(file_name, index=False)
    append_dataframe_list = pd.read_csv(file_name)
    plot_graph(append_dataframe_list, args)


