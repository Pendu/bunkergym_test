import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
import argparse
from multiprocessing import Process
from distutils.util import strtobool
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import torch
from stable_baselines3 import PPO, DQN, A2C
from sb3_contrib import TRPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common import results_plotter
from stable_baselines3.common.results_plotter import load_results, ts2xy
from env import SutcoEnv
from gym.wrappers import FlattenObservation
import pandas as pd
os.environ['CUDA_VISIBLE_DEVICES'] = ''
torch.set_num_threads(1)


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


def plot_reward(seed, args):
    """
    Evaluates a trained agent and creates plots of its performance.

    Attributes
    ----------
    seed : int
        Seed used for the experiment
    args : argparse.Namespace
        Arguments passed to the script
    """

    overwrite_episode_length = 1200
    deterministic_policy = True
    save_fig = True
    ent_coef = 0.
    gamma = 0.99
    n_steps = args.n_steps
    seed = seed

    config_file = args.config_file

    fig_format = 'svg'  # png'

    run_name = f"{args.RL_agent}_{args.config_file.replace('.json', '')}_seed_{seed}_budget_{args.budget}_n_steps_{args.n_steps}"

    fig_name = 'run_trained_agent_deterministic_policy_' + run_name

    log_dir = os.path.dirname(os.path.abspath(__file__)) + '/logs/' + run_name + '/'

    budget = int(run_name.split("budget_")[-1].split("_")[0])  # Parse budget from run name

    results_plotter.plot_results([log_dir], budget, results_plotter.X_TIMESTEPS, f"{args.RL_agent} Sutco")

    def moving_average(values, window):
        """
        Smooth values by doing a moving average
        :param values: (numpy array)
        :param window: (int)
        :return: (numpy array)
        """
        weights = np.repeat(1.0, window) / window
        return np.convolve(values, weights, 'valid')

    def plot_results(log_folder, title='Learning Curve'):
        """
        plot the results

        :param log_folder: (str) the save location of the results to plot
        :param title: (str) the title of the task to plot
        """
        x, y = ts2xy(load_results(log_folder), 'timesteps')
        y = moving_average(y, window=50)
        # Truncate x
        x = x[len(x) - len(y):]
        dataframe = pd.DataFrame(data={'x': x, 'y': y, 'seed': [seed] * len(x)})
        dataframe.to_csv(log_dir + 'train_reward_' + run_name + '.csv', index=False)
        fig = plt.figure(title)
        plt.plot(x, y)
        plt.xlabel('Number of Timesteps')
        plt.ylabel('Rewards')
        plt.title(title + " Smoothed")
        plt.show()

    plot_results(log_dir)


    log_dir_results = os.path.dirname(os.path.abspath(__file__)) + '/results/'
    os.makedirs(log_dir_results, exist_ok=True)

    plt.savefig(log_dir_results + 'train_reward_' + run_name + '.%s' % fig_format)




if __name__ == "__main__":

    # get the arguments
    args = parse_args()

    seeds = range(1, args.n_seeds)
    processes = [Process(target=plot_reward, args=(s, args)) for s in seeds]

    for p in processes:
        p.start()

    for p in processes:
        p.join()
