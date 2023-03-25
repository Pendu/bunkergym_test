import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common import results_plotter
from env import SutcoEnv
from gym.wrappers import FlattenObservation
import argparse
from multiprocessing import Process

os.environ['CUDA_VISIBLE_DEVICES'] = ''
torch.set_num_threads(1)

#list_constants_example.json
#list_constants_example_2.json
#1bunker_1press.json

def parse_args():
    parser = argparse.ArgumentParser()

    # Experiment specific arguments
    parser.add_argument("--config-file", type=str, default='1bunker_1press.json',
                        help="The name of the config file for the env")
    parser.add_argument("--budget", type=int, default=10000,
                        help="total number of timesteps of the experiments")
    
    args = parser.parse_args()

    return args
    
def inference(seed,args):
    """Evaluates a trained agent and creates plots of its performance.

    Parameters
    ----------
    log_dir : str
        log directory in which the agent was saved
    config_file : str, optional
        name of the config file used for the environment, by default '1bunker_1press.json'
    overwrite_episode_length : int, optional
        if a different episode length is desired in inference, set that length here
    deterministic_policy : bool, optional
        whether to sample from deterministic policy, by default True
    fig_name : str, optional
        figure name for the generated plots, by default ''
    format : str, optional
        file format of the generated plots, by default 'png'
    save_fig : bool, optional
        whether to save the figures to file, by default False
    """
    
    # log_dir, config_file='1bunker_2press.json'
    
    overwrite_episode_length=1200
    deterministic_policy=True
    #fig_name=''
    #format='png'
    save_fig=True
    budget= args.budget
    ent_coef=0.
    gamma=0.99
    #n_steps=6144
    n_steps=2048
    seed = seed 
   
    
    
    config_file = args.config_file
    
    fig_format = 'png'
    run_name = f"ppo_{args.config_file.replace('.json','')}_seed_{seed}_budget_{args.budget}_ent-coef_{ent_coef}_gamma_{gamma}_steps_{n_steps}"
    
    fig_name='run_trained_agent_deterministic_policy_' + run_name
    
    
    log_dir = os.path.dirname(os.path.abspath(__file__)) + '/logs/' + run_name + '/'
    print(log_dir)
    
    budget = int(run_name.split("budget_")[-1].split("_")[0]) # Parse budget from run name
    env_config = run_name.split('ppo_')[-1].split("_seed")[0] + ".json" # Parse config from run name

    results_plotter.plot_results([log_dir], budget, results_plotter.X_TIMESTEPS, "PPO Sutco")
    plt.savefig('train_reward_' + run_name + '.%s' % fig_format)


    env = SutcoEnv.from_json(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../configs/" + config_file))
    env = FlattenObservation(env)

    if overwrite_episode_length:
        env.max_episode_length = overwrite_episode_length
    env = Monitor(env)
    obs = env.reset()  
    model = PPO.load(log_dir + "best_model.zip")

    # Keep track of state variables (volumes), actions, and rewards
    volumes = []
    actions = []
    rewards = []

    step = 0

    episode_length = 0

    # Run episode
    while True:
        episode_length += 1
        action, _ = model.predict(obs, deterministic=deterministic_policy)
        actions.append(action)
        obs, reward, done, info = env.step(action)
        volumes.append(obs[env.n_presses:].copy())
        rewards.append(reward)
        # print('obs =', obs, 'reward =', reward, 'done =', done)
        if done:
            break
        step += 1

    # Plot state variables
    fig = plt.figure(figsize=(15, 10))
    env_unwrapped = env.unwrapped
    if deterministic_policy:  
        fig.suptitle("PPO on Sutco env. Trained agent, deterministic actions")
    else:
        fig.suptitle("PPO on Sutco env. Trained agent, non-deterministic actions")

    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313)

    ax1.set_title("Volume")
    ax2.set_title("Action")
    ax3.set_title("Reward")
    # ax3.set_title("Reward. Cumulative reward: {:.2f}".format(sum(rewards)))

    ax1.grid()
    ax2.grid()
    ax3.grid()

    ax1.set_ylim(top=40)
    ax2.set_yticks(list(range(env_unwrapped.action_space.n)))
    ax3.set_ylim(bottom=-0.1, top=1)

    plt.xlabel("Steps")

    default_color = "#1f77b4"  # Default Matplotlib blue color
    color_code = {"C1-10": "#872657",  # raspberry
                  "C1-20": "#0000FF",  # blue
                  "C1-30": "#FFA500",  # orange
                  "C1-40": "#008000",  # green
                  "C1-50": "#B0E0E6",  # powderblue
                  "C1-60": "#FF00FF",  # fuchsia
                  "C1-70": "#800080",  # purple
                  "C1-80": "#FF4500",  # orangered
                  "C2-10": "#DB7093",  # palevioletred
                  "C2-20": "#FF8C69",  # salmon1
                  "C2-40": "#27408B",  # royalblue4
                  "C2-50": "#54FF9F",  # seagreen1
                  "C2-60": "#FF3E96",  # violetred1
                  "C2-70": "#FFD700",  # gold1
                  "C2-80": "#7FFF00",  # chartreuse1
                  "C2-90": "#D2691E",  # chocolate
                  }
    line_width = 3

    # Plot volumes for each bunker
    for i in range(env_unwrapped.n_bunkers):
        ax1.plot(np.array(volumes)[:, i], linewidth=3, label=env_unwrapped.enabled_bunkers[i],
                 color=color_code[env_unwrapped.enabled_bunkers[i]])
    ax1.legend()

    # Plot actions
    x_axis = range(episode_length)
    for i in x_axis:
        if actions[i] == 0:  # Action: "do nothing"
            ax2.scatter(i, actions[i], linewidth=line_width, color=default_color)
        elif actions[i] in range(1, env_unwrapped.n_bunkers + 1):  # Action: "use Press 1"
            ax2.scatter(i,
                        actions[i],
                        linewidth=line_width,
                        color=color_code[env_unwrapped.enabled_bunkers[actions[i] - 1]],
                        marker="^")
            ax2.legend(handles=[mlines.Line2D([], [], color='black', marker='^', linestyle='None', label='Press 1')])
        elif actions[i] in range(env_unwrapped.n_bunkers + 1, env_unwrapped.n_bunkers * 2 + 1):  # Action: "use Press 2"
            ax2.scatter(i,
                        actions[i],
                        linewidth=line_width,
                        color=color_code[env_unwrapped.enabled_bunkers[actions[i] - env_unwrapped.n_bunkers - 1]],
                        marker="x")
            ax2.legend(handles=[mlines.Line2D([], [], color='black', marker='x', linestyle='None', label='Press 2')])
            
        else:
            print("Unrecognised action: ", actions[i])


    # Plot rewards
    ax3.scatter(range(episode_length), rewards, linewidth=line_width, color=default_color, clip_on=False)
    ax3.annotate("Cumul. reward: {:.2f}".format(sum(rewards)), xy=(0.9, 0.9), xycoords='axes fraction', fontsize=14)

    if save_fig:
        # Save plot
        plt.savefig(fig_name + '.%s' % fig_format, dpi='figure', format=fig_format)

    plt.show()


if __name__ == "__main__":
    
    # get the arguments
    args = parse_args()
        
    seeds = range(1, 4)
    #seeds = 13
    
    #inference(13,args)
    processes = [Process(target=inference, args=(s,args)) for s in seeds]

    for p in processes:
        p.start()

    for p in processes:
        p.join()
    
    
    
