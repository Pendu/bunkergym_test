import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))

from datetime import datetime
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common import results_plotter
from multiprocessing import Process
from callbacks import *
from env import SutcoEnv
from gym.wrappers import FlattenObservation
import argparse

os.environ['CUDA_VISIBLE_DEVICES'] = ''
torch.set_num_threads(1)

def parse_args():
    parser = argparse.ArgumentParser()

    # Experiment specific arguments
    parser.add_argument("--config-file", type=str, default='1bunker_1press.json',
                        help="The name of the config file for the env")
    parser.add_argument("--budget", type=int, default=10000,
                        help="total number of timesteps of the experiments")
    
    args = parser.parse_args()

    return args


#list_constants_example.json
#list_constants_example_2.json
#1bunker_1press.json

def train(seed, args):
    
    
    """Trains an agent on the benchmark environment.

    Parameters
    ----------
    seed : number
        seed with which to initialize the random number generator
    config_file : str, optional
        name of the config file with which to create the environment, by default '1bunker_1press.json'
    budget : int, optional
        number of training steps, by default 10000000
    ent_coef : float, optional
        ppo entropy coefficient for loss calculation, by default 0.
    gamma : float, optional
        ppo discount factor, by default 0.99
    n_steps : int, optional
        ppo number of steps per policy update, by default 2048
    """
    
    #config_file='list_constants_example_2.json'
    config_file = args.config_file
    budget= args.budget
    ent_coef=0.
    gamma=0.99
    n_steps=6144
    #n_steps=2048
    seed = seed 
    
    name = 'ppo_' + config_file.replace(".json", "") \
           +  '_seed_' + str(seed) + '_budget_' + str(budget) + '_ent-coef_' \
           + str(ent_coef) + '_gamma_' + str(gamma) + "_steps_" + str(n_steps)

    log_dir = os.path.dirname(os.path.abspath(__file__)) + '/logs/' + name + '/'
    os.makedirs(log_dir, exist_ok=True)

    # Create and wrap the environment
    env = SutcoEnv.from_json(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../configs/" + config_file))
    env = FlattenObservation(env)
    env = Monitor(env, log_dir)
    
    

    # Create the callback
    auto_save_callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)

    #model = PPO("MultiInputPolicy", env, seed=seed, verbose=0, ent_coef=ent_coef, gamma=gamma, n_steps=n_steps)
    model = PPO("MlpPolicy", env, seed=seed, verbose=0, ent_coef=ent_coef, gamma=gamma, n_steps=n_steps)


    # To use Tensorboard, see example below.
    # model = PPO("MultiInputPolicy", env, seed=seed, verbose=0, ent_coef=ent_coef, gamma=gamma, n_steps=n_steps,
    #             tensorboard_log="./ppo_sutco_tensorboard/")

    start = datetime.now()

    model.learn(total_timesteps=budget, callback=auto_save_callback, tb_log_name="seed_" + str(seed))

    print("Total training time: ", datetime.now() - start)

    # Plot various results
    #
    # Training reward
    results_plotter.plot_results([log_dir], budget, results_plotter.X_TIMESTEPS, "PPO Sutco")
    #
    # Average episodic reward of best policy
    del model
    model = PPO.load(log_dir + "best_model.zip", env=env)
    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10, deterministic=True)
    print("Average episodic reward: {:} \u00B1 {:}".format(mean_reward, std_reward))

    # Run trained agent and plot state variables
    # run_trained_agent(log_dir, fig_name='run_trained_agent_' + name, save_fig=False,
    #                   max_episode_length=max_episode_length)

    
    
if __name__ == "__main__":
    
    # get the arguments
    args = parse_args()
    
    seeds = range(1, 4)
    processes = [Process(target=train, args=(s,args)) for s in seeds]

    for p in processes:
        p.start()

    for p in processes:
        p.join()