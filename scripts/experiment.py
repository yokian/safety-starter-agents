#!/usr/bin/env python
import gym 
import safety_gym
import safe_rl
from safe_rl.utils.run_utils import setup_logger_kwargs
from safe_rl.utils.mpi_tools import mpi_fork, proc_id
import sys
sys.path.append('/home/jeappen/code/zikang/HiSaRL') # Adding zikang repo
from shrl.envs.point import PointNav as zikenv
# from shrl.envs.car import CarNav as zikenv

import wandb

def main(robot, task, algo, seed, exp_name=None, cpu=4):

    # Verify experiment
    robot_list = ['point', 'car', 'doggo']
    task_list = ['goal1', 'goal2', 'button1', 'button2', 'push1', 'push2']
    algo_list = ['ppo', 'ppo_lagrangian', 'trpo', 'trpo_lagrangian', 'cpo','sac']

    algo = algo.lower()
    task = task.capitalize()
    robot = robot.capitalize()
    assert algo in algo_list, "Invalid algo"
    assert task.lower() in task_list, "Invalid task"
    assert robot.lower() in robot_list, "Invalid robot"

    # Hyperparameters
    if exp_name is None:
        exp_name = algo + '_' + robot + task
    if robot=='Doggo':
        num_steps = 1e8
        steps_per_epoch = 60000
    else:
        num_steps = 1e7
        steps_per_epoch = 30000

    # Copied from run_polopt_agent
    config = {
    "ent_reg": 0.,
    
    "cost_lim":0,
    "penalty_init":1.,
    "penalty_lr":5e-2,
    "target_kl" : 0.01,
    "vf_lr":1e-3,
    "vf_iters":80, 
    }

    
    epochs = int(num_steps / steps_per_epoch)
    save_freq = 50

    config["epochs"] = epochs
    # wandb.config["cost_lim"] = cost_lim
    # wandb.config["target_kl"] = target_kl


    # Fork for parallelizing
    mpi_fork(cpu)

    if proc_id() == 0:
        # For using wandb with mpi
        wandb.init(project="hisarl-baselines", entity="csbric", config=config)
        config = wandb.config # For allowing hparam sweep?
        # TODO: see if we can sweep when using mpi

    # Prepare Logger
    exp_name = exp_name or (algo + '_' + robot.lower() + task.lower())
    logger_kwargs = setup_logger_kwargs(exp_name, seed)

    # Algo and Env
    algo = eval('safe_rl.'+algo)
    env_name = 'Safexp-'+robot+task+'-v0'

    algo(env_fn=lambda: zikenv(),
         ac_kwargs=dict(
             hidden_sizes=(256, 256),
            ),
        #  epochs=epochs,
         steps_per_epoch=steps_per_epoch,
         save_freq=save_freq,
        #  target_kl=target_kl,
        #  cost_lim=cost_lim,
         seed=seed,
         logger_kwargs=logger_kwargs,
         wandb=wandb,
         **config
         )



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--robot', type=str, default='Point')
    parser.add_argument('--task', type=str, default='Goal1')
    parser.add_argument('--algo', type=str, default='ppo')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--exp_name', type=str, default='')
    parser.add_argument('--cpu', type=int, default=1)
    args = parser.parse_args()
    exp_name = args.exp_name if not(args.exp_name=='') else None
    main(args.robot, args.task, args.algo, args.seed, exp_name, args.cpu)