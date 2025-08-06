"""
Evaluate trained PPO models from Stable Baselines 3 (score‑based evaluation)
"""

import argparse
import os
import glob
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime

import gymnasium as gym
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import WarpFrame
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack, VecTransposeImage
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

from stable_baselines3.common.policies import ActorCriticCnnPolicyImproved
from stable_baselines3.common.policies import ActorCriticCnnPolicyImproved_model4


from components import ScoreRewardEnv
from components import StochasticFrameSkip
from components import SoftRewardEnv

import retro

import pdb 


def make_retro(*, game, state=None, max_episode_steps=20000, **kwargs):
    state = state or retro.State.DEFAULT
    env = retro.make(game, state, **kwargs)
    env = StochasticFrameSkip(env, n=4, stickprob=0.25, SEED=50)
    if max_episode_steps is not None:
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
    return env

def wrap_deepmind_retro(env):
    env = WarpFrame(env)
    env = ScoreRewardEnv(env)
    return env

def extract_steps(filename):
    name = os.path.basename(filename)
    steps_str = name.replace("ppo_checkpoint_", "").replace("_steps.zip", "")
    return int(steps_str)

def find_model_files(log_dir):
    model_files = {}
    final_model_path = os.path.join(log_dir, "final_model.zip")
    if os.path.exists(final_model_path):
        model_files["final"] = final_model_path
    best_model_path = os.path.join(log_dir, "best_model", "best_model.zip")
    if os.path.exists(best_model_path):
        model_files["best"] = best_model_path
    checkpoint_dir = os.path.join(log_dir, "checkpoints")
    if os.path.exists(checkpoint_dir):
        for f in sorted(glob.glob(os.path.join(checkpoint_dir, "*.zip")), key=extract_steps):
            name = os.path.basename(f)
            if "ppo_checkpoint_" in name:
                timesteps = name.replace("ppo_checkpoint_", "").replace("_steps.zip", "")
                model_files[f"checkpoint_{timesteps}"] = f
    return model_files

def evaluate_model(model_path, env, n_eval_episodes=10, render=False):
    print(f"\nEvaluating model: {model_path}")
    try:
        model = PPO.load(model_path, custom_objects={"policy_class": ActorCriticCnnPolicyImproved})
        mean_reward, std_reward = evaluate_policy(
            model, env,
            n_eval_episodes=n_eval_episodes,
            render=render,
            deterministic=True
        )
        print(f"Mean score: {mean_reward:.2f} ± {std_reward:.2f}")
        return mean_reward, std_reward
    except Exception as e:
        print(f"Error when evaluating the model: {e}")
        return None, None

def create_evaluation_report(results, output_dir):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(output_dir, f"evaluation_report_{ts}.txt")
    with open(path, "w") as f:
        f.write("Evaluation report\n")
        f.write(f"Date {datetime.now()}\n\n")
        for name, (mean, std) in results.items():
            if mean is None:
                f.write(f"{name}: error\n")
            else:
                f.write(f"{name}: {mean:.2f} ± {std:.2f}\n")
        valid = {k: v for k, v in results.items() if v[0] is not None}
        if valid:
            best = max(valid.items(), key=lambda x: x[1][0])
            f.write(f"\nBest model {best[0]} with score {best[1][0]:.2f}\n")
    print(f"Report saved in {path}")
    return path

def plot_evaluation_results(results, output_dir):

        
    plt.rcParams.update({
        "text.usetex": False,           
        "font.family": "serif",         
        "mathtext.fontset": "cm",       
        "font.size": 12,
    })

    valid = {k: v for k, v in results.items() if v[0] is not None}
    
    #breakpoint()

    if not valid:
        print("No valid results to plot")
        return
    
    names = list(valid.keys())

    names = [int(item.split("_")[1]) for item in names[1:]]

    means = [v[0] for v in valid.values()]
    means = means[1:]

    stds  = [v[1] for v in valid.values()]
    stds  = stds[1:]

    plt.figure(figsize=(12, 6))
    #plt.errorbar(np.array(names)/100000, means, yerr=stds, fmt="o-", capsize=5, capthick=2)

    plt.plot(np.array(names)/100000, means, "o-", color="C0", label="Mean score")

    plt.errorbar(
        np.array(names)/100000,
        means,
        yerr=stds,
        fmt="none",         
        ecolor="C0",           
        elinewidth=2,
        capsize=5,
        capthick=1.5,
        alpha=0.3                
    )

    ax = plt.gca()
    ax.ticklabel_format(style='plain', axis='x')  

    
    #plt.xticks(range(len(names)), names)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.xlabel(r"Steps [$\times 100000$]", fontsize=14)
    plt.ylabel(r"Mean score", fontsize=14)

    plt.grid(alpha=0.3)
    plt.tight_layout()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(output_dir, f"evaluation_plot_{ts}.pdf")
    plt.savefig(path, dpi=300, bbox_inches="tight")

    print(f"Plot saved in: {path}")

def save_evaluation_results(results, output_dir):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    np.save(os.path.join(output_dir, f"results_{ts}.npy"), results)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", required=True)
    parser.add_argument("--game",    default="Riverraid-Atari2600")
    parser.add_argument("--state",   default=None)
    parser.add_argument("--scenario", default=None)
    parser.add_argument("--n_eval_episodes", type=int, default=10)
    parser.add_argument("--render", action="store_true")

    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    args.output_dir = args.output_dir or os.path.join(args.log_dir, "evaluation")
    os.makedirs(args.output_dir, exist_ok=True)

    state = args.state or retro.State.DEFAULT

    def make_env():
        env = make_retro(game=args.game, state=state, scenario=args.scenario)
        env = wrap_deepmind_retro(env)
        env = Monitor(env)
        return env

    eval_env = VecTransposeImage(VecFrameStack(SubprocVecEnv([make_env]), n_stack=4))

    model_files = find_model_files(args.log_dir)
    if not model_files:
        print(f"No models were found in {args.log_dir}")
        return

    print("Found models")
    for name, path in model_files.items():
        print(f"  {name}: {path}")

    results = {}
    for name, path in model_files.items():
        mean, std = evaluate_model(path, eval_env, args.n_eval_episodes, args.render)
        results[name] = (mean, std)

    create_evaluation_report(results, args.output_dir)
    plot_evaluation_results(results,  args.output_dir)
    save_evaluation_results(results,  args.output_dir)


    eval_env.close()
    print("\nASSESSMENT COMPLETED")
    print(f"Results {args.output_dir}")


if __name__ == "__main__":
    main()
