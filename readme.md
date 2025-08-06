
# Learning to play River Raid™ 

This project, part of the AE4350: Bio-inspired Intelligence and Learning for Aerospace Applications course, focuses on training a Reinforcement Learning (RL) agent to play the classic 1982 video game, River Raid.


## Environment Setup

This project is designed to run **on Linux**, due to dependencies such as `stable-retro`, which are incompatible with Windows or macOS. This project has been implemented using the Python 3.12.x version. This project and its dependencies require a **64-bit** system.  

Using a virtual environment (`venv`) is strongly recommended. Do not use `conda` for this project, as some packages included in this repository are known to break in conda environments. 

---

### 0. Create a work directory

Open a terminal to create a work directory, (e.g. on the desktop). This folder will contain the python environment, and the Riverraid directory.

```bash
mkdir RiverraidMain
```

Enter that directory: 

```bash
cd RiverraidMain
```


### 1. Clone the Repository

Clone this repository to your work directory:

```bash
git clone https://github.com/ahsawa/RiverraidPPO.git
```

### 2. Create a Virtual Environment 

To keep dependencies isolated, it’s strongly recommended to create a Python virtual environment.  

Before doing this, make sure the `venv` module is available. If not, install it with:

```bash
sudo apt install python3.12-venv
```

Now: 

```bash
python3.12 -m venv .venv
```

Activate this environment: 

```bash
source .venv/bin/activate
```

### 3. Install requirements

To install the required packages, first, enter the cloned directory.

```bash

cd RiverraidPPO
```

Then, execute the following to automatically install the first collection of required packages: 

```bash
pip install --upgrade pip  # (Optional) Upgrade pip first
pip install -r requirements.txt
```

### 4. Install stable-retro

This project uses a custom version of: `stable-retro`, therefore, it should be installed in an editable mode, and after that, install the modified files.

First, make sure that you are in the RiverraidPPO folder (you should be there already)

After that: 

```bash
git clone https://github.com/Farama-Foundation/stable-retro.git
cd stable-retro
pip3 install -e .
```

At that point the editable version of the official version of stable-retro should be installed.

To install the custom files, first move back to the RiverraidPPO folder: 

```bash
cd ..
```

Then, delete the off-the-shelf Riverraid folder:  

```bash
rm -rf stable-retro/retro/data/stable/Riverraid-Atari2600/
```

And replace it with: 

```bash
cp -r Riverraid-Atari2600 stable-retro/retro/data/stable/
```



### 5. Install stable-baselines3 

Now, install the custom version of the stable-baselines3: 

```bash
pip install -e ./stable-baselines3
```

### 6. (Optional) Verify the installation:


```bash
pip list | grep -E "stable-retro|stable-baselines3"
```

## Development of the models: 

To launch a training session using Proximal Policy Optimization (PPO), run the mainstable.py script from within the RiverraidPPO folder. This script initializes the environment, sets up vectorized parallel environments, and manages logging, curriculum learning, and model saving. 

#### Command-line arguments for mainstable.py

You can customize the training process via command-line arguments, summarized in the following table: 

| Argument              | Default               | Description                                                                 |
|-----------------------|------------------------|-----------------------------------------------------------------------------|
| `--game`             | `Riverraid-Atari2600` | Name of the game environment (do not modify unless using another ROM).     |
| `--state`            | `None`                | Initial game state. If `None`, uses the default state.                     |
| `--scenario`         | `scenario`            | Reward system ("scenario" for a advanced reward system, and "scenario2" for the basic reward system).|
| `--reward_select`    | `soft`                | Type of reward clipping: `soft` (tanh) or `clip` (hard).              |
| `--curriculum_select`| `0`                   | Enables (`1`) or disables (`0`) curriculum learning.                        |
| `--total_timesteps`  | `5000000`             | Number of timesteps to train the agent.                                    |
| `--log_dir`          | `logs`                | Directory where logs and model checkpoints will be saved.                  |
| `--save_freq`        | `3125`                | Frequency (steps) for saving model checkpoints (multiplied by the number of environments, by default 16; therefore 50000 steps).                     |
| `--seed`             | `10`                  | Random seed for reproducibility.                                           |
| `--lr`               | `2.5e-4`              | Base learning rate for PPO (scaled during training).                       |
| `--clip`             | `0.15`                | PPO clipping parameter for the objective function.                         |

Note: The policy architecture used in training is fixed in the script as "CnnPolicyImproved", which is a lightweight, customized convolutional neural network designed for this project, discussed in the report. The original Stable-Baselines3 policy, "CnnPolicy", can also be used but requires manually modifying the script at the model instantiation:

Example (execute from the RiverraidPPO folder, with the custom environment): 

```bash
python mainstable.py --reward_select soft --curriculum_select 1 --total_timesteps 10000000
```

#### Evaluating Trained Models with mainevaluate.py

The mainevaluate.py script is used to evaluate trained PPO models saved during training. It automatically finds all saved checkpoints, including the final and best models, inside a given experiment’s logs directory. For each model, it runs multiple evaluation episodes (default 10) in a parallelized environment and calculates the mean score and standard deviation. The script generates a detailed text report, a plot of the scores over training steps, and saves raw evaluation results for further analysis. To run it, provide the path to the logs folder using the --log_dir argument.

| Argument            | Default                | Description                                                 |
| ------------------- | ---------------------- | ----------------------------------------------------------- |
| `--log_dir`         | **(required)**         | Path to the directory containing logs and trained models.   |
| `--game`            | `Riverraid-Atari2600`  | Name of the game environment to evaluate.                   |
| `--state`           | `None`                 | Initial state of the game (uses default if not specified).  |
| `--n_eval_episodes` | `10`                   | Number of episodes to evaluate each model.                  |

The script `run_evaluate.sh` is used to run `mainevaluate.py` over all the cases inside a directory. 


#### Generating performance metrics with mainstats.py


After running the training experiments with mainstable.py, it is recommended to run mainstats.py. This script processes each experiment folder inside the logs directory, reads the evaluation .npy files, and calculates statistics such as mean scores, standard deviations, efficiency (mean score per timestep), and a normalized area under the curve (AUC) to measure robustness. It then saves these summary statistics as text files within each evaluation folder. These statistics are used to evaluate each case in this project. 

#### Plotting cases into single PDF

To create a plot from the test cases, run the script mainplotting.py. This script gathers evaluation results from all experiment runs stored in the `logs` folder and combines them into a single plot. It displays the mean values with error bars for each case. The final combined plot is saved as a PDF file in the `output` directory. To visually review your training results, simply run this script and check the generated plots inside the `output` folder.

### Utility Scripts (utils/)

This folder contains additional helper scripts that support training and evaluation. Two of the most relevant ones are:


`create_video.py`

This script generates a video of a trained agent playing the game. It loads a selected model checkpoint and records a gameplay session as a video file. This is especially useful for visualizing and presenting the agent’s behavior after training.

`analyze_policy.py`

This tool helps you inspect the neural network architecture used by a trained policy. It prints out the model layers, number of parameters, and structure, which is helpful for debugging or understanding how your custom policy (e.g. CnnPolicyImproved) is built.

## Results

Below is a sample video (evaluation_video.mp4) that helps visualize and analyze the agent’s performance during evaluation. The next video shown corresponds to the model discussed in the report and represents its final behavior after training.

https://github.com/user-attachments/assets/de0b688c-3dee-4271-82ce-90f92032a3e5

### Other logs

RiverraidPPO contains three main folders that group results from different experiment batches:

#### logs_10M/

Contains the results of the initial exploration phase, where each model was trained for 10 million timesteps. These experiments were launched automatically using the `run_10M.sh` script to avoid running them manually.

#### logs_params/

Stores results from experiments focused on testing different network architectures.

#### logs_hyper/

Includes results related to hyperparameter tuning and their effect on training performance.

Note: Not all .zip model files are included here to keep the repository size low (the full size would exceed 80 GB). Only the final checkpoint, however, as discussed in the report this is not necessarily the best model, since some experiments saved their best checkpoint earlier.

## References

*   J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, *Proximal Policy Optimization Algorithms*, 2017. doi: 10.48550/ARXIV.1707.06347. [Online]. Available: https://arxiv.org/abs/1707.06347.
*   M. Poliquin, *Stable Retro, a maintained fork of OpenAI’s gym-retro*, GitHub repository, 2025. [Online]. Available: https://github.com/Farama-Foundation/stable-retro.
*   A. Raffin, A. Hill, A. Gleave, A. Kanervisto, M. Ernestus, and N. Dormann, “Stable-Baselines3: Reliable Reinforcement Learning Implementations,” *Journal of Machine Learning Research*, vol. 22, no. 268, pp. 1–8, 2021. [Online]. Available: http://jmlr.org/papers/v22/20-1364.html.
*   V. Mnih, K. Kavukcuoglu, D. Silver, et al., “Human-level control through deep reinforcement learning,” *Nature*, vol. 518, no. 7540, pp. 529–533, Feb. 2015. doi: 10.1038/nature14236. [Online]. Available: https://www.nature.com/articles/nature14236.


