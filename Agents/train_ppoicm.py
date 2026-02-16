# checkout https://github.com/geekyutao/PyTorch-PPO/blob/master/PPO_discrete.py
# checkout https://github.com/john-cardiff/-cyborg-cage-2
import os
import torch
import numpy as np
from CybORG import CybORG
from CybORG.Agents import RedMeanderAgent, B_lineAgent
from CybORG.Agents.Wrappers import ChallengeWrapper
import inspect
from PPOICMAgent import PPOICMAgent
from tqdm import tqdm
import random
import csv
import os


PATH = str(inspect.getfile(CybORG))
PATH = PATH[:-10] + '/Shared/Scenarios/Scenario2.yaml'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train(env, input_dims, action_space,
          max_episodes, max_timesteps, update_timestep, K_epochs, eps_clip,
          gamma, lr, betas, icm_beta, ckpt_folder, print_interval=10, save_interval=100):

    agent = PPOICMAgent(
        input_dims,
        action_space,
        gamma,
        lr,
        eps_clip,
        K_epochs,
        betas,

    )

    # ADDED: CSV reward logging setup 
    rewards_dir = os.path.join(os.getcwd(), "rewards")
    os.makedirs(rewards_dir, exist_ok=True)

    log_path = os.path.join(rewards_dir, "episode_rewards.csv")
    if not os.path.exists(log_path):
        with open(log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["episode", "reward_sum"])


    red_agents = [B_lineAgent, RedMeanderAgent]
    running_reward, time_step = 0, 0

    for i_episode in tqdm(range(1, max_episodes + 1), desc="Training"):
        red_agent = random.choice(red_agents)
        cyborg = CybORG(PATH, 'sim', agents={'Red': red_agent})
        env = ChallengeWrapper(env=cyborg, agent_name="Blue")

        state = env.reset()

        # ADDED: accumulate per-episode reward 
        ep_reward = 0.0

        for t in range(max_timesteps):
            time_step += 1
            action = agent.get_action(state) # adds action to memory as well
            next_state, reward, done, _ = env.step(action)

            # ADDED: accumulate reward 
            ep_reward += reward

            # Compute intrinsic reward and shape the reward
            intrinsic_reward = agent.compute_intrinsic_reward(state, next_state, action)
            shaped_reward = reward + agent.icm_beta * intrinsic_reward

            agent.store(next_state, shaped_reward, done) # stores next_state, shaped_reward, done in memory
            state = next_state

            if time_step % update_timestep == 0:
                agent.train()
                agent.clear_memory()
                time_step = 0

            running_reward += reward

        agent.end_episode()

        # ADDED: write episode reward to CSV 
        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([i_episode, ep_reward])

        if i_episode % save_interval == 0:
            ckpt = os.path.join(ckpt_folder, '{}.pth'.format(i_episode))
            torch.save(agent.ppo.policy.state_dict(), ckpt)
            print('Checkpoint saved')

        if i_episode % print_interval == 0:
            running_reward = int((running_reward / print_interval))
            print('Episode {} \t Avg reward: {}'.format(i_episode, running_reward))
            running_reward = 0

if __name__ == '__main__':

    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    folder = 'ppo_icm_agent'
    ckpt_folder = os.path.join(os.getcwd(), "Models", folder)
    if not os.path.exists(ckpt_folder):
        os.makedirs(ckpt_folder)

    CYBORG = CybORG(PATH, 'sim', agents={
        'Red': B_lineAgent
    })

    env = ChallengeWrapper(env=CYBORG, agent_name="Blue")
    input_dims = env.observation_space.shape[0]
    action_space = env.action_space.n

    print_interval = 50
    save_interval = 500
    max_episodes = 500000
    max_timesteps = 100

    update_timesteps = 20000
    K_epochs = 6
    eps_clip = 0.2
    gamma = 0.99
    lr = 0.002
    icm_beta = 0.02

    train(env, input_dims, action_space,
              max_episodes=max_episodes, max_timesteps=max_timesteps,
              update_timestep=update_timesteps, K_epochs=K_epochs,
              eps_clip=eps_clip, gamma=gamma, lr=lr,
              betas=[0.9, 0.990], icm_beta=icm_beta, ckpt_folder=ckpt_folder,
              print_interval=print_interval, save_interval=save_interval)