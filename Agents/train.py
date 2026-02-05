# checkout https://github.com/geekyutao/PyTorch-PPO/blob/master/PPO_discrete.py
# checkout https://github.com/john-cardiff/-cyborg-cage-2
import os
import torch
import numpy as np
from CybORG import CybORG
from CybORG.Agents import RedMeanderAgent, B_lineAgent
from CybORG.Agents.Wrappers import ChallengeWrapper
import inspect
from Agents.PPOAgent import PPOAgent
from tqdm import tqdm
import random

PATH = str(inspect.getfile(CybORG))
PATH = PATH[:-10] + '/Shared/Scenarios/Scenario2.yaml'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train(env, input_dims, action_space,
          max_episodes, max_timesteps, update_timestep, K_epochs, eps_clip,
          gamma, lr, betas, ckpt_folder, checkpoint_path=None, print_interval=10, save_interval=100):

    agent = PPOAgent(input_dims, action_space, gamma, lr, eps_clip, K_epochs, betas)
    
    # Load checkpoint if provided
    start_episode = 1
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        try:
            agent.load_checkpoint(checkpoint_path)
            
            checkpoint_name = os.path.basename(checkpoint_path)
            if checkpoint_name.endswith('.pth'):
                try:
                    episode_num = int(checkpoint_name.split('.')[0])
                    start_episode = episode_num + 1
                    print(f"Resuming training from episode {start_episode}")
                except ValueError:
                    print("Could not parse episode number from checkpoint filename, starting from episode 1")
                    
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Starting training from scratch")
            start_episode = 1
    else:
        print("No checkpoint provided or checkpoint not found, starting training from scratch")
    
    red_agents = [B_lineAgent, RedMeanderAgent]
    running_reward, time_step = 0, 0

    for i_episode in tqdm(range(start_episode, max_episodes + 1), desc="Training"):
        red_agent = random.choice(red_agents)
        cyborg = CybORG(PATH, 'sim', agents={'Red': red_agent})
        env = ChallengeWrapper(env=cyborg, agent_name="Blue")
        
        state = env.reset()
        for t in range(max_timesteps):
            time_step += 1
            action = agent.get_action(state)
            state, reward, done, _ = env.step(action)
            agent.store(state, reward, done)

            if time_step % update_timestep == 0:
                agent.train()
                agent.clear_memory()
                time_step = 0

            running_reward += reward

        agent.end_episode()

        if i_episode % save_interval == 0:
            ckpt = os.path.join(ckpt_folder, '{}.pth'.format(i_episode))
            torch.save(agent.policy.state_dict(), ckpt)
            print('Checkpoint saved')

        if i_episode % print_interval == 0:
            running_reward = int((running_reward / print_interval))
            print('Episode {} \t Avg reward: {}'.format(i_episode, running_reward))
            running_reward = 0

if __name__ == '__main__':

    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    folder = 'new'
    ckpt_folder = os.path.join(os.getcwd(), "Models", folder)
    if not os.path.exists(ckpt_folder):
        os.makedirs(ckpt_folder)

    # Path to resume training (optional)
    checkpoint_path = os.path.join(ckpt_folder, "68000.pth")
    # checkpoint_path = None  # Set to None to start from scratch

    CYBORG = CybORG(PATH, 'sim', agents={
        'Red': B_lineAgent
    })
    
    env = ChallengeWrapper(env=CYBORG, agent_name="Blue")
    input_dims = env.observation_space.shape[0]
    action_space = env.action_space.n
    
    print_interval = 50
    save_interval = 200
    max_episodes = 500000
    max_timesteps = 100
  
    update_timesteps = 20000
    K_epochs = 6
    eps_clip = 0.2
    gamma = 0.99
    lr = 0.002

    train(env, input_dims, action_space,
              max_episodes=max_episodes, max_timesteps=max_timesteps,
              update_timestep=update_timesteps, K_epochs=K_epochs,
              eps_clip=eps_clip, gamma=gamma, lr=lr,
              betas=[0.9, 0.990], ckpt_folder=ckpt_folder,
              checkpoint_path=checkpoint_path,
              print_interval=print_interval, save_interval=save_interval)

# 155000 EP
# 217350 EP