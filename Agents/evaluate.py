import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from CybORG import CybORG
from CybORG.Agents import RedMeanderAgent, B_lineAgent
from CybORG.Agents.Wrappers import ChallengeWrapper
import inspect
from PPOAgent import PPOAgent
import random

PATH = str(inspect.getfile(CybORG))
PATH = PATH[:-10] + '/Shared/Scenarios/Scenario2.yaml'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate():
    model_paths = [
        # "Agents/Models/PPO/one.pth",
        "Agents/Models/PPO/two.pth", 
        # "Agents/Models/PPO/three.pth"
    ]
    
    red_agents = {
        'B_line': B_lineAgent,
        'RedMeander': RedMeanderAgent,
        'Random': [B_lineAgent, RedMeanderAgent]
    }
    episode_counts = [30, 50, 100]

    results = {}
    cyborg = CybORG(PATH, 'sim', agents={'Red': B_lineAgent})
    env = ChallengeWrapper(env=cyborg, agent_name="Blue")
    input_dims = env.observation_space.shape[0]
    action_space = env.action_space.n
    
    for i, model_path in enumerate(model_paths):
        if not os.path.exists(model_path):
            print(f"Warning: Model {model_path} not found, skipping...")
            continue
            
        print(f"Loading model {i+1}: {model_path}")
        agent = PPOAgent(input_dims, action_space)
        agent.load_checkpoint(model_path)
        
        model_results = {}
        
        for scenario_name, red_agent in red_agents.items():
            scenario_results = {}
            for num_episodes in episode_counts:
                print(f"  Evaluating {scenario_name} scenario with {num_episodes} episodes...")
                
                episode_rewards = []
                
                for episode in range(num_episodes):
                    if scenario_name == 'Random':
                        chosen_agent = random.choice(red_agent)
                    else:
                        chosen_agent = red_agent
                        
                    cyborg = CybORG(PATH, 'sim', agents={'Red': chosen_agent})
                    env = ChallengeWrapper(env=cyborg, agent_name="Blue")
                    
                    state = env.reset()
                    total_reward = 0
                    
                    for step in range(100):  # max 100 steps per episode
                        action = agent.get_action(state)
                        state, reward, done, _ = env.step(action)
                        total_reward += reward
                        
                        if done:
                            break
                    
                    episode_rewards.append(total_reward)
                    agent.end_episode()
                
                scenario_results[num_episodes] = {
                    'rewards': episode_rewards,
                    'mean_reward': np.mean(episode_rewards),
                    'std_reward': np.std(episode_rewards),
                    'min_reward': np.min(episode_rewards),
                    'max_reward': np.max(episode_rewards)
                }
            
            model_results[scenario_name] = scenario_results
        
        results[f"Model_{i+1}"] = model_results
    
    return results

def plot_results(results):
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle('Model Evaluation Results', fontsize=16, fontweight='bold')
    
    colors = ['blue', 'red', 'green']
    
    for model_idx, (model_name, model_results) in enumerate(results.items()):           
        for scenario_idx, (scenario_name, scenario_results) in enumerate(model_results.items()):
            ax = axes[model_idx, scenario_idx]
            
            episode_counts = list(scenario_results.keys())
            mean_rewards = [scenario_results[ep]['mean_reward'] for ep in episode_counts]
            std_rewards = [scenario_results[ep]['std_reward'] for ep in episode_counts]
            
            bars = ax.bar(episode_counts, mean_rewards, 
                         yerr=std_rewards, 
                         color=colors[model_idx], 
                         alpha=0.7, 
                         capsize=5)
            
            for bar, mean_val in zip(bars, mean_rewards):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                       f'{mean_val:.1f}', ha='center', va='bottom', fontsize=8)
            
            ax.set_title(f'{model_name} vs {scenario_name}', fontsize=10, fontweight='bold')
            ax.set_xlabel('Number of Episodes')
            ax.set_ylabel('Mean Reward')
            ax.grid(True, alpha=0.3)
            
            all_rewards = []
            for ep_data in scenario_results.values():
                all_rewards.extend(ep_data['rewards'])
            y_min, y_max = min(all_rewards), max(all_rewards)
            y_range = y_max - y_min
            ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    
    plt.savefig('evaluation_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    
    for model_name, model_results in results.items():
        print(f"\n{model_name}:")
        print("-" * 40)
        
        for scenario_name, scenario_results in model_results.items():
            print(f"  {scenario_name} scenario:")
            
            for num_episodes, stats in scenario_results.items():
                print(f"    {num_episodes} episodes: "
                      f"Mean={stats['mean_reward']:.2f} ± {stats['std_reward']:.2f} "
                      f"(Min={stats['min_reward']:.2f}, Max={stats['max_reward']:.2f})")

def plot_model_two_line_graph(results):
    """Create a line graph specifically for Model Two showing performance across episode counts and agent types."""
    if "Model_2" not in results:
        print("Warning: Model_2 not found in results")
        return
    
    model_results = results["Model_2"]
    episode_counts = [30, 50, 100]
    agent_types = ['B_line', 'RedMeander', 'Random']
    
    # Colors for different agent types
    colors = ['blue', 'red', 'green']
    markers = ['o', 's', '^']
    
    plt.figure(figsize=(10, 6))
    
    for i, agent_type in enumerate(agent_types):
        if agent_type not in model_results:
            continue
            
        mean_rewards = []
        std_rewards = []
        
        for num_episodes in episode_counts:
            if num_episodes in model_results[agent_type]:
                stats = model_results[agent_type][num_episodes]
                mean_rewards.append(stats['mean_reward'])
                std_rewards.append(stats['std_reward'])
            else:
                mean_rewards.append(0)
                std_rewards.append(0)
        
        # Plot line with error bars
        plt.errorbar(episode_counts, mean_rewards, yerr=std_rewards, 
                    color=colors[i], marker=markers[i], linewidth=2, 
                    markersize=8, capsize=5, capthick=2, 
                    label=f'{agent_type} Agent', alpha=0.8)
        
        # Add data point labels
        for j, (ep, mean_val) in enumerate(zip(episode_counts, mean_rewards)):
            plt.annotate(f'{mean_val:.1f}', 
                        (ep, mean_val), 
                        textcoords="offset points", 
                        xytext=(0,10), 
                        ha='center', 
                        fontsize=9)
    
    plt.title('PPO Agent Performance Across Different Episode Counts and Agent Types', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Number of Episodes', fontsize=12)
    plt.ylabel('Mean Reward', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Set x-axis ticks to match episode counts
    plt.xticks(episode_counts)
    
    # Adjust y-axis limits for better visualization
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('model_two_line_graph.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nModel Two line graph saved as 'model_two_line_graph.png'")

if __name__ == '__main__':
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    
    print("Starting model evaluation...")
    print("="*50)
    
    results = evaluate()
    print(results)  # TODO: Remove this
    plot_results(results)
    
    plot_model_two_line_graph(results)
    
    print("\nEvaluation completed! Results saved to 'evaluation_results.png' and 'model_two_line_graph.png'")
