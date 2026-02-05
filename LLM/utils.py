import logging
from dataclasses import dataclass
from typing import List, Dict, Any
import json
import statistics
from collections import defaultdict
import os
import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class EpisodeResult:
    episode_id: int
    total_reward: float
    steps: int
    actions_taken: List[str]
    action_names: List[str]  # Add action names tracking
    final_state: str
    duration: float
    red_agent_type: str
    max_steps: int = 100  # Add max_steps to track time step configuration

@dataclass
class TimeStepResult:
    max_steps: int
    episodes: List[EpisodeResult]
    summary: Dict[str, Any]

@dataclass
class EvaluationResults:
    config: Dict[str, Any]
    episodes: List[EpisodeResult]
    summary: Dict[str, Any]
    time_step_results: List[TimeStepResult] = None  # Add time step breakdown
    
def save_results(config, results: EvaluationResults):
    try:
        # Create intelligent folder structure
        model_name = config.get('hyperparams', {}).get('model_name', 'unknown_model')
        temperature = config.get('hyperparams', {}).get('temperature', 'unknown_temp')
        red_agent = config.get('red_agent', 'unknown_agent')
        prompt_name = config.get('prompt_name', 'unknown_prompt')
        
        # Clean names for folder structure
        model_name_clean = model_name.replace('/', '_').replace('-', '_')
        temperature_clean = str(temperature).replace('.', '_')
        
        # Create folder structure: results/model_name/red_agent/prompt/temperature/
        results_dir = os.path.join("results", model_name_clean, red_agent, prompt_name, f"temp_{temperature_clean}")
        os.makedirs(results_dir, exist_ok=True)
        
        # Get current timestamp for uniqueness
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Get max_steps from config or from episodes
        max_steps = config.get('max_steps', 100)
        if results.episodes:
            max_steps = max(ep.max_steps for ep in results.episodes)
        
        # Create filename with metadata
        filename = f"{max_steps}steps_{timestamp}.json"
        output_file = os.path.join(results_dir, filename)
        
        data = {
            "config": {
                "backend_type": results.config.get('backend_type'),
                "prompt_name": results.config.get('prompt_name'),
                "episodes": results.config.get('episodes'),
                "max_steps": results.config.get('max_steps'),
                "red_agent": results.config.get('red_agent'),
                "model_name": model_name,
                "timestamp": timestamp
            },
            "summary": results.summary,
            "episodes": [
                {
                    "episode_id": ep.episode_id,
                    "total_reward": ep.total_reward,
                    "steps": ep.steps,
                    "duration": ep.duration,
                    "red_agent_type": ep.red_agent_type,
                    "max_steps": ep.max_steps,
                    "actions_taken": ep.actions_taken,
                    "action_names": ep.action_names
                }
                for ep in results.episodes
            ]
        }
        
        # Add time step breakdown if available
        if results.time_step_results:
            data["time_step_breakdown"] = [
                {
                    "max_steps": tsr.max_steps,
                    "summary": tsr.summary,
                    "episodes": [
                        {
                            "episode_id": ep.episode_id,
                            "total_reward": ep.total_reward,
                            "steps": ep.steps,
                            "duration": ep.duration,
                            "red_agent_type": ep.red_agent_type,
                            "actions_taken": ep.actions_taken,
                            "action_names": ep.action_names
                        }
                        for ep in tsr.episodes
                    ]
                }
                for tsr in results.time_step_results
            ]
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Results saved to {output_file}")
        
        # Create matplotlib visualization
        create_evaluation_plots(results, output_file.replace('.json', '_plots.png'))
        
        return output_file
        
    except Exception as e:
        logger.error(f"Failed to save results: {e}")
        return None

def create_evaluation_plots(results: EvaluationResults, output_file: str):
    """Create matplotlib visualizations for evaluation results"""
    try:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'LLM Agent Evaluation Results - {results.config.get("hyperparams", {}).get("model_name", "Unknown Model")}', 
                    fontsize=16, fontweight='bold')
        
        # Plot 1: Reward distribution
        rewards = [ep.total_reward for ep in results.episodes]
        axes[0, 0].hist(rewards, bins=min(10, len(rewards)), alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].axvline(np.mean(rewards), color='red', linestyle='--', label=f'Mean: {np.mean(rewards):.2f}')
        axes[0, 0].set_title('Reward Distribution')
        axes[0, 0].set_xlabel('Total Reward')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Steps vs Reward scatter plot
        steps = [ep.steps for ep in results.episodes]
        axes[0, 1].scatter(steps, rewards, alpha=0.6, color='green')
        axes[0, 1].set_title('Steps vs Reward')
        axes[0, 1].set_xlabel('Steps Taken')
        axes[0, 1].set_ylabel('Total Reward')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Episode progression (reward over episodes)
        episode_ids = [ep.episode_id for ep in results.episodes]
        axes[0, 2].plot(episode_ids, rewards, marker='o', linestyle='-', color='purple', alpha=0.7)
        axes[0, 2].set_title('Reward Progression Over Episodes')
        axes[0, 2].set_xlabel('Episode ID')
        axes[0, 2].set_ylabel('Total Reward')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Plot 4: Action distribution (NEW)
        action_counts = defaultdict(int)
        for ep in results.episodes:
            if hasattr(ep, 'action_names') and ep.action_names:
                for action_name in ep.action_names:
                    action_counts[action_name] += 1
        
        if action_counts:
            actions = list(action_counts.keys())
            counts = list(action_counts.values())
            # Sort by count for better visualization
            sorted_indices = np.argsort(counts)[::-1]
            actions = [actions[i] for i in sorted_indices]
            counts = [counts[i] for i in sorted_indices]
            
            # Truncate long action names for display
            display_actions = [action[:20] + '...' if len(action) > 20 else action for action in actions]
            
            bars = axes[1, 0].bar(range(len(actions)), counts, color='orange', alpha=0.7)
            axes[1, 0].set_title('Action Distribution')
            axes[1, 0].set_xlabel('Actions')
            axes[1, 0].set_ylabel('Count')
            axes[1, 0].set_xticks(range(len(actions)))
            axes[1, 0].set_xticklabels(display_actions, rotation=45, ha='right')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Add count labels on bars
            for bar, count in zip(bars, counts):
                height = bar.get_height()
                axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 0.1,
                               f'{count}', ha='center', va='bottom', fontsize=8)
        else:
            axes[1, 0].text(0.5, 0.5, 'No action data available', 
                           ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Action Distribution')
        
        # Plot 5: Action frequency by timestep (NEW)
        if results.time_step_results:
            timesteps = [tsr.max_steps for tsr in results.time_step_results]
            avg_rewards = [tsr.summary['avg_reward'] for tsr in results.time_step_results]
            axes[1, 1].bar(timesteps, avg_rewards, color='lightblue', alpha=0.7)
            axes[1, 1].set_title('Average Reward by Time Steps')
            axes[1, 1].set_xlabel('Max Steps')
            axes[1, 1].set_ylabel('Average Reward')
            axes[1, 1].grid(True, alpha=0.3)
        else:
            # Alternative: Red agent breakdown
            if 'red_agent_breakdown' in results.summary:
                agent_types = list(results.summary['red_agent_breakdown'].keys())
                agent_avg_rewards = [results.summary['red_agent_breakdown'][agent]['avg_reward'] 
                                   for agent in agent_types]
                axes[1, 1].bar(agent_types, agent_avg_rewards, color='lightcoral', alpha=0.7)
                axes[1, 1].set_title('Average Reward by Red Agent Type')
                axes[1, 1].set_xlabel('Red Agent Type')
                axes[1, 1].set_ylabel('Average Reward')
                axes[1, 1].tick_params(axis='x', rotation=45)
                axes[1, 1].grid(True, alpha=0.3)
            else:
                axes[1, 1].text(0.5, 0.5, 'No additional data available', 
                               ha='center', va='center', transform=axes[1, 1].transAxes)
                axes[1, 1].set_title('Additional Metrics')
        
        # Plot 6: Action type breakdown (NEW)
        action_types = defaultdict(int)
        for ep in results.episodes:
            if hasattr(ep, 'action_names') and ep.action_names:
                for action_name in ep.action_names:
                    # Extract action type (first word before space)
                    action_type = action_name.split()[0] if action_name else 'Unknown'
                    # Condense decoy actions
                    if action_type.startswith('Decoy'):
                        action_type = 'Decoy'
                    action_types[action_type] += 1
        
        if action_types:
            types = list(action_types.keys())
            type_counts = list(action_types.values())
            
            axes[1, 2].pie(type_counts, labels=types, autopct='%1.1f%%', startangle=90)
            axes[1, 2].set_title('Action Type Distribution')
        else:
            axes[1, 2].text(0.5, 0.5, 'No action type data', 
                           ha='center', va='center', transform=axes[1, 2].transAxes)
            axes[1, 2].set_title('Action Type Distribution')
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Plots saved to {output_file}")
        
    except Exception as e:
        logger.error(f"Failed to create plots: {e}")

def calculate_summary(episode_results: List[EpisodeResult]) -> Dict[str, Any]:
    if not episode_results:
        return {"error": "No episodes completed"}
    
    rewards = [ep.total_reward for ep in episode_results]
    steps = [ep.steps for ep in episode_results]
    durations = [ep.duration for ep in episode_results]
    
    red_agent_results = {}
    for ep in episode_results:
        agent_type = ep.red_agent_type
        if agent_type not in red_agent_results:
            red_agent_results[agent_type] = []
        red_agent_results[agent_type].append(ep.total_reward)
    
    red_agent_stats = {}
    for agent_type, agent_rewards in red_agent_results.items():
        red_agent_stats[agent_type] = {
            "count": len(agent_rewards),
            "avg_reward": statistics.mean(agent_rewards),
            "success_rate": sum(1 for r in agent_rewards if r > 0) / len(agent_rewards)
        }
    
    # Calculate action statistics
    action_counts = defaultdict(int)
    action_types = defaultdict(int)
    
    for ep in episode_results:
        if hasattr(ep, 'action_names') and ep.action_names:
            for action_name in ep.action_names:
                action_counts[action_name] += 1
                # Extract action type (first word before space)
                action_type = action_name.split()[0] if action_name else 'Unknown'
                # Condense parameterized actions
                if action_type.startswith('Decoy'):
                    action_type = 'Decoy'
                elif action_type in ['Analyse', 'Remove', 'Restore', 'Monitor']:
                    action_type = action_type  # Keep as is, already condensed
                action_types[action_type] += 1
    
    # Get most common actions
    most_common_actions = sorted(action_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    most_common_types = sorted(action_types.items(), key=lambda x: x[1], reverse=True)[:5]
    
    summary = {
        "total_episodes": len(episode_results),
        "avg_reward": statistics.mean(rewards),
        "std_reward": statistics.stdev(rewards) if len(rewards) > 1 else 0,
        "min_reward": min(rewards),
        "max_reward": max(rewards),
        "avg_steps": statistics.mean(steps),
        "avg_duration": statistics.mean(durations),
        "total_duration": sum(durations),
        "red_agent_breakdown": red_agent_stats,
        "action_analysis": {
            "total_actions": sum(action_counts.values()),
            "unique_actions": len(action_counts),
            "most_common_actions": most_common_actions,
            "action_type_breakdown": dict(action_types),
            "most_common_types": most_common_types
        }
    }
    
    return summary

def calculate_time_step_summary(episode_results: List[EpisodeResult]) -> List[TimeStepResult]:
    """Group episodes by max_steps and calculate summary for each time step"""
    time_step_groups = defaultdict(list)
    
    for ep in episode_results:
        time_step_groups[ep.max_steps].append(ep)
    
    time_step_results = []
    for max_steps, episodes in time_step_groups.items():
        summary = calculate_summary(episodes)
        time_step_results.append(TimeStepResult(
            max_steps=max_steps,
            episodes=episodes,
            summary=summary
        ))
    
    return time_step_results
        
def print_results(results: EvaluationResults):
    print("\n" + "="*60)
    print("LLM BLUE AGENT EVALUATION RESULTS")
    print("="*60)
    
    print(f"\nConfiguration:")
    print(f"  Backend: {results.config.get('backend_type', 'gemini')}")
    print(f"  Prompt: {results.config.get('prompt_name', 'zero_shot')}")
    print(f"  Episodes: {results.config.get('episodes', 10)}")
    print(f"  Red Agent: {results.config.get('red_agent', 'random')}")
    
    print(f"\nOverall Summary:")
    print(f"  Total Episodes: {results.summary['total_episodes']}")
    print(f"  Average Reward: {results.summary['avg_reward']:.2f}")
    print(f"  Reward Std Dev: {results.summary['std_reward']:.2f}")
    print(f"  Min/Max Reward: {results.summary['min_reward']:.2f} / {results.summary['max_reward']:.2f}")
    print(f"  Average Steps: {results.summary['avg_steps']:.1f}")
    print(f"  Average Duration: {results.summary['avg_duration']:.2f}s")
    print(f"  Total Duration: {results.summary['total_duration']:.2f}s")
    
    if 'red_agent_breakdown' in results.summary:
        print(f"\nRed Agent Breakdown:")
        for agent_type, stats in results.summary['red_agent_breakdown'].items():
            print(f"  {agent_type}: {stats['count']} episodes, "
                  f"avg reward: {stats['avg_reward']:.2f}")
    
    # Print action analysis if available
    if 'action_analysis' in results.summary:
        action_analysis = results.summary['action_analysis']
        print(f"\nAction Analysis:")
        print(f"  Total Actions: {action_analysis['total_actions']}")
        print(f"  Unique Actions: {action_analysis['unique_actions']}")
        
        if action_analysis['most_common_actions']:
            print(f"  Most Common Actions:")
            for action, count in action_analysis['most_common_actions']:
                print(f"    {action}: {count} times")
        
        if action_analysis['most_common_types']:
            print(f"  Most Common Action Types:")
            for action_type, count in action_analysis['most_common_types']:
                print(f"    {action_type}: {count} times")
    
    # Print time step breakdown if available
    if results.time_step_results:
        print(f"\nTime Step Breakdown:")
        for tsr in results.time_step_results:
            print(f"\n  Max Steps: {tsr.max_steps}")
            print(f"    Episodes: {tsr.summary['total_episodes']}")
            print(f"    Avg Reward: {tsr.summary['avg_reward']:.2f}")
            print(f"    Avg Steps: {tsr.summary['avg_steps']:.1f}")
            print(f"    Avg Duration: {tsr.summary['avg_duration']:.2f}s")
            
            if 'red_agent_breakdown' in tsr.summary:
                for agent_type, stats in tsr.summary['red_agent_breakdown'].items():
                    print(f"    {agent_type}: {stats['count']} episodes, "
                          f"avg reward: {stats['avg_reward']:.2f}")
    
    print("\n" + "="*60)