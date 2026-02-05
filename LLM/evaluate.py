import inspect, random, numpy as np
import logging, time, copy, pprint
from tqdm import tqdm
from typing import Dict, Any, List, Optional
from prettytable import PrettyTable

from CybORG import CybORG
from CybORG.Agents import RedMeanderAgent, B_lineAgent, SleepAgent
from CybORG.Agents.Wrappers import ChallengeWrapper, BlueTableWrapper
from CybORG.Agents.SimpleAgents import BlueMonitorAgent
from CybORG.Shared.Results import Results

from configs import prompts
from blue_agent import LLMAgent, LLMPolicy
from utils import EpisodeResult, EvaluationResults, save_results, print_results, calculate_summary, calculate_time_step_summary
from configs.action_to_index import ACTION_MAPPING

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LLMAgentEvaluator:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.results = []
        
        PATH = str(inspect.getfile(CybORG))
        self.PATH = PATH[:-10] + '/Shared/Scenarios/Scenario2.yaml'
        
    def create_agent(self, env) -> LLMAgent:
        obs_space = env.observation_space
        agent = LLMAgent(
            name="Blue",
            policy=LLMPolicy,
            obs_space=obs_space,
            llm_config=self.config
        )
        logger.info(f"Created LLM agent with LLMPolicy and obs_space: {obs_space}")
        return agent
    
    def create_red_agent(self) -> Any:
        red_agent_type = self.config.get('red_agent', 'random')
        
        if red_agent_type == 'bline':
            return B_lineAgent
        elif red_agent_type == 'meander':
            return RedMeanderAgent
        else:
            return SleepAgent()
    
    def create_environment(self, red_agent) -> tuple:
        cyborg = CybORG(self.PATH, 'sim', agents={'Red': red_agent})
        env = ChallengeWrapper(env=cyborg, agent_name="Blue")
        return cyborg, env
    
    def get_action_name(self, action_index: int) -> str:
        """Convert action index to action name using ACTION_MAPPING"""
        # Create reverse mapping from index to action name
        reverse_mapping = {v: k for k, v in ACTION_MAPPING.items()}
        return reverse_mapping.get(action_index, f"Unknown_Action_{action_index}")
    
    def run_episode(self, agent: LLMAgent, episode_id: int, env=None, max_steps:int =100) -> List[EpisodeResult]:
        logger.info(f"Starting episode {episode_id} with results tracking")
        start_time = time.time()
        actions_taken = []
        action_names = []  # Track action names
        total_reward = 0.0
        steps = 0
        red_agent = self.create_red_agent()
        cyborg, env = self.create_environment(red_agent)
        state = env.reset()
        
        episode_results = []
        timesteps = [30, 50, 100]
        
        for step in tqdm(range(max_steps), desc="Evaluating"):
            action = agent.get_action(state)
            action_name = self.get_action_name(action)
            
            actions_taken.append(f"Step {step}: Action {action}")
            action_names.append(action_name)  # Store action name
            
            next_state, reward, done, _ = env.step(action)
            result = Results(observation=state, action=action, reward=reward)
            logger.info(f"Step {step}: Action {action} ({action_name}) resulted in state: {next_state} and reward: {reward}")
            state = next_state
            total_reward += reward
            steps += 1
            
            if steps in timesteps:
                duration = time.time() - start_time
                intermediate_result = EpisodeResult(
                    episode_id=episode_id,
                    total_reward=total_reward,
                    steps=steps,
                    actions_taken=actions_taken.copy(),
                    action_names=action_names.copy(),  # Include action names
                    final_state=str(state),
                    duration=duration,
                    red_agent_type=type(red_agent).__name__,
                    max_steps=steps
                )
                episode_results.append(intermediate_result)
                logger.info(f"Episode {episode_id} intermediate result at step {steps}: reward={total_reward:.2f}")
            
            if done: break
            logger.info((str(cyborg.get_last_action('Blue')), str(cyborg.get_last_action('Red')), _))
            # time.sleep(60)
        
        logger.info(f"Episode {episode_id} completed: reward={total_reward:.2f}, steps={steps}")
        agent.end_episode()
        return episode_results
    
    def evaluate(self, episodes=None, max_steps=None) -> EvaluationResults:
        logger.info("Starting LLM Blue Agent evaluation")
        # logger.info(f"Configuration: {self.config}")
        # Create a temp env to get obs_space
        red_agent = self.create_red_agent()
        _, env = self.create_environment(red_agent)
        agent = self.create_agent(env)
        episode_results = []
        n_episodes = episodes if episodes is not None else self.config.get('episodes', 10)
        max_steps = max_steps if max_steps is not None else self.config.get('max_steps', 30)
        
        for episode_id in range(n_episodes):
            result = self.run_episode(agent, episode_id, max_steps=max_steps)
            episode_results.append(result)
            
        summary = calculate_summary(episode_results)
        results = EvaluationResults(
            config=self.config,
            episodes=episode_results,
            summary=summary
        )
        save_results(self.config, results)
        logger.info("Evaluation completed")
        return results    

if __name__ == "__main__":
    config = {
        'llm': "gemini",
        'hyperparams': {"model_name": "gemini-2.5-pro", "temperature": 0.7},
        'red_agent': "bline",
        'max_steps': 100,
        'episodes': 1
    }
    random.seed(0)
    np.random.seed(0)
    
    temperatures = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    red_agents = ["meander", "bline"]
    # models = ["gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-2.5-pro"]
    models = ["gemini-2.5-pro"]
    prompts = ["base", "strategy", "chain-of-thought"]
    for model in models:
        for red_agent in red_agents:
            for prompt in prompts:
                for temperature in temperatures:
                    print(f"Model: {model}, Red Agent: {red_agent}, Prompt: {prompt}, Temperature: {temperature}")
                    config['hyperparams']['model_name'] = model
                    config['red_agent'] = red_agent
                    config['prompt_name'] = prompt
                    config['hyperparams']['temperature'] = temperature
                    logger.info(f"Running evaluation with config: {config}")
                    evaluator = LLMAgentEvaluator(config)
                    red_agent_instance = evaluator.create_red_agent()
                    _, env = evaluator.create_environment(red_agent_instance)
                    agent = evaluator.create_agent(env)
                    
                    all_episode_results = []
                    for episode_id in range(config['episodes']):
                        episode_results = evaluator.run_episode(agent, episode_id, max_steps=config['max_steps'])
                        all_episode_results.extend(episode_results)
                    
                    overall_summary = calculate_summary(all_episode_results)
                    time_step_results = calculate_time_step_summary(all_episode_results)
                    
                    comprehensive_results = EvaluationResults(
                        config=config,
                        episodes=all_episode_results,
                        summary=overall_summary,
                        time_step_results=time_step_results
                    )
                    
                    save_results(config, comprehensive_results)
                    print(f"Avg Reward: {overall_summary['avg_reward']:.2f}")
                    
                    
                    
# Have something quantative in the dicussion.
# Expand the background, add more details.
# Remove the hybrid, 
# Try to find out if there is an ensemble of experts. 
# Anther possibility, is to use LLM maybe when latency use RL or something like that. 
# Use the RL as a backup.
