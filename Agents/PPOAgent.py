import torch
from CybORG.Agents import BaseAgent
from CybORG.Shared.Results import Results
from PPO.PPO import PPO

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PPOAgent(BaseAgent):

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        gamma: float = 0.99,
        lr: float = 2.5e-4,
        eps_clip: float = 0.2,
        K_epochs: int = 4,
        betas: tuple[float, float] = (0.9, 0.999),
    ) -> None:
        super().__init__()
        self.agent = PPO(
            state_dim=state_dim,
            action_dim=action_dim,
            lr=lr,
            gamma=gamma,
            eps_clip=eps_clip,
            K_epochs=K_epochs,
            betas=betas
        )
        self.policy = self.agent.policy
        self.policy_old = self.agent.policy_old
        
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model weights from a checkpoint file.
        
        Parameters
        ----------
        checkpoint_path : str
            Path to the checkpoint file (.pth)
        """
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            self.policy.load_state_dict(checkpoint)
            self.policy_old.load_state_dict(checkpoint)
            print(f"Successfully loaded checkpoint from {checkpoint_path}")
        except Exception as e:
            print(f"Error loading checkpoint from {checkpoint_path}: {e}")
            raise
        
    def end_episode(self) -> None:
        self.clear_memory()
        
    def clear_memory(self) -> None:
        self.agent.memory.clear()
        
    def store(self, next_state, rewards, done):
        self.agent.store(next_state, rewards, done)
        
    def train(self):
        self.agent.update()

    def get_action(self, observation, action_space=None, hidden=None):
        state = torch.tensor(observation, dtype=torch.float32).unsqueeze(0).to(device)
        action = self.agent.select_action(state)
        return int(action.item())
