import torch
from CybORG.Agents import BaseAgent
from CybORG.Shared.Results import Results
from PPO.PPO import PPO
from ICM import ICM
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PPOICMAgent(BaseAgent):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        gamma: float = 0.99,
        lr: float = 2.5e-4,
        eps_clip: float = 0.2,
        K_epochs: int = 4,
        betas: tuple[float, float] = (0.9, 0.999),
        curiosity_beta: float = 0.01,
    ) -> None:
        super().__init__()
        self.ppo = PPO(
            state_dim=state_dim,
            action_dim=action_dim,
            lr=lr,
            gamma=gamma,
            eps_clip=eps_clip,
            K_epochs=K_epochs,
            betas=betas
        )
        self.icm = ICM(state_dim, action_dim).to(device)
        self.icm_beta = curiosity_beta
        self.icm_opt = torch.optim.Adam(self.icm.parameters(), lr=lr)
        self.action_dim = action_dim

    def end_episode(self) -> None:
        self.ppo.memory.clear()

    def clear_memory(self) -> None:
        self.ppo.memory.clear()

    def store(self, next_state, reward, done):
        if not isinstance(next_state, torch.Tensor):
            next_state = torch.tensor(next_state, dtype=torch.float32)
        self.ppo.store(next_state, reward, done)

    def compute_intrinsic_reward(self, state, next_state, action):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(device)
        action_onehot = F.one_hot(torch.tensor([action], dtype=torch.long), num_classes=self.action_dim).to(device)
        with torch.no_grad():
            intrinsic_reward, _, _ = self.icm(state, next_state, action_onehot)
        return intrinsic_reward.item()

    def train(self):
        #? should we batch this?
        memory = self.ppo.memory
        old_states = torch.squeeze(torch.stack(memory.states).to(device)).detach()
        old_next_states = torch.squeeze(torch.stack(memory.next_states).to(device)).detach()
        old_actions = torch.squeeze(torch.stack(memory.actions).to(device)).detach()
        actions_one_hot = F.one_hot(old_actions, num_classes=self.action_dim).float().to(device)

        pred_intrinsic_rewards, pred_action_logits, pred_next_feat = self.icm(old_states, old_next_states, actions_one_hot)
        
        # Forward loss: MSE between predicted and actual next features
        feat_next = self.icm.encoder(old_next_states)
        fwd_loss = 0.5 * self.icm.mse(pred_next_feat, feat_next)
        # Inverse loss: Cross-entropy between predicted and actual actions
        inv_loss = self.icm.ce(pred_action_logits, old_actions)
        
        fwd_loss = fwd_loss.mean() if fwd_loss.ndim > 0 else fwd_loss
        inv_loss = inv_loss.mean() if inv_loss.ndim > 0 else inv_loss

        icm_loss = fwd_loss + inv_loss

        self.icm_opt.zero_grad()
        icm_loss.backward()
        self.icm_opt.step()

        self.ppo.update() # shaped rewards are already in memory

    def get_action(self, observation, action_space=None, hidden=None):
        state = torch.tensor(observation, dtype=torch.float32).unsqueeze(0).to(device)
        action = self.ppo.select_action(state)
        self.last_state = state
        return int(action.item())

    def set_initial_values(self, action_space, observation):
        pass
