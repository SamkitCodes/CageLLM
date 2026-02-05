import torch
import torch.nn as nn

class ICM(nn.Module):
    """Simple implementation of the Intrinsic Curiosity Module."""

    def __init__(self, state_dim: int, action_dim: int, encoding_dim: int = 128):
        super().__init__()

        self.encoding_dim = encoding_dim

        # Encoder that transforms raw observations to a feature space
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, encoding_dim),
            nn.ReLU(),
        )

        # Forward model predicts next state features from current features and the action taken
        self.forward_model = nn.Sequential(
            nn.Linear(encoding_dim + action_dim, encoding_dim),
            nn.ReLU(),
            nn.Linear(encoding_dim, encoding_dim),
        )

        # Inverse model predicts the taken action from the pair state features
        self.inverse_model = nn.Sequential(
            nn.Linear(encoding_dim * 2, encoding_dim),
            nn.ReLU(),
            nn.Linear(encoding_dim, action_dim),
        )

        self.mse = nn.MSELoss(reduction='mean')
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(
        self,
        state_prev: torch.Tensor,
        state_next: torch.Tensor,
        action_onehot: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Encode into features
        feat_prev = self.encoder(state_prev)
        feat_next = self.encoder(state_next)

        # Predict next feature given previous feature and action
        forward_input = torch.cat([feat_prev, action_onehot], dim=-1)
        pred_next_feat = self.forward_model(forward_input)

        # Predict the taken action (inverse model) from state features
        inverse_input = torch.cat([feat_prev, feat_next], dim=-1)
        pred_action_logits = self.inverse_model(inverse_input)

        # Curiosity reward is the MSE of the forward model (Prediction error)
        reward = 0.5 * self.mse(pred_next_feat, feat_next).mean(dim=-1)

        return reward, pred_action_logits, pred_next_feat
