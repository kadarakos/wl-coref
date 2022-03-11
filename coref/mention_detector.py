import torch
from typing import Optional


class MentionDetector(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        dropout_rate: float,
        k: Optional[int] = None,
        n_layers: int = 2,
    ):
        super().__init__()
        self.lstm = torch.nn.LSTM(
            input_size,
            input_size,
            batch_first=True
        )
        layers = []
        for i in range(n_layers):
            layers.extend([torch.nn.Linear(input_size,
                                           input_size),
                           torch.nn.LeakyReLU(),
                           torch.nn.Dropout(dropout_rate)])
        layers.extend([torch.nn.Linear(input_size, 1)])
        self.net = torch.nn.Sequential(*layers)
        self.k = k

    def forward(self, word_features):
        mention_scores = self.net(word_features)
        return mention_scores

    def scores2ij(self, scores):
        N = scores.shape[0]
        pair_mask = torch.arange(N)
        pair_mask = pair_mask.unsqueeze(1) - pair_mask.unsqueeze(0)
        pair_mask = torch.log((pair_mask > 0).to(torch.float))
        pair_mask = pair_mask.to(scores.device)
        scores = scores.squeeze()
        scores_tiled = scores.tile((N, 1))
        scores_ij = scores_tiled + scores_tiled.T
        scores_ij += pair_mask
        return scores_ij
