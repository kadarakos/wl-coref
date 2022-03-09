import torch


class MentionDetector(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        n_layers: int,
        dropout_rate: float,
        k: int
    ):
        super().__init__()
        layers = []
        for i in range(n_layers):
            layers.extend([torch.nn.Linear(hidden_size if i else input_size,
                                           hidden_size),
                           torch.nn.LeakyReLU(),
                           torch.nn.Dropout(dropout_rate)])
        layers.extend([torch.nn.Linear(hidden_size, 1)])
        self.net = torch.nn.Sequential(*layers)
        self.k = k

    def forward(self, mentions):
        N = mentions.shape[0]
        pair_mask = torch.arange(N)
        pair_mask = pair_mask.unsqueeze(1) - pair_mask.unsqueeze(0)
        pair_mask = torch.log((pair_mask > 0).to(torch.float))
        pair_mask = pair_mask.to(mentions.device)
        mention_scores = self.net(mentions)
        mention_scores = mention_scores.squeeze()
        scores_tiled = mention_scores.tile((N, 1))
        scores_ij = scores_tiled + scores_tiled.T
        scores_ij += pair_mask
        top_scores, indices = torch.topk(scores_ij,
                                         k=min(self.k, len(scores_ij)),
                                         dim=1, sorted=False)
        return top_scores, indices
