from typing import List, Tuple

import torch
from coref.anaphoricity_scorer import AnaphoricityScorer
from coref.pairwise_encoder import DistancePairwiseEncoder
from coref.rough_scorer import RoughScorer
from coref.mention_detector import MentionDetector
from coref.utils import add_dummy


class CorefScorer(torch.nn.Module):
    """Combines all coref modules together to find coreferent spans.

    Attributes:
        config (coref.config.Config): the model's configuration,
            see config.toml for the details
        epochs_trained (int): number of epochs the model has been trained for

    Submodules (in the order of their usage in the pipeline):
        rough_scorer (RoughScorer)
        pw (PairwiseEncoder)
        a_scorer (AnaphoricityScorer)
        sp (SpanPredictor)
    """
    def __init__(
        self,
        device: str,
        dist_emb_size: int,
        hidden_size: int,
        n_layers: int,
        dropout_rate: float,
        roughk: int,
        batch_size: int
    ):
        super().__init__()
        """
        A newly created model is set to evaluation mode.

        Args:
            config_path (str): the path to the toml file with the configuration
            section (str): the selected section of the config file
            epochs_trained (int): the number of epochs finished
                (useful for warm start)
        """
        # TODO don't hardcode tok2vec size
        bert_emb = 1024
        self.roughk = roughk
        self.lstm = torch.nn.LSTM(
            input_size=bert_emb,
            hidden_size=bert_emb,
            batch_first=True,
        ).to(device)
        self.mention_detector = MentionDetector(
            bert_emb,
            bert_emb,
            2,
            dropout_rate
        ).to(device)
        self.rough_scorer = RoughScorer(
             bert_emb,
             dropout_rate,
             roughk
         ).to(device)
        self.pw = DistancePairwiseEncoder(
            dist_emb_size,
            dropout_rate
        ).to(device)
        pair_emb = bert_emb * 3 + self.pw.shape
        self.a_scorer = AnaphoricityScorer(
           pair_emb,
           hidden_size,
           n_layers,
           dropout_rate
        ).to(device)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.batch_size = batch_size

    def forward(
        self,
        word_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        This is a massive method, but it made sense to me to not split it into
        several ones to let one see the data flow.

        Args:
            word_features: torch.Tensor containing word encodings
        Returns:
            coreference scores and top indices
        """
        # words           [n_words, span_emb]
        # cluster_ids     [n_words]
        word_features = torch.unsqueeze(word_features, dim=0)
        words, _ = self.lstm(word_features)
        words = words.squeeze()
        words = self.dropout(words)
        # meniton_scores        [n_words, 1]
        # mention_scores_ij     [n_words, n_words]
        mention_scores = self.mention_detector(words)
        mention_scores_ij = self.mention_detector.scores2ij(mention_scores)
        mention_scores_ij = mention_scores_ij[:, :self.roughk]
        # Obtain bilinear scores and leave only top-k antecedents for each word
        # top_rough_scores  [n_words, n_ants]
        # top_indices       [n_words, n_ants]
        top_rough_scores, top_indices = self.rough_scorer(words)
        # Get pairwise features [n_words, n_ants, n_pw_features]
        pw = self.pw(top_indices)
        batch_size = self.batch_size
        a_scores_lst: List[torch.Tensor] = []

        for i in range(0, len(words), batch_size):
            pw_batch = pw[i:i + batch_size]
            words_batch = words[i:i + batch_size]
            top_indices_batch = top_indices[i:i + batch_size]
            top_rough_scores_batch = top_rough_scores[i:i + batch_size]

            # a_scores_batch    [batch_size, n_ants]
            a_scores_batch = self.a_scorer(
                all_mentions=words, mentions_batch=words_batch,
                pw_batch=pw_batch, top_indices_batch=top_indices_batch,
                top_rough_scores_batch=top_rough_scores_batch
            )
            a_scores_lst.append(a_scores_batch)
        coref_scores = torch.cat(a_scores_lst, dim=0)
        coref_scores += add_dummy(mention_scores_ij)
        return coref_scores, mention_scores, top_indices
