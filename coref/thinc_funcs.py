import os
import spacy

from datetime import datetime
from typing import Tuple, List

from thinc.types import Floats2d, Ints1d, Ints2d
from thinc.util import xp2torch, torch2xp
from thinc.api import PyTorchWrapper, Model, reduce_mean
from thinc.api import ArgsKwargs
from spacy_transformers.architectures import transformer_tok2vec_v3
from spacy_transformers.span_getters import configure_strided_spans

from coref.utils import GraphNode
from coref.coref_model import CorefScorer
from coref.span_predictor import SpanPredictor


# XXX this global nlp feels sketchy
nlp = spacy.blank("en")


def convert_coref_scorer_inputs(
    model: Model,
    X: Floats2d,
    is_train: bool
):
    word_features = xp2torch(X, requires_grad=False)
    return ArgsKwargs(args=(word_features, ), kwargs={}), lambda dX: []


def convert_coref_scorer_outputs(
    model: Model,
    inputs_outputs,
    is_train: bool
):
    _, outputs = inputs_outputs
    coref_scores, mention_scores, indices = outputs

    def convert_for_torch_backward(
        dY: Tuple[Floats2d, Floats2d]
    ) -> ArgsKwargs:
        dY_coref, dY_mention = xp2torch(dY[0]), xp2torch(dY[1])
        return ArgsKwargs(
            args=([coref_scores],),
            kwargs={"grad_tensors": [dY_coref, dY_mention]},
        )

    coref_scores_xp = torch2xp(coref_scores)
    mention_scores_xp = torch2xp(mention_scores.squeeze())
    indices_xp = torch2xp(indices)
    return (
        coref_scores_xp,
        mention_scores_xp,
        indices_xp
    ), convert_for_torch_backward


def convert_span_predictor_inputs(
    model: Model,
    X: Tuple[Ints1d, Floats2d, Ints1d],
    is_train: bool
):
    sent_id = xp2torch(X[0], requires_grad=False)
    word_features = xp2torch(X[1], requires_grad=False)
    head_ids = xp2torch(X[2], requires_grad=False)
    argskwargs = ArgsKwargs(args=(sent_id, word_features, head_ids), kwargs={})
    return argskwargs, lambda dX: []


def spaCyRoBERTa(
) -> Model[spacy.tokens.Doc, List[Floats2d]]:
    """Configures and returns RoBERTa from spacy-transformers."""
    return transformer_tok2vec_v3(
        name='roberta-large',
        get_spans=configure_strided_spans(400, 350),
        tokenizer_config={'add_prefix_space': True},
        pooling=reduce_mean()
    )


def doc2tensors(
    xp,
    doc: spacy.tokens.Doc
) -> Tuple[Ints1d, Ints1d, Ints1d, Ints1d, Ints1d]:
    sent_ids = [token._.sent_i for token in doc]
    cluster_ids = [token._.cluster_id for token in doc]
    mention_labels = [token._.is_head for token in doc]
    head2span = sorted(doc._.coref_head2span)
    if not head2span:
        heads, starts, ends = [], [], []
    else:
        heads, starts, ends = zip(*head2span)
    sent_ids = xp.asarray(sent_ids)
    cluster_ids = xp.asarray(cluster_ids)
    mention_labels = xp.asarray(mention_labels)
    heads = xp.asarray(heads)
    starts = xp.asarray(starts)
    ends = xp.asarray(ends) - 1
    return sent_ids, cluster_ids, mention_labels, heads, starts, ends


def configure_pytorch_modules(config):
    """
    Initializes CorefScorer and SpanPredictor Pytorch modules.
    """
    coref_scorer = PyTorchWrapper(
        CorefScorer(
            config.device,
            config.embedding_size,
            config.hidden_size,
            config.n_hidden_layers,
            config.dropout_rate,
            config.rough_k,
            config.a_scoring_batch_size
        ),
        convert_inputs=convert_coref_scorer_inputs,
        convert_outputs=convert_coref_scorer_outputs
    )
    span_predictor = PyTorchWrapper(
        SpanPredictor(
            1024,
            config.sp_embedding_size,
            config.device
        ),
        convert_inputs=convert_span_predictor_inputs
    )
    return coref_scorer, span_predictor


def save_state(model, span_predictor, optimizer, config):
    """Serialize CorefScorer and SpanPredictor."""
    time = datetime.strftime(datetime.now(), "%Y.%m.%d_%H.%M")
    span_path = os.path.join(
        config.data_dir,
        f"span-{config.section}"
        f"_(e{model.attrs['epochs_trained']}_{time}).pt"
    )
    coref_path = os.path.join(config.data_dir,
                        f"coref-{config.section}"
                        f"_(e{model.attrs['epochs_trained']}_{time}).pt")
    with model.use_params(optimizer.averages):
        model.to_disk(coref_path)
    with span_predictor.use_params(optimizer.averages):
        span_predictor.to_disk(span_path)


def load_state(
    model,
    span_predictor,
    config,
    coref_path,
    span_path,
):
    """Deserialize CorefScorer and SpanPredictor."""
    model.from_disk(coref_path)
    span_predictor.from_disk(span_path)
    return model, span_predictor


def predict_span_clusters(span_predictor: Model,
                          sent_ids: Ints1d,
                          words: Floats2d,
                          clusters: List[Ints1d]):
    """
    Predicts span clusters based on the word clusters.

    Args:
        doc (Doc): the document data
        words (torch.Tensor): [n_words, emb_size] matrix containing
            embeddings for each of the words in the text
        clusters (List[List[int]]): a list of clusters where each cluster
            is a list of word indices

    Returns:
        List[List[Span]]: span clusters
    """
    if not clusters:
        return []

    xp = span_predictor.ops.xp
    heads_ids = xp.asarray(sorted(i for cluster in clusters for i in cluster))
    scores = span_predictor.predict((sent_ids, words, heads_ids))
    starts = scores[:, :, 0].argmax(axis=1).tolist()
    ends = (scores[:, :, 1].argmax(axis=1) + 1).tolist()

    head2span = {
        head: (start, end)
        for head, start, end in zip(heads_ids.tolist(), starts, ends)
    }

    return [[head2span[head] for head in cluster]
            for cluster in clusters]


def _clusterize(
        model,
        scores: Floats2d,
        top_indices: Ints2d
):
    xp = model.ops.xp
    antecedents = scores.argmax(axis=1) - 1
    not_dummy = antecedents >= 0
    coref_span_heads = xp.arange(0, len(scores))[not_dummy]
    antecedents = top_indices[coref_span_heads, antecedents[not_dummy]]
    n_words = scores.shape[0]
    nodes = [GraphNode(i) for i in range(n_words)]
    for i, j in zip(coref_span_heads.tolist(), antecedents.tolist()):
        nodes[i].link(nodes[j])
        assert nodes[i] is not nodes[j]

    clusters = []
    for node in nodes:
        if len(node.links) > 0 and not node.visited:
            cluster = []
            stack = [node]
            while stack:
                current_node = stack.pop()
                current_node.visited = True
                cluster.append(current_node.id)
                stack.extend(link for link in current_node.links if not link.visited)
            assert len(cluster) > 1
            clusters.append(sorted(cluster))
    return sorted(clusters)
