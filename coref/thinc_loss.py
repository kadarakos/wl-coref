from typing import Tuple, List
from thinc.types import Ints1d, Ints2d, Floats1d, Floats2d, Floats3d, FloatsXd
from thinc.api import Model
from thinc.util import to_categorical


def add_dummy(xp, tensor: FloatsXd) -> FloatsXd:
    """ Prepends zeros to the first dimension of tensor."""
    shape: List[int] = list(tensor.shape)
    shape[1] = 1
    dummy = xp.zeros(shape)
    output = xp.concatenate((dummy, tensor), axis=1)
    return output


def get_ground_truth(
    xp,
    cluster_ids: Ints1d,
    top_indices: Ints1d,
    valid_pair_map: Ints2d
) -> Ints2d:
    """
    cluster_ids: tensor of shape [n_words], containing cluster indices
        for each word. Non-gold words have cluster id of zero.
    top_indices: tensor of shape [n_words, n_ants],
        indices of antecedents of each word
    valid_pair_map: boolean tensor of shape [n_words, n_ants],
        whether for pair at [i, j] (i-th word and j-th word)
        j < i is True
    Returns:
        tensor of shape [n_words, n_ants + 1] (dummy added),
            containing 1 at position [i, j] if i-th and j-th words corefer.
    """
    # [n_words, n_ants]
    y = cluster_ids[top_indices] * valid_pair_map
    # -1 for non-gold words
    y[y == 0] = -1
    # [n_words, n_cands + 1]
    y = add_dummy(xp, y)
    # True if coreferent
    y = (y == xp.expand_dims(cluster_ids, 1))
    # For all rows with no gold antecedents setting dummy to True
    y[y.sum(axis=1) == 0, 0] = True
    y = y.astype('float')
    return y


def coref_loss(
    model: Model,
    cluster_ids: Ints1d,
    scores: Floats2d,
    top_indices: Ints2d,
) -> Floats2d:
    """
    Compute the negative marginal log-likelihood loss
    and it's gradient for the coreference-scorer.
    """
    xp = model.ops.xp
    pair_mask = xp.arange(scores.shape[0])
    pair_mask = xp.expand_dims(pair_mask, 1) - xp.expand_dims(pair_mask, 0)
    pair_mask = xp.log((pair_mask > 0)).astype('float')
    pair_mask = pair_mask[:, :scores.shape[1] - 1]
    gscores = get_ground_truth(
        xp,
        cluster_ids,
        top_indices,
        (pair_mask > float("-inf"))
    )
    log_marg = model.ops.softmax(
        scores + xp.log(gscores),
        axis=1
    )
    log_norm = model.ops.softmax(scores, axis=1)
    grad = log_norm - log_marg
    loss = float((grad ** 2).sum())
    return loss, grad


def span_loss(
    model: Model,
    span_scores: Floats3d,
    starts: Ints1d,
    ends: Ints1d
) -> Tuple[float, Floats3d]:
    """
    Compute the sum of categorical-cross entropy losses
    for the span-start and span-end scores and the
    corresponding gradient.
    """
    start_scores = span_scores[:, :, 0]
    end_scores = span_scores[:, :, 1]
    n_classes = start_scores.shape[1]
    start_probs = model.ops.softmax(start_scores, axis=1)
    end_probs = model.ops.softmax(end_scores, axis=1)
    start_targets = to_categorical(starts, n_classes)
    end_targets = to_categorical(ends, n_classes)
    start_grads = (start_probs - start_targets)
    end_grads = (end_probs - end_targets)
    grads = model.ops.xp.stack((start_grads, end_grads), axis=2)
    loss = float((grads ** 2).sum())
    return loss, grads


def mention_loss(
    model: Model,
    mention_scores: Floats1d,
    mention_labels: Ints1d
):
    """
    Loss function for the mention detector.
    """
    mention_probs = model.ops.sigmoid(mention_scores)
    grads = mention_probs - mention_labels
    loss = float((grads ** 2).sum())
    return loss, grads
