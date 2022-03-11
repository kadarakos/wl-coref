from typing import Tuple

import argparse
import tqdm
from contextlib import contextmanager
import datetime
import random
import time

import numpy as np  # type: ignore
import torch        # type: ignore

from coref.cluster_checker import ClusterChecker
from coval import Evaluator, get_cluster_info, b_cubed, muc, ceafe, lea
from thinc.api import require_gpu
from thinc.api import Adam
from coref.utils import _load_config, prfscore, mergespans
from coref.thinc_funcs import configure_pytorch_modules, doc2tensors
from coref.thinc_funcs import spaCyRoBERTa, _clusterize
from coref.thinc_funcs import detect_mention_heads, detect_mention_spans
from coref.thinc_funcs import predict_span_clusters
from coref.thinc_funcs import load_state, save_state
from coref.thinc_loss import coref_loss, span_loss, mention_loss
from process_litbank import load_docs


def train(
    config,
    model,
    span_predictor,
    mention_detector
):
    """
    Trains all the trainable blocks in the model using the config provided.
    """
    docs = load_docs('train-litbank.spacy')
    optimizer = Adam(config.learning_rate)
    docs_ids = list(range(len(docs)))
    best_val_score = 0
    encoder = spaCyRoBERTa()
    encoder.initialize()

    for epoch in range(model.attrs['epochs_trained'], config.train_epochs):
        running_c_loss = 0.0
        running_s_loss = 0.0
        running_m_loss = 0.0
        random.shuffle(docs_ids)
        pbar = tqdm.tqdm(docs_ids, unit="docs", ncols=0)
        for doc_id in pbar:
            doc = docs[doc_id]
            # Get data for CorefScorer
            sent_ids, cluster_ids, mention_labels, heads, starts, ends = doc2tensors(
                model.ops.xp,
                doc
            )
            word_features, _ = encoder([doc], False)
            word_features = word_features[0]
            mention_scores, m_backprop = mention_detector.begin_update(word_features)
            # Run CorefScorer
            (
                coref_scores,
                top_indices
            ), c_backprop = model.begin_update(word_features)
            # Compute coref loss
            c_loss, c_grads = coref_loss(
                model,
                cluster_ids,
                coref_scores,
                top_indices
            )
            m_loss, m_grads = mention_loss(
                model,
                mention_scores,
                mention_labels
            )
            # Update CorefScorer and MentionDetector
            c_backprop(c_grads)
            m_backprop(m_grads)
            model.finish_update(optimizer)
            mention_detector.finish_update(optimizer)
            # Maybe run span-predictor
            if starts.size and ends.size:
                span_scores, backprop_span = span_predictor.begin_update(
                    (
                        sent_ids,
                        word_features,
                        heads
                    )
                )
                s_loss, s_grads = span_loss(
                    span_predictor,
                    span_scores,
                    starts,
                    ends
                )
                backprop_span(s_grads)
                span_predictor.finish_update(optimizer)
                del span_scores
            else:
                s_loss = 0

            running_c_loss += c_loss
            running_s_loss += s_loss
            running_m_loss += m_loss

            del coref_scores
            del top_indices
            del m_loss

            pbar.set_description(
                f"Epoch {epoch + 1}:"
                f" {doc._.document_id:26}"
                f" c_loss: {running_c_loss / (pbar.n + 1):<.5f}"
                f" s_loss: {running_s_loss / (pbar.n + 1):<.5f}"
                f" m_loss: {running_m_loss / (pbar.n + 1):<.5f}"
            )

        model.attrs['epochs_trained'] += 1
        val_score = evaluate(
            config,
            model,
            span_predictor,
            mention_detector
        )
        if val_score > best_val_score:
            best_val_score = val_score
            print("New best {}".format(best_val_score))
            save_state(
                model,
                span_predictor,
                mention_detector,
                optimizer,
                config
            )


@torch.no_grad()
def evaluate(
    config,
    model,
    span_predictor,
    mention_detector,
    data_split: str = "dev",
    word_level_conll: bool = False
) -> Tuple[float, Tuple[float, float, float]]:

    encoder = spaCyRoBERTa()
    encoder.initialize()
    docs = load_docs('dev-litbank.spacy')
    n_docs = len(docs)
    running_loss = 0.0
    w_checker = ClusterChecker()
    s_checker = ClusterChecker()
    muc_evaluator = Evaluator(muc)
    bcubed_evaluator = Evaluator(b_cubed)
    ceafe_evaluator = Evaluator(ceafe)
    lea_evaluator = Evaluator(lea)
    s_correct = 0
    s_total = 0
    muc_score = 0.
    bcubed_score = 0.
    ceafe_score = 0.
    lea_score = 0.
    mention_head_p = 0.
    mention_head_r = 0.
    mention_head_f = 0.

    pbar = tqdm.tqdm(docs, unit="docs", ncols=0)
    for i, doc in enumerate(pbar):
        doc = docs[i]
        sent_ids, cluster_ids, mention_labels, heads, starts, ends = doc2tensors(
            model.ops.xp,
            doc
        )
        word_features, _ = encoder([doc], False)
        word_features = word_features[0]
        # Get data for SpanPredictor
        # Run CorefScorer
        coref_scores, top_indices = model.predict(word_features)
        mention_pred_heads = detect_mention_heads(
            mention_detector,
            word_features
        )
        # Compute coreference loss
        c_loss, c_grads = coref_loss(
            model,
            cluster_ids,
            coref_scores,
            top_indices
        )
        word_clusters = _clusterize(
             span_predictor,
             coref_scores,
             top_indices
         )
        running_loss += c_loss
        if starts.size and ends.size:
            span_scores = span_predictor.predict(
                (
                    sent_ids,
                    word_features,
                    heads
                )
            )
            span_clusters = predict_span_clusters(
                span_predictor,
                sent_ids,
                word_features,
                word_clusters
            )
            mention_pred_spans = detect_mention_spans(
                mention_detector,
                span_predictor,
                sent_ids,
                word_features
            )
            span_clusters = mergespans(
                span_clusters,
                mention_pred_spans
            )
            pred_starts = span_scores[:, :, 0].argmax(axis=1)
            pred_ends = span_scores[:, :, 1].argmax(axis=1)
            s_correct += ((starts == pred_starts) * (ends == pred_ends)).sum()
            s_total += len(pred_starts)

        w_checker.add_predictions(doc._.word_clusters, word_clusters)
        w_lea = w_checker.total_lea
        s_checker.add_predictions(doc._.coref_clusters, span_clusters)
        s_lea = s_checker.total_lea

        cluster_info = get_cluster_info(
            span_clusters,
            doc._.coref_clusters
        )
        muc_evaluator.update(cluster_info)
        bcubed_evaluator.update(cluster_info)
        ceafe_evaluator.update(cluster_info)
        lea_evaluator.update(cluster_info)
        muc_score += muc_evaluator.get_f1()
        bcubed_score += bcubed_evaluator.get_f1()
        ceafe_score += ceafe_evaluator.get_f1()
        lea_score += lea_evaluator.get_f1()
        predicted_mentions = mention_detector.ops.alloc1f(len(word_features))
        predicted_mentions[mention_pred_heads] = 1.0
        m_p, m_r, m_f = prfscore(
            predicted_mentions,
            mention_labels
        )
        mention_head_p += m_p
        mention_head_r += m_r
        mention_head_f += m_f

        pbar.set_description(
            f"{data_split}:"
            f" | WL: "
            f" loss: {running_loss / (pbar.n + 1):<.5f},"
            f" f1: {w_lea[0]:.5f},"
            f" p: {w_lea[1]:.5f},"
            f" r: {w_lea[2]:<.5f}"
            f" | SL: "
            f" sa: {s_correct / s_total:<.5f},"
            f" f1: {s_lea[0]:.5f},"
            f" p: {s_lea[1]:.5f},"
            f" r: {s_lea[2]:<.5f}"
        )
    print("LEA", s_lea[0])
    print("Paul LEA", lea_score/n_docs)
    print("MUC", muc_score/n_docs)
    print("CEAFE", ceafe_score/n_docs)
    print("BCUB", bcubed_score/n_docs)
    print("Mention head precision:", m_p/n_docs)
    print("Mention head recall:", m_r/n_docs)
    print("Mention head f-score:", m_f/n_docs)
    eval_score = w_lea[0] + s_lea[0]
    return eval_score


@contextmanager
def output_running_time():
    """ Prints the time elapsed in the context """
    start = int(time.time())
    try:
        yield
    finally:
        end = int(time.time())
        delta = datetime.timedelta(seconds=end - start)
        print(f"Total running time: {delta}")


def seed(value: int) -> None:
    """ Seed random number generators to get reproducible results """
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)           # type: ignore
    torch.backends.cudnn.deterministic = True   # type: ignore
    torch.backends.cudnn.benchmark = False      # type: ignore


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("mode", choices=("train", "eval"))
    argparser.add_argument("experiment")
    argparser.add_argument("--config-file", default="config.toml")
    argparser.add_argument("--data-split", choices=("train", "dev", "test"),
                           default="test",
                           help="Data split to be used for evaluation."
                                " Defaults to 'test'."
                                " Ignored in 'train' mode.")
    argparser.add_argument("--batch-size", type=int,
                           help="Adjust to override the config value if you're"
                                " experiencing out-of-memory issues")
    argparser.add_argument("--coref-weights",
                           help="Path to file with weights to load."
                                " If not supplied, in 'eval' mode the latest"
                                " weights of the experiment will be loaded;"
                                " in 'train' mode no weights will be loaded.")
    argparser.add_argument("--span-weights",
                           help="Path to file with weights to load."
                                " If not supplied, in 'eval' mode the latest"
                                " weights of the experiment will be loaded;"
                                " in 'train' mode no weights will be loaded.")
    argparser.add_argument("--mention-weights",
                           help="Path to file with weights to load."
                                " If not supplied, in 'eval' mode the latest"
                                " weights of the experiment will be loaded;"
                                " in 'train' mode no weights will be loaded.")
    argparser.add_argument("--word-level", action="store_true",
                           help="If set, output word-level conll-formatted"
                                " files in evaluation modes. Ignored in"
                                " 'train' mode.")
    args = argparser.parse_args()
    config = _load_config(args.config_file, args.experiment)

    if args.batch_size:
        config.a_scoring_batch_size = args.batch_size

    require_gpu()
    seed(2020)
    model, span_predictor, mention_detector = configure_pytorch_modules(config)

    if args.mode == "train":
        model.attrs['epochs_trained'] = 0
        with output_running_time():
            train(
                config,
                model,
                span_predictor,
                mention_detector
            )
    else:
        model, span_predictor = load_state(
            model,
            span_predictor,
            mention_detector,
            config,
            args.coref_weights,
            args.span_weights,
            args.mention_weights
        )
        evaluate(
            config,
            model,
            span_predictor,
            mention_detector,
            data_split=args.data_split,
            word_level_conll=args.word_level
        )
