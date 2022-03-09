"""
This module just takes the generated Doc objects
from the wl-coref library and converts them
to spacy.tokens.Doc
"""
import os
import logging
import jsonlines
import tqdm
import spacy

from collections import defaultdict
from typing import List, Dict, Tuple
from spacy.tokens import Doc, Token, Span, SpanGroup, DocBin


DATA_DIR = "."
FILENAME = "data/english_{}_head.jsonlines"
LOGGING_LEVEL = logging.WARNING  # DEBUG to output all duplicate spans
SPLITS = ("development", "test", "train")
nlp = spacy.blank("en")


def _get_word_clusters(doc: Doc):
    """Group words into mention clusters."""
    word_clusters_dict: Dict[int, List[int]] = defaultdict(list)
    word_clusters_list: List[List[int]] = []

    for token in doc:
        if token._.coref_cluster_i is not None:
            word_clusters_dict[token._.coref_cluster_i].append(token.i)

    for cluster in sorted(word_clusters_dict.keys()):
        word_clusters_list.append(word_clusters_dict[cluster])

    return word_clusters_list


def _get_span_clusters(doc: Doc) -> List[List[Tuple[int, int]]]:
    """Group spans into mention clusters."""
    span_clusters_dict: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    span_clusters_list: List[List[Tuple[int, int]]] = []

    for span in doc.spans['mentions']:
        span_clusters_dict[span._.coref_cluster_i].append(
            (span.start, span.end)
        )

    for cluster in sorted(span_clusters_dict.keys()):
        span_clusters_list.append(span_clusters_dict[cluster])

    return span_clusters_list


def _get_cluster_id(token: Token):
    """
    For training we use the 0-th "empty" cluster for
    negative labels and shift the cluster ids by + 1.
    """
    cluster_id = token._.coref_cluster_i
    if cluster_id is not None:
        return cluster_id + 1
    else:
        return 0


def _sentids_to_sentstarts(
    sent_ids: List[int]
) -> List[int]:
    """
    Convert sentence id per token to sentence start indicators.
    """
    sent_starts = [1]
    for i in range(1, len(sent_ids)):
        start = int(sent_ids[i] != sent_ids[i - 1])
        sent_starts.append(start)
    return sent_starts


def load_spacy_data(path):
    return list(
        DocBin(
            store_user_data=True
        ).from_disk(
            path
        ).get_docs(
            nlp.vocab
        )
    )


# Book keeping ids for OntoNotes ConLL evaluation
Doc.set_extension("document_id", default=None)
Doc.set_extension("part_id", default=None)
Doc.set_extension("coref_head2span", default=None)
# Used for sentence-level masking for SpanPredictor
Token.set_extension('sent_i', default=None)
# Mention clusters on word-level (span heads)
Token.set_extension('coref_cluster_i', default=None)
Token.set_extension('cluster_id', getter=_get_cluster_id)
Token.set_extension('is_head', default=False)
Doc.set_extension(
    "word_clusters",
    getter=_get_word_clusters,
)
# Mention clusters on span-level
Span.set_extension('coref_cluster_i', default=None)
Doc.set_extension(
    "coref_clusters",
    getter=_get_span_clusters,
)


if __name__ == "__main__":
    logging.basicConfig(level=LOGGING_LEVEL)
    path = os.path.join(DATA_DIR, FILENAME)
    for split in SPLITS:
        with jsonlines.open(path.format(split, ""), mode='r') as inf:
            # inf = pickle.load(f)
            doc_bin = DocBin(store_user_data=True)
            for doc in tqdm.tqdm(inf):
                sent_starts = _sentids_to_sentstarts(doc['sent_id'])
                spacy_doc = Doc(
                    vocab=nlp.vocab,
                    words=doc['cased_words'],
                    sent_starts=sent_starts
                )
                spacy_doc._.document_id = doc['document_id']
                spacy_doc._.part_id = doc['part_id']
                spacy_doc._.coref_head2span = doc['head2span']
                assert len(spacy_doc) == len(doc['sent_id'])
                for i, token in enumerate(spacy_doc):
                    token = spacy_doc[i]
                    token._.sent_i = doc['sent_id'][i]
                for i, cluster in enumerate(doc['word_clusters']):
                    for position in cluster:
                        token = spacy_doc[position]
                        token._.coref_cluster_i = i
                        token._.is_head = True
                mention_spans = []
                for i, cluster in enumerate(doc['span_clusters']):
                    for start, end in cluster:
                        mention_span = spacy_doc[start:end]
                        mention_span._.coref_cluster_i = i
                        mention_spans.append(mention_span)
                spacy_doc.spans['mentions'] = SpanGroup(
                    spacy_doc,
                    name='mentions',
                    spans=mention_spans
                )
                doc_bin.add(spacy_doc)
            if split == "development":
                split = 'dev'
            doc_bin.to_disk("{}.spacy".format(split))
