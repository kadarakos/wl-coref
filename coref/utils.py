""" Contains functions not directly linked to coreference resolution """

from typing import List, Set

import torch
import toml

from coref.config import Config

EPSILON = 1e-7

class GraphNode:
    def __init__(self, node_id: int):
        self.id = node_id
        self.links: Set[GraphNode] = set()
        self.visited = False

    def link(self, another: "GraphNode"):
        self.links.add(another)
        another.links.add(self)

    def __repr__(self) -> str:
        return str(self.id)


def _load_config(
    config_path: str,
    section: str
) -> Config:
    config = toml.load(config_path)
    default_section = config["DEFAULT"]
    current_section = config[section]
    unknown_keys = (set(current_section.keys())
                    - set(default_section.keys()))
    if unknown_keys:
        raise ValueError(f"Unexpected config keys: {unknown_keys}")
    return Config(section, **{**default_section, **current_section})


def add_dummy(tensor: torch.Tensor, eps: bool = False):
    """ Prepends zeros (or a very small value if eps is True)
    to the first (not zeroth) dimension of tensor.
    """
    kwargs = dict(device=tensor.device, dtype=tensor.dtype)
    shape: List[int] = list(tensor.shape)
    shape[1] = 1
    if not eps:
        dummy = torch.zeros(shape, **kwargs)          # type: ignore
    else:
        dummy = torch.full(shape, EPSILON, **kwargs)  # type: ignore
    output = torch.cat((dummy, tensor), dim=1)
    return output
