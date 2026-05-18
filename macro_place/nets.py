"""
Net connectivity extraction for SA optimization.
"""

import torch

from macro_place._plc import PlacementCost
from macro_place.benchmark import Benchmark


def extract_edges(
    benchmark: Benchmark, plc: PlacementCost
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Extract a weighted edge list from the plc net representation.

    Each multi-pin net is expanded into hard-macro pairs weighted by
    `1 / (macros_in_net - 1)` so larger nets do not dominate as strongly.
    """
    name_to_bidx: dict[str, int] = {}
    for bidx, plc_idx in enumerate(benchmark.hard_macro_indices):
        name_to_bidx[plc.modules_w_pins[plc_idx].get_name()] = bidx

    edge_dict: dict[tuple[int, int], float] = {}
    for driver, sinks in plc.nets.items():
        macros: set[int] = set()
        for pin in [driver] + sinks:
            parent = pin.split("/")[0]
            if parent in name_to_bidx:
                macros.add(name_to_bidx[parent])
        if len(macros) < 2:
            continue
        macro_list = sorted(macros)
        weight = 1.0 / (len(macro_list) - 1)
        for i in range(len(macro_list)):
            for j in range(i + 1, len(macro_list)):
                pair = (macro_list[i], macro_list[j])
                edge_dict[pair] = edge_dict.get(pair, 0.0) + weight

    if not edge_dict:
        return torch.zeros(0, 2, dtype=torch.long), torch.zeros(0)

    edges = torch.tensor(list(edge_dict.keys()), dtype=torch.long)
    weights = torch.tensor([edge_dict[edge] for edge in edge_dict], dtype=torch.float32)
    return edges, weights
