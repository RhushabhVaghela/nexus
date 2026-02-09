import numpy as np

from nexus.core.mri.spatial_activation_mapper import SpatialActivationMapper


def make_identity_layer(hidden_dim: int):
    def layer_fn(x):
        # If x is numpy array, return x; if scalar, wrap
        arr = np.asarray(x)
        # simple linear transform: add 0.1 to break zeros
        return arr + 0.1

    return layer_fn


def test_mapper_builds_maps_for_two_layers():
    samples = [np.zeros(8), np.ones(8)]
    layer_count = 2

    def loader(idx):
        return make_identity_layer(8)

    mapper = SpatialActivationMapper(
        layer_count=layer_count, layer_loader=loader, threshold=0.05
    )
    maps = mapper.build_spatial_map(samples)

    assert isinstance(maps, dict)
    assert 0 in maps and 1 in maps

    # Two samples per layer
    assert len(maps[0]) == 2
    for layer_idx in maps:
        for sample_idx, info in maps[layer_idx].items():
            assert "indices" in info and "values" in info
            assert info["indices"].dtype == int or info["indices"].dtype == np.int64
