"""Stage 4 scoring helpers: handcrafted features, score fusion, and metrics.

The package is import-light on purpose. Import the submodules you need directly::

    from scoring.handcrafted_features import describe_image, pair_features
    from scoring.fusion import FusionScorer, load_fusion_config
    from scoring.metrics import eer_threshold

Nothing here imports PyTorch, so these helpers can be used from a plain NumPy script.
"""
