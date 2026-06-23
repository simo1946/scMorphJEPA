"""scMorphJEPA: Self-supervised cell morphology learning via spatial JEPA + SIGReg.

First image-based JEPA for fluorescence microscopy. No EMA, no augmentations.
Stage 1 of the CellAgora research program.

Top-level names (build_scmorphjepa, ScMorphJEPA, ...) are lazily imported so that
the torch-free analysis layer (scmorphjepa.analysis) can be used without importing
torch. See PEP 562.
"""

__version__ = "0.1.8"
__model_name__ = "scMorphJEPA"

__all__ = [
    "build_scmorphjepa", "load_trained_model",
    "ScMorphJEPA", "SpatialPredictor", "SIGReg",
    "__version__", "__model_name__",
]

_LAZY = {
    "build_scmorphjepa": "scmorphjepa.models.builder",
    "load_trained_model": "scmorphjepa.models.builder",
    "ScMorphJEPA": "scmorphjepa.models.cell_jepa",
    "SpatialPredictor": "scmorphjepa.models.cell_jepa",
    "SIGReg": "scmorphjepa.models.cell_jepa",
}


def __getattr__(name):
    if name in _LAZY:
        import importlib
        module = importlib.import_module(_LAZY[name])
        return getattr(module, name)
    raise AttributeError(f"module 'scmorphjepa' has no attribute '{name}'")
