"""Helpers for importing project modules without TensorFlow installed."""

from __future__ import annotations

import importlib
import sys
import types
from typing import Mapping, Optional, Sequence, Tuple


class TensorFlowStub(types.ModuleType):
    """Minimal TensorFlow stub used by unit tests that do not execute TF ops."""

    def __init__(self, gpus: Optional[Sequence[str]] = None) -> None:
        """Initialize the TensorFlow stub."""
        super().__init__("tensorflow")
        self.memory_growth_calls = []
        self.float32 = "float32"
        self.int32 = "int32"
        experimental = types.SimpleNamespace(
            list_physical_devices=lambda device_type: list(gpus or []),
            set_memory_growth=self._set_memory_growth,
        )
        self.config = types.SimpleNamespace(experimental=experimental)
        self.constant = lambda value, dtype=None: value

    def _set_memory_growth(self, gpu: str, enabled: bool) -> None:
        """Record GPU memory-growth requests made by the code under test."""
        self.memory_growth_calls.append((gpu, enabled))


def make_tf_stub(gpus: Optional[Sequence[str]] = None) -> TensorFlowStub:
    """Create a TensorFlow stub for modules that only need import-time symbols."""
    return TensorFlowStub(gpus=gpus)


def import_project_module(
    module_name: str,
    extra_modules: Optional[Mapping[str, types.ModuleType]] = None,
    clear_modules: Optional[Sequence[str]] = None,
    tensorflow_stub: Optional[TensorFlowStub] = None,
) -> Tuple[types.ModuleType, TensorFlowStub]:
    """Import a project module while temporarily stubbing heavyweight dependencies."""
    extra_modules = extra_modules or {}
    clear_modules = tuple(clear_modules or ())
    tensorflow_stub = tensorflow_stub or make_tf_stub()

    original_modules = {"tensorflow": sys.modules.get("tensorflow")}
    for name, module in extra_modules.items():
        original_modules[name] = sys.modules.get(name)

    sys.modules["tensorflow"] = tensorflow_stub
    for name, module in extra_modules.items():
        sys.modules[name] = module

    for name in (module_name,) + clear_modules:
        sys.modules.pop(name, None)

    try:
        imported_module = importlib.import_module(module_name)
    finally:
        for name, original in original_modules.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original

    return imported_module, tensorflow_stub
