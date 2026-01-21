# -*- coding: utf-8 -*-
"""
modeling/registry.py

Central place to register/construct models.

This keeps:
- CLI (`models_update.py`)
- UI (`ui.py`)
in sync, and makes it easier to add future models.
"""

from __future__ import annotations

from typing import List

from .base import Model
from .models_fhkq import FhkqModel
from .models_laowang import LaowangModel


def build_models(
    *,
    only: str = "both",
    workers: int = 16,
    laowang_top: int = 200,
    laowang_min_score: float = 0.0,
) -> List[Model]:
    models: List[Model] = []
    o = str(only or "both").strip().lower()
    if o in {"both", "laowang"}:
        models.append(
            LaowangModel(
                top_n=int(laowang_top),
                min_score=float(laowang_min_score),
                workers=int(workers),
            )
        )
    if o in {"both", "fhkq"}:
        models.append(FhkqModel())
    return models

