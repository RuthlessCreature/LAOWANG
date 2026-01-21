# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

import pandas as pd


class Model(Protocol):
    name: str

    def ensure_tables(self, engine) -> None:  # noqa: ANN001
        ...

    def compute(self, *, engine, trade_date: str, workers: int) -> pd.DataFrame:  # noqa: ANN001
        ...

    def save(self, *, engine, trade_date: str, df: pd.DataFrame) -> int:  # noqa: ANN001
        ...


@dataclass(frozen=True)
class RunResult:
    model_name: str
    trade_date: str
    status: str
    row_count: int
    message: str = ""
    started_at: Optional[str] = None
    finished_at: Optional[str] = None

