"""Capa de transformación y controles de calidad para el dashboard."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date, datetime

import pandas as pd


def filter_dates(
    frame: pd.DataFrame,
    start: date | datetime | pd.Timestamp,
    end: date | datetime | pd.Timestamp,
    column: str = "mes",
) -> pd.DataFrame:
    """Filtra inclusivamente un rango temporal y devuelve una copia."""
    start_ts = pd.Timestamp(start).to_period("M").to_timestamp()
    end_ts = pd.Timestamp(end).to_period("M").to_timestamp()
    mask = frame[column].between(start_ts, end_ts, inclusive="both")
    return frame.loc[mask].copy()


def filter_values(frame: pd.DataFrame, column: str, values: Iterable[str]) -> pd.DataFrame:
    selected = list(values)
    return frame.loc[frame[column].isin(selected)].copy() if selected else frame.iloc[0:0].copy()


def port_monthly(frame: pd.DataFrame) -> pd.DataFrame:
    """Agrega las filas por carga a un grano mensual único."""
    measures = [
        "sum_importacion",
        "sum_exportacion",
        "sum_transbordo",
        "toneladas_totales",
        "toneladas_comercio_exterior",
    ]
    return frame.groupby("mes", as_index=False)[measures].sum().sort_values("mes")


def quality_profile(
    name: str,
    frame: pd.DataFrame,
    key: list[str],
    date_column: str | None,
    measures: list[str],
) -> dict[str, object]:
    """Calcula controles estables de completitud, unicidad, fechas y dominio."""
    null_cells = int(frame.isna().sum().sum())
    duplicated_rows = int(frame.duplicated().sum())
    duplicated_keys = int(frame.duplicated(key).sum())
    invalid_dates = int(frame[date_column].isna().sum()) if date_column else 0
    negative_values = int((frame[measures] < 0).sum().sum()) if measures else 0
    return {
        "dataset": name,
        "filas": len(frame),
        "columnas": len(frame.columns),
        "celdas_nulas": null_cells,
        "filas_duplicadas": duplicated_rows,
        "llaves_duplicadas": duplicated_keys,
        "fechas_invalidas": invalid_dates,
        "valores_negativos": negative_values,
        "estado": "Aprobado"
        if not any([null_cells, duplicated_rows, duplicated_keys, invalid_dates, negative_values])
        else "Revisar",
    }


def missing_months(frame: pd.DataFrame, column: str = "mes") -> list[pd.Timestamp]:
    """Devuelve meses ausentes entre el mínimo y máximo observado."""
    observed = pd.DatetimeIndex(frame[column].dropna().unique()).to_period("M")
    if observed.empty:
        return []
    expected = pd.period_range(observed.min(), observed.max(), freq="M")
    return [period.to_timestamp() for period in expected.difference(observed)]
