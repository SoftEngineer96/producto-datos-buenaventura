"""Definiciones centralizadas de KPIs del dashboard."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ImportKpis:
    """KPIs aduaneros para el rango filtrado."""

    cif_total_usd: float
    peso_neto_total_kg: float
    declaraciones_total: int
    observaciones_mensuales: int
    ultimo_mes: pd.Timestamp | None
    variacion_cif_mensual_pct: float | None


def calculate_import_kpis(frame: pd.DataFrame) -> ImportKpis:
    """Calcula KPIs; la variación compara los dos últimos meses seleccionados."""
    ordered = frame.sort_values("mes")
    variation = None
    if len(ordered) >= 2:
        previous = float(ordered.iloc[-2]["cif_usd"])
        current = float(ordered.iloc[-1]["cif_usd"])
        if previous != 0:
            variation = (current / previous - 1) * 100
    return ImportKpis(
        cif_total_usd=float(ordered["cif_usd"].sum()),
        peso_neto_total_kg=float(ordered["peso_neto_kg"].sum()),
        declaraciones_total=int(ordered["n_registros"].sum()),
        observaciones_mensuales=len(ordered),
        ultimo_mes=ordered["mes"].max() if not ordered.empty else None,
        variacion_cif_mensual_pct=variation,
    )


KPI_DEFINITIONS = pd.DataFrame(
    [
        {
            "KPI": "Valor CIF total",
            "Definición": "Suma del valor CIF en dólares corrientes de las declaraciones de importación de ADUA 35.",
            "Fuente": "DANE IMPO · serie_mensual_aduanera_v4.parquet",
        },
        {
            "KPI": "Peso neto",
            "Definición": "Suma del peso neto declarado; se presenta en millones de toneladas.",
            "Fuente": "DANE IMPO · serie_mensual_aduanera_v4.parquet",
        },
        {
            "KPI": "Declaraciones",
            "Definición": "Suma de n_registros de la serie mensual; no equivale a contenedores ni movimientos portuarios.",
            "Fuente": "DANE IMPO · serie_mensual_aduanera_v4.parquet",
        },
        {
            "KPI": "Variación mensual CIF",
            "Definición": "Cambio porcentual entre los dos últimos meses disponibles dentro del filtro.",
            "Fuente": "Cálculo del dashboard sobre DANE IMPO",
        },
        {
            "KPI": "Toneladas portuarias",
            "Definición": "Suma de importación, exportación y transbordo reportada por la zona portuaria.",
            "Fuente": "Supertransporte · dataset 5r3g-zv5z",
        },
    ]
)
