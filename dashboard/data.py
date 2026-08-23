"""Capa de carga y validación de las fuentes reales del dashboard.

Este módulo no contiene reglas de presentación. Prioriza los archivos versionados
en ``data/raw`` para que una clonación limpia funcione también en la nube.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
SURFACE = ROOT / "data" / "surface"
EDA_OUTPUTS = ROOT / "notebooks" / "salidas_eda"


class DataValidationError(ValueError):
    """La estructura de una fuente no permite calcular métricas confiables."""


@dataclass(frozen=True)
class DashboardData:
    """Fuentes verificadas que alimentan las vistas del dashboard."""

    imports_monthly: pd.DataFrame
    port_cargo_monthly: pd.DataFrame
    societies_annual: pd.DataFrame
    societies_by_cargo: pd.DataFrame
    countries_snapshot: pd.DataFrame
    chapters_snapshot: pd.DataFrame
    source_catalog: pd.DataFrame
    seasonality: pd.DataFrame
    container_series: pd.DataFrame
    hhi_annual: pd.DataFrame
    port_extremes: pd.DataFrame
    integrated_correlations: pd.DataFrame
    model_metrics: pd.DataFrame
    ablation: pd.DataFrame
    interval_coverage: pd.DataFrame
    availability: pd.DataFrame
    quality_card: dict


def _require_columns(frame: pd.DataFrame, required: set[str], source: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise DataValidationError(f"{source}: faltan columnas requeridas: {', '.join(missing)}")


def validate_core_quality(
    frame: pd.DataFrame,
    *,
    source: str,
    key: list[str],
    required: list[str],
    nonnegative: list[str],
) -> None:
    """Impide que problemas críticos de calidad lleguen a los KPIs."""
    nulls = frame[required].isna().sum()
    nulls = nulls.loc[nulls > 0]
    if not nulls.empty:
        detail = ", ".join(f"{column}={int(count)}" for column, count in nulls.items())
        raise DataValidationError(f"{source}: valores nulos en campos críticos: {detail}")
    duplicate_keys = int(frame.duplicated(key).sum())
    if duplicate_keys:
        raise DataValidationError(
            f"{source}: {duplicate_keys} llaves duplicadas para {', '.join(key)}"
        )
    negative_count = int((frame[nonnegative] < 0).sum().sum())
    if negative_count:
        raise DataValidationError(
            f"{source}: {negative_count} valores negativos en medidas no negativas"
        )


def load_imports_monthly() -> pd.DataFrame:
    """Carga la serie aduanera mensual agregada de DANE IMPO."""
    path = RAW / "aduanas" / "serie_mensual_aduanera_v4.parquet"
    frame = pd.read_parquet(path)
    _require_columns(
        frame,
        {"mes", "cif_usd", "fob_usd", "peso_neto_kg", "peso_bruto_kg", "n_registros"},
        str(path.relative_to(ROOT)),
    )
    result = frame.copy()
    result["mes"] = (
        pd.to_datetime(result["mes"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    )
    numeric = ["cif_usd", "fob_usd", "peso_neto_kg", "peso_bruto_kg", "n_registros"]
    result[numeric] = result[numeric].apply(pd.to_numeric, errors="coerce")
    result["cif_kg"] = result["cif_usd"].div(result["peso_neto_kg"].replace(0, pd.NA))
    validate_core_quality(
        result,
        source=str(path.relative_to(ROOT)),
        key=["mes"],
        required=["mes", *numeric],
        nonnegative=numeric,
    )
    return result.sort_values("mes").reset_index(drop=True)


def load_port_cargo_monthly() -> pd.DataFrame:
    """Carga importación, exportación y transbordo portuario por mes y carga."""
    path = RAW / "puertos" / "trafico_portuario_buenaventura.csv"
    frame = pd.read_csv(path)
    _require_columns(
        frame,
        {
            "anno_vigencia",
            "mes_vigencia",
            "tipo_carga",
            "sum_importacion",
            "sum_exportacion",
            "sum_transbordo",
        },
        str(path.relative_to(ROOT)),
    )
    result = frame.copy()
    result["mes"] = pd.to_datetime(
        {"year": result["anno_vigencia"], "month": result["mes_vigencia"], "day": 1},
        errors="coerce",
    )
    measures = ["sum_importacion", "sum_exportacion", "sum_transbordo"]
    result[measures] = result[measures].apply(pd.to_numeric, errors="coerce")
    result["toneladas_totales"] = result[measures].sum(axis=1)
    result["toneladas_comercio_exterior"] = result[["sum_importacion", "sum_exportacion"]].sum(
        axis=1
    )
    validate_core_quality(
        result,
        source=str(path.relative_to(ROOT)),
        key=["mes", "tipo_carga"],
        required=["mes", "tipo_carga", *measures],
        nonnegative=measures,
    )
    return result.sort_values(["mes", "tipo_carga"]).reset_index(drop=True)


def load_societies_annual() -> pd.DataFrame:
    """Carga actividad anual por sociedad portuaria."""
    path = RAW / "puertos" / "terminales_por_anio.csv"
    frame = pd.read_csv(path)
    _require_columns(
        frame,
        {
            "anno_vigencia",
            "sociedad_portuaria",
            "sum_importacion",
            "sum_exportacion",
            "sum_transbordo",
        },
        str(path.relative_to(ROOT)),
    )
    result = frame.copy()
    measures = ["sum_importacion", "sum_exportacion", "sum_transbordo"]
    result[measures] = result[measures].apply(pd.to_numeric, errors="coerce")
    result["toneladas_totales"] = result[measures].sum(axis=1)
    validate_core_quality(
        result,
        source=str(path.relative_to(ROOT)),
        key=["anno_vigencia", "sociedad_portuaria"],
        required=["anno_vigencia", "sociedad_portuaria", *measures],
        nonnegative=measures,
    )
    return result.sort_values(["anno_vigencia", "sociedad_portuaria"]).reset_index(drop=True)


def load_societies_by_cargo() -> pd.DataFrame:
    """Carga el cruce histórico sociedad portuaria–tipo de carga."""
    path = RAW / "puertos" / "terminales_por_tipo_carga.csv"
    frame = pd.read_csv(path)
    _require_columns(
        frame,
        {
            "sociedad_portuaria",
            "tipo_carga",
            "sum_importacion",
            "sum_exportacion",
            "sum_transbordo",
        },
        str(path.relative_to(ROOT)),
    )
    result = frame.copy()
    measures = ["sum_importacion", "sum_exportacion", "sum_transbordo"]
    result[measures] = result[measures].apply(pd.to_numeric, errors="coerce")
    result["toneladas_totales"] = result[measures].sum(axis=1)
    validate_core_quality(
        result,
        source=str(path.relative_to(ROOT)),
        key=["sociedad_portuaria", "tipo_carga"],
        required=["sociedad_portuaria", "tipo_carga", *measures],
        nonnegative=measures,
    )
    return result


def _optional_csv(path: Path, columns: list[str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=columns)
    frame = pd.read_csv(path)
    _require_columns(frame, set(columns), str(path.relative_to(ROOT)))
    return frame


def _surface_csv(
    filename: str,
    columns: list[str],
    *,
    dates: tuple[str, ...] = (),
    numeric: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Carga un resultado analítico versionado y normaliza sus tipos."""
    frame = _optional_csv(SURFACE / filename, columns)
    result = frame.copy()
    for column in dates:
        result[column] = pd.to_datetime(result[column], errors="coerce")
    if numeric:
        result[list(numeric)] = result[list(numeric)].apply(pd.to_numeric, errors="coerce")
    return result


def load_dashboard_data() -> DashboardData:
    """Carga todas las fuentes sin acoplarlas a Streamlit."""
    quality_path = SURFACE / "ficha_calidad.json"
    quality_card = (
        json.loads(quality_path.read_text(encoding="utf-8")) if quality_path.exists() else {}
    )
    return DashboardData(
        imports_monthly=load_imports_monthly(),
        port_cargo_monthly=load_port_cargo_monthly(),
        societies_annual=load_societies_annual(),
        societies_by_cargo=load_societies_by_cargo(),
        countries_snapshot=_optional_csv(
            EDA_OUTPUTS / "participacion_pais.csv",
            ["codigo_pais", "participacion_cif_pct", "acumulado_pct"],
        ),
        chapters_snapshot=_optional_csv(
            EDA_OUTPUTS / "participacion_capitulo.csv",
            ["capitulo", "participacion_cif_pct"],
        ),
        source_catalog=_optional_csv(
            SURFACE / "catalogo_fuentes.csv",
            list(pd.read_csv(SURFACE / "catalogo_fuentes.csv", nrows=0).columns)
            if (SURFACE / "catalogo_fuentes.csv").exists()
            else [],
        ),
        seasonality=_surface_csv(
            "estacionalidad_aduanera.csv",
            ["variable", "mes_num", "indice_estacional", "n_observaciones"],
            numeric=("mes_num", "indice_estacional", "n_observaciones"),
        ),
        container_series=_surface_csv(
            "serie_contenerizada.csv",
            [
                "mes",
                "ton_cont_importacion",
                "ton_cont_exportacion",
                "ton_cont_transbordo",
                "ton_contenerizada",
            ],
            dates=("mes",),
            numeric=(
                "ton_cont_importacion",
                "ton_cont_exportacion",
                "ton_cont_transbordo",
                "ton_contenerizada",
            ),
        ),
        hhi_annual=_surface_csv(
            "hhi_anual_sociedades.csv",
            ["anio", "hhi", "sociedades_que_reportan", "periodo_completo"],
            numeric=("anio", "hhi", "sociedades_que_reportan"),
        ),
        port_extremes=_surface_csv(
            "extremos_portuarios.csv",
            ["mes", "variable", "valor", "z_robusto", "fuera_iqr", "decision"],
            dates=("mes",),
            numeric=("valor", "z_robusto"),
        ),
        integrated_correlations=_surface_csv(
            "correlaciones_integradas.csv",
            ["lag", "correlacion", "n", "banda_95", "significativa", "usable_sin_fuga", "par"],
            numeric=("lag", "correlacion", "n", "banda_95"),
        ),
        model_metrics=_surface_csv(
            "metricas_modelos_portuarios.csv",
            [
                "modelo",
                "ventana",
                "tipo_ventana",
                "n_cortes",
                "wape_pct",
                "mae",
                "rmse",
                "mape_pct",
                "mase_1",
                "mase_12",
                "sesgo",
                "sesgo_rel_pct",
                "error_maximo",
                "cortes_ganados_vs_naive12_pct",
                "skill_vs_naive12_pct",
                "segundos",
                "objetivo",
            ],
            numeric=(
                "ventana",
                "n_cortes",
                "wape_pct",
                "mae",
                "rmse",
                "mape_pct",
                "mase_1",
                "mase_12",
                "sesgo",
                "sesgo_rel_pct",
                "error_maximo",
                "cortes_ganados_vs_naive12_pct",
                "skill_vs_naive12_pct",
                "segundos",
            ),
        ),
        ablation=_surface_csv(
            "ablacion_multidominio.csv",
            ["conjunto", "n_vars", "wape_pct", "ganancia_vs_A_pp"],
            numeric=("n_vars", "wape_pct", "ganancia_vs_A_pp"),
        ),
        interval_coverage=_surface_csv(
            "cobertura_intervalos_portuarios.csv",
            [
                "caso",
                "nivel_nominal",
                "cobertura_empirica",
                "ic95_cobertura",
                "n_evaluados",
                "ancho_promedio",
                "ancho_relativo_promedio_pct",
                "casos_fuera",
                "meses_fuera",
                "tipo",
                "metodo",
                "calibrado",
                "veredicto",
                "nombre_honesto",
            ],
            numeric=(
                "nivel_nominal",
                "cobertura_empirica",
                "n_evaluados",
                "ancho_promedio",
                "ancho_relativo_promedio_pct",
                "casos_fuera",
            ),
        ),
        availability=_surface_csv(
            "calendario_disponibilidad.csv",
            ["fuente", "ultimo_mes", "rezago_dias", "disponible_para_pronosticar_mes_siguiente"],
            numeric=("rezago_dias",),
        ),
        quality_card=quality_card,
    )
