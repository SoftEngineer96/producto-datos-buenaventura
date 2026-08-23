"""Construcción de las series mensuales: CIF, peso neto y CIF por kilogramo.

El CIF por kilogramo es un valor unitario implícito agregado. No es un precio:
lo afecta la mezcla de mercancías, el seguro y el flete. Se calcula sobre los
agregados mensuales, no como promedio de cocientes por registro, porque el
promedio de razones no equivale a la razón de los totales.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config


def filtrar_aduana(df: pd.DataFrame, adua: int | str | None = None,
                   columna: str = "adua") -> pd.DataFrame:
    """Filtra por código de aduana comparando como texto normalizado.

    ADUA 35 identifica la aduana de Buenaventura como unidad de registro
    administrativo. No equivale a la operación física de la terminal portuaria.
    """
    adua = config.ADUA_OBJETIVO if adua is None else adua
    if columna not in df.columns:
        raise KeyError(f"No existe la columna {columna}")
    objetivo = str(adua).strip().lstrip("0") or "0"
    s = df[columna].astype("string").str.strip().str.lstrip("0").fillna("")
    return df.loc[s == objetivo].copy()


def serie_mensual(df: pd.DataFrame, *, columna_fecha: str = "fecha",
                  por: list[str] | None = None) -> pd.DataFrame:
    """Agrega a mensual. Si `por` se indica, agrega también por esas dimensiones.

    Devuelve cif_usd, peso_neto_kg, n_registros y cif_kg derivado del agregado.
    """
    d = df.copy()
    d[columna_fecha] = pd.to_datetime(d[columna_fecha], errors="coerce")
    d = d.dropna(subset=[columna_fecha])
    d["mes"] = d[columna_fecha].dt.to_period("M").dt.to_timestamp()

    llaves = ["mes"] + (por or [])
    agregados = {}
    for c in ("cif_usd", "fob_usd", "peso_neto_kg", "peso_bruto_kg"):
        if c in d.columns:
            agregados[c] = (c, "sum")
    agregados["n_registros"] = (columna_fecha, "size")

    out = d.groupby(llaves, dropna=False).agg(**agregados).reset_index()
    return agregar_cif_kg(out)


def agregar_cif_kg(df: pd.DataFrame) -> pd.DataFrame:
    """P19. cif_kg = cif_usd / peso_neto_kg, con control de división por cero.

    Los meses sin peso quedan en NaN, nunca en 0 ni en infinito: un valor
    unitario no calculable no es un valor unitario de cero.
    """
    out = df.copy()
    if {"cif_usd", "peso_neto_kg"}.issubset(out.columns):
        peso = pd.to_numeric(out["peso_neto_kg"], errors="coerce")
        cif = pd.to_numeric(out["cif_usd"], errors="coerce")
        out[config.DERIVADA] = cif.divide(peso.where(peso > 0))
        out[config.DERIVADA] = out[config.DERIVADA].replace([np.inf, -np.inf], np.nan)
    return out


def cif_kg_por_registro(df: pd.DataFrame, *, minimo_kg: float = 0.0) -> pd.Series:
    """Valor unitario implícito por registro, para las distribuciones de P19.

    Se excluyen los registros sin peso o con peso <= minimo_kg y se deja
    constancia de cuántos fueron en el atributo `.attrs`.
    """
    peso = pd.to_numeric(df.get("peso_neto_kg"), errors="coerce")
    cif = pd.to_numeric(df.get("cif_usd"), errors="coerce")
    valido = peso.notna() & (peso > minimo_kg) & cif.notna()
    s = (cif[valido] / peso[valido]).replace([np.inf, -np.inf], np.nan)
    s.attrs["excluidos_sin_peso"] = int((~valido).sum())
    s.attrs["pct_excluidos"] = float((~valido).mean() * 100) if len(df) else 0.0
    return s


def asegurar_continuidad(serie: pd.DataFrame, columna_mes: str = "mes") -> pd.DataFrame:
    """Reindexa a un calendario mensual completo y deja NaN donde falte el mes.

    Nunca interpola en silencio: un mes ausente debe verse como ausente.
    """
    s = serie.copy()
    s[columna_mes] = pd.to_datetime(s[columna_mes])
    s = s.sort_values(columna_mes).set_index(columna_mes)
    completo = pd.date_range(s.index.min(), s.index.max(), freq="MS")
    s = s.reindex(completo)
    s.index.name = columna_mes
    return s.reset_index()


def meses_faltantes(serie: pd.DataFrame, columna_mes: str = "mes",
                    columna_valor: str = "cif_usd") -> list[str]:
    s = asegurar_continuidad(serie, columna_mes)
    return list(s.loc[s[columna_valor].isna(), columna_mes].dt.strftime("%Y-%m"))


def media_movil(serie: pd.Series, ventana: int = 12) -> pd.Series:
    """Media móvil centrada en falso: usa solo el pasado, para no mirar adelante."""
    return serie.rolling(ventana, min_periods=ventana).mean()


def resumen_serie(serie: pd.DataFrame, columnas: tuple[str, ...] | None = None) -> pd.DataFrame:
    """Ficha descriptiva de la serie mensual: n, rango, media, mediana, CV."""
    columnas = columnas or (*config.OBJETIVOS, config.DERIVADA)
    filas = []
    for c in columnas:
        if c not in serie.columns:
            continue
        s = pd.to_numeric(serie[c], errors="coerce").dropna()
        filas.append({
            "variable": c, "n_meses": int(s.size),
            "inicio": str(serie["mes"].min())[:7], "fin": str(serie["mes"].max())[:7],
            "media": float(s.mean()), "mediana": float(s.median()),
            "desviacion": float(s.std()),
            "cv": float(s.std() / s.mean()) if s.mean() else None,
            "minimo": float(s.min()), "maximo": float(s.max()),
            "asimetria": float(s.skew()), "curtosis": float(s.kurt()),
        })
    return pd.DataFrame(filas)
