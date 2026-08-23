"""Construcción de variables predictoras sin fuga.

Regla central: toda transformación que dependa de los datos (rezagos, medias
móviles, escaladores, imputadores) se ajusta dentro del conjunto de
entrenamiento de cada corte, nunca sobre la serie completa.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def agregar_rezagos(df: pd.DataFrame, columna: str,
                    lags: tuple[int, ...] = (1, 2, 3, 6, 12)) -> pd.DataFrame:
    out = df.copy()
    for k in lags:
        out[f"{columna}_lag{k}"] = out[columna].shift(k)
    return out


def agregar_medias_moviles(df: pd.DataFrame, columna: str,
                           ventanas: tuple[int, ...] = (3, 6, 12),
                           desplazamiento: int = 1) -> pd.DataFrame:
    """Medias móviles calculadas sobre la serie YA desplazada.

    El desplazamiento de 1 es obligatorio: una media que incluya el mes t no
    puede usarse para predecir el mes t.
    """
    out = df.copy()
    base = out[columna].shift(desplazamiento)
    for v in ventanas:
        out[f"{columna}_ma{v}"] = base.rolling(v, min_periods=v).mean()
        out[f"{columna}_std{v}"] = base.rolling(v, min_periods=v).std()
    return out


def agregar_calendario(df: pd.DataFrame, columna_mes: str = "mes") -> pd.DataFrame:
    """Variables de calendario. No dependen de los datos, no pueden filtrar."""
    out = df.copy()
    fechas = pd.to_datetime(out[columna_mes])
    out["mes_num"] = fechas.dt.month
    out["trimestre"] = fechas.dt.quarter
    out["anio"] = fechas.dt.year
    out["dias_del_mes"] = fechas.dt.days_in_month
    out["tendencia"] = np.arange(len(out))
    # codificación cíclica: diciembre y enero quedan contiguos
    out["mes_sin"] = np.sin(2 * np.pi * out["mes_num"] / 12)
    out["mes_cos"] = np.cos(2 * np.pi * out["mes_num"] / 12)
    return out


def agregar_externa(df: pd.DataFrame, externa: pd.DataFrame, columna: str,
                    lags: tuple[int, ...] = (1, 2, 3), columna_mes: str = "mes") -> pd.DataFrame:
    """Une una variable externa y la rezaga. Nunca se usa el valor contemporáneo."""
    out = df.merge(externa[[columna_mes, columna]], on=columna_mes, how="left")
    for k in lags:
        out[f"{columna}_lag{k}"] = out[columna].shift(k)
    return out.drop(columns=[columna])


# Conjuntos para el análisis de ablación (P45)
def conjuntos_ablacion(columnas: list[str], objetivo: str) -> dict[str, list[str]]:
    """Define los cinco conjuntos de variables que compara P45."""
    rezagos = [c for c in columnas if "_lag" in c and c.startswith(objetivo)]
    moviles = [c for c in columnas if ("_ma" in c or "_std" in c) and c.startswith(objetivo)]
    calendario = [c for c in columnas
                  if c in {"mes_num", "trimestre", "dias_del_mes", "tendencia",
                           "mes_sin", "mes_cos"}]
    trm = [c for c in columnas if c.startswith("trm")]
    oni = [c for c in columnas if c.startswith("oni")]
    return {
        "solo_rezagos": rezagos,
        "rezagos_moviles": rezagos + moviles,
        "rezagos_calendario": rezagos + calendario,
        "rezagos_trm": rezagos + trm,
        "rezagos_oni": rezagos + oni,
        "completo": rezagos + moviles + calendario + trm + oni,
    }


def construir_matriz(serie: pd.DataFrame, objetivo: str, *,
                     lags=(1, 2, 3, 6, 12), ventanas=(3, 6, 12),
                     externas: dict[str, pd.DataFrame] | None = None,
                     columna_mes: str = "mes") -> tuple[pd.DataFrame, pd.Series]:
    """Devuelve (X, y) alineados y sin filas incompletas.

    y es el valor del mes t; todas las columnas de X provienen de t-1 o anterior.
    """
    d = serie[[columna_mes, objetivo]].copy().sort_values(columna_mes).reset_index(drop=True)
    d = agregar_rezagos(d, objetivo, lags)
    d = agregar_medias_moviles(d, objetivo, ventanas, desplazamiento=1)
    d = agregar_calendario(d, columna_mes)
    for nombre, ext in (externas or {}).items():
        d = agregar_externa(d, ext, nombre, lags=(1, 2, 3), columna_mes=columna_mes)

    y = d[objetivo]
    X = d.drop(columns=[objetivo, columna_mes, "anio"], errors="ignore")
    completo = X.notna().all(axis=1) & y.notna()
    X, y = X.loc[completo].reset_index(drop=True), y.loc[completo].reset_index(drop=True)
    X.attrs["mes"] = d.loc[completo, columna_mes].reset_index(drop=True)
    return X, y


def vif(X: pd.DataFrame) -> pd.DataFrame:
    """P46. Factor de inflación de varianza. VIF > 10 indica redundancia fuerte.

    Justifica el uso de Ridge: con rezagos y medias móviles la colinealidad es
    esperable, y la penalización L2 es la respuesta razonable.
    """
    Xn = X.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all").dropna()
    Xn = Xn.astype(float)
    filas = []
    for c in Xn.columns:
        otras = Xn.drop(columns=[c])
        if otras.shape[1] == 0:
            continue
        A = np.column_stack([np.ones(len(otras)), otras.values])
        beta, *_ = np.linalg.lstsq(A, Xn[c].values, rcond=None)
        pred = A @ beta
        ss_res = float(((Xn[c].values - pred) ** 2).sum())
        ss_tot = float(((Xn[c].values - Xn[c].values.mean()) ** 2).sum())
        r2 = 1 - ss_res / ss_tot if ss_tot else 0.0
        filas.append({"variable": c, "r2": r2,
                      "vif": float("inf") if r2 >= 1 else 1 / (1 - r2),
                      "redundante": bool(r2 >= 0.9)})
    return pd.DataFrame(filas).sort_values("vif", ascending=False).reset_index(drop=True)
