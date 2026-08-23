"""Intervalos de predicción y medición de su cobertura (P51).

Prohibición explícita del proyecto: un intervalo NUNCA se deriva del WAPE ni de
ninguna métrica de error puntual. Se construye con los errores fuera de muestra
del backtest y su cobertura se mide, no se declara.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config


def intervalo_por_cuantiles(errores: np.ndarray, prediccion: float,
                            nivel: float | None = None) -> tuple[float, float]:
    """Intervalo aditivo: cuantiles empíricos de los errores fuera de muestra.

    Apropiado cuando P23 concluye que la dispersión NO crece con el nivel.
    """
    nivel = config.NIVEL_NOMINAL if nivel is None else nivel
    e = np.asarray(errores, float)
    e = e[~np.isnan(e)]
    if e.size < 8:
        return (np.nan, np.nan)
    alfa = (1 - nivel) / 2
    return (float(prediccion + np.quantile(e, alfa)),
            float(prediccion + np.quantile(e, 1 - alfa)))


def intervalo_proporcional(errores_rel: np.ndarray, prediccion: float,
                           nivel: float | None = None) -> tuple[float, float]:
    """Intervalo multiplicativo: cuantiles de los errores relativos.

    Apropiado cuando P23 concluye que la dispersión crece con el nivel.
    """
    nivel = config.NIVEL_NOMINAL if nivel is None else nivel
    e = np.asarray(errores_rel, float)
    e = e[~np.isnan(e)]
    if e.size < 8:
        return (np.nan, np.nan)
    alfa = (1 - nivel) / 2
    return (float(prediccion * (1 + np.quantile(e, alfa))),
            float(prediccion * (1 + np.quantile(e, 1 - alfa))))


def intervalos_conformales(detalle: pd.DataFrame, nivel: float | None = None,
                           minimo_calibracion: int = 12,
                           proporcional: bool = False) -> pd.DataFrame:
    """Construye intervalos de forma honesta: para el corte t solo se usan los
    errores de los cortes anteriores a t (calibración expansiva).

    Usar todos los errores del backtest, incluido el del propio corte, infla la
    cobertura medida y produce un intervalo que parece mejor calibrado de lo
    que está.
    """
    nivel = config.NIVEL_NOMINAL if nivel is None else nivel
    d = detalle.sort_values("corte").reset_index(drop=True).copy()
    obs, pred = d["observado"].to_numpy(float), d["prediccion"].to_numpy(float)
    err = obs - pred
    err_rel = np.divide(err, pred, out=np.full_like(err, np.nan), where=pred != 0)

    lo, hi = np.full(len(d), np.nan), np.full(len(d), np.nan)
    for t in range(len(d)):
        if t < minimo_calibracion:
            continue
        if proporcional:
            lo[t], hi[t] = intervalo_proporcional(err_rel[:t], pred[t], nivel)
        else:
            lo[t], hi[t] = intervalo_por_cuantiles(err[:t], pred[t], nivel)

    d["limite_inferior"] = lo
    d["limite_superior"] = hi
    dentro = np.where(np.isnan(lo), np.nan, ((obs >= lo) & (obs <= hi)).astype(float))
    d["dentro_intervalo"] = dentro          # float con NaN donde no hay calibración
    d["ancho"] = hi - lo
    d["ancho_relativo_pct"] = d["ancho"] / pred * 100
    d["nivel_nominal"] = nivel
    d["metodo"] = "cuantiles empíricos de errores fuera de muestra, calibración expansiva"
    d["tipo"] = "proporcional" if proporcional else "aditivo"
    return d


def medir_cobertura(intervalos: pd.DataFrame, nivel: float | None = None) -> dict:
    """P51. Cobertura empírica, ancho promedio y casos fuera.

    Si la cobertura empírica no alcanza la nominal, el intervalo se recalibra o
    se renombra según lo que realmente cubre. No se deja el rótulo del 80 %
    sobre un intervalo que cubre el 60 %.
    """
    nivel = config.NIVEL_NOMINAL if nivel is None else nivel
    d = intervalos.dropna(subset=["dentro_intervalo"])
    n = len(d)
    if n == 0:
        return {"n_evaluados": 0, "cobertura_empirica": np.nan,
                "veredicto": "sin cortes suficientes para calibrar"}
    cob = float(d["dentro_intervalo"].mean())
    # error estándar binomial, para no sobreinterpretar con pocos cortes
    ee = float(np.sqrt(cob * (1 - cob) / n)) if n else np.nan
    fuera = d.loc[~d["dentro_intervalo"].astype(bool)]
    return {
        "n_evaluados": n,
        "nivel_nominal": nivel,
        "cobertura_empirica": cob,
        "error_estandar": ee,
        "ic95_cobertura": (max(0.0, cob - 1.96 * ee), min(1.0, cob + 1.96 * ee)),
        "ancho_promedio": float(d["ancho"].mean()),
        "ancho_relativo_promedio_pct": float(d["ancho_relativo_pct"].mean()),
        "casos_fuera": len(fuera),
        "meses_fuera": list(fuera["mes"].astype(str)),
        "metodo": str(d["metodo"].iloc[0]),
        "tipo": str(d["tipo"].iloc[0]),
        "calibrado": bool(cob + 1.96 * ee >= nivel),
        "veredicto": _veredicto(cob, nivel, ee),
        "nombre_honesto": f"intervalo con cobertura empírica del {cob*100:.0f} %",
    }


def _veredicto(cob: float, nivel: float, ee: float) -> str:
    if cob + 1.96 * ee < nivel:
        return (f"subcobertura: cubre {cob*100:.1f} % frente al {nivel*100:.0f} % declarado. "
                "Recalibrar o renombrar el intervalo.")
    if cob - 1.96 * ee > nivel + 0.10:
        return (f"sobrecobertura: cubre {cob*100:.1f} %. El intervalo es más ancho de lo "
                "necesario y comunica menos precisión de la que hay.")
    return f"cobertura compatible con el nivel declarado ({cob*100:.1f} %)."


def tabla_cobertura(resultados: dict[str, pd.DataFrame], nivel: float | None = None) -> pd.DataFrame:
    """Consolida la cobertura por objetivo y ventana en un solo archivo."""
    filas = []
    for clave, inter in resultados.items():
        m = medir_cobertura(inter, nivel)
        m["caso"] = clave
        m["meses_fuera"] = "; ".join(m.get("meses_fuera", []))
        m["ic95_cobertura"] = f"[{m['ic95_cobertura'][0]:.3f}, {m['ic95_cobertura'][1]:.3f}]" \
            if isinstance(m.get("ic95_cobertura"), tuple) else ""
        filas.append(m)
    cols = ["caso", "nivel_nominal", "cobertura_empirica", "ic95_cobertura", "n_evaluados",
            "ancho_promedio", "ancho_relativo_promedio_pct", "casos_fuera", "meses_fuera",
            "tipo", "metodo", "calibrado", "veredicto", "nombre_honesto"]
    df = pd.DataFrame(filas)
    return df[[c for c in cols if c in df.columns]]
