"""Métricas de evaluación de pronóstico.

Incluye lo que el proyecto original no reportaba: MASE, sesgo, error máximo y
porcentaje de cortes ganados. Un promedio bajo puede esconder que el modelo
pierde en la mayoría de los meses.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _arrays(y, yhat):
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    if y.shape != yhat.shape:
        raise ValueError(f"Formas distintas: {y.shape} vs {yhat.shape}")
    m = ~(np.isnan(y) | np.isnan(yhat))
    return y[m], yhat[m]


def mae(y, yhat) -> float:
    y, yhat = _arrays(y, yhat)
    return float(np.mean(np.abs(y - yhat)))


def rmse(y, yhat) -> float:
    y, yhat = _arrays(y, yhat)
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def mape(y, yhat) -> float:
    """MAPE en porcentaje. Indefinido si hay ceros: se excluyen y se avisa."""
    y, yhat = _arrays(y, yhat)
    m = y != 0
    if not m.any():
        return float("nan")
    return float(np.mean(np.abs((y[m] - yhat[m]) / y[m])) * 100)


def wape(y, yhat) -> float:
    """WAPE en porcentaje: suma de errores absolutos sobre suma de observados.

    Preferido sobre MAPE porque no explota con valores pequeños.
    """
    y, yhat = _arrays(y, yhat)
    denom = np.sum(np.abs(y))
    if denom == 0:
        return float("nan")
    return float(np.sum(np.abs(y - yhat)) / denom * 100)


def sesgo(y, yhat) -> float:
    """Error medio con signo. Positivo = el modelo subestima."""
    y, yhat = _arrays(y, yhat)
    return float(np.mean(y - yhat))


def sesgo_relativo(y, yhat) -> float:
    y, yhat = _arrays(y, yhat)
    denom = np.mean(np.abs(y))
    return float(np.mean(y - yhat) / denom * 100) if denom else float("nan")


def error_maximo(y, yhat) -> float:
    y, yhat = _arrays(y, yhat)
    return float(np.max(np.abs(y - yhat))) if y.size else float("nan")


def mase(y, yhat, y_entrenamiento, estacionalidad: int = 1) -> float:
    """MASE escalado por el error naive de la serie de entrenamiento.

    estacionalidad=1 -> naive de un paso; =12 -> naive estacional.
    Un MASE < 1 significa que el modelo supera a esa referencia.
    """
    y, yhat = _arrays(y, yhat)
    ytr = np.asarray(y_entrenamiento, dtype=float)
    ytr = ytr[~np.isnan(ytr)]
    if ytr.size <= estacionalidad:
        return float("nan")
    escala = np.mean(np.abs(ytr[estacionalidad:] - ytr[:-estacionalidad]))
    if escala == 0:
        return float("nan")
    return float(np.mean(np.abs(y - yhat)) / escala)


def cortes_ganados(y, yhat, yhat_referencia) -> float:
    """Porcentaje de cortes en que el modelo tiene menor error absoluto que la referencia."""
    y1, m1 = _arrays(y, yhat)
    _, r1 = _arrays(y, yhat_referencia)
    if y1.size == 0:
        return float("nan")
    return float(np.mean(np.abs(y1 - m1) < np.abs(y1 - r1)) * 100)


def skill_score(y, yhat, yhat_referencia) -> float:
    """Mejora porcentual del MAE frente a la referencia. Positivo = mejor."""
    base = mae(y, yhat_referencia)
    if base == 0:
        return float("nan")
    return float((1 - mae(y, yhat) / base) * 100)


def resumen(y, yhat, *, y_entrenamiento=None, yhat_referencia=None,
            estacionalidad: int = 12) -> dict:
    """Batería completa de métricas para una serie de cortes de backtest."""
    out = {
        "n_cortes": len(_arrays(y, yhat)[0]),
        "wape_pct": wape(y, yhat),
        "mae": mae(y, yhat),
        "rmse": rmse(y, yhat),
        "mape_pct": mape(y, yhat),
        "sesgo": sesgo(y, yhat),
        "sesgo_rel_pct": sesgo_relativo(y, yhat),
        "error_maximo": error_maximo(y, yhat),
    }
    if y_entrenamiento is not None:
        out["mase_1"] = mase(y, yhat, y_entrenamiento, 1)
        out[f"mase_{estacionalidad}"] = mase(y, yhat, y_entrenamiento, estacionalidad)
    if yhat_referencia is not None:
        out["cortes_ganados_pct"] = cortes_ganados(y, yhat, yhat_referencia)
        out["skill_score_pct"] = skill_score(y, yhat, yhat_referencia)
    return out


def tabla_resumen(resultados: dict[str, dict]) -> pd.DataFrame:
    """Convierte {nombre_modelo: resumen} en una tabla ordenada por WAPE."""
    df = pd.DataFrame(resultados).T.reset_index().rename(columns={"index": "modelo"})
    return df.sort_values("wape_pct").reset_index(drop=True)


def diferencia_significativa(y, yhat_a, yhat_b, umbral_relativo: float = 0.05) -> dict:
    """¿La diferencia entre dos modelos es material o es ruido?

    Motivada por el caso del proyecto original: Ridge 7,3403 % contra HGB 7,3470 %
    es una diferencia de 0,007 puntos que no sostiene la afirmación de superioridad.
    """
    wa, wb = wape(y, yhat_a), wape(y, yhat_b)
    dif = wa - wb
    rel = abs(dif) / max(wa, wb) if max(wa, wb) else float("nan")
    ea = np.abs(np.asarray(y, float) - np.asarray(yhat_a, float))
    eb = np.abs(np.asarray(y, float) - np.asarray(yhat_b, float))
    return {
        "wape_a_pct": wa,
        "wape_b_pct": wb,
        "diferencia_pp": dif,
        "diferencia_relativa": rel,
        "cortes_gana_a_pct": float(np.mean(ea < eb) * 100),
        "material": bool(rel >= umbral_relativo),
        "veredicto": ("diferencia material" if rel >= umbral_relativo
                      else "empate técnico: no declarar ganador"),
    }
