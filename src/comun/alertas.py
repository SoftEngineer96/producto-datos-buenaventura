"""Reglas de alerta: de una cifra a una señal accionable (P52).

Una predicción sola no dice qué hacer. Estas reglas responden: ¿el próximo mes
se aparta lo suficiente del comportamiento reciente como para activar una
revisión por país, producto o capítulo?

Dos advertencias de diseño:
  1. Los umbrales se calibran SOLO con la ventana de entrenamiento. Calibrarlos
     con toda la serie es circular.
  2. Una alerta explica su razón y no constituye una orden operativa.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

NIVELES = ("normal", "seguimiento", "alerta")


@dataclass
class ReglasAlerta:
    """Umbrales sobre la VARIACIÓN, no sobre el nivel.

    Comparar el pronóstico contra la mediana histórica del nivel no funciona en una
    serie con tendencia: cualquier mes reciente queda en el percentil 100 y a muchas
    desviaciones de una mediana calculada sobre catorce años. La alerta se dispararía
    siempre y dejaría de informar. Por eso la referencia es la distribución de las
    variaciones intermensual e interanual, que sí es estacionaria.
    """
    z_seguimiento: float = 1.5
    z_alerta: float = 2.5
    variacion_seguimiento_pct: float = 10.0
    variacion_alerta_pct: float = 20.0
    mediana_variacion_anual: float = np.nan
    mad_variacion_anual: float = np.nan
    mediana_entrenamiento: float = np.nan
    mad_entrenamiento: float = np.nan
    periodo_calibracion: str = ""
    nota: str = ("Umbrales calibrados únicamente con la ventana de entrenamiento. "
                 "Una alerta es una señal de revisión analítica, no una orden operativa.")

    def a_json(self, ruta=None) -> str:
        texto = json.dumps(asdict(self), ensure_ascii=False, indent=2, default=str)
        if ruta:
            from pathlib import Path
            Path(ruta).write_text(texto, encoding="utf-8")
        return texto


def calibrar(serie_entrenamiento: pd.Series, columna_mes: pd.Series | None = None,
             **kwargs) -> ReglasAlerta:
    """Calcula la referencia robusta con la que se comparará el pronóstico."""
    x = pd.to_numeric(serie_entrenamiento, errors="coerce").dropna()
    variacion = x.pct_change().dropna() * 100
    anual = (x.pct_change(12).dropna() * 100) if len(x) > 12 else pd.Series(dtype=float)
    r = ReglasAlerta(**kwargs)
    r.mediana_entrenamiento = float(x.median())
    r.mad_entrenamiento = float((x - x.median()).abs().median())
    if len(anual) > 8:
        r.mediana_variacion_anual = float(anual.median())
        r.mad_variacion_anual = float((anual - anual.median()).abs().median())
    if len(variacion) > 8:
        r.variacion_seguimiento_pct = float(variacion.abs().quantile(0.75))
        r.variacion_alerta_pct = float(variacion.abs().quantile(0.90))
    if columna_mes is not None and len(columna_mes):
        r.periodo_calibracion = f"{str(columna_mes.iloc[0])[:7]} a {str(columna_mes.iloc[-1])[:7]}"
    return r


def evaluar(prediccion: float, historia: pd.Series, reglas: ReglasAlerta, *,
            limite_inferior: float = np.nan, limite_superior: float = np.nan,
            estacionalidad: int = 12) -> dict:
    """Clasifica el pronóstico y explica por qué.

    Compara contra cuatro referencias distintas, porque una cifra aislada no le
    dice nada al usuario: mes anterior, mismo mes del año anterior, media móvil
    de 12 meses y posición en la distribución histórica.
    """
    h = pd.to_numeric(historia, errors="coerce").dropna()
    if h.empty:
        raise ValueError("Historia vacía: no hay contra qué comparar")

    ultimo = float(h.iloc[-1])
    mismo_mes_anterior = float(h.iloc[-estacionalidad]) if len(h) >= estacionalidad else np.nan
    media_movil = float(h.iloc[-estacionalidad:].mean()) if len(h) >= estacionalidad else float(h.mean())

    var_mes = (prediccion / ultimo - 1) * 100 if ultimo else np.nan
    var_anio = (prediccion / mismo_mes_anterior - 1) * 100 if mismo_mes_anterior else np.nan
    dif_ma = (prediccion / media_movil - 1) * 100 if media_movil else np.nan
    percentil = float((h < prediccion).mean() * 100)

    # z sobre la variación interanual, que es estacionaria. Calcularlo sobre el
    # nivel haría que toda la serie reciente pareciera anómala por la tendencia.
    mad = reglas.mad_variacion_anual
    z = ((0.6745 * (var_anio - reglas.mediana_variacion_anual) / mad)
         if mad and not np.isnan(var_anio) else np.nan)

    razones, nivel = [], "normal"

    def _subir(a: str, motivo: str):
        nonlocal nivel
        if NIVELES.index(a) > NIVELES.index(nivel):
            nivel = a
        razones.append(motivo)

    if not np.isnan(z):
        if abs(z) >= reglas.z_alerta:
            _subir("alerta", f"el crecimiento interanual previsto ({var_anio:+.1f} %) se aparta "
                             f"{abs(z):.1f} desviaciones robustas de su comportamiento típico "
                             f"({reglas.mediana_variacion_anual:+.1f} %)")
        elif abs(z) >= reglas.z_seguimiento:
            _subir("seguimiento", f"el crecimiento interanual previsto ({var_anio:+.1f} %) se aparta "
                                  f"{abs(z):.1f} desviaciones robustas de su comportamiento típico")
    if not np.isnan(var_mes):
        if abs(var_mes) >= reglas.variacion_alerta_pct:
            _subir("alerta", f"variación de {var_mes:+.1f} % frente al mes anterior")
        elif abs(var_mes) >= reglas.variacion_seguimiento_pct:
            _subir("seguimiento", f"variación de {var_mes:+.1f} % frente al mes anterior")
    if not np.isnan(dif_ma) and abs(dif_ma) >= reglas.variacion_alerta_pct:
        _subir("seguimiento", f"se aparta {dif_ma:+.1f} % de la media móvil de doce meses")
    fuera_del_intervalo = (
        not np.isnan(limite_inferior) and not np.isnan(limite_superior)
        and (prediccion < limite_inferior or prediccion > limite_superior))
    if fuera_del_intervalo:
        _subir("alerta", "el pronóstico cae fuera de su propio intervalo de predicción")

    if not razones:
        razones.append("el pronóstico se mantiene dentro del comportamiento histórico esperado")

    return {
        "prediccion": float(prediccion),
        "limite_inferior": float(limite_inferior) if not np.isnan(limite_inferior) else None,
        "limite_superior": float(limite_superior) if not np.isnan(limite_superior) else None,
        "nivel": nivel,
        "razones": razones,
        "variacion_vs_mes_anterior_pct": None if np.isnan(var_mes) else round(var_mes, 2),
        "variacion_vs_mismo_mes_anio_anterior_pct": None if np.isnan(var_anio) else round(var_anio, 2),
        "diferencia_vs_media_movil_12_pct": None if np.isnan(dif_ma) else round(dif_ma, 2),
        "percentil_historico": round(percentil, 1),
        "percentil_ultimos_24m": round(float((h.iloc[-24:] < prediccion).mean() * 100), 1),
        "z_robusto": None if np.isnan(z) else round(float(z), 2),
        "descargo": ("Señal de revisión analítica. No constituye una orden operativa "
                     "ni una medición de la operación física del puerto."),
    }


def matriz_decision() -> pd.DataFrame:
    """Tabla que se muestra al usuario para que entienda qué hacer con cada nivel."""
    return pd.DataFrame([
        {"nivel": "normal", "significado": "dentro del comportamiento histórico esperado",
         "accion_sugerida": "seguimiento de rutina del informe mensual"},
        {"nivel": "seguimiento", "significado": "variación relevante pero dentro del rango histórico",
         "accion_sugerida": "revisar composición por país y capítulo antes de cerrar el informe"},
        {"nivel": "alerta", "significado": "desviación fuera del umbral definido o fuera del intervalo",
         "accion_sugerida": "revisión dirigida por país y capítulo, y verificación de la fuente"},
    ])
