"""Modelos y validación walk-forward.

Todos los modelos implementan la misma interfaz mínima (ajustar/predecir) para
que el backtest sea idéntico entre líneas base y modelos de aprendizaje: si el
protocolo de evaluación difiere, la comparación no significa nada.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .. import config
from . import metricas


# ============================================================ líneas base
class LineaBase:
    nombre = "base"

    def ajustar(self, y: pd.Series, X: pd.DataFrame | None = None):
        self.y_ = pd.Series(y).reset_index(drop=True)
        return self

    def predecir(self, X: pd.DataFrame | None = None) -> float:
        raise NotImplementedError


class Naive1(LineaBase):
    """Predice el último valor observado. Referencia mínima absoluta."""
    nombre = "naive_1"

    def predecir(self, X=None) -> float:
        return float(self.y_.iloc[-1])


class NaiveEstacional(LineaBase):
    """Predice el valor del mismo mes del año anterior."""
    nombre = "naive_12"

    def __init__(self, periodo: int = 12):
        self.periodo = periodo

    def predecir(self, X=None) -> float:
        if len(self.y_) < self.periodo:
            return float(self.y_.iloc[-1])
        return float(self.y_.iloc[-self.periodo])


class Drift(LineaBase):
    """Extrapola la pendiente media entre el primer y el último punto."""
    nombre = "drift"

    def predecir(self, X=None) -> float:
        y = self.y_
        if len(y) < 2:
            return float(y.iloc[-1])
        pendiente = (y.iloc[-1] - y.iloc[0]) / (len(y) - 1)
        return float(y.iloc[-1] + pendiente)


# ============================================================ modelos ML
class ModeloSklearn(LineaBase):
    """Envoltura para estimadores de scikit-learn con escalado dentro del fold."""

    def __init__(self, estimador, nombre: str, escalar: bool = True, log: bool = False):
        self.estimador = estimador
        self.nombre = nombre
        self.escalar = escalar
        self.log = log      # decidido por P23, no por costumbre

    def ajustar(self, y: pd.Series, X: pd.DataFrame | None = None):
        from sklearn.preprocessing import StandardScaler
        if X is None:
            raise ValueError(f"{self.nombre} requiere matriz de features")
        self.X_cols_ = list(X.columns)
        Xv = np.asarray(X, dtype=float)
        yv = np.asarray(y, dtype=float)
        if self.log:
            if (yv <= 0).any():
                raise ValueError("No se puede transformar en logaritmo con valores <= 0")
            yv = np.log(yv)
        # El escalador se ajusta SOLO con el entrenamiento de este corte.
        self.escalador_ = StandardScaler().fit(Xv) if self.escalar else None
        Xt = self.escalador_.transform(Xv) if self.escalador_ else Xv
        self.modelo_ = self.estimador.fit(Xt, yv)
        return self

    def predecir(self, X: pd.DataFrame | None = None) -> float:
        Xv = np.asarray(X, dtype=float).reshape(1, -1)
        Xt = self.escalador_.transform(Xv) if self.escalador_ else Xv
        p = float(self.modelo_.predict(Xt)[0])
        return float(np.exp(p)) if self.log else p


def catalogo_modelos(alpha: float = 1.0, log: bool = False) -> dict[str, LineaBase]:
    """Modelos mínimos exigidos por la especificación. ETS y SARIMA son opcionales."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge
    return {
        "naive_1": Naive1(),
        "naive_12": NaiveEstacional(config.ESTACIONALIDAD),
        "drift": Drift(),
        "ridge": ModeloSklearn(Ridge(alpha=alpha, random_state=None), "ridge", log=log),
        "hgb": ModeloSklearn(
            HistGradientBoostingRegressor(random_state=config.SEMILLA, max_iter=200),
            "hgb", escalar=False, log=log),
    }


# ============================================================ backtest
@dataclass
class ResultadoBacktest:
    nombre: str
    detalle: pd.DataFrame          # una fila por corte
    resumen: dict

    def errores(self) -> np.ndarray:
        return (self.detalle["observado"] - self.detalle["prediccion"]).to_numpy(float)


def backtest_walk_forward(y: pd.Series, modelo: LineaBase, *, X: pd.DataFrame | None = None,
                          n_cortes: int = 24, min_entrenamiento: int | None = None,
                          expansiva: bool = True, meses: pd.Series | None = None
                          ) -> ResultadoBacktest:
    """Validación de un paso adelante.

    En cada corte t: se entrena con todo lo anterior a t y se predice t. La
    ventana expansiva usa toda la historia; la móvil usa solo los últimos
    `min_entrenamiento` meses, lo que se adapta mejor a cambios de régimen.
    """
    y = pd.Series(y).reset_index(drop=True)
    n = len(y)
    min_ent = min_entrenamiento or config.MIN_ENTRENAMIENTO
    inicio = n - n_cortes
    if inicio < min_ent:
        raise ValueError(
            f"Serie insuficiente: {n} meses no alcanzan para {n_cortes} cortes "
            f"con {min_ent} meses mínimos de entrenamiento.")

    filas = []
    for t in range(inicio, n):
        ini_ent = 0 if expansiva else max(0, t - min_ent)
        y_ent = y.iloc[ini_ent:t]
        if X is not None:
            X_ent, X_pred = X.iloc[ini_ent:t], X.iloc[[t]]
            modelo.ajustar(y_ent, X_ent)
            pred = modelo.predecir(X_pred)
        else:
            modelo.ajustar(y_ent)
            pred = modelo.predecir()
        filas.append({
            "corte": t,
            "mes": (str(meses.iloc[t])[:7] if meses is not None else t),
            "n_entrenamiento": len(y_ent),
            "observado": float(y.iloc[t]),
            "prediccion": float(pred),
            "error": float(y.iloc[t] - pred),
            "error_abs": abs(float(y.iloc[t] - pred)),
        })

    detalle = pd.DataFrame(filas)
    resumen = metricas.resumen(
        detalle["observado"], detalle["prediccion"],
        y_entrenamiento=y.iloc[:inicio], estacionalidad=config.ESTACIONALIDAD)
    resumen["modelo"] = modelo.nombre
    resumen["ventana"] = n_cortes
    resumen["tipo_ventana"] = "expansiva" if expansiva else "movil"
    return ResultadoBacktest(modelo.nombre, detalle, resumen)


def comparar_modelos(y: pd.Series, modelos: dict[str, LineaBase], *,
                     X: pd.DataFrame | None = None, ventanas=(24,),
                     meses: pd.Series | None = None, expansiva: bool = True
                     ) -> tuple[pd.DataFrame, dict[str, ResultadoBacktest]]:
    """P47, P48 y P50. Backtest de todos los modelos en todas las ventanas."""
    filas, resultados = [], {}
    import time
    for ventana in ventanas:
        for nombre, modelo in modelos.items():
            usa_X = isinstance(modelo, ModeloSklearn)
            try:
                t0 = time.perf_counter()
                res = backtest_walk_forward(
                    y, modelo, X=X if usa_X else None, n_cortes=ventana,
                    meses=meses, expansiva=expansiva)
                res.resumen["segundos"] = round(time.perf_counter() - t0, 3)
                resultados[f"{nombre}_v{ventana}"] = res
                filas.append(res.resumen)
            except ValueError as e:
                filas.append({"modelo": nombre, "ventana": ventana, "error": str(e)})
    tabla = pd.DataFrame(filas)
    # cortes ganados y MASE contra naive_12 de la misma ventana
    for ventana in ventanas:
        ref = resultados.get(f"naive_12_v{ventana}")
        if ref is None:
            continue
        for k, res in resultados.items():
            if not k.endswith(f"_v{ventana}"):
                continue
            gan = metricas.cortes_ganados(
                res.detalle["observado"], res.detalle["prediccion"], ref.detalle["prediccion"])
            skill = metricas.skill_score(
                res.detalle["observado"], res.detalle["prediccion"], ref.detalle["prediccion"])
            m = (tabla["modelo"] == res.nombre) & (tabla["ventana"] == ventana)
            tabla.loc[m, "cortes_ganados_vs_naive12_pct"] = gan
            tabla.loc[m, "skill_vs_naive12_pct"] = skill
    orden = [c for c in ["modelo", "ventana", "tipo_ventana", "n_cortes", "wape_pct",
                         "mae", "rmse", "mape_pct", "mase_1", "mase_12", "sesgo",
                         "sesgo_rel_pct", "error_maximo", "cortes_ganados_vs_naive12_pct",
                         "skill_vs_naive12_pct", "segundos", "error"] if c in tabla.columns]
    return tabla[orden].sort_values(["ventana", "wape_pct"]).reset_index(drop=True), resultados


def ablacion(y: pd.Series, X: pd.DataFrame, conjuntos: dict[str, list[str]], *,
             ventana: int = 24, meses: pd.Series | None = None,
             alpha: float = 1.0, log: bool = False) -> pd.DataFrame:
    """P45. Desempeño por conjunto de variables.

    Una variable externa se conserva solo si mejora de forma estable fuera de
    muestra. Que el algoritmo la acepte no es argumento.
    """
    from sklearn.linear_model import Ridge
    filas = []
    for nombre, cols in conjuntos.items():
        cols = [c for c in cols if c in X.columns]
        if not cols:
            filas.append({"conjunto": nombre, "n_variables": 0, "error": "sin variables"})
            continue
        modelo = ModeloSklearn(Ridge(alpha=alpha), f"ridge_{nombre}", log=log)
        try:
            res = backtest_walk_forward(y, modelo, X=X[cols], n_cortes=ventana, meses=meses)
            fila = {"conjunto": nombre, "n_variables": len(cols)}
            fila.update({k: v for k, v in res.resumen.items() if k != "modelo"})
            filas.append(fila)
        except ValueError as e:
            filas.append({"conjunto": nombre, "n_variables": len(cols), "error": str(e)})
    tabla = pd.DataFrame(filas)
    if "wape_pct" in tabla.columns:
        base = tabla.loc[tabla["conjunto"] == "solo_rezagos", "wape_pct"]
        if not base.empty and pd.notna(base.iloc[0]):
            tabla["ganancia_vs_solo_rezagos_pp"] = base.iloc[0] - tabla["wape_pct"]
    return tabla


def estabilidad(tabla: pd.DataFrame) -> pd.DataFrame:
    """Dispersión del WAPE de cada modelo entre ventanas.

    Un modelo que gana en una ventana y pierde en otra no es mejor: es inestable.
    """
    if "wape_pct" not in tabla.columns:
        return pd.DataFrame()
    g = tabla.groupby("modelo")["wape_pct"].agg(["mean", "std", "min", "max", "count"])
    g["rango_pp"] = g["max"] - g["min"]
    g["estable"] = g["std"].fillna(0) < 1.0
    return g.reset_index().sort_values("mean")
