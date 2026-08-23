"""Análisis exploratorio: bloques 3 y 4 del EDA (P17 a P32).

Cada función responde a una pregunta identificada y devuelve una tabla. La
interpretación se escribe en el notebook y se registra con el Trazador.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------- P17, P18, P19
def describir_distribucion(s: pd.Series, nombre: str = "") -> pd.DataFrame:
    """Percentiles, asimetría y curtosis. La escala log se usa al graficar."""
    x = pd.to_numeric(s, errors="coerce").dropna()
    if x.empty:
        return pd.DataFrame()
    q = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
    fila = {"variable": nombre or getattr(s, "name", ""), "n": int(x.size),
            "media": float(x.mean()), "desviacion": float(x.std()),
            "asimetria": float(x.skew()), "curtosis": float(x.kurt()),
            "minimo": float(x.min()), "maximo": float(x.max())}
    fila.update({f"p{int(p*100):02d}": float(x.quantile(p)) for p in q})
    return pd.DataFrame([fila])


# ---------------------------------------------------------------- P20
def concentracion_registros(valores: pd.Series, cortes=(0.001, 0.01, 0.05, 0.10)) -> pd.DataFrame:
    """Qué porcentaje del total aportan los registros más grandes."""
    x = pd.to_numeric(valores, errors="coerce").dropna().sort_values(ascending=False)
    total = x.sum()
    filas = []
    for c in cortes:
        k = max(1, int(np.ceil(len(x) * c)))
        filas.append({"top_pct": c * 100, "n_registros": k,
                      "aporte_pct": float(x.iloc[:k].sum() / total * 100) if total else None})
    return pd.DataFrame(filas)


# ---------------------------------------------------------------- P21, P22
def z_robusto(s: pd.Series) -> pd.Series:
    """Z basado en mediana y MAD: no se lo lleva por delante un solo extremo."""
    x = pd.to_numeric(s, errors="coerce")
    mediana = x.median()
    mad = (x - mediana).abs().median()
    if mad == 0 or np.isnan(mad):
        return pd.Series(np.zeros(len(x)), index=x.index)
    return 0.6745 * (x - mediana) / mad


def detectar_extremos(serie: pd.DataFrame, columnas: list[str],
                      columna_mes: str = "mes", umbral_z: float = 3.5,
                      k_iqr: float = 1.5) -> pd.DataFrame:
    """P21. Marca meses extremos por IQR y por z robusto. No elimina nada."""
    filas = []
    for c in columnas:
        if c not in serie.columns:
            continue
        x = pd.to_numeric(serie[c], errors="coerce")
        q1, q3 = x.quantile(0.25), x.quantile(0.75)
        iqr = q3 - q1
        z = z_robusto(x)
        marca = ((x < q1 - k_iqr * iqr) | (x > q3 + k_iqr * iqr) | (z.abs() > umbral_z))
        for idx in serie.index[marca.fillna(False)]:
            filas.append({
                "mes": str(serie.loc[idx, columna_mes])[:7], "variable": c,
                "valor": float(x.loc[idx]), "z_robusto": float(z.loc[idx]),
                "fuera_iqr": bool((x.loc[idx] < q1 - k_iqr * iqr) or (x.loc[idx] > q3 + k_iqr * iqr)),
                "decision": "pendiente de investigar",
            })
    return pd.DataFrame(filas, columns=["mes", "variable", "valor", "z_robusto",
                                        "fuera_iqr", "decision"])


def cruzar_extremos(extremos: pd.DataFrame) -> pd.DataFrame:
    """P22. ¿Un mes extremo lo fue por valor, por cantidad o por ambos?"""
    if extremos.empty:
        return extremos
    piv = (extremos.assign(marca=True)
           .pivot_table(index="mes", columns="variable", values="marca",
                        aggfunc="any", fill_value=False).reset_index())
    cols = [c for c in piv.columns if c != "mes"]
    piv["n_indicadores"] = piv[cols].sum(axis=1)
    piv["tipo"] = np.where(piv["n_indicadores"] > 1, "conjunto", "aislado")
    return piv


# ---------------------------------------------------------------- P23
def heterocedasticidad(serie: pd.DataFrame, columna: str, columna_mes: str = "mes",
                       ventana: int = 12) -> pd.DataFrame:
    """P23. ¿La dispersión crece con el nivel?

    Decide dos cosas: si conviene modelar en logaritmo y si el intervalo de
    predicción debe ser proporcional al nivel en lugar de constante.
    Debe ejecutarse ANTES de la fase de modelado.
    """
    x = pd.to_numeric(serie[columna], errors="coerce")
    nivel = x.rolling(ventana, min_periods=ventana).mean()
    disp = x.rolling(ventana, min_periods=ventana).std()
    cv = disp / nivel
    valido = nivel.notna() & disp.notna()
    corr = float(np.corrcoef(nivel[valido], disp[valido])[0, 1]) if valido.sum() > 2 else np.nan
    # correlación en logaritmos: pendiente ~1 sugiere dispersión proporcional
    pendiente = np.nan
    if valido.sum() > 2 and (nivel[valido] > 0).all() and (disp[valido] > 0).all():
        pendiente = float(np.polyfit(np.log(nivel[valido]), np.log(disp[valido]), 1)[0])
    detalle = pd.DataFrame({
        "mes": serie[columna_mes], "valor": x,
        "nivel_movil": nivel, "dispersion_movil": disp, "cv_movil": cv})
    detalle.attrs["variable"] = columna
    detalle.attrs["corr_nivel_dispersion"] = corr
    detalle.attrs["pendiente_log_log"] = pendiente
    detalle.attrs["recomendacion"] = _recomendacion_heterocedasticidad(corr, pendiente)
    return detalle


def _recomendacion_heterocedasticidad(corr: float, pendiente: float) -> str:
    if np.isnan(corr):
        return "datos insuficientes para decidir"
    if corr > 0.5 and (np.isnan(pendiente) or pendiente > 0.5):
        return ("dispersión crece con el nivel: considerar transformación logarítmica "
                "e intervalo proporcional al nivel")
    if corr < 0.2:
        return "dispersión aproximadamente constante: intervalo aditivo admisible"
    return "evidencia mixta: comparar ambas formas en el backtest"


# ---------------------------------------------------------------- P27, P28, P29
def descomponer_stl(serie: pd.Series, periodo: int = 12) -> pd.DataFrame:
    """P27. Descomposición STL. Requiere statsmodels; si falta, usa medias móviles."""
    s = pd.to_numeric(serie, errors="coerce").dropna()
    try:
        from statsmodels.tsa.seasonal import STL
        res = STL(s, period=periodo, robust=True).fit()
        return pd.DataFrame({"observado": s, "tendencia": res.trend,
                             "estacional": res.seasonal, "residuo": res.resid})
    except Exception:
        tendencia = s.rolling(periodo, center=True, min_periods=periodo).mean()
        detrend = s - tendencia
        estacional = detrend.groupby(detrend.index.month if hasattr(detrend.index, "month")
                                     else np.arange(len(detrend)) % periodo).transform("mean")
        return pd.DataFrame({"observado": s, "tendencia": tendencia,
                             "estacional": estacional,
                             "residuo": s - tendencia - estacional})


def indices_estacionales(serie: pd.DataFrame, columna: str,
                         columna_mes: str = "mes") -> pd.DataFrame:
    """P28. Índice estacional multiplicativo por mes calendario (base 100)."""
    d = serie[[columna_mes, columna]].copy()
    d[columna_mes] = pd.to_datetime(d[columna_mes])
    d["mes_num"] = d[columna_mes].dt.month
    d["valor"] = pd.to_numeric(d[columna], errors="coerce")
    media_global = d["valor"].mean()
    idx = (d.groupby("mes_num")["valor"].mean() / media_global * 100).round(4)
    return pd.DataFrame({"variable": columna, "mes_num": idx.index,
                         "indice_estacional": idx.values,
                         "n_observaciones": d.groupby("mes_num")["valor"].count().values})


def autocorrelaciones(serie: pd.Series, max_lag: int = 24) -> pd.DataFrame:
    """P29. ACF y PACF hasta el rezago indicado, con banda de confianza al 95 %."""
    s = pd.to_numeric(serie, errors="coerce").dropna()
    n = len(s)
    banda = 1.96 / np.sqrt(n) if n else np.nan
    acf_vals = [1.0] + [float(s.autocorr(lag=k)) for k in range(1, max_lag + 1)]
    pacf_vals = [np.nan] * (max_lag + 1)
    try:
        from statsmodels.tsa.stattools import pacf
        p = pacf(s, nlags=min(max_lag, n // 2 - 1), method="ywm")
        pacf_vals[:len(p)] = list(map(float, p))
    except Exception:
        pass
    return pd.DataFrame({"lag": range(max_lag + 1), "acf": acf_vals,
                         "pacf": pacf_vals, "banda_95": banda,
                         "significativo": [abs(v) > banda if not np.isnan(banda) else None
                                           for v in acf_vals]})


# ---------------------------------------------------------------- P30, P31
def puntos_de_cambio(serie: pd.Series, minimo_segmento: int = 12) -> pd.DataFrame:
    """P30. Detección simple de cambio de nivel por búsqueda binaria de la media.

    No pretende sustituir una prueba formal: identifica candidatos para que la
    decisión de ventana móvil o expansiva quede justificada.
    """
    x = pd.to_numeric(serie, errors="coerce").dropna().reset_index(drop=True)
    n = len(x)
    if n < 2 * minimo_segmento:
        return pd.DataFrame(columns=["posicion", "media_antes", "media_despues",
                                     "cambio_pct", "estadistico"])
    filas = []
    for k in range(minimo_segmento, n - minimo_segmento):
        a, b = x[:k], x[k:]
        s2 = (a.var(ddof=1) / len(a)) + (b.var(ddof=1) / len(b))
        t = abs(a.mean() - b.mean()) / np.sqrt(s2) if s2 > 0 else 0.0
        filas.append({"posicion": k, "media_antes": float(a.mean()),
                      "media_despues": float(b.mean()),
                      "cambio_pct": float((b.mean() / a.mean() - 1) * 100) if a.mean() else None,
                      "estadistico": float(t)})
    df = pd.DataFrame(filas).sort_values("estadistico", ascending=False)
    return df.reset_index(drop=True)


def comparar_subperiodos(serie: pd.DataFrame, columna: str, cortes: list[str],
                         columna_mes: str = "mes") -> pd.DataFrame:
    """P31. Media, mediana, desviación y CV por subperiodo declarado."""
    d = serie[[columna_mes, columna]].copy()
    d[columna_mes] = pd.to_datetime(d[columna_mes])
    # Los cortes se ordenan y se descartan los que caen fuera del rango: si llegan
    # desordenados, los subperiodos se solapan y las comparaciones dejan de ser válidas.
    cortes_validos = sorted({pd.Timestamp(c) for c in cortes
                             if d[columna_mes].min() < pd.Timestamp(c) < d[columna_mes].max()})
    limites = [d[columna_mes].min(), *cortes_validos, d[columna_mes].max()]
    filas = []
    for i in range(len(limites) - 1):
        ini, fin = limites[i], limites[i + 1]
        m = (d[columna_mes] >= ini) & (d[columna_mes] < fin if i < len(limites) - 2
                                       else d[columna_mes] <= fin)
        x = pd.to_numeric(d.loc[m, columna], errors="coerce").dropna()
        if x.empty:
            continue
        filas.append({"subperiodo": f"{ini:%Y-%m} a {fin:%Y-%m}", "n_meses": int(x.size),
                      "media": float(x.mean()), "mediana": float(x.median()),
                      "desviacion": float(x.std()),
                      "cv": float(x.std() / x.mean()) if x.mean() else None})
    out = pd.DataFrame(filas)
    if len(out) > 1:
        out["cambio_mediana_pct"] = out["mediana"].pct_change() * 100
    return out
