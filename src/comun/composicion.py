"""Composición y contribución por país y capítulo: P33 a P39.

Es la parte que convierte un total agregado en algo explicable: dos meses con
el mismo CIF pueden tener composiciones completamente distintas.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def participaciones(df: pd.DataFrame, dimension: str, valor: str,
                    top: int = 15) -> pd.DataFrame:
    """P33 y P35. Participación porcentual y acumulada de cada categoría."""
    g = (df.groupby(dimension, dropna=False)[valor].sum()
         .sort_values(ascending=False).reset_index())
    total = g[valor].sum()
    g["participacion_pct"] = g[valor] / total * 100 if total else np.nan
    g["acumulado_pct"] = g["participacion_pct"].cumsum()
    g["rango"] = np.arange(1, len(g) + 1)
    return g.head(top) if top else g


def hhi(df: pd.DataFrame, dimension: str, valor: str) -> float:
    """Índice de Herfindahl-Hirschman sobre participaciones en porcentaje.

    Escala 0–10.000. Por convención: <1.500 desconcentrado, 1.500–2.500
    moderadamente concentrado, >2.500 concentrado.
    """
    s = df.groupby(dimension, dropna=False)[valor].sum()
    total = s.sum()
    if total <= 0:
        return float("nan")
    return float(((s / total * 100) ** 2).sum())


def hhi_mensual(df: pd.DataFrame, dimension: str, valor: str,
                columna_mes: str = "mes") -> pd.DataFrame:
    """P36. Evolución de la concentración: ¿la canasta se diversifica?"""
    filas = []
    for mes, sub in df.groupby(columna_mes):
        filas.append({"mes": mes, "dimension": dimension, "variable": valor,
                      "hhi": hhi(sub, dimension, valor),
                      "n_categorias": int(sub[dimension].nunique())})
    return pd.DataFrame(filas).sort_values("mes").reset_index(drop=True)


def clasificar_hhi(valor: float) -> str:
    if np.isnan(valor):
        return "sin dato"
    if valor < 1500:
        return "desconcentrado"
    if valor <= 2500:
        return "moderadamente concentrado"
    return "concentrado"


def contribucion_variacion(df: pd.DataFrame, dimension: str, valor: str,
                           columna_mes: str = "mes", mes_actual=None,
                           mes_anterior=None, top: int = 10) -> pd.DataFrame:
    """P37 y P38. Descompone la variación mensual del total por categoría.

    La suma de las contribuciones reproduce exactamente la variación total: es
    lo que permite decir "el total subió y estos tres orígenes lo explican".
    """
    d = df.copy()
    meses = sorted(d[columna_mes].unique())
    if mes_actual is None:
        mes_actual = meses[-1]
    if mes_anterior is None:
        idx = list(meses).index(mes_actual)
        if idx == 0:
            raise ValueError("No hay mes anterior para comparar")
        mes_anterior = meses[idx - 1]

    a = d.loc[d[columna_mes] == mes_anterior].groupby(dimension)[valor].sum()
    b = d.loc[d[columna_mes] == mes_actual].groupby(dimension)[valor].sum()
    comp = pd.concat([a.rename("anterior"), b.rename("actual")], axis=1).fillna(0.0)
    comp["contribucion_abs"] = comp["actual"] - comp["anterior"]
    total_var = comp["contribucion_abs"].sum()
    base = comp["anterior"].sum()
    comp["contribucion_pct_del_total_anterior"] = (
        comp["contribucion_abs"] / base * 100 if base else np.nan)
    comp["participacion_en_la_variacion_pct"] = (
        comp["contribucion_abs"] / total_var * 100 if total_var else np.nan)
    comp = comp.sort_values("contribucion_abs", key=abs, ascending=False).reset_index()
    comp.attrs["mes_anterior"] = str(mes_anterior)[:7]
    comp.attrs["mes_actual"] = str(mes_actual)[:7]
    comp.attrs["variacion_total"] = float(total_var)
    return comp.head(top) if top else comp


def valor_unitario_por_categoria(df: pd.DataFrame, dimension: str,
                                 top: int = 20) -> pd.DataFrame:
    """P34. CIF/kg por categoría, calculado sobre los agregados, no como media de razones."""
    g = df.groupby(dimension, dropna=False).agg(
        cif_usd=("cif_usd", "sum"), peso_neto_kg=("peso_neto_kg", "sum"),
        n_registros=("cif_usd", "size")).reset_index()
    g["cif_kg"] = g["cif_usd"].divide(g["peso_neto_kg"].where(g["peso_neto_kg"] > 0))
    g["participacion_cif_pct"] = g["cif_usd"] / g["cif_usd"].sum() * 100
    g["participacion_peso_pct"] = g["peso_neto_kg"] / g["peso_neto_kg"].sum() * 100
    return g.sort_values("cif_usd", ascending=False).head(top).reset_index(drop=True)


def efecto_mezcla(df: pd.DataFrame, dimension: str, columna_mes: str = "mes",
                  mes_actual=None, mes_anterior=None) -> pd.DataFrame:
    """P34. Separa cuánto del cambio en CIF/kg vino del valor unitario de cada
    categoría y cuánto de un cambio en la mezcla de participaciones.

    Descomposición aditiva clásica:
        efecto_precio  = sum( w_ant * (u_act - u_ant) )
        efecto_mezcla  = sum( (w_act - w_ant) * u_act )
    """
    meses = sorted(df[columna_mes].unique())
    mes_actual = mes_actual if mes_actual is not None else meses[-1]
    if mes_anterior is None:
        mes_anterior = meses[list(meses).index(mes_actual) - 1]

    def _u(m):
        g = df.loc[df[columna_mes] == m].groupby(dimension).agg(
            cif=("cif_usd", "sum"), peso=("peso_neto_kg", "sum"))
        g["u"] = g["cif"].divide(g["peso"].where(g["peso"] > 0))
        g["w"] = g["peso"] / g["peso"].sum()
        return g

    a, b = _u(mes_anterior), _u(mes_actual)
    j = a[["u", "w"]].join(b[["u", "w"]], lsuffix="_ant", rsuffix="_act", how="outer").fillna(0)
    j["efecto_precio"] = j["w_ant"] * (j["u_act"] - j["u_ant"])
    j["efecto_mezcla"] = (j["w_act"] - j["w_ant"]) * j["u_act"]
    j = j.reset_index()
    j.attrs["mes_anterior"] = str(mes_anterior)[:7]
    j.attrs["mes_actual"] = str(mes_actual)[:7]
    return j


def cruce_extremos(df: pd.DataFrame, meses_extremos: list, columna_mes: str = "mes",
                   dim_a: str = "pais_origen", dim_b: str = "capitulo",
                   valor: str = "cif_usd", top: int = 10) -> pd.DataFrame:
    """P39. Tabla cruzada país por capítulo restringida a los meses extremos.

    Ejecutar solo si la granularidad de la fuente lo permite.
    """
    d = df.loc[df[columna_mes].astype(str).str[:7].isin([str(m)[:7] for m in meses_extremos])]
    if d.empty:
        return pd.DataFrame()
    piv = d.pivot_table(index=dim_a, columns=dim_b, values=valor, aggfunc="sum", fill_value=0)
    principales = piv.sum(axis=1).sort_values(ascending=False).head(top).index
    cols = piv.sum(axis=0).sort_values(ascending=False).head(top).index
    return piv.loc[principales, cols].reset_index()
