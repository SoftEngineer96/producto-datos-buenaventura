"""Dominio portuario: tráfico de carga de la zona portuaria de Buenaventura.

Fuente: Superintendencia de Transporte, dataset `5r3g-zv5z` de datos.gov.co.
Granularidad original: zona portuaria × sociedad portuaria × tipo de carga × mes.
Unidad: toneladas. Licencia CC BY-SA 4.0.

Advertencia de la propia fuente: las cifras pueden actualizarse cuando una sociedad
portuaria reporta un error de transmisión, de modo que la serie está sujeta a
revisión igual que los microdatos del DANE.
"""
from __future__ import annotations

import pandas as pd

from . import config

# Los movimientos que suman al tráfico total de la zona portuaria.
MOVIMIENTOS = ("sum_importacion", "sum_exportacion", "sum_transbordo")


def cargar(ruta: str | None = None) -> pd.DataFrame:
    """Lee el CSV crudo y construye la columna de mes."""
    ruta = ruta or (config.RAW_PUERTOS / "trafico_portuario_buenaventura.csv")
    d = pd.read_csv(ruta)
    d["mes"] = pd.to_datetime(
        dict(year=d["anno_vigencia"], month=d["mes_vigencia"], day=1))
    for c in MOVIMIENTOS:
        d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0.0)
    return d.sort_values(["mes", "tipo_carga"]).reset_index(drop=True)


def serie_mensual(d: pd.DataFrame) -> pd.DataFrame:
    """Serie mensual total de la zona portuaria, en toneladas.

    `toneladas_totales` suma importación, exportación y transbordo. El transbordo
    se reporta aparte porque **no entra ni sale del país**: es carga que cambia de
    buque. Confundirlo con comercio exterior infla el total.
    """
    g = (d.groupby("mes")[list(MOVIMIENTOS)].sum().reset_index()
         .rename(columns={"sum_importacion": "ton_importacion",
                          "sum_exportacion": "ton_exportacion",
                          "sum_transbordo": "ton_transbordo"}))
    g["toneladas_totales"] = g[["ton_importacion", "ton_exportacion", "ton_transbordo"]].sum(axis=1)
    g["toneladas_comercio_exterior"] = g["ton_importacion"] + g["ton_exportacion"]
    return g


def serie_por_tipo(d: pd.DataFrame) -> pd.DataFrame:
    """Serie mensual por tipo de carga, con participación dentro del mes."""
    g = (d.groupby(["mes", "tipo_carga"])[list(MOVIMIENTOS)].sum().reset_index()
         .rename(columns={"sum_importacion": "ton_importacion",
                          "sum_exportacion": "ton_exportacion",
                          "sum_transbordo": "ton_transbordo"}))
    g["toneladas"] = g[["ton_importacion", "ton_exportacion", "ton_transbordo"]].sum(axis=1)
    total = g.groupby("mes")["toneladas"].transform("sum")
    g["participacion_pct"] = g["toneladas"] / total * 100
    return g


def serie_contenerizada(d: pd.DataFrame) -> pd.DataFrame:
    """Serie de la carga contenerizada, en toneladas.

    **No es TEU.** El dataset publica toneladas por tipo de carga, no unidades de
    contenedor ni TEU. Presentarlo como TEU sería inventar una unidad que la
    fuente no entrega.
    """
    c = d.loc[d["tipo_carga"] == "CONTENEDORES"]
    g = serie_mensual(c).rename(columns={
        "toneladas_totales": "ton_contenerizada",
        "ton_importacion": "ton_cont_importacion",
        "ton_exportacion": "ton_cont_exportacion",
        "ton_transbordo": "ton_cont_transbordo"})
    return g[["mes", "ton_cont_importacion", "ton_cont_exportacion",
              "ton_cont_transbordo", "ton_contenerizada"]]


def continuidad(serie: pd.DataFrame) -> dict:
    """Comprueba que la serie mensual no tenga huecos."""
    s = serie.sort_values("mes")
    esperado = pd.date_range(s["mes"].min(), s["mes"].max(), freq="MS")
    faltan = sorted(set(esperado) - set(s["mes"]))
    return {"meses_esperados": len(esperado), "meses_observados": len(s),
            "meses_faltantes": [f"{m:%Y-%m}" for m in faltan],
            "continua": not faltan}


def control_calidad(d: pd.DataFrame) -> pd.DataFrame:
    """Controles de dominio: negativos, ceros totales y meses incompletos."""
    filas = []
    for c in MOVIMIENTOS:
        filas.append({"variable": c, "n": len(d),
                      "n_negativos": int((d[c] < 0).sum()),
                      "n_ceros": int((d[c] == 0).sum()),
                      "minimo": float(d[c].min()), "maximo": float(d[c].max())})
    por_mes = d.groupby("mes")["tipo_carga"].nunique()
    filas.append({"variable": "tipos_de_carga_por_mes", "n": len(por_mes),
                  "n_negativos": 0,
                  "n_ceros": int((por_mes < len(config.TIPOS_CARGA)).sum()),
                  "minimo": float(por_mes.min()), "maximo": float(por_mes.max())})
    return pd.DataFrame(filas)
