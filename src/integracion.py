"""Integración entre dominios: aduanero y portuario.

Clasificación de la relación, siguiendo la regla del proyecto:

* **directa** — existe una llave común y verificable registro a registro;
* **agregada** — las fuentes se relacionan por mes, zona o categoría;
* **contextual** — una fuente explica un periodo, pero no puede unirse.

Entre una declaración de importación y un movimiento de carga portuaria **no
existe llave pública**. La declaración identifica una operación aduanera; el
tráfico portuario identifica toneladas por sociedad portuaria y mes. Por eso la
integración de este proyecto es **agregada por mes**, nunca directa.

Además, los conceptos no son equivalentes:

| | Aduanas (DANE) | Puerto (Supertransporte) |
|---|---|---|
| Unidad | declaración de importación | movimiento de carga |
| Cobertura | ADUA 35, solo importaciones | zona portuaria, todos los flujos |
| Peso | peso neto de la mercancía | toneladas movilizadas, con embalaje |
| Incluye | solo importación | importación, exportación y transbordo |

Comparar peso neto aduanero con toneladas portuarias **no debe hacerse como
igualdad**: se comparan como series complementarias y normalizadas.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def unir_agregado(aduanas: pd.DataFrame, puertos: pd.DataFrame,
                  columna_mes: str = "mes") -> pd.DataFrame:
    """Unión agregada por mes. Devuelve solo los meses presentes en ambas fuentes."""
    a = aduanas.copy()
    p = puertos.copy()
    a[columna_mes] = pd.to_datetime(a[columna_mes])
    p[columna_mes] = pd.to_datetime(p[columna_mes])
    return a.merge(p, on=columna_mes, how="inner").sort_values(columna_mes).reset_index(drop=True)


def reporte_cobertura(aduanas: pd.DataFrame, puertos: pd.DataFrame,
                      columna_mes: str = "mes") -> pd.DataFrame:
    """Embudo de integración: cuántos meses quedan vinculados y cuántos no."""
    a = set(pd.to_datetime(aduanas[columna_mes]))
    p = set(pd.to_datetime(puertos[columna_mes]))
    comunes = a & p
    return pd.DataFrame([{
        "meses_aduanas": len(a), "meses_puertos": len(p),
        "meses_vinculados": len(comunes),
        "pct_aduanas_vinculado": round(len(comunes) / len(a) * 100, 2) if a else 0,
        "pct_puertos_vinculado": round(len(comunes) / len(p) * 100, 2) if p else 0,
        "solo_aduanas": len(a - p), "solo_puertos": len(p - a),
        "periodo_comun": (f"{min(comunes):%Y-%m} a {max(comunes):%Y-%m}" if comunes else ""),
        "tipo_integracion": "agregada por mes",
        "llave": "mes calendario",
        "cardinalidad": "1:1 tras agregar ambos dominios a mensual",
        "riesgo_duplicacion": "nulo: ambas fuentes se agregan antes de unir",
    }])


def matriz_relaciones() -> pd.DataFrame:
    """Declaración explícita de qué se puede unir con qué y cómo."""
    return pd.DataFrame([
        {"origen": "DANE IMPO (declaraciones)", "destino": "Supertransporte (tráfico)",
         "llave": "mes calendario", "tipo": "agregada", "granularidad_comun": "mensual",
         "justificacion": "No existe llave pública entre declaración y movimiento portuario"},
        {"origen": "DANE IMPO", "destino": "TRM", "llave": "mes calendario",
         "tipo": "agregada", "granularidad_comun": "mensual",
         "justificacion": "La TRM diaria se mensualiza con criterio de corte declarado"},
        {"origen": "DANE IMPO", "destino": "ONI", "llave": "mes calendario",
         "tipo": "contextual", "granularidad_comun": "mensual",
         "justificacion": "Indicador climático; explica periodos, no operaciones"},
        {"origen": "DANE IMPO", "destino": "Catálogo de eventos", "llave": "rango de fechas",
         "tipo": "contextual", "granularidad_comun": "periodo",
         "justificacion": "Coincidencia temporal documentada, sin afirmar causalidad"},
        {"origen": "Supertransporte", "destino": "DIMAR (arribos)", "llave": "—",
         "tipo": "no viable", "granularidad_comun": "—",
         "justificacion": "DIMAR solo publica PDF trimestrales sin serie tabular"},
        {"origen": "Cualquiera", "destino": "ETA/ATA/permanencias", "llave": "—",
         "tipo": "no viable", "granularidad_comun": "—",
         "justificacion": "Sin fuente pública histórica; requiere AIS comercial o terminal"},
    ])


def comparar_normalizado(integrado: pd.DataFrame, col_a: str, col_b: str,
                         columna_mes: str = "mes") -> pd.DataFrame:
    """Compara dos series de dominios distintos en escala normalizada (base 100).

    Se normaliza porque las unidades no son equivalentes: comparar kilogramos
    aduaneros con toneladas portuarias en el mismo eje sugeriría una igualdad
    que no existe.
    """
    d = integrado[[columna_mes, col_a, col_b]].dropna().copy()
    for c in (col_a, col_b):
        d[f"{c}_base100"] = d[c] / d[c].iloc[0] * 100
    return d


def correlacion_rezagada(integrado: pd.DataFrame, col_a: str, col_b: str,
                         max_lag: int = 6) -> pd.DataFrame:
    """Correlación cruzada entre dominios sobre series diferenciadas.

    Solo los rezagos >= 1 son utilizables sin fuga. Una correlación no demuestra
    que un dominio cause el otro: ambos responden al mismo comercio subyacente.
    """
    a = pd.to_numeric(integrado[col_a], errors="coerce").diff()
    b = pd.to_numeric(integrado[col_b], errors="coerce").diff()
    filas = []
    for k in range(max_lag + 1):
        br = b.shift(k)
        m = a.notna() & br.notna()
        n = int(m.sum())
        r = float(np.corrcoef(a[m], br[m])[0, 1]) if n > 2 else np.nan
        banda = 1.96 / np.sqrt(n) if n > 2 else np.nan
        filas.append({"lag": k, "correlacion": r, "n": n, "banda_95": banda,
                      "significativa": bool(abs(r) > banda) if n > 2 else None,
                      "usable_sin_fuga": k >= 1})
    out = pd.DataFrame(filas)
    out.attrs["par"] = f"{col_a} ~ {col_b}"
    return out
