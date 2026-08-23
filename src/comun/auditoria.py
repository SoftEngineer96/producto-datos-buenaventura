"""Auditoría de calidad de datos: P02 a P16.

Cada función devuelve un DataFrame listo para guardarse. Ninguna corrige datos
en silencio: separan el diagnóstico de la decisión, que se registra en la
bitácora de exclusiones (P08).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config


# ---------------------------------------------------------------- P02, P08
def embudo_registros(etapas: dict[str, int]) -> pd.DataFrame:
    """P02. Conteo por etapa con pérdida absoluta y relativa frente a la anterior."""
    filas, previo = [], None
    for etapa, n in etapas.items():
        perdida = None if previo is None else previo - n
        filas.append({
            "etapa": etapa,
            "registros": n,
            "perdida_abs": perdida,
            "perdida_pct": (None if previo in (None, 0) else round(perdida / previo * 100, 4)),
        })
        previo = n
    return pd.DataFrame(filas)


class BitacoraExclusiones:
    """P08. Registra cada regla de exclusión y cuántas filas quitó."""

    def __init__(self):
        self._filas: list[dict] = []

    def excluir(self, df: pd.DataFrame, mascara: pd.Series, regla: str,
                justificacion: str, reversible: bool = True) -> pd.DataFrame:
        """Excluye las filas donde `mascara` es True y deja constancia."""
        n = int(mascara.sum())
        self._filas.append({
            "regla": regla,
            "justificacion": justificacion,
            "registros_excluidos": n,
            "pct_del_total": round(n / len(df) * 100, 6) if len(df) else 0.0,
            "reversible": reversible,
        })
        return df.loc[~mascara].copy()

    def a_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self._filas, columns=[
            "regla", "justificacion", "registros_excluidos", "pct_del_total", "reversible"])


# ---------------------------------------------------------------- P03, P04
def perfil_esquemas(dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """P03 y P04. Presencia de columnas y tipo de dato por archivo o vigencia."""
    filas = []
    for nombre, df in dfs.items():
        for col in df.columns:
            filas.append({
                "origen": nombre,
                "columna": col,
                "dtype": str(df[col].dtype),
                "n_filas": len(df),
                "nulos": int(df[col].isna().sum()),
                "unicos": int(df[col].nunique(dropna=True)),
            })
    perfil = pd.DataFrame(filas)
    if perfil.empty:
        return perfil
    conflictos = (perfil.groupby("columna")["dtype"].nunique()
                  .rename("tipos_distintos").reset_index())
    return perfil.merge(conflictos, on="columna", how="left")


# ---------------------------------------------------------------- P05
def cobertura_mensual(fechas: pd.Series) -> pd.DataFrame:
    """P05. Calendario esperado contra observado; marca los meses faltantes."""
    f = pd.to_datetime(pd.Series(fechas).dropna())
    if f.empty:
        return pd.DataFrame(columns=["mes", "observado", "n_registros"])
    periodos = f.dt.to_period("M")
    conteo = periodos.value_counts().sort_index()
    esperado = pd.period_range(periodos.min(), periodos.max(), freq="M")
    return pd.DataFrame({
        "mes": esperado.astype(str),
        "observado": [p in conteo.index for p in esperado],
        "n_registros": [int(conteo.get(p, 0)) for p in esperado],
    })


# ---------------------------------------------------------------- P06
def duplicados_por_capa(df: pd.DataFrame, clave: list[str], capa: str,
                        ejemplos: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """P06. Duplicados exactos y duplicados por clave de negocio."""
    clave = [c for c in clave if c in df.columns]
    dup_exactos = int(df.duplicated().sum())
    dup_clave = int(df.duplicated(subset=clave).sum()) if clave else 0
    resumen = pd.DataFrame([{
        "capa": capa, "n_filas": len(df), "clave": " + ".join(clave),
        "duplicados_exactos": dup_exactos, "duplicados_por_clave": dup_clave,
        "pct_duplicados_clave": round(dup_clave / len(df) * 100, 6) if len(df) else 0.0,
    }])
    muestra = (df.loc[df.duplicated(subset=clave, keep=False)].head(ejemplos)
               if clave and dup_clave else df.head(0))
    return resumen, muestra


# ---------------------------------------------------------------- P07
def validar_codigos(df: pd.DataFrame,
                    longitudes: dict[str, int] | None = None) -> pd.DataFrame:
    """P07. Longitud y formato de los códigos que deben conservar ceros iniciales."""
    longitudes = longitudes or config.CODIGOS_TEXTO
    filas = []
    for col, largo in longitudes.items():
        if col not in df.columns:
            filas.append({"columna": col, "presente": False, "longitud_esperada": largo,
                          "longitud_modal": None, "n_longitud_incorrecta": None,
                          "es_texto": None, "riesgo_ceros_perdidos": None})
            continue
        s = df[col].astype("string").fillna("")
        largos = s.str.len()
        es_texto = pd.api.types.is_string_dtype(df[col]) or df[col].dtype == object
        filas.append({
            "columna": col, "presente": True, "longitud_esperada": largo,
            "longitud_modal": int(largos.mode().iloc[0]) if not largos.empty else None,
            "n_longitud_incorrecta": int((largos != largo).sum()),
            "es_texto": bool(es_texto),
            # si el código viene como número, los ceros iniciales ya se perdieron
            "riesgo_ceros_perdidos": bool(not es_texto),
        })
    return pd.DataFrame(filas)


# ---------------------------------------------------------------- P09, P10, P15
def completitud(df: pd.DataFrame, columnas: list[str],
                por: str = "fecha") -> pd.DataFrame:
    """P09. Porcentaje de nulos por variable y periodo."""
    cols = [c for c in columnas if c in df.columns]
    if por not in df.columns:
        return pd.DataFrame({"variable": cols,
                             "pct_nulos": [df[c].isna().mean() * 100 for c in cols]})
    g = df.assign(_mes=pd.to_datetime(df[por]).dt.to_period("M").astype(str)).groupby("_mes")
    out = (g[cols].apply(lambda x: x.isna().mean() * 100)
           .reset_index().melt(id_vars="_mes", var_name="variable", value_name="pct_nulos"))
    return out.rename(columns={"_mes": "mes"})


def valores_fuera_de_dominio(df: pd.DataFrame,
                             reglas: dict[str, tuple[float, float]] | None = None) -> pd.DataFrame:
    """P10. Negativos, ceros e imposibles según reglas de rango declaradas."""
    reglas = reglas or {
        "cif_usd": (0, np.inf), "fob_usd": (0, np.inf),
        "peso_neto_kg": (0, np.inf), "peso_bruto_kg": (0, np.inf),
    }
    filas = []
    for col, (lo, hi) in reglas.items():
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        filas.append({
            "variable": col, "minimo_permitido": lo, "maximo_permitido": hi,
            "n_negativos": int((s < 0).sum()), "n_ceros": int((s == 0).sum()),
            "n_fuera_rango": int(((s < lo) | (s > hi)).sum()),
            "n_no_numericos": int(s.isna().sum() - df[col].isna().sum()),
            "minimo_observado": float(s.min()) if s.notna().any() else None,
            "maximo_observado": float(s.max()) if s.notna().any() else None,
        })
    return pd.DataFrame(filas)


def codigos_no_identificados(df: pd.DataFrame,
                             columnas: tuple[str, ...] = ("pais_origen", "subpartida", "capitulo"),
                             marcadores: tuple[str, ...] = ("", "0", "00", "000", "999", "ZZ", "NA")) -> pd.DataFrame:
    """P15. Proporción de registros con código ausente o desconocido."""
    filas = []
    for col in columnas:
        if col not in df.columns:
            continue
        s = df[col].astype("string")
        desconocido = s.isna() | s.str.strip().isin(marcadores)
        filas.append({
            "columna": col, "n_desconocidos": int(desconocido.sum()),
            "pct_desconocidos": round(desconocido.mean() * 100, 4),
        })
    return pd.DataFrame(filas)


# ---------------------------------------------------------------- P11, P12
def coherencia_cif_fob(df: pd.DataFrame) -> pd.DataFrame:
    """P11. Razón CIF/FOB. Debe ser >= 1: el CIF incluye seguro y flete."""
    if not {"cif_usd", "fob_usd"}.issubset(df.columns):
        return pd.DataFrame()
    cif = pd.to_numeric(df["cif_usd"], errors="coerce")
    fob = pd.to_numeric(df["fob_usd"], errors="coerce")
    razon = cif.divide(fob.where(fob > 0))
    return pd.DataFrame([{
        "n_comparables": int(razon.notna().sum()),
        "razon_p01": float(razon.quantile(0.01)) if razon.notna().any() else None,
        "razon_mediana": float(razon.median()) if razon.notna().any() else None,
        "razon_p99": float(razon.quantile(0.99)) if razon.notna().any() else None,
        "n_cif_menor_fob": int((razon < 1).sum()),
        "pct_cif_menor_fob": round((razon < 1).mean() * 100, 4) if razon.notna().any() else None,
        "n_fob_cero_o_nulo": int((fob.isna() | (fob <= 0)).sum()),
    }])


def coherencia_pesos(df: pd.DataFrame) -> pd.DataFrame:
    """P12. El peso bruto debe ser >= al peso neto."""
    if not {"peso_neto_kg", "peso_bruto_kg"}.issubset(df.columns):
        return pd.DataFrame()
    neto = pd.to_numeric(df["peso_neto_kg"], errors="coerce")
    bruto = pd.to_numeric(df["peso_bruto_kg"], errors="coerce")
    razon = bruto.divide(neto.where(neto > 0))
    return pd.DataFrame([{
        "n_comparables": int(razon.notna().sum()),
        "razon_mediana": float(razon.median()) if razon.notna().any() else None,
        "n_bruto_menor_neto": int((bruto < neto).sum()),
        "pct_bruto_menor_neto": round((bruto < neto).mean() * 100, 4),
        "n_neto_cero": int((neto == 0).sum()),
        # una razón cercana a 1000 sugiere mezcla de kg con toneladas
        "sospecha_unidad_mixta": bool(((razon > 100) | (razon < 0.01)).sum() > 0),
    }])


# ---------------------------------------------------------------- P16
def reconciliar_totales(serie_propia: pd.Series, serie_oficial: pd.Series,
                        tolerancia: float | None = None) -> pd.DataFrame:
    """P16. Diferencia contra el boletín oficial con tolerancia declarada de antemano.

    La tolerancia se fija en config.TOLERANCIA_RECONCILIACION ANTES de ejecutar,
    para no elegirla después de ver el resultado.
    """
    tol = config.TOLERANCIA_RECONCILIACION if tolerancia is None else tolerancia
    comp = pd.DataFrame({"propio": serie_propia, "oficial": serie_oficial}).dropna()
    comp["dif_abs"] = comp["propio"] - comp["oficial"]
    comp["dif_rel"] = comp["dif_abs"] / comp["oficial"].replace(0, np.nan)
    comp["dentro_tolerancia"] = comp["dif_rel"].abs() <= tol
    comp["tolerancia_declarada"] = tol
    return comp.reset_index().rename(columns={"index": "mes"})
