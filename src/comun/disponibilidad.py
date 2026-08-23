"""Disponibilidad real de las variables y auditoría de fuga: P43 y P44.

Una variable puede referirse al mes anterior y aun así no estar publicada a la
fecha del pronóstico. La DIAN entrega la información de importaciones al DANE
con un plazo de hasta 45 días después del mes de referencia, y el DANE valida y
difunde después. Un rezago de un mes no garantiza disponibilidad.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd


@dataclass
class Variable:
    """Ficha de disponibilidad de una variable predictora."""
    nombre: str
    periodicidad: str                  # diaria, mensual
    dias_publicacion: int              # días tras el fin del mes de referencia
    rezago_usado: int                  # en meses, tal como entra al modelo
    sujeta_revision: bool
    fuente: str
    nota: str = ""

    def disponible_para(self, mes_objetivo: pd.Timestamp) -> bool:
        """¿El dato del mes (objetivo - rezago) está publicado antes de que empiece
        el mes objetivo? Es la condición mínima para usarlo sin fuga."""
        mes_objetivo = pd.Timestamp(mes_objetivo).to_period("M").to_timestamp()
        mes_ref = mes_objetivo - pd.DateOffset(months=self.rezago_usado)
        fin_ref = (mes_ref + pd.offsets.MonthEnd(1))
        publicacion = fin_ref + pd.Timedelta(days=self.dias_publicacion)
        return publicacion < mes_objetivo


# Fichas por defecto. Ajustar dias_publicacion con el calendario oficial vigente.
VARIABLES_POR_DEFECTO: list[Variable] = [
    Variable("cif_usd", "mensual", 45, 1, True,
             "DANE — microdatos IMPO",
             "La DIAN entrega hasta 45 días después del mes de referencia."),
    Variable("peso_neto_kg", "mensual", 45, 1, True, "DANE — microdatos IMPO", ""),
    Variable("trm", "diaria", 0, 1, False,
             "Banco de la República",
             "Disponible el mismo día; se declara el criterio de corte usado."),
    Variable("oni", "mensual", 15, 1, True,
             "NOAA CPC",
             "El ONI se revisa cuando se actualiza la climatología base."),
]


def calendario_disponibilidad(variables: list[Variable] | None = None,
                              meses: pd.DatetimeIndex | None = None) -> pd.DataFrame:
    """P43. Matriz variable × mes objetivo con la disponibilidad efectiva."""
    variables = variables or VARIABLES_POR_DEFECTO
    if meses is None:
        return pd.DataFrame([asdict(v) for v in variables])
    filas = []
    for v in variables:
        for m in meses:
            mes_ref = pd.Timestamp(m) - pd.DateOffset(months=v.rezago_usado)
            fin_ref = mes_ref + pd.offsets.MonthEnd(1)
            filas.append({
                "variable": v.nombre,
                "mes_objetivo": pd.Timestamp(m).strftime("%Y-%m"),
                "mes_referencia": mes_ref.strftime("%Y-%m"),
                "rezago_meses": v.rezago_usado,
                "dias_publicacion": v.dias_publicacion,
                "fecha_publicacion_estimada": (fin_ref + pd.Timedelta(days=v.dias_publicacion)).date(),
                "disponible": v.disponible_para(m),
                "sujeta_revision": v.sujeta_revision,
                "fuente": v.fuente,
            })
    return pd.DataFrame(filas)


def variables_no_disponibles(cal: pd.DataFrame) -> pd.DataFrame:
    """Filas donde la variable NO estaba publicada al momento de predecir."""
    return cal.loc[~cal["disponible"]].copy()


# ---------------------------------------------------------------- P44
def comparar_versiones(v_antigua: pd.DataFrame, v_nueva: pd.DataFrame,
                       columna_mes: str = "mes", columnas: tuple[str, ...] = ("cif_usd",),
                       tolerancia: float = 0.0) -> pd.DataFrame:
    """P44. Diferencias mes a mes entre dos descargas de la misma serie.

    Si el DANE revisa, hay que decidir si se evalúa contra la primera
    publicación (lo que el analista vio) o contra la serie revisada.
    """
    a = v_antigua.set_index(columna_mes)
    b = v_nueva.set_index(columna_mes)
    filas = []
    for c in columnas:
        if c not in a.columns or c not in b.columns:
            continue
        j = pd.concat([a[c].rename("version_antigua"), b[c].rename("version_nueva")],
                      axis=1)
        j["dif_abs"] = j["version_nueva"] - j["version_antigua"]
        j["dif_rel_pct"] = j["dif_abs"] / j["version_antigua"].replace(0, pd.NA) * 100
        j["revisado"] = j["dif_abs"].abs() > tolerancia
        j["variable"] = c
        filas.append(j.reset_index())
    return pd.concat(filas, ignore_index=True) if filas else pd.DataFrame()


# ---------------------------------------------------------------- auditoría
def auditar_features(X: pd.DataFrame, y: pd.Series, columna_mes: pd.Series) -> pd.DataFrame:
    """Prueba de fuga por construcción: ninguna columna de X puede tener una
    correlación perfecta con el objetivo contemporáneo.

    Un |r| ~ 1 con y en el mismo mes es la firma típica de que una variable
    entró sin rezagar.
    """
    filas = []
    for c in X.columns:
        x = pd.to_numeric(X[c], errors="coerce")
        m = x.notna() & pd.to_numeric(y, errors="coerce").notna()
        if m.sum() < 3:
            continue
        r = float(x[m].corr(pd.to_numeric(y, errors="coerce")[m]))
        filas.append({"feature": c, "corr_con_objetivo_contemporaneo": r,
                      "sospecha_fuga": bool(abs(r) > 0.999)})
    return pd.DataFrame(filas)


def informe_fuga(cal: pd.DataFrame, auditoria_X: pd.DataFrame,
                 notas: str = "") -> str:
    """Genera el contenido de docs/auditoria_fuga_temporal.md."""
    no_disp = variables_no_disponibles(cal)
    sospechosas = auditoria_X.loc[auditoria_X["sospecha_fuga"]] if not auditoria_X.empty else auditoria_X
    lineas = [
        "# Auditoría de fuga temporal",
        "",
        f"Generado por `src/disponibilidad.py` el {pd.Timestamp.now():%Y-%m-%d %H:%M}.",
        "",
        "## 1. Disponibilidad de variables (P43)",
        "",
        f"- Filas evaluadas: {len(cal)}",
        f"- Casos con variable NO publicada a la fecha de predicción: {len(no_disp)}",
        "",
    ]
    if len(no_disp):
        lineas += ["Variables comprometidas:", ""]
        for v, sub in no_disp.groupby("variable"):
            lineas.append(f"- `{v}`: {len(sub)} cortes afectados "
                          f"({sub['mes_objetivo'].min()} a {sub['mes_objetivo'].max()})")
        lineas.append("")
    else:
        lineas += ["Ninguna variable se usó antes de su fecha de publicación estimada.", ""]

    lineas += ["## 2. Correlación contemporánea de las features (auditoría estructural)", ""]
    if len(sospechosas):
        lineas.append("**Features con correlación casi perfecta con el objetivo del mismo mes:**")
        lineas.append("")
        for _, r in sospechosas.iterrows():
            lineas.append(f"- `{r['feature']}` (r = {r['corr_con_objetivo_contemporaneo']:.4f})")
        lineas += ["", "Revisar el desplazamiento de estas variables antes de reportar métricas.", ""]
    else:
        lineas += ["Ninguna feature presenta correlación casi perfecta con el objetivo "
                   "contemporáneo.", ""]

    lineas += [
        "## 3. Señal de alarma heredada",
        "",
        "El proyecto original reporta una mejora de 55,5 % sobre la línea base estacional "
        "en CIF. Una diferencia de esa magnitud es posible, pero también es el patrón "
        "característico de una fuga. Si tras esta auditoría la mejora se mantiene, dejar "
        "constancia explícita de las pruebas que la respaldan.",
        "",
    ]
    if notas:
        lineas += ["## 4. Notas del ejecutor", "", notas, ""]
    return "\n".join(lineas)
