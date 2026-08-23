"""Variables externas TRM y ONI y su relación con los objetivos: P40 a P42.

Advertencia permanente: una correlación cruzada no demuestra causalidad. Estas
variables se conservan solo si aportan de forma estable fuera de muestra (P45).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def mensualizar_trm(trm_diaria: pd.DataFrame, columna_fecha: str = "fecha",
                    columna_valor: str = "trm", corte: str = "fin_de_mes") -> pd.DataFrame:
    """Convierte la TRM diaria a mensual con un criterio de corte explícito.

    El criterio importa para la fuga: 'fin_de_mes' usa el último dato del mes,
    que está disponible el mismo día; 'promedio' usa el promedio del mes.
    """
    d = trm_diaria.copy()
    d[columna_fecha] = pd.to_datetime(d[columna_fecha])
    d["mes"] = d[columna_fecha].dt.to_period("M").dt.to_timestamp()
    if corte == "promedio":
        out = d.groupby("mes")[columna_valor].mean()
    elif corte == "fin_de_mes":
        out = d.sort_values(columna_fecha).groupby("mes")[columna_valor].last()
    else:
        raise ValueError("corte debe ser 'fin_de_mes' o 'promedio'")
    return out.rename("trm").reset_index().assign(criterio_corte=corte)


def diferenciar(s: pd.Series, log: bool = True) -> pd.Series:
    """Estacionariza: log-diferencia si la serie es positiva, diferencia simple si no.

    Correlacionar series con tendencia produce correlaciones espurias altas; por
    eso P40–P42 exigen trabajar sobre series estacionarias.
    """
    x = pd.to_numeric(s, errors="coerce")
    if log and (x.dropna() > 0).all():
        return np.log(x).diff()
    return x.diff()


def correlacion_cruzada(objetivo: pd.Series, externa: pd.Series, max_lag: int = 12,
                        estacionaria: bool = True) -> pd.DataFrame:
    """P40–P42. Correlación del objetivo con la externa rezagada 0..max_lag.

    Un rezago k significa: la externa de hace k meses contra el objetivo de hoy.
    Solo los rezagos >= 1 son utilizables como predictores sin fuga.
    """
    y = diferenciar(objetivo) if estacionaria else pd.to_numeric(objetivo, errors="coerce")
    x = diferenciar(externa) if estacionaria else pd.to_numeric(externa, errors="coerce")
    n = min(len(y), len(x))
    y, x = y.iloc[-n:].reset_index(drop=True), x.iloc[-n:].reset_index(drop=True)
    filas = []
    for k in range(max_lag + 1):
        xr = x.shift(k)
        m = y.notna() & xr.notna()
        n_ef = int(m.sum())
        r = float(np.corrcoef(y[m], xr[m])[0, 1]) if n_ef > 2 else np.nan
        banda = 1.96 / np.sqrt(n_ef) if n_ef > 2 else np.nan
        filas.append({"lag": k, "correlacion": r, "n": n_ef, "banda_95": banda,
                      "significativa": bool(abs(r) > banda) if n_ef > 2 else None,
                      "usable_sin_fuga": k >= 1})
    out = pd.DataFrame(filas)
    out.attrs["advertencia"] = "Correlación no implica causalidad."
    return out


def resumen_utilidad(cc: pd.DataFrame) -> dict:
    """Resume si alguna correlación rezagada usable es significativa."""
    usable = cc.loc[cc["usable_sin_fuga"] & cc["significativa"].fillna(False)]
    if usable.empty:
        return {"aporta_senal_aparente": False,
                "mejor_lag": None, "mejor_correlacion": None,
                "nota": "ningún rezago usable es significativo; candidata a descartarse en P45"}
    mejor = usable.reindex(usable["correlacion"].abs().sort_values(ascending=False).index).iloc[0]
    return {"aporta_senal_aparente": True, "mejor_lag": int(mejor["lag"]),
            "mejor_correlacion": float(mejor["correlacion"]),
            "nota": "señal aparente; confirmar con ablación fuera de muestra (P45)"}
