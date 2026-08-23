"""Fixtures sintéticas para probar la lógica del pipeline.

IMPORTANTE: estos datos son ARTIFICIALES y generados con semilla fija. No
representan importaciones reales ni pueden citarse como resultado del proyecto.
Existen para verificar que el código hace lo que dice, antes de que lleguen los
microdatos oficiales.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

SEMILLA = 20260805


@pytest.fixture(scope="session")
def rng():
    return np.random.default_rng(SEMILLA)


@pytest.fixture(scope="session")
def serie_sintetica() -> pd.DataFrame:
    """Serie mensual con tendencia, estacionalidad, quiebre de nivel y
    dispersión proporcional al nivel (heterocedástica a propósito, para que
    P23 tenga algo que detectar)."""
    r = np.random.default_rng(SEMILLA)
    n = 173
    meses = pd.date_range("2012-01-01", periods=n, freq="MS")
    t = np.arange(n)
    nivel = 1.5e8 * (1 + 0.004 * t)
    nivel[120:] *= 1.35                                   # quiebre de régimen
    estacional = 1 + 0.12 * np.sin(2 * np.pi * (t % 12) / 12 - 0.6)
    ruido = r.normal(0, 0.06, n)
    cif = nivel * estacional * (1 + ruido)
    peso = cif / (1.9 + 0.15 * np.sin(2 * np.pi * (t % 12) / 12) + r.normal(0, 0.05, n))
    df = pd.DataFrame({"mes": meses, "cif_usd": cif, "peso_neto_kg": peso})
    df["cif_kg"] = df["cif_usd"] / df["peso_neto_kg"]
    df["n_registros"] = r.integers(30_000, 45_000, n)
    return df


@pytest.fixture(scope="session")
def microdatos_sinteticos() -> pd.DataFrame:
    """Microdatos con la forma esperada tras la homologación, incluidos
    defectos deliberados: duplicados, negativos, CIF<FOB y códigos sin ceros."""
    r = np.random.default_rng(SEMILLA + 1)
    n = 6_000
    fechas = pd.to_datetime(r.choice(pd.date_range("2024-01-01", "2026-03-01", freq="D"), n))
    paises = r.choice(["156", "840", "076", "484", "999"], n, p=[.35, .25, .15, .20, .05])
    capitulos = r.choice(["27", "84", "85", "10", "39"], n)
    cif = r.lognormal(9, 1.4, n)
    df = pd.DataFrame({
        "adua": "35",
        "fecha": fechas,
        "pais_origen": paises,
        "capitulo": capitulos,
        "subpartida": [c + str(r.integers(10_000_000, 99_999_999)) for c in capitulos],
        "cif_usd": cif,
        "fob_usd": cif / r.uniform(1.02, 1.25, n),
        "peso_neto_kg": cif / r.uniform(0.5, 6.0, n),
    })
    df["peso_bruto_kg"] = df["peso_neto_kg"] * r.uniform(1.01, 1.20, n)
    # defectos deliberados: primero se siembran, luego se duplica, para que el
    # conteo de duplicados exactos sea determinista (40).
    df.loc[df.index[:15], "cif_usd"] = -abs(df.loc[df.index[:15], "cif_usd"])  # negativos
    df.loc[df.index[20:30], "peso_neto_kg"] = 0                    # sin peso
    df = pd.concat([df, df.iloc[:40]], ignore_index=True)          # 40 duplicados exactos
    otra_aduana = df.iloc[:500].copy()
    otra_aduana["adua"] = "11"
    return pd.concat([df, otra_aduana], ignore_index=True)


@pytest.fixture(scope="session")
def trm_sintetica() -> pd.DataFrame:
    r = np.random.default_rng(SEMILLA + 2)
    dias = pd.date_range("2012-01-01", "2026-06-30", freq="D")
    valores = 2000 + np.cumsum(r.normal(0.4, 12, len(dias)))
    return pd.DataFrame({"fecha": dias, "trm": valores})
