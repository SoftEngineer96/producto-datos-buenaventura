"""Series, valor unitario implícito y detección de heterocedasticidad."""
import numpy as np
import pandas as pd
import pytest

from src.comun import eda, series


def test_cif_kg_es_razon_de_agregados_no_media_de_razones(microdatos_sinteticos):
    f = series.filtrar_aduana(microdatos_sinteticos)
    f = f.loc[f["cif_usd"] > 0]
    m = series.serie_mensual(f)
    fila = m.iloc[5]
    assert fila["cif_kg"] == pytest.approx(fila["cif_usd"] / fila["peso_neto_kg"])


def test_cif_kg_no_produce_infinitos_con_peso_cero():
    df = pd.DataFrame({"mes": pd.to_datetime(["2026-01-01"]),
                       "cif_usd": [1000.0], "peso_neto_kg": [0.0]})
    out = series.agregar_cif_kg(df)
    assert pd.isna(out["cif_kg"].iloc[0])


def test_cif_kg_por_registro_reporta_cuantos_excluyo(microdatos_sinteticos):
    s = series.cif_kg_por_registro(microdatos_sinteticos)
    assert s.attrs["excluidos_sin_peso"] > 0
    assert np.isfinite(s.dropna()).all()


def test_continuidad_deja_nan_no_interpola():
    df = pd.DataFrame({"mes": pd.to_datetime(["2026-01-01", "2026-03-01"]),
                       "cif_usd": [1.0, 3.0]})
    c = series.asegurar_continuidad(df)
    assert len(c) == 3
    assert pd.isna(c.loc[1, "cif_usd"])
    assert series.meses_faltantes(df) == ["2026-02"]


def test_media_movil_no_mira_hacia_adelante():
    s = pd.Series(range(24), dtype=float)
    mm = series.media_movil(s, 12)
    assert pd.isna(mm.iloc[10])
    assert mm.iloc[11] == pytest.approx(np.mean(range(12)))


def test_heterocedasticidad_se_detecta_en_serie_proporcional(serie_sintetica):
    d = eda.heterocedasticidad(serie_sintetica, "cif_usd")
    assert d.attrs["corr_nivel_dispersion"] > 0.3
    assert "logar" in d.attrs["recomendacion"] or "mixta" in d.attrs["recomendacion"]


def test_heterocedasticidad_no_se_inventa_en_serie_homocedastica():
    r = np.random.default_rng(7)
    n = 120
    df = pd.DataFrame({"mes": pd.date_range("2016-01-01", periods=n, freq="MS"),
                       "x": 1000 + r.normal(0, 50, n)})
    d = eda.heterocedasticidad(df, "x")
    assert abs(d.attrs["corr_nivel_dispersion"]) < 0.6


def test_z_robusto_resiste_un_extremo():
    """Un solo valor gigante no debe arrastrar la escala, como sí ocurre con
    el z clásico basado en media y desviación."""
    s = pd.Series([10, 11, 9, 10, 12, 8, 10, 11, 9, 1000.0])
    z = eda.z_robusto(s)
    z_clasico = (s - s.mean()) / s.std()
    assert abs(z.iloc[-1]) > 100
    assert abs(z_clasico.iloc[-1]) < 4          # el clásico casi no lo ve
    assert (z.iloc[:-1].abs() < 5).all()


def test_z_robusto_devuelve_ceros_si_no_hay_dispersion():
    """Caso degenerado explícito: con MAD = 0 no se puede escalar."""
    z = eda.z_robusto(pd.Series([10.0] * 9 + [1000.0]))
    assert (z == 0).all()


def test_extremos_no_eliminan_datos(serie_sintetica):
    ext = eda.detectar_extremos(serie_sintetica, ["cif_usd"])
    assert set(ext.columns) >= {"mes", "variable", "valor", "decision"}
    assert (ext["decision"] == "pendiente de investigar").all()


def test_indices_estacionales_promedian_cien(serie_sintetica):
    idx = eda.indices_estacionales(serie_sintetica, "cif_usd")
    assert len(idx) == 12
    assert idx["indice_estacional"].mean() == pytest.approx(100, abs=2.0)


def test_autocorrelacion_alta_en_lag1(serie_sintetica):
    acf = eda.autocorrelaciones(serie_sintetica["cif_usd"], max_lag=13)
    assert acf.loc[acf["lag"] == 0, "acf"].iloc[0] == 1.0
    assert acf.loc[acf["lag"] == 1, "acf"].iloc[0] > 0.5


def test_punto_de_cambio_encuentra_el_quiebre(serie_sintetica):
    pc = eda.puntos_de_cambio(serie_sintetica["cif_usd"])
    assert not pc.empty
    assert abs(int(pc.iloc[0]["posicion"]) - 120) < 30
