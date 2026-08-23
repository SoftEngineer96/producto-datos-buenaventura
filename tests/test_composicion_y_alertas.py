"""Composición, contribución y reglas de alerta."""
import numpy as np
import pandas as pd
import pytest

from src.comun import alertas, composicion, externas, series


@pytest.fixture(scope="module")
def mensual_pais(microdatos_sinteticos):
    f = series.filtrar_aduana(microdatos_sinteticos)
    f = f.loc[(f["cif_usd"] > 0) & (f["peso_neto_kg"] > 0)]
    return series.serie_mensual(f, por=["pais_origen"])


def test_participaciones_suman_cien(mensual_pais):
    p = composicion.participaciones(mensual_pais, "pais_origen", "cif_usd", top=0)
    assert p["participacion_pct"].sum() == pytest.approx(100.0)
    assert p["acumulado_pct"].iloc[-1] == pytest.approx(100.0)


def test_hhi_maximo_con_un_solo_origen():
    df = pd.DataFrame({"pais_origen": ["156"] * 5, "cif_usd": [1.0] * 5})
    assert composicion.hhi(df, "pais_origen", "cif_usd") == pytest.approx(10_000.0)


def test_hhi_minimo_con_origenes_iguales():
    df = pd.DataFrame({"pais_origen": list("abcdefghij"), "cif_usd": [1.0] * 10})
    assert composicion.hhi(df, "pais_origen", "cif_usd") == pytest.approx(1000.0)
    assert composicion.clasificar_hhi(1000.0) == "desconcentrado"


def test_contribuciones_reproducen_la_variacion_total(mensual_pais):
    c = composicion.contribucion_variacion(mensual_pais, "pais_origen", "cif_usd", top=0)
    assert c["contribucion_abs"].sum() == pytest.approx(c.attrs["variacion_total"])


def test_efecto_mezcla_es_aditivo(mensual_pais):
    e = composicion.efecto_mezcla(mensual_pais, "pais_origen")
    assert {"efecto_precio", "efecto_mezcla"}.issubset(e.columns)
    assert np.isfinite(e["efecto_precio"].sum())


def test_valor_unitario_por_categoria_usa_agregados(mensual_pais):
    v = composicion.valor_unitario_por_categoria(mensual_pais, "pais_origen")
    fila = v.iloc[0]
    assert fila["cif_kg"] == pytest.approx(fila["cif_usd"] / fila["peso_neto_kg"])


# ------------------------------------------------------------------ externas
def test_correlacion_cruzada_marca_los_rezagos_usables(serie_sintetica, trm_sintetica):
    trm = externas.mensualizar_trm(trm_sintetica)
    j = serie_sintetica.merge(trm, on="mes", how="inner")
    cc = externas.correlacion_cruzada(j["cif_usd"], j["trm"], max_lag=6)
    assert not bool(cc.loc[cc["lag"] == 0, "usable_sin_fuga"].iloc[0])
    assert bool(cc.loc[cc["lag"] == 1, "usable_sin_fuga"].iloc[0])
    assert cc["correlacion"].abs().max() <= 1.0


def test_series_con_tendencia_se_diferencian_antes_de_correlacionar():
    """Dos series con tendencia propia y ruido independiente correlacionan casi
    perfecto en niveles y casi nada tras diferenciar. Por eso P40–P42 exigen
    trabajar sobre series estacionarias."""
    r = np.random.default_rng(5)
    a = pd.Series(np.arange(120, dtype=float) + r.normal(0, 2, 120) + 100)
    b = pd.Series(np.arange(120, dtype=float) * 3 + r.normal(0, 6, 120) + 500)
    cruda = abs(np.corrcoef(a, b)[0, 1])
    cc = externas.correlacion_cruzada(a, b, max_lag=1, estacionaria=True)
    assert cruda > 0.98                          # correlación espuria por tendencia
    assert abs(cc.loc[0, "correlacion"]) < 0.5   # desaparece al diferenciar


def test_criterio_de_corte_de_trm_queda_registrado(trm_sintetica):
    t = externas.mensualizar_trm(trm_sintetica, corte="fin_de_mes")
    assert (t["criterio_corte"] == "fin_de_mes").all()


# ------------------------------------------------------------------ alertas
def test_umbrales_se_calibran_solo_con_entrenamiento(serie_sintetica):
    entrenamiento = serie_sintetica.iloc[:120]
    r = alertas.calibrar(entrenamiento["cif_usd"], entrenamiento["mes"])
    assert r.mediana_entrenamiento == pytest.approx(entrenamiento["cif_usd"].median())
    assert r.periodo_calibracion.startswith("2012-01")


def test_valor_tipico_es_normal(serie_sintetica):
    h = serie_sintetica["cif_usd"].iloc[:120]
    r = alertas.calibrar(h)
    res = alertas.evaluar(float(h.iloc[-1]) * 1.01, h, r)
    assert res["nivel"] == "normal"


def test_desviacion_grande_dispara_alerta(serie_sintetica):
    h = serie_sintetica["cif_usd"].iloc[:120]
    r = alertas.calibrar(h)
    res = alertas.evaluar(float(h.median()) * 2.5, h, r)
    assert res["nivel"] == "alerta"
    assert res["razones"]


def test_toda_alerta_explica_su_razon_y_no_ordena(serie_sintetica):
    h = serie_sintetica["cif_usd"].iloc[:120]
    r = alertas.calibrar(h)
    res = alertas.evaluar(float(h.median()) * 1.9, h, r)
    assert len(res["razones"]) >= 1
    assert "no constituye una orden operativa" in res["descargo"].lower()


def test_prediccion_fuera_del_intervalo_es_alerta(serie_sintetica):
    h = serie_sintetica["cif_usd"].iloc[:120]
    r = alertas.calibrar(h)
    p = float(h.iloc[-1])
    res = alertas.evaluar(p, h, r, limite_inferior=p * 1.5, limite_superior=p * 2.0)
    assert res["nivel"] == "alerta"


def test_matriz_de_decision_tiene_los_tres_niveles():
    m = alertas.matriz_decision()
    assert list(m["nivel"]) == ["normal", "seguimiento", "alerta"]
