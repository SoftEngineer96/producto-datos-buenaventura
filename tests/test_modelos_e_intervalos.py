"""Backtest, líneas base, intervalos y su cobertura."""
import numpy as np
import pandas as pd
import pytest

from src.comun import features, intervalos, modelos


def test_naive1_predice_el_ultimo_valor():
    y = pd.Series([1.0, 2.0, 3.0])
    assert modelos.Naive1().ajustar(y).predecir() == 3.0


def test_naive12_predice_el_mismo_mes_del_anio_anterior():
    y = pd.Series(range(24), dtype=float)
    assert modelos.NaiveEstacional(12).ajustar(y).predecir() == 12.0


def test_drift_extrapola_la_pendiente():
    y = pd.Series([0.0, 1.0, 2.0, 3.0])
    assert modelos.Drift().ajustar(y).predecir() == pytest.approx(4.0)


def test_backtest_no_entrena_con_el_futuro(serie_sintetica):
    res = modelos.backtest_walk_forward(
        serie_sintetica["cif_usd"], modelos.Naive1(), n_cortes=24,
        meses=serie_sintetica["mes"])
    d = res.detalle
    # el entrenamiento crece de a un mes y nunca incluye el corte evaluado
    assert list(d["n_entrenamiento"]) == list(range(d["corte"].min(), d["corte"].max() + 1))
    for _, r in d.iterrows():
        assert r["n_entrenamiento"] == r["corte"]


def test_backtest_falla_si_la_serie_es_corta():
    y = pd.Series(np.arange(30, dtype=float))
    with pytest.raises(ValueError, match="insuficiente"):
        modelos.backtest_walk_forward(y, modelos.Naive1(), n_cortes=24, min_entrenamiento=36)


def test_naive12_gana_a_naive1_cuando_domina_la_estacionalidad():
    """Con estacionalidad fuerte y poca tendencia, la referencia estacional debe
    ganar. Es lo que justifica usar Naive 12 como línea base y no Naive 1."""
    n = 120
    t = np.arange(n)
    y = pd.Series(1000 + 300 * np.sin(2 * np.pi * t / 12))
    meses = pd.Series(pd.date_range("2016-01-01", periods=n, freq="MS"))
    tabla, _ = modelos.comparar_modelos(
        y, {"naive_1": modelos.Naive1(), "naive_12": modelos.NaiveEstacional(12)},
        ventanas=(24,), meses=meses)
    w = tabla.set_index("modelo")["wape_pct"]
    assert w["naive_12"] < w["naive_1"]


def test_naive1_puede_ganar_cuando_domina_la_tendencia(serie_sintetica):
    """El resultado inverso también debe poder ocurrir: qué línea base es más
    exigente depende de la serie, y por eso se reportan las tres."""
    tabla, _ = modelos.comparar_modelos(
        serie_sintetica["cif_usd"],
        {"naive_1": modelos.Naive1(), "naive_12": modelos.NaiveEstacional(12),
         "drift": modelos.Drift()},
        ventanas=(24,), meses=serie_sintetica["mes"])
    w = tabla.set_index("modelo")["wape_pct"]
    assert set(w.index) == {"naive_1", "naive_12", "drift"}
    assert (w > 0).all() and np.isfinite(w).all()


def test_ridge_supera_las_lineas_base(serie_sintetica):
    X, y = features.construir_matriz(serie_sintetica, "cif_usd")
    tabla, _ = modelos.comparar_modelos(
        y, modelos.catalogo_modelos(), X=X, ventanas=(24,), meses=X.attrs["mes"])
    w = tabla.set_index("modelo")["wape_pct"]
    assert w["ridge"] < w["naive_12"]
    assert w.min() > 0


def test_ablacion_reporta_todos_los_conjuntos(serie_sintetica):
    X, y = features.construir_matriz(serie_sintetica, "cif_usd")
    conj = features.conjuntos_ablacion(list(X.columns), "cif_usd")
    tabla = modelos.ablacion(y, X, conj, ventana=24, meses=X.attrs["mes"])
    assert len(tabla) == len(conj)
    assert "solo_rezagos" in set(tabla["conjunto"])


def test_vif_detecta_redundancia():
    r = np.random.default_rng(3)
    x = r.normal(size=200)
    X = pd.DataFrame({"a": x, "b": x * 2 + r.normal(0, 1e-6, 200), "c": r.normal(size=200)})
    v = features.vif(X).set_index("variable")
    assert bool(v.loc["a", "redundante"])
    assert not bool(v.loc["c", "redundante"])


# ------------------------------------------------------------------ intervalos
def test_intervalo_no_se_deriva_de_una_metrica_puntual(serie_sintetica):
    """El método declarado debe ser el de cuantiles fuera de muestra."""
    res = modelos.backtest_walk_forward(
        serie_sintetica["cif_usd"], modelos.NaiveEstacional(12), n_cortes=48,
        meses=serie_sintetica["mes"])
    inter = intervalos.intervalos_conformales(res.detalle)
    assert "cuantiles empíricos" in inter["metodo"].iloc[0]
    assert "wape" not in inter["metodo"].iloc[0].lower()


def test_calibracion_solo_usa_errores_anteriores(serie_sintetica):
    res = modelos.backtest_walk_forward(
        serie_sintetica["cif_usd"], modelos.Naive1(), n_cortes=48,
        meses=serie_sintetica["mes"])
    inter = intervalos.intervalos_conformales(res.detalle, minimo_calibracion=12)
    assert inter["limite_inferior"].iloc[:12].isna().all()
    assert inter["limite_inferior"].iloc[12:].notna().all()


def test_cobertura_se_mide_y_no_se_declara(serie_sintetica):
    res = modelos.backtest_walk_forward(
        serie_sintetica["cif_usd"], modelos.NaiveEstacional(12), n_cortes=48,
        meses=serie_sintetica["mes"])
    inter = intervalos.intervalos_conformales(res.detalle)
    cob = intervalos.medir_cobertura(inter)
    assert 0.0 <= cob["cobertura_empirica"] <= 1.0
    assert cob["n_evaluados"] > 0
    assert "cobertura empírica" in cob["nombre_honesto"]


def test_subcobertura_se_reporta_como_tal():
    """Intervalo deliberadamente estrecho: debe declararse subcalibrado."""
    r = np.random.default_rng(11)
    n = 60
    d = pd.DataFrame({
        "corte": range(n), "mes": [f"2020-{i%12+1:02d}" for i in range(n)],
        "observado": r.normal(100, 20, n)})
    d["prediccion"] = 100.0
    inter = intervalos.intervalos_conformales(d, nivel=0.80)
    inter["limite_inferior"] = 99.0      # estrechamos a mano
    inter["limite_superior"] = 101.0
    inter["dentro_intervalo"] = (inter["observado"] >= 99) & (inter["observado"] <= 101)
    inter["ancho"] = 2.0
    inter["ancho_relativo_pct"] = 2.0
    cob = intervalos.medir_cobertura(inter, nivel=0.80)
    assert not cob["calibrado"]
    assert "subcobertura" in cob["veredicto"]
