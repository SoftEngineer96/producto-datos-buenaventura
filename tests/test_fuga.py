"""La prueba que decide si los resultados del modelo son defendibles."""
import numpy as np
import pandas as pd
import pytest

from src.comun import disponibilidad, features


def test_features_solo_usan_pasado(serie_sintetica):
    """Ninguna columna de X puede coincidir con el objetivo del mismo mes."""
    X, y = features.construir_matriz(serie_sintetica, "cif_usd")
    for c in X.columns:
        r = abs(np.corrcoef(X[c].values, y.values)[0, 1])
        assert r < 0.999, f"{c} parece contener el valor contemporáneo (r={r:.4f})"


def test_media_movil_esta_desplazada(serie_sintetica):
    d = features.agregar_medias_moviles(serie_sintetica.copy(), "cif_usd", (3,))
    # la media de 3 en el mes t debe usar t-1, t-2, t-3
    esperado = serie_sintetica["cif_usd"].iloc[[3, 4, 5]].mean()
    assert d["cif_usd_ma3"].iloc[6] == pytest.approx(esperado)


def test_rezago_uno_es_el_valor_anterior(serie_sintetica):
    d = features.agregar_rezagos(serie_sintetica.copy(), "cif_usd", (1,))
    assert d["cif_usd_lag1"].iloc[10] == pytest.approx(serie_sintetica["cif_usd"].iloc[9])


def test_auditor_detecta_fuga_sembrada(serie_sintetica):
    X, y = features.construir_matriz(serie_sintetica, "cif_usd")
    X_sucia = X.copy()
    X_sucia["objetivo_sin_rezagar"] = y.values      # fuga deliberada
    aud = disponibilidad.auditar_features(X_sucia, y, serie_sintetica["mes"])
    sospechosas = set(aud.loc[aud["sospecha_fuga"], "feature"])
    assert "objetivo_sin_rezagar" in sospechosas


def test_variable_publicada_tarde_no_esta_disponible():
    """Un rezago de un mes NO alcanza si la fuente publica 45 días después."""
    v = disponibilidad.Variable("cif_usd", "mensual", 45, 1, True, "DANE")
    assert not v.disponible_para(pd.Timestamp("2026-06-01"))


def test_variable_publicada_a_tiempo_si_esta_disponible():
    v = disponibilidad.Variable("cif_usd", "mensual", 45, 3, True, "DANE")
    assert v.disponible_para(pd.Timestamp("2026-06-01"))


def test_trm_diaria_esta_disponible_con_rezago_uno():
    v = disponibilidad.Variable("trm", "diaria", 0, 1, False, "BanRep")
    assert v.disponible_para(pd.Timestamp("2026-06-01"))


def test_calendario_marca_los_cortes_comprometidos():
    meses = pd.date_range("2026-01-01", periods=6, freq="MS")
    cal = disponibilidad.calendario_disponibilidad(meses=meses)
    no_disp = disponibilidad.variables_no_disponibles(cal)
    assert "cif_usd" in set(no_disp["variable"])


def test_informe_de_fuga_menciona_la_senal_de_alarma():
    meses = pd.date_range("2026-01-01", periods=3, freq="MS")
    cal = disponibilidad.calendario_disponibilidad(meses=meses)
    texto = disponibilidad.informe_fuga(cal, pd.DataFrame(columns=["feature", "sospecha_fuga"]))
    assert "55,5" in texto and "fuga" in texto.lower()


def test_comparar_versiones_detecta_revision():
    a = pd.DataFrame({"mes": ["2026-01", "2026-02"], "cif_usd": [100.0, 200.0]})
    b = pd.DataFrame({"mes": ["2026-01", "2026-02"], "cif_usd": [100.0, 205.0]})
    d = disponibilidad.comparar_versiones(a, b)
    assert bool(d.loc[d["mes"] == "2026-02", "revisado"].iloc[0])
    assert not bool(d.loc[d["mes"] == "2026-01", "revisado"].iloc[0])
