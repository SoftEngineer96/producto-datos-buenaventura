"""Las métricas son la vara con la que se mide todo: se prueban contra valores
calculados a mano, no contra otra implementación."""
import numpy as np
import pytest

from src.comun import metricas


def test_wape_caso_conocido():
    y = [100, 200, 300]
    yhat = [110, 180, 330]
    # |10| + |20| + |30| = 60 sobre 600 = 10 %
    assert metricas.wape(y, yhat) == pytest.approx(10.0)


def test_mae_rmse():
    y, yhat = [0, 0, 0], [1, -1, 2]
    assert metricas.mae(y, yhat) == pytest.approx(4 / 3)
    assert metricas.rmse(y, yhat) == pytest.approx(np.sqrt(6 / 3))


def test_sesgo_positivo_significa_subestimacion():
    assert metricas.sesgo([100, 100], [90, 90]) == pytest.approx(10.0)


def test_mase_igual_a_uno_si_el_modelo_es_el_naive():
    y_tr = [10, 12, 14, 16, 18]
    y = [20, 22]
    yhat = [18, 20]          # naive de un paso sobre la continuación
    assert metricas.mase(y, yhat, y_tr, estacionalidad=1) == pytest.approx(1.0)


def test_mase_menor_que_uno_si_el_modelo_supera_al_naive():
    y_tr = list(range(0, 100, 5))
    assert metricas.mase([100, 105], [100, 105], y_tr, 1) == pytest.approx(0.0)


def test_wape_no_explota_con_ceros_pero_mape_los_excluye():
    y, yhat = [0, 100], [10, 110]
    assert np.isfinite(metricas.wape(y, yhat))
    assert metricas.mape(y, yhat) == pytest.approx(10.0)


def test_cortes_ganados():
    y = [10, 10, 10]
    a = [10, 12, 15]     # gana solo el primero
    b = [11, 11, 11]
    assert metricas.cortes_ganados(y, a, b) == pytest.approx(100 / 3)


def test_empate_tecnico_se_declara_como_tal():
    """Réplica del caso Ridge 7,3403 % contra HGB 7,3470 % del proyecto original."""
    rng = np.random.default_rng(1)
    y = rng.normal(1000, 100, 24)
    a = y + rng.normal(0, 30, 24)
    b = a + rng.normal(0, 0.05, 24)      # prácticamente el mismo modelo
    r = metricas.diferencia_significativa(y, a, b)
    assert not r["material"]
    assert "empate" in r["veredicto"]
