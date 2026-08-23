"""Los controles de calidad deben encontrar los defectos que sembramos."""
import pandas as pd

from src.comun import auditoria, series


def test_duplicados_detectados(microdatos_sinteticos):
    resumen, muestra = auditoria.duplicados_por_capa(
        microdatos_sinteticos, ["fecha", "subpartida", "cif_usd"], "landing")
    assert resumen.loc[0, "duplicados_exactos"] == 40
    assert not muestra.empty


def test_negativos_detectados(microdatos_sinteticos):
    fuera = auditoria.valores_fuera_de_dominio(microdatos_sinteticos)
    cif = fuera.loc[fuera["variable"] == "cif_usd"].iloc[0]
    assert cif["n_negativos"] >= 15


def test_pesos_en_cero_detectados(microdatos_sinteticos):
    fuera = auditoria.valores_fuera_de_dominio(microdatos_sinteticos)
    peso = fuera.loc[fuera["variable"] == "peso_neto_kg"].iloc[0]
    assert peso["n_ceros"] >= 10


def test_cif_menor_que_fob_se_detecta_y_es_marginal(microdatos_sinteticos):
    """El CIF incluye seguro y flete: CIF < FOB solo puede venir de un error.

    Los únicos casos son los negativos sembrados y sus duplicados; deben ser
    detectados y representar una fracción mínima del total.
    """
    coh = auditoria.coherencia_cif_fob(microdatos_sinteticos)
    assert coh.loc[0, "n_cif_menor_fob"] > 0
    assert coh.loc[0, "pct_cif_menor_fob"] < 1.0
    assert coh.loc[0, "razon_mediana"] > 1.0


def test_codigos_como_texto_conservan_ceros(microdatos_sinteticos):
    val = auditoria.validar_codigos(microdatos_sinteticos, {"pais_origen": 3})
    assert not bool(val.loc[0, "riesgo_ceros_perdidos"])


def test_codigos_numericos_marcan_riesgo(microdatos_sinteticos):
    df = microdatos_sinteticos.copy()
    df["pais_origen"] = df["pais_origen"].astype(int)   # así se pierden los ceros
    val = auditoria.validar_codigos(df, {"pais_origen": 3})
    assert bool(val.loc[0, "riesgo_ceros_perdidos"])


def test_cobertura_mensual_marca_hueco():
    fechas = pd.to_datetime(["2024-01-15", "2024-02-10", "2024-04-01"])
    cob = auditoria.cobertura_mensual(pd.Series(fechas))
    assert len(cob) == 4
    assert not bool(cob.loc[cob["mes"] == "2024-03", "observado"].iloc[0])


def test_embudo_registra_perdidas():
    e = auditoria.embudo_registros({"raw": 1000, "adua_35": 400, "trusted": 380})
    assert e.loc[1, "perdida_abs"] == 600
    assert e.loc[2, "perdida_pct"] == 5.0


def test_bitacora_es_reversible(microdatos_sinteticos):
    b = auditoria.BitacoraExclusiones()
    m = microdatos_sinteticos["cif_usd"] < 0
    limpio = b.excluir(microdatos_sinteticos, m, "cif negativo", "valor imposible")
    assert len(limpio) == len(microdatos_sinteticos) - int(m.sum())
    assert b.a_dataframe().loc[0, "registros_excluidos"] == int(m.sum())


def test_reconciliacion_usa_tolerancia_declarada():
    propia = pd.Series([100.0, 200.0], index=["2026-01", "2026-02"])
    oficial = pd.Series([100.2, 210.0], index=["2026-01", "2026-02"])
    rec = auditoria.reconciliar_totales(propia, oficial, tolerancia=0.005)
    assert bool(rec.loc[0, "dentro_tolerancia"])
    assert not bool(rec.loc[1, "dentro_tolerancia"])


def test_filtro_de_aduana_conserva_solo_la_35(microdatos_sinteticos):
    f = series.filtrar_aduana(microdatos_sinteticos)
    assert set(f["adua"].unique()) == {"35"}
    assert len(f) < len(microdatos_sinteticos)
