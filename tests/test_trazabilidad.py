"""Trazabilidad: la numeración P01–P52 está fija y la evidencia es obligatoria."""
import pytest

from src.comun import trazabilidad


def test_catalogo_tiene_52_preguntas():
    assert len(trazabilidad.cargar_catalogo()) == 52


def test_numeracion_congelada():
    trazabilidad.validar_ids()          # lanza si hay hueco o renumeración


def test_los_siete_bloques_estan_presentes():
    c = trazabilidad.cargar_catalogo()
    assert c["bloque"].nunique() == 7


def test_bloque_maritimo_declara_su_inviabilidad():
    """P31 a P38 son el bloque marítimo: la especificación admite cerrarlas como
    no viables, pero exige que el catálogo lo declare de antemano."""
    c = trazabilidad.cargar_catalogo().set_index("pregunta")
    maritimas = [f"P{i}" for i in range(31, 39)]
    assert all(c.loc[p, "estado_previsto"] == "no viable" for p in maritimas)


def test_preguntas_de_integracion_presentes():
    c = trazabilidad.cargar_catalogo().set_index("pregunta")
    assert "directa" in c.loc["P42", "enunciado"].lower()
    assert "vinculado" in c.loc["P43", "enunciado"].lower()


def test_codigo_sin_salida_no_cuenta_como_ejecutada():
    tz = trazabilidad.Trazador()
    tz.registrar("P23", hallazgo="la carga portuaria crece")   # sin archivo ni figura
    assert "P23" in tz.pendientes()


def test_con_archivo_si_cuenta_como_ejecutada(tmp_path):
    f = tmp_path / "serie.csv"
    f.write_text("a,b\n1,2\n", encoding="utf-8")
    tz = trazabilidad.Trazador()
    tz.registrar("P23", archivo=f, hallazgo="ok")
    assert "P23" not in tz.pendientes()


def test_no_se_admite_una_pregunta_fuera_del_catalogo():
    tz = trazabilidad.Trazador()
    with pytest.raises(KeyError):
        tz.registrar("P53", hallazgo="inventada")


def test_resumen_reporta_pendientes():
    assert "0/52" in trazabilidad.Trazador().resumen()
