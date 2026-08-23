"""Pruebas de rerun e interacción de la aplicación Streamlit completa."""

from __future__ import annotations

from datetime import date

from streamlit.testing.v1 import AppTest


def dashboard_app() -> AppTest:
    return AppTest.from_file("dashboard/app.py", default_timeout=30).run()


def test_default_view_renders_real_kpis_without_exceptions():
    app = dashboard_app()
    assert not app.exception
    assert [metric.value for metric in app.metric] == [
        "204,5",
        "165,0 M t",
        "6,70 M",
        "173",
        "2,1 %",
    ]


def test_partial_date_range_rerun_is_safe():
    app = dashboard_app()
    app.date_input[0].set_value((date(2024, 1, 1),)).run()
    assert not app.exception
    assert app.metric[3].value == "1"


def test_all_views_render_without_exceptions():
    app = dashboard_app()
    for page in [
        "Dinámica aduanera",
        "Operación portuaria",
        "Composición y territorio",
        "Integración y modelos",
        "Detalle descargable",
        "Calidad y metodología",
    ]:
        app.radio[0].set_value(page).run()
        assert not app.exception, page


def test_empty_cargo_filter_has_an_explicit_state():
    app = dashboard_app()
    app.multiselect[0].set_value([]).run()
    assert not app.exception
    assert any("Seleccione al menos un tipo de carga" in info.value for info in app.info)
