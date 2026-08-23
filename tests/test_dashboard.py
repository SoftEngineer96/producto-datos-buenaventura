"""Pruebas de reconciliación y calidad del dashboard BI."""

from __future__ import annotations

import pandas as pd
import pytest

from dashboard.data import DataValidationError, load_dashboard_data, validate_core_quality
from dashboard.metrics import calculate_import_kpis
from dashboard.transforms import filter_dates, missing_months, port_monthly, quality_profile


@pytest.fixture(scope="module")
def dashboard_data():
    return load_dashboard_data()


def test_real_sources_load_with_expected_grains(dashboard_data):
    assert len(dashboard_data.imports_monthly) == 173
    assert len(dashboard_data.port_cargo_monthly) == 508
    assert dashboard_data.imports_monthly["mes"].is_monotonic_increasing
    assert dashboard_data.port_cargo_monthly["mes"].notna().all()
    assert len(dashboard_data.seasonality) == 12
    assert len(dashboard_data.container_series) == 102
    assert len(dashboard_data.model_metrics) == 20
    assert len(dashboard_data.integrated_correlations) == 21


def test_import_kpis_reconcile_to_source(dashboard_data):
    source = dashboard_data.imports_monthly
    kpis = calculate_import_kpis(source)
    assert kpis.cif_total_usd == pytest.approx(float(source["cif_usd"].sum()))
    assert kpis.peso_neto_total_kg == pytest.approx(float(source["peso_neto_kg"].sum()))
    assert kpis.declaraciones_total == int(source["n_registros"].sum())
    expected_change = (
        float(source.iloc[-1]["cif_usd"]) / float(source.iloc[-2]["cif_usd"]) - 1
    ) * 100
    assert kpis.variacion_cif_mensual_pct == pytest.approx(expected_change)


def test_date_filter_is_inclusive(dashboard_data):
    filtered = filter_dates(
        dashboard_data.imports_monthly,
        pd.Timestamp("2024-01-01"),
        pd.Timestamp("2024-12-31"),
    )
    assert len(filtered) == 12
    assert filtered["mes"].min() == pd.Timestamp("2024-01-01")
    assert filtered["mes"].max() == pd.Timestamp("2024-12-01")


def test_port_monthly_reconciles_all_cargo_rows(dashboard_data):
    raw = dashboard_data.port_cargo_monthly
    monthly = port_monthly(raw)
    for measure in ["sum_importacion", "sum_exportacion", "sum_transbordo"]:
        assert monthly[measure].sum() == pytest.approx(raw[measure].sum())
    assert len(missing_months(monthly)) == 0


def test_core_data_quality_controls_pass(dashboard_data):
    profile = quality_profile(
        "Aduanas",
        dashboard_data.imports_monthly,
        ["mes"],
        "mes",
        ["cif_usd", "fob_usd", "peso_neto_kg", "peso_bruto_kg", "n_registros"],
    )
    assert profile["estado"] == "Aprobado"
    assert profile["celdas_nulas"] == 0
    assert profile["llaves_duplicadas"] == 0
    assert len(missing_months(dashboard_data.imports_monthly)) == 0


def test_critical_quality_gate_rejects_duplicate_keys():
    invalid = pd.DataFrame(
        {
            "mes": [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-01")],
            "valor": [1.0, 2.0],
        }
    )
    with pytest.raises(DataValidationError, match="llaves duplicadas"):
        validate_core_quality(
            invalid,
            source="prueba",
            key=["mes"],
            required=["mes", "valor"],
            nonnegative=["valor"],
        )
