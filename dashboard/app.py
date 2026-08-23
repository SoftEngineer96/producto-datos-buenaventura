"""Dashboard ejecutivo de importaciones y actividad portuaria de Buenaventura.

Ejecutar desde la raíz del repositorio:

    streamlit run dashboard/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dashboard.data import DataValidationError, load_dashboard_data  # noqa: E402
from dashboard.metrics import KPI_DEFINITIONS, calculate_import_kpis  # noqa: E402
from dashboard.transforms import (  # noqa: E402
    filter_dates,
    filter_values,
    missing_months,
    port_monthly,
    quality_profile,
)

BLUE = "#153a5b"
GOLD = "#d89a3d"
ORANGE = "#b85c38"

st.set_page_config(
    page_title="Importaciones de Buenaventura · BI",
    page_icon="⚓",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _load_css() -> None:
    css_path = ROOT / "dashboard" / "style.css"
    if css_path.exists():
        st.markdown(
            f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True
        )


@st.cache_data(show_spinner="Validando fuentes reales…")
def get_data():
    return load_dashboard_data()


def fmt_number(value: float, decimals: int = 1) -> str:
    return f"{value:,.{decimals}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def line_chart(frame: pd.DataFrame, x: str, y: str, title: str, subtitle: str) -> alt.Chart:
    return (
        alt.Chart(frame, title=alt.TitleParams(title, subtitle=[subtitle], anchor="start"))
        .mark_line(
            color=BLUE, strokeWidth=2.5, point=alt.OverlayMarkDef(size=28, filled=True, color=GOLD)
        )
        .encode(
            x=alt.X(f"{x}:T", title=None, axis=alt.Axis(format="%Y", labelAngle=0)),
            y=alt.Y(f"{y}:Q", title=None, scale=alt.Scale(zero=False)),
            tooltip=[
                alt.Tooltip(f"{x}:T", title="Mes", format="%Y-%m"),
                alt.Tooltip(f"{y}:Q", title=title, format=",.2f"),
            ],
        )
        .properties(height=315)
    )


def horizontal_bar(
    frame: pd.DataFrame, category: str, value: str, title: str, subtitle: str
) -> alt.Chart:
    return (
        alt.Chart(frame, title=alt.TitleParams(title, subtitle=[subtitle], anchor="start"))
        .mark_bar(color=GOLD, cornerRadiusEnd=4)
        .encode(
            x=alt.X(f"{value}:Q", title=None, scale=alt.Scale(zero=True)),
            y=alt.Y(f"{category}:N", title=None, sort="-x", axis=alt.Axis(labelLimit=360)),
            tooltip=[
                alt.Tooltip(f"{category}:N", title="Categoría"),
                alt.Tooltip(f"{value}:Q", title="Valor", format=",.2f"),
            ],
        )
        .properties(height=max(260, 42 * len(frame)))
    )


def multi_line_chart(
    frame: pd.DataFrame,
    *,
    x: str,
    y: str,
    series: str,
    title: str,
    subtitle: str,
    height: int = 330,
) -> alt.Chart:
    """Línea multiserie con identidad explícita y paleta estable."""
    return (
        alt.Chart(frame, title=alt.TitleParams(title, subtitle=[subtitle], anchor="start"))
        .mark_line(strokeWidth=2.3, point=alt.OverlayMarkDef(size=20, filled=True))
        .encode(
            x=alt.X(f"{x}:T", title=None, axis=alt.Axis(format="%Y", labelAngle=0)),
            y=alt.Y(f"{y}:Q", title=None, scale=alt.Scale(zero=False)),
            color=alt.Color(
                f"{series}:N",
                title=None,
                scale=alt.Scale(range=[BLUE, GOLD, ORANGE, "#718355", "#8d6a9f"]),
            ),
            tooltip=[
                alt.Tooltip(f"{x}:T", title="Periodo", format="%Y-%m"),
                alt.Tooltip(f"{series}:N", title="Serie"),
                alt.Tooltip(f"{y}:Q", title="Valor", format=",.2f"),
            ],
        )
        .properties(height=height)
    )


_load_css()

try:
    data = get_data()
except (FileNotFoundError, DataValidationError, ValueError) as exc:
    st.error("No fue posible cargar las fuentes del dashboard.")
    st.code(str(exc))
    st.stop()

imports = data.imports_monthly
port = data.port_cargo_monthly
societies = data.societies_annual

min_date = min(imports["mes"].min(), port["mes"].min()).date()
max_date = max(imports["mes"].max(), port["mes"].max()).date()

st.sidebar.markdown("## ⚓ Buenaventura BI")
st.sidebar.caption("Importaciones, carga portuaria y evidencia de calidad")
page = st.sidebar.radio(
    "Navegación",
    [
        "Resumen ejecutivo",
        "Dinámica aduanera",
        "Operación portuaria",
        "Composición y territorio",
        "Integración y modelos",
        "Detalle descargable",
        "Calidad y metodología",
    ],
)
st.sidebar.divider()
st.sidebar.markdown("### Filtros globales")
date_value = st.sidebar.date_input(
    "Periodo",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
    format="YYYY-MM-DD",
)
if isinstance(date_value, tuple) and len(date_value) == 2:
    start_date, end_date = date_value
elif isinstance(date_value, tuple) and len(date_value) == 1:
    # Streamlit emite una tupla de un elemento mientras el usuario completa el rango.
    # Tratarla como un solo mes evita una excepción durante el rerun intermedio.
    start_date = end_date = date_value[0]
else:
    start_date = end_date = date_value

cargo_options = sorted(port["tipo_carga"].dropna().unique())
selected_cargo = st.sidebar.multiselect("Tipo de carga", cargo_options, default=cargo_options)
society_options = sorted(societies["sociedad_portuaria"].dropna().unique())
selected_societies = st.sidebar.multiselect(
    "Sociedad portuaria", society_options, default=society_options
)

imports_filtered = filter_dates(imports, start_date, end_date)
port_filtered = filter_values(
    filter_dates(port, start_date, end_date), "tipo_carga", selected_cargo
)
start_year, end_year = pd.Timestamp(start_date).year, pd.Timestamp(end_date).year
societies_filtered = societies.loc[
    societies["anno_vigencia"].between(start_year, end_year)
    & societies["sociedad_portuaria"].isin(selected_societies)
].copy()

st.sidebar.divider()
st.sidebar.caption(
    "Fuente aduanera: DANE IMPO · ADUA 35\n\n" "Fuente portuaria: Supertransporte · 5r3g-zv5z"
)

st.markdown('<p class="eyebrow">PRODUCTO DE DATOS · UNIVERSIDAD LIBRE</p>', unsafe_allow_html=True)
st.title("Importaciones y actividad portuaria de Buenaventura")
st.caption(
    f"Datos reales agregados · Aduanas hasta {imports['mes'].max():%Y-%m} · "
    f"Puerto hasta {port['mes'].max():%Y-%m} · Vista actual: {start_date:%Y-%m} a {end_date:%Y-%m}"
)

if imports_filtered.empty and page not in {"Calidad y metodología", "Integración y modelos"}:
    st.warning("El periodo seleccionado no contiene observaciones aduaneras.")

if page == "Resumen ejecutivo":
    st.markdown("## Panorama del periodo")
    if not imports_filtered.empty:
        kpis = calculate_import_kpis(imports_filtered)
        k1, k2, k3 = st.columns(3)
        k1.metric("Valor CIF · US$ mil M", fmt_number(kpis.cif_total_usd / 1e9))
        k2.metric("Peso neto", f"{fmt_number(kpis.peso_neto_total_kg / 1e9)} M t")
        k3.metric("Declaraciones", f"{fmt_number(kpis.declaraciones_total / 1e6, 2)} M")
        k4, k5 = st.columns(2)
        k4.metric("Meses observados", f"{kpis.observaciones_mensuales}")
        variation = kpis.variacion_cif_mensual_pct
        k5.metric(
            "Variación CIF mensual",
            "—" if variation is None else f"{fmt_number(variation)} %",
            help="Compara los dos últimos meses disponibles dentro del filtro.",
        )

        st.altair_chart(
            line_chart(
                imports_filtered.assign(cif_millones=imports_filtered["cif_usd"] / 1e6),
                "mes",
                "cif_millones",
                "Tendencia mensual del valor CIF",
                "US$ millones corrientes · DANE IMPO, ADUA 35",
            ),
            use_container_width=True,
        )

    left, right = st.columns(2)
    monthly_port = port_monthly(port_filtered) if not port_filtered.empty else pd.DataFrame()
    with left:
        if monthly_port.empty:
            st.info("Seleccione al menos un tipo de carga para ver la tendencia portuaria.")
        else:
            st.altair_chart(
                line_chart(
                    monthly_port.assign(
                        importacion_miles_t=monthly_port["sum_importacion"] / 1_000
                    ),
                    "mes",
                    "importacion_miles_t",
                    "Carga portuaria de importación",
                    "Miles de toneladas · tipos de carga seleccionados",
                ),
                use_container_width=True,
            )
    with right:
        if not port_filtered.empty:
            cargo_rank = (
                port_filtered.groupby("tipo_carga", as_index=False)["sum_importacion"]
                .sum()
                .assign(importacion_millones_t=lambda x: x["sum_importacion"] / 1e6)
            )
            st.altair_chart(
                horizontal_bar(
                    cargo_rank,
                    "tipo_carga",
                    "importacion_millones_t",
                    "Importación por tipo de carga",
                    "Millones de toneladas · periodo seleccionado",
                ),
                use_container_width=True,
            )

    st.info(
        "La serie aduanera mide declaraciones de importación; la serie portuaria mide movimientos de carga "
        "e incluye exportación y transbordo. Se comparan como señales agregadas, nunca como registros equivalentes."
    )

elif page == "Dinámica aduanera":
    st.markdown("## Evolución, variación y estacionalidad aduanera")
    if imports_filtered.empty:
        st.info("Amplíe el periodo para consultar la dinámica aduanera.")
    else:
        indexed = imports_filtered[["mes", "cif_usd", "peso_neto_kg", "n_registros"]].copy()
        labels = {
            "cif_usd": "Valor CIF",
            "peso_neto_kg": "Peso neto",
            "n_registros": "Declaraciones",
        }
        for column in labels:
            baseline = float(indexed.iloc[0][column])
            indexed[column] = indexed[column].div(baseline).mul(100) if baseline else pd.NA
        indexed_long = indexed.melt("mes", var_name="serie", value_name="indice_base_100")
        indexed_long["serie"] = indexed_long["serie"].map(labels)
        st.altair_chart(
            multi_line_chart(
                indexed_long,
                x="mes",
                y="indice_base_100",
                series="serie",
                title="Valor, volumen y declaraciones en una escala comparable",
                subtitle=f"Índice base 100 en {imports_filtered['mes'].min():%Y-%m} · DANE IMPO, ADUA 35",
            ),
            use_container_width=True,
        )

        left, right = st.columns(2)
        with left:
            changes = imports_filtered[["mes", "cif_usd"]].copy()
            changes["variacion_pct"] = changes["cif_usd"].pct_change().mul(100)
            changes = changes.dropna(subset=["variacion_pct"])
            if changes.empty:
                st.info("Se requieren al menos dos meses para calcular variaciones.")
            else:
                change_chart = (
                    alt.Chart(
                        changes,
                        title=alt.TitleParams(
                            "Variación mensual del valor CIF",
                            subtitle=["Cambio frente al mes anterior · %"],
                            anchor="start",
                        ),
                    )
                    .mark_bar()
                    .encode(
                        x=alt.X("mes:T", title=None, axis=alt.Axis(format="%Y", labelAngle=0)),
                        y=alt.Y("variacion_pct:Q", title=None),
                        color=alt.condition(
                            alt.datum.variacion_pct >= 0,
                            alt.value(BLUE),
                            alt.value(ORANGE),
                        ),
                        tooltip=[
                            alt.Tooltip("mes:T", title="Mes", format="%Y-%m"),
                            alt.Tooltip("variacion_pct:Q", title="Variación", format=".1f"),
                        ],
                    )
                    .properties(height=300)
                )
                st.altair_chart(change_chart, use_container_width=True)
        with right:
            seasonal = data.seasonality.loc[data.seasonality["variable"] == "cif_usd"].copy()
            seasonal["mes_nombre"] = seasonal["mes_num"].map(
                dict(
                    enumerate(
                        [
                            "Ene",
                            "Feb",
                            "Mar",
                            "Abr",
                            "May",
                            "Jun",
                            "Jul",
                            "Ago",
                            "Sep",
                            "Oct",
                            "Nov",
                            "Dic",
                        ],
                        1,
                    )
                )
            )
            points = (
                alt.Chart(
                    seasonal,
                    title=alt.TitleParams(
                        "Patrón estacional histórico del CIF",
                        subtitle=["Índice: 100 = mes promedio · serie completa 2012-01 a 2026-05"],
                        anchor="start",
                    ),
                )
                .mark_circle(size=115, color=GOLD)
                .encode(
                    x=alt.X("mes_nombre:N", title=None, sort=list(seasonal["mes_nombre"])),
                    y=alt.Y("indice_estacional:Q", title=None, scale=alt.Scale(domain=[90, 107])),
                    tooltip=[
                        alt.Tooltip("mes_nombre:N", title="Mes"),
                        alt.Tooltip("indice_estacional:Q", title="Índice", format=".1f"),
                        alt.Tooltip("n_observaciones:Q", title="Observaciones"),
                    ],
                )
            )
            baseline = (
                alt.Chart(pd.DataFrame({"referencia": [100]}))
                .mark_rule(color=BLUE, strokeDash=[5, 4])
                .encode(y="referencia:Q")
            )
            st.altair_chart((baseline + points).properties(height=300), use_container_width=True)

    st.caption(
        "El índice base 100 compara trayectorias relativas; no implica que dólares, kilogramos y declaraciones "
        "sean magnitudes equivalentes. La estacionalidad usa la serie histórica completa para conservar estabilidad."
    )

elif page == "Operación portuaria":
    st.markdown("## Tráfico, concentración y eventos portuarios")
    container = filter_dates(data.container_series, start_date, end_date)
    if container.empty:
        st.info("El periodo seleccionado no contiene observaciones de carga contenerizada.")
    else:
        container_long = container.melt(
            "mes",
            value_vars=["ton_cont_importacion", "ton_cont_exportacion", "ton_cont_transbordo"],
            var_name="flujo",
            value_name="toneladas",
        )
        container_long["flujo"] = container_long["flujo"].map(
            {
                "ton_cont_importacion": "Importación",
                "ton_cont_exportacion": "Exportación",
                "ton_cont_transbordo": "Transbordo",
            }
        )
        area = (
            alt.Chart(
                container_long,
                title=alt.TitleParams(
                    "Composición mensual de la carga contenerizada",
                    subtitle=["Toneladas; la fuente no publica TEU"],
                    anchor="start",
                ),
            )
            .mark_area(opacity=0.82)
            .encode(
                x=alt.X("mes:T", title=None, axis=alt.Axis(format="%Y", labelAngle=0)),
                y=alt.Y("toneladas:Q", title=None, stack="zero"),
                color=alt.Color(
                    "flujo:N",
                    title=None,
                    scale=alt.Scale(
                        domain=["Importación", "Exportación", "Transbordo"],
                        range=[BLUE, GOLD, ORANGE],
                    ),
                ),
                tooltip=[
                    alt.Tooltip("mes:T", title="Mes", format="%Y-%m"),
                    alt.Tooltip("flujo:N", title="Flujo"),
                    alt.Tooltip("toneladas:Q", title="Toneladas", format=",.0f"),
                ],
            )
            .properties(height=330)
        )
        st.altair_chart(area, use_container_width=True)

    left, right = st.columns(2)
    with left:
        hhi = data.hhi_annual.loc[data.hhi_annual["anio"].between(start_year, end_year)].copy()
        if hhi.empty:
            st.info("No hay concentración anual dentro del periodo seleccionado.")
        else:
            hhi["corte"] = hhi["periodo_completo"].map(
                lambda value: "Completo" if str(value).lower() == "true" else "Parcial"
            )
            hhi_chart = (
                alt.Chart(
                    hhi,
                    title=alt.TitleParams(
                        "Concentración anual por sociedad portuaria",
                        subtitle=["Índice HHI de 0 a 10.000 · 2026 es un corte parcial"],
                        anchor="start",
                    ),
                )
                .mark_line(color=BLUE, strokeWidth=2.4, point=True)
                .encode(
                    x=alt.X("anio:O", title=None),
                    y=alt.Y("hhi:Q", title=None, scale=alt.Scale(zero=True)),
                    tooltip=[
                        "anio:O",
                        alt.Tooltip("hhi:Q", format=",.0f"),
                        "corte:N",
                        "sociedades_que_reportan:Q",
                    ],
                )
                .properties(height=300)
            )
            st.altair_chart(hhi_chart, use_container_width=True)
    with right:
        activity = societies_filtered.copy()
        if activity.empty:
            st.info("No hay actividad anual para las sociedades seleccionadas.")
        else:
            activity["millones_t"] = activity["toneladas_totales"] / 1e6
            activity["sociedad_corta"] = activity["sociedad_portuaria"].replace(
                {
                    "COMPANIA DE PUERTOS ASOCIADOS S.A.": "COMPAS",
                    "GRUPO PORTUARIO S.A": "Grupo Portuario",
                    "SOCIEDAD PORTUARIA REGIONAL DE BUENAVENTURA S.A": "SPRBUN",
                    "SOCIEDAD PORTUARIA TERMINAL DE CONTENEDORES DE BUENAVENTURA S.A.": "TCBUEN",
                    "SOCIEDAD PUERTO INDUSTRIAL AGUADULCE S.A.": "Aguadulce",
                }
            )
            society_chart = (
                alt.Chart(
                    activity,
                    title=alt.TitleParams(
                        "Actividad anual por sociedad",
                        subtitle=["Millones de toneladas · selección global"],
                        anchor="start",
                    ),
                )
                .mark_line(point=True, strokeWidth=2)
                .encode(
                    x=alt.X("anno_vigencia:O", title=None),
                    y=alt.Y("millones_t:Q", title=None, scale=alt.Scale(zero=True)),
                    color=alt.Color("sociedad_corta:N", title="Sociedad"),
                    tooltip=[
                        "anno_vigencia:O",
                        "sociedad_portuaria:N",
                        alt.Tooltip("millones_t:Q", title="Millones t", format=".2f"),
                    ],
                )
                .properties(height=300)
            )
            st.altair_chart(society_chart, use_container_width=True)

    extremes = data.port_extremes.loc[
        data.port_extremes["mes"].between(pd.Timestamp(start_date), pd.Timestamp(end_date))
    ]
    st.markdown("### Meses extremos detectados")
    if extremes.empty:
        st.success("No hay meses clasificados como extremos dentro del periodo seleccionado.")
    else:
        shown = extremes.assign(
            mes=extremes["mes"].dt.strftime("%Y-%m"),
            valor_millones_t=extremes["valor"] / 1e6,
        )[["mes", "valor_millones_t", "z_robusto", "decision"]]
        st.dataframe(shown, hide_index=True, use_container_width=True)

elif page == "Integración y modelos":
    st.markdown("## Integración, capacidad predictiva e incertidumbre")
    st.info(
        "Esta vista resume validaciones históricas del Colab. No cambia con el filtro de fechas porque los "
        "modelos requieren ventanas y cortes comparables definidos durante el backtest."
    )
    correlations = data.integrated_correlations.copy()
    correlations["par_visible"] = correlations["par"].replace(
        {
            "cif_usd~trm_cop_usd": "CIF ↔ TRM",
            "toneladas_totales~trm_cop_usd": "Toneladas ↔ TRM",
            "toneladas_totales~oni_anomalia": "Toneladas ↔ ONI",
        }
    )
    corr_chart = (
        alt.Chart(
            correlations,
            title=alt.TitleParams(
                "Correlaciones cruzadas por rezago",
                subtitle=["Coeficiente de Pearson · líneas punteadas = banda aproximada de 95 %"],
                anchor="start",
            ),
        )
        .mark_line(point=True, strokeWidth=2)
        .encode(
            x=alt.X("lag:Q", title="Rezago (meses)", axis=alt.Axis(values=list(range(7)))),
            y=alt.Y("correlacion:Q", title=None, scale=alt.Scale(domain=[-0.25, 0.25])),
            color=alt.Color(
                "par_visible:N",
                title="Par analítico",
                scale=alt.Scale(range=[BLUE, GOLD, ORANGE]),
            ),
            tooltip=[
                "par_visible:N",
                "lag:Q",
                alt.Tooltip("correlacion:Q", format=".3f"),
                "n:Q",
            ],
        )
        .properties(height=315)
    )
    band = (
        alt.Chart(pd.DataFrame({"limite": [-0.196, 0.196]}))
        .mark_rule(color="#6b7280", strokeDash=[5, 4])
        .encode(y="limite:Q")
    )
    st.altair_chart(corr_chart + band, use_container_width=True)
    st.caption(
        "Ningún par supera la banda de significancia; la vista no presenta correlación como causalidad."
    )

    model_left, model_right = st.columns([1, 2])
    objectives = sorted(data.model_metrics["objetivo"].unique())
    windows = sorted(int(value) for value in data.model_metrics["ventana"].unique())
    with model_left:
        objective = st.selectbox("Indicador modelado", objectives)
        window = st.selectbox("Ventana mínima de entrenamiento", windows, index=len(windows) - 1)
    model_view = data.model_metrics.loc[
        (data.model_metrics["objetivo"] == objective) & (data.model_metrics["ventana"] == window)
    ].sort_values("wape_pct")
    best = model_view.iloc[0]
    with model_right:
        m1, m2, m3 = st.columns(3)
        m1.metric("Mejor modelo", str(best["modelo"]).upper())
        m2.metric("WAPE", f"{fmt_number(float(best['wape_pct']))} %")
        m3.metric("Skill vs. Naive 12", f"{fmt_number(float(best['skill_vs_naive12_pct']))} %")

    model_chart = (
        alt.Chart(
            model_view,
            title=alt.TitleParams(
                "Error fuera de muestra por modelo",
                subtitle=[f"WAPE % · {objective} · ventana {window} meses · menor es mejor"],
                anchor="start",
            ),
        )
        .mark_bar(cornerRadiusEnd=4)
        .encode(
            x=alt.X("wape_pct:Q", title=None, scale=alt.Scale(zero=True)),
            y=alt.Y("modelo:N", title=None, sort="x"),
            color=alt.condition(
                alt.datum.modelo == str(best["modelo"]), alt.value(GOLD), alt.value(BLUE)
            ),
            tooltip=[
                "modelo:N",
                alt.Tooltip("wape_pct:Q", title="WAPE %", format=".2f"),
                alt.Tooltip("mase_12:Q", title="MASE 12", format=".2f"),
                alt.Tooltip("n_cortes:Q", title="Cortes"),
            ],
        )
        .properties(height=260)
    )
    st.altair_chart(model_chart, use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.altair_chart(
            horizontal_bar(
                data.ablation.sort_values("wape_pct"),
                "conjunto",
                "wape_pct",
                "Valor incremental de otras fuentes",
                "WAPE % · menor es mejor · historia propia como referencia",
            ),
            use_container_width=True,
        )
    with right:
        coverage = data.interval_coverage.copy()
        coverage["cobertura_pct"] = coverage["cobertura_empirica"] * 100
        coverage_chart = (
            alt.Chart(
                coverage,
                title=alt.TitleParams(
                    "Cobertura empírica de intervalos",
                    subtitle=["Objetivo nominal 80 % · ancho relativo disponible en tooltip"],
                    anchor="start",
                ),
            )
            .mark_bar(color=GOLD, cornerRadiusEnd=4)
            .encode(
                x=alt.X("cobertura_pct:Q", title=None, scale=alt.Scale(domain=[0, 100])),
                y=alt.Y("caso:N", title=None, sort="-x", axis=alt.Axis(labelLimit=260)),
                tooltip=[
                    "caso:N",
                    alt.Tooltip("cobertura_pct:Q", title="Cobertura %", format=".1f"),
                    alt.Tooltip(
                        "ancho_relativo_promedio_pct:Q", title="Ancho relativo %", format=".1f"
                    ),
                    "n_evaluados:Q",
                ],
            )
            .properties(height=260)
        )
        target = (
            alt.Chart(pd.DataFrame({"objetivo": [80]}))
            .mark_rule(color=BLUE, strokeDash=[5, 4])
            .encode(x="objetivo:Q")
        )
        st.altair_chart(coverage_chart + target, use_container_width=True)

elif page == "Composición y territorio":
    st.markdown("## ¿Qué carga se mueve y por qué sociedades?")
    left, right = st.columns(2)
    with left:
        if not port_filtered.empty:
            cargo_mix = (
                port_filtered.groupby("tipo_carga", as_index=False)["toneladas_totales"]
                .sum()
                .assign(toneladas_millones=lambda x: x["toneladas_totales"] / 1e6)
            )
            st.altair_chart(
                horizontal_bar(
                    cargo_mix,
                    "tipo_carga",
                    "toneladas_millones",
                    "Movimiento total por tipo de carga",
                    "Importación + exportación + transbordo · millones de toneladas",
                ),
                use_container_width=True,
            )
    with right:
        if societies_filtered.empty:
            st.info("No hay sociedades portuarias para el periodo y selección actuales.")
        else:
            society_rank = (
                societies_filtered.groupby("sociedad_portuaria", as_index=False)[
                    "toneladas_totales"
                ]
                .sum()
                .assign(toneladas_millones=lambda x: x["toneladas_totales"] / 1e6)
            )
            st.altair_chart(
                horizontal_bar(
                    society_rank,
                    "sociedad_portuaria",
                    "toneladas_millones",
                    "Movimiento por sociedad portuaria",
                    "Millones de toneladas · años completos o parciales dentro del filtro",
                ),
                use_container_width=True,
            )

    st.markdown("### Especialización histórica sociedad–carga")
    heat = data.societies_by_cargo.loc[
        data.societies_by_cargo["sociedad_portuaria"].isin(selected_societies)
        & data.societies_by_cargo["tipo_carga"].isin(selected_cargo)
    ].copy()
    if not heat.empty:
        heat["millones_t"] = heat["toneladas_totales"] / 1e6
        chart = (
            alt.Chart(
                heat,
                title=alt.TitleParams(
                    "Matriz de especialización portuaria",
                    subtitle=[
                        "Millones de toneladas históricas; esta fuente no ofrece grano mensual"
                    ],
                    anchor="start",
                ),
            )
            .mark_rect(stroke="white")
            .encode(
                x=alt.X("tipo_carga:N", title=None, axis=alt.Axis(labelAngle=-25, labelLimit=180)),
                y=alt.Y("sociedad_portuaria:N", title=None, axis=alt.Axis(labelLimit=360)),
                color=alt.Color(
                    "millones_t:Q",
                    title="Millones t",
                    scale=alt.Scale(range=["#f5eee4", GOLD, ORANGE]),
                ),
                tooltip=[
                    "sociedad_portuaria",
                    "tipo_carga",
                    alt.Tooltip("millones_t:Q", format=",.2f"),
                ],
            )
            .properties(height=330)
        )
        st.altair_chart(chart, use_container_width=True)

    st.markdown("### Cobertura comercial y geográfica disponible")
    geo1, geo2 = st.columns(2)
    with geo1:
        st.markdown("#### Países de origen — corte histórico")
        if data.countries_snapshot.empty:
            st.info("No hay una tabla de participación por país versionada.")
        else:
            countries = data.countries_snapshot.copy()
            countries["origen"] = countries["codigo_pais"].map(lambda value: f"Código DANE {value}")
            st.altair_chart(
                horizontal_bar(
                    countries,
                    "origen",
                    "participacion_cif_pct",
                    "Participación CIF de los principales orígenes",
                    "% del CIF histórico · códigos pendientes de homologación a nombres",
                ),
                use_container_width=True,
            )
    with geo2:
        st.markdown("#### Capítulos arancelarios — corte histórico")
        if data.chapters_snapshot.empty:
            st.info("No hay una tabla de participación por capítulo versionada.")
        else:
            chapters = data.chapters_snapshot.copy()
            chapters["capitulo_hs"] = chapters["capitulo"].map(
                lambda value: f"Capítulo HS {int(value):02d}"
            )
            st.altair_chart(
                horizontal_bar(
                    chapters,
                    "capitulo_hs",
                    "participacion_cif_pct",
                    "Participación CIF por capítulo",
                    "% del CIF histórico · principales cinco capítulos",
                ),
                use_container_width=True,
            )
    st.warning(
        "No se dibuja un mapa: la fuente portuaria está filtrada a una sola zona (Buenaventura) y la tabla "
        "aduanera disponible solo conserva códigos de país agregados. Un mapa con un punto o etiquetas no "
        "homologadas sugeriría más precisión geográfica de la que existe."
    )

elif page == "Detalle descargable":
    st.markdown("## Registros filtrados y descarga")
    dataset_name = st.selectbox(
        "Tabla",
        [
            "Aduanera mensual",
            "Portuaria mensual por carga",
            "Sociedades portuarias por año",
            "Métricas de modelos",
            "Correlaciones integradas",
            "Cobertura de intervalos",
        ],
    )
    if dataset_name == "Aduanera mensual":
        detail = imports_filtered.copy()
        filename = "buenaventura_aduanas_filtrado.csv"
    elif dataset_name == "Portuaria mensual por carga":
        detail = port_filtered.copy()
        filename = "buenaventura_puerto_filtrado.csv"
    elif dataset_name == "Sociedades portuarias por año":
        detail = societies_filtered.copy()
        filename = "buenaventura_sociedades_filtrado.csv"
    elif dataset_name == "Métricas de modelos":
        detail = data.model_metrics.copy()
        filename = "buenaventura_metricas_modelos.csv"
    elif dataset_name == "Correlaciones integradas":
        detail = data.integrated_correlations.copy()
        filename = "buenaventura_correlaciones.csv"
    else:
        detail = data.interval_coverage.copy()
        filename = "buenaventura_cobertura_intervalos.csv"

    search = st.text_input(
        "Buscar en dimensiones de texto", placeholder="Ej.: CONTENEDORES o AGUADULCE"
    )
    if search:
        text_columns = detail.select_dtypes(include="object").columns
        if len(text_columns):
            mask = pd.Series(False, index=detail.index)
            for column in text_columns:
                mask |= detail[column].astype(str).str.contains(search, case=False, na=False)
            detail = detail.loc[mask]

    st.caption(f"{len(detail):,} filas después de aplicar filtros".replace(",", "."))
    st.dataframe(detail, hide_index=True, use_container_width=True, height=520)
    st.download_button(
        "Descargar CSV filtrado",
        data=detail.to_csv(index=False).encode("utf-8-sig"),
        file_name=filename,
        mime="text/csv",
        type="primary",
    )

elif page == "Calidad y metodología":
    st.markdown("## Calidad, trazabilidad y definiciones")
    profiles = pd.DataFrame(
        [
            quality_profile(
                "Aduanas mensuales",
                imports,
                ["mes"],
                "mes",
                ["cif_usd", "fob_usd", "peso_neto_kg", "peso_bruto_kg", "n_registros"],
            ),
            quality_profile(
                "Puerto por mes y carga",
                port,
                ["mes", "tipo_carga"],
                "mes",
                ["sum_importacion", "sum_exportacion", "sum_transbordo"],
            ),
            quality_profile(
                "Sociedades por año",
                societies,
                ["anno_vigencia", "sociedad_portuaria"],
                None,
                ["sum_importacion", "sum_exportacion", "sum_transbordo"],
            ),
        ]
    )
    approved = int((profiles["estado"] == "Aprobado").sum())
    question_status = data.quality_card.get("preguntas", {})
    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Fuentes aprobadas", f"{approved} / {len(profiles)}")
    q2.metric("Nulos detectados", f"{int(profiles['celdas_nulas'].sum())}")
    q3.metric("Duplicados de llave", f"{int(profiles['llaves_duplicadas'].sum())}")
    q4.metric("Preguntas ejecutadas", f"{question_status.get('ejecutada', '—')} / 52")
    st.dataframe(profiles, hide_index=True, use_container_width=True)

    missing_imports = missing_months(imports)
    missing_port = missing_months(port_monthly(port))
    if missing_imports or missing_port:
        st.warning(
            f"Meses ausentes — Aduanas: {len(missing_imports)} · Puerto: {len(missing_port)}. "
            "Revise la continuidad antes de comparar periodos."
        )
    else:
        st.success(
            "Las series aduanera y portuaria son continuas entre su primer y último mes observado."
        )

    st.markdown("### Diccionario de KPIs")
    st.dataframe(KPI_DEFINITIONS, hide_index=True, use_container_width=True)

    st.markdown("### Fuentes y alcance")
    if not data.source_catalog.empty:
        st.dataframe(data.source_catalog, hide_index=True, use_container_width=True)
    if not data.availability.empty:
        st.markdown("### Corte y rezago de publicación")
        st.dataframe(data.availability, hide_index=True, use_container_width=True)
    st.markdown(
        """
        - **Grano aduanero:** un mes agregado para la aduana 35; no hay microdatos personales en el dashboard.
        - **Grano portuario:** mes × tipo de carga; el corte por sociedad solo está disponible por año.
        - **Moneda:** dólares corrientes. La variación mezcla precio, tipo de cambio y volumen.
        - **Comparabilidad:** peso neto aduanero y toneladas portuarias son conceptos distintos.
        - **Actualización:** al reemplazar los archivos versionados conservando el esquema, Streamlit invalida la caché y recalcula la vista.
        """
    )

st.divider()
st.caption(
    "Dashboard de solo lectura · Sin credenciales embebidas · Las alertas y variaciones son señales analíticas, no decisiones operativas."
)
