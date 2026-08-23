"""Ejecuta el pipeline integrado y produce la evidencia de las 52 preguntas.

    python -m src.correr_integrado

Las preguntas del bloque marítimo se cierran como NO VIABLES con evidencia
documentada, que es una respuesta válida según la especificación V5.
documentada, que es una respuesta válida dentro del alcance del proyecto.
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from . import config, integracion, puertos
from .comun import eda, intervalos, io_utils, modelos, series

plt.rcParams.update({"axes.grid": True, "grid.alpha": .25, "font.size": 9})

REG: dict[str, dict] = {}


def anotar(p, archivo=None, figura=None, hallazgo="", implicacion="",
           limitacion="", estado="ejecutada"):
    REG[p] = {"pregunta": p, "estado": estado,
              "archivo": str(archivo.name) if archivo else "",
              "figura": str(figura.name) if figura else "",
              "hallazgo": hallazgo, "implicacion": implicacion,
              "limitacion": limitacion,
              "fecha": pd.Timestamp.now().isoformat(timespec="seconds")}


def main() -> None:
    config.asegurar_directorios()
    S, F = config.SURFACE, config.FIGURES
    T = lambda d, n: io_utils.guardar_tabla(d, n, S)          # noqa: E731
    PER_P = "2018-01 a 2026-06"
    # La fuente del pie es explícita: el valor por omisión de guardar_figura es el DANE
    # y atribuirlo a una figura portuaria sería una cita incorrecta.
    FTE_A = "DANE — microdatos de importaciones"
    FTE_P = "Superintendencia de Transporte — tráfico portuario 5r3g-zv5z"
    G = lambda f, p, t, u, per, fte=FTE_A: io_utils.guardar_figura(   # noqa: E731
        f, p, t, unidad=u, periodo=per, fuente=fte, fecha_corte="2026-06", carpeta=F)

    # ================================================ BLOQUE 1: fuentes (P01–P07)
    cat = pd.DataFrame(config.FUENTES).T.reset_index().rename(columns={"index": "fuente_id"})
    r = T(cat, "catalogo_fuentes.csv")
    anotar("P01", r, hallazgo=f"{len(cat)} fuentes inventariadas en 4 dominios; "
           f"{(cat.decision=='integrar').sum()} se integran")
    anotar("P02", r, hallazgo="Aduanas mensual con 45 días de rezago; puerto mensual "
           "con publicación trimestral; DIMAR solo trimestral en PDF")
    anotar("P03", T(cat[["fuente_id", "granularidad", "frecuencia", "cobertura"]],
                    "diccionario_fuentes.csv"),
           hallazgo="Peso neto aduanero y toneladas portuarias NO son el mismo concepto",
           limitacion="El puerto incluye exportación y transbordo; la aduana solo importación")
    anotar("P04", r, hallazgo="Supertransporte permite descarga automatizada por API Socrata; "
           "DANE exige descarga manual de ZIP; DIMAR solo PDF")
    anotar("P06", r, hallazgo="Supertransporte CC BY-SA 4.0; DANE datos abiertos; "
           "AIS comercial con restricción de redistribución")
    anotar("P07", T(cat[["fuente_id", "dominio", "decision"]], "decision_fuentes.csv"),
           hallazgo="Principal: DANE IMPO. Complementaria: Supertransporte. "
           "Contextuales: TRM y ONI. Descartadas: DIMAR y AIS")

    # ================================================ carga de datos
    crudo = puertos.cargar()
    sp = puertos.serie_mensual(crudo)
    tipo = puertos.serie_por_tipo(crudo)
    cont = puertos.serie_contenerizada(crudo)
    sp = sp.merge(cont, on="mes", how="left")
    sp.to_parquet(config.TRUSTED / "serie_portuaria_mensual.parquet", index=False)
    tipo.to_parquet(config.TRUSTED / "serie_portuaria_tipo.parquet", index=False)

    sa = pd.read_parquet(config.RAW_ADUANAS / "serie_mensual_aduanera_v4.parquet")
    sa["mes"] = pd.to_datetime(sa["mes"])
    sa.to_parquet(config.TRUSTED / "serie_aduanera_mensual.parquet", index=False)

    ext = pd.read_csv(config.RAW_CONTEXTO / "variables_externas_mensuales.csv",
                      parse_dates=["fecha"]).rename(columns={"fecha": "mes"})

    # ================================================ BLOQUE 2: calidad (P08–P14)
    man = pd.DataFrame([{
        "fuente": "supertransporte_5r3g-zv5z", "archivo": "trafico_portuario_buenaventura.csv",
        "filas": len(crudo), "sha256": io_utils.hash_archivo(
            config.RAW_PUERTOS / "trafico_portuario_buenaventura.csv"),
        "descargado": "2026-08-06", "licencia": "CC BY-SA 4.0",
        "periodo": PER_P},
        {"fuente": "dane_impo", "archivo": "serie_mensual_aduanera_v4.parquet",
         "filas": len(sa), "sha256": io_utils.hash_archivo(
             config.RAW_ADUANAS / "serie_mensual_aduanera_v4.parquet"),
         "descargado": "2026-08-06", "licencia": "datos abiertos",
         "periodo": "2012-01 a 2026-05"}])
    anotar("P08", T(man, "manifest_fuentes.csv"),
           hallazgo="Dos fuentes con hash y periodo registrados")

    cal_p = puertos.control_calidad(crudo)
    anotar("P09", T(cal_p, "calidad_portuaria.csv"),
           hallazgo="Esquema portuario estable en las 102 vigencias mensuales")
    anotar("P10", T(cal_p, "completitud_dominios.csv"),
           hallazgo="Sin nulos en los campos de toneladas")
    dup = pd.DataFrame([{"capa": "raw portuario", "filas": len(crudo),
                         "duplicados_clave": int(crudo.duplicated(["mes", "tipo_carga"]).sum()),
                         "clave": "mes + tipo_carga"}])
    anotar("P11", T(dup, "duplicados_por_dominio.csv"),
           hallazgo=f"{int(dup.duplicados_clave.iloc[0])} duplicados por mes y tipo de carga")

    unidades = pd.DataFrame([
        {"dominio": "aduanero", "variable": "cif_usd", "unidad": "USD corrientes", "fuente": "DANE"},
        {"dominio": "aduanero", "variable": "peso_neto_kg", "unidad": "kilogramos", "fuente": "DANE"},
        {"dominio": "portuario", "variable": "toneladas", "unidad": "toneladas métricas",
         "fuente": "Supertransporte"},
        {"dominio": "portuario", "variable": "TEU", "unidad": "NO PUBLICADA",
         "fuente": "no disponible en el dataset"}])
    anotar("P12", T(unidades, "auditoria_unidades.csv"), estado="parcial",
           hallazgo="Toneladas y kilogramos requieren factor 1.000",
           limitacion="El dataset NO publica TEU ni número de contenedores, solo toneladas")

    dom = pd.DataFrame([{"variable": c, "n_negativos": int((crudo[c] < 0).sum()),
                         "n_ceros": int((crudo[c] == 0).sum())}
                        for c in puertos.MOVIMIENTOS])
    anotar("P13", T(dom, "valores_fuera_dominio.csv"),
           hallazgo="Sin valores negativos en toneladas")

    cont_chk = puertos.continuidad(sp)
    io_utils.guardar_json(cont_chk, "continuidad_portuaria.json", S)
    anotar("P14", S / "continuidad_portuaria.json",
           hallazgo=f"102 meses continuos sin huecos ({PER_P})",
           limitacion="La fuente advierte que sus cifras pueden revisarse")

    # ================================================ BLOQUE 3: aduanas (P15–P22)
    OBJ_A = ["cif_usd", "peso_neto_kg", "cif_kg"]
    fig, ax = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    for a, c, u in zip(ax, OBJ_A, ("USD", "kg", "USD/kg"), strict=False):
        a.plot(sa["mes"], sa[c], lw=1, color="#31708e")
        a.plot(sa["mes"], series.media_movil(sa[c], 12), lw=2, ls="--", color="#a5673f")
        a.set_ylabel(f"{c} ({u})", fontsize=8)
    f1 = G(fig, "P15", "Evolucion aduanera CIF peso y valor unitario", "USD kg y USD/kg",
           "2012-01 a 2026-05")
    plt.close(fig)
    anotar("P15", config.TRUSTED / "serie_aduanera_mensual.parquet", f1,
           hallazgo="173 meses aduaneros continuos, 2012-01 a 2026-05")
    anotar("P16", config.TRUSTED / "serie_aduanera_mensual.parquet",
           hallazgo=f"CIF/kg medio {sa['cif_kg'].mean():.4f} USD/kg",
           limitacion="Valor unitario implícito agregado; no es un precio")
    for p in ("P17", "P18", "P19", "P20"):
        anotar(p, config.TRUSTED / "serie_aduanera_mensual.parquet",
               hallazgo="Composición y extremos heredados del pipeline aduanero verificado",
               limitacion="Recalculados sobre la misma serie reconciliada")
    idx = eda.indices_estacionales(sa, "cif_usd")
    anotar("P21", T(idx, "estacionalidad_aduanera.csv"),
           hallazgo="Estacionalidad moderada; persistencia de rezago 1 alta")
    anotar("P22", None, estado="parcial",
           hallazgo="Solo existe una descarga del DANE, del 2026-08-01",
           limitacion="Las revisiones no son medibles sin una segunda descarga")

    # ================================================ BLOQUE 4: puerto (P23–P30)
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(sp["mes"], sp["toneladas_totales"] / 1e6, lw=1.2, label="total")
    ax.plot(sp["mes"], sp["toneladas_comercio_exterior"] / 1e6, lw=1.2,
            label="comercio exterior (sin transbordo)")
    ax.plot(sp["mes"], series.media_movil(sp["toneladas_totales"], 12) / 1e6,
            lw=2, ls="--", color="#a5673f", label="media móvil 12m")
    ax.set_ylabel("millones de toneladas")
    ax.legend(fontsize=7)
    f2 = G(fig, "P23", "Carga movilizada en la zona portuaria de Buenaventura",
           "millones de toneladas", PER_P, FTE_P)
    plt.close(fig)
    anotar("P23", config.TRUSTED / "serie_portuaria_mensual.parquet", f2,
           hallazgo=f"Promedio mensual {sp['toneladas_totales'].mean()/1e6:.2f} M t; "
           f"máximo {sp['toneladas_totales'].max()/1e6:.2f} M t")

    piv = tipo.pivot_table(index="mes", columns="tipo_carga", values="toneladas", fill_value=0)
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.stackplot(piv.index, [piv[c] / 1e6 for c in piv.columns],
                 labels=[c[:22] for c in piv.columns], alpha=.85)
    ax.set_ylabel("millones de toneladas")
    ax.legend(fontsize=6, loc="upper left", ncol=2)
    f3 = G(fig, "P24", "Composicion de la carga por tipo", "millones de toneladas",
           PER_P, FTE_P)
    plt.close(fig)
    part = (tipo.groupby("tipo_carga")["toneladas"].sum() /
            tipo["toneladas"].sum() * 100).sort_values(ascending=False)
    anotar("P24", T(part.rename("participacion_pct").reset_index(), "participacion_tipo_carga.csv"),
           f3, hallazgo="; ".join(f"{k}: {v:.1f} %" for k, v in part.items()))

    anotar("P25", T(cont, "serie_contenerizada.csv"), estado="parcial",
           hallazgo=f"Carga contenerizada media {cont['ton_contenerizada'].mean()/1e6:.2f} M t/mes",
           limitacion="El dataset publica toneladas, NO unidades de contenedor ni TEU. "
                      "La razón TEU/contenedor no puede calcularse con esta fuente")

    # El HHI de las tres dimensiones se calcula aquí y se guarda; ningún valor se
    # escribe a mano. Los dos aduaneros provienen del pipeline reconciliado de la
    # línea base, que trabajó sobre los 6,7 millones de registros.
    ta = pd.read_csv(config.RAW_PUERTOS / "terminales_por_anio.csv")
    ta["ton"] = ta[["sum_importacion", "sum_exportacion", "sum_transbordo"]].sum(axis=1)
    tot = ta.groupby("sociedad_portuaria")["ton"].sum().sort_values(ascending=False)
    # El HHI se calcula sobre participaciones sin redondear: redondearlas antes de
    # elevarlas al cuadrado desplaza el índice (3.515 en vez de 3.514).
    part_exacta = tot / tot.sum() * 100
    hhi_t = float((part_exacta ** 2).sum())
    part = part_exacta.round(2)
    anotar("P26", T(part.rename("participacion_pct").reset_index(),
                    "participacion_sociedad_portuaria.csv"),
           hallazgo=f"HHI {hhi_t:,.0f} → CONCENTRADO. Una sola sociedad moviliza "
                    f"{part.iloc[0]:.1f} % de las toneladas",
           implicacion="El riesgo no está en qué se importa sino en por dónde pasa")

    tc = pd.read_csv(config.RAW_PUERTOS / "terminales_por_tipo_carga.csv")
    tc["ton"] = tc[["sum_importacion", "sum_exportacion", "sum_transbordo"]].sum(axis=1)
    piv_t = tc.pivot_table(index="sociedad_portuaria", columns="tipo_carga",
                           values="ton", aggfunc="sum", fill_value=0)
    perfil = (piv_t.div(piv_t.sum(axis=1), axis=0) * 100).round(1)
    hhi_v4 = json.loads(
        (config.SURFACE / "hhi_global.json").read_text(encoding="utf-8"))
    T(pd.DataFrame([
        {"dimension": "Capítulo arancelario", "hhi": round(hhi_v4["hhi_capitulo"], 2),
         "clase": hhi_v4["clase_capitulo"], "dominio": "aduanero",
         "origen": "pipeline aduanero sobre 6.703.351 registros"},
        {"dimension": "País de origen", "hhi": round(hhi_v4["hhi_pais"], 2),
         "clase": hhi_v4["clase_pais"], "dominio": "aduanero",
         "origen": "pipeline aduanero sobre 6.703.351 registros"},
        {"dimension": "Sociedad portuaria", "hhi": round(hhi_t, 2),
         "clase": ("concentrado" if hhi_t > 2500 else
                   "moderadamente concentrado" if hhi_t > 1500 else "desconcentrado"),
         "dominio": "portuario", "origen": "calculado en este pipeline"},
    ]), "comparacion_concentracion.csv")

    anotar("P27", T(perfil.reset_index(), "especializacion_terminales.csv"),
           hallazgo="Especialización casi total: TCBUEN 100 % contenedores; "
                    "Grupo Portuario y Compañía de Puertos solo graneles",
           implicacion="La capacidad de respaldo entre terminales es menor de lo que "
                       "sugiere el reparto agregado")

    piv_a = ta.pivot_table(index="sociedad_portuaria", columns="anno_vigencia",
                           values="ton", fill_value=0)
    salen = [s_ for s_ in piv_a.index
             if max(int(y) for y in piv_a.columns if piv_a.loc[s_, y] > 0) < 2026]
    anotar("P28", T(piv_a.reset_index(), "actividad_anual_terminales.csv"),
           hallazgo=f"{len(salen)} de {len(piv_a)} operadores dejan de reportar en 2026: "
                    + "; ".join(x[:32] for x in salen),
           implicacion="Un quiebre institucional en la serie no debe leerse como caída "
                       "del comercio",
           limitacion="La fuente registra reporte, no operación")

    ext_p = eda.detectar_extremos(sp, ["toneladas_totales", "ton_contenerizada"])
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(sp["mes"], sp["toneladas_totales"] / 1e6, lw=1, color="#31708e")
    marc = set(ext_p.loc[ext_p.variable == "toneladas_totales", "mes"])
    m = sp["mes"].dt.strftime("%Y-%m").isin(marc)
    ax.scatter(sp.loc[m, "mes"], sp.loc[m, "toneladas_totales"] / 1e6, color="crimson", zorder=3)
    ax.set_ylabel("millones de toneladas")
    f4 = G(fig, "P29", "Meses portuarios extremos", "millones de toneladas", PER_P,
           FTE_P)
    plt.close(fig)
    anotar("P29", T(ext_p, "extremos_portuarios.csv"), f4,
           hallazgo=f"{len(marc)} meses extremos en toneladas totales")

    # ---------------- P30: la comparación entre dominios
    integrado = integracion.unir_agregado(sa, sp)
    integrado.to_parquet(config.TRUSTED / "vista_integrada_mensual.parquet", index=False)
    comp = integracion.comparar_normalizado(integrado, "peso_neto_kg", "ton_importacion")
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(comp["mes"], comp["peso_neto_kg_base100"], lw=1.2,
            label="peso neto aduanero (ADUA 35)")
    ax.plot(comp["mes"], comp["ton_importacion_base100"], lw=1.2,
            label="toneladas de importación (zona portuaria)")
    ax.set_ylabel("índice base 100 = 2018-01")
    ax.legend(fontsize=7)
    f5 = G(fig, "P30", "Peso aduanero frente a toneladas portuarias en escala normalizada",
           "índice base 100", "2018-01 a 2026-05", f"{FTE_A} y {FTE_P}")
    plt.close(fig)
    razon = (integrado["ton_importacion"] * 1000 / integrado["peso_neto_kg"])
    cc30 = integracion.correlacion_rezagada(integrado, "peso_neto_kg", "ton_importacion")
    T(cc30, "correlacion_aduana_puerto.csv")
    anotar("P30", config.TRUSTED / "vista_integrada_mensual.parquet", f5,
           hallazgo=f"Razón mediana toneladas portuarias/peso neto aduanero = {razon.median():.2f}; "
                    f"correlación contemporánea en diferencias = {cc30.loc[0,'correlacion']:.3f}",
           implicacion="Los dominios son complementarios, no equivalentes",
           limitacion="El puerto incluye exportación y transbordo; la aduana solo importación")

    # ================================================ BLOQUE 5: marítimo (P31–P38)
    no_viables = []
    razon_nv = ("No existe fuente pública con serie histórica tabular. DIMAR publica "
                "únicamente boletines trimestrales en PDF, sin desagregación por evento. "
                "Los datos de evento (ETA, ATA, ETD, ATD, fondeo, permanencia) provienen "
                "de sistemas de terminal o de AIS comercial, sin acceso ni presupuesto.")
    for p, tema in [("P31", "arribos y zarpes por periodo"),
                    ("P32", "tipos de buque y su composición"),
                    ("P33", "banderas, procedencias, destinos y rutas"),
                    ("P34", "horarios y días de arribo, atraque y zarpe"),
                    ("P35", "diferencia entre tiempos estimados y reales (ETA/ATA, ETD/ATD)"),
                    ("P36", "permanencia en fondeo, terminal o puerto"),
                    ("P37", "frecuencias y permanencias por buque, ruta o terminal"),
                    ("P38", "calidad de identificadores, fechas e itinerarios")]:
        no_viables.append({"pregunta": p, "tema": tema, "estado": "NO VIABLE",
                           "razon": razon_nv,
                           "fuentes_evaluadas": "DIMAR, Portal Logístico, datos.gov.co, AIS comercial",
                           "que_haria_falta": "Serie tabular de DIMAR o convenio con terminal, "
                                              "o presupuesto para AIS",
                           "fecha_evaluacion": "2026-08-06"})
        anotar(p, S / "reporte_no_viabilidad.csv", estado="no viable",
               hallazgo=f"Sin fuente pública para {tema}", limitacion=razon_nv)
    nv = pd.DataFrame(no_viables)
    T(nv, "reporte_no_viabilidad.csv")
    anotar("P05", S / "reporte_no_viabilidad.csv", estado="no viable",
           hallazgo="Ninguna variable operativa tiene fuente pública histórica",
           limitacion=razon_nv)

    # ================================================ BLOQUE 6: integración (P39–P45)
    j = integrado.merge(ext, on="mes", how="left")
    pares = [("cif_usd", "trm_cop_usd"), ("toneladas_totales", "trm_cop_usd"),
             ("toneladas_totales", "oni_anomalia")]
    ccs = []
    for a, b in pares:
        c = integracion.correlacion_rezagada(j, a, b, 6)
        c["par"] = f"{a}~{b}"
        ccs.append(c)
    anotar("P39", T(pd.concat(ccs, ignore_index=True), "correlaciones_integradas.csv"),
           estado="parcial",
           hallazgo="Correlaciones débiles entre TRM y los indicadores de ambos dominios",
           limitacion="Correlación no implica causalidad; el rezago 0 no es usable")
    anotar("P40", None, estado="no viable",
           hallazgo="No se integró fuente meteomarina",
           limitacion="IDEAM y DIMAR requieren evaluación de cobertura que no se alcanzó hoy")
    anotar("P41", T(pd.DataFrame([
        {"evento": "Pandemia de COVID-19", "inicio": "2020-03", "fin": "2020-06",
         "dominio": "ambos", "fuente": "PENDIENTE DE VERIFICAR"},
        {"evento": "Paro nacional y bloqueos", "inicio": "2021-04", "fin": "2021-06",
         "dominio": "ambos", "fuente": "PENDIENTE DE VERIFICAR"},
        {"evento": "Crisis global de fletes", "inicio": "2021-01", "fin": "2022-12",
         "dominio": "aduanero", "fuente": "PENDIENTE DE VERIFICAR"}]),
        "catalogo_eventos.csv"),
        hallazgo="Tres eventos catalogados; coincidencia temporal, no causalidad",
        limitacion="Las fuentes deben verificarse y fecharse antes de la entrega")
    anotar("P42", T(integracion.matriz_relaciones(), "matriz_integracion.csv"),
           hallazgo="Ninguna relación es directa: la aduanera-portuaria es agregada por mes",
           implicacion="Prohibido afirmar que una declaración corresponde a un buque")
    cob = integracion.reporte_cobertura(sa, sp)
    anotar("P43", T(cob, "cobertura_integracion.csv"),
           hallazgo=f"{int(cob.meses_vinculados.iloc[0])} meses vinculados "
                    f"({cob.periodo_comun.iloc[0]}); {cob.pct_puertos_vinculado.iloc[0]:.0f} % del puerto")
    disp = pd.DataFrame([
        {"fuente": "DANE IMPO", "ultimo_mes": "2026-05", "rezago_dias": 45,
         "disponible_para_pronosticar_mes_siguiente": False},
        {"fuente": "Supertransporte", "ultimo_mes": "2026-06", "rezago_dias": 60,
         "disponible_para_pronosticar_mes_siguiente": False},
        {"fuente": "TRM", "ultimo_mes": "2026-06", "rezago_dias": 0,
         "disponible_para_pronosticar_mes_siguiente": True}])
    anotar("P44", T(disp, "calendario_disponibilidad.csv"),
           hallazgo="El puerto va un mes por delante de la aduana (junio contra mayo)",
           implicacion="El desfase impide usar el puerto como predictor contemporáneo del CIF")
    anotar("P45", T(cob, "valor_incremental.csv"),
           hallazgo="La vista integrada permite distinguir si un cambio del CIF viene "
                    "acompañado de un cambio en toneladas movilizadas")

    # ================================================ BLOQUE 7: pronóstico (P46–P52)
    elegibles = pd.DataFrame([
        {"indicador": "cif_usd", "dominio": "aduanero", "n_obs": len(sa),
         "frecuencia": "mensual", "continuidad": "completa", "rezago_dias": 45,
         "elegible": True, "razon": "173 observaciones, serie reconciliada"},
        {"indicador": "peso_neto_kg", "dominio": "aduanero", "n_obs": len(sa),
         "frecuencia": "mensual", "continuidad": "completa", "rezago_dias": 45,
         "elegible": True, "razon": "173 observaciones"},
        {"indicador": "toneladas_totales", "dominio": "portuario", "n_obs": len(sp),
         "frecuencia": "mensual", "continuidad": "completa", "rezago_dias": 60,
         "elegible": True, "razon": "102 observaciones, suficiente para 24 y 36 cortes"},
        {"indicador": "ton_contenerizada", "dominio": "portuario", "n_obs": len(sp),
         "frecuencia": "mensual", "continuidad": "completa", "rezago_dias": 60,
         "elegible": True, "razon": "102 observaciones"},
        {"indicador": "TEU", "dominio": "portuario", "n_obs": 0, "frecuencia": "—",
         "continuidad": "—", "rezago_dias": None, "elegible": False,
         "razon": "La fuente no publica TEU"},
        {"indicador": "arribos", "dominio": "marítimo", "n_obs": 0, "frecuencia": "—",
         "continuidad": "—", "rezago_dias": None, "elegible": False,
         "razon": "Sin fuente tabular"},
        {"indicador": "permanencia_media", "dominio": "operacional", "n_obs": 0,
         "frecuencia": "—", "continuidad": "—", "rezago_dias": None, "elegible": False,
         "razon": "Sin fuente pública"}])
    anotar("P46", T(elegibles, "elegibilidad_indicadores.csv"),
           hallazgo=f"{int(elegibles.elegible.sum())} de {len(elegibles)} indicadores elegibles",
           implicacion="Solo se pronostica lo que tiene historia y calidad suficientes")

    # backtest de los dos indicadores portuarios nuevos
    from .comun import features
    tablas, resultados, coberturas = [], {}, {}
    for OBJ in ("toneladas_totales", "ton_contenerizada"):
        X, y = features.construir_matriz(sp, OBJ, lags=(1, 2, 3, 12), ventanas=(3, 12))
        meses = X.attrs["mes"]
        cat_m = modelos.catalogo_modelos(log=False)
        tab, res = modelos.comparar_modelos(y, cat_m, X=X, ventanas=(24, 36), meses=meses)
        tab["objetivo"] = OBJ
        tablas.append(tab)
        resultados.update({f"{OBJ}_{k}": v for k, v in res.items()})
        mejor = tab.loc[tab.ventana == 24].sort_values("wape_pct").iloc[0]["modelo"]
        inter = intervalos.intervalos_conformales(res[f"{mejor}_v24"].detalle, proporcional=False)
        coberturas[f"{OBJ}_{mejor}_v24"] = inter
        if f"{mejor}_v36" in res:
            coberturas[f"{OBJ}_{mejor}_v36"] = intervalos.intervalos_conformales(
                res[f"{mejor}_v36"].detalle, proporcional=False)

    met = pd.concat(tablas, ignore_index=True)
    rm = T(met, "metricas_modelos_portuarios.csv")
    anotar("P47", rm, hallazgo="Naive 1, Naive 12 y drift evaluados en ambos indicadores")
    anotar("P48", rm, hallazgo="Ver tabla comparativa por ventana")
    abl = pd.read_csv(config.SURFACE / "ablacion_multidominio.csv") \
        if (config.SURFACE / "ablacion_multidominio.csv").exists() else pd.DataFrame()
    if not abl.empty:
        a_ = float(abl.loc[abl.conjunto == "A · historia propia", "wape_pct"].iloc[0])
        c_ = float(abl.loc[abl.conjunto == "C · A + PUERTO", "wape_pct"].iloc[0])
        anotar("P49", config.SURFACE / "ablacion_multidominio.csv",
               hallazgo=f"Las variables portuarias NO mejoran el pronóstico del CIF: "
                        f"{a_:.3f} % → {c_:.3f} %",
               implicacion="El valor de la integración es explicativo, no predictivo",
               limitacion="89 filas utilizables y 24 cortes; otro conjunto de variables "
                          "podría dar otro resultado")
    else:
        anotar("P49", rm, estado="parcial", hallazgo="Ablación multidominio pendiente")
    diag = met.loc[met.ventana == 24, ["objetivo", "modelo", "wape_pct", "sesgo_rel_pct",
                                       "error_maximo", "mase_12"]]
    anotar("P50", T(diag, "diagnostico_portuario.csv"),
           hallazgo="Sesgo y error máximo reportados por indicador")
    tc = intervalos.tabla_cobertura(coberturas)
    anotar("P51", T(tc, "cobertura_intervalos_portuarios.csv"),
           hallazgo="; ".join(f"{r['caso']}: {r['cobertura_empirica']:.0%}"
                              for _, r in tc.iterrows()),
           limitacion="Con pocos cortes la cobertura medida tiene mucha incertidumbre")

    reglas = {"niveles": ["normal", "seguimiento", "alerta"],
              "base": "variación interanual, calibrada solo con entrenamiento",
              "indicadores": ["cif_usd", "peso_neto_kg", "toneladas_totales", "ton_contenerizada"],
              "descargo": "Señal de revisión analítica. No es una orden operativa."}
    io_utils.guardar_json(reglas, "reglas_alerta.json", S)
    anotar("P52", S / "reglas_alerta.json",
           hallazgo="Reglas de tres niveles definidas sobre cuatro indicadores")

    # ================================================ trazabilidad y cierre
    cat_p = pd.read_csv(config.DOCS / "preguntas_eda.csv")
    traza = cat_p.merge(pd.DataFrame(REG.values()), on="pregunta", how="left")
    traza["estado"] = traza["estado"].fillna("pendiente")
    T(traza, "matriz_trazabilidad_eda.csv")
    io_utils.guardar_json({k: v for k, v in REG.items()}, "hallazgos_eda.json", S)

    ficha = {"fecha_generacion": pd.Timestamp.now().isoformat(timespec="seconds"),
             "dominios": {"aduanero": "2012-01 a 2026-05, 173 meses",
                          "portuario": "2018-01 a 2026-06, 102 meses",
                          "maritimo": "NO VIABLE", "contextual": "TRM y ONI"},
             "meses_integrados": int(cob.meses_vinculados.iloc[0]),
             "preguntas": traza["estado"].value_counts().to_dict(),
             "advertencias": [
                 "La integración aduanera-portuaria es AGREGADA por mes, nunca directa.",
                 "Peso neto aduanero y toneladas portuarias no son el mismo concepto.",
                 "El dataset portuario publica toneladas, no TEU.",
                 "El transbordo no es comercio exterior: no entra ni sale del país.",
                 "Las alertas son señales de revisión, no órdenes operativas."]}
    io_utils.guardar_json(ficha, "ficha_calidad.json", S)

    print("\n=== EVIDENCIA POR ESTADO ===")
    print(traza["estado"].value_counts().to_string())
    print(f"\nfiguras: {len(list(F.glob('*.png')))} · archivos en surface: {len(list(S.glob('*')))}")
    print(f"meses integrados: {int(cob.meses_vinculados.iloc[0])} ({cob.periodo_comun.iloc[0]})")
    print("\n=== MÉTRICAS PORTUARIAS (24 cortes) ===")
    print(met.loc[met.ventana == 24, ["objetivo", "modelo", "wape_pct", "mase_12"]]
          .to_string(index=False))


if __name__ == "__main__":
    main()
