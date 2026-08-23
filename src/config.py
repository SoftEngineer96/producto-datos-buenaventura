"""Rutas, fuentes y parámetros del proyecto integrado.

Reemplaza al config de V4: cambian las rutas y el universo de fuentes.
"""
from __future__ import annotations

import os
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
DATA = RAIZ / "data"
RAW = DATA / "raw"
RAW_ADUANAS, RAW_PUERTOS = RAW / "aduanas", RAW / "puertos"
RAW_MARITIMO, RAW_CONTEXTO = RAW / "maritimo", RAW / "contexto"
LANDING, TRUSTED, SURFACE = DATA / "landing", DATA / "trusted", DATA / "surface"
REPORTS = RAIZ / "reports"
FIGURES, TABLES, DOCUMENTS = REPORTS / "figures", REPORTS / "tables", REPORTS / "documents"
MODELS, DOCS, DASHBOARD = RAIZ / "models", RAIZ / "docs", RAIZ / "dashboard"


# --------------------------------------------------------------------------------------
# Entorno de ejecución: local, repositorio clonado, Google Colab y Google Drive.
#
# Se amplía la misma lógica de rutas que ya usaba el proyecto en lugar de introducir un
# segundo sistema de configuración. `RAIZ` se sigue derivando del propio archivo, de modo
# que un clon de GitHub funciona sin tocar nada. Lo único que puede cambiar es dónde vive
# la carpeta de datos, que es lo que exige Drive.
# --------------------------------------------------------------------------------------
def en_colab() -> bool:
    """True si el proceso corre dentro de Google Colab."""
    import sys
    return "google.colab" in sys.modules or bool(os.environ.get("COLAB_RELEASE_TAG"))


def usar_datos_en(raiz_datos: str | Path) -> Path:
    """Reapunta las capas de datos a otra ubicación, por ejemplo Google Drive.

    Reasigna las mismas constantes que ya usa todo el pipeline; no crea nombres nuevos
    ni un segundo juego de rutas. Devuelve la nueva raíz de datos.

        from src import config
        config.usar_datos_en("/content/drive/MyDrive/buenaventura/data")
    """
    global DATA, RAW, RAW_ADUANAS, RAW_PUERTOS, RAW_MARITIMO, RAW_CONTEXTO
    global LANDING, TRUSTED, SURFACE
    DATA = Path(raiz_datos)
    RAW = DATA / "raw"
    RAW_ADUANAS, RAW_PUERTOS = RAW / "aduanas", RAW / "puertos"
    RAW_MARITIMO, RAW_CONTEXTO = RAW / "maritimo", RAW / "contexto"
    LANDING, TRUSTED, SURFACE = DATA / "landing", DATA / "trusted", DATA / "surface"
    asegurar_directorios()
    return DATA


def montar_drive(subcarpeta: str = "buenaventura") -> Path | None:
    """Monta Google Drive y apunta las capas de datos allí. Sin efecto fuera de Colab.

    Los datos quedan persistentes entre sesiones de Colab, que es el problema real:
    el sistema de archivos de Colab se borra al cerrar el entorno.
    """
    if not en_colab():
        print("No se está ejecutando en Colab; las rutas de datos no se modifican.")
        return None
    from google.colab import drive
    drive.mount("/content/drive", force_remount=False)
    destino = Path("/content/drive/MyDrive") / subcarpeta / "data"
    usar_datos_en(destino)
    print(f"Datos persistentes en: {destino}")
    return destino


# Permite fijar la ubicación de los datos sin tocar código, útil en integración continua.
if os.environ.get("BUENAVENTURA_DATA"):
    usar_datos_en(os.environ["BUENAVENTURA_DATA"])


def asegurar_directorios() -> None:
    for d in (RAW_ADUANAS, RAW_PUERTOS, RAW_MARITIMO, RAW_CONTEXTO, LANDING,
              TRUSTED, SURFACE, FIGURES, TABLES, DOCUMENTS, MODELS, DOCS, DASHBOARD):
        d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------- dominio
ADUA_OBJETIVO = 35
ZONA_PORTUARIA = "BUENAVENTURA"

TIPOS_CARGA = ("CONTENEDORES", "GENERAL", "GRANEL LIQUIDO",
               "GRANEL SOLIDO DIFER. DE CARBON", "CARBON AL GRANEL")

# Constantes que esperan los módulos heredados de V4 (src/comun). Se conservan con el
# mismo nombre que allí para no romper sus dependencias: el principio es adaptar la
# configuración a los componentes que ya funcionan, no renombrarlos.
OBJETIVOS = ("cif_usd", "peso_neto_kg")
DERIVADA = "cif_kg"                 # valor unitario implícito. NO es un precio.
ADUA_COLUMNA = "adua"

COLUMNAS_CANONICAS = {
    "adua": "adua", "fecha": "fecha", "pais_origen": "pais_origen",
    "subpartida": "subpartida", "capitulo": "capitulo", "cif_usd": "cif_usd",
    "fob_usd": "fob_usd", "peso_neto_kg": "peso_neto_kg", "peso_bruto_kg": "peso_bruto_kg",
}
CODIGOS_TEXTO = {"adua": 2, "pais_origen": 3, "subpartida": 10, "capitulo": 2}

# Catálogo de las 52 preguntas utilizado por el pipeline y la trazabilidad.
CATALOGO_PREGUNTAS = DOCS / "preguntas_eda.csv"

SEMILLA = 20260806
ESTACIONALIDAD = 12
NIVEL_NOMINAL = 0.80
MIN_ENTRENAMIENTO = 36
VENTANAS_BACKTEST = (24, 36)
TOLERANCIA_RECONCILIACION = 0.005

# --------------------------------------------------------------------- fuentes
FUENTES = {
    "dane_impo": {
        "dominio": "aduanero",
        "entidad": "DANE",
        "nombre": "Estadísticas de importaciones (IMPO), microdatos anonimizados",
        "url": "https://microdatos.dane.gov.co/index.php/catalog/856",
        "formato": "CSV dentro de ZIP anidados",
        "granularidad": "declaración",
        "frecuencia": "mensual",
        "cobertura": "2012-01 a 2026-05",
        "rezago_dias": 45,
        "licencia": "Datos abiertos, uso público con cita de fuente",
        "decision": "integrar",
    },
    "supertransporte_trafico": {
        "dominio": "portuario",
        "entidad": "Superintendencia de Transporte",
        "nombre": "Tráfico Portuario Marítimo en Colombia (5r3g-zv5z)",
        "url": "https://www.datos.gov.co/resource/5r3g-zv5z.csv",
        "formato": "CSV vía API Socrata",
        "granularidad": "zona portuaria × sociedad portuaria × tipo de carga × mes",
        "frecuencia": "mensual (publicación trimestral)",
        "cobertura": "2018-01 a 2026-06",
        "rezago_dias": 60,
        "licencia": "CC BY-SA 4.0",
        "decision": "integrar",
    },
    "banrep_trm": {
        "dominio": "contextual",
        "entidad": "Banco de la República",
        "nombre": "Tasa Representativa del Mercado",
        "url": "https://www.banrep.gov.co/es/estadisticas/trm",
        "formato": "CSV",
        "granularidad": "diaria",
        "frecuencia": "diaria",
        "cobertura": "2012-01 a 2026-06",
        "rezago_dias": 0,
        "licencia": "Datos abiertos",
        "decision": "contexto",
    },
    "noaa_oni": {
        "dominio": "contextual",
        "entidad": "NOAA Climate Prediction Center",
        "nombre": "Oceanic Niño Index",
        "url": "https://origin.cpc.ncep.noaa.gov/products/analysis_monitoring/ensostuff/ONI_v5.php",
        "formato": "CSV",
        "granularidad": "mensual",
        "frecuencia": "mensual",
        "cobertura": "2012-01 a 2026-06",
        "rezago_dias": 15,
        "licencia": "Dominio público",
        "decision": "contexto",
    },
    "dimar_trafico": {
        "dominio": "marítimo",
        "entidad": "DIMAR",
        "nombre": "Estadísticas trimestrales de tráfico y transporte marítimo",
        "url": "https://www.dimar.mil.co/operaciones-estadisticas/estadisticas-de-trafico-maritimo-internacional/publicaciones-estadisticas-trimestrales-de-trafico-maritimo",
        "formato": "PDF",
        "granularidad": "trimestral, agregada",
        "frecuencia": "trimestral",
        "cobertura": "por verificar",
        "rezago_dias": 90,
        "licencia": "Uso público con cita",
        "decision": "no integrada — solo PDF, sin serie tabular descargable",
    },
    "ais_comercial": {
        "dominio": "operacional",
        "entidad": "proveedores comerciales de AIS",
        "nombre": "Posición y eventos de buques (ETA/ATA/ETD/ATD)",
        "url": "—",
        "formato": "API de pago",
        "granularidad": "evento",
        "frecuencia": "tiempo real",
        "cobertura": "—",
        "rezago_dias": 0,
        "licencia": "comercial, con restricciones de redistribución",
        "decision": "descartada — sin acceso ni presupuesto",
    },
}
