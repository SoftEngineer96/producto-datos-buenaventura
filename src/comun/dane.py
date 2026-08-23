"""Lector de los microdatos de importaciones del DANE.

El formato cambia entre vigencias. Verificado sobre los paquetes 2012–2026:

| Vigencia | Separador | Decimal | Ceros iniciales | Ausentes            |
|----------|-----------|---------|-----------------|---------------------|
| 2012–2016| coma      | punto   | perdidos        | 1.79769313486232e+308 |
| 2017–2020| coma      | punto   | conservados     | vacío               |
| 2021–2026| punto y coma | coma | conservados     | vacío               |

El centinela 1.797…e+308 es el máximo de un flotante de doble precisión: es el
«missing» de Stata/SPSS filtrado al CSV. Tratarlo como número contamina cualquier
agregación, así que se convierte a nulo antes de operar.

Estructura de los paquetes: `Impo_AAAA.zip` → zip mensual → `Mes.csv`. El nombre
del zip mensual varía (`Impo_2012/Enero.zip`, `01. Enero 2026.zip`, `Mayo 2025.zip`),
por lo que se resuelve por coincidencia del nombre del mes, no por posición.
"""
from __future__ import annotations

import io
import re
import unicodedata
import zipfile
from pathlib import Path

import pandas as pd

# Centinela de valor ausente heredado de Stata/SPSS.
CENTINELA = 1.79769313486232e+308
UMBRAL_CENTINELA = 1e300

MESES = {1: "enero", 2: "febrero", 3: "marzo", 4: "abril", 5: "mayo", 6: "junio",
         7: "julio", 8: "agosto", 9: "septiembre", 10: "octubre",
         11: "noviembre", 12: "diciembre"}

# Nombres del DANE → nombres canónicos del proyecto.
COLUMNAS = {
    "FECH": "fech", "ADUA": "adua", "PAISGEN": "pais_origen", "PAISPRO": "pais_procedencia",
    "PAISCOM": "pais_compra", "DEPTODES": "depto_destino", "VIATRANS": "via_transporte",
    "PBK": "peso_bruto_kg", "PNK": "peso_neto_kg", "NABAN": "subpartida",
    "VAFODO": "fob_usd", "FLETE": "flete_usd", "VACID": "cif_usd", "SEGUROS": "seguros_usd",
    "CODA": "unidad", "REGIMEN": "regimen",
}
NUMERICAS = ["peso_bruto_kg", "peso_neto_kg", "fob_usd", "flete_usd", "cif_usd", "seguros_usd"]
TEXTO = ["adua", "pais_origen", "pais_procedencia", "pais_compra", "subpartida", "unidad"]


def _norm(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()


def _limpiar_encabezado(c: str) -> str:
    """Normaliza un nombre de columna.

    Las vigencias 2017 y 2019 traen marca de orden de bytes (BOM) que, leída como
    latin-1, aparece como los caracteres «Ï»¿» pegados al primer nombre. Si no se
    quita, la columna FECH deja de encontrarse y se pierde la fecha del registro.
    """
    c = c.strip().replace('"', "").replace("\ufeff", "")
    c = re.sub(r"^[^A-Za-z]+", "", c)
    return c.upper()


def detectar_separador(cabecera: str) -> str:
    """El separador es el que más veces aparece en la línea de cabecera."""
    return ";" if cabecera.count(";") > cabecera.count(",") else ","


def a_numero(s: pd.Series, separador: str) -> pd.Series:
    """Convierte a float respetando la convención decimal de cada vigencia.

    Con separador de campos ';' el decimal es coma y el millar es punto.
    Con ',' el decimal es punto. El centinela de Stata se vuelve nulo.
    """
    x = s.astype("string").str.strip().str.replace('"', "", regex=False)
    if separador == ";":
        x = x.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    x = pd.to_numeric(x, errors="coerce")
    return x.mask(x.abs() >= UMBRAL_CENTINELA)


def fecha_desde_fech(fech: pd.Series) -> pd.Series:
    """FECH viene como AAMM (1201 = enero de 2012, 2605 = mayo de 2026)."""
    f = fech.astype("string").str.strip().str.replace('"', "", regex=False).str.zfill(4)
    anio = 2000 + pd.to_numeric(f.str[:2], errors="coerce")
    mes = pd.to_numeric(f.str[2:], errors="coerce")
    return pd.to_datetime(dict(year=anio, month=mes, day=1), errors="coerce")


def rutas_mensuales(zip_anual: str | Path) -> dict[str, str]:
    """Mapa {mes: ruta interna} dentro del paquete anual.

    Coexisten dos disposiciones en los paquetes del DANE:
      * zip anidado por mes  (`Impo_2012/Enero.zip`, `01. Enero 2026.zip`);
      * CSV suelto en carpeta (`Agosto 2024/Agosto.csv`, sin zip intermedio).
    Se admiten ambas y se resuelve por el nombre del mes, no por posición.
    """
    with zipfile.ZipFile(zip_anual) as z:
        nombres = z.namelist()
    internos = [n for n in nombres if n.lower().endswith(".zip")]
    if not internos:
        internos = [n for n in nombres if n.lower().endswith(".csv")]
    salida = {}
    for n in internos:
        etiqueta = _norm(n)          # incluye la carpeta, que a veces lleva el mes
        for nombre in MESES.values():
            if nombre in etiqueta and nombre not in salida:
                salida[nombre] = n
                break
    return salida


def leer_mes(zip_anual: str | Path, mes: int, *, columnas: dict | None = None
             ) -> tuple[pd.DataFrame, dict]:
    """Lee un mes de un paquete anual y devuelve (DataFrame homologado, metadatos)."""
    columnas = columnas or COLUMNAS
    zip_anual = Path(zip_anual)
    mapa = rutas_mensuales(zip_anual)
    objetivo = MESES[mes]
    if objetivo not in mapa:
        raise FileNotFoundError(f"{objetivo} no está en {zip_anual.name}")

    ruta_interna = mapa[objetivo]
    with zipfile.ZipFile(zip_anual) as z:
        crudo = z.read(ruta_interna)
    if ruta_interna.lower().endswith(".csv"):
        # El paquete guarda el CSV directamente, sin zip mensual intermedio.
        contenedor = None
        nombre_csv = ruta_interna
        abrir = lambda: io.BytesIO(crudo)  # noqa: E731
    else:
        contenedor = zipfile.ZipFile(io.BytesIO(crudo))
        csvs = [n for n in contenedor.namelist() if n.lower().endswith(".csv")]
        if not csvs:
            raise FileNotFoundError(f"Sin CSV dentro de {ruta_interna}")
        nombre_csv = csvs[0]
        abrir = lambda: contenedor.open(nombre_csv)  # noqa: E731

    with abrir() as f:
        cabecera = f.readline().decode("latin-1", errors="replace")
    sep = detectar_separador(cabecera)
    n_descartadas = 0
    try:
        with abrir() as f:
            df = pd.read_csv(f, sep=sep, encoding="latin-1", dtype=str,
                             low_memory=False, quotechar='"')
    except pd.errors.ParserError:
        # Defectos reales de la fuente: campos con el separador sin entrecomillar
        # y, en 2023, saltos de línea dentro de un campo de texto. Las filas
        # afectadas se descartan y se cuentan; nunca se reparan en silencio.
        with abrir() as f:
            df = pd.read_csv(f, sep=sep, encoding="latin-1", dtype=str,
                             low_memory=False, quotechar='"', on_bad_lines="skip")
        with abrir() as f:
            n_lineas = sum(1 for _ in f) - 1
        n_descartadas = max(0, n_lineas - len(df))
    finally:
        if contenedor is not None:
            contenedor.close()

    df.columns = [_limpiar_encabezado(c) for c in df.columns]
    meta = {"paquete": zip_anual.name, "archivo": nombre_csv, "separador": sep,
            "lineas_descartadas_por_formato": n_descartadas,
            "decimal": "coma" if sep == ";" else "punto",
            "n_registros_nacional": len(df),
            "columnas_originales": len(df.columns)}

    presentes = {k: v for k, v in columnas.items() if k in df.columns}
    faltantes = [k for k in columnas if k not in df.columns]
    meta["columnas_faltantes"] = faltantes

    out = df[list(presentes)].rename(columns=presentes)
    for c in NUMERICAS:
        if c in out.columns:
            out[c] = a_numero(out[c], sep)
    for c in TEXTO:
        if c in out.columns:
            out[c] = (out[c].astype("string").str.strip()
                      .str.replace('"', "", regex=False)
                      .replace({"": pd.NA}))
            # el centinela también aparece en columnas de código
            out[c] = out[c].mask(out[c].str.startswith("1.797", na=False))
    if "fech" in out.columns:
        out["fecha"] = fecha_desde_fech(out["fech"])
        out = out.drop(columns=["fech"])
    if "subpartida" in out.columns:
        out["subpartida"] = out["subpartida"].str.zfill(10)
        out["capitulo"] = out["subpartida"].str[:2]

    # Filas partidas por un salto de línea dentro de un campo quedan sin fecha o
    # sin valor. Se cuentan aquí para que la bitácora de exclusiones (P08) las
    # recoja en lugar de que desaparezcan sin dejar rastro.
    meta["filas_sin_fecha"] = int(out["fecha"].isna().sum()) if "fecha" in out else 0
    meta["filas_sin_cif"] = int(out["cif_usd"].isna().sum()) if "cif_usd" in out else 0
    return out, meta


def filtrar_adua(df: pd.DataFrame, adua: int | str = 35) -> pd.DataFrame:
    """Filtra por código de aduana normalizando ceros a la izquierda."""
    objetivo = str(adua).strip().lstrip("0") or "0"
    s = df["adua"].astype("string").str.strip().str.lstrip("0")
    return df.loc[s == objetivo].copy()


def paquetes(carpeta: str | Path) -> list[Path]:
    """Paquetes anuales ordenados cronológicamente."""
    ps = sorted(Path(carpeta).glob("Impo_*.zip"))
    def clave(p):
        m = re.search(r"(\d{4})(?:_(\d))?", p.stem)
        return (int(m.group(1)), int(m.group(2) or 0)) if m else (0, 0)
    return sorted(ps, key=clave)
