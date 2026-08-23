"""Entrada/salida con trazabilidad: hash, metadatos y guardado de figuras.

Regla del proyecto: ninguna cifra llega al documento sin pasar por un archivo.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .. import config


def ahora_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def hash_archivo(ruta: str | Path, algoritmo: str = "sha256", bloque: int = 1 << 20) -> str:
    """Hash de integridad de un archivo, leído por bloques."""
    h = hashlib.new(algoritmo)
    with open(ruta, "rb") as f:
        for chunk in iter(lambda: f.read(bloque), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_dataframe(df: pd.DataFrame) -> str:
    """Hash reproducible del contenido de un DataFrame (orden de filas incluido)."""
    return hashlib.sha256(
        pd.util.hash_pandas_object(df, index=True).values.tobytes()
    ).hexdigest()


def guardar_tabla(df: pd.DataFrame, nombre: str, carpeta: Path | None = None,
                  indice: bool = False) -> Path:
    """Guarda una tabla en CSV o Parquet según la extensión y devuelve la ruta."""
    carpeta = carpeta or config.TABLES
    carpeta.mkdir(parents=True, exist_ok=True)
    ruta = carpeta / nombre
    if ruta.suffix == ".parquet":
        df.to_parquet(ruta, index=indice)
    elif ruta.suffix in (".csv", ".txt"):
        df.to_csv(ruta, index=indice, encoding="utf-8")
    else:
        raise ValueError(f"Extensión no soportada: {ruta.suffix}")
    return ruta


def guardar_json(obj: Any, nombre: str, carpeta: Path | None = None) -> Path:
    carpeta = carpeta or config.TABLES
    carpeta.mkdir(parents=True, exist_ok=True)
    ruta = carpeta / nombre
    ruta.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str),
                    encoding="utf-8")
    return ruta


def guardar_figura(fig, pregunta: str, titulo: str, *, unidad: str, periodo: str,
                   fuente: str = "DANE — microdatos de importaciones",
                   fecha_corte: str = "", carpeta: Path | None = None) -> Path:
    """Guarda una figura con pie obligatorio: unidad, periodo, fuente y corte.

    El nombre del archivo lleva el identificador de la pregunta (P01–P52), que es
    lo que permite enlazarla con la matriz de trazabilidad.
    """
    carpeta = carpeta or config.FIGURES
    carpeta.mkdir(parents=True, exist_ok=True)
    pie = (f"Unidad: {unidad} · Periodo: {periodo} · Fuente: {fuente}"
           f" · Fecha de corte: {fecha_corte or 'sin definir'}")
    fig.suptitle(f"{pregunta}. {titulo}", fontsize=11, y=0.995)
    fig.text(0.01, 0.005, pie, fontsize=7, va="bottom", ha="left", wrap=True)
    ruta = carpeta / f"{pregunta}_{_slug(titulo)}.png"
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    return ruta


def _slug(texto: str, maximo: int = 60) -> str:
    import re
    import unicodedata
    t = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode()
    t = re.sub(r"[^a-zA-Z0-9]+", "_", t).strip("_").lower()
    return t[:maximo]


def manifiesto_de(rutas: list[Path], fuente_id: str = "") -> pd.DataFrame:
    """Construye el manifiesto de fuentes exigido por P01."""
    filas = []
    for r in sorted(rutas):
        r = Path(r)
        if not r.is_file():
            continue
        meta = config.FUENTES.get(fuente_id, {})
        filas.append({
            "fuente_id": fuente_id,
            "nombre_fuente": meta.get("nombre", ""),
            "url": meta.get("url", ""),
            "archivo": r.name,
            "ruta_relativa": str(r.relative_to(config.RAIZ)),
            "tamano_bytes": r.stat().st_size,
            "sha256": hash_archivo(r),
            "fecha_descarga": datetime.fromtimestamp(r.stat().st_mtime).isoformat(timespec="seconds"),
            "fecha_registro": ahora_iso(),
            "licencia": meta.get("licencia", ""),
        })
    return pd.DataFrame(filas)
