"""Registro de evidencia por pregunta P01–P52.

Reglas de trazabilidad:
  * los identificadores P01–P52 no se renumeran;
  * una pregunta no está ejecutada si existe el código pero no la salida;
  * cada salida se enlaza con archivo, figura y sección del documento.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .. import config
from . import io_utils

CATALOGO = getattr(config, "CATALOGO_PREGUNTAS", config.DOCS / "preguntas_eda.csv")
REGISTRO = config.SURFACE / "matriz_trazabilidad.csv"
HALLAZGOS = config.SURFACE / "hallazgos_eda.json"

# El catálogo declara pregunta, bloque, enunciado y estado previsto. Se exige
# solo el núcleo común para que el módulo siga siendo fácil de mantener.
_COLUMNAS = ["pregunta", "bloque", "enunciado"]


def cargar_catalogo() -> pd.DataFrame:
    """Catálogo oficial de las 52 preguntas."""
    if not CATALOGO.exists():
        raise FileNotFoundError(
            f"Falta el catálogo de preguntas en {CATALOGO}. "
            "Se genera con docs/preguntas_p01_p52.csv del repositorio."
        )
    df = pd.read_csv(CATALOGO, dtype=str).fillna("")
    faltan = [c for c in _COLUMNAS if c not in df.columns]
    if faltan:
        raise ValueError(f"El catálogo no tiene las columnas {faltan}")
    return df


def validar_ids(df: pd.DataFrame | None = None) -> None:
    """Verifica que los identificadores sean exactamente P01…P52, sin huecos."""
    df = cargar_catalogo() if df is None else df
    esperados = [f"P{i:02d}" for i in range(1, 53)]
    obtenidos = list(df["pregunta"])
    if obtenidos != esperados:
        faltantes = sorted(set(esperados) - set(obtenidos))
        sobrantes = sorted(set(obtenidos) - set(esperados))
        raise AssertionError(
            f"Numeración alterada. Faltan {faltantes}; sobran {sobrantes}. "
            "Renumerar exige actualizar el Anexo A del EDA en el mismo commit."
        )


class Trazador:
    """Acumula la evidencia de cada pregunta y la vuelca a disco.

    Uso típico dentro del notebook de EDA::

        tz = Trazador()
        tz.registrar("P24", archivo=ruta_csv, figura=ruta_png,
                     hallazgo="...", interpretacion="...",
                     implicacion="...", limitacion="...")
        tz.exportar()
    """

    def __init__(self, catalogo: pd.DataFrame | None = None):
        self.catalogo = catalogo if catalogo is not None else cargar_catalogo()
        validar_ids(self.catalogo)
        self._filas: dict[str, dict] = {}

    def registrar(self, pregunta: str, *, archivo: str | Path | None = None,
                  figura: str | Path | None = None, seccion_documento: str = "",
                  hallazgo: str = "", interpretacion: str = "",
                  implicacion: str = "", limitacion: str = "") -> None:
        if pregunta not in set(self.catalogo["pregunta"]):
            raise KeyError(f"{pregunta} no pertenece al catálogo congelado P01–P52")
        self._filas[pregunta] = {
            "pregunta": pregunta,
            "archivo": _rel(archivo),
            "figura": _rel(figura),
            "seccion_documento": seccion_documento,
            "hallazgo": hallazgo,
            "interpretacion": interpretacion,
            "implicacion": implicacion,
            "limitacion": limitacion,
            "fecha_registro": io_utils.ahora_iso(),
            "ejecutada": bool(archivo or figura),
        }

    def pendientes(self) -> list[str]:
        hechas = {p for p, f in self._filas.items() if f["ejecutada"]}
        return [p for p in self.catalogo["pregunta"] if p not in hechas]

    def a_dataframe(self) -> pd.DataFrame:
        base = self.catalogo[["pregunta", "bloque", "enunciado"]].copy()
        reg = pd.DataFrame(self._filas.values()) if self._filas else pd.DataFrame(
            columns=["pregunta", "archivo", "figura", "seccion_documento", "hallazgo",
                     "interpretacion", "implicacion", "limitacion", "fecha_registro",
                     "ejecutada"])
        out = base.merge(reg, on="pregunta", how="left")
        out["ejecutada"] = out["ejecutada"].fillna(False).astype(bool)
        return out.fillna("")

    def exportar(self) -> tuple[Path, Path]:
        """Escribe matriz_trazabilidad.csv y hallazgos_eda.json en data/surface."""
        config.SURFACE.mkdir(parents=True, exist_ok=True)
        df = self.a_dataframe()
        df.to_csv(REGISTRO, index=False, encoding="utf-8")
        HALLAZGOS.write_text(
            json.dumps(self._filas, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8")
        return REGISTRO, HALLAZGOS

    def resumen(self) -> str:
        total = len(self.catalogo)
        hechas = total - len(self.pendientes())
        return f"{hechas}/{total} preguntas con evidencia · pendientes: {self.pendientes()}"


def _rel(ruta: str | Path | None) -> str:
    if ruta in (None, ""):
        return ""
    p = Path(ruta)
    try:
        return str(p.relative_to(config.RAIZ))
    except ValueError:
        return str(p)
