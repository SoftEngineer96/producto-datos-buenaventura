# Proyecto integrado de Buenaventura

**Producto de datos que integra información aduanera y portuaria de Buenaventura**

Universidad Libre, Seccional Cali · Ingeniería del Producto de Ciencia de Datos
Juan Manuel Tejada Fajardo · Jesús Alejandro Guerrero · 2026

**Dashboard público:**
<https://producto-datos-buenaventura-ffmg5xrgalkuhetnbntp2d.streamlit.app/>

La guía de uso, actualización y despliegue está en [`docs/DASHBOARD_BI.md`](docs/DASHBOARD_BI.md).

---

## Entregables

- [`Documento_Academico_Buenaventura.pdf`](reports/documents/Documento_Academico_Buenaventura.pdf)
- [`Documento_Academico_Buenaventura.docx`](reports/documents/Documento_Academico_Buenaventura.docx)
- [`Presentacion_Buenaventura.pdf`](reports/Presentacion_Buenaventura.pdf)
- [`Presentacion_Buenaventura.pptx`](reports/Presentacion_Buenaventura.pptx)

---

## Qué hace y qué no hace

Integra dos dominios con fuentes oficiales verificadas, los relaciona de forma
**agregada por mes**, y pronostica únicamente los indicadores que tienen historia y
calidad suficientes.

**No construye el dominio marítimo ni el operacional.** Diez de las 52 preguntas están
cerradas como no viables porque las fuentes públicas consultadas no ofrecen las series
históricas necesarias. La búsqueda y la justificación de cada cierre quedan registradas
en la matriz de trazabilidad.

---

## Estado de los cuatro dominios

| Dominio | Fuente | Cobertura | Estado |
|---|---|---|---|
| **Aduanero** | DANE IMPO, microdatos | 2012-01 a 2026-05 · **173 meses** | integrado |
| **Portuario** | Supertransporte `5r3g-zv5z` | 2018-01 a 2026-06 · **102 meses** | integrado |
| **Marítimo** | DIMAR | solo boletines trimestrales en PDF | **no viable** |
| **Operacional** | ETA, ATA, permanencias | sin fuente pública histórica | **no viable** |
| Contextual | TRM, ONI | 2012-01 a 2026-06 | contexto |

**Meses integrados: 101** (2018-01 a 2026-05), que es la intersección de ambos dominios.

---

## Las 52 preguntas

| Estado | Cantidad | Significado |
|---|---|---|
| ejecutada | **38** | con resultado y evidencia verificable |
| no viable | **10** | con evidencia documentada de por qué |
| parcial | **4** | respondida hasta el límite permitido por la fuente |

Ninguna quedó sin respuesta. `data/surface/matriz_trazabilidad_eda.csv` tiene el detalle
y `data/surface/reporte_no_viabilidad.csv` justifica cada cierre.

---

## Hallazgos

### El dataset portuario es mensual, no trimestral

Al comienzo se consideró que Supertransporte solo publicaba boletines trimestrales en
PDF. Esa conclusión no era correcta. El dataset `5r3g-zv5z` de datos.gov.co entrega datos
**mensual**, por zona portuaria, sociedad portuaria y tipo de carga, desde enero de 2018,
con licencia CC BY-SA 4.0 y descarga automatizable por API.

Eso convirtió un dominio que parecía inviable en uno con 102 observaciones mensuales.

### El puerto va por delante de la aduana

El tráfico portuario llega hasta **junio de 2026**; los microdatos aduaneros hasta
**mayo**. El puerto publica con menos rezago.

Consecuencia práctica: el dato portuario no sirve como predictor contemporáneo del CIF,
porque cuando se pronostica el CIF de un mes tampoco está publicado el puerto de ese mes.
Ambos comparten el mismo rezago de publicación.

### El modelo funciona para toneladas totales y no para carga contenerizada

Backtest walk-forward, 24 cortes:

| Indicador | Mejor modelo | WAPE | Naive 1 | ¿Gana el modelo? |
|---|---|---|---|---|
| Toneladas totales | Ridge | **6,61 %** | 9,17 % | **sí, 28 % mejor** |
| Carga contenerizada | **Naive 1** | **8,55 %** | 8,55 % | **no** |

Para la carga contenerizada, Ridge obtiene 9,45 %: **peor que repetir el último valor**.
El resultado se reporta tal cual. Un indicador que no se puede pronosticar mejor que la
referencia trivial no debe presentarse con un modelo encima.

### La integración no puede ser directa

No existe llave pública entre una declaración de importación y un movimiento de carga
portuaria. Los conceptos tampoco son equivalentes:

| | Aduanas (DANE) | Puerto (Supertransporte) |
|---|---|---|
| Unidad | declaración de importación | movimiento de carga |
| Cobertura | ADUA 35, solo importación | zona portuaria, todos los flujos |
| Peso | peso neto de la mercancía | toneladas movilizadas |
| Incluye | solo importación | importación, exportación **y transbordo** |

El transbordo es carga que cambia de buque **sin entrar ni salir del país**. Sumarlo al
comercio exterior infla el total. La serie portuaria separa ambas cosas.

---

## Estructura

```
proyecto_integrado_buenaventura/
├── data/
│   ├── raw/{aduanas,puertos,maritimo,contexto}/   fuentes inmutables con hash
│   └── landing/ · trusted/ · surface/             salidas analíticas y trazabilidad
├── src/
│   ├── comun/        métricas, modelos, calidad y trazabilidad
│   ├── config.py     rutas y fuentes
│   ├── puertos.py    procesamiento del dominio portuario
│   ├── integracion.py unión agregada y matriz de relaciones
│   └── correr_integrado.py  orquestador de las 52 preguntas
├── dashboard/app.py  siete vistas, solo lectura
├── notebooks/        EDA reproducible de las 52 preguntas
├── reports/          documento, presentación y figuras finales
└── docs/             catálogo de preguntas y guía del dashboard
```

---

## Reproducir

```bash
pip install -r requirements.txt
python -m src.correr_integrado
streamlit run dashboard/app.py
```

### Ejecutar en Google Colab

Abrir `notebooks/EDA_52_Preguntas_Buenaventura.ipynb` en Colab y ejecutar todas las
celdas. El cuaderno clona el repositorio cuando hace falta, instala sus dependencias y usa
los datos versionados del proyecto. No es necesario subir carpetas manualmente.

`notebooks/00_entorno.ipynb` ofrece una comprobación adicional: prepara las rutas, ejecuta
las pruebas y corre el pipeline antes de continuar con el EDA.

El dashboard funciona en una clonación limpia porque carga las fuentes reales
versionadas en `data/raw`; no depende de archivos locales ignorados por Git. Incluye
filtros de fecha, carga y sociedad portuaria, KPIs documentados, gráficos ejecutivos,
controles de calidad y descarga de los registros filtrados.

La arquitectura, comparación de plataformas y guía completa de publicación gratuita
están en [`docs/DASHBOARD_BI.md`](docs/DASHBOARD_BI.md).

La descarga portuaria es reproducible con la consulta exacta documentada en
`data/raw/puertos/README_fuente.md`.

---

## Límites declarados

1. ADUA 35 es una unidad de registro administrativo, no la operación física del puerto.
2. La integración aduanera-portuaria es **agregada por mes**, nunca directa.
3. Peso neto aduanero y toneladas portuarias **no son el mismo concepto**.
4. El dataset portuario publica **toneladas, no TEU**. La razón TEU por contenedor no
   puede calcularse con esta fuente.
5. El transbordo no es comercio exterior.
6. El CIF está en dólares corrientes: parte de la variación es precio, no volumen.
7. Las alertas son señales de revisión analítica, no órdenes operativas.
8. Correlación no implica causalidad.

## Trabajo futuro

| Tema | Motivo |
|---|---|
| Obtener una segunda descarga comparable del DANE | permitiría medir revisiones de publicación en P22 |
| Incorporar fuentes marítimas y operacionales autorizadas | permitiría ampliar las preguntas hoy cerradas como no viables |
| Traducir códigos de país y capítulo con catálogos oficiales | mejoraría la lectura sin introducir equivalencias manuales |
| Validar el tablero con usuarios del sector | permitiría contrastar la utilidad de las alertas y vistas |

## Pruebas del dashboard

```bash
python -m pytest tests/test_dashboard.py tests/test_dashboard_app.py -q
python -m ruff check dashboard tests/test_dashboard.py tests/test_dashboard_app.py
```

Las pruebas reconcilian los KPIs con las fuentes, verifican las llaves, los rangos de
fecha, la agregación mensual portuaria y las siete vistas mediante el motor de pruebas
de Streamlit.

