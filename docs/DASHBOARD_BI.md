# Dashboard BI de importaciones de Buenaventura

## Despliegue

El dashboard está listo para publicarse desde la rama `main`, usando el archivo
`dashboard/app.py` y Python 3.12 en Streamlit Community Cloud. Cuando se cree el
despliegue, la URL pública se puede agregar en esta sección.

## Decisión

Se eligió **Streamlit Community Cloud**. Es la alternativa que mejor conserva la
arquitectura Python existente, lee Parquet/CSV directamente, permite filtros y descargas
sin construir un backend separado y se actualiza desde GitHub. La documentación oficial
indica que Community Cloud permite crear, desplegar y administrar aplicaciones gratis:
<https://docs.streamlit.io/deploy/streamlit-community-cloud>.

## Comparación de alternativas

| Alternativa | Publicación gratuita | Encaje con este repositorio | Actualización | Decisión |
|---|---:|---|---|---|
| Streamlit Community Cloud | Sí, para la aplicación pública | Nativo: Python, pandas, Parquet y CSV | Un `push` a GitHub redespliega la aplicación | **Elegida** |
| Looker Studio | Sí, versión sin costo | Requiere subir/replicar CSV; no ejecuta las transformaciones Python del repositorio | La carga CSV se anexa y exige controlar duplicados | Viable para reportes simples, no es la fuente de verdad aquí |
| Power BI Desktop | Desktop gratuito | Buen modelado local, pero desconectado del pipeline Python versionado | Compartir en workspaces y con otros usuarios requiere licencia de pago | Descartada por la restricción de costo de compartir |
| GitHub Pages | Sí en repositorios públicos | Solo sirve contenido HTML/CSS/JS estático; habría que reimplementar filtros y transformación | Requiere un proceso de construcción y publicar extractos | Descartada por duplicar la lógica de datos |

Fuentes oficiales consultadas:

- Streamlit Community Cloud y redespliegue desde GitHub:
  <https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app>
- Looker Studio sin costo y carga CSV:
  <https://cloud.google.com/looker/docs/studio-comparison> y
  <https://cloud.google.com/looker/docs/studio/upload-csv-files-to-looker-studio>
- Licencias para compartir en Power BI:
  <https://learn.microsoft.com/en-us/power-bi/fundamentals/end-user-license>
- GitHub Pages como hosting estático:
  <https://docs.github.com/en/enterprise-cloud@latest/pages/getting-started-with-github-pages/what-is-github-pages>

## Arquitectura implementada

```text
data/raw + data/surface + notebooks/salidas_eda
                       │
                       ▼
dashboard/data.py          carga, esquema y tipos
                       │
                       ▼
dashboard/transforms.py    filtros, agregaciones y calidad
                       │
                       ▼
dashboard/metrics.py       definiciones y cálculo de KPIs
                       │
                       ▼
dashboard/app.py           interfaz Streamlit y gráficos Altair
```

La presentación no recalcula reglas escondidas: consume funciones probadas por separado.
Los datos son agregados y públicos; no se usan secretos ni credenciales.

## Cobertura visual

El dashboard organiza los resultados del Colab en siete vistas orientadas a decisión. No
duplica las 55 salidas gráficas del notebook: conserva las visualizaciones con valor
ejecutivo y deja la evidencia exploratoria completa en el cuaderno ejecutado.

| Vista | Contenido principal |
|---|---|
| Resumen ejecutivo | KPIs, tendencia CIF, carga de importación y mezcla de carga |
| Dinámica aduanera | Índices comparables, variación mensual y estacionalidad |
| Operación portuaria | Carga contenerizada, HHI, actividad por sociedad y extremos |
| Composición y territorio | Cargas, sociedades, especialización, países y capítulos |
| Integración y modelos | Correlaciones, backtests, ablación e intervalos predictivos |
| Detalle descargable | Seis tablas filtrables y exportables |
| Calidad y metodología | Calidad, cobertura, definiciones, fuentes y rezagos |

## Indicadores

| Indicador | Fórmula | Grano / alcance |
|---|---|---|
| Valor CIF total | `SUM(cif_usd)` | Meses aduaneros del filtro |
| Peso neto | `SUM(peso_neto_kg)` | Meses aduaneros del filtro |
| Declaraciones | `SUM(n_registros)` | Registros administrativos de ADUA 35 |
| Meses observados | `COUNT(filas mensuales)` | Meses aduaneros del filtro |
| Variación mensual CIF | `(CIF último / CIF anterior - 1) × 100` | Dos últimos meses disponibles del filtro |
| Toneladas portuarias | Suma de importación, exportación y transbordo | Mes × tipo de carga |

El peso neto aduanero no se suma con las toneladas portuarias porque representan conceptos
distintos. Los dólares son corrientes y no aíslan inflación ni tasa de cambio.

## Ejecutar localmente

Requiere Python 3.11 o 3.12.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
streamlit run dashboard/app.py
```

Abrir <http://localhost:8501>. No se requiere archivo `.env`.

## Tutorial de uso, clic a clic

1. Abrir la URL pública indicada en **Acceso público** y esperar a que aparezca el título
   **Importaciones y actividad portuaria de Buenaventura**. Si la aplicación estaba
   inactiva, Streamlit puede tardar unos segundos en reactivarla.
2. En computador, usar el panel izquierdo. En celular, pulsar la flecha `>` de la esquina
   superior izquierda para abrirlo y la flecha `<` para cerrarlo.
3. En **Periodo**, hacer clic en el rango y elegir la fecha inicial y la final. Los KPI,
   gráficos y tablas se recalculan con ese intervalo.
4. En **Tipo de carga**, mantener o quitar categorías. En **Sociedad portuaria**, hacer lo
   mismo con las terminales. **Clear all** deja el filtro sin selecciones; se puede volver
   a elegir una o varias opciones desde la lista.
5. Abrir **Resumen ejecutivo** para leer valor CIF, peso neto, declaraciones, meses y
   variación mensual, además de las tres tendencias principales.
6. Abrir **Dinámica aduanera** para comparar índices base 100, variaciones y
   estacionalidad. Abrir **Operación portuaria** para carga contenerizada, concentración
   HHI, sociedades y periodos extremos.
7. Abrir **Composición y territorio** para comparar cargas, sociedades, especialización,
   países y capítulos arancelarios. Los países aparecen con el código DANE disponible en
   la fuente.
8. Abrir **Integración y modelos** para revisar correlaciones, resultados de backtest,
   ablación e intervalos predictivos; estas señales no sustituyen una decisión operativa.
9. Abrir **Detalle descargable**, localizar la tabla deseada y pulsar **Download as CSV**
   en su esquina superior derecha. El navegador guardará la tabla correspondiente.
10. Abrir **Calidad y metodología** antes de citar resultados. Confirmar `3 / 3` fuentes
    aprobadas, cero nulos, cero duplicados y revisar el diccionario de KPI, alcance y corte.
11. Para ampliar un gráfico, pulsar **View fullscreen**. Para volver, cerrar la vista
    ampliada o presionar `Esc`.

## Actualizar los datos

1. Actualizar las fuentes inmutables o ejecutar `python -m src.correr_integrado`.
2. Conservar los nombres y esquemas documentados en `dashboard/data.py`.
3. Ejecutar las pruebas y abrir la vista **Calidad y metodología**.
4. Confirmar corte, nulos, duplicados de llave, fechas inválidas y valores negativos.
5. Hacer commit y push. Community Cloud detectará el cambio en GitHub.

No subir tokens, archivos `.env`, `secrets.toml`, microdatos personales ni fuentes que no
tengan autorización de publicación. Si en el futuro se usa una API privada, los secretos
se configuran en la administración de Streamlit y nunca en el repositorio.

## Publicar gratis paso a paso

La aplicación ya está publicada. Estos pasos sirven para reconstruirla o crear otra
instancia:

1. Integrar la rama aprobada en `main` y verificar que el repositorio siga siendo público.
2. Entrar a <https://share.streamlit.io> e iniciar sesión.
3. Conectar la cuenta de GitHub `TEFAELBRO`.
4. Seleccionar `TEFAELBRO/producto-datos-buenaventura`.
5. Elegir la rama `main` y el archivo `dashboard/app.py`.
6. Seleccionar Python 3.12 en opciones avanzadas. No agregar secretos: el dashboard no los usa.
7. Pulsar **Deploy** y esperar a que finalice la instalación de `requirements.txt`.
8. Probar filtros, vistas y la descarga CSV desde la URL `*.streamlit.app`.

El primer intento con la versión automática Python 3.14 no pudo instalar
`pyarrow==17.0.0`. Se corrigió seleccionando Python 3.12 en la configuración de la
aplicación; la reconstrucción instaló las 59 dependencias y arrancó correctamente.

La publicación crea una aplicación pública. Los límites de recursos de Community Cloud
pueden cambiar; para este repositorio los extractos son pequeños y se almacenan en caché.

## Limitaciones

- La serie aduanera está agregada por mes. No permite filtrar por empresa ni producto a
  nivel de declaración.
- País y capítulo solo están disponibles como cortes históricos top-5. Los países siguen
  codificados y no se tradujeron para evitar asignaciones no verificadas.
- La fuente portuaria corresponde a una sola zona. Un mapa con un punto no aportaría una
  comparación geográfica real.
- El corte por sociedad portuaria es anual, por lo que el filtro temporal se aplica por año.
- El último año/mes puede estar incompleto y debe interpretarse con la fecha de corte visible.
- La actualización es por cambio de archivos en GitHub; no es una conexión transaccional en vivo.

