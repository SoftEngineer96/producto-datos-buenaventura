# Recurso aduanero para el cuaderno

Esta carpeta forma parte del repositorio y no debe subirse manualmente a Colab. Contiene
la versión CSV que utiliza el EDA:

| Archivo | Por qué está aquí |
|---|---|
| `serie_aduanera_mensual.csv` | Exportación en CSV de `data/trusted/serie_aduanera_mensual.parquet`, para que el cuaderno no dependa de pyarrow en Colab |

El tráfico portuario, la información por sociedad y las variables externas se leen desde
`data/raw/`, que es su ubicación canónica.

La función `ruta_dato()` permite ejecutar el cuaderno desde la raíz del repositorio o
desde `notebooks/`. En Colab, el propio cuaderno clona primero el proyecto completo.
