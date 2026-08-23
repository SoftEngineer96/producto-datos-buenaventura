# Ejecutar el EDA en Google Colab

El cuaderno necesita los datos y módulos del proyecto completo. Por eso no es necesario
subir carpetas o archivos manualmente.

## Pasos

1. Abrir `EDA_52_Preguntas_Buenaventura.ipynb` en Google Colab.
2. Seleccionar **Entorno de ejecución → Ejecutar todo**.
3. La primera celda instala las dependencias y, si hace falta, clona el repositorio.
4. Esperar a que termine la ejecución y revisar el resumen de las 52 preguntas.

También se puede ejecutar primero `00_entorno.ipynb`. Ese cuaderno prepara el proyecto,
corre las pruebas y genera las salidas del pipeline antes de abrir el EDA.

## Datos portuarios

El EDA comprueba si la API pública de datos.gov.co está disponible, pero realiza los
cálculos con los archivos versionados en `data/raw/puertos/`. Así, una actualización de
la API no cambia los resultados de la entrega. Los análisis por sociedad portuaria usan
`terminales_por_anio.csv` y `terminales_por_tipo_carga.csv`.

## Salidas

Los resultados quedan en `notebooks/salidas_eda/`:

- figuras utilizadas por las preguntas;
- archivos CSV con resultados numéricos;
- `matriz_trazabilidad_eda.csv` con el estado de las 52 preguntas.

## Estado esperado

| Estado | Cantidad |
|---|---:|
| ejecutada | 38 |
| ejecutada parcialmente | 4 |
| no viable | 10 |

Ninguna pregunta queda sin respuesta o sin justificación. Una pregunta parcial conserva
el resultado que sí puede calcularse y explica qué parte no permite responder la fuente.
