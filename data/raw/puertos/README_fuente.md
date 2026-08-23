# Fuente portuaria — Supertransporte

- **Entidad:** Superintendencia de Transporte
- **Dataset:** Tráfico Portuario Marítimo En Colombia (`5r3g-zv5z`)
- **Portal:** https://www.datos.gov.co/Transporte/Trafico-Portuario-Mar-timo-En-Colombia/5r3g-zv5z
- **API:** https://www.datos.gov.co/resource/5r3g-zv5z.csv
- **Licencia:** Creative Commons Attribution · Share Alike 4.0 International (CC BY-SA 4.0)
- **Cobertura:** desde el 1 de enero de 2018
- **Frecuencia declarada:** trimestral · **granularidad del dato: mensual**
- **Filas totales:** 9.571 nacionales · 1.111 de la zona portuaria de Buenaventura
- **Actualización del dataset:** 2026-08-01
- **Unidad:** toneladas
- **Advertencia de la fuente:** las cifras pueden actualizarse cuando las sociedades
  portuarias reportan errores de transmisión. Solo la Supertransporte puede certificarlas.

## Consulta exacta usada (reproducible)

```
GET https://www.datos.gov.co/resource/5r3g-zv5z.csv
  ?$select=anno_vigencia,mes_vigencia,tipo_carga,
           sum(importacion),sum(exportacion),sum(transbordo)
  &$where=zona_portuaria='BUENAVENTURA'
  &$group=anno_vigencia,mes_vigencia,tipo_carga
  &$limit=2000
```

Descargado el 2026-08-06. La capa `landing` guarda el resultado tal cual se recibió.
