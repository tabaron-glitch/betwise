# Protocolo de ajuste de parámetros (fijado ANTES de correrlo)

Fecha: 8-nov-2026. Este archivo se guarda en git antes de ejecutar la búsqueda, para que
no se pueda cambiar el criterio después de ver los resultados.

## Antecedente

Primera corrida con datos reales (parámetros fijados a priori: xi=0.0023, reg=0.05,
ventana=1460 días), prueba 17-ene-2026 a 20-sep-2026, 479 partidos:
log loss modelo 1,0269 vs mercado sin margen 1,0049 (z = −1,95). El modelo pierde.

Ese periodo de prueba ya se miró una vez. Esta será la **segunda y última mirada** con
este diseño; cualquier ajuste posterior requiere un periodo de prueba nuevo (partidos futuros).

## Validación (para elegir parámetros)

- Periodo: 2023-08-01 a 2026-01-16. No se solapa con la prueba.
- Datos: solo resultados de openfootball (no se usan cuotas).
- Mismo walk-forward que el backtest: lotes de 14 días, cada lote se predice con datos
  anteriores a su fecha de corte; mínimo 150 partidos de entrenamiento.
- Métrica: log loss medio del 1X2, Premier y LaLiga juntas.

## Rejilla (decidida a priori)

- xi (decaimiento diario): 0.001, 0.0019, 0.0023, 0.003, 0.0045, 0.006
- reg (regularización de ataque/defensa): 0.05, 1, 3, 10
- ventana de entrenamiento (días): 730, 1460

48 combinaciones.

## Regla de elección

Se elige la combinación de menor log loss en validación. Si la mejora frente a la
configuración actual (0.0023, 0.05, 1460) es menor que 1 error estándar de la diferencia
pareada, se conserva la actual (no hay evidencia de mejora).

## Prueba final

Con la combinación elegida, una sola corrida de `run_backtest.py` en 17-ene-2026 a
20-sep-2026 frente a Pinnacle sin margen. Se informa el resultado sea cual sea.
Criterio sin cambios: hay evidencia a favor del modelo solo si z > 2.
