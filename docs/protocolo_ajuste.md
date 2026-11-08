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

## Resultado (8-nov-2026)

- Validación: 1.910 partidos, 48 combinaciones. Log loss entre 0,9663 y ~0,98.
- Mejor: xi=0.0019, reg=1, ventana=1460 → 0,9663, mejora frente a la actual de 0,0016
  con EE 0,0018. **Menor que 1 EE: por regla se conserva la configuración actual.**
- La superficie es plana (diferencias ≤ 0,002), mientras la brecha contra el mercado en la
  prueba es 0,022: ajustar estos parámetros no puede cerrarla.
- Como la configuración elegida es la misma de la primera corrida, la prueba final es esa
  corrida (log loss modelo 1,0269 vs mercado 1,0049, z = −1,95). No se hizo otra mirada.
- Conclusión: el criterio de la fase 0 NO se cumple. No hay base estadística para apostar.
