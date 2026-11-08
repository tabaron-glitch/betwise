# BetWise AI — fase 0

Motor estadístico para fútbol (Premier League y LaLiga) con una pregunta central:
**¿el modelo supera a las probabilidades del mercado sin margen, fuera de muestra?**
Si no lo supera, no hay base para apostar; el proyecto sigue siendo útil como herramienta de análisis.

> No garantiza ganancias. Las probabilidades son estimaciones. Juega con responsabilidad.

## Inicio rápido

```bash
pip install -e ".[dev]"
pytest                                   # 13 pruebas (incluye prueba de no-fuga de información)

# Prueba de tubería con datos SINTÉTICOS (no son resultados reales)
python scripts/run_backtest.py --synthetic

# Backtest con datos REALES (descarga CSV de football-data.co.uk, requiere internet)
python scripts/run_backtest.py --leagues E0 SP1 --test-start 2024-08-01

# Pronóstico de un partido con las cuotas que tú ves (nunca se inventan cuotas)
python scripts/predict_match.py --league E0 --home Arsenal --away Chelsea --odds 1.95 3.60 4.00
```

## Qué hay

| Archivo | Función |
|---|---|
| `src/betwise/odds.py` | Probabilidad implícita, margen, EV, cuota justa y mínima, Kelly fraccionado con tope |
| `src/betwise/data.py` | Descarga con caché y estandarización de CSV; elige cuotas fila a fila (Pinnacle cierre > promedio cierre > Bet365...) |
| `src/betwise/dixon_coles.py` | Modelo Dixon-Coles con peso temporal, matriz de marcadores y mercados derivados (1X2, totales, ambos anotan, marcadores) |
| `src/betwise/backtest.py` | Walk-forward sin fuga, Brier, log loss, calibración, prueba pareada contra el mercado |
| `src/betwise/synthetic.py` | Datos simulados solo para probar el código |

## Cómo leer el backtest

1. **Log loss y Brier del modelo vs mercado vs base.** El mercado sin margen es el rival real.
2. **z de la diferencia de log loss.** Con |z| < 2 no hay evidencia de que el modelo sea mejor o peor.
3. **ROI simulado:** solo sensibilidad. Usa cuotas del archivo (normalmente de cierre), y elegir el umbral con los mismos datos sobreajusta.
4. Detectar un ROI de +3% con cuotas ~2.0 requiere del orden de 4.000 apuestas.

## Limitaciones conocidas

- Falta el cierre de mercado vs apertura (CLV): necesita cuotas capturadas antes y después; llega con el ledger (fase 1).
- Sin xG (FBref perdió los datos avanzados de Opta en enero de 2026).
- Sin lesiones ni alineaciones: son contexto manual por ahora.
- Los CSV tienen cuotas de varias casas y temporadas: verifica qué columnas trae cada liga/temporada (`odds_source`).
- Temporadas 2020/21 (sin público) tienen ventaja local atípica; el peso temporal lo atenúa, pero conviene revisarlo.

## Reglas del repositorio público

- Nunca subir claves (`.env` está ignorado) ni los CSV descargados (`data/raw/`, el sitio reclama copyright).
- Los resultados reales del backtest se publican como `predicciones_backtest.csv` solo si decides hacerlo.

## Siguiente (fase 1)

Ledger inmutable de pronósticos, captura diaria de calendario/cuotas con GitHub Actions y panel Streamlit.
