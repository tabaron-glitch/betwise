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

# Backtest con datos REALES
#   resultados: openfootball (dominio público, CC0)
#   cuotas de cierre de Pinnacle: OddsPapi (plan gratis; clave en .env: ODDSPAPI_KEY=...)
python scripts/update_odds.py --leagues E0 SP1 --start 2026-01-01   # la 1.ª vez tarda ~1 h
python scripts/run_backtest.py --leagues E0 SP1 --test-start 2026-01-17

# Pronóstico de un partido con las cuotas que tú ves (nunca se inventan cuotas)
#   nombres libres ("Man City", "atletico"); usa openfootball + football-data.org (FOOTBALLDATA_KEY)
python scripts/predict_match.py --league E0 --home "Man City" --away Arsenal --odds 1.95 3.60 4.00
```

## Panel visual (solo en tu computador)

Doble clic en `abrir_panel.bat` (o `.venv/Scripts/streamlit run app/panel.py`). Se abre en el navegador
en http://localhost:8501 y tiene tres pestañas: pronosticar un partido (consulta, no se registra), el
libro de pronósticos y el informe modelo vs mercado. Lee el libro desde la copia local del repositorio
privado `betwise-datos` (carpeta vecina; otra ruta con la variable `BETWISE_DATOS`) y nunca lo modifica.
No se publica en internet: el libro contiene cuotas de OddsPapi, que no se pueden redistribuir.
Requiere `pip install -e .[panel]`.

## Qué hay

| Archivo | Función |
|---|---|
| `src/betwise/odds.py` | Probabilidad implícita, margen, EV, cuota justa y mínima, Kelly fraccionado con tope |
| `src/betwise/data.py` | Lector de CSV de football-data.co.uk (bloqueado en Colombia; sin uso hasta tener concepto legal) |
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

## Fuentes de datos

| Fuente | Qué aporta | Notas |
|---|---|---|
| openfootball/football.json | Resultados desde 2010/11 | CC0. Se corrige un año mal escrito en Premier 2019/20 y se descartan partidos sin marcador. Se actualiza con semanas de retraso. |
| OddsPapi | Cuota de cierre de Pinnacle (1X2, más/menos 2,5, ambos anotan) | Histórico desde mediados de enero de 2026. Términos prohíben redistribuir: todo queda en `data/raw/`. |
| football-data.co.uk | Resultados y cuotas desde 1993 | Bloqueado en Colombia por Coljuegos. El lector sigue en `data.py`, sin uso, hasta tener concepto legal. |

Los nombres de equipos se traducen con `data/team_names.csv`; si aparece uno sin equivalencia el programa se detiene.

## Reglas del repositorio público

- Nunca subir claves (`.env` está ignorado) ni datos descargados (`data/raw/`: copyright de terceros y términos de OddsPapi).
- Los resultados reales del backtest se publican como `predicciones_backtest.csv` solo si decides hacerlo.

## Fase 1: registro hacia adelante

Resultado de la fase 0 (ver `docs/protocolo_ajuste.md`): el modelo NO supera a Pinnacle sin margen.
La fase 1 acumula una prueba limpia con partidos futuros.

```bash
python scripts/daily_run.py --ledger ledger/betwise.db   # pronostica (48 h) y liquida; 2 consultas OddsPapi
python scripts/report.py --ledger ledger/betwise.db --csv results/informe_fase1.csv
```

- `src/betwise/ledger.py`: libro SQLite solo de inserciones, con huella SHA-256 encadenada.
- `src/betwise/daily.py`: pronósticos antes del inicio; liquidación con cuota de apertura, al pronóstico y de cierre.
- `deploy/betwise-datos/`: plantilla de GitHub Actions para un repositorio PRIVADO de datos
  (el libro contiene cuotas de OddsPapi, que no se pueden redistribuir).
- Claves en `.env`: `ODDSPAPI_KEY`, `FOOTBALLDATA_KEY`.
