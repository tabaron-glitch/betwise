"""Panel visual de BetWise (solo en tu computador).

    .venv/Scripts/streamlit run app/panel.py

Lee el libro desde la copia local del repositorio privado betwise-datos (no lo modifica).
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from betwise import daily
from betwise.ledger import LedgerError
from betwise.panel_data import OUTCOMES, forecast, ledger_view, read_ledger
from betwise.report import DISCLAIMER, summary
from betwise.sources import footballdata_org as fd

ROOT = Path(__file__).resolve().parents[1]
DATOS = Path(os.environ.get("BETWISE_DATOS", ROOT.parent / "betwise-datos"))
LEDGER = DATOS / "ledger" / "betwise.db"
LEAGUES = {"E0": "Premier League", "SP1": "LaLiga"}

st.set_page_config(page_title="BetWise", page_icon="⚽", layout="wide")
os.chdir(ROOT)  # cachés en data/raw y .env del proyecto


@st.cache_data(ttl=6 * 3600, show_spinner="Cargando resultados (openfootball + football-data.org)…")
def load_results(day: str):
    try:
        client = fd.Client(fd.load_key())
    except Exception:
        client = None
    results, _, current = daily.load_all_results(list(LEAGUES), pd.Timestamp.now(tz="UTC"), client,
                                                 verbose=False)
    return results, {d: sorted(t) for d, t in current.items()}, client is not None


def pct(x):
    return f"{x:.1%}"


st.title("⚽ BetWise")
st.warning(f"**Herramienta de análisis, no de apuestas.** El backtest no mostró ventaja sobre el mercado "
           f"(z = −1,95). {DISCLAIMER}")

with st.sidebar:
    st.header("Libro de pronósticos")
    st.caption(f"Copia local: `{DATOS}`")
    if st.button("⬇️ Traer lo último de GitHub", use_container_width=True):
        r = subprocess.run(["git", "-C", str(DATOS), "pull", "--ff-only"], capture_output=True, text=True)
        if r.returncode == 0:
            st.success("Sin cambios: ya tenías lo último." if "up to date" in r.stdout else "Libro actualizado.")
        else:
            st.error(f"No se pudo actualizar: {r.stderr.strip()}")
    st.divider()
    st.caption("La tarea automática corre cada día ~6:17 a. m. (hora Colombia) en GitHub y guarda "
               "los pronósticos ANTES de cada partido. Este panel solo los lee.")

tab_p, tab_l, tab_i = st.tabs(["🔮 Pronosticar un partido", "📒 Libro", "📊 Informe modelo vs mercado"])

# ---------------------------------------------------------------- Pronóstico de consulta
with tab_p:
    st.caption("Consulta: no se registra en el libro. Las cuotas, si las pones, son las que tú ves; "
               "nunca se inventan.")
    results, teams, has_api = load_results(str(pd.Timestamp.now(tz="UTC").date()))
    if not has_api:
        st.info("Sin FOOTBALLDATA_KEY en .env: solo openfootball (pueden faltar jornadas recientes).")
    c1, c2, c3 = st.columns([1, 2, 2])
    div = c1.selectbox("Liga", list(LEAGUES), format_func=LEAGUES.get)
    home = c2.selectbox("Local", teams[div], key=f"h_{div}")
    away = c3.selectbox("Visitante", teams[div], index=1, key=f"a_{div}")
    use_odds = st.checkbox("Comparar con cuotas que yo veo (decimales)")
    odds = None
    if use_odds:
        o1, o2, o3 = st.columns(3)
        typed = [c.number_input(n, 1.01, 100.0, None, 0.01, placeholder="ej. 2.10")
                 for c, n in zip((o1, o2, o3), ("Cuota local", "Cuota empate", "Cuota visita"))]
        if all(x is not None for x in typed):
            odds = typed
        else:
            st.caption("Escribe las tres cuotas tal como las ves en la casa (no hay valores por defecto).")
    if home == away:
        st.error("Escoge dos equipos distintos.")
    else:
        f = forecast(results, div, home, away, pd.Timestamp.now(tz="UTC"), odds)
        days_old = (pd.Timestamp.now().normalize() - f["last_result"]).days
        st.caption(f"Datos hasta {f['last_result'].date()} · entrenado con {f['n_train']} partidos · "
                   f"parámetros {daily.PARAMS}")
        if days_old > 10:
            st.warning(f"El último resultado cargado es de hace {days_old} días.")
        for t in f["unknown"]:
            st.warning(f"{t} no tiene partidos en la ventana (recién ascendido): pronóstico menos confiable.")

        st.subheader(f"{home}  vs  {away}")
        cols = st.columns(3)
        for i, (name, c) in enumerate(zip(OUTCOMES, cols)):
            c.metric(f"{name} (modelo)", pct(f["p"][i]), help=f"Cuota justa del modelo: {f['fair'][i]:.2f}")
            c.progress(float(f["p"][i]))
            c.caption(f"Cuota justa: **{f['fair'][i]:.2f}**")

        g1, g2, g3 = st.columns(3)
        g1.metric("Goles esperados", f"{f['xg'][0]:.2f} – {f['xg'][1]:.2f}")
        g2.metric("Más de 2.5 goles", pct(f["over"][2.5]),
                  help=f"Más de 1.5: {pct(f['over'][1.5])} · Más de 3.5: {pct(f['over'][3.5])}")
        g3.metric("Ambos anotan", pct(f["btts"]))

        left, right = st.columns(2)
        with left:
            st.markdown("**Marcadores más probables**")
            st.dataframe(pd.DataFrame([{"Marcador": f"{h}-{a}", "Probabilidad": pct(q)} for h, a, q in f["top"]]),
                         hide_index=True, use_container_width=True)
        with right:
            st.markdown(f"**Matriz de marcadores** (filas: goles de {home}; columnas: de {away})")
            mat = pd.DataFrame(f["matrix"] * 100).round(1)
            st.dataframe(mat.style.format("{:.1f}%"),
                         use_container_width=True)

        if odds is not None:
            st.markdown("---")
            st.markdown(f"**Contra las cuotas que pusiste** · margen de la casa {pct(f['margin'])}")
            st.dataframe(pd.DataFrame({
                "": OUTCOMES, "Cuota": f["odds"], "Mercado sin margen": [pct(x) for x in f["market"]],
                "Modelo": [pct(x) for x in f["p"]], "EV teórico": [f"{x:+.1%}" for x in f["ev"]]}),
                hide_index=True, use_container_width=True)
            st.error("NO APOSTAR: el modelo no ha demostrado ventaja. Si el modelo y el mercado discrepan, "
                     "lo más probable es que el mercado tenga razón. Un EV teórico positivo aquí NO es valor.")

# ---------------------------------------------------------------- Libro
tables, counts, settled, err = {}, {}, pd.DataFrame(), None
try:
    tables, counts, settled = read_ledger(LEDGER)
except (FileNotFoundError, LedgerError) as e:
    err = e

with tab_l:
    if err:
        st.error(f"{err}")
    else:
        st.success(f"Libro verificado (huellas OK): {counts}")
        v = ledger_view(tables)
        if v.empty:
            st.info("Aún no hay pronósticos registrados. La pausa FIFA termina el 21 de noviembre; los "
                    "primeros deberían aparecer hacia el 19–20 de noviembre. Pulsa «Traer lo último de "
                    "GitHub» en la barra lateral para actualizar.")
        else:
            fil = st.radio("Mostrar", ["Todos", "Pendientes", "Jugados"], horizontal=True)
            if fil != "Todos":
                v = v[v.estado == fil[:-1]]
            show = pd.DataFrame({
                "Inicio (UTC)": v.kickoff.str[:16].str.replace("T", " "), "Liga": v["div"].map(LEAGUES),
                "Partido": v.home + " vs " + v.away, "Estado": v.estado, "Marcador": v.marcador,
                "Modelo 1/X/2": [f"{a:.0%} / {b:.0%} / {c:.0%}" for a, b, c in v[["p_h", "p_d", "p_a"]].to_numpy()],
                "Mercado cierre 1/X/2": ["" if np.isnan(a) else f"{a:.0%} / {b:.0%} / {c:.0%}"
                                         for a, b, c in v[["m_h", "m_d", "m_a"]].to_numpy()],
                "Registrado": v.created_at.str[:16].str.replace("T", " ")})
            st.dataframe(show, hide_index=True, use_container_width=True)
            st.download_button("Descargar CSV (Excel)", v.to_csv(index=False).encode("utf-8-sig"),
                               "libro_betwise.csv", "text/csv")

# ---------------------------------------------------------------- Informe
with tab_i:
    if err:
        st.error(f"{err}")
    else:
        st.code(summary(settled), language=None, wrap_lines=True)
        st.caption("Log loss más bajo = mejor. «dif» positiva significaría que el modelo le gana al mercado; "
                   "con |z| < 2 no hay evidencia. Hacen falta cientos de partidos para concluir algo.")
