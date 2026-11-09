"""Panel visual de BetWise (solo en tu computador).

    .venv/Scripts/streamlit run app/panel.py      (o doble clic en abrir_panel.bat)

Muestra los próximos partidos con las probabilidades del modelo, el análisis de cada uno,
la simulación con dinero ficticio, el libro y el informe. Lee el libro desde la copia local
del repositorio privado betwise-datos (no lo modifica).
"""
from __future__ import annotations

import html
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from betwise import daily
from betwise import simulation as sim
from betwise.ledger import LedgerError
from betwise.panel_data import (OUTCOMES, day_label, forecast, ledger_view, read_ledger, upcoming_board,
                                verdict)
from betwise.report import DISCLAIMER, summary
from betwise.sources import footballdata_org as fd
from betwise.teams import UnknownTeamError, to_canonical

ROOT = Path(__file__).resolve().parents[1]
DATOS = Path(os.environ.get("BETWISE_DATOS", ROOT.parent / "betwise-datos"))
LEDGER = DATOS / "ledger" / "betwise.db"
SIM_DB = Path(os.environ.get("BETWISE_SIM", ROOT / "simulacion" / "cuaderno.db"))  # local, nunca se sube
LEAGUES = {"E0": "Premier League", "SP1": "LaLiga"}
LEAGUE_TAG = {"E0": "ENG", "SP1": "ESP"}
DAYS_AHEAD = 14

st.set_page_config(page_title="BetWise", page_icon="⚽", layout="wide")
os.chdir(ROOT)  # cachés en data/raw y .env del proyecto

st.markdown("""
<style>
:root { --bg:#0b1220; --card:#131c2e; --card2:#1a2540; --line:#24314f; --txt:#e8edf7; --mut:#8a97b3;
        --acc:#19c3a0; --warn:#f5b301; --bad:#ef4444; }
.stApp { background: var(--bg); }
.block-container { padding-top: 1.2rem; max-width: 1200px; }
.bw-top { display:flex; align-items:center; justify-content:space-between; gap:12px; flex-wrap:wrap;
          background:linear-gradient(90deg,#0f1a30,#132442); border:1px solid var(--line);
          border-radius:14px; padding:14px 20px; margin-bottom:14px; }
.bw-logo { font-size:1.6rem; font-weight:800; color:var(--txt); letter-spacing:.5px; }
.bw-logo span { color:var(--acc); }
.bw-pill { background:rgba(239,68,68,.15); color:#fca5a5; border:1px solid rgba(239,68,68,.45);
           border-radius:999px; padding:6px 14px; font-weight:700; font-size:.85rem; }
.bw-day { color:var(--acc); font-weight:800; font-size:1.05rem; margin:18px 0 6px; text-transform:none;
          border-bottom:1px solid var(--line); padding-bottom:6px; }
.bw-meta { color:var(--mut); font-size:.78rem; }
.bw-tag { display:inline-block; background:var(--card2); color:var(--txt); border-radius:6px;
          padding:1px 7px; font-size:.7rem; font-weight:700; margin-right:6px; }
.bw-teams { color:var(--txt); font-weight:700; font-size:1rem; line-height:1.5; }
.bw-odds { display:flex; gap:8px; }
.bw-odd { flex:1; background:var(--card2); border:1px solid var(--line); border-radius:10px;
          text-align:center; padding:6px 4px; }
.bw-odd .k { color:var(--mut); font-size:.72rem; font-weight:700; }
.bw-odd .p { color:var(--txt); font-weight:800; font-size:1.05rem; }
.bw-odd .f { color:var(--acc); font-size:.75rem; }
.bw-odd.fav { border-color:var(--acc); }
.bw-verdict { background:rgba(239,68,68,.12); border:1px solid rgba(239,68,68,.5); border-radius:12px;
              padding:12px 16px; color:#fecaca; margin:8px 0 14px; }
.bw-verdict b { color:#fff; font-size:1.15rem; }
div[data-testid="stVerticalBlockBorderWrapper"] { background:var(--card); border-color:var(--line) !important;
              border-radius:14px; }
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ datos (con caché)
def _fd_client():
    try:
        return fd.Client(fd.load_key())
    except Exception:
        return None


@st.cache_data(ttl=6 * 3600, show_spinner="Cargando resultados…")
def load_results(day: str):
    client = _fd_client()
    results, _, current = daily.load_all_results(list(LEAGUES), pd.Timestamp.now(tz="UTC"), client,
                                                 verbose=False)
    return results, {d: sorted(t) for d, t in current.items()}, client is not None


@st.cache_resource(show_spinner="Ajustando el modelo…")
def load_models(day: str):
    results = load_results(day)[0]
    now = pd.Timestamp.now(tz="UTC")
    return {d: daily.fit_for_cutoff(results, d, now) for d in LEAGUES}


@st.cache_data(ttl=3600, show_spinner="Buscando los próximos partidos…")
def load_calendar(day: str):
    client = _fd_client()
    if client is None:
        return pd.DataFrame(), "Sin FOOTBALLDATA_KEY en .env: no puedo traer el calendario."
    teams = load_results(day)[1]
    start = pd.Timestamp.now(tz="UTC")
    frames, skipped = [], []
    try:
        for d in LEAGUES:
            c = fd.scheduled_matches(client, d, str(start.date()), str((start + pd.Timedelta(days=DAYS_AHEAD)).date()))
            c = c[c["kickoff"] > start]
            keep = []
            for i, r in c.iterrows():
                try:
                    c.loc[i, ["home", "away"]] = to_canonical([r["home"], r["away"]], "footballdata", d, set(teams[d]))
                    keep.append(i)
                except UnknownTeamError:
                    skipped.append(f"{r['home']} vs {r['away']}")
            frames.append(c.loc[keep])
    except fd.FootballDataError as e:
        return pd.DataFrame(), f"No se pudo traer el calendario: {e}"
    msg = f"Partidos omitidos por nombre desconocido: {skipped}" if skipped else ""
    return pd.concat(frames, ignore_index=True), msg


def pct(x):
    return f"{x:.1%}"


def money(x):
    return f"${x:,.0f}".replace(",", ".")


TODAY = str(pd.Timestamp.now(tz="UTC").date())
results, teams, has_api = load_results(TODAY)
models = load_models(TODAY)
v_title, v_why = verdict()


# ------------------------------------------------------------------ análisis de un partido
def render_analysis(div, home, away, kp: str, match_date=None):
    f = forecast(results, div, home, away, pd.Timestamp.now(tz="UTC"), fitted=models[div])
    st.markdown(f"<div class='bw-verdict'><b>Veredicto: {v_title}</b><br>{html.escape(v_why)}</div>",
                unsafe_allow_html=True)
    for t in f["unknown"]:
        st.warning(f"{t} no tiene partidos en la ventana (recién ascendido): pronóstico menos confiable.")
    fav = int(np.argmax(f["p"]))
    boxes = "".join(
        f"<div class='bw-odd{' fav' if i == fav else ''}'><div class='k'>{k} · {n}</div>"
        f"<div class='p'>{pct(f['p'][i])}</div><div class='f'>cuota justa {f['fair'][i]:.2f}</div></div>"
        for i, (k, n) in enumerate(zip(("1", "X", "2"), OUTCOMES)))
    st.markdown(f"<div class='bw-odds'>{boxes}</div>", unsafe_allow_html=True)
    st.caption("Cuota justa = la que pagaría una casa sin margen si el modelo acertara. Una apuesta solo "
               "tendría sentido con una cuota MAYOR y con un modelo validado (hoy no lo está).")
    g1, g2, g3 = st.columns(3)
    g1.metric("Goles esperados", f"{f['xg'][0]:.2f} – {f['xg'][1]:.2f}")
    g2.metric("Más de 2.5 goles", pct(f["over"][2.5]),
              help=f"Más de 1.5: {pct(f['over'][1.5])} · Más de 3.5: {pct(f['over'][3.5])}")
    g3.metric("Ambos anotan", pct(f["btts"]))
    with st.expander("Marcadores más probables y matriz"):
        a, b = st.columns(2)
        a.dataframe(pd.DataFrame([{"Marcador": f"{h}-{x}", "Probabilidad": pct(q)} for h, x, q in f["top"]]),
                    hide_index=True, use_container_width=True)
        b.caption(f"Filas: goles de {home} · columnas: de {away}")
        b.dataframe(pd.DataFrame(f["matrix"] * 100).style.format("{:.1f}%"), use_container_width=True)
    st.caption(f"Datos hasta {f['last_result'].date()} · {f['n_train']} partidos · parámetros {daily.PARAMS}")

    st.markdown("##### Comparar con las cuotas que ves en la casa")
    o1, o2, o3 = st.columns(3)
    typed = [c.number_input(n, 1.01, 100.0, None, 0.01, placeholder="ej. 2.10", key=f"{kp}_o{i}")
             for i, (c, n) in enumerate(zip((o1, o2, o3), ("Cuota 1 (local)", "Cuota X (empate)", "Cuota 2 (visita)")))]
    if not all(x is not None for x in typed):
        st.caption("Escribe las tres cuotas tal como las ves (no hay valores por defecto; nunca se inventan).")
        return
    f = forecast(results, div, home, away, pd.Timestamp.now(tz="UTC"), typed, fitted=models[div])
    st.dataframe(pd.DataFrame({
        "": OUTCOMES, "Cuota": f["odds"], "Mercado sin margen": [pct(x) for x in f["market"]],
        "Modelo": [pct(x) for x in f["p"]], "EV teórico": [f"{x:+.1%}" for x in f["ev"]]}),
        hide_index=True, use_container_width=True)
    st.caption(f"Margen de la casa: {f['margin']:.1%}. Si modelo y mercado discrepan, lo más probable es que "
               "el mercado tenga razón.")

    st.markdown("##### 🧪 Simulación con dinero ficticio")
    prop = sim.propose(f["p"], f["odds"])
    if prop is None:
        st.info("Ninguna opción pasa la regla de práctica (probabilidad − 3 puntos × cuota ≤ 1): "
                "ni en simulación se apuesta.")
        return
    with sim.Notebook(SIM_DB) as nb:
        stt = nb.state()
    amount = f" · monto ficticio {money(prop['fraccion'] * stt['banca'])}" if stt["iniciada"] else ""
    st.warning(f"**SIMULADO – NO ES RECOMENDACIÓN.** La regla de práctica marcaría **{prop['nombre']}** a "
               f"{prop['cuota']:.2f} (EV pesimista {prop['ev_pesimista']:+.1%}; {prop['fraccion']:.2%} de la banca)"
               f"{amount}.")
    if not stt["iniciada"]:
        st.caption("Para registrarla, primero fija tu banca ficticia en la pestaña «Simulación».")
        return
    today = pd.Timestamp.now(tz="America/Bogota").date()
    if match_date is None:
        match_date = st.date_input("Fecha del partido", value=today, min_value=today, format="DD/MM/YYYY",
                                   key=f"{kp}_date")
        st.caption("Pon la fecha real del partido: con ella se busca el resultado para liquidarla.")
    if st.button("Registrar en la simulación (dinero ficticio)", key=f"{kp}_reg"):
        try:
            with sim.Notebook(SIM_DB) as nb:
                r = nb.register(div, home, away, match_date, prop, results)
            st.success(f"Registrada: {prop['nombre']} a {prop['cuota']:.2f} por {money(r['monto'])} ficticios. "
                       "Se liquida sola cuando llegue el resultado.")
        except sim.SimulationError as e:
            st.error(str(e))


@st.dialog("Análisis del partido", width="large")
def analysis_dialog(row: dict):
    st.markdown(f"<span class='bw-tag'>{LEAGUE_TAG[row['div']]}</span>"
                f"<span class='bw-meta'>{LEAGUES[row['div']]} · jornada {row.get('matchday') or '—'} · "
                f"{row['local']:%d/%m %H:%M} (hora Colombia)</span>", unsafe_allow_html=True)
    st.subheader(f"{row['home']} vs {row['away']}")
    render_analysis(row["div"], row["home"], row["away"], kp=f"dlg_{row['div']}_{row['home']}_{row['away']}",
                    match_date=row["local"].date())


# ------------------------------------------------------------------ encabezado
st.markdown(f"<div class='bw-top'><div class='bw-logo'>Bet<span>Wise</span> ⚽</div>"
            f"<div class='bw-pill'>Veredicto general: {v_title} · modelo sin ventaja demostrada</div></div>",
            unsafe_allow_html=True)

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
    st.caption("La tarea automática corre cada día ~6:17 a. m. (hora Colombia) en GitHub y guarda los "
               "pronósticos ANTES de cada partido. Este panel solo los lee.")
    st.divider()
    st.caption(DISCLAIMER)

tab_c, tab_b, tab_s, tab_l, tab_i = st.tabs(["⚽ Próximos partidos", "🔎 Buscar partido", "🧪 Simulación",
                                             "📒 Libro", "📊 Informe"])

# ------------------------------------------------------------------ próximos partidos
with tab_c:
    cal, msg = load_calendar(TODAY)
    if not has_api:
        st.info("Sin FOOTBALLDATA_KEY en .env: solo openfootball (pueden faltar jornadas recientes).")
    if msg:
        st.caption(msg)
    league = st.segmented_control("Liga", ["Todas", *LEAGUES.values()], default="Todas",
                                  label_visibility="collapsed")
    board = upcoming_board(cal, {d: m[0] for d, m in models.items()}) if not cal.empty else cal
    if not board.empty and league and league != "Todas":
        board = board[board["div"].map(LEAGUES) == league]
    if board.empty:
        st.info(f"No hay partidos de Premier League ni LaLiga en los próximos {DAYS_AHEAD} días "
                "(por ejemplo, durante una pausa de selecciones). Vuelve más adelante.")
    else:
        today = pd.Timestamp.now(tz="America/Bogota").date()
        for label, g in board.groupby(board["local"].map(lambda t: t.date()), sort=True):
            st.markdown(f"<div class='bw-day'>{day_label(g.iloc[0]['local'], today)}</div>", unsafe_allow_html=True)
            for _, r in g.iterrows():
                with st.container(border=True):
                    a, b, c = st.columns([1.1, 2.4, 3.2], vertical_alignment="center")
                    hora = f"{r['local']:%H:%M}" if r["hora_confirmada"] else "Por confirmar"
                    a.markdown(f"<div class='bw-teams'>{hora}</div><span class='bw-tag'>{LEAGUE_TAG[r['div']]}</span>"
                               f"<span class='bw-meta'>J{r['matchday'] or '—'}</span>", unsafe_allow_html=True)
                    warn = "" if r["conocidos"] else " <span class='bw-meta'>· equipo sin historial</span>"
                    b.markdown(f"<div class='bw-teams'>{html.escape(r['home'])}<br>{html.escape(r['away'])}</div>{warn}",
                               unsafe_allow_html=True)
                    p = [r["p_h"], r["p_d"], r["p_a"]]
                    fav = int(np.argmax(p))
                    with c:
                        st.markdown("<div class='bw-odds'>" + "".join(
                            f"<div class='bw-odd{' fav' if i == fav else ''}'><div class='k'>{k}</div>"
                            f"<div class='p'>{pct(p[i])}</div><div class='f'>justa {1 / p[i]:.2f}</div></div>"
                            for i, k in enumerate(("1", "X", "2"))) + "</div>", unsafe_allow_html=True)
                        if st.button("Ver análisis", key=f"open_{r['div']}_{r['home']}_{r['away']}",
                                     use_container_width=True):
                            analysis_dialog(r.to_dict())
        st.caption("Porcentajes = probabilidad del modelo (1 local, X empate, 2 visita); «justa» = cuota sin margen. "
                   "No son cuotas de ninguna casa. Veredicto: NO APOSTAR.")

# ------------------------------------------------------------------ buscar partido (manual)
with tab_b:
    st.caption("Cualquier cruce de la temporada, aunque no esté en el calendario. Consulta: no se registra.")
    c1, c2, c3 = st.columns([1, 2, 2])
    div = c1.selectbox("Liga", list(LEAGUES), format_func=LEAGUES.get)
    home = c2.selectbox("Local", teams[div], key=f"h_{div}")
    away = c3.selectbox("Visitante", teams[div], index=1, key=f"a_{div}")
    if home == away:
        st.error("Escoge dos equipos distintos.")
    else:
        st.subheader(f"{home} vs {away}")
        render_analysis(div, home, away, kp="manual")

# ------------------------------------------------------------------ simulación
with tab_s:
    st.warning("**Dinero FICTICIO para practicar. NO es una recomendación de apuesta.** Reglas fijas: "
               "probabilidad del modelo − 3 puntos, 1/4 de Kelly con tope 2 % por apuesta, máximo 5 % por día y "
               "10 % por semana, pausa si la banca cae 15 %. Perder nunca sube el monto (sin martingala).")
    with sim.Notebook(SIM_DB) as nb:
        if nb.initial_bankroll() is None:
            ini = st.number_input("Banca inicial ficticia (pesos)", 100_000, 1_000_000_000, 1_000_000, 100_000)
            st.caption("Se fija una sola vez y no se puede cambiar, para que la práctica sea honesta.")
            if st.button("Empezar la simulación"):
                nb.start(ini)
                st.rerun()
        else:
            n_new = nb.settle(results)
            if n_new:
                st.success(f"Se liquidaron {n_new} apuestas simuladas con resultados nuevos.")
            stt = nb.state()
            c = st.columns(4)
            c[0].metric("Banca ficticia", money(stt["banca"]), f"{money(stt['resultado'])}")
            c[1].metric("Rendimiento (ROI)", "—" if np.isnan(stt["roi"]) else f"{stt['roi']:+.1%}")
            c[2].metric("Ganadas / liquidadas", f"{stt['ganadas']} / {stt['liquidadas']}")
            c[3].metric("Pendientes", stt["pendientes"])
            if stt["pausada"]:
                st.error(f"En pausa: la banca cayó {stt['caida']:.0%} desde su máximo.")
            st.caption(f"Hoy usado: {money(stt['hoy'])} de {money(sim.DAILY_LIMIT * stt['banca'])} · "
                       f"7 días: {money(stt['semana'])} de {money(sim.WEEKLY_LIMIT * stt['banca'])}")
            b = nb.bets()
            if b.empty:
                st.info("Aún no hay apuestas simuladas. En «Próximos partidos» abre «Ver análisis» y escribe las "
                        "cuotas que ves en la casa.")
            else:
                show = pd.DataFrame({
                    "Fecha": b.fecha_partido, "Partido": b.home + " vs " + b.away,
                    "Opción": b.opcion.map(dict(enumerate(OUTCOMES))), "Cuota": b.cuota,
                    "Monto": b.monto.map(money),
                    "Marcador": [("" if pd.isna(h) else f"{int(h)}-{int(a)}") for h, a in zip(b.hg, b.ag)],
                    "Estado": ["Pendiente" if pd.isna(g) else ("Ganada" if g else "Perdida") for g in b.ganada],
                    "Resultado": [("" if pd.isna(r) else money(r)) for r in b.resultado]})
                st.dataframe(show.iloc[::-1], hide_index=True, use_container_width=True)
                st.download_button("Descargar cuaderno (CSV para Excel)", b.to_csv(index=False).encode("utf-8-sig"),
                                   "simulacion_betwise.csv", "text/csv")
            st.caption("Con pocas apuestas el resultado es pura suerte: hacen falta cientos para sacar conclusiones.")

# ------------------------------------------------------------------ libro e informe
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
            st.info("Aún no hay pronósticos registrados. Los primeros deberían aparecer hacia el 19–20 de "
                    "noviembre. Pulsa «Traer lo último de GitHub» en la barra lateral para actualizar.")
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

with tab_i:
    if err:
        st.error(f"{err}")
    else:
        st.code(summary(settled), language=None, wrap_lines=True)
        st.caption("Log loss más bajo = mejor. «dif» positiva significaría que el modelo le gana al mercado; "
                   "con |z| < 2 no hay evidencia. Hacen falta cientos de partidos para concluir algo.")
