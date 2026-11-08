import io
import json
import math
import urllib.error

import pytest

from betwise.sources import oddspapi as op

START = "2026-04-11T11:30:00.000Z"


def rows(*items):
    return {"players": {"0": [{"createdAt": t, "price": p, "active": a} for t, p, a in items]}}


# Historial armado a mano: incluye cotización en vivo, suspendida y desordenada
HIST = {"fixtureId": "x", "bookmakers": {"pinnacle": {"markets": {
    "101": {"outcomes": {
        "101": rows(("2026-04-11T10:00:00Z", 1.50, True), ("2026-04-11T11:20:00Z", 1.49, True),
                    ("2026-04-11T11:25:00Z", 1.40, False),          # suspendida: se ignora
                    ("2026-04-11T12:00:00Z", 30.1, True)),          # en vivo: se ignora
        "102": rows(("2026-04-11T11:29:00Z", 4.70, True), ("2026-04-10T09:00:00Z", 4.50, True)),
        "103": rows(("2026-04-11T11:10:00Z", 6.90, True)),
    }},
    "1010": {"outcomes": {"1010": rows(("2026-04-11T11:00:00Z", 1.80, True))}},
}}}}


def test_cierre_toma_ultima_activa_antes_del_inicio():
    c = op.closing_prices(HIST, START)
    assert (c["o_h"], c["o_d"], c["o_a"]) == (1.49, 4.70, 6.90)
    assert c["o_over25"] == 1.80 and math.isnan(c["o_under25"]) and math.isnan(c["o_btts_y"])
    assert c["quote_time"].startswith("2026-04-11T11:10")  # la más vieja de las tres del 1X2
    assert c["min_before_ko"] == pytest.approx(20.0)


def test_cierre_sin_datos_del_bookmaker():
    c = op.closing_prices({"bookmakers": {}}, START)
    assert math.isnan(c["o_h"]) and c["quote_time"] == ""


def test_load_key_lee_env_y_rechaza_plantilla(tmp_path, monkeypatch):
    monkeypatch.delenv("ODDSPAPI_KEY", raising=False)
    env = tmp_path / ".env"
    env.write_text("OTRA=1\nODDSPAPI_KEY=abc123\n", encoding="utf-8")
    assert op.load_key(env) == "abc123"
    env.write_text("ODDSPAPI_KEY=PEGA_AQUI_TU_CLAVE\n", encoding="utf-8")
    with pytest.raises(op.OddsPapiError):
        op.load_key(env)


def test_la_clave_no_aparece_en_errores():
    def boom(req, timeout):
        raise urllib.error.HTTPError(req.full_url, 500, "err", {}, io.BytesIO(b""))

    c = op.Client("CLAVE-SECRETA", cooldown=0, opener=boom)
    with pytest.raises(op.OddsPapiError) as e:
        c.get("fixtures", tournamentId=17)
    assert "CLAVE-SECRETA" not in str(e.value) and "CLAVE-SECRETA" not in repr(c)


def test_404_devuelve_none_y_no_cuenta_como_consulta_facturable():
    def nf(req, timeout):
        raise urllib.error.HTTPError(req.full_url, 404, "nf", {}, io.BytesIO(b""))

    c = op.Client("k", cooldown=0, opener=nf)
    assert c.get("historical-odds", billable=False, fixtureId="x") is None
    assert c.billable_calls == 0


def test_update_closing_usa_cache_y_no_repite(tmp_path):
    fx = [{"fixtureId": "f1", "startTime": START, "participant1Name": "Arsenal FC",
           "participant2Name": "AFC Bournemouth"}]
    (tmp_path / "fixtures").mkdir()
    (tmp_path / "fixtures" / "E0_2026-04.json").write_text(json.dumps(fx), encoding="utf-8")
    calls = []

    class Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake(req, timeout):
        calls.append(req.full_url)
        return Resp(json.dumps(HIST).encode())

    c = op.Client("k", cooldown=0, opener=fake)
    df = op.update_closing(c, "E0", "2026-04-01", "2026-04-30", cache_dir=tmp_path, verbose=False)
    assert len(df) == 1 and df.status.iloc[0] == "ok" and df.o_h.iloc[0] == 1.49
    assert len(calls) == 1 and "historical-odds" in calls[0]  # el calendario vino de la caché
    op.update_closing(c, "E0", "2026-04-01", "2026-04-30", cache_dir=tmp_path, verbose=False)
    assert len(calls) == 1  # el partido ya estaba: no se vuelve a pedir
