import io
import urllib.error

import pytest

from betwise.sources import footballdata_org as fd

# Forma tomada del ejemplo de la documentación v4 (armado a mano)
DATA = {"matches": [
    {"utcDate": "2026-11-07T23:30:00Z", "status": "FINISHED", "homeTeam": {"name": "Arsenal FC"},
     "awayTeam": {"name": "Chelsea FC"}, "score": {"fullTime": {"home": 2, "away": 1}}},
    {"utcDate": "2026-11-08T15:00:00Z", "status": "IN_PLAY", "homeTeam": {"name": "A"},
     "awayTeam": {"name": "B"}, "score": {"fullTime": {"home": 0, "away": 0}}},
    {"utcDate": "2026-11-08T17:30:00Z", "status": "FINISHED", "homeTeam": {"name": "C"},
     "awayTeam": {"name": "D"}, "score": {"fullTime": {"home": None, "away": None}}},
]}


def test_parse_solo_terminados_con_marcador():
    df = fd.parse_matches(DATA, "E0")
    assert len(df) == 1
    r = df.iloc[0]
    assert (r.home, r.away, r.hg, r.ag) == ("Arsenal FC", "Chelsea FC", 2, 1)
    assert str(r.date.date()) == "2026-11-07"  # día UTC


def test_clave_va_en_cabecera_y_no_en_errores():
    seen = {}

    def boom(req, timeout):
        seen["h"] = req.get_header("X-auth-token")
        raise urllib.error.HTTPError(req.full_url, 403, "x", {}, io.BytesIO(b""))

    c = fd.Client("SECRETO", cooldown=0, opener=boom)
    with pytest.raises(fd.FootballDataError) as e:
        c.get("competitions/PL/matches", status="FINISHED")
    assert seen["h"] == "SECRETO"
    assert "SECRETO" not in str(e.value) and "SECRETO" not in repr(c)


def test_load_key(tmp_path, monkeypatch):
    monkeypatch.delenv("FOOTBALLDATA_KEY", raising=False)
    env = tmp_path / ".env"
    env.write_text("ODDSPAPI_KEY=x\nFOOTBALLDATA_KEY=abc\n", encoding="utf-8")
    assert fd.load_key(env) == "abc"
    env.write_text("FOOTBALLDATA_KEY=PEGA_AQUI_TU_CLAVE\n", encoding="utf-8")
    with pytest.raises(fd.FootballDataError):
        fd.load_key(env)
