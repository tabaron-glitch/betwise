import json

from betwise.sources import openfootball as of

# Fragmento armado a mano con los tres formatos de marcador vistos en openfootball
DATA = {
    "name": "Prueba",
    "matches": [
        {"round": "Matchday 1", "date": "2025-08-15", "time": "20:00", "team1": "Liverpool FC",
         "team2": "AFC Bournemouth", "score": {"ft": [4, 2], "ht": [1, 0]}},
        {"round": "Matchday 1", "date": "2025-08-16", "time": "12:30", "team1": "Aston Villa FC",
         "team2": "Newcastle United FC", "score": [0, 0]},
        {"round": "Matchday 38", "date": "2026-05-24", "team1": "Arsenal FC", "team2": "Everton FC"},
        {"round": "Matchday 2", "date": "2025-08-22", "team1": "A", "team2": "B", "score": {"ht": [1, 0]}},
    ],
}


def test_parse_formatos_de_marcador():
    df, skipped = of.parse(DATA, "E0", "2526")
    assert skipped == 2  # sin score y con solo medio tiempo
    assert list(df.home) == ["Liverpool FC", "Aston Villa FC"]
    assert list(df.hg) == [4, 0] and list(df.ag) == [2, 0]
    assert df.hg.dtype.kind == "i"
    assert (df["div"] == "E0").all() and (df["season"] == "2526").all()


def test_corrige_anio_mal_escrito_solo_cuando_corresponde():
    ts = of.pd.Timestamp
    # caso real: Bournemouth 1-4 Newcastle se jugó el 1-jul-2020, el archivo dice 2019
    assert of.fix_season_year(ts("2019-07-01"), "1920") == ts("2020-07-01")
    assert of.fix_season_year(ts("2019-07-26"), "1920") == ts("2020-07-26")
    assert of.fix_season_year(ts("2019-08-09"), "1920") == ts("2019-08-09")
    assert of.fix_season_year(ts("2020-07-19"), "1920") == ts("2020-07-19")
    data = {"matches": [{"date": "2019-07-01", "team1": "A", "team2": "B", "score": {"ft": [1, 4]}}]}
    df, _ = of.parse(data, "E0", "1920")
    assert df.date.iloc[0] == ts("2020-07-01") and df.attrs["fechas_corregidas"] == 1


def test_season_folder():
    assert of.season_folder("2425") == "2024-25"
    assert of.season_folder("1920") == "2019-20"


def test_load_results_usa_cache_sin_internet(tmp_path, monkeypatch):
    (tmp_path / "E0_2526.json").write_text(json.dumps(DATA), encoding="utf-8")

    def no_network(*a, **k):
        raise AssertionError("no debía conectarse a internet")

    monkeypatch.setattr(of.urllib.request, "urlopen", no_network)
    df = of.load_results(["E0"], ["2526"], cache_dir=tmp_path, verbose=False)
    assert len(df) == 2
