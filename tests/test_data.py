import io

import numpy as np
import pandas as pd

from betwise.data import standardize

# CSV con el formato de columnas de football-data.co.uk (construido a mano para la prueba)
CSV = """Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,B365H,B365D,B365A,PSCH,PSCD,PSCA,AvgH,AvgD,AvgA
E0,16/08/2024,Man United,Fulham,1,0,1.6,4.2,5.25,1.62,4.1,5.5,1.6,4.1,5.4
E0,17/08/2024,Arsenal,Wolves,2,0,1.2,6.5,13,,,,1.2,6.4,12.5
E0,18/08/2024,Chelsea,Man City,0,2,4.0,3.8,1.8,,,,,,
,,,,,,,,,,,,,,
"""


def test_standardize_formato_football_data():
    raw = pd.read_csv(io.StringIO(CSV))
    df = standardize(raw, "E0", "2425")
    assert len(df) == 3  # la fila vacía se descarta
    assert df.date.iloc[0] == pd.Timestamp("2024-08-16")  # día primero
    assert list(df.odds_source) == ["Pinnacle cierre", "Promedio", "Bet365"]  # preferencia fila a fila
    assert np.isfinite(df[["o_h", "o_d", "o_a"]].to_numpy()).all()
    assert df.o_h.iloc[0] == 1.62 and df.o_h.iloc[1] == 1.2 and df.o_h.iloc[2] == 4.0
    assert df.hg.dtype.kind == "i"
