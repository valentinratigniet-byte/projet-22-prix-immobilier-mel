"""L'estimateur de l'application (stdlib + JSON) doit redonner exactement les prédictions statsmodels."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import estimateur as E  # noqa: E402
import model as M  # noqa: E402

pytestmark = pytest.mark.skipif(not M.DATA.exists(), reason="pipeline non exécuté (src/prepare.py)")


@pytest.mark.parametrize("type_bien", ["Appartement", "Maison"])
def test_estimateur_egal_statsmodels(type_bien):
    df = M.load()
    est = json.loads((M.OUT / "estimateur.json").read_text(encoding="utf-8"))
    m = M.fit(df, type_bien)
    ech = df[df["type"] == type_bien].sample(300, random_state=1)
    attendu = M.predict_prix(m, ech).to_numpy()
    obtenu = [E.estimer(est, type_bien, r.surface, r.pieces_c, r.dependance, r.terrain, r.quartier, r.trimestre)["prix"]
              for r in ech.itertuples()]
    np.testing.assert_allclose(obtenu, attendu, rtol=1e-9)


def test_cellule_identique_au_modele():
    df = M.load()
    est = json.loads((M.OUT / "estimateur.json").read_text(encoding="utf-8"))
    ech = df[~df["quartier"].str.endswith("(reste)")].sample(500, random_state=2)
    for r in ech.itertuples():
        assert E.quartier(est, r.type, r.lat, r.lon, r.commune) == (r.quartier, True)
