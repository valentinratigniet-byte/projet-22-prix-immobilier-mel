"""Le nettoyage DVF sur des lignes construites à la main : chaque piège a son cas."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import prepare  # noqa: E402

BASE = {"date_mutation": "2023-05-02", "nature_mutation": "Vente", "code_commune": "59350", "commune": "Lille",
        "nombre_pieces_principales": 3, "latitude": 50.63, "longitude": 3.06, "code_nature_culture": None,
        "surface_terrain": None, "id_parcelle": "P1"}


def ligne(mutation, type_local, prix, surface=None, **kw):
    return {**BASE, "id_mutation": mutation, "type_local": type_local, "valeur_fonciere": prix,
            "surface_reelle_bati": surface, **kw}


def test_vente_de_plusieurs_logements_ecartee():
    df = pd.DataFrame([ligne("M1", "Appartement", 600_000, 60), ligne("M1", "Appartement", 600_000, 55),
                       ligne("M2", "Appartement", 200_000, 50)])
    sales, _ = prepare.to_sales(df)
    assert list(sales.index) == ["M2"]  # M1 : prix total de 2 logements, pas le prix de l'un d'eux


def test_dependance_admise_et_signalee():
    df = pd.DataFrame([ligne("M1", "Appartement", 200_000, 50), ligne("M1", "Dépendance", 200_000)])
    sales, _ = prepare.to_sales(df)
    assert sales.loc["M1", "dependance"] == 1 and sales.loc["M1", "surface"] == 50


def test_local_commercial_ecarte():
    df = pd.DataFrame([ligne("M1", "Maison", 300_000, 100),
                       ligne("M1", "Local industriel. commercial ou assimilé", 300_000, 80)])
    sales, _ = prepare.to_sales(df)
    assert sales.empty


def test_terrain_dedoublonne():
    # DVF répète la ligne de terrain pour chaque local : 2 locaux x même parcelle de 400 m² = 400 m², pas 800.
    terrain = {"code_nature_culture": "S", "surface_terrain": 400}
    df = pd.DataFrame([ligne("M1", "Maison", 300_000, 100, **terrain), ligne("M1", "Dépendance", 300_000, **terrain)])
    sales, _ = prepare.to_sales(df)
    assert sales.loc["M1", "terrain"] == 400


def test_residence_de_studios_ecartee_copropriete_conservee():
    rows = [ligne(f"S{i}", "Appartement", 90_000, 19) for i in range(6)]  # 6 studios, même point
    # Grande copropriété : 12 reventes au même point, surfaces normales. Prix identiques pour que le
    # filtre des 1 % de prix extrêmes n'interfère pas avec la règle testée ici.
    rows += [ligne(f"C{i}", "Appartement", 180_000, 60, latitude=50.70) for i in range(12)]
    sales, funnel = prepare.to_sales(pd.DataFrame(rows))
    s = prepare.clean(sales, funnel)
    assert set(s["id_mutation"].str[0]) == {"C"}
    assert len(s) == 12


def test_distance_lille():
    assert prepare.distance_km(*prepare.LILLE_CENTRE) == 0
    assert 9 < prepare.distance_km(50.6909, 3.1817) < 11  # Grand'Place de Roubaix, ~10 km
