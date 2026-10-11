"""Garde-fous sur les résultats du pipeline (rejoué en CI avant ces tests) : si un changement de
nettoyage ou de modèle les fait tomber, la CI casse au lieu de publier une conclusion fausse."""
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "outputs"


def test_le_modele_bat_la_methode_naive():
    v = pd.read_csv(OUT / "validation.csv")
    assert (v["erreur_mediane_modele_pct"] < v["erreur_mediane_naive_pct"]).all()


def test_baisse_2024_vs_2022_significative():
    t = pd.read_csv(OUT / "test_hausse_taux.csv")
    assert (t["ecart_2024_vs_2022_pct"] < 0).all() and (t["p_value"] < 0.001).all()


def test_aucune_adresse_dans_les_sorties_publiees():
    # Conditions de réutilisation DVF : seules des données agrégées par quartier sont publiées.
    q = pd.read_csv(OUT / "quartiers.csv")
    assert (q["ventes"] >= 30).all()
    assert not {"adresse_nom_voie", "id_parcelle", "id_mutation"} & set(q.columns)
