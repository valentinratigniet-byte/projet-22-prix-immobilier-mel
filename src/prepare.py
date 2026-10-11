"""DVF brut (une ligne par lot/parcelle) -> une ligne par vente d'un logement unique, MEL, 2021-2025.

Piège central de DVF : `valeur_fonciere` est le prix de la mutation entière, répété sur chaque
ligne. Une vente de 3 appartements apparaît 3 fois avec le prix total. On ne garde donc que les
mutations qui portent sur exactement un logement (maison ou appartement), dépendances admises.
Chaque filtre est compté dans `docs/entonnoir.csv` : rien n'est écarté sans trace.
"""
import json
import math
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW, OUT = ROOT / "data" / "raw", ROOT / "data" / "processed"
LILLE_CENTRE = (50.6366, 3.0635)  # Grand'Place de Lille
LOGEMENTS = {"Maison", "Appartement"}


def distance_km(lat, lon, ref=LILLE_CENTRE):
    """Distance à vol d'oiseau (approximation équirectangulaire, < 0,1 % d'erreur à cette échelle)."""
    k = math.pi / 180
    x = (lon - ref[1]) * k * math.cos(ref[0] * k)
    y = (lat - ref[0]) * k
    return 6371 * (x**2 + y**2) ** 0.5


def load_raw() -> pd.DataFrame:
    mel = {c["code"]: c["nom"] for c in json.loads((RAW / "communes_mel.json").read_text(encoding="utf-8"))}
    frames = [pd.read_csv(f, dtype={"code_commune": str, "id_parcelle": str}, low_memory=False)
              for f in sorted(RAW.glob("dvf_59_*.csv.gz"))]
    df = pd.concat(frames, ignore_index=True)
    df = df[df["code_commune"].isin(mel)].copy()
    df["commune"] = df["code_commune"].map(mel)  # nom officiel (fusions de communes incluses)
    return df


def to_sales(df: pd.DataFrame) -> tuple[pd.DataFrame, list[tuple[str, int]]]:
    funnel = [("Mutations DVF dans la MEL, 2021-2025", df["id_mutation"].nunique())]

    df = df[df["nature_mutation"] == "Vente"]
    funnel.append(("Ventes simples (hors VEFA, échanges, adjudications, terrains à bâtir)", df["id_mutation"].nunique()))

    g = df.groupby("id_mutation")
    n_log = g["type_local"].apply(lambda s: s.isin(LOGEMENTS).sum())
    n_autre = g["type_local"].apply(lambda s: s.eq("Local industriel. commercial ou assimilé").sum())
    n_dep = g["type_local"].apply(lambda s: s.eq("Dépendance").sum())
    keep = n_log[(n_log == 1) & (n_autre == 0)].index
    df = df[df["id_mutation"].isin(keep)]
    funnel.append(("Un seul logement vendu (dépendances admises, pas de local commercial)", len(keep)))

    # Terrain : une ligne par (parcelle, culture), répétée pour chaque local -> dédoublonner avant de sommer.
    terrain = (df.dropna(subset=["surface_terrain"])
                 .drop_duplicates(["id_mutation", "id_parcelle", "code_nature_culture", "surface_terrain"])
                 .groupby("id_mutation")["surface_terrain"].sum())
    log = df[df["type_local"].isin(LOGEMENTS)].set_index("id_mutation")
    sales = pd.DataFrame({
        "date": pd.to_datetime(log["date_mutation"]),
        "commune": log["commune"],
        "code_commune": log["code_commune"],
        "type": log["type_local"],
        "prix": log["valeur_fonciere"],
        "surface": log["surface_reelle_bati"],
        "pieces": log["nombre_pieces_principales"],
        "terrain": terrain.reindex(log.index).fillna(0),
        "dependance": n_dep.reindex(log.index).gt(0).astype(int),
        "lat": log["latitude"],
        "lon": log["longitude"],
    })
    return sales, funnel


def clean(sales: pd.DataFrame, funnel: list) -> pd.DataFrame:
    s = sales.dropna(subset=["prix", "surface", "pieces", "lat", "lon"])
    funnel.append(("Prix, surface, pièces et géolocalisation renseignés", len(s)))

    s = s[s["surface"].between(10, 400) & s["pieces"].between(1, 12) & (s["prix"] >= 15_000)]
    s = s[~((s["type"] == "Appartement") & (s["surface"] > 250))]
    s = s[~((s["type"] == "Maison") & (s["terrain"] > 5_000))]  # corps de ferme, propriétés agricoles
    funnel.append(("Biens résidentiels plausibles (10-400 m², 1-12 pièces, prix ≥ 15 k€, terrain ≤ 5 000 m²)", len(s)))

    s = s.assign(prix_m2=s["prix"] / s["surface"])
    q = s.groupby("type")["prix_m2"].quantile([0.01, 0.99]).unstack()
    s = s[s.apply(lambda r: q.loc[r["type"], 0.01] <= r["prix_m2"] <= q.loc[r["type"], 0.99], axis=1)]
    funnel.append(("Prix au m² hors 1 % extrêmes de chaque côté, par type (ventes familiales, erreurs de saisie)", len(s)))

    # Résidences étudiantes / de services vendues lot par lot (investissement LMNP) : même point GPS,
    # même année, studios. Ce n'est pas le marché résidentiel (ex. 200 studios à Villeneuve-d'Ascq en
    # 2021 à 4 760 €/m², qui doublaient la médiane communale). Les grandes copropriétés qui se revendent
    # au fil de l'eau (10-30 ventes/an au même point, surfaces normales) sont conservées.
    s = s.assign(annee=s["date"].dt.year)
    appt = s["type"] == "Appartement"
    grappe = s[appt].groupby(["lat", "lon", "annee"])["surface"].agg(["size", "median"])
    residences = grappe[(grappe["size"] >= 5) & (grappe["median"] < 30)].index
    s = s[~(appt & s.set_index(["lat", "lon", "annee"]).index.isin(residences))]
    funnel.append(("Hors résidences étudiantes/de services vendues lot par lot (≥ 5 studios au même point la même année)", len(s)))

    s = s.assign(
        annee=s["date"].dt.year,
        trimestre=s["date"].dt.to_period("Q").astype(str),
        dist_lille_km=distance_km(s["lat"], s["lon"]).round(2),
    )
    return s.reset_index()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sales, funnel = to_sales(load_raw())
    s = clean(sales, funnel)
    s.to_parquet(OUT / "ventes_mel.parquet", index=False)
    f = pd.DataFrame(funnel, columns=["etape", "ventes"])
    f["ecartees"] = (-f["ventes"].diff()).fillna(0).astype(int)
    f.to_csv(ROOT / "docs" / "entonnoir.csv", index=False)
    print(f.to_string(index=False))
    print(s.groupby("type")["prix_m2"].describe().round(0))


if __name__ == "__main__":
    main()
