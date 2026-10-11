"""Prix des logements de la métropole lilloise — carte des quartiers, estimateur, indice.

N'embarque que des agrégats par quartier (≥ 30 ventes) et les coefficients du modèle :
aucune vente individuelle (conditions de réutilisation de DVF)."""
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd
import pydeck as pdk
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import estimateur as E  # noqa: E402

REPO = "https://github.com/valentinratigniet-byte/projet-22-prix-immobilier-mel"
LILLE = (50.6366, 3.0635)

st.set_page_config(page_title="Prix immobilier MEL", page_icon="🏠", layout="wide")


@st.cache_data
def charger():
    est = json.loads((ROOT / "outputs" / "estimateur.json").read_text(encoding="utf-8"))
    q = pd.read_csv(ROOT / "outputs" / "quartiers.csv")
    return est, q[q["fiable"]], pd.read_csv(ROOT / "outputs" / "indice_prix.csv"), pd.read_csv(ROOT / "outputs" / "validation.csv")


@st.cache_data(ttl=3600, show_spinner=False)
def geocoder(adresse: str):
    """API Adresse (Etat) : recherche orientée vers Lille ; renvoie le meilleur résultat ou None."""
    params = urllib.parse.urlencode({"q": adresse, "limit": 1, "lat": LILLE[0], "lon": LILLE[1]})
    with urllib.request.urlopen(f"https://api-adresse.data.gouv.fr/search/?{params}", timeout=10) as r:
        f = json.load(r)["features"]
    if not f:
        return None
    p, (lon, lat) = f[0]["properties"], f[0]["geometry"]["coordinates"]
    return {"label": p["label"], "citycode": p["citycode"], "score": p["score"], "lat": lat, "lon": lon}


def couleur(prime: float) -> list:
    """Décote (vert) → médiane (crème) → prime (rouge), saturée à ±40 %."""
    t = max(-1.0, min(1.0, prime / 40))
    vert, creme, rouge = (47, 163, 107), (250, 243, 221), (217, 83, 79)
    a, b = (creme, rouge) if t > 0 else (creme, vert)
    t = abs(t)
    return [int(a[i] + (b[i] - a[i]) * t) for i in range(3)] + [190]


def polygone(cle: str, deg: list) -> list:
    i, j = (int(x) for x in cle.split("_"))
    la, lo = i * deg[0], j * deg[1]
    dla, dlo = deg[0] / 2, deg[1] / 2
    return [[lo - dlo, la - dla], [lo + dlo, la - dla], [lo + dlo, la + dla], [lo - dlo, la + dla]]


est, quartiers, indice, validation = charger()
eur = lambda x: f"{x:,.0f} €".replace(",", " ")  # noqa: E731

st.title("🏠 Prix des logements de la métropole lilloise")
st.caption("73 271 ventes notariées réelles (DVF 2021-2025, 95 communes) · régression hédonique à effets fixes "
           f"quartier et trimestre · [méthode et code]({REPO}) · [note de synthèse]({REPO}/blob/main/docs/note-decideur.md)")

onglet_estim, onglet_carte, onglet_indice = st.tabs(["💶 Estimer un bien", "🗺️ Carte des quartiers", "📈 Évolution des prix"])

with onglet_estim:
    g, d = st.columns([1, 1], gap="large")
    with g:
        adresse = st.text_input("Adresse dans la métropole lilloise", "12 rue de la Monnaie, Lille")
        type_bien = st.radio("Type de bien", ["Appartement", "Maison"], horizontal=True)
        c1, c2 = st.columns(2)
        surface = c1.number_input("Surface habitable (m²)", 10, 400, 65 if type_bien == "Appartement" else 100)
        pieces = c2.number_input("Pièces principales", 1, 12, 3 if type_bien == "Appartement" else 5)
        terrain = st.number_input("Terrain (m²)", 0, 5000, 200) if type_bien == "Maison" else 0
        dependance = st.checkbox("Cave, parking ou garage vendu avec le bien")
        prix_demande = st.number_input("Prix demandé (facultatif, €)", 0, 3_000_000, 0, step=5_000)

    with d:
        try:
            geo = geocoder(adresse) if adresse.strip() else None
        except OSError:
            geo = None
            st.error("Le service de géolocalisation (API Adresse) ne répond pas, réessayez dans un instant.")
        if geo is None:
            st.info("Saisissez une adresse de la métropole lilloise.")
        elif geo["citycode"] not in est["communes"]:
            st.warning(f"« {geo['label']} » n'est pas dans l'une des 95 communes de la MEL.")
        else:
            commune = est["communes"][geo["citycode"]]
            try:
                cle, fine = E.quartier(est, type_bien, geo["lat"], geo["lon"], commune)
            except KeyError:
                st.warning(f"Pas assez de ventes de type « {type_bien.lower()} » à {commune} pour estimer.")
            else:
                r = E.estimer(est, type_bien, surface, pieces, dependance, terrain, cle)
                st.caption(f"📍 {geo['label']}" + ("" if geo["score"] > 0.6 else " · ⚠️ correspondance approximative"))
                st.metric("Prix de marché estimé", eur(r["prix"]), help="Prix au dernier trimestre observé, "
                          f"{est['trimestre_reference']}, à caractéristiques et quartier donnés.")
                st.write(f"**{eur(r['prix_m2'])}/m²** · 8 ventes comparables sur 10 se situent entre "
                         f"**{eur(r['bas'])}** et **{eur(r['haut'])}**.")
                st.caption("Référence : quartier (cellule d'environ 550 m)" if fine
                           else f"Référence : {commune}, hors quartiers détaillés (trop peu de ventes de ce type ici)")
                if prix_demande:
                    ecart = 100 * (prix_demande / r["prix"] - 1)
                    if prix_demande > r["haut"]:
                        st.error(f"Prix demandé **{ecart:+.0f} %** au-dessus de l'estimation, au-delà de la fourchette "
                                 "habituelle : argument de négociation, sauf qualité particulière (rénovation, DPE, vue).")
                    elif prix_demande < r["bas"]:
                        st.warning(f"Prix demandé **{ecart:+.0f} %** sous l'estimation, en dessous de la fourchette "
                                   "habituelle : à examiner de près (travaux, DPE, copropriété).")
                    else:
                        st.success(f"Prix demandé **{ecart:+.0f} %** par rapport à l'estimation : dans la fourchette du marché.")
        st.caption("L'état du bien, le DPE, l'étage et les travaux ne figurent pas dans les données de ventes : "
                   "l'estimation est un point de départ de négociation, pas une expertise.")

with onglet_carte:
    type_carte = st.radio("Type de bien ", ["Appartement", "Maison"], horizontal=True, key="carte")
    q = quartiers[quartiers["type"] == type_carte].copy()
    q["polygon"] = q["quartier"].map(lambda c: polygone(c, est["cellule_deg"]))
    q["couleur"] = q["prime_pct"].map(couleur)
    q["prime_txt"] = q["prime_pct"].map(lambda x: f"{x:+.0f} %")
    q["prix_txt"] = q["prix_m2_median"].map(lambda x: f"{x:,.0f} €/m²".replace(",", " "))
    st.pydeck_chart(pdk.Deck(
        layers=[pdk.Layer("PolygonLayer", q, get_polygon="polygon", get_fill_color="couleur",
                          get_line_color=[255, 255, 255, 120], line_width_min_pixels=0.5, pickable=True)],
        initial_view_state=pdk.ViewState(latitude=50.66, longitude=3.09, zoom=10.3),
        tooltip={"html": "<b>{commune}</b><br/>À bien égal : <b>{prime_txt}</b> vs quartier médian"
                         "<br/>Prix médian observé : {prix_txt}<br/>{ventes} ventes 2021-2025"},
    ), height=560)
    st.caption(f"{len(q)} quartiers d'au moins 30 ventes. Vert : moins cher à bien égal ; rouge : plus cher. "
               "Les quartiers dont un seul immeuble fait plus de la moitié des ventes sont masqués (prime non fiable).")
    c1, c2 = st.columns(2)
    cols = {"commune": "Commune", "prime_pct": "Prime à bien égal (%)", "prix_m2_median": "Prix médian (€/m²)", "ventes": "Ventes"}
    c1.write("**Les plus abordables à bien égal**")
    c1.dataframe(q.nsmallest(8, "prime_pct")[list(cols)].rename(columns=cols).round(0), hide_index=True)
    c2.write("**Les plus chers à bien égal**")
    c2.dataframe(q.nlargest(8, "prime_pct")[list(cols)].rename(columns=cols).round(0), hide_index=True)

with onglet_indice:
    st.write("**Prix à qualité constante** (base 100 = 1er trimestre 2021) : l'évolution des prix une fois les "
             "biens comparés à l'identique, sans l'effet des changements dans ce qui se vend.")
    st.line_chart(indice.pivot(index="trimestre", columns="type", values="indice"), color=["#137A8B", "#E4A93C"])
    st.write("À bien égal, les prix de 2024 sont inférieurs à ceux de 2022 de **3,9 %** (appartements) et "
             "**7,1 %** (maisons), un écart très significatif (p < 0,001) : la trace de la hausse des taux.")
    with st.expander("Fiabilité du modèle"):
        st.dataframe(validation.rename(columns={
            "type": "Type", "ventes_test": "Ventes test", "erreur_mediane_modele_pct": "Erreur médiane modèle (%)",
            "erreur_mediane_naive_pct": "Erreur médiane méthode naïve (%)", "part_a_10pct_modele": "À ±10 % — modèle (%)",
            "part_a_10pct_naive": "À ±10 % — naïve (%)"}), hide_index=True)
        st.caption("Validation sur 20 % de ventes jamais vues par le modèle. Méthode naïve : prix médian au m² de "
                   "la commune et de l'année × surface.")
