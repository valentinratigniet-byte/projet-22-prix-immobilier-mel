"""Estimation d'un prix à partir de outputs/estimateur.json : aucune dépendance au-delà de la
bibliothèque standard, pour tourner dans l'application Streamlit (et être testée contre statsmodels)."""
import math


def cellule(lat: float, lon: float, deg: list) -> str:
    """Même découpage que src/model.py:quartiers() (round Python = round pandas : arrondi au pair)."""
    return f"{round(lat / deg[0])}_{round(lon / deg[1])}"


def quartier(est: dict, type_bien: str, lat: float, lon: float, commune: str) -> tuple[str, bool]:
    """Cellule du modèle si elle a assez de ventes, sinon « commune (reste) ». Renvoie (clé, est_cellule)."""
    q = est["types"][type_bien]["quartiers"]
    c = cellule(lat, lon, est["cellule_deg"])
    if c in q:
        return c, True
    if f"{commune} (reste)" in q:
        return f"{commune} (reste)", False
    raise KeyError(f"Aucune référence de prix pour {commune} ({type_bien})")


def estimer(est: dict, type_bien: str, surface: float, pieces: int, dependance: bool,
            terrain: float, quartier_cle: str, trimestre: str | None = None) -> dict:
    m = est["types"][type_bien]
    log_prix = (m["intercept"] + m["log_surface"] * math.log(surface)
                + m["pieces"][str(min(max(int(pieces), 1), 6))]
                + m["dependance"] * int(dependance)
                + m["log_terrain"] * math.log1p(terrain if type_bien == "Maison" else 0)
                + m["quartiers"][quartier_cle]
                + m["trimestres"][trimestre or est["trimestre_reference"]])
    prix = math.exp(log_prix) * m["smearing"]
    bas, haut = m["fourchette"]
    return {"prix": prix, "bas": prix * bas, "haut": prix * haut, "prix_m2": prix / surface}
