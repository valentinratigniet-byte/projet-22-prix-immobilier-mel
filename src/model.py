"""Régression hédonique du prix des logements de la MEL (DVF 2021-2025).

Un modèle par type de bien (coefficients lisibles plutôt qu'un modèle unique à interactions) :
    log(prix) = b1·log(surface) + pièces + dépendance [+ b·log(1+terrain) pour les maisons]
                + effet fixe quartier + effet fixe trimestre
Écarts-types robustes (HC3). Le « quartier » est une cellule d'environ 550 m de côté ; une cellule
de moins de MIN_VENTES ventes (par type) est regroupée avec le reste de sa commune. Les effets fixes
neutralisent le niveau de prix du quartier et du trimestre : les coefficients se lisent « à quartier
et date identiques », et l'effet quartier lui-même mesure combien un quartier coûte à bien égal.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "ventes_mel.parquet"
OUT = ROOT / "outputs"

FORMULES = {
    "Appartement": "log_prix ~ log_surface + C(pieces_c) + dependance + C(quartier) + C(trimestre)",
    "Maison": "log_prix ~ log_surface + C(pieces_c) + dependance + log_terrain + C(quartier) + C(trimestre)",
}
SEED = 22
CELLULE_DEG = (0.005, 0.0075)  # ≈ 555 m en latitude, ≈ 530 m en longitude à 50,6° N
MIN_VENTES = 30


def quartiers(s: pd.DataFrame) -> pd.Series:
    """Cellule de grille, ou « commune (reste) » si la cellule a trop peu de ventes de ce type."""
    cell = ((s["lat"] / CELLULE_DEG[0]).round().astype(int).astype(str) + "_"
            + (s["lon"] / CELLULE_DEG[1]).round().astype(int).astype(str))
    n = cell.groupby([s["type"], cell]).transform("size")
    return cell.where(n >= MIN_VENTES, s["commune"] + " (reste)")


def load() -> pd.DataFrame:
    s = pd.read_parquet(DATA)
    return s.assign(
        log_prix=np.log(s["prix"]),
        log_surface=np.log(s["surface"]),
        log_terrain=np.log1p(s["terrain"]),
        pieces_c=s["pieces"].clip(upper=6).astype(int),  # 6 = « 6 pièces et plus »
        quartier=quartiers(s),
    )


def fit(df: pd.DataFrame, type_bien: str):
    return smf.ols(FORMULES[type_bien], data=df[df["type"] == type_bien]).fit(cov_type="HC3")


def predict_prix(model, df: pd.DataFrame) -> pd.Series:
    """Retour à l'échelle des euros avec correction de Duan (smearing) : exp(E[log]) sous-estime E[prix]."""
    smear = np.mean(np.exp(model.resid))
    return np.exp(model.predict(df)) * smear


def holdout(df: pd.DataFrame, type_bien: str) -> dict:
    """80/20 aléatoire : modèle hédonique contre la méthode naïve « médiane €/m² de la commune × surface »."""
    d = df[df["type"] == type_bien]
    test = d.sample(frac=0.2, random_state=SEED)
    train = d.drop(test.index)
    m = fit(train, type_bien)
    test = test[test["quartier"].isin(train["quartier"]) & test["trimestre"].isin(train["trimestre"])]
    erreur_modele = (predict_prix(m, test) / test["prix"] - 1).abs()
    med_m2 = train.groupby(["commune", "annee"])["prix_m2"].median()
    naif = test.set_index(["commune", "annee"]).index.map(med_m2).to_numpy() * test["surface"].to_numpy()
    erreur_naive = (pd.Series(naif, index=test.index) / test["prix"] - 1).abs()
    return {
        "type": type_bien,
        "ventes_test": len(test),
        "erreur_mediane_modele_pct": round(100 * erreur_modele.median(), 1),
        "erreur_mediane_naive_pct": round(100 * erreur_naive.median(), 1),
        "part_a_10pct_modele": round(100 * (erreur_modele <= 0.10).mean(), 1),
        "part_a_10pct_naive": round(100 * (erreur_naive <= 0.10).mean(), 1),
    }


def coefficients(model, type_bien: str) -> pd.DataFrame:
    keep = [k for k in model.params.index if not k.startswith(("C(quartier)", "C(trimestre)", "Intercept"))]
    t = pd.DataFrame({
        "type": type_bien,
        "variable": keep,
        "coef": model.params[keep],
        "ic95_bas": model.conf_int().loc[keep, 0],
        "ic95_haut": model.conf_int().loc[keep, 1],
        "p_value": model.pvalues[keep],
    })
    # Effet en % sur le prix pour les variables binaires / catégorielles (exp(b) - 1)
    t["effet_pct"] = np.where(t["variable"].str.startswith(("log_",)), np.nan, 100 * (np.exp(t["coef"]) - 1))
    return t.reset_index(drop=True)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    df = load()
    coefs, valid, resid = [], [], []
    for type_bien in FORMULES:
        m = fit(df, type_bien)
        coefs.append(coefficients(m, type_bien))
        valid.append(holdout(df, type_bien))
        d = df[df["type"] == type_bien]
        resid.append(d.assign(residu=m.resid, prix_modele=predict_prix(m, d)))
        print(f"{type_bien}: n={int(m.nobs)}, R² ajusté={m.rsquared_adj:.3f}")
    pd.concat(coefs).to_csv(OUT / "coefficients.csv", index=False)
    pd.DataFrame(valid).to_csv(OUT / "validation.csv", index=False)
    pd.concat(resid).to_parquet(ROOT / "data" / "processed" / "ventes_residus.parquet", index=False)
    print(pd.DataFrame(valid).to_string(index=False))
    print(pd.concat(coefs).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
