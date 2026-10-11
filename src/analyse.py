"""Résultats métier à partir des modèles hédoniques : indice de prix à qualité constante,
test de la baisse post-hausse des taux, prime/décote de chaque quartier, figures."""
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import model as M  # noqa: E402

OUT = M.OUT
PETROL, AMBRE, VERT, ROUGE, GRIS = "#137A8B", "#E4A93C", "#2FA36B", "#D9534F", "#8A9599"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def indice_prix(model, type_bien: str) -> pd.DataFrame:
    """Indice trimestriel à qualité constante (base 100 = 2021T1), IC 95 % robustes."""
    rows = [{"trimestre": "2021Q1", "indice": 100.0, "ic_bas": 100.0, "ic_haut": 100.0}]
    ci = model.conf_int()
    for k in [k for k in model.params.index if k.startswith("C(trimestre)")]:
        rows.append({"trimestre": k.split("[T.")[1].rstrip("]"),
                     "indice": 100 * np.exp(model.params[k]),
                     "ic_bas": 100 * np.exp(ci.loc[k, 0]), "ic_haut": 100 * np.exp(ci.loc[k, 1])})
    return pd.DataFrame(rows).assign(type=type_bien).sort_values("trimestre")


def test_hausse_des_taux(model, type_bien: str) -> dict:
    """H0 : le prix à qualité constante est le même en 2024 qu'en 2022 (moyenne des 4 trimestres)."""
    names = list(model.params.index)
    w = np.zeros(len(names))
    for annee, signe in (("2022", -1), ("2024", 1)):
        for q in range(1, 5):
            w[names.index(f"C(trimestre)[T.{annee}Q{q}]")] = signe / 4
    t = model.t_test(w)
    ecart = float(np.squeeze(t.effect))
    ci = np.squeeze(t.conf_int())
    return {"type": type_bien, "ecart_2024_vs_2022_pct": round(100 * (np.exp(ecart) - 1), 1),
            "ic95_bas_pct": round(100 * (np.exp(ci[0]) - 1), 1), "ic95_haut_pct": round(100 * (np.exp(ci[1]) - 1), 1),
            "p_value": float(np.squeeze(t.pvalue))}


def effets_quartiers(model, df: pd.DataFrame, type_bien: str) -> pd.DataFrame:
    """Prime/décote de chaque cellule par rapport à la médiane des cellules, à bien et date égaux."""
    d = df[df["type"] == type_bien]
    fe = {k.split("[T.")[1].rstrip("]"): v for k, v in model.params.items() if k.startswith("C(quartier)")}
    base = sorted(d["quartier"].unique())[0]  # modalité de référence de patsy
    fe[base] = 0.0
    g = d.groupby("quartier").agg(ventes=("prix", "size"), lat=("lat", "mean"), lon=("lon", "mean"),
                                 commune=("commune", lambda x: x.mode().iloc[0]),
                                 prix_m2_median=("prix_m2", "median"))
    # Part de la cellule venant du premier immeuble (même point GPS) : au-delà de 50 %, la « prime du
    # quartier » mesure surtout cet immeuble (ex. résidence récente revendue en bloc, que DVF ne signale
    # pas comme neuve hors VEFA). Signalée, et exclue de la carte.
    top = d.groupby(["quartier", "lat", "lon"]).size().groupby(level=0).max()
    g["part_premier_immeuble"] = (top / g["ventes"]).round(2)
    g["fiable"] = g["part_premier_immeuble"] <= 0.5
    g["effet_log"] = g.index.map(fe)
    g["prime_pct"] = 100 * (np.exp(g["effet_log"] - g["effet_log"].median()) - 1)
    g = g[~g.index.str.endswith("(reste)")]  # seules les vraies cellules ont une position fiable
    return g.reset_index().assign(type=type_bien).drop(columns="effet_log")


def figures(indices: pd.DataFrame, coefs: pd.DataFrame, q: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    for t, c in (("Appartement", PETROL), ("Maison", AMBRE)):
        d = indices[indices["type"] == t]
        x = range(len(d))
        ax.plot(x, d["indice"], color=c, lw=2, label=t)
        ax.fill_between(x, d["ic_bas"], d["ic_haut"], color=c, alpha=0.15)
    ax.set_xticks(range(0, len(d), 4), d["trimestre"].iloc[::4].str[:4])
    ax.axhline(100, color=GRIS, lw=0.8, ls=":")
    ax.set_title("Prix à qualité constante, MEL (base 100 = 1er trimestre 2021, IC 95 %)", loc="left")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "indice_prix.png", dpi=150)

    fig, ax = plt.subplots(figsize=(7, 7))
    d = q[(q["type"] == "Appartement") & q["fiable"]]
    sc = ax.scatter(d["lon"], d["lat"], c=d["prime_pct"].clip(-40, 40), s=np.sqrt(d["ventes"]) * 6,
                    cmap="RdYlGn_r", vmin=-40, vmax=40, edgecolors="white", linewidths=0.3)
    fig.colorbar(sc, ax=ax, shrink=0.7, label="Prime (+) ou décote (−) du quartier, %")
    ax.set_title("Appartements : ce que coûte chaque quartier à bien égal", loc="left")
    ax.set_aspect(1 / np.cos(np.radians(50.63)))
    ax.set_xticks([]), ax.set_yticks([])
    reperes = ["Lille", "Roubaix", "Tourcoing", "Villeneuve-d'Ascq", "Armentières", "Wattignies", "Marcq-en-Barœul"]
    for com, r in d[d["commune"].isin(reperes)].groupby("commune")[["lon", "lat"]].mean().iterrows():
        ax.annotate(com, (r["lon"], r["lat"]), fontsize=8, color="#222", ha="center", va="center",
                    bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "none", "alpha": 0.8})
    fig.tight_layout()
    fig.savefig(OUT / "carte_quartiers_appartements.png", dpi=150)

    d = coefs[coefs["variable"].isin(["dependance"]) | coefs["variable"].str.startswith("C(pieces_c)")].copy()
    d["libelle"] = d["variable"].str.replace(r"C\(pieces_c\)\[T\.(\d)\]", r"\1 pièces (vs 1)", regex=True) \
                                .str.replace("6 pièces", "6+ pièces").replace({"dependance": "Dépendance (cave, parking)"})
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharex=True)
    for ax, (t, c) in zip(axes, (("Appartement", PETROL), ("Maison", AMBRE))):
        e = d[d["type"] == t]
        lo, hi = 100 * (np.exp(e["ic95_bas"]) - 1), 100 * (np.exp(e["ic95_haut"]) - 1)
        ax.errorbar(e["effet_pct"], range(len(e)), xerr=[e["effet_pct"] - lo, hi - e["effet_pct"]],
                    fmt="o", color=c, ecolor=GRIS, capsize=3)
        ax.set_yticks(range(len(e)), e["libelle"])
        ax.axvline(0, color=GRIS, lw=0.8)
        ax.set_title(t, loc="left")
        ax.set_xlabel("Effet sur le prix, % (surface égale)")
    fig.tight_layout()
    fig.savefig(OUT / "effets_caracteristiques.png", dpi=150)
    plt.close("all")


def main() -> None:
    df = M.load()
    indices, tests, quart, coefs = [], [], [], []
    for t in M.FORMULES:
        m = M.fit(df, t)
        indices.append(indice_prix(m, t))
        tests.append(test_hausse_des_taux(m, t))
        quart.append(effets_quartiers(m, df, t))
        coefs.append(M.coefficients(m, t))
    indices, quart, coefs = pd.concat(indices), pd.concat(quart), pd.concat(coefs)
    indices.round(2).to_csv(OUT / "indice_prix.csv", index=False)
    pd.DataFrame(tests).to_csv(OUT / "test_hausse_taux.csv", index=False)
    quart.round(4).to_csv(OUT / "quartiers.csv", index=False)
    figures(indices, coefs, quart)
    print(pd.DataFrame(tests).to_string(index=False))
    print("quartiers signalés (un immeuble > 50 % des ventes) :", (~quart["fiable"]).groupby(quart["type"]).sum().to_dict())
    a = quart[(quart["type"] == "Appartement") & quart["fiable"]].sort_values("prime_pct")
    print(a[["commune", "ventes", "prix_m2_median", "prime_pct"]].head(6).round(1).to_string(index=False))
    print(a[["commune", "ventes", "prix_m2_median", "prime_pct"]].tail(6).round(1).to_string(index=False))


if __name__ == "__main__":
    main()
