"""Télécharge DVF géolocalisé (Nord, 2021-2025) et la liste officielle des 95 communes de la MEL."""
import json
import urllib.request
from pathlib import Path

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
YEARS = range(2021, 2026)
DVF_URL = "https://files.data.gouv.fr/geo-dvf/latest/csv/{year}/departements/59.csv.gz"
MEL_URL = "https://geo.api.gouv.fr/epcis/200093201/communes?fields=code,nom,population"


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    for year in YEARS:
        dest = RAW / f"dvf_59_{year}.csv.gz"
        if not dest.exists():
            urllib.request.urlretrieve(DVF_URL.format(year=year), dest)
        print(f"{dest.name}: {dest.stat().st_size / 1e6:.1f} Mo")
    with urllib.request.urlopen(MEL_URL) as r:
        communes = json.load(r)
    (RAW / "communes_mel.json").write_text(json.dumps(communes, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"communes MEL : {len(communes)}")


if __name__ == "__main__":
    main()
