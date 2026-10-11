"""L'application démarre sans exception ; l'API Adresse est simulée (pas de réseau en test)."""
import io
import json
from pathlib import Path
from unittest import mock

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app" / "app.py"
pytestmark = pytest.mark.skipif(not (APP.parents[1] / "outputs" / "estimateur.json").exists(), reason="pas d'export")
REPONSE = {"features": [{"properties": {"label": "12 Rue de la Monnaie 59800 Lille", "citycode": "59350", "score": 0.98},
                         "geometry": {"coordinates": [3.063804, 50.640769]}}]}


def lancer(urlopen):
    st.cache_data.clear()  # sinon la géolocalisation du test précédent est resservie
    with mock.patch("urllib.request.urlopen", urlopen):
        at = AppTest.from_file(str(APP), default_timeout=60)
        at.run()
    assert not at.exception
    return at


def test_estimation_affichee():
    at = lancer(lambda *a, **k: io.BytesIO(json.dumps(REPONSE).encode()))
    prix = [m for m in at.metric if m.label == "Prix de marché estimé"]
    assert prix and prix[0].value.endswith("€")


def test_api_adresse_en_panne():
    at = lancer(mock.Mock(side_effect=OSError("timeout")))
    assert any("ne répond pas" in e.value for e in at.error)
