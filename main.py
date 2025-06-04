from fastapi import FastAPI, Request, Form, Depends
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import logging
from datetime import datetime
import os

app = FastAPI()

# Configuration des dossiers
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Charger le modèle
model_path = "modele_energie.pkl"
try:
    model = joblib.load(model_path)
    logger.info("Modèle chargé avec succès")
except Exception as e:
    logger.error(f"Erreur lors du chargement du modèle: {str(e)}")
    raise e

# Mapping des mois
MONTH_NAMES = {
    1: "Janvier", 2: "Février", 3: "Mars", 4: "Avril",
    5: "Mai", 6: "Juin", 7: "Juillet", 8: "Août",
    9: "Septembre", 10: "Octobre", 11: "Novembre", 12: "Décembre"
}

# Schéma d'entrée
class PredictionInput(BaseModel):
    household_size: float
    heating_cooling: float  # ⚠️ Changé en float (1.0 ou 0.0)
    living_area: float
    month_num: int
    season: str
    housing_type: str

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(
    request: Request,
    household_size: float = Form(...),
    heating_cooling: str = Form(...),
    living_area: float = Form(...),
    month_num: int = Form(...),
    season: str = Form(...),
    housing_type: str = Form(...)
):
    try:
        # Conversion des données
        season_encoded = {"winter": 0, "spring": 1, "summer": 2, "fall": 3}.get(season, 0)
        heating_cooling_encoded = 1.0 if heating_cooling == "yes" else 0.0
        housing_type_encoded = 1.0 if housing_type == "house" else 0.0

        # Création du DataFrame
        input_data = {
            "household_size": [household_size],
            "heating_cooling": [heating_cooling_encoded],
            "living_area": [living_area],
            "month_num": [month_num],
            "season": [season_encoded],
            "housing_type": [housing_type_encoded]
        }
        input_df = pd.DataFrame(input_data)

        # Prédiction
        prediction = model.predict(input_df)
        prediction_value = round(float(prediction[0]), 2)
        
        # Préparation des données pour l'affichage
        details = {
            "month_name": MONTH_NAMES.get(month_num, "Inconnu"),
            "month_num": month_num,
            "season": season,
            "household_size": household_size,
            "housing_type": "Appartement" if housing_type == "apartment" else "Maison",
            "living_area": f"{living_area:.1f}",
            "heating_cooling": "Oui" if heating_cooling == "yes" else "Non",
            "prediction": prediction_value,  # Valeur numérique
            "prediction_formatted": f"{prediction_value:,.2f}".replace(",", " ")  # Format "28 385.37"
        }

        return templates.TemplateResponse("result.html", {
            "request": request,
            "details": details,
            "prediction": details["prediction"],  # Double sécurité
            "prediction_formatted": details["prediction_formatted"]
        })

    except Exception as e:
        logger.error(f"Erreur: {str(e)}")
        return templates.TemplateResponse("error.html", {
            "request": request,
            "error": f"Erreur lors du calcul: {str(e)}"
        })
@app.get("/model_features")
async def get_model_features():
    return {"features": ['household_size', 'heating_cooling', 'living_area', 'month_num', 'season', 'housing_type']}