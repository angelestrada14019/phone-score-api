
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Union, Optional
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from analyzer import PhoneAnalyzer
import random
import asyncio
import time
import joblib
import os
import numpy as np
from pathlib import Path

app = FastAPI(title="Smartphone Evaluation API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_path = Path("phone_analyzer_pro.pkl")
try:
    analyzer = joblib.load(model_path)
    print(f"Successfully loaded model from {model_path}")
except Exception as e:
    print(f"Warning: Could not load model: {e}")
    analyzer = None


class SmartphoneInput(BaseModel):
    internal_storage: int  # in GB
    storage_ram: int  # in GB
    expandable_storage: Union[float, str]  # in TB or "NA"
    primary_camera: str  # e.g. "108MP + 12MP + 5MP + 5MP"
    display: str  # e.g. "Full HD+ Dynamic AMOLED 2X"
    network: str  # e.g. "5G, 4G, 3G, 2G"
    battery: str  # in mAh

class MetricsOutput(BaseModel):
    gaming_potential: int  # 0-100
    battery_performance: int  # 0-100
    photography: int  # 0-100
    display_quality: int  # 0-100

class SmartphoneEvaluation(SmartphoneInput):
    id: str
    overall_score: int  # 0-100
    performance_category: str  # "LOW" | "MID" | "HIGH"
    price_range: str  # e.g. "$300 - $400"
    user_recommendation: str
    metrics: MetricsOutput

# Sample data for initial display
sample_smartphones = [
{
    'internal_storage(GB)': 256,
    'storage_ram(GB)': 8,
    'expandable_storage(TB)': 1,
    'primary_camera': '108MP + 12MP + 5MP + 5MP',
    'display': 'Full HD+ Dynamic AMOLED 2X DisplayHD',
    'network': '5G, 4G, 3G, 2G',
    'battery': '5000  mAh'
},
{
    'internal_storage(GB)': 64,
    'storage_ram(GB)': 8,
    'expandable_storage(TB)': np.nan,
    'primary_camera': '50MP + 10MP + 12MP',
    'display': 'LCDHD',
    'network': '5G, 4G, 3G, 2G',
    'battery': '6000  mAh'
},
{
    'internal_storage(GB)': 100,
    'storage_ram(GB)': 12,
    'expandable_storage(TB)': 1,
    'primary_camera': '108MP + 8MP + 2MP',
    'display': 'Full HD+ Super AMOLED Plus DisplayHD',
    'network': '5G, 4G, 3G, 2G',
    'battery': '4400  mAh'
}
]

# Adapt input data to match model's expected format
def adapt_input_for_model(input_data: SmartphoneInput):

    if isinstance(input_data.expandable_storage, str) and input_data.expandable_storage.upper() == "NA":
        expandable_storage = np.nan
    else:
        expandable_storage = float(input_data.expandable_storage)

    
    return {
        'internal_storage(GB)': input_data.internal_storage,
        'storage_ram(GB)': input_data.storage_ram,
        'expandable_storage(TB)': expandable_storage,
        'primary_camera': input_data.primary_camera,
        'display': input_data.display,
        'network': input_data.network,
        'battery': input_data.battery
    }

def map_category_to_api(category: str) -> str:
    """Maps model category to API category format"""
    category = category.upper()
    if category in ["HIGH", "ALTA"]:
        return "HIGH"
    elif category in ["MID", "MEDIA"]:
        return "MID"
    else:
        return "LOW"

# Fallback evaluation if model loading fails
def fallback_evaluate_smartphone(input_data: SmartphoneInput) -> dict:
    """
    Fallback evaluation when the model is not available
    """
    print("Using fallback evaluation since model could not be loaded")
    
    # Calculate a weighted score based on specs
    storage_score = (input_data.internal_storage / 512) * 100
    ram_score = (input_data.storage_ram / 16) * 100
    
    # Calculate camera score based on MP values
    cameras = [int(''.join(filter(str.isdigit, c))) for c in input_data.primary_camera.split("+") if any(char.isdigit() for char in c)]
    mp_sum = sum(cameras)
    camera_score = (mp_sum / 200) * 100
    
    # Calculate display score
    display_score = 0
    if "amoled" in input_data.display.lower():
        display_score = 90
    elif "oled" in input_data.display.lower():
        display_score = 85
    elif "lcd" in input_data.display.lower():
        display_score = 70
    else:
        display_score = 60
    
    # Add bonus for resolution
    if "full hd" in input_data.display.lower() or "1080p" in input_data.display.lower():
        display_score += 5
    elif "2k" in input_data.display.lower() or "1440p" in input_data.display.lower():
        display_score += 8
    elif "4k" in input_data.display.lower():
        display_score += 10
    
    display_score = min(100, display_score)
    
    # Calculate network score
    network_score = 0
    if "5g" in input_data.network.lower():
        network_score = 100
    elif "4g" in input_data.network.lower() or "lte" in input_data.network.lower():
        network_score = 80
    else:
        network_score = 60
    
    # Calculate battery score
    battery_score = (input_data.battery / 6000) * 100
    
    # Combined weighted score
    combined_score = (
        (storage_score * 0.15) +
        (ram_score * 0.25) +
        (camera_score * 0.2) +
        (display_score * 0.15) +
        (network_score * 0.1) +
        (battery_score * 0.15)
    )
    
    # Determine category based on score
    if combined_score >= 80:
        performance_category = "HIGH"
    elif combined_score >= 60:
        performance_category = "MID"
    else:
        performance_category = "LOW"
        
    # Calculate price estimation
    base_price = (
        input_data.internal_storage * 0.5 + 
        input_data.storage_ram * 100 + 
        mp_sum * 3 + 
        (300 if "amoled" in input_data.display.lower() else 
         250 if "oled" in input_data.display.lower() else 
         150 if "lcd" in input_data.display.lower() else 100) + 
        (200 if "5g" in input_data.network.lower() else 100) + 
        input_data.battery / 20
    )
    
    min_price = round(base_price * 0.9)
    max_price = round(base_price * 1.1)
    price_range = f"${min_price} - ${max_price}"
    
    # Calculate specific metrics
    gaming_potential = round((ram_score * 0.6) + (battery_score * 0.2) + (storage_score * 0.2))
    battery_performance = round(battery_score)
    photography = round(camera_score)
    display_quality = round(display_score)
    
    return {
        "performance_category": performance_category,
        "price_range": price_range,
        "metrics": {
            "gaming_potential": gaming_potential,
            "battery_performance": battery_performance,
            "photography": photography,
            "display_quality": display_quality
        },
        "overall_score": round(combined_score)
    }

# ML model evaluation
def evaluate_smartphone(input_data: SmartphoneInput) -> SmartphoneEvaluation:
    """
    Main evaluation function that uses the loaded model or falls back to calculation
    """
    # Generate a random ID
    id = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=7))
    
    try:
        if analyzer is not None:
            # Convert input to the format expected by the model
            model_input = adapt_input_for_model(input_data)
            print(f"Using model to evaluate: {model_input}")
            
            # Use the model to evaluate the smartphone
            result = analyzer.evaluate_phone(model_input)
            print(f"Model evaluation result: {result}")
            
            # Extract results from the model output
            performance_category = map_category_to_api(result['gama'])
            price_range = f"${result['precio_estimado']}"
            
            # Extract technical scores
            metrics = {
                "gaming_potential": int(result['puntajes_tecnicos'].get('gaming', 50)),
                "battery_performance": int(result['puntajes_tecnicos'].get('bateria', 50)),
                "photography": int(result['puntajes_tecnicos'].get('fotografia', 50)),
                "display_quality": int(result['puntajes_tecnicos'].get('pantalla', 50))
            }
            
            # Calculate overall score as average of metrics
            overall_score = round(sum(metrics.values()) / len(metrics))
            
        else:
            print("Model not available, using fallback evaluation")
            # Use fallback evaluation if model isn't available
            # fallback_result = fallback_evaluate_smartphone(input_data)
            # performance_category = fallback_result["performance_category"]
            # price_range = fallback_result["price_range"]
            # metrics = fallback_result["metrics"]
            # overall_score = fallback_result["overall_score"]
    
    except Exception as e:
        print(f"Error using model, falling back to calculation: {str(e)}")
        # fallback_result = fallback_evaluate_smartphone(input_data)
        # performance_category = fallback_result["performance_category"]
        # price_range = fallback_result["price_range"]
        # metrics = fallback_result["metrics"]
        # overall_score = fallback_result["overall_score"]
    
    # Generate user recommendation
    if performance_category == "HIGH" and metrics["gaming_potential"] > 80:
        user_recommendation = "Ideal para gaming intensivo y uso prolongado"
    elif performance_category == "HIGH" and metrics["photography"] > 85:
        user_recommendation = "Excelente para fotografía profesional y redes sociales"
    elif performance_category == "MID" and metrics["battery_performance"] > 80:
        user_recommendation = "Perfecto para usuarios que requieren larga duración de batería"
    elif performance_category == "HIGH":
        user_recommendation = "Recomendado para usuarios exigentes multipropósito"
    elif performance_category == "MID":
        user_recommendation = "Buen equilibrio para uso diario"
    else:
        user_recommendation = "Adecuado para uso básico"
    
    return SmartphoneEvaluation(
        id=id,
        internal_storage=input_data.internal_storage,
        storage_ram=input_data.storage_ram,
        expandable_storage=input_data.expandable_storage,
        primary_camera=input_data.primary_camera,
        display=input_data.display,
        network=input_data.network,
        battery=input_data.battery,
        overall_score=overall_score,
        performance_category=performance_category,
        price_range=price_range,
        user_recommendation=user_recommendation,
        metrics=MetricsOutput(**metrics)
    )

# API Endpoints
@app.get("/")
def read_root():
    return {"status": "ok", "message": "Smartphone Evaluation API is running"}

@app.get("/smartphones/samples", response_model=List[SmartphoneInput])
def get_sample_smartphones():
    return sample_smartphones

@app.post("/smartphones/evaluate", response_model=SmartphoneEvaluation)
async def evaluate_smartphone_endpoint(smartphone: SmartphoneInput):
    # Simulate API delay
    await asyncio.sleep(1.5)
    
    try:
        print(f"Evaluating smartphone with specs: {smartphone}")
        result = evaluate_smartphone(smartphone)
        print(f"Evaluation result: {result}")
        return result
    except Exception as e:
        print(f"Error evaluating smartphone: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error evaluating smartphone: {str(e)}")

# This would be used to run the server
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
