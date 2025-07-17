from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any
import csv
import os

app = FastAPI()

# Enable CORS for all origins (for demo purposes)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = os.path.join(os.path.dirname(__file__), '../data/emission')

_emit_cache = None
_cams_cache = None

@app.get("/emit")
def get_emit() -> List[Dict[str, Any]]:
    global _emit_cache
    if _emit_cache is not None:
        return _emit_cache
    file_path = os.path.join(DATA_DIR, 'emit_clean.csv')
    results = []
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # Parse numeric fields
            for field in ["latitude", "longitude", "flux t/h", "uncertainty t/h"]:
                if field in row:
                    try:
                        row[field] = float(row[field])
                    except Exception:
                        pass
            results.append(row)
    _emit_cache = results
    return results

@app.get("/cams")
def get_cams() -> List[Dict[str, Any]]:
    global _cams_cache
    if _cams_cache is not None:
        return _cams_cache
    file_path = os.path.join(DATA_DIR, 'cams_clean.csv')
    results = []
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # Parse numeric fields
            for field in ["latitude", "longitude", "flux t/h", "uncertainty t/h"]:
                if field in row:
                    try:
                        row[field] = float(row[field])
                    except Exception:
                        pass
            results.append(row)
    _cams_cache = results
    return results 