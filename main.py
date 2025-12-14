from fastapi import FastAPI, Query, HTTPException
from typing import Dict
from final_tide_constituent import get_constituents_by_latlon

app = FastAPI(root_path="/tide")

@app.get("/", tags=["Root"])
def read_root():
    return {"message": "welcome to tide prediction application"}
@app.get("/tide-constituents/")
def tide_constituents(lat: float = Query(..., description="Latitude"), lon: float = Query(..., description="Longitude")) -> Dict[str, Dict[str, float]]:
    try:
        result = get_constituents_by_latlon(lat, lon)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
