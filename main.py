from fastapi import FastAPI, Query, HTTPException
from typing import Dict
from final_tide_constituent import get_constituents_by_latlon
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import pyTMD
import json

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

def load_constituents_from_dict(data: dict):
    c = []
    amp = []
    ph = []
    for name, vals in data.items():
        c.append(name)
        amp.append(vals.get('amplitude'))
        ph.append(vals.get('phase'))
    return np.array(amp, dtype=float), np.array(ph, dtype=float), c

def datetime2datetnum(time: np.ndarray):
    epoch = np.datetime64('1992-01-01T00:00:00', 's')
    time_s = time.astype('datetime64[s]')
    return ((time_s - epoch).astype(float)) / (3600 * 24)

@app.get("/tide-extrema/")
def tide_extrema(
    lat: float = Query(..., description="Latitude"),
    lon: float = Query(..., description="Longitude"),
    start: str = Query(..., description="Start time in YYYY-MM-DDTHH:MM:SS format"),
    end: str = Query(..., description="End time in YYYY-MM-DDTHH:MM:SS format")
):
    try:
        # Parse times
        t0 = datetime.fromisoformat(start)
        t1 = datetime.fromisoformat(end)
        dt = timedelta(minutes=1)
        times = np.array([t0 + i*dt for i in range(int((t1 - t0)/dt) + 1)], dtype='datetime64[ns]')

        # Get constituents
        data = get_constituents_by_latlon(lat, lon)
        amp, ph, c = load_constituents_from_dict(data)

        # Predict tide
        time_tmd = datetime2datetnum(times)
        cph = -1j * ph * np.pi / 180.0
        hc = amp * np.exp(cph)
        TIDE = pyTMD.predict.time_series(time_tmd, hc, c)
        MINOR = pyTMD.predict.infer_minor(time_tmd, hc, c)
        TIDE.data[:] += MINOR
        tide = TIDE.data

        # Find highs and lows
        series = pd.DataFrame({'time': times.astype('datetime64[s]'), 'tide': tide})
        vals = series['tide'].values
        diff = np.diff(vals)
        max_idx = []
        min_idx = []
        for i in range(1, len(vals)-1):
            prev = diff[i-1]
            next_ = diff[i] if i < len(diff) else 0.0
            if prev > 0 and next_ <= 0:
                max_idx.append(i)
            elif prev < 0 and next_ >= 0:
                min_idx.append(i)
        extra = []
        for i in max_idx:
            extra.append(('High', series.loc[i, 'time'], series.loc[i, 'tide']))
        for i in min_idx:
            extra.append(('Low', series.loc[i, 'time'], series.loc[i, 'tide']))
        extra.sort(key=lambda x: x[1])
        offset = pd.Timedelta(hours=12)
        json_rows = [
            {
                'type': t.lower(),
                'datetime': (pd.to_datetime(tm) + offset).strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z',
                'height': round(float(h), 3)
            }
            for t, tm, h in extra
        ]
        return json_rows
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
