from datetime import datetime, timedelta
import json
import numpy as np
import pyTMD
import matplotlib.pyplot as plt
import pandas as pd


def load_constituents(json_path:str):
	with open(json_path, 'r') as f:
		data = json.load(f)
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


def main():
	# Time vector
	t0 = datetime(2025, 12, 10)
	t1 = datetime(2025, 12, 25)
	dt = timedelta(minutes=1)
	times = np.array([t0 + i*dt for i in range(int((t1 - t0)/dt) + 1)], dtype='datetime64[ns]')

	# Load constituents from lautoka.json
	amp, ph, c = load_constituents('lautoka.json')

	# Use provided snippet logic to predict tide
	
	time_tmd = datetime2datetnum(times)
	cph = -1j * ph * np.pi / 180.0
	hc = amp * np.exp(cph)
	TIDE = pyTMD.predict.time_series(time_tmd, hc, c)
	MINOR = pyTMD.predict.infer_minor(time_tmd, hc, c)
	TIDE.data[:] += MINOR
	tide = TIDE.data
    
	# Build alternating High/Low extrema sequence across the whole period
	series = pd.DataFrame({'time': times.astype('datetime64[s]'), 'tide': tide})
	vals = series['tide'].values
	# Find local maxima and minima using sign changes in first differences
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
	# Merge extrema by time order
	extra = []
	for i in max_idx:
		extra.append(('High', series.loc[i, 'time'], series.loc[i, 'tide']))
	for i in min_idx:
		extra.append(('Low', series.loc[i, 'time'], series.loc[i, 'tide']))
	extra.sort(key=lambda x: x[1])
	# Write JSON alternating high/low rows with ISO 8601 timestamps (T, Z), and height
	offset = pd.Timedelta(hours=12)
	json_rows = [
		{
			'type': t.lower(),
			'datetime': (pd.to_datetime(tm) + offset).strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z',
			'height': round(float(h), 3)
		}
		for t, tm, h in extra
	]
	print(json.dumps(json_rows, indent=2))





if __name__ == '__main__':
	main()