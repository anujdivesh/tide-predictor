# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 2025

author: gregoires 

pyTMD wrapper class object for the TPXO model.

Tested against pyTMD versions:
    - 2.2.4
    - 2.2.5

"""

from datetime import datetime
import itertools
import numpy as np
import os
import pyTMD
from pyTMD.io.OTIS import read_otis_grid
import sys


class TideModel:

    # root = os.path.join('S:\\', 'GEM', 'FJ_NAB', 'O_M', 'Oceans', 'Active', 'MARINE_SURVEYS_SURVEYS', 'Greggy', 'tideModels')
    root = os.path.join('D:\\', 'tideModels')
    EPOCH = np.datetime64('1992-01-01T00:00:00', 's')

    def __init__(self):
        
        # Instantiate the tide model
        self.Model = pyTMD.io.model(directory=self.root)
        self.Model.format = 'OTIS'
        self.Model.grid_file = os.path.join(self.root, 'TPXO10v2', 'DATA', 'grid_tpxo10')
        self.Model.model_file = os.path.join(self.root, 'TPXO10v2', 'DATA', 'h_tpxo10.v2')
        self.Model.name = 'TPXO10v2'
        self.Model.projection = '4326'
        self.Model.type = 'z'
        self.Model.variable = 'tide_ocean'

        # Load the global model grid and store its land mask (False values) and coordinates
        x, y, _, mz, _, _ = read_otis_grid(self.Model.grid_file)
        mlon, mlat = np.meshgrid(x, y)
        
        # Store the values and flatten the arrays for ease of use
        self.mlon = mlon.flatten()
        self.mlat = mlat.flatten()
        
        # Convert the interger flags to boolean (0=land, 1=ocean)
        self.modelMsk = np.full(mz.shape, False)
        self.modelMsk[mz==1] = True
        self.modelMsk = self.modelMsk.flatten()

        
    @staticmethod
    def datetime2datetnum(time:np.ndarray[datetime]|np.ndarray[np.datetime64])->np.ndarray[float]:
        """
        Convert a time vector of datetimes or datetime64 to float with regard to
        pyTMD temporal datum.

        :param np.ndarray[datetime] | np.ndarray[np.datetime64] time: Time vector
        
        
        :return np.ndarray[float]: Time vector converted to float wrt. pyTMD temporal datum
        """
        time = time.astype('datetime64[s]')
        datenum = (time - TideModel.EPOCH).astype(float) / (3600 * 24)
        return datenum
    
    
    def rayleigh_criterion(self, lon, lat, recordDuration=None):

        # Extract all the constants available from the tide model
        amp, _, c = self.Model.extract_constants(lon, lat, **{'extrapolate' : True})
        omega = pyTMD.arguments.frequency(c) * 86400 / (2 * np.pi) # Conversion from rad/s to cycle/day

        # Conversion from ma.masked_array to classic array
        amp = amp.__array__()[0]

        # Estimate the limit of the rayleigh criterion
        res = 1.0 / recordDuration
        n = len(c)
        keep = np.ones(n, dtype=bool)
        issues = []
        for ii, jj in itertools.combinations(range(n), 2):
            df = abs(omega[ii] - omega[jj])
            if df < res:
                issues.append((c[ii], c[jj], df))
                if amp is not None:
                    if amp[ii] >= amp[jj]:
                        keep[jj] = False
                    else:
                        keep[ii] = False

        if amp is not None:
            recommended = [ c for c, k in zip(c, keep) if k ]
        else:
            recommended = c

        out = {
            'resolution_cpd': res,
            'unresolvable_pairs': issues,
            'recommended_constituents': recommended
            }
        return out

    @staticmethod
    def generate_constants_from_tide_elevation(
            time:np.ndarray[datetime]|np.ndarray[np.datetime64],
            record:np.ndarray[float],
            constituents:list[str]
        )->tuple[np.ndarray[float], np.ndarray[float]]:
        """
        Wrapper function to estimate the harmonic constants from a tidal record.


        :param np.ndarray[datetime] | np.ndarray[np.datetime64] time: Array containing the timestamps at
        which the data was recorded
        
        :param np.ndarray[float] record: Surface Elevation record.
        
        :param list[str] constituents: List of VALID harmonic constant names
        
        
        :return np.ndarray[float] amplitude: Harmonic amplitudes (in meters)
        :return np.ndarray[float] phase: Harmonic phases (in degrees)
        """


        # Convert datetime into datenum and generate the constants
        time_tmd = TideModel.datetime2datetnum(time)
        amplitude, phase = pyTMD.solve.constants(
                            t=time_tmd,
                            ht=record,
                            constituents=constituents
                        )
        return amplitude, phase

    def generate_tide_elevation_from_constants(self,
            time:np.ndarray[datetime]|np.ndarray[np.datetime64],
            lon:float|np.ndarray[float],
            lat:float|np.ndarray[float],
            amp:float|np.ndarray[float]=None,
            ph:float|np.ndarray[float]=None,
            c:list[str]=None,
            nearest:bool=True
            )->np.ndarray[float]:
        """
        Wrapper function to generate the tide elevation from Harmonic constants.

        :param np.ndarray[datetime] | np.ndarray[np.datetime64] time: 
        Array containing the timestamps at which the tide will be estimated

        :param float | np.ndarray[float] lon: Longitude at which the tide will be estimated.
        Pass lon = None if you have the harmonic constants

        :param float | np.ndarray[float] lat: Latitude at which the tide will be estimated.
        Pass lat = None if you have the harmonic constants

        :param float | np.ndarray[float] amp: Harmonic amplitudes (in meters), defaults to None

        :param float | np.ndarray[float] ph: Harmonic phases (in degrees), defaults to None

        :param list[str] c: List of VALID harmonic constant names, defaults to None

        :param bool nearest: Select the nearest valid ocean point from the global model, defaults to True


        :return np.ndarray[float]: Estimated tide
        """

        if any([amp is None, ph is None, c is None]):
            # Retrieve the nearest valid point
            if nearest:
                lon, lat = self.get_model_nearest_point(lon, lat)
            
            # Extract constituents from tide model
            amp, ph, c = self.Model.extract_constants(lon, lat, **{'extrapolate' : True})
            

        # Convert datetime into datenum
        time_tmd = self.datetime2datetnum(time)

        # Estimate the complexe phase and constituent oscillation
        cph = -1j * ph * np.pi / 180.0
        hc = amp * np.exp(cph)

        # Predict tidal elevation
        TIDE = pyTMD.predict.time_series(time_tmd, hc, c)
        MINOR = pyTMD.predict.infer_minor(time_tmd, hc, c)
        TIDE.data[:] += MINOR
        arr = TIDE
        return arr.data
    
    def get_model_nearest_point(self, lon:float, lat:float)->tuple[float, float]:
        """
        Quick way to retrieve the closest valid ocean point from the global tide model

        :param float lon: Longitude at which the tide will be estimated.
        Pass lon = None if you have the harmonic constants

        :param float lat: Latitude at which the tide will be estimated.
        Pass lat = None if you have the harmonic constants

        :return tuple[float, float]: Nearest valid longitude and latitude
        """
        def planar_distance(xp, yp, x, y):
            distance = np.sqrt((xp - x)**2 + (yp - y)**2)
            return distance

        mlon = self.mlon[self.modelMsk]
        mlat = self.mlat[self.modelMsk]

        if lon < 0:
            lon += 360

        distance = planar_distance(lon, lat, mlon, mlat)
        idx = np.nanargmin(distance)

        xout, yout = mlon[idx], mlat[idx]
        return xout, yout
    
    
    




