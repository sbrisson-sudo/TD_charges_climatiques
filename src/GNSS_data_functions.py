import numpy as np
import pandas as pd

from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d

from datetime import datetime,timedelta

import os 

def butter_lowpass_filter(time, data, cut_period, order=4):
    
    # 1. resample
    time_reg = np.linspace(time.min(), time.max(), 1000)
    interp = interp1d(time, data, kind='linear', fill_value='extrapolate')
    data_reg = interp(time_reg)
    
    # 2. filter
    sampling_rate = 1./(time_reg[1]-time_reg[0])
    nyquist = 0.5 * sampling_rate
    cut_frequency = 1.0 / cut_period
    normal_cutoff = cut_frequency / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data_reg = filtfilt(b, a, data_reg)
    
    # 3. interpolate back on original data point
    interp = interp1d(time_reg, filtered_data_reg, kind='linear', fill_value='extrapolate')
    filtered_data = interp(time)
    
    return filtered_data

def decimal_year_to_date(decimal_years):
    
    # Extract the years and fractions of years
    years = np.floor(decimal_years)
    fractions_of_year = decimal_years - years

    # Calculate the days in the year and day of the year
    days_in_year = 365  # Adjust for leap years if needed
    days_of_year = (fractions_of_year * days_in_year).astype(int)

    # Create datetime objects using vectorized operations
    date_objects = np.array([datetime(int(year), 1, 1) + timedelta(days=int(day) - 1) for year, day in zip(years, days_of_year)])
    
    return date_objects

def load_GNSS_tenv_data(data_dir):

    gnss_df = None
    stations = {}

    list_files = os.listdir(data_dir)

    for file in list_files:
                
        if not(file[-5:] == "tenv3"): continue
        
        # Load GNSS data
        gnss_df_tmp = pd.read_fwf(os.path.join(data_dir,file), infer_nrows=10)

        # Extraction site, longitude de réference, latitude et longitude de la station
        site, lon_ref, lat, lon = gnss_df_tmp.iloc[0][["site","reflon","_latitude(deg)","_longitude(deg)"]]
        stations[site] = (lat,lon)

        # Somme des colonnes déplacements
        for col_sum, (col_int,col_dec) in zip(["east_orig","north_orig","up_orig"], [('_e0(m)', '__east(m)'), ('____n0(m)', '_north(m)'), ('u0(m)', '____up(m)')]):
            # gnss_df_tmp[col_sum] = gnss_df_tmp[col_int] + gnss_df_tmp[col_dec]
            gnss_df_tmp[col_sum] = gnss_df_tmp[col_dec]
            
        # Construction colonne datetime
        gnss_df_tmp["Date"] = decimal_year_to_date(gnss_df_tmp["yyyy.yyyy"])

        # Conserver uniquement colonnes intéressantes : 
        gnss_df_tmp = gnss_df_tmp[["site","Date","yyyy.yyyy","east_orig","north_orig","up_orig"]]

        # Remove linear trend
        for col1,col2 in zip(["east_orig","north_orig","up_orig"],["east_detrended","north_detrended","up_detrended"]):
            coeffs = np.polyfit(gnss_df_tmp["yyyy.yyyy"], gnss_df_tmp[col1], 1)
            fit_line = fit_line = np.poly1d(coeffs)
            gnss_df_tmp[col2] = gnss_df_tmp[col1] - fit_line(gnss_df_tmp["yyyy.yyyy"])
                
        # Remove outliers by applying low-pass filters then having a relative difference threshold
        
        cut_period = 1/6 # 2 months
        ts_abs_diff = 1.5e-2 # 1.5cm
        
        N_pts = len(gnss_df_tmp)
        
        for col1,col2,col3 in zip(["east_detrended","north_detrended","up_detrended"],["east_dt_filtered","north_dt_filtered","up_dt_filtered"],["east_dt_filter_res","north_dt_filter_res","up_dt_filter_res"]):
            # Filtering
            gnss_df_tmp[col2] = butter_lowpass_filter(gnss_df_tmp["yyyy.yyyy"], gnss_df_tmp[col1], cut_period)
            # Computing relative residuals
            gnss_df_tmp[col3] = np.abs((gnss_df_tmp[col1] - gnss_df_tmp[col2]))
            # Drop data point when above threshold
            gnss_df_tmp = gnss_df_tmp[gnss_df_tmp[col3] <= ts_abs_diff]
            
        print(f"Station {site} : {N_pts - len(gnss_df_tmp)} outliers removed.")
                    
        # Append to df
        gnss_df_tmp = gnss_df_tmp.rename(columns={"site":"Station"})
        if gnss_df is None : 
            gnss_df = gnss_df_tmp[["Station","Date",'up_orig', 'north_orig', 'east_orig','up_detrended', 'north_detrended', 'east_detrended','up_dt_filtered', 'north_dt_filtered', 'east_dt_filtered']]
        else : 
            gnss_df = pd.concat([gnss_df,gnss_df_tmp[["Station","Date",'up_orig', 'north_orig', 'east_orig','up_detrended', 'north_detrended', 'east_detrended','up_dt_filtered', 'north_dt_filtered', 'east_dt_filtered']]], ignore_index=True)

    gnss_df = gnss_df.set_index(["Station","Date"])
    
    return gnss_df,stations