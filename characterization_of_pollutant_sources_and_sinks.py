# Developed by Saeed Farhoodi et al. (Ph.D. Student in Architectural Engineering at Illinois Tech, Chicago, IL)
# Updated on: July 26, 2025
# Contact: Saeed Farhoodi (Email: sfarhoodi@hawk.iit.edu)
# Copyright (c): The Built Environment Research Group (BERG). Webpage: https://built-envi.com/
# This code is licensed for personal or academic use only.
# Redistribution, modification, or commercial use requires prior written permission.


#..........%%%%%.........Project Description.........%%%%%..........#
# // This code is part of a research initiative under the Built Environment Research Group (BERG) lab
# // focused on HUD project. The code encompasses data processing, analysis, and visualization techniques 
# // for assessing PAC efficacy and indoor air quality in real-world settings


#..........%%%%%.........Highlights.........%%%%%..........#
# - Automates the identification of prominent peaks and their corresponding decay events
# - Detects background periods with minimal indoor source influence
# - Characterizes indoor source strengths using time-series indoor PM data


#..........%%%%%.........Code Description.........%%%%%..........#
# // The code follows a functional programming paradigm and is organized into two main sections:
# // 1) Settings, 2) Functions, and 3) Body of the code

# // In the setting section, the setting and constants used for data organization and processing are defined
# // and also different homes are labeled based on their category
# // In the "Functions" section, different functions are defined to handle data processing and perform specific calculations for
# // the purpose of a) detecting prominent peaks, b)identifying steady-state background periods, and c) characterizing decay events
# // The main body of the code logically calls these functions in sequence.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.dates as mdates
import matplotlib.lines as mlines
import os
from scipy.signal import find_peaks
from scipy.stats import linregress
from pytz import timezone
import seaborn as sns
import shutil
from scipy.stats import kstest, norm
import pickle

#..........%%%%%.........Data Organization and Processing.........%%%%%..........#

#..........%%%%%.........Settings.........%%%%%..........#

# // In this section, the constants and home labels are defined, which will be used later in the code within the functions.
# // These parameters are meaningful when the "Functions" and the "Body" of the code are carefully reviewed.

TIMEZONE = 'America/Chicago'
tz = timezone(TIMEZONE)

START_DATE = pd.to_datetime('2023-05-24 8:00:00').tz_localize(tz)
END_DATE = pd.to_datetime('2023-05-24 15:00:00').tz_localize(tz)

ENABLE_DATE_FILTER = False
ENABLE_EXPORT_CSV_PA = False
ENABLE_EXPORT_STATS = True
ENABLE_PLOT_RESULT = False

DECAY_PLOT_DEF = False
DECAY_PLOT_SAVE = False
DECAY_PLOT_SHOW = False

PROMINENCE_SHOW = False

RESAMPLING_INTERVAL = '2T'  # resampling interval to be used for PLL PA data resampling
TIME_INTERVAL = 2  # PA data resolution
GRADIENT_THRESHOLD = 1.8  # gradient threshold during steady-state background periods (can be adjusted based on variabilty across the dataset)
SS_SAMPLE_THRESHOLD = 12  # minimum samples of data during steady-state background periods
MIN_HEIGHT = 20  # minimum peak height to be used in the find_peaks function
MIN_DISTANCE = 1  # minimum peak distance to be used in the find_peaks function
MIN_DEFAULT_PROMINENCE = 10  # minimum default peak prominence
MIN_CUSTOM_PROMINENCE = 20  # minimum custom peak prominence
R_SQUARED_THRESHOLD = 0.9  # R2 threshold to exclude decay events that does not follow the psedo inverse kinetic
SAFETY_FACTOR = 4  # safety factor considered to capture fluctuations over steady-state background periods
PEAK_VALLEY_MIN_SAMPLES = 10  # minimum number of samples during each decay event
MIN_TO_DAY_CONST = 1440  # (60*24) convert day to minute
CHANNELS_SAMPLES_THRESHOLD = 100

CADR_OFF = 0 # in CFM
CADR_LOW = 50 # in CFM
CADR_MEDIUM = 80 # in CFM
CADR_HIGH = 160 # in CFM

PL_GAP_THRESHOLD = pd.Timedelta(minutes=10)  # threshold to be used for PLL gap detection post resampling
PA_GAP_THRESHOLD = pd.Timedelta(minutes=25)  # to be used for PurpleAir (PA) gap detection post resampling

MIN_TIME_DELTA = 5
MAX_TIME_DELTA = 60
MIN_Y = 0.25
MAX_Y = 4

BINS = [0, 1, 42, 57, 80, 118]  # PAC active power ranges for different modes
LABELS = ['off', 'error', 'low', 'medium', 'high']

# ambiguous periods resulting from DST
AMBIGUOUS_PERIODS = {
    2023: (pd.to_datetime('2023-11-05 00:59:00').tz_localize(tz, ambiguous='NaT'),
           pd.to_datetime('2023-11-05 02:01:00').tz_localize(tz, ambiguous='NaT')),
    2024: (pd.to_datetime('2024-11-03 00:59:00').tz_localize(tz, ambiguous='NaT'),
           pd.to_datetime('2024-11-03 02:01:00').tz_localize(tz, ambiguous='NaT'))
}

DST_PERIODS = [
    (pd.Timestamp('2023-03-12', tz='UTC'), pd.Timestamp('2023-11-05', tz='UTC')),
    (pd.Timestamp('2024-03-10', tz='UTC'), pd.Timestamp('2024-11-03', tz='UTC'))
]

plt.rcParams.update({'font.size': 55})
plt.rcParams.update({'font.family': 'Times New Roman'})

home_categories = {
    'HUD001': 'active-smoker'
    }

housing_type = {
    'HUD001': 'apartment'
    }


#..........%%%%%.........Functions.........%%%%%..........#

# // This function processes and organizes plug load logger data and performs the following tasks:
# // 1: Organize the data in the Chicago accounting for daylight saving time adjustments.
# // 2: Resampling the data with forward filling 
# // 3. Identifying missing time periods
    
def process_pl_data(directory_pl, pl_csv_file):
    
    file_path = os.path.join(directory_pl, pl_csv_file)
    
    try:
        df_PL = pd.read_csv(file_path, encoding='utf-8-sig', skiprows=1, low_memory=False)
    except UnicodeDecodeError:
        df_PL = pd.read_csv(file_path, encoding='ISO-8859-1', skiprows=1, low_memory=False)
    
    df_PL.columns.values[1] = "date_time"
    df_PL.columns.values[4] = "active_power(W)"
    df_PL = df_PL[['date_time', 'active_power(W)']]
    
    df_PL['date_time'] = pd.to_datetime(df_PL['date_time'], errors='coerce').dt.tz_localize('UTC')
    
    start_time = df_PL['date_time'].min()
    shift_hours = 5 if any(start_time >= start and start_time <= end for start, end in DST_PERIODS) else 6
    df_PL['date_time'] = df_PL['date_time'] + np.timedelta64(shift_hours, 'h')
    df_PL['date_time'] = df_PL['date_time'].dt.tz_convert(TIMEZONE)
    
    for start_ambiguous, end_ambiguous in AMBIGUOUS_PERIODS.values():
        df_PL = df_PL[~((df_PL['date_time'] >= start_ambiguous) & (df_PL['date_time'] < end_ambiguous))]
    
    if ENABLE_DATE_FILTER:
        df_PL = df_PL[(df_PL['date_time'] >= START_DATE) & (df_PL['date_time'] <= END_DATE)]
    
    df_PL = df_PL.dropna(subset=['date_time', 'active_power(W)'])
    
    df_PL.set_index('date_time', inplace=True)
    
    gaps_pl = df_PL.index.to_series().diff() > PL_GAP_THRESHOLD
    
    gap_periods_pl = []
    for i in range(len(gaps_pl)):
        if gaps_pl.iloc[i]:
            gap_periods_pl.append((df_PL.index[i-1], df_PL.index[i]))
    
    df_PL = df_PL.resample(RESAMPLING_INTERVAL).mean().ffill()
    
    for start, end in gap_periods_pl:
        df_PL = df_PL[~((df_PL.index > start) & (df_PL.index < end))]
    
    df_PL['pl_mode'] = pd.cut(df_PL['active_power(W)'], bins=BINS, labels=LABELS, right=False, include_lowest=True)
    
    return df_PL


# // This function processes and organizes Purple Air data and performs the following tasks:
# // 1: Organize the data in the Chicago accounting for daylight saving time adjustments.
# // 2: Resampling the data with forward filling 
# // 3. Identifying missing time periods

def process_pa_data(directory_pa, column_mapping):
    
    df = pd.DataFrame()
    df_PA = pd.DataFrame()
    
    csv_files = [os.path.join(directory_pa, f) for f in os.listdir(directory_pa) if f.endswith('.csv')]
    
    for file_path in csv_files:
        try:
            df_temporary = pd.read_csv(file_path, encoding='utf-8-sig')
        except UnicodeDecodeError:
            df_temporary = pd.read_csv(file_path, encoding='ISO-8859-1')
        #print(file_path)
        #print(df_temporary.shape[1])
        df = pd.concat([df, df_temporary], ignore_index=True, sort=False)
    
    df = df.drop(['mac_address', 'firmware_ver', 'hardware', 'current_dewpoint_f', 'adc', 'mem', 'rssi',
                  'uptime', 'pm2.5_aqi_cf_1', 'pm2.5_aqi_atm', 'pm2.5_aqi_cf_1_b', 'pm2.5_aqi_atm_b', 'gas'], axis=1)    
    
    df_PA['date_time'] = pd.to_datetime(df.iloc[:, column_mapping['date_time']], errors='coerce')
    
    if df_PA['date_time'].dt.tz is None:
        df_PA['date_time'] = df_PA['date_time'].dt.tz_localize('UTC')
        
    df_PA['date_time'] = df_PA['date_time'].dt.tz_convert(TIMEZONE)
    
    for column_name, column_idx in column_mapping.items():
        if column_name != 'date_time':
            df_PA[column_name] = pd.to_numeric(df.iloc[:, column_idx], errors='coerce')
    
    df_PA['pm1.0_cf_1'] = (df_PA['pm1.0_cf_1_a'] + df_PA['pm1.0_cf_1_b'])/2
    df_PA['pm2.5_cf_1'] = (df_PA['pm2.5_cf_1_a'] + df_PA['pm2.5_cf_1_b'])/2
    df_PA['pm10.0_cf_1'] = (df_PA['pm10.0_cf_1_a'] + df_PA['pm10.0_cf_1_b'])/2
    df_PA['pm1.0_atm'] = (df_PA['pm1.0_atm_a'] + df_PA['pm1.0_atm_b'])/2
    df_PA['pm2.5_atm'] = (df_PA['pm2.5_atm_a'] + df_PA['pm2.5_atm_b'])/2
    df_PA['pm10.0_atm'] = (df_PA['pm10.0_atm_a'] + df_PA['pm10.0_atm_b'])/2
    df_PA['0.3_um_count'] = (df_PA['0.3_um_count_a'] + df_PA['0.3_um_count_b'])/2
    df_PA['0.5_um_count'] = (df_PA['0.5_um_count_a'] + df_PA['0.5_um_count_b'])/2
    df_PA['1.0_um_count'] = (df_PA['1.0_um_count_a'] + df_PA['1.0_um_count_b'])/2
    df_PA['2.5_um_count'] = (df_PA['2.5_um_count_a'] + df_PA['2.5_um_count_b'])/2
    df_PA['5.0_um_count'] = (df_PA['5.0_um_count_a'] + df_PA['5.0_um_count_b'])/2
    
    df_PA['pm2.5_alt'] = 3*(0.00030418*(df_PA['0.3_um_count'] - df_PA['0.5_um_count']) + 0.0018512 *(df_PA['0.5_um_count'] - df_PA['1.0_um_count']) + 0.02069706 * (df_PA['1.0_um_count'] - df_PA['2.5_um_count']))
    df_PA['pm2.5_alt_a'] = 3*(0.00030418*(df_PA['0.3_um_count_a'] - df_PA['0.5_um_count_a']) + 0.0018512 *(df_PA['0.5_um_count_a'] - df_PA['1.0_um_count_a']) + 0.02069706 * (df_PA['1.0_um_count_a'] - df_PA['2.5_um_count_a']))
    df_PA['pm2.5_alt_b'] = 3 *(0.00030418*(df_PA['0.3_um_count_b'] - df_PA['0.5_um_count_b']) + 0.0018512 *(df_PA['0.5_um_count_b'] - df_PA['1.0_um_count_b']) + 0.02069706 * (df_PA['1.0_um_count_b'] - df_PA['2.5_um_count_b']))
    
    #df_PA.to_csv('D:\OneDrive - Illinois Institute of Technology\PhD\Research\HUD Project-Indoor Air Quality\Analysis\HUD - Saeed\output.csv')
    
    df_PA = df_PA.dropna(subset=['date_time', 'pm2.5_alt'])
    
    df_PA.set_index('date_time', inplace=True)
    
    gaps_pa = df_PA.index.to_series().diff() > PA_GAP_THRESHOLD
    
    gap_periods_pa = []
    for i in range(len(gaps_pa)):
        if gaps_pa.iloc[i]:
            gap_periods_pa.append((df_PA.index[i-1], df_PA.index[i]))
    
    df_PA = df_PA.resample(RESAMPLING_INTERVAL).mean().ffill()
    
    for start, end in gap_periods_pa:
        df_PA = df_PA[~((df_PA.index > start) & (df_PA.index < end))]
    
    return df_PA


# // This function combines PL and PA data:

def concat_pl_pa_data(df_PA, df_PL):
    
    df_PA = pd.concat([df_PA, df_PL], axis=1)
    df_PA.reset_index(inplace=True)
    
    df_PA = df_PA.dropna(subset=['date_time', 'pm2.5_alt', 'active_power(W)', 'pl_mode'])
    
    for start_ambiguous, end_ambiguous in AMBIGUOUS_PERIODS.values():
        df_PA = df_PA[~((df_PA['date_time'] >= start_ambiguous) & (df_PA['date_time'] < end_ambiguous))]
    
    df_PA.reset_index(drop=True, inplace=True)
    
    return df_PA



# // This function identifies steady-state periods, which are essential for calculating loss rates.
# // It has two main sections: 
# //   1) Detecting primary steay-state regions utilizing a fixed gradient threshold, 
# //   2) Revising them using dynamic gradient thresholds

def detect_steady_state_periods(df_PA):
    
    adjusted_gradients = {}
    
    df_PA['gradient'] = df_PA['pm2.5_alt'].diff().abs()
    steady_state = (df_PA['gradient'] < GRADIENT_THRESHOLD).astype(int)
    df_PA['steady_state'] = steady_state.groupby((steady_state != steady_state.shift()).cumsum()).transform('size') * steady_state
    df_PA['steady_state'] = df_PA['steady_state'].apply(lambda x: x if x >= SS_SAMPLE_THRESHOLD else 0)
    df_PA['steady_state_group'] = (df_PA['steady_state'].diff(1) != 0).cumsum()
    steady_state_groups = df_PA[df_PA['steady_state'] > 0]['steady_state_group'].unique()
    
    adjusted_gradients = {group: SAFETY_FACTOR * df_PA[df_PA['steady_state_group'] == group]['gradient'].quantile(0.75) for group in steady_state_groups}
    
    df_PA['adjusted_gradient_threshold'] = df_PA['steady_state_group'].map(adjusted_gradients).fillna(GRADIENT_THRESHOLD)
    df_PA['adjusted_gradient'] = df_PA.apply(lambda row: row['gradient'] < row['adjusted_gradient_threshold'], axis=1).astype(int)
    df_PA['adjusted_steady_state'] = df_PA['adjusted_gradient'].groupby((df_PA['adjusted_gradient'] != df_PA['adjusted_gradient'].shift()).cumsum()).transform('size') * df_PA['adjusted_gradient']
    df_PA['adjusted_steady_state'] = df_PA['adjusted_steady_state'].apply(lambda x: x if x >= SS_SAMPLE_THRESHOLD else 0)
    df_PA['adjusted_steady_state_group'] = (df_PA['adjusted_steady_state'].diff(1) != 0).cumsum()
    
    return df_PA


# // This function identifies prominent peaks and their corresponding valleys.

def capture_peaks_and_valleys(df_PA, home_name):
    
    # step 1: detect peaks without prominence filtering
    primary2_peaks, _ = find_peaks(df_PA['pm2.5_alt'], height=MIN_HEIGHT, distance=MIN_DISTANCE, prominence=MIN_DEFAULT_PROMINENCE)
    
    # step 2: calculate custom prominence for all detected peaks
    custom_prominences = []
    for i, peak in enumerate(primary2_peaks):
        # Find reference valleys
        left_valley = df_PA['pm2.5_alt'][:peak].min() if i == 0 else df_PA['pm2.5_alt'][primary2_peaks[i - 1]:peak].min()
        right_valley = df_PA['pm2.5_alt'][peak:].min() if i == len(primary2_peaks) - 1 else df_PA['pm2.5_alt'][peak:primary2_peaks[i + 1]].min()
        
        # Reference valley is the higher of the two valleys
        reference_valley = max(left_valley, right_valley)
        
        # Calculate prominence
        custom_local_prominence = df_PA['pm2.5_alt'][peak] - reference_valley
        custom_prominences.append(custom_local_prominence)
    
    # step 3: filter peaks based on custom prominence
    peaks = [peak for peak, prominence in zip(primary2_peaks, custom_prominences) if prominence > MIN_CUSTOM_PROMINENCE]
    prominences = [prominence for prominence in custom_prominences if prominence > MIN_CUSTOM_PROMINENCE]
    
    
    valleys = []
    
    for i in range(len(peaks) - 1):
        start, end = peaks[i], peaks[i + 1]
        valley = df_PA['pm2.5_alt'][start:end].idxmin()
        valleys.append(valley)
    
    #print(f'length all peaks: {len(peaks)}')
    #print(f'length all valleys: {len(valleys)}')
    #print(peaks)
    
    # Store peak data
    peak_data = []
    for peak, prom in zip(peaks, prominences):
        peak_data.append({
            'home_ID': home_name,
            'peak_index': peak,
            'peak_value': df_PA['pm2.5_alt'][peak],
        })
    
    return peaks, valleys, prominences, peak_data



# // This function filters peaks and valleys to select the valid ones qualified for loss rate calculation.

def filter_peaks_and_valleys_and_loss_calculation(df_PA, peaks, valleys):
    
    valid_peaks = []
    valid_valleys = []
    decay_data = []
    valid_Cbgs_mean = []
    valid_loss_rates = []
    
    for peak_index, valley_index in zip(peaks, valleys):
        if df_PA.loc[valley_index, 'adjusted_steady_state'] == 0:
            continue
        
        current_ss_group_index = df_PA.loc[valley_index, 'adjusted_steady_state_group']
        first_ss_cell = df_PA[df_PA['adjusted_steady_state_group'] == current_ss_group_index].index[0]   # first point of steady-state region
        #first_ss_cell = df_PA[df_PA['adjusted_steady_state_group'] == current_ss_group_index]['pm2.5_alt'].idxmin()   # the primary valley index
        ss_groups_between = df_PA.loc[peak_index:valley_index, 'adjusted_steady_state_group'].unique()
        
        if len(ss_groups_between) > 3:
            continue
        
        peak_value = df_PA.loc[peak_index, 'pm2.5_alt']
        valley_value = df_PA.loc[first_ss_cell, 'pm2.5_alt']
        
        if peak_value < MIN_HEIGHT:
            continue
        
        peak_time = df_PA.loc[peak_index, 'date_time']
        C0 = df_PA.loc[peak_index, 'pm2.5_alt']
        pl_mode = df_PA.loc[peak_index, 'pl_mode']
        current_ss_group = df_PA.loc[df_PA['adjusted_steady_state_group'] == current_ss_group_index, 'pm2.5_alt']
        Cbg_mean = current_ss_group.mean()
        Cbg_std = current_ss_group.std()
        
        times = []
        y_values = []
        
        for t in range(peak_index, first_ss_cell + 1):
            Ct = df_PA.loc[t, 'pm2.5_alt']
            #y = -np.log((Ct - Cbg) / (C0 - Cbg))
            y = -np.log(np.clip((Ct - Cbg_mean) / (C0 - Cbg_mean), 1e-10, None))  #Ensure that the input to the natural log function is valid
            time_delta = (df_PA.loc[t, 'date_time'] - peak_time).total_seconds() / 60
            
            if not np.isnan(y):
                #if MIN_Y <= y <= MAX_Y:  # The following condition can be based on y instead of time_delta
                if MIN_TIME_DELTA <= time_delta <= MAX_TIME_DELTA:
                    times.append(time_delta)
                    y_values.append(y)
        
        if len(times) < PEAK_VALLEY_MIN_SAMPLES:
            continue
        
        slope, intercept, r_value, p_value, std_err = linregress(times, y_values)
        if r_value**2 < R_SQUARED_THRESHOLD:
            continue
        
        valid_peaks.append(peak_index)
        valid_valleys.append(first_ss_cell)
        valid_Cbgs_mean.append(Cbg_mean)
        valid_loss_rates.append(slope)
        
        
        #.....Uncertainty Calculation.....
        # Uncertainty contribution from Cbg
        uncertainty_Cbg = Cbg_std / Cbg_mean if Cbg_mean != 0 else 0  # In other word, relative standard deviation of Cbg
        # Uncertainty contribution from regression std. error
        uncertainty_err = std_err / slope
        # Total uncertainty
        total_uncertainty = ( uncertainty_Cbg**2 + uncertainty_err**2 )**0.5
        
        
        decay_data.append({
            'peak_time': peak_time,
            'peak_value': peak_value,
            'slope': slope,
            'r_squared_linear': r_value**2,
            'intercept': intercept,
            'pl_mode': pl_mode,
            'Cbg_mean': Cbg_mean,
            'Cbg_std': Cbg_std,
            'valley_value': valley_value,
            'gradient_threshold': df_PA.loc[valley_index, 'adjusted_gradient_threshold'],
            'times': times,
            'y_values': y_values,
            'std_err': std_err,
            'uncertainty_Cbg': uncertainty_Cbg,
            'uncertainty_err': uncertainty_err,
            'total_uncertainty': total_uncertainty
        })
        
    return valid_peaks, valid_valleys, valid_Cbgs_mean, valid_loss_rates, decay_data



def export_decay_statistics(decay_data):
    
    decay_numbers = list(range(1, len(decay_data) + 1))
    peak_times = [data['peak_time'] for data in decay_data]
    peak_values = [data['peak_value'] for data in decay_data]
    slopes = [data['slope'] for data in decay_data]
    r_squared_values = [data['r_squared_linear'] for data in decay_data]
    intercepts = [data['intercept'] for data in decay_data]
    pl_modes = [data['pl_mode'] for data in decay_data]
    Cbgs_mean = [data['Cbg_mean'] for data in decay_data]
    Cbgs_std = [data['Cbg_std'] for data in decay_data]
    valley_values = [data['valley_value'] for data in decay_data]
    gradient_thresholds = [data['gradient_threshold'] for data in decay_data]
    std_errs = [data['std_err'] for data in decay_data]
    uncertainty_Cbgs = [data['uncertainty_Cbg'] for data in decay_data]
    uncertainty_errs = [data['uncertainty_err'] for data in decay_data]
    total_uncertainties = [data['total_uncertainty'] for data in decay_data]
    
    
    df_decay = pd.DataFrame({
        'decay_number': decay_numbers,
        'home_ID': home_name,
        'home_category': category,
        'peak_time': peak_times,
        'peak_value': [round(peak_value, 2) for peak_value in peak_values],
        'slope': [round(slope*60, 2) for slope in slopes],
        'r_squared': [round(r_squared, 2) for r_squared in r_squared_values],
        'intercept': [round(intercept, 2) for intercept in intercepts],
        'pl_mode': pl_modes,
        'Cbg_mean': [round(Cbg_mean, 2) for Cbg_mean in Cbgs_mean],
        'Cbg_std': [round(Cbg_std, 2) for Cbg_std in Cbgs_std],
        'valley_value': [round(valley_value, 2) for valley_value in valley_values],
        'gradient_threshold': [round(gradient_threshold, 2) for gradient_threshold in gradient_thresholds],
        'std_err': [round(std_err, 2) for std_err in std_errs],
        'uncertainty_Cbg': [round(uncertainty_Cbg, 2) for uncertainty_Cbg in uncertainty_Cbgs],
        'uncertainty_err': [round(uncertainty_err, 2) for uncertainty_err in uncertainty_errs],
        'total_uncertainty': [round(total_uncertainty, 2) for total_uncertainty in total_uncertainties]
    })
    
    return df_decay



# // This function extracts some general statistics of PA data

def export_general_statistics(df_PA, peaks, valleys, valid_peaks, valid_valleys, decay_data):
    
    df_PA['rel_diff'] = np.where((df_PA['pm2.5_alt_a'] != 0) | (df_PA['pm2.5_alt_b'] != 0),
         abs(df_PA['pm2.5_alt_a'] - df_PA['pm2.5_alt_b']) / (df_PA['pm2.5_alt_a'] + df_PA['pm2.5_alt_b']), np.nan)
    
    mean_rel_diff = df_PA['rel_diff'].mean()
    std_rel_diff = df_PA['rel_diff'].std()
    
    mean_total = df_PA['pm2.5_alt'].mean()
    std_total = df_PA['pm2.5_alt'].std()
    q25_total = df_PA['pm2.5_alt'].quantile(0.25)
    median_total = df_PA['pm2.5_alt'].median()
    q75_total = df_PA['pm2.5_alt'].quantile(0.75)
    
    mean_total_off = df_PA[(df_PA['pl_mode'] == 'off')]['pm2.5_alt'].mean()
    std_total_off = df_PA[(df_PA['pl_mode'] == 'off')]['pm2.5_alt'].std()
    mean_total_low = df_PA[(df_PA['pl_mode'] == 'low')]['pm2.5_alt'].mean()
    std_total_low = df_PA[(df_PA['pl_mode'] == 'low')]['pm2.5_alt'].std()
    mean_total_medium = df_PA[(df_PA['pl_mode'] == 'medium')]['pm2.5_alt'].mean()
    std_total_medium = df_PA[(df_PA['pl_mode'] == 'medium')]['pm2.5_alt'].std()
    mean_total_high = df_PA[(df_PA['pl_mode'] == 'high')]['pm2.5_alt'].mean()
    std_total_high = df_PA[(df_PA['pl_mode'] == 'high')]['pm2.5_alt'].std()
    
    
    mean_ss = df_PA[df_PA['adjusted_steady_state'] > 0]['pm2.5_alt'].mean()
    std_ss = df_PA[df_PA['adjusted_steady_state'] > 0]['pm2.5_alt'].std()
    q25_ss = df_PA[df_PA['adjusted_steady_state'] > 0]['pm2.5_alt'].quantile(0.25)
    median_ss = df_PA[df_PA['adjusted_steady_state'] > 0]['pm2.5_alt'].median()
    q75_ss = df_PA[df_PA['adjusted_steady_state'] > 0]['pm2.5_alt'].quantile(0.75)
    
    mean_ss_off = df_PA[(df_PA['adjusted_steady_state'] > 0) & (df_PA['pl_mode'] == 'off')]['pm2.5_alt'].mean()
    std_ss_off = df_PA[(df_PA['adjusted_steady_state'] > 0) & (df_PA['pl_mode'] == 'off')]['pm2.5_alt'].std()
    mean_ss_low = df_PA[(df_PA['adjusted_steady_state'] > 0) & (df_PA['pl_mode'] == 'low')]['pm2.5_alt'].mean()
    std_ss_low = df_PA[(df_PA['adjusted_steady_state'] > 0) & (df_PA['pl_mode'] == 'low')]['pm2.5_alt'].std()
    mean_ss_medium = df_PA[(df_PA['adjusted_steady_state'] > 0) & (df_PA['pl_mode'] == 'medium')]['pm2.5_alt'].mean()
    std_ss_medium = df_PA[(df_PA['adjusted_steady_state'] > 0) & (df_PA['pl_mode'] == 'medium')]['pm2.5_alt'].std()
    mean_ss_high = df_PA[(df_PA['adjusted_steady_state'] > 0) & (df_PA['pl_mode'] == 'high')]['pm2.5_alt'].mean()
    std_ss_high = df_PA[(df_PA['adjusted_steady_state'] > 0) & (df_PA['pl_mode'] == 'high')]['pm2.5_alt'].std()
    
    mean_peaks = df_PA.loc[peaks, 'pm2.5_alt'].mean()
    num_peaks = len(peaks)
    num_valid_peaks = len(valid_peaks)
    
    total_day = (df_PA.shape[0] * TIME_INTERVAL) / MIN_TO_DAY_CONST
    total_day_ss = (df_PA[df_PA['adjusted_steady_state'] > 0].shape[0] * TIME_INTERVAL) / MIN_TO_DAY_CONST
    off_day = (df_PA[df_PA['pl_mode'] == 'off'].shape[0] * TIME_INTERVAL) / MIN_TO_DAY_CONST
    low_day = (df_PA[df_PA['pl_mode'] == 'low'].shape[0] * TIME_INTERVAL) / MIN_TO_DAY_CONST
    medium_day = (df_PA[df_PA['pl_mode'] == 'medium'].shape[0] * TIME_INTERVAL) / MIN_TO_DAY_CONST
    high_day = (df_PA[df_PA['pl_mode'] == 'high'].shape[0] * TIME_INTERVAL) / MIN_TO_DAY_CONST
    error_day = total_day - (off_day + low_day + medium_day + high_day)
    
    total_day = total_day - error_day
    
    off_percentage = (off_day / total_day) * 100
    low_percentage = (low_day / total_day) * 100
    medium_percentage = (medium_day / total_day) * 100
    high_percentage = (high_day / total_day) * 100
    
    cadr = (off_day*CADR_OFF + low_day*CADR_LOW + medium_day*CADR_MEDIUM + high_day*CADR_HIGH) / total_day
    cadrv = ( ((off_day*CADR_OFF + low_day*CADR_LOW + medium_day*CADR_MEDIUM + high_day*CADR_HIGH) / total_day) / volume ) * 60
    
    
    df_general_statistics = pd.DataFrame({
        'home_ID': home_name,
        'home_category': category,
        
        'rel_diff_mean': [round(mean_rel_diff, 2)],
        'rel_diff_std': [round(std_rel_diff, 2)],
        
        'pm2.5_mean(allmodes)': [round(mean_total, 2)],
        'pm2.5_std(allmodes)': [round(std_total, 2)],
        'pm2.5_q25(allmodes)': [round(q25_total, 2)],
        'pm2.5_median(allmodes)': [round(median_total, 2)],
        'pm2.5_q75(allmodes)': [round(q75_total, 2)],
        
        'pm2.5_mean(off)': [round(mean_total_off, 2)],
        'pm2.5_std(off)': [round(std_total_off, 2)],
        'pm2.5_mean(low)': [round(mean_total_low, 2)],
        'pm2.5_std(low)': [round(std_total_low, 2)],
        'pm2.5_mean(medium)': [round(mean_total_medium, 2)],
        'pm2.5_std(medium)': [round(std_total_medium, 2)],
        'pm2.5_mean(high)': [round(mean_total_high, 2)],
        'pm2.5_std(high)': [round(std_total_high, 2)],
        
        
        'pm2.5_ss_mean(allmodes)': [round(mean_ss, 2)],
        'pm2.5_ss_std(allmodes)': [round(std_ss, 2)],
        'pm2.5_ss_q25(allmodes)': [round(q25_ss, 2)],
        'pm2.5_ss_median(allmodes)': [round(median_ss, 2)],
        'pm2.5_ss_q75(allmodes)': [round(q75_ss, 2)],
        
        'pm2.5_ss_mean(off)': [round(mean_ss_off, 2)],
        'pm2.5_ss_std(off)': [round(std_ss_off, 2)],
        'pm2.5_ss_mean(low)': [round(mean_ss_low, 2)],
        'pm2.5_ss_std(low)': [round(std_ss_low, 2)],
        'pm2.5_ss_mean(medium)': [round(mean_ss_medium, 2)],
        'pm2.5_ss_std(medium)': [round(std_ss_medium, 2)],
        'pm2.5_ss_mean(high)': [round(mean_ss_high, 2)],
        'pm2.5_ss_std(high)': [round(std_ss_high, 2)],
        
        'pm2.5_mean(peaks)': [round(mean_peaks, 2)],
        'num_detected_peaks': num_peaks,
        'num_valid_peaks': num_valid_peaks,
        
        'total_duration(day)': [round(total_day, 2)],
        'total_duration_ss(day)': [round(total_day_ss, 2)],
        'off_duration(day)': [round(off_day, 2)],
        'low_duration(day)': [round(low_day, 2)],
        'medium_duration(day)': [round(medium_day, 2)],
        'high_duration(day)': [round(high_day, 2)],
        'error(day)': [round(error_day, 2)],
        
        'off_duration(%)': [round(off_percentage, 2)],
        'low_duration(%)': [round(low_percentage, 2)],
        'medium_duration(%)': [round(medium_percentage, 2)],
        'high_duration(%)': [round(high_percentage, 2)],
        
        'cadr(cfm)': [round(cadr, 2)],
        'cadr/v(1/h)': [round(cadrv, 2)],
        
    })
    
    return df_PA, df_general_statistics



#..........%%%%%.........Body of the Code.........%%%%%..........#

# // The body of the code logically calls the functions.

df_decay_stats_all_homes = pd.DataFrame()
df_emission_stats_all_homes = pd.DataFrame()
df_general_stats_all_homes = pd.DataFrame()
df_PA_dict = {}
all_peak_data = []

df_volume = pd.read_excel(os.path.join(os.getcwd(), 'Home_Volume.xlsx'))

directory_pl = os.path.join(os.getcwd(), 'PL Data')
pl_csv_files = [f for f in os.listdir(directory_pl) if f.endswith('.csv')]

for idx, pl_csv_file in enumerate(pl_csv_files):
    
    home_name = f'HUD{pl_csv_file[2:5]}'
    print(home_name)
    
    category = home_categories.get(home_name, 'unknown')
    
    volume = df_volume.loc[df_volume['home_name'] == home_name, 'volume(ft3)'].values[0]
    
    df_PL = process_pl_data(directory_pl, pl_csv_file)
    
    pa_folder_name = f'PA{pl_csv_file[2:5]}'
    directory_pa = os.path.join(os.getcwd(), 'PA Data', pa_folder_name)
    column_mapping = {'date_time': 0,
                      'temperaure(F)': 1,
                      'rh(%)': 2,
                      'pressure': 3,
                      
                      'pm1.0_cf_1_a': 4,
                      'pm2.5_cf_1_a': 5,
                      'pm10.0_cf_1_a': 6,
                      'pm1.0_atm_a': 7,
                      'pm2.5_atm_a': 8,
                      'pm10.0_atm_a': 9,
                      '0.3_um_count_a': 10,
                      '0.5_um_count_a': 11,
                      '1.0_um_count_a': 12,
                      '2.5_um_count_a': 13,
                      '5.0_um_count_a': 14,
                      '10.0_um_count_a': 15,
                      
                      'pm1.0_cf_1_b': 16,
                      'pm2.5_cf_1_b': 17,
                      'pm10.0_cf_1_b': 18,
                      'pm1.0_atm_b': 19,
                      'pm2.5_atm_b': 20,
                      'pm10.0_atm_b': 21,
                      '0.3_um_count_b': 22,
                      '0.5_um_count_b': 23,
                      '1.0_um_count_b': 24,
                      '2.5_um_count_b': 25,
                      '5.0_um_count_b': 26,
                      '10.0_um_count_b': 27,
                      }
    df_PA = process_pa_data(directory_pa, column_mapping=column_mapping)
    
    df_PA = concat_pl_pa_data(df_PA, df_PL)
    
    df_PA = detect_steady_state_periods(df_PA)
    
    peaks, valleys, prominences, peak_data = capture_peaks_and_valleys(df_PA, home_name)
    all_peak_data.extend(peak_data)
    
    valid_peaks, valid_valleys, valid_Cbgs_mean, valid_loss_rates, decay_data = filter_peaks_and_valleys_and_loss_calculation(df_PA, peaks, valleys)
    
    
    df_decay = export_decay_statistics(decay_data)
    df_decay_stats_all_homes = pd.concat([df_decay_stats_all_homes, df_decay], ignore_index=True)
    
    df_PA, df_general_statistics = export_general_statistics(df_PA, peaks, valleys, valid_peaks, valid_valleys, decay_data)
    df_general_stats_all_homes = pd.concat([df_general_stats_all_homes, df_general_statistics], ignore_index=True)
    
    if ENABLE_EXPORT_CSV_PA:
        df_PA.to_csv(os.path.join(os.getcwd(), 'Results', 'Combined Data', f'{home_name}.csv'), index=False)
    
    df_PA_dict[home_name] = df_PA


df_peak_data = pd.DataFrame(all_peak_data)

save_path = os.path.join(os.getcwd(), 'Results', 'Processed_Data')
os.makedirs(save_path, exist_ok=True)

# Save all the data in one pickle file
with open(os.path.join(save_path, 'processed_data.pkl'), 'wb') as f:
    pickle.dump({
        'df_decay_stats_all_homes': df_decay_stats_all_homes,
        'df_emission_stats_all_homes': df_emission_stats_all_homes,
        'df_general_stats_all_homes': df_general_stats_all_homes,
        'df_PA_dict': df_PA_dict,
        'df_peak_data': df_peak_data,
    }, f)

