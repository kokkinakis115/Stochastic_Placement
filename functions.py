# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 15:59:10 2024

@author: PANAGIOTIS
"""

import numpy as np
# import pyfeng as pf
import matplotlib.pyplot as plt
import pandas as pd
import random
# import time
# import os
import statistics
# from statistics import mean
# from scipy.stats import norm
# import seaborn as sns
# from warnings import filterwarnings
import math
# import matplotlib.pyplot as plt
# import networkx as nx
from collections import deque
# from scipy.special import ndtri
from statistics import NormalDist



def create_random_scheme(simulation):
    scheme = []
    for app in simulation.applications_stack:
        array = np.random.randint(simulation.num_of_total_nodes, size=app.num_of_ms)
        scheme.append(deque(array))
    return scheme

def plot_time_series(time_series, freq):
    dates = pd.date_range(start='00:00', periods=len(time_series), freq=freq)

    data = pd.DataFrame({'date': dates, 'value': time_series})
    
    # Plot the time-series data
    plt.plot(data['date'], data['value'])
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.xticks(rotation = 45)
    plt.title("Usage Time Series")
    plt.show()
    
def produce_volatility_dict(workload_volatility_array):
    
    perc_50 = np.percentile(workload_volatility_array, 50)
    perc_90 = np.percentile(workload_volatility_array, 90)
    low_vol = []
    med_vol = []
    high_vol = []
    for index, vol in enumerate(workload_volatility_array):
        if vol < perc_50:
            low_vol.append(index)
        elif vol > perc_90:
            high_vol.append(index)
        else:
            med_vol.append(index)
    vol_dict = {'low': low_vol, 'medium': med_vol, 'high': high_vol}
    return vol_dict

def calculate_volatility(time_series):
    wl_change = [0] + [(time_series[i]-time_series[i-1])/time_series[i] for i in range(1, len(time_series))]
    time_series_volatility = statistics.stdev(wl_change)
    return time_series_volatility

def produce_workload(intensity, vol, workloads, vol_dict):
    
    intensity_dict = {'low': 1, 'medium': 5, 'high': 7}
    wl = np.zeros(len(workloads[0]))
    for i in range(intensity_dict[intensity]):
        wl += workloads[random.choice(vol_dict[vol])]
    
    return np.append(wl, wl[-1])

    
def create_wl(sigma, mu, lambd, users=100, duration_in_hours=12):

    duration = duration_in_hours*3600
    wl = []
    
    for user in range(users):
        start_time = int(random.uniform(0, duration-1)) #12 hours
        end_time = int(max(start_time+1, min(duration, start_time + random.normalvariate(mu=mu, sigma=sigma)))) #duration around 5 minutes
        norm_cpu_requested = random.expovariate(lambd = lambd)
        
        new_row = [start_time, end_time, norm_cpu_requested]
        wl.append([new_row])
    wl = np.array(wl).reshape((users,3))
    urllc_df = pd.DataFrame(wl, columns=['start time', 'end time', 'normalized cpu requested'])
    urllc_df['start time'] = urllc_df['start time'].astype(int)
    urllc_df['end time'] = urllc_df['end time'].astype(int)
    
    return urllc_df  



def compute_strike_price1(delta, texp, cp, vol, intr, spot):
    exponent1 = intr*spot*texp
    exponent2 = -cp*NormalDist().inv_cdf(cp*np.exp(exponent1)*delta)*vol*math.sqrt(texp)
    additive = 0.5*vol**2*texp
    predicted_strike = spot*np.exp(exponent2+additive)
    return predicted_strike

def compute_strike_price2(delta, texp, cp, vol, intr, spot):
    # exponent1 = intr*spot*texp
    exponent2 = -cp*NormalDist().inv_cdf(cp*delta)*vol*math.sqrt(texp)
    additive = 0.5*vol**2*texp
    predicted_strike = spot*np.exp(exponent2+additive)
    return predicted_strike