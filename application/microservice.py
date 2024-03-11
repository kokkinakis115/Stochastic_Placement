# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 12:38:27 2024

@author: PANAGIOTIS
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import functions

class Microservice: #microservice class, time_series is created upon initialization based on given intensity of application

    def __init__(self, app_id, ms_id, intensity, volatility, duration):
        self.app_id = app_id
        self.ms_id = ms_id
        self.intensity = intensity
        self.volatility = volatility
        self.duration = duration
        self.assigned_to_node = None
        # self._create_usage1(duration_in_hours=duration+1, intensity=intensity, volatility=volatility)
        # self._create_usage2(intensity=intensity, volatility=volatility)
        
    def __repr__(self):
        return f"Microservice #{self.ms_id}, assigned to Node #{self.assigned_to_node}"

    def _assign_to_node(self, timeslot_duration, timeslot_nr, node):
        self.assigned_to_node = node.id
        return node._assign_task(self.usage_time_series[timeslot_duration*timeslot_nr], self.ms_id)
                
    def _create_usage1(self, duration_in_hours, intensity, volatility):
        
        sigma_dict = {'low': 15*60, 'medium': 20*60 , 'high': 30*60}
        mu_dict = {'low': 1800, 'medium': 3600 , 'high': 2*3600}
        lambd_dict = {'low': 50, 'medium': 20 , 'high': 10}
        
        sigma = sigma_dict[volatility]
        mu = mu_dict[intensity]
        lambd = lambd_dict[intensity]
        
        wl = functions.create_wl(sigma=sigma, mu=mu, lambd=lambd, users=200, duration_in_hours=duration_in_hours)
        
        sigma_dict_vol = {'low': 0.0005, 'medium': 0.001 , 'high': 0.002}
        sigma = sigma_dict_vol[volatility]
        mu = 0
        
        time_series = np.zeros(duration_in_hours*3600+1)
        for i in range(len(wl)):
            duration = wl.loc[i, 'end time']+1-wl.loc[i, 'start time']
            # time_series[wl.loc[i, 'start time']:wl.loc[i, 'end time']+1] += np.ones(duration)*wl.loc[i, 'normalized cpu requested']+(sigma * np.random.randn(duration) + mu) 
            
            time_series[wl.loc[i, 'start time']:wl.loc[i, 'end time']+1] += np.ones(duration)*wl.loc[i, 'normalized cpu requested']#+(sigma * np.random.randn(duration) + mu) 

            if (intensity != 'low'):
                time_series += sigma * np.random.randn(duration_in_hours*3600+1) + mu
            
            time_series = np.clip(time_series, 0.001, 1000) 
        
        self.usage_time_series = time_series[3600:]
        return
    
    def _create_usage2(self, workloads, vol_dict):
        self.usage_time_series = functions.produce_workload(intensity=self.intensity, vol=self.volatility, workloads=workloads, vol_dict=vol_dict)
        return

    def _plot_usage(self):
        dates = pd.date_range(start='00:00', periods=self.duration, freq='1s')

        data = pd.DataFrame({'date': dates, 'value': self.usage_time_series})
        
        # Plot the time-series data
        plt.plot(data['date'], data['value'])
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.xticks(rotation = 45)
        plt.title("Usage Time Series")
        plt.show()