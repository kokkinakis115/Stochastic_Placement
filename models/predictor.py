import numpy as np
# import pyfeng as pf
import matplotlib.pyplot as plt
import pandas as pd

import functions

class BSM_predictor:
    
    def __init__(self, intr = 0.0, texp = 3600, lookback_duration = 3600, threshold = 0.1):
        self.time_slot_duration = texp
        self.intr = intr
        self.lookback = lookback_duration
        # self.starting_strike = strike
        self.threshold = threshold
        return
        
    
    def _timeslot_volatility2(self, time_series, lookback):
        data = pd.DataFrame({'value': time_series})
        data['log'] = np.log(data['value'] / data['value'].shift(1))
        df = data['log'].rolling(window=lookback).std()
        return df
        
    def _compute_options(self, workload, starting_strike):
        texp = self.time_slot_duration #1 day in trading = 1 second in workload
        cp = 1

        num_of_time_slots = len(workload) // self.time_slot_duration
        predictions_array = [starting_strike]
        # max_array = []
        deltas_array = [0]
        volatility = self._timeslot_volatility2(workload, self.lookback)
        
        # strike_price = self.starting_strike
        
        for timeslot_num in range(num_of_time_slots): #compute loop for each timeslot (1h)
            # lookback_timeseries = workload[max((timeslot_num+1)*self.time_slot_duration-self.lookback, 0):min((timeslot_num+1)*self.time_slot_duration, len(workload)+1)]
            
            # timeslot_timeseries = workload[timeslot_num*self.time_slot_duration:min((timeslot_num+1)*self.time_slot_duration, len(workload)+1)]
            # spot_price = timeslot_timeseries[-1]
            
            spot_price = workload[(timeslot_num+1)*self.time_slot_duration]
            
            vol = volatility.loc[(timeslot_num+1)*self.time_slot_duration]
                
            prediction = min(2.5*spot_price, functions.compute_strike_price2(spot=spot_price, vol=vol, texp=texp, intr=self.intr, delta=self.threshold, cp=cp))
            
            predictions_array.append(prediction)
            # max_array.append(max(timeslot_timeseries))
            
        self.predictions = predictions_array[:-1]
        # self.max_array = max_array
        predictions = np.array([starting_strike])
        # maxima = np.array([])
        
        for timeslot_num in range(num_of_time_slots): 
            predictions = np.append(predictions, np.ones(self.time_slot_duration)*self.predictions[timeslot_num])
            # maxima = np.append(maxima, np.ones(self.time_slot_duration)*self.max_array[timeslot_num])
        
        self.predictions_time_series = predictions
        # self.maxima_time_series = maxima
        self.deltas_array = deltas_array

        return

    def _predict_next_timeslot(self, microservice, prediction_timeslot_num):

        timeslot_num = prediction_timeslot_num-1
        texp = self.time_slot_duration 
        cp = 1

        workload = microservice.usage_time_series

        volatility = self._timeslot_volatility2(workload, self.lookback)

        timeslot_timeseries = workload[timeslot_num*self.time_slot_duration:min((timeslot_num+1)*self.time_slot_duration, len(workload)+1)]
        spot_price = timeslot_timeseries[-1]

        vol = volatility.loc[(timeslot_num+1)*self.time_slot_duration]

        prediction = min(2*spot_price, functions.compute_strike_price2(spot=spot_price, vol=vol, texp=texp, intr=self.intr, delta=self.threshold, cp=cp))        

        return prediction


    def _plot_workload_and_options(self, workload):

        # num_of_time_slots = len(workload) // self.time_slot_duration

        dates1 = pd.date_range(start='00:00', periods=len(workload), freq='5min')
        dates2 = pd.date_range(start='00:00', periods=len(self.predictions_time_series), freq='5min')

        data = pd.DataFrame({'date': dates1, 'value': workload})

        # Set the 'date' column as the index
        data.set_index('date', inplace=True)

        # Plot the time-series data
        plt.figure(dpi=1200)
        plt.plot(data.index, data['value'], label = "CPU Workload")
        plt.plot(dates2, self.predictions_time_series, 'g--', label = "Model Predictions")
        # plt.plot(data.index, self.maxima_time_series, label = "Time Slot Max")
        plt.xlabel('Time')
        plt.ylabel('Number of CPUs')
        plt.xticks(rotation = 45)
        plt.title(f'Parameters: Δ = {self.threshold*100}%, Timeslot Duration = {self.time_slot_duration*5}min')
        # plt.title(f'Workload Prediction for {workload.users} users in a timespan of {workload.size} minutes\n T = {self.time_slot_duration}, r = {self.intr}, M = {self.lookback}, t = {self.texp}')
        plt.legend()
        plt.savefig("plot.png", bbox_inches='tight')
        plt.show()

        return