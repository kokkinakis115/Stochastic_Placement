
class FixedModel:
    
    def __init__(self, cpu_resources):
        self.cpu = cpu_resources
    
    def _predict_next_timeslot(self, microservice, prediction_timeslot_num):
        return float(self.cpu)
        
