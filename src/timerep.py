from abc import ABCMeta, abstractmethod
import numpy as np
from cronex import CronExpression
import math
from datetime import datetime
import collections

class TimeRepresentation:
    """
    General abstract class for representing time when the bank operations should occur.
    Actual implementation classes that represent time should implement this class.
    """
    __metaclass__ = ABCMeta

#     @classmethod
#     def version(self): return "1.0"
    @abstractmethod
    def evaluate(self,datetime): raise NotImplementedError #implemented method should return bool

class Cron(TimeRepresentation):
    """
    Class for representing cron times.
    Note:
        The package we use for cron parsing (cronex) supports using repeaters (for periodicity). 
        To use repeater put '%number' in expression.
    """
    def __init__(self,expression):
        """
        Args:
            expression (str) : classic cron expression string
        """
        self.cronjob = CronExpression(expression)
    
    def evaluate(self,datetime):
        self.last_time = datetime
        return self.cronjob.check_trigger(datetime.timetuple()[:5])
    
    def evaluate_expression(self):
        return self.cronjob.numerical_tab
    
    def get_expression(self):
        return self.cronjob.string_tab
    
    def get_epoch(self):
        return self.cronjob.epoch
    
    def set_epoch(self,epoch):
        """
        Setting the epoch will probably be required for repeaters to work correctly. The epoch is important since
        it is the start from which the repeaters will measure the period.
        """
        try:
            datetime(*epoch[:5])
        except:
            raise ValueError("Some value in the tuple doesn't match a correct date. Use the format: year, month,day,hour,minute.")
        if len(epoch) == 6:
            self.cronjob.epoch = epoch
        else:
            raise ValueError("Use tuple with 6 values being in this order: year,month,day,hour,minute,utc_offset!")
            
class TimeDistribution(TimeRepresentation):
    """
    Class for representing time distributions by using different granularities of time.
    The specified granularities used are : month, day, hour and minute.
    TODO: Implement weekday granularity. 
    TODO: Implement some form of counter? maxcount? for a period of time
    """
    
    def __init__(self, td_dict):
        """
        Args:
            td_dict (dictionary) : should be a dictionary with keys of any from "month", "day", "hour", "minute".
                Then the values in those keys should be either probability dictionaries (f.e. for hour there
                should be keys from 0 to 59 with their respective probabilities)
        """
        # USING ORDERED DICTs for the probabilities
        self.timedist = td_dict #dict contains distributions for : month,week, day, hour, minute
        # the distributions supplied should be either a histogram of probabilities (in our case list is enough)
        # or a discrete distribution
        # bud histogram alebo binomial distribution
        #for every such supplied list we should check that the probabilities add up to 1
        for k,v in self.timedist.items():
            self.timedist[k]=self.check_probabilities2(v)

    def check_probabilities(self, prob_dict):
        """
        Check that the sum of the probabilities for the specified time granularity is equal to 1.
        Args:
            prob_dict (dict or scipy.stats type) : should be either a dictionary of probabilities or
                a random variable from the scipy.stats module
        """
        #prob_dict doesnt necessarily need to be a dictionary, we can use scipy distribution.. 
        if TimeDistribution.getname(prob_dict).startswith('scipy.stats'):
            return prob_dict
        if np.sum(prob_dict.values()) == 1:
            return prob_dict
        else:
            values=list(prob_dict.values())
            return collections.OrderedDict(zip(prob_dict.keys(),np.array(values)/np.sum(values)))
    
    def check_probabilities2(self,prob_dict):
        """
        Check that the probabilities for the specified time granularity are in the range
        of 0<=x<=1.
        """
        if TimeDistribution.getname(prob_dict).startswith('scipy.stats'):
            return prob_dict
        values=list(map(lambda x: x if 0<= x <= 1 else 1 if x > 1 else 0, prob_dict.values()))
        return collections.OrderedDict(zip(prob_dict.keys(),values))
        
    @classmethod
    def getname(cls,obj):
        """Gets the 'class' name from the object. From my experience it also gives the full module path which is 
        very beneficial for us since that's what we need to identify scipy random variables.
        """
        return str(obj.__class__).split("'")[1]
        
    def evaluate(self,timestamp): 
        """
        Check for a certain timestamp whether the operation should execute.
        """
        #TODO: should consider how to evaluate when there is both month and week defined
        datetime_names ={"month":"day", "week":"weekday","day":"hour", "hour":"minute","minute":"second"}
        counter=0
        for name,prob_dict in self.timedist.items():
            if self.prob_eval(prob_dict,getattr(timestamp,datetime_names[name])):
                counter+=1
                continue
            else:
                return (False,counter)
        return (True,counter)
        
    def prob_eval(self,prob_dict,value):
        #whats the difference betweeen generating a new random number between 0 and 1 for all of the time lengths and just 
        # using one generated random number for all of them ? does it matter or not? 
        #check if prob_dict really is a dictionary or it could be a scipy random variable
        if TimeDistribution.getname(prob_dict).startswith('scipy.stats') : #this is a scipy random variable
            generated=prob_dict.rvs() #generate a random number from the distribution
            # check if it matches the value
            generated = math.ceil(generated)
            return True if generated == value else False
        else: #it should be a dictionary
            if value not in prob_dict:
                return False
            return True if np.random.random() < prob_dict[value] else False
        
class PeriodicTime(TimeRepresentation):
    
    def __init__(self,time_start,time_period,granularity):
        """
        Class for representing periodic times.
        Args:
            time_start (datetime) : the start time of the operation (when the first operation should be executed)
            time_period (timedelta) : the period of the operation (how often should the operation be executed
                from the start point).
            granularity (str) : can be several values - from least specific to most specific
                - should be either 'day','hour','minute'. We don't specify months and years because that kind of period 
                can and should be represented in days in this case.
        """
        self.time_start = time_start
        self.time_period = time_period
        self.multiplicator = 0
        if granularity not in ["day","hour","minute"]:
            raise ValueError("Specified granularity for the Periodic Time class is wrong! The value is:", granularity)
        self.granularity = granularity
        self.granul_list = ["year","month","day","hour","minute"]
        index = self.granul_list.index(granularity)
        self.granul_list = self.granul_list[:index+1]
    
    def total_time(self,delta,timetype):
        """
        Get total time from timedelta object - in days or hours or minutes.
        """
        seconds = delta.total_seconds()
        if timetype == "days":
            return delta.days
        elif timetype == "hours":
            return seconds // 3600
        elif timetype == "minutes":
            return (seconds) // 60
        else:
            raise ValueError("Wrong value for timetype parameter! value is : ", timetype)        
        
    def evaluate(self,timestamp):
        """
        Args:
            timestamp (timedelta) : timestamp that we want to evaluate the operation on"""
        delta = timestamp - self.time_start
        timetype=self.granularity+"s"
        result,remainder = divmod(self.total_time(delta,timetype),self.total_time(self.time_period,timetype))
        return True if remainder == 0 else False