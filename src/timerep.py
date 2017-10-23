from abc import ABCMeta, abstractmethod
import numpy as np
from cronex import CronExpression
import math
from datetime import datetime
import collections
from itertools import cycle

class TimeRepresentation(metaclass=ABCMeta):
    """
    General abstract class for representing time when the bank operations should occur.
    Actual implementation classes that represent time should implement this class.
    """

    @abstractmethod
    def evaluate(self,datetime): raise NotImplementedError #implemented method should return bool

class Cron(TimeRepresentation):
    """
    Class for representing cron times. This class is meant to be used when we can be certain that on certain days
    or in certain times, the transaction should be executing. 
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
        try: # test if the specified parameters are right in terms of a correct datetime
            datetime(*epoch[:5])
        except:
            raise ValueError("Some value in the tuple doesn't match a correct date. Use the format: year, month,day,hour,minute.")
        if len(epoch) == 6:
            self.cronjob.epoch = epoch
        else:
            raise ValueError("Use tuple with 6 values being in this order: year,month,day,hour,minute,utc_offset!")
            
class TimeDistribution(TimeRepresentation):
    """
    Update as of 23.10.2017: This class shouldn't be used because it doesn't preserve the required probabilities.
        A better implementation is TimeDistribExp that we should be using.
    Class for representing time distributions by using different granularities of time.
    The specified granularities used are : month, day, hour and minute. This class is meant to be used when 
    we want to define the time for transaction's execution as a probability on a certain day. 
    TODO: Afaik there is no way to specify that the probability of transaction to execute should increase
    with the time getting later and later.. is that okay?
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



class TimeDistribExp(TimeRepresentation):
    """
    This is a different,better,working version of TimeDistribution class. Class for representing time distributions 
        by using different granularities of time.
    The specified granularities used are : month, day, hour and minute. This class is meant to be used when 
        we want to define the time for transaction's execution as a probability on a certain day. 
    TODO: Implement weekday granularity. 
    TODO: we cannot work with scipy random variables using this class, so far we can only use histograms 
        or rather dictionaries with (time_value,probability) tuples
    Kedze zo sekvencie dni, mesiacov, atd, je tazke povedat ze kedy chcem mat zaciatok (napr mesiace jan,jun,nov s nenulovou
    pravdepodobnostou - ako viem kedy vygenerovat novy prvok --> uzivatel mohol zamyslat sekvenciu jan,jun,nov alebo
    jun,nov,jan atd atd,preto uzivatel by si mal sam vediet specifikovat kedy jeho obdobie konci.
    Philosophy of this class:
        As input we get some distribution over certain time periods (month,day,hour,etc.). We also get number of transactions
        to generate in this period, and when the sequence of the biggest time period is over (more info to write here!!)
    """
    __months=list(range(1,13))
    __days=list(range(1,32))
    __hours=list(range(0,24))
    __minutes=list(range(0,60))
    __timetypes={"month":TimeDistribExp.__months,"day":TimeDistribExp.__days,
                   "hour":TimeDistribExp.__hours,"minute":TimeDistribExp.__minutes}

    def crange(start, stop, modulo):
        """
        Implements circular range. Useful for when we want to represent start of period and end of period.
            When we know that we are in the interval between the end of period and start of period that is <end;start>
            then we can safely assume we can generate new schedules for transactions.
        """
        result = []
        index = start
        while index != stop:
            result.append(index)
            index = (index + 1) % modulo
        return result
    
    def get_biggest_timetype(self):
        """
        The biggest timetype will serve as something to keep track of the start and end of period for which we generate
            the schedules of transactions. Depending on what is the specified biggest timetype (f.e. "month","day","hour")
            the schedule will change.
        """
        a={"month":2,"day":1.5,"hour":1,"minute":0.5}
        mx=0 #max value
        mx_tt="second"
        for timetype in self.timedist.keys():
            if a[timetype] > mx:
                mx=a[timetype]
                mx_tt=timetype
        return mx_tt
    
    def get_tt_reordered(self):
        """
        Moves what should be the last element of the biggest timetype values array to the end of the array. 
            This way we have clearly defined start at the start of array and end at the end of the array.
        """
        biggest_tt_arr=list(self.timedist[self.biggest_timetype].keys())
        end_ind=biggest_tt_arr.index(self.last_biggest_timetype_value)
        while biggest_tt_arr.index(self.last_biggest_timetype_value) != (len(biggest_tt_arr)-1):
            biggest_tt_arr += [biggest_tt_arr.pop(0)]
        # mali by sme si vytvorit set v ktorom su vsetky hodnoty od zaciatku obdobia do konca obdobia
        # treba na to mozno vyuzit cycle vtedy ked pojdeme do noveho mesiaca/dna/whatever
        return biggest_tt_arr
    
    def __init__(self, td_dict,last_biggest_timetype_val,num_transactions):
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
        
        self.gen_next()
        self.biggest_timetype=self.get_biggest_timetype()
        self.should_gen_next=False
        self.last_biggest_timetype_value=last_biggest_timetype_val
        self.num_transactions=num_transactions
        self.cur_tx_counter=0
        self.tx_timestamps_to_execute=[]
        for _ in range(num_transactions):
            self.tx_timestamps_to_execute.append(self.gen_next())
        #self.get_end_of_seq(list(self.timedist[self.biggest_timetype].keys()),self.biggest_timetype)#self.next_timetype[self.biggest_timetype]
        biggest_tt_general=TimeDistribExp.__timetypes[self.biggest_timetype]
        biggest_tt=self.get_tt_reordered()
        start=biggest_tt[0]
        end=biggest_tt[-1]
        self.active_range=crange(start,end+1,max(biggest_tt_general)+1)
#         pool=cycle_with_index(biggest_tt,biggest_tt.index(STARTPOINT))
    
    def get_end_of_seq(self,timetype):
        pool = cycle(TimeDistribExp.timetypes[timetype])
        return pool
    
    def check_probabilities2(self,prob_dict):
        """
        Check that the probabilities for the specified time granularity are in the range
        of 0<=x<=1 and also check that they sum up to one.
        """
        if TimeDistribution.getname(prob_dict).startswith('scipy.stats'):
            return prob_dict
        values=list(map(lambda x: x if 0<= x <= 1 else 1 if x > 1 else 0, prob_dict.values()))
        velka_suma=sum(values)
        values=[(val/velka_suma) for val in values]
        return collections.OrderedDict(zip(prob_dict.keys(),values))
        
    @classmethod
    def getname(cls,obj):
        """Gets the 'class' name from the object. From my experience it also gives the full module path which is 
        very beneficial for us since that's what we need to identify scipy random variables.
        """
        return str(obj.__class__).split("'")[1]
        
    def check_timestamp(self,object_timestamp,sent_timestamp):
        for timetype,value in object_timestamp.items():
            if timetype == "executed":
                continue
            if value != getattr(sent_timestamp,timetype):
                return False
        return True
        
    def evaluate(self,timestamp): 
        """
        Check for a certain timestamp whether the operation should execute.
        """
        #TODO: should consider how to evaluate when there is both month and week defined

        if (self.cur_tx_counter > 0) and getattr(timestamp,self.biggest_timetype) not in self.active_range:
#     and self.biggest_tt_start < getattr(timestamp,self.biggest_timetype) > self.last_biggest_timetype_value:
            print ("Vygenerovanie novych timestampov")
            self.tx_timestamps_to_execute=[]
            for _ in range(self.num_transactions):
                self.tx_timestamps_to_execute.append(self.gen_next())
            self.should_gen_next=False
            self.cur_tx_counter=0
        
#         else:
            # don't generate new next_timestamp because conditions weren't met
            # the conditions (2) are: the transaction with the last generated timestamp should have been executed,
            # the number of transactions in some period should have been met
            # the period must be over
            # conditions (1) are: we're currently in the period where the transactions should be executing
            # and we haven't yet made the desired amount of transactions
            # there's a clear problem with this implementation: if we randomly generate timestamp at the end of the period
            # as our first timestamp, then we will probably not be able to generate the desired amount of transactions 
            # that we wished to
            # solutions : randomly generate the amount of num_transactions at once
        

        for object_timestamp in self.tx_timestamps_to_execute:
            if object_timestamp["executed"] == True:
                continue
            return_val=self.check_timestamp(object_timestamp,timestamp)
            if return_val == True:
                self.cur_tx_counter+=1
                object_timestamp["executed"]=True
                return True
#                 break
        
        return False
    
    def gen_next(self):
        """
        Generate the next timestamp when the transaction should be executed.
        """
        granularity=["month","day","hour","minute"]
        next_timestamp={}
        for timetype in granularity:
            if timetype in self.timedist:
                prob_dct=self.timedist[timetype]
                next_timestamp[timetype]=np.random.choice(list(prob_dct.keys()),p=list(prob_dct.values()))
        
#         self.next_timestamp = next_timestamp
        next_timestamp["executed"]=False
        return next_timestamp