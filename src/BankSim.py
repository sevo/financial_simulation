from mesa import Agent, Model
from mesa.time import RandomActivation
import random, datetime, names, pandas as pd
from collections import Counter
from mesa.datacollection import DataCollector
from enum import Enum
import numpy as np
from scipy.stats import lognorm, dirichlet
from scipy import stats
import scipy
import math
import yaml
import importlib
from abc import ABCMeta, abstractmethod

################################################### globals ###################################################
def load_config(path):
    # Read YAML file and return the config
    with open(path, 'r') as stream:
        config = yaml.load(stream)
        return config

config = load_config("config.yaml")

def get_func_from_pkg(pkgname,funcname):
    rr = importlib.import_module(pkgname)
    return getattr(rr,funcname)

def call_func(func,args,kwargs,args_for_eval,kwargs_for_eval,dct):
    #dct is variable dictionary used for storing variables we might need
    evaled_args = list(args) #shallow copy of a list
    evaled_kwargs = dict(kwargs) #shallow copy of dict, should be enough?
    for index in args_for_eval:
        evaled_args[index] = eval(evaled_args[index])
    for index in kwargs_for_eval:
        evaled_kwargs[index] = eval(evaled_kwargs[index])
    return func(*evaled_args,**evaled_kwargs)

def get_cfg_func_result(fp_func,variable_dct={}):
    if type(fp_func) != dict:
        return fp_func
    func =  get_func_from_pkg(fp_func["package"],fp_func["funcname"])
    required_kwords=["args","kwargs","args_for_eval","kwargs_for_eval"]
    required_dtypes=[list,dict,list,list]
    for i,keyword in enumerate(required_kwords):
        if keyword not in fp_func:
            fp_func[keyword]=required_dtypes[i]()
    result = call_func(func,fp_func["args"],fp_func["kwargs"],fp_func["args_for_eval"],fp_func["kwargs_for_eval"],variable_dct)
    return result

def get_time_from_cfg(cfg):
    timelength={
    "days":get_cfg_func_result(cfg["days"],{}),
    "hours":get_cfg_func_result(cfg["hours"],{}),
    "minutes":get_cfg_func_result(cfg["minutes"],{}),
    "seconds":get_cfg_func_result(cfg["seconds"],{})}
    return timelength

################################################### globals-end ###################################################

################################################### small stuff ###################################################
class SimulatorError(Exception):
    """
    Exception raised for errors in the simulator.
    
    Args:
        msg (str) : explanation of the error
    """
    def __init__(self, msg):
        self.msg = msg
    def __str__(self):
        return self.msg

class TransactionType(Enum):
    # TODO : premenit vsetky referencie ktore idu cez povodny nazov 'TrType' na aktualny 'TransactionType'
    """Enumeration class for transaction types.
    
    * DEPOSIT = put real cash into an account (doesn't necessarily need to be the account of the person who puts the money in).
    * WITHDRAWAL = withdraw real cash from an account (probably the person doing it must be the owner of the account)
    * TRANSFER = transfer of funds into another account (cashless)
    * INKASO = cash collection in english; the owner of the account authorizes another account (usually a big company) to
      transfer funds from the owner's account to that account. The company may then transfer funds without any more approval.
    * ABP = automatic bill payment; is a recurring (scheduled) type of transfer
    """
    DEPOSIT = 1 #vklad penazi (hotovostny)
    WITHDRAWAL = 2 #vyber penazi (hotovostny)
    TRANSFER = 3 #platba penazi na ucet -- inak vraj su 2 typy: ACH a wire transfer ale to len urcuje rychlost vykonania
    INKASO = 4 #anglicky cash collection 
    ABP = 5 # automatic bill payment - trvaly prikaz
    
    def __str__(self):
        """Convert enum type into string."""
        return str(self.name)

class Transaction():
    """
    Class which represents a single transaction, which should equal one row in a dataframe.
    """
    def __init__(self,sender, receiver, tr_type, amount, timestamp, step_count):
        """
        Args:
            sender (BankAgent) : person that is making the transaction (agent)
            receiver (BankAgent) : the other party receiving the goods from the transaction (agent)
            tr_type (TrType) : transaction type 
            amount (float) : the transaction amount
            timestamp (datetime) : timestamp of when the transaction was done
            step_count (int) : the step count of the model when the transaction was executed
        """
        self.sender=sender
        self.receiver=receiver
        self.sender_name=sender.name
        self.receiver_name=receiver.name
        self.tr_type=tr_type
        self.amount=amount
        self.timestamp=timestamp
        self.sender_id = self.sender.unique_id
        self.receiver_id = self.receiver.unique_id
        self.step_count = step_count
        self.sender_label = self.sender.label
        self.receiver_label = self.receiver.label
        
    def to_dict(self):
        """Creates dictionary for a transaction which will then be used for creating a dataframe of transactions"""
        return dict((key, value) for key, value in self.__dict__.items() if not key in ["day_scheduled","sender","receiver"])

class StepTime():
    """Class for converting time to number of steps for simulator"""
    #we can define our own kind of "imaginary time" where we can define our own rules such as all months have 30 days 
    def __init__(self,step_length):
        """
        Args:
            step_length (timedelta) : model's step_length specified by the timedelta type
        """
        self.step_length = step_length #is of type timedelta
        self.total_seconds = step_length.total_seconds()
        self.second = 1 / self.total_seconds
        self.minute = 60 / self.total_seconds
        self.hour = 3600 / self.total_seconds
        self.day = (24 * 3600) / self.total_seconds
        self.week = 7*(24 * 3600) / self.total_seconds
        
    def time_to_steps(self,days,hours=0,minutes=0,seconds=0):
        """Return the closest possible time in number of steps (round always to nearest larger integer)
        
        Args:
            days (int) : how many days to convert to steps 
            hours (int) : how many ours to convert to steps
            minutes (int) : how many minutes to convert to steps
            seconds (int) : how many minutes to convert to steps
        """
        return math.ceil(days*self.day + hours * self.hour + minutes * self.minute + seconds * self.second)
        
    def timedelta_to_steps(td):
        pass


class Scheduler:
    """General scheduler class."""
    def add(self,Agent):
        pass
    def step(self,stepcount):
        pass
    
class RandomScheduler(Scheduler):
    """
    Random scheduler.
    """
    def __init__():
        self.super()
        self.agents=[]
        
    def add(self,Agent):
        self.agents.append(Agent)
    
    def step(self,stepcount):
        perm=np.random.permutation(len(self.agents))
        for index in perm:
            agents[index].step(stepcount)

################################################### small stuff-end ###################################################
class ConnectionToOperationTransformer:
    __metaclass__ = ABCMeta
    
    def __init__(self,amount_distribution,timing_distribution):
        """
        Args:
            amount_distribution (DistributionOfDistributions) : ...
            timing_distribution (DistributionOfTimingDistributions) : ...
        """
        
    @abstractmethod
    def transform(self,connections): raise NotImplementedError #method not implemented


class Operation:
    """
    General abstract class for representing an operation that will generate a bank transaction when it occurs.
    Actual implementation classes that represent different types of operations should implement this class.
    """
    __metaclass__ = ABCMeta
    
    @abstractmethod
    def execute(self,stepcount): raise NotImplementedError #implemented method should return bool
        
class ScheduledOperation(Operation):
    def __init__(self,start,end,amount_distribution,time_distribution):
        """
        Args:
            start (int): ...
            end (int): ...
            amount_distribution (Distribution): ...
            time_distribution (TimeRepresentation): ...
        """
        self.start=start
        self.end=end
        self.amount_distribution=amount_distribution
        self.time_distribution=time_distribution
        
    def execute(self,stepcount):
        if self.start <= stepcount <= self.end:
            # we are in the interval for which the operation should be executable
            # we will probably need to turn the stepcount into a datetime
            if self.time_distribution.evaluate(stepcount) == True: # the operation should be executed on this stepcount
                amount=self.amount_distribution.sample()
                # now let's create a transaction... but we don't have info about origin and target accounts..
                return transaction
        return None
    
class RandomOperation(Operation):
    def __init__(self, friends_distributions_of_amount,friends_list,friends_probabilities):
        self.friends_distributions_of_amount=friends_distributions_of_amount
        self.friends_list=friends_list
        self.friends_probabilities=friends_probabilities
    def execute(self,stepcount):
        for index,target in enumerate(friends_list):
            if random.random() <= self.friends_probabilities[index]:
                #probability requirement is satisfied
                amount=self.friends_distributions_of_amount[index].sample()
                #create transaction
                return transaction
        return None


class Scheduler:
    """General scheduler class."""
    def add(self,Agent):
        pass
    def step(self,stepcount):
        pass
    
class RandomScheduler(Scheduler):
    """
    Random scheduler.
    """
    def __init__():
        self.super()
        self.agents=[]
        
    def add(self,Agent):
        self.agents.append(Agent)
    
    def step(self,stepcount):
        perm=np.random.permutation(len(self.agents))
        for index in perm:
            agents[index].step(stepcount)



class BankAgent(Agent):
    """ An agent generating transactions."""
    def __init__(self, unique_id, model, account_number, name, balance, bank_country,bank_name):
        """
        Args:
            unique_id (int): unique_id of the agent, needed for mesa Agent type
            model (mesa.Model): simulation model (BankModel), needed for mesa agent initialization
            account_number (int) : number (ID) of the account in the bank
            name (str): name of agent
            balance (float) : float representing balance on the account
            bank_country (str) : identifier of the bank's country
            bank_name (str) : bank's name
        """
        super().__init__(unique_id, model)
        self.name = name
        self.balance = balance
        self.bank_country = bank_country
        self.bank_name = bank_name
        self.operations = []
        
    def set_label(self,label):
        """This method sets the agent's label (normal,fraudster)."""
        self.label=label
    
    def amount_2decimal(self,amount):
        """Usually generated floats can be generated up to many decimal places. We just need two. Convert to 2 decimal places.
        
        Args:
            amount (float) : transaction amount, but can be any float that we want to convert to only 2 decimal places
        """
        return float("%.2f" % amount)
        
    def add_operation(operation):
        self.operations.append(operation)
        
    def step(self,stepcount):
        """
        Agent step function. Called at every step of the model (when the model's step function is called).
        This function calls the method ``execute`` on every operation that is associated with the agent.
        It also puts whatever transactions that the agent executed into the model's global transaction array.
            
        Raises:
            SimulatorError : error raised if there is not enough agents to do meaningful transactions with
        """
        if len(self.model.schedule.agents) < 2:
            raise SimulatorError('We need atleast two agents to make meaningful transactions')
        
        executed_transactions = []
        for operation in self.operations:
            transaction=operation.execute(stepcount)
             #when an operation executes, it should create an executed transaction in the bank
            if transaction != None:
                executed_transactions.append(transaction)
        
        self.model.transactions.extend(executed_transactions)


class BankModel(Model):
    """
    A bank model which serves the purpose of a simulator class.
    """
    
    def __init__(self, step_length=None):
        """
        Args:
            step_length (timedelta) : the length of one step of simulator (f.e. one second, one hour, one day)
        
        **Attributes**
        
        Attributes:
            schedule (?) : scheduler for agent activation (controls the order of calling agents' step functions)
            time (datetime) : the current time of the model. Isn't really used for much, just to have some kind of idea 
                about how much time is passing in the model (although the primary time-keeping method is step counting)
            transactions (list(Transaction)) : list of all transactions executed in the simulation
            step_count (int) : counts how many steps the model has executed so far
            step_time (StepTime) : StepTime instance which we then use to convert time into steps
        """
        cfg = config["model"]
        self.step_length = step_length
        self.schedule = RandomActivation(self)
        starttime=cfg["starttime"]
        self.time = datetime.datetime(year=starttime["year"],month=starttime["month"],day=starttime["day"],hour=8,minute=0)
        self.transactions=[] # list of all transactions
        self.initialized = False
        self.agents=set()
    
    def add_agent(Agent):
        self.agents.add(Agent)
        self.schedule.add(Agent)

    #do one step of simulation, doing what we want to do at each step + calling agent's step functions using scheduler.step()
    def step(self):
        """
        Model's step function that is supposed to run the logic performed at each step of the simulator.
        """
        self.schedule.step()
        self.time+=self.step_length #we will increase the time after each step by the timedelta specified in model's constructor
        self.step_count+=1
    
    #run the model, parameter: number of steps to execute
    def run_model(self,num_of_steps):
        """
        Run the model for a certain amount of steps.
        
        Args:
            num_of_steps (int) : number of steps to run the model for
        """
        if self.initialized == False: #the model hasn't been initialized yet (no agents, no connections, no operations)
            self.generate_agents() #first generate agents
            self.generate_connections() #when agents are generated, we can generate connections between them
            self.generate_operations() #when connections are generated, we can generate concrete operations
            self.initialized = True
    
        for _ in range(num_of_steps):
            self.step()
    
    def transactions_to_df(self):
        #we have a custom column order
        """
        Create a ``pandas`` dataframe from existing Transaction objects in ``model.transactions`` list variable.
        For this dataframe we have specified a custom column ordering inside the function.
        """
        transdict={}
        counter=1
        for x in self.transactions:
            transdict[counter]=x.to_dict()
            counter+=1
        df = pd.DataFrame.from_dict(transdict,orient='index')
        cols = df.columns.tolist()
        custom_cols = ['sender_name','receiver_name','tr_type','amount','timestamp','sender_id','receiver_id','step_count']
        for col in cols: #in case there will be additional columns which we don't yet know about
            if col not in custom_cols:
                custom_cols.append(col)
        df = df[custom_cols]
        return df

