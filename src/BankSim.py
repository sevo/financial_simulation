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

stepcount=0
steptime=None
current_timestamp=None

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

class AccountGenerator:
    __next_id=0

    def get_unique_ID():
        AccountGenerator.__next_id+=1
        return AccountGenerator.__next_id

class Distribution:
    def __init__(self,random_var):
        """
        Args:
            random_var (scipy.stats.*) : a random variable (we mainly expect it to be from package `scipy.stats`),
                supporting function `rvs()`.
        """
        self.random_var = random_var

    def sample(self,size=1):
        """
        Args:
            size (int) : specify the size of the returned array of samples from the distribution
        """
        #IMPORTANT! allow size to be more so that this thing can be easily used for arbitrary long sizes
        if size == 1:
            return self.random_var.rvs()
        elif size>1:
            return self.random_var.rvs(size=size)
        else:
            raise ValueError("Wrong value for parameter: size.")

class ExactAmount(Distribution):
    def __init__(self,exact_var):
        self.exact_var=exact_var

    def sample(self,size=1):
        if size==1:
            return self.exact_var
        elif size > 1:
            return np.array([self.exact_var for _ in range(size)])
        else:
            raise ValueError("Wrong value for parameter: size.")


class DistributionOfDistributions():
    def __init__(self,distribution_class,distribution_param_list):
        """
        Distribution for generating distributions of a certain determined type (determined by distribution_func)
        Args:
            distribution_class (str) : a string containing the full path to the desired function. For example you can
                specify path to uniform random variable from scipy stats like this `scipy.stats.uniform`.
            distribution_param_list (list(Distribution)) : list of `Distribution` objects, each of which corresponds to
                a parameter that the distribution_class function should be instantiated with.
        """
        self.distribution_module=distribution_class[:distribution_class.rindex('.')]
        self.distribution_class=distribution_class[distribution_class.rindex('.')+1:]
        self.distribution_param_list=distribution_param_list
    
    def sample_distribution(self,size=1):
        """
        Should return an object of type `Distribution`. This method instantiates the class specified in constructor called
            `distribution_class`.
        Args:
            size (int) : how many `Distribution` objects we wish to get
        """
        imported_module = importlib.import_module(self.distribution_module)
        function=getattr(imported_module,self.distribution_class)
        result_distributions=[]
        for i in range(size):
            args=[d.sample() for d in self.distribution_param_list]
            result_distributions.append(function(*args)) #function(*args,**kwargs)

        if size == 1:
            return result_distributions[0]
        elif size > 1:
            return result_distributions
        else:
            raise ValueError("Wrongly specified size. Size must be greater than zero!")

class ConnectionGenerator(metaclass=ABCMeta):

    @abstractmethod
    def generate_connections(self,num_connections,agents):
        """
        Generate `num_connections` amount of connections between agents in the list `agents`.
        Args:
            num_connections (int) : number of connections to generate
            agents (list(Agent)) : list of agents between which the connections should be generated
        Returns a list of connections.
        """
        pass
    
class RandomConnectionGenerator(ConnectionGenerator):
    """
    Generates random connections between agents.
    """
    def __init__(self, probability_distribution):
        """
        Args:
            probability_distribution (Distribution) : The probability distribution from which we will generate probabilities
                of agents executing a transaction between each other.
        """
        self.probability_distribution=probability_distribution
    
    def generate_connections(self,num_connections,agents):
        """
        Args:
            num_of_connections (int) : how many connections to generate
            agents (BankAgent) : list of agents that we want to generate the connections between
        """
        connections = []
        
        origins=np.random.choice(agents, num_connections, replace=True)
        targets=np.random.choice(agents, num_connections, replace=True)
        probabilities=self.probability_distribution.sample(size=num_connections)
        #generate the connections
        for i in range(num_connections):
            connections.append(Connection(origins[i],probabilities[i],targets[i]))
        return connections

class Connection:
    def __init__(self,origin,probability,target):
        """
        Args:
            origin (BankAgent) : the origin agent (where the money comes from)
            probability (float) : the probability which says how likely it is that this connection going to be used.
                This probability can/should be used when transforming the `Connection` to an `RandomTimeOperation`, but
                probably not when transforming into a `ScheduledOperation`.
            target (BankAgent) : the target agent (where the money goes to)
        """
        self.origin=origin
        if (0<=probability<=1) == False:
            raise ValueError("Probability of connection should be in range 0 <= prob <= 1")
        self.probability=probability
        self.target=target
        

class ConnectionToOperationTransformer(metaclass=ABCMeta):
    """
    Abstract class.
    Transforms every `Connection` into `Operation`. Agents have a list of `Operation` objects which they should execute.
    """

    def __init__(self,amount_distribution,timing_distribution):
        """
        Args:
            amount_distribution (DistributionOfDistributions) : ...
            timing_distribution (DistributionOfTimingDistributions) : ...
        """
        self.amount_distribution=amount_distribution
        self.timing_distribution=timing_distribution

    @abstractmethod
    def transform(self,connections): 
        raise NotImplementedError #method not implemented


class Operation(metaclass=ABCMeta):
    """
    General abstract class for representing an operation that will generate a bank transaction when it occurs.
    Actual implementation classes that represent different types of operations should implement this class.
    """
    
    def __init__(self):
        pass

    @abstractmethod
    def execute(self,timestamp,stepcount): raise NotImplementedError #implemented method should return bool
        
class ScheduledOperation(Operation):
    # what we need for this class is to be able to generate transactions
    # transactions need to have the following:
    # sender, receiver, tr_type, amount, timestamp, step_count.. the timestamp isnt really necessary
    # but the other things are

    def __init__(self,sender,receiver,tr_type,start,end,amount_distribution,time_distribution):
        """
        Args:
            start (datetime): the starting timestamp of doing these transactions
            end (datetime): timestamp after which no transactions should be made
            amount_distribution (Distribution): ...
            time_distribution (TimeRepresentation): ...
        """
        self.start=start
        self.end=end
        self.amount_distribution=amount_distribution
        self.time_distribution=time_distribution
        self.sender=sender
        self.receiver=receiver
        self.tr_type=tr_type
        
    def execute(self,timestamp,stepcount):
        transactions=[] #only for providing compatibility
        if self.start <= timestamp <= self.end:
            # we are in the interval for which the operation should be executable
            # we will probably need to turn the stepcount into a datetime
            if self.time_distribution.evaluate(timestamp) == True: # the operation should be executed on this stepcount
                amount=self.amount_distribution.sample()
                transactions.append(Transaction(self.sender,self.receiver,self.tr_type,amount,timestamp,stepcount))
        return transactions # return list of zero or one transaction(s)
    
class RandomOperation(Operation):
    def __init__(self, sender,tr_type,friends_distributions_of_amount,friends_list,friends_probabilities):
        self.friends_distributions_of_amount=friends_distributions_of_amount #asi typu DistributionOfDistributions?
        self.friends_list=friends_list
        self.friends_probabilities=friends_probabilities
        self.tr_type=tr_type
        self.sender=sender

    def execute(self,timestamp,stepcount):
        print ("ROP: global stepcount = {}".format(stepcount))
        transactions=[]
        for index,receiver in enumerate(self.friends_list):
            if random.random() <= self.friends_probabilities[index]:
                #probability requirement is satisfied
                amount=self.friends_distributions_of_amount[index].sample()
                #create transaction
                transactions.append(Transaction(self.sender,receiver,self.tr_type,amount,timestamp,stepcount))
        return transactions


class Scheduler(metaclass=ABCMeta):
    """General scheduler class."""
    @abstractmethod
    def add(self,Agent):
        pass

    @abstractmethod
    def step(self,stepcount):
        pass
    
class RandomScheduler(Scheduler):
    """
    Random scheduler.
    """
    def __init__(self):
        self.agents=[]
        
    def add(self,Agent):
        self.agents.append(Agent)
    
    def step(self,current_timestamp,stepcount):
        perm=np.random.permutation(len(self.agents))
        for index in perm:
            self.agents[index].step(current_timestamp,stepcount)



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
        
    def add_operation(self,operation):
        self.operations.append(operation)
        
    def step(self,current_timestamp,stepcount):
        """
        Agent step function. Called at every step of the model (when the model's step function is called).
        This function calls the method ``execute`` on every operation that is associated with the agent.
        It also puts whatever transactions that the agent executed into the model's global transaction array.
            
        Raises:
            SimulatorError : error raised if there is not enough agents to do meaningful transactions with
        """
        # TODO: think about sending the stepcount to the agent and what we really want to use..
        # so far there is disconnect between somewhere using stepcount, and somewhere using timestamp
        # should be unified
        if len(self.model.schedule.agents) < 2:
            raise SimulatorError('We need atleast two agents to make meaningful transactions')
        
        executed_transactions = []
        # global current_timestamp
        for operation in self.operations:
            transactions=operation.execute(current_timestamp,stepcount)
             #when an operation executes, it should create an executed transaction in the bank
            executed_transactions.extend(transactions)
        
        self.model.transactions.extend(executed_transactions)


class BankModel(Model,metaclass=ABCMeta):
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
        self.schedule = RandomScheduler()#RandomActivation(self)
        self.step_count=0
        starttime=cfg["starttime"]
        self.time = datetime.datetime(year=starttime["year"],month=starttime["month"],day=starttime["day"],hour=8,minute=0)
        self.transactions=[] # list of all transactions
        self.initialized = False
        self.agents=list() # probably better to be a list than a set
        global steptime
        steptime=StepTime(self.step_length)
    
    @abstractmethod
    def generate_agents(self):
        pass

    @abstractmethod
    def generate_connections(self):
        pass

    @abstractmethod
    def generate_operations(self):
        pass

    def add_agent(self,Agent):
        self.agents.append(Agent)
        self.schedule.add(Agent)

    #do one step of simulation, doing what we want to do at each step + calling agent's step functions using scheduler.step()
    def step(self):
        """
        Model's step function that is supposed to run the logic performed at each step of the simulator.
        """
        global stepcount, current_timestamp
        if self.step_count == 0: #initialization of these variables
            stepcount=self.step_count
            current_timestamp=self.time

        self.schedule.step(current_timestamp,stepcount)
        self.time+=self.step_length #we will increase the time after each step by the timedelta specified in model's constructor
        self.step_count+=1
        stepcount=self.step_count
        current_timestamp=self.time

    
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

