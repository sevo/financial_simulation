# financial_simulation

Source code for financial data simulation published in:
* Mocko, M. and Ševcech, J., 2018, July. Simulation of Bank Transaction Data. In International Workshop on Multi-Agent Systems and Agent-Based Simulation (pp. 99-114). Springer, Cham.


Run tests from the main directory : 
`
python -m unittest discover -v tests
`

Class list:
- BankAgent
- BankModel
- Transaction
- TransactionType
- SimulatorError
- StepTime
- Scheduler
- RandomScheduler
- Operation
- ScheduledOperation
- RandomTimeOperation
- ConnectionToOperationTrasformer
- SimulatorError
- AccountGenerator
- Distribution
- ExactAmount
- ConnectionGenerator
- RandomConnectionGenerator
- DistributionOfDistributions
