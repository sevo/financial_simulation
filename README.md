# financial_simulation

Run tests from the main directory : 
`
python -m unittest discover -v tests
`

Classes that are somewhat implemented:
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

Current TODOs:
- premenit odkazovanie sa na typ transakcie cez 'TrType' na 'TransactionType'
- triedu Operation a jej podtriedy
- triedu BankAgentFactory
- triedu ConnectionToOperationTransformer

There are certain classes that aren't even in the code yet. 