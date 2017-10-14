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
- implementovat ConnectionGenerator a aspon jednu jeho podtriedu
- implementovat Connection
- implementovat AccountGenerator
- implementovat Distribution a podtriedu ExactAmount
- implementovat DistributionOfDistributions
- implementovat DistributionOfTimingDistributions
- implementovat TimingDistributionFactory
- opravit triedu TimeDistribution tak, aby sa vedela chovat takym sposobom, ze ked sa nepodari uskutocnit transakciu (pravdepodobnost nepustila), tak na konci toho obdobia, kedy by transakcia mala byt uskutocnena, sa aj uskutocni (bud stylom ze pravdepodobnost postupne rastie, alebo stylom ze ked skratka nebola usutocnena transakcia tak nakonci proste poviem ze ma sa uskutocnit)


There are certain classes that aren't even in the code yet. 