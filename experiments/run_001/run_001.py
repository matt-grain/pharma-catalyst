from pharma_agents.crew import PharmaAgentsCrew

crew = PharmaAgentsCrew()
inputs = {
    'property': 'solubility (logS)',
    'baseline_rmse': '1.3175',
    'experiment_history': 'No previous experiments',
    'proposal': 'Increase n_estimators from 100 to 200'
}
result = crew.crew().kickoff(inputs=inputs)
print(result)
