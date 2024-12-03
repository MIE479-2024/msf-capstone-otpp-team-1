from bond_optimizer import BondPortfolioOptimizer
from bond_optimizer.BondPortfolioOptimizer import Constraint

# Stage 1: Loading in data
optimizer = BondPortfolioOptimizer()
data_file_path = 'data/bonds_w_exp_returns_new.csv'
optimizer.read_csv(data_file_path)

# Stage 2: Setting up optimizer configuration (Excel)​
optimizer.read_in_configuration('data/input_configuration_example.csv')

# Stage 3: Optimizing and getting results​
optimize_results = optimizer.optimize()
print('optimize_results', optimize_results)
optimizer.export_to_xlsx('data/output.xlsx')

# Infeasible constraint here
# new_infeasible_constraint = Constraint(metric='DTS', scope='Sector', bucket='ALL', lowerbound=-20000, upperbound=-10000, constraint_priority=1, violation_penalty=200)
# optimizer.add_constraint(new_infeasible_constraint)