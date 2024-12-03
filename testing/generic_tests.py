from bond_optimizer import BondPortfolioOptimizer
from bond_optimizer.BondPortfolioOptimizer import Constraint
from bond_optimizer.utils.helpers import get_sample_prev_data

from datetime import datetime
import math


# Initialize your library
optimizer = BondPortfolioOptimizer()

# Run a test method
data_file_path = 'data/bond_data_rev1.csv'
#optimizer.test(data_file_path)

optimizer.clear_data()
optimizer.read_csv('data/bonds_w_exp_returns_new.csv')

optimizer.set_optimization_configuration({'input_date': datetime(2025,1,2)}) #set the date for the simulation
optimizer.create_working_data()

prev_portfolio = get_sample_prev_data(optimizer._raw_data, 2025,1,1)
optimizer.set_optimization_configuration({'prev_portfolio': prev_portfolio}) # pass in the weights of the current portfolio, pre-optimization


# add constraints
constraints = [Constraint('Cardinality', 'Sector', 'Technology', 5, 10,1), Constraint('WeightIfSelected', 'Security', 'ALL', 0.001, 1,1)]
optimizer.add_constraints(constraints)

# Optimize, storing output to variable
test_res = optimizer.optimize()

# Output

# Output structure (preliminary)
# dict: A dictionary containing the following keys:
#             - 'OptimizationResult' (str): Description of success or failure of optimization and causes. (not implemented yet)
#             - 'OverallStats' (df): Dataframe with portfolio statistics. Each row is one of Portfolio
#                                         (optimized turnover budget constrained portfolio),
#                                         Unconstrained (optimized turnover budget unconstrained portfolio), and Benchmark
#                           'EXPECTED_RET', 'EXPECTED_RET_PEN_UNCERTAINTY', 'DURATION', 'DTS',
#                           'LIQUIDITY_SCORE', 'NUM_POSITIONS', 'TURNOVER', 'NET_TURNOVER',
#                           'T_COST', 'WEIGHT', 'POS_EXITED', 'POS_ENTERED'
#                                         e.g., test_res['OverallStats'].loc['Portfolio'].EXPECTED_RET gets the expected return of the optimized turnover budget constrained portfolio
#             - 'AggregatedContributions' (df): Get the statistics and portfolio contributions of SECTOR, MATURITY, and LIQUIDITY breakdowns of the optimized turnover budget constrained portfolio
#               e.g., AggContributions[(AggContributions['AGGREGATION']== 'SECTOR') & (AggContributions['NAME']== 'Technology')].iloc[0].DURATION gets the duration contribution of technology sector (this is NOT the duration of the tech sector)
#             - 'PositionLevelSummary' (df): Get the weights and information for each stock in the portfolio
#                    'ISIN', 'ISSUER', 'SECTOR', 'WEIGHT', 'EXPECTED_RETURN',
#                        'RETURN_STD_DEV', 'TURNOVER', 'BENCHMARK_WEIGHT'
#             - 'ViolatedConstraints': TBD

print(test_res['AggregatedContributions'].columns)
AggContributions = test_res['AggregatedContributions']
print(AggContributions[(AggContributions['AGGREGATION']== 'SECTOR') & (AggContributions['NAME']== 'Technology')].iloc[0].DURATION)

PositionLevelSummary = test_res['PositionLevelSummary']
print(PositionLevelSummary.columns)


## WEIGHT TESTS
# # Test1
# optimizer.add_constraint(Constraint('WeightIfSelected', 'Security', 'ALL', 0.001, 1,1))
# optimizer.add_constraint(Constraint('Weight', 'Sector', 'ALL', 0.01, 0.2 ,1))
# optimizer.add_constraint(Constraint('Weight', 'Issuer', 'Company4', 0.05, 1,1))


# Test2
# optimizer.add_constraint(Constraint('WeightIfSelected', 'Security', 'ALL', 0.001, 1,1))
# optimizer.add_constraint(Constraint('Weight', 'Sector', 'ALL', 0.01, 0.2 ,1))
# optimizer.add_constraint(Constraint('Weight', 'Issuer', 'ALL', 0.01, 0.1, 1))


# Test3
# optimizer.add_constraint(Constraint('WeightIfSelected', 'Security', 'ALL', 0.001, 1,1))
# optimizer.add_constraint(Constraint('Weight', 'Sector', 'ALL', 0.01, 0.2 ,1))
# optimizer.add_constraint(Constraint('Weight', 'Liquidity', 'ALL', 0.05, 0.2, 1))

# Test4
# optimizer.add_constraint(Constraint('WeightIfSelected', 'Security', 'ALL', 0.001, 1,1))
# optimizer.add_constraint(Constraint('Weight', 'Sector', 'ALL', 0.01, 0.2 ,1))
# optimizer.add_constraint(Constraint('Weight', 'Maturity', 'ALL', 0.2, 1, 1))

# test_res = optimizer.optimize()
# print(test_res)

## DURATION TESTS
# Test1
# optimizer.add_constraint(Constraint('WeightIfSelected', 'Security', 'ALL', 0.001, 1,1))
# optimizer.add_constraint(Constraint('DTS', 'Sector', 'ALL', None, 4, 1))
# optimizer.add_constraint(Constraint('DTS', 'Issuer', 'ALL', None, 4, 1))

# Test2
# optimizer.add_constraint(Constraint('WeightIfSelected', 'Security', 'ALL', 0.001, 1,1))
# # Because the DTS of our optimized is actually higher than the benchmark and we always compare to benchmark, this upper bound won't do anything unless it's actually negative
# optimizer.add_constraint(Constraint('DTS', 'Portfolio', None, None, -5, 1))

# Test3
# optimizer.add_constraint(Constraint('WeightIfSelected', 'Security', 'ALL', 0.01, 1,1))
# optimizer.add_constraint(Constraint('Cardinality', 'Sector', 'ALL', 1, 100, 1))
# optimizer.add_constraint(Constraint('DTS', 'Sector', 'Consumer Goods', None, -2, 1))

# test_res = optimizer.optimize()
# print(test_res)

## DTS TESTS
# Test1
# optimizer.add_constraint(Constraint('WeightIfSelected', 'Security', 'ALL', 0.01, 1,1))
# optimizer.add_constraint(Constraint('DTS', 'Portfolio', None, None, 1, 1))

# Test2
# optimizer.add_constraint(Constraint('WeightIfSelected', 'Security', 'ALL', 0.01, 1,1))
# optimizer.add_constraint(Constraint('DTS', 'Sector', 'ALL', None, 1, 1))

# Test3
# optimizer.add_constraint(Constraint('WeightIfSelected', 'Security', 'ALL', 0.01, 1,1))
# optimizer.add_constraint(Constraint('DTS', 'Sector', 'ALL', None, 1, 1))
# optimizer.add_constraint(Constraint('Cardinality', 'Sector', 'ALL', 1, 5, 1))

#
# test_res = optimizer.optimize()
# print(test_res)



## WeightDeviation TESTS
# Test1 => uniform portfolio essentially
# optimizer.add_constraint(Constraint('WeightIfSelected', 'Security', 'ALL', 0.001, 1,1))
# optimizer.add_constraint(Constraint('WeightDeviation', 'Security', 'ALL', 0.00001, 0.00001 ,1))
# optimizer.add_constraint(Constraint('WeightDeviation', 'Issuer', 'Company4', 0.05, 1,1))


# Test2
# optimizer.add_constraint(Constraint('WeightIfSelected', 'Security', 'ALL', 0.001, 1,1))
# optimizer.add_constraint(Constraint('WeightDeviation', 'Sector', 'ALL', 0.01, 0.01 ,1))
#

#Baseline Portfolio

# constraints = [Constraint('WeightIfSelected', 'Security', 'ALL', 0.005, 0.03,1),
#                Constraint('DTS', 'Portfolio', None, 0, 700, 1),
#                Constraint('Cardinality', 'Sector', 'ALL', 3, 15, 1),
#                 Constraint('WeightDeviation', 'Sector', 'ALL', 0, 0.05 ,1),
#                  Constraint('WeightDeviation', 'Issuer', 'ALL', 0, 0.02 ,1),
#                  Constraint('DTS', 'Sector', 'ALL', 0, 500, 1),
#                  Constraint('Duration', 'Maturity', 'ALL', 0, 1, 1),
#                  Constraint('Weight', 'Liquidity', 7, 0, 0.02, 1),
#                  Constraint('Weight', 'Liquidity', 8, 0, 0.05, 1),
#                  Constraint('Weight', 'Liquidity', 9, 0, 0.05, 1),
#                  Constraint('Duration', 'Portfolio', None, 0, 2, 1)]

# optimizer.add_constraint(Constraint('WeightIfSelected', 'Security', 'ALL', 0.005, 1,1))
# optimizer.add_constraint(Constraint('DTS', 'Portfolio', None, 0, 700, 1))
# optimizer.add_constraint(Constraint('Cardinality', 'Sector', 'ALL', 3, math.inf, 1))
# optimizer.add_constraint(Constraint('WeightDeviation', 'Sector', 'ALL', 0, 0.05 ,1))
# optimizer.add_constraint(Constraint('WeightDeviation', 'Issuer', 'ALL', 0, 0.02 ,1))
# optimizer.add_constraint(Constraint('DTS', 'Sector', 'ALL', 0, 500, 1))
# optimizer.add_constraint(Constraint('Duration', 'Maturity', 'ALL', 0, 1, 1))
# optimizer.add_constraint(Constraint('Weight', 'Liquidity', 7, 0, 0.02, 1))
# optimizer.add_constraint(Constraint('Weight', 'Liquidity', 8, 0, 0.05, 1))
# optimizer.add_constraint(Constraint('Weight', 'Liquidity', 9, 0, 0.05, 1))
# optimizer.add_constraint(Constraint('Duration', 'Portfolio', None, 0, 2, 1))


# optimizer.add_constraints(constraints)
# test_res = optimizer.optimize()
# print(test_res)