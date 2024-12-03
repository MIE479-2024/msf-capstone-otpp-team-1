import csv
from datetime import date
import math
from typing import TypeVar, List

from gurobipy import Model, GRB
import pandas as pd

# Absolute imports from other files
from bond_optimizer.utils.helpers import compute_spread_t_cost, filter_dataset, scope_map, get_param_row
from bond_optimizer.utils.initial_constraint_check import initial_constraint_checks

pd.set_option('future.no_silent_downcasting', True)

# Input Excel file column types
CSV_DATA_TYPES = {
    "ISIN": "string",
    "issuer": "string",
    "maturity": "string", # convert to datetime on read
    "coupon": "float",
    "sector": "string",
    "OAS": "float",
    "liquidity_score": "float",
    "expected_return": "float",
    "return_std_dev": "float",
    "Date": "string",
    "duration": "float",
    "bid_price": "float",
    "ask_price": "float"
}

# Types
TDataFrame = TypeVar('TDataFrame', bound=pd.DataFrame)

# Constraint constants
MetricTypes = ['DTS', 'Duration', 'Weight', 'Cardinality', 'WeightDeviation', 'WeightIfSelected']
ScopeTypes = ['Portfolio', 'Sector', 'Issuer', 'Maturity', 'Liquidity', 'Security']
# We are given 10 liquidity levels here
LiquidityLevels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# We define maturity levels as 1: 0-3 years, 2: 3-8 years, 3: 8+ years
MaturityLevels = [1, 2, 3]

# Refer to 2.1.1 optimization_configuration 
CONFIGURATION_PARAMETERS = set(['constraints', 'benchmark', 'prev_portfolio', 'input_date', 'robust', 'allow_constraint_violation', 'enable_soft_constraints', 'transaction_cost_limit', 't_cost_func'])

class Constraint:
    '''
    The purpose of this class is to represent constraints which we'll add to our
    model. We add it as a separate class as we foresee users testing out and
    sharing constraints. The interface will easily allow them to do so and keep
    track of which constraints are being applied to the model.

    1. metric: the metric which we are adding a constraint to. One of MetricTypes
    2. scope: the scope at which we'll apply the constraint. One of ScopeTypes
    3. bucket: For Liquidity/Maturity, the level which we are limiting our constraining to.
    For Portfolio, this will be None. For Sector/Issuer, either give the name of
    the Sector/Issuer which we should apply this constraint to or "ALL" if it
    should apply to all issuers/sectors.
    4. lowerbound: the lower bound for the constraint
    5. upperbound: the upper bound for the constraint
    6. constraint_priority: the priority of the constraint
    7. violation_penalty: penalty associated with violating this constraint
    '''
    def __init__(self, metric: str, scope: str, bucket, lowerbound: int, upperbound: int, constraint_priority: int, violation_penalty: int = 5):
        self.validate_constraint(metric, scope, bucket)
        self.metric = metric
        self.scope = scope
        self.bucket = bucket
        self.lowerbound = lowerbound
        self.upperbound = upperbound
        self.constraint_priority = constraint_priority
        self.violation_penalty = violation_penalty

    def validate_constraint(self, metric, scope, bucket):
        '''
        We guarantee that all constraints which are successfully created can be added to our model.
        '''
        if metric not in MetricTypes:
            raise Exception('Error: Unknown Metric Type')
        elif scope not in ScopeTypes:
            raise Exception('Error: Unknown Scope Type')

        if scope == 'Liquidity' and bucket not in LiquidityLevels + ['ALL']:
            raise Exception('Error: Invalid LiquidityLevel bucket')
        elif scope == 'Maturity' and bucket not in MaturityLevels + ['ALL']:
            raise Exception('Error: Invalid MaturityLevels bucket')

    def __repr__(self):
        '''Print the constraint in a human-friendly readable format'''
        return str({
            'Object ID': id(self),
            'Metric': self.metric,
            'Scope': self.scope,
            'Bucket': self.bucket,
            'Lowerbound': self.lowerbound,
            'Upperbound': self.upperbound,
            'Violation Penalty': self.violation_penalty
        })

class BondPortfolioOptimizer:
    def __init__(self):
        '''
        Refer to 2.1 Variable Definitions

        The following will briefly go over what each variable represents.

        1. raw_data: the raw data in pandas dataframe (df) format
        2. raw_data_recent: the most recent (date-wise) raw data, in df format
        3. working_data: the data which our model will be working with to optimize
        4. sectors: a set of all the unique sectors bonds belong to
        5. data_length: length of working_data
        6. benchmark: weights of benchmark portfolio
        7. benchmark_metrics: benchmarket portfolio metrics
        8. constraints: list of tuples (priority, Constraint)
        9. constraint_priority: priority of the next constraint we add
        10. slack_variables_constraints: list of tuples (slackVariable, penalty) for constraints
        11. input_date: specifies the date on which we are optimizing on. If None, use most recent date in data
        12. prev_portfolio: specifies previous portfolio. If sum is less than 1, we assume the rest is held in cash. Passed in as a dataframe with ISIN and PREV_WEIGHT mandatory columns.
        13. t_cost_func: Computes the turnover cost coefficient for each bond

        '''
        # data
        self._raw_data = []
        self._raw_data_recent = []
        self._working_data = []
        self._sectors = set()
        self._data_length = 0
        self._result = {}

        # helpers
        self._benchmark_metrics = {} # dictionary containing metrics of the benchmark portfolio
        self._slack_variables_constraints = []
        self._slack_variables_to_constraint_mapping = []

        # Configuration for optimizer. Can be changed by user
        self.optimization_configuration = {
            'constraints': [],
            'benchmark': None,
            'prev_portfolio': None,
            'input_date': None,
            'robust': False,
            'allow_constraint_violation': False,
            'enable_soft_constraints': False,
            'transaction_cost_limit': 10**9,
            't_cost_func': compute_spread_t_cost
        }

    def clear_data(self):
        '''Resets optimizer variables back to state in __init__'''
        # data
        self._raw_data_recent = []
        self._working_data = []
        self._sectors = set()
        self._data_length = 0

        # helpers
        self._benchmark_metrics = {} # dictionary containing metrics of the benchmark portfolio
        self._slack_variables_constraints = []
        self._slack_variables_to_constraint_mapping = []

        self.optimization_configuration = {
            'constraints': [],
            'benchmark': None,
            'prev_portfolio': None,
            'input_date': None,
            'robust': False,
            'allow_constraint_violation': False,
            'enable_soft_constraints': False,
            'transaction_cost_limit': 10**9,
            't_cost_func': compute_spread_t_cost
        }

    def read_in_configuration(self, file_path):
        """
        Reads in a configuration CSV file. The format should be the same as in data/input_configuration_file_example.csv
        Sets the params as specified in the file.

        Additionally, creates and adds in the constraint objects
        """
        # Open and read the CSV file
        with open(file_path, mode='r') as file:
            reader = csv.reader(file)

            rows_to_skip = ['Params', 'Constraints', 'Metric']
            constraints_flag = False

            # Iterate through rows
            for row in reader:
                if 'Constraints' in row:
                    constraints_flag = True
                if any(skip in cell for cell in row for skip in rows_to_skip) or row[0] == '':
                        continue
                # Reading in parameters
                elif not constraints_flag:
                    key, value = get_param_row(key=row[0], value=row[1])
                    self.set_optimization_configuration({key: value})
                # Reading in constraints
                else:
                    # Create a constraint object
                    metric, scope, bucket, lb, ub, penalty = row[0], row[1], row[2], row[3], row[4], float(row[5])
                    # Small checks - data processing
                    if lb == 'N/A' or lb == '':
                        lb = -10**9
                    if ub == 'N/A' or ub == '':
                        ub = 10**9
                    lb = float(lb)
                    ub = float(ub)
                    if (scope == 'Liquidity' or scope == 'Maturity') and bucket != 'ALL':
                        bucket = int(bucket)

                    new_constraint = Constraint(metric, scope, bucket, lb, ub, 1, penalty)
                    self.add_constraint(new_constraint)

    def set_optimization_configuration(self, configuration: dict):
        '''Takes in a dictionary of user configuration settings, as specified in optimization_configuration'''
        configuration_parameters = configuration.keys()
        unrecognized_keys = set(configuration_parameters) - CONFIGURATION_PARAMETERS
        if len(unrecognized_keys) != 0:
            raise Exception(f"The following parameters were not recognized: {', '.join(unrecognized_keys)}")

        if 'input_date' in configuration_parameters:
            self.set_date(configuration['input_date'])
            del configuration['input_date']

        self.optimization_configuration = {**self.optimization_configuration, **configuration}

    def set_date(self, input_date: date):
        '''Sets the date from raw_data which our optimizer will run on. When empty, we use the most recent date.'''
        if not isinstance(input_date, date):
            raise TypeError("input_date must be a datetime.date object")
        self.optimization_configuration['input_date'] = input_date
        self.create_working_data()

    def add_prev_portfolio_weights(self):
        '''
        Sets the previous portfolio weights, as defined in optimization_configuration.
        '''
        columns_to_keep = ['ISIN', 'PREV_WEIGHT']
        if self.optimization_configuration['prev_portfolio'] is not None:
            self.optimization_configuration['prev_portfolio'] = self.optimization_configuration['prev_portfolio'].loc[:, columns_to_keep]

            # Drop PREV_WEIGHT if it exists, since we the user defines it
            self._working_data = self._working_data.drop(['PREV_WEIGHT'], axis=1, errors='ignore')
            self._working_data = (self._working_data).merge(self.optimization_configuration['prev_portfolio'], on = ['ISIN'], how='left')
            self._working_data['PREV_WEIGHT'] = self._working_data['PREV_WEIGHT'].fillna(0)
        else:
            self._working_data['PREV_WEIGHT'] = 0

    def get_constraints(self):
        '''Returns user constraints added to optimizer.'''
        return self.optimization_configuration['constraints']

    def add_constraint(self, constraint: Constraint):
        '''This is used to add constraints into our model.'''
        # set up benchmark metrics => needed for initial constraints check. We assume constraints are added after benchmark
        if len(self._working_data) == 0:
            self._benchmark_metrics = self.get_metrics(self._working_data)
        elif self.optimization_configuration['benchmark'] is None:
            num_assets = len(self._working_data)
            self._working_data = self._working_data.drop('WEIGHT', axis=1, errors='ignore')
            self._working_data['WEIGHT'] = 1/num_assets if num_assets != 0 else 0
            self._benchmark_metrics = self.get_metrics(self._working_data)
        else:
            self._working_data = self._working_data.drop('WEIGHT', axis=1, errors='ignore')
            self._working_data = pd.merge(self.optimization_configuration['benchmark'], self._working_data, on='ISIN', how='right').fillna(0)
            self._benchmark_metrics = self.get_metrics(self._working_data)

        # check current constraint is OK
        violated_constraints = initial_constraint_checks(self._working_data, [constraint], self._benchmark_metrics)
        if len(violated_constraints) != 0:
            raise Exception("The following constraint can never be satisfied", violated_constraints[0])

        self.optimization_configuration['constraints'].append((constraint.constraint_priority, constraint))

    def add_constraints(self, constraints: List[Constraint]):
        '''Adds a list of user constraints to the optimizer.'''
        for constraint in constraints:
            self.add_constraint(constraint)

    def add_cash(self):
        '''Returns a copy of the dataset passed in but with the cash asset added'''
        with_cash = self._working_data.copy(deep=True)
        cash = {'ISIN': 'Cash', 'ISSUER': '', 'SECTOR': '', 'OAS': 0, 'LIQUIDITY_SCORE': 10, 'EXPECTED_RETURN': 0, 'RETURN_STD_DEV': 0, 'RETURN_VARIANCE': 0, 'DURATION': 0, 'BID_PRICE': 0, 'ASK_PRICE': 0, 'MATURITY_SCORE': 1, 'PREV_WEIGHT':0, 'T_COST_COEFF': 0, 'RETURN_UNCERTAINTY_PENALTY': 0, 'WEIGHT': 0}
        with_cash = pd.concat([with_cash, pd.DataFrame([cash])], ignore_index=True)
        return with_cash

    def test(self, csv_file):
        '''Reads in the csv_file and runs the optimizer on it'''
        self.read_csv(csv_file)
        self.optimize()

    def read_csv(self, file):
        '''Reads in csv file and populates raw_data'''
        self.clear_data()

        # read into pandas and convert columns to datetime
        pd.set_option('display.max_columns', None)
        df = pd.read_csv(file, dtype=CSV_DATA_TYPES)

        df.columns = map(str.upper, df.columns)
        df['MATURITY'] = pd.to_datetime(df['MATURITY'], errors='coerce')
        df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')
        self._raw_data = df

        # Fill in values for working_data
        self._sectors = set(df['SECTOR'].unique())
        self.create_working_data()

    def select_recent_data(self):
        '''Populates raw_data_recent with only data from most recent date'''
        df = self._raw_data
        max_date = df['DATE'].max()
        self._raw_data_recent = df.loc[df['DATE'] == max_date]

    def create_working_data(self):
        '''
        Populates working_data with only columns which we'll use for optimization.
        Additionally, only get rows which belong to the most recent date.
        '''
        self.select_recent_data()
        df = self._raw_data_recent

        # if we have self.optimization_configuration['input_date'], use that date instead for working_data
        if self.optimization_configuration['input_date'] is not None:
            df = self._raw_data
            df = df.loc[df['DATE'] == self.optimization_configuration['input_date']]
        else:
            self.optimization_configuration['input_date'] = df['DATE'].max()

        working_data = df[['ISIN', 'ISSUER', 'SECTOR', 'OAS', 'LIQUIDITY_SCORE', 'EXPECTED_RETURN', 'RETURN_STD_DEV', 'DURATION', 'MATURITY', 'BID_PRICE', 'ASK_PRICE']].copy(deep=True)
        # variance
        working_data.loc[:, 'RETURN_VARIANCE'] = df['RETURN_STD_DEV'] ** 2
        # maturity score
        working_data.loc[:, 'YEARS'] = (working_data['MATURITY'] - self.optimization_configuration['input_date']).dt.days/365
        working_data.loc[:, 'MATURITY_SCORE'] = pd.cut(
            working_data['YEARS'],
            bins=[-float('inf'), 3, 8, float('inf')],
            labels=[1, 2, 3]
        ).astype(int)
        working_data = working_data.drop(columns='YEARS')
        working_data = working_data.drop(columns='MATURITY')
        # liquidity score
        working_data['LIQUIDITY_SCORE'] = pd.cut(
            working_data['LIQUIDITY_SCORE'],
            bins=[-float('inf'), *LiquidityLevels],
            labels=LiquidityLevels
        ).astype(int)
        working_data['RETURN_UNCERTAINTY_PENALTY'] = 1.96*working_data['RETURN_STD_DEV']/math.sqrt(180)
        self._working_data = working_data
        self._data_length = len(working_data)

    def get_metrics(self, dataset: TDataFrame):
        '''
        This function returns the metrics of a dataset which you pass in (weights).
        Note we expect the dataset to match the format of working_data (columns-wise)
        It should have WEIGHT, ISIN, ISSUER, SECTOR, OAS, LIQUIDITY_SCORE, EXPECTED_RETURN,
        RETURN_STD_DEV, RETURN_VARIANCE, DURATION.

        Returns a dictionary with the following keys:
        total_return, total_liquidity, total_dts, total_duration, total_variance
        '''
        N = len(dataset)
        metrics = {
            'total_return': 0,
            'total_liquidity': 0,
            'total_dts': 0,
            'total_duration': 0,
            'total_variance': 0
        }
        if N == 0: # no weights anywhere so total for everything is 0
            return metrics

        metrics['total_return']= (dataset['EXPECTED_RETURN'] * dataset['WEIGHT']).sum()
        metrics['total_variance'] = (dataset['RETURN_VARIANCE'] * dataset['WEIGHT']**2).sum()
        metrics['total_liquidity'] = (dataset['LIQUIDITY_SCORE'] * dataset['WEIGHT']).sum()
        metrics['total_dts'] = (dataset['OAS'] * dataset['DURATION'] * dataset['WEIGHT']).sum()
        metrics['total_duration'] = (dataset['DURATION'] * dataset['WEIGHT']).sum()

        return metrics

    def add_constraint_to_gurobi(self, model, constraint: Constraint, sector_benchmark_metrics, filtered_dataset, weights, cardinalities):
        '''Helper function for adding constraints to gurobi'''
        # Add the slack binary variable
        slack = model.addVar(vtype=GRB.BINARY, name=f"slack_Constraint_{constraint.metric}")

        # Add the constraint depending on the type
        # For slack variables, we use the "Big M approach"
        M = 10**9
        self._slack_variables_constraints.append((slack, constraint.violation_penalty))
        self._slack_variables_to_constraint_mapping.append(constraint)
        slack_contribution = slack * M if self.optimization_configuration['allow_constraint_violation'] else 0

        match constraint.metric:
            case 'DTS':
                model.addConstr(
                    sum(weights[i] * (filtered_dataset['OAS'] * filtered_dataset['DURATION'])[i] for i in filtered_dataset.index) <= sector_benchmark_metrics['total_dts'] + constraint.upperbound + slack_contribution, f"Constraint_{constraint.metric}"
                )

            case 'Duration':
                model.addConstr(
                    sum(weights[i] * filtered_dataset['DURATION'][i] for i in filtered_dataset.index) <= sector_benchmark_metrics['total_duration'] + constraint.upperbound + slack_contribution, f"Constraint_{constraint.metric}"
                )

            case 'Weight':
                model.addConstr(
                    sum(weights[i] for i in filtered_dataset.index) <= constraint.upperbound + slack_contribution, f"Constraint_up{constraint.metric}"
                )

                model.addConstr(
                    sum(weights[i] for i in filtered_dataset.index) >= constraint.lowerbound - slack_contribution, f"Constraint_lb{constraint.metric}"
                )

            case 'WeightDeviation':
                model.addConstr(
                    sum(weights[i] for i in filtered_dataset.index) - filtered_dataset['WEIGHT'].sum() <= constraint.upperbound + slack_contribution, f"Constraint_up{constraint.metric}"
                )
                model.addConstr(
                    filtered_dataset['WEIGHT'].sum() - sum(weights[i] for i in filtered_dataset.index) <= constraint.lowerbound + slack_contribution, f"Constraint_up{constraint.metric}"
                )

            case 'Cardinality':
                model.addConstr(
                    sum(cardinalities[i] for i in filtered_dataset.index) <= constraint.upperbound + slack_contribution, f"Constraint_up{constraint.metric}"
                )
                model.addConstr(
                    sum(cardinalities[i] for i in filtered_dataset.index) >= constraint.lowerbound - slack_contribution, f"Constraint_lb{constraint.metric}"
                )

            case 'WeightIfSelected':
                model.addConstrs(
                    (weights[i]  >= constraint.lowerbound *  cardinalities[i] for i in filtered_dataset.index), f"Constraint_up{constraint.metric}"
                )

                model.addConstrs(
                    (weights[i]  <= constraint.upperbound *  cardinalities[i] for i in filtered_dataset.index), f"Constraint_lb{constraint.metric}"
                )

    def optimize(self):
        '''Runs the Gurobi optimizer and returns a dictionary, as specified in 2.3 Results.'''
        # Get benchmark_metrics. If user didn't set a benchmark portfolio, we use the equally weighted one by default
        if len(self._working_data) == 0:
            self._benchmark_metrics = self.get_metrics(self._working_data)
        elif self.optimization_configuration['benchmark'] is None:
            num_assets = len(self._working_data)
            self._working_data = self._working_data.drop('WEIGHT', axis=1, errors='ignore')
            self._working_data['WEIGHT'] = 1/num_assets if num_assets != 0 else 0
            self._benchmark_metrics = self.get_metrics(self._working_data)
        else:
            self._working_data = self._working_data.drop('WEIGHT', axis=1, errors='ignore')
            self._working_data = pd.merge(self.optimization_configuration['benchmark'], self._working_data, on='ISIN', how='right').fillna(0)
            self._benchmark_metrics = self.get_metrics(self._working_data)

        violated_constraints = set()
        # Will break out of this if we allow constraints to be violated or no constraints are violated
        while True:
            violated_constraints_flag = False
            self._slack_variables_constraints = []
            self._slack_variables_to_constraint_mapping = []

            # Modified to add for transaction cost.
            self.add_prev_portfolio_weights()
            dataset = self.add_cash()
            dataset = self.optimization_configuration['t_cost_func'](dataset)

            # Gurobi setup
            model = Model("Optimal")
            model.Params.LogToConsole = 0
            weights = []
            cardinalities = []
            turnover_change = []
            transaction_dummy = []

            # Weight + cardinality variables
            for i in range(len(dataset)):
                weight_var = model.addVar(name=f"{dataset['ISIN'][i]}", lb=0.0, ub=1.0)
                cardinal_var = model.addVar(name=f"y_i-{i}|{dataset['ISIN'][i]}", vtype=GRB.BINARY)
                weights.append(weight_var)
                cardinalities.append(cardinal_var)
                model.addConstr(weight_var <= cardinal_var)
                model.addConstr(weight_var * 10**9 >= cardinal_var)

            # Full Investment constraint
            model.addConstr(sum(weights[i] for i in range(len(dataset))) == 1.0, "Full Investment")

            # t_cost constraints
            transaction_dummy = model.addVars(len(dataset), name="abs_turnover", lb=0.0, ub=1.0)
            turnover_change = model.addVars(len(dataset), name="turnover", lb=-1.0, ub=1.0)
            model.addConstrs(((weights[i] - dataset["PREV_WEIGHT"][i]) == turnover_change[i] for i in range(len(weights))), name="turnover")

            for i in range(len(dataset)):
                model.addGenConstrAbs(transaction_dummy[i], turnover_change[i])
            # Add user input constraints
            for i, (_, constraint) in enumerate(self.optimization_configuration['constraints']):
                # If constraint has been violated on a previous run, don't include for current run
                if constraint in violated_constraints:
                    continue
                scope = constraint.scope
                bucket = constraint.bucket

                # If bucket is ALL:  if ScopeTypes is in ['Maturity', 'Liquidity'] iterate through all unique values as defined in constraint constants and define a constraint for each.
                # if ScopeTypes is in ['Portfolio', 'Sector', 'Issuer'], iterate through all unique values found in the data and define a constraint for each.
                if bucket == 'ALL':
                    scope_key = scope_map(scope)
                    buckets = set(self._working_data[scope_key].unique()) # Don't want cash

                    for bucket in buckets:
                        # Filter data based on scope
                        filtered_dataset = filter_dataset(dataset, scope, bucket)
                        sector_benchmark_metrics = self.get_metrics(filtered_dataset)

                        # Add the constraint to gurobi model
                        self.add_constraint_to_gurobi(model, constraint, sector_benchmark_metrics, filtered_dataset, weights, cardinalities)
                else:
                    # Filter data based on scope
                    filtered_dataset = filter_dataset(dataset, scope, bucket)
                    sector_benchmark_metrics = self.get_metrics(filtered_dataset)

                    # Add the constraint to gurobi model
                    self.add_constraint_to_gurobi(model, constraint, sector_benchmark_metrics, filtered_dataset, weights, cardinalities)


            slack_contribution = sum(slack * penalty for slack, penalty in self._slack_variables_constraints) if self.optimization_configuration['allow_constraint_violation'] else 0
            # Objective function
            if not self.optimization_configuration['robust']:
                # Objective = expected return - transaction costs - violated constraints
                model.setObjective(
                    sum(weights[i] * dataset['EXPECTED_RETURN'][i] for i in range(len(dataset)))
                    - slack_contribution,
                    GRB.MAXIMIZE
                )
            else:
                # Objective = expected return - transaction costs - violated constraints - robust
                model.setObjective(
                    sum(weights[i] * dataset['EXPECTED_RETURN'][i] for i in range(len(dataset)))
                    - slack_contribution
                    - sum(dataset['RETURN_UNCERTAINTY_PENALTY'][i] * weights[i] for i in range(len(dataset))) ,
                    GRB.MAXIMIZE
                )

            # We first optimize without the transaction cost constraint.
            model.optimize()
            # If unable to optimize => User constraint violated!
            if model.status != GRB.OPTIMAL:
                dataset_without_turnover_constraint = []
                unconstrained_turnover_portfolio_metrics = {}
                unconstrained_turnover_portfolio_metrics['Type'] = 'UnconstrainedTurnover'
                break

            # Else, save results of run without transaction const constraint.
            dataset_without_turnover_constraint = dataset.copy(deep=True)
            dataset_without_turnover_constraint['WEIGHT'] = [weights[i].X for i in range(len(dataset))]
            dataset_without_turnover_constraint['TURNOVER'] = [turnover_change[i].X for i in range(len(dataset))]
            dataset_without_turnover_constraint['CARDINALITY'] = [cardinalities[i].X for i in range(len(dataset))]
            unconstrained_turnover_portfolio_metrics = self.get_detailed_portfolio_metrics(dataset_without_turnover_constraint)
            unconstrained_turnover_portfolio_metrics['Type'] = 'UnconstrainedTurnover'

            # Add constraint for turnover and re-optimize
            model.addConstr(sum([transaction_dummy[i] * dataset["T_COST_COEFF"][i] for i in range(len(dataset))]) <= self.optimization_configuration['transaction_cost_limit'])
            model.optimize()
            # If unable to optimize => Turnover constraint violated
            if model.status != GRB.OPTIMAL:
                break

            # Iterate through slack variables to see which constraints are violated, if any
            for i, slack_variable in enumerate(self._slack_variables_constraints):
                if slack_variable[0].X == 1:
                    violated_constraints.add(self._slack_variables_to_constraint_mapping[i])
                    violated_constraints_flag = True

            # No new constraints were violated! No more runs needed
            if violated_constraints_flag == False or not self.optimization_configuration['allow_constraint_violation'] or self.optimization_configuration['enable_soft_constraints']:
                break
        if model.status == GRB.OPTIMAL:
            model_vars = [weights, cardinalities, turnover_change, transaction_dummy]
            output_res = self.get_optimization_results(model, dataset, model_vars, list(violated_constraints), unconstrained_turnover_portfolio_metrics)
            output_res['OptimizationStatus'] = 'SUCCESS'
        else:
            model_vars = [[], [], [], []]
            output_res = self.get_optimization_results(model, dataset, model_vars, list(violated_constraints), unconstrained_turnover_portfolio_metrics, successful_optimization=False)
            # This means we failed optimization before adding turnover constraint
            if len(dataset_without_turnover_constraint) == 0:
                output_res['OptimizationStatus'] = 'FAILED (Reason - Infeasible, user constraint violated)'
            else:
                output_res['OptimizationStatus'] = 'FAILED (Reason - Infeasible, inadequate turnover budget)'

        # Prints out resulting output_res
        # for key in output_res.keys():
        #     print(key)
        #     print(output_res[key])
        #     print('\n')
        self._result = output_res
        return output_res


    def get_optimization_results(self, model, dataset, model_vars, violated_constraints, unconstrained_turnover_portfolio_metrics, successful_optimization = True):
        '''Returns summarized results in a dictionary after having run the Gurobi optimizer.'''
        weights = model_vars[0]
        cardinalities = model_vars[1]
        turnover_change = model_vars[2]
        transaction_dummy = model_vars[3]
        dataset['BENCHMARK_WEIGHT'] = dataset['WEIGHT'] # assign the benchmark weight to benchmark column
        if successful_optimization:
            dataset['WEIGHT'] = [weights[i].X for i in range(len(dataset))]
            dataset['TURNOVER'] = [turnover_change[i].X for i in range(len(dataset))]
            dataset['CARDINALITY'] = [cardinalities[i].X for i in range(len(dataset))]
        else:
            dataset['WEIGHT'] = dataset['PREV_WEIGHT']
            dataset['TURNOVER'] = [0] * len(dataset)
            dataset['CARDINALITY'] = [1 if dataset['PREV_WEIGHT'][i] != 0 else 0 for i in range(len(dataset))]

        # portfolio_composition = pd.DataFrame({
        #     'Name':  [weights[i].VarName for i in range(len(dataset))],
        #     'Issuer': dataset['ISSUER'],
        #     'Sector': dataset['SECTOR'],
        #     'Maturity Bucket': dataset['MATURITY_SCORE'],
        #     'Liquidity Bucket': dataset['LIQUIDITY_SCORE'],
        #     'Weight': [weights[i].X for i in range(len(dataset))],
        #     'Prev Weight': dataset['PREV_WEIGHT'],
        #     'Turnover':[turnover_change[i].X for i in range(len(dataset))],
        #     'Cardinality':[cardinalities[i].X for i in range(len(dataset))],
        #     'Duration': dataset['DURATION'],
        #     'DTS': dataset['OAS'] * dataset['DURATION'],
        #     'EXPECTED_RETURN': dataset['EXPECTED_RETURN'],
        #     'RETURN_VARIANCE': dataset['RETURN_VARIANCE'],
        #     'OAS': dataset['OAS'],
        #     'bid_price': dataset['BID_PRICE'],
        #     'ask_price': dataset['ASK_PRICE'],
        # })

        dataset.sort_values('WEIGHT', ascending = False, inplace = True)

        # overall stats
        # stats for portfolio
        portfolio_metrics = self.get_detailed_portfolio_metrics(dataset)
        # If optimization was unsuccessful, unconstrained portfolio is same as constrained portfolio
        if not successful_optimization:
            unconstrained_turnover_portfolio_metrics = portfolio_metrics.copy()
            unconstrained_turnover_portfolio_metrics['Type'] = 'UnconstrainedTurnover'

        #stats for benchmark
        benchmark_dataset = dataset.copy()
        benchmark_dataset['WEIGHT'] = benchmark_dataset['BENCHMARK_WEIGHT']
        benchmark_dataset['TURNOVER'] = math.nan
        benchmark_dataset['T_COST_COEFF'] = math.nan
        benchmark_metrics = self.get_detailed_portfolio_metrics(benchmark_dataset)

        portfolio_metrics['Type'] = 'ConstrainedTurnover'
        benchmark_metrics['Type'] = 'Benchmark'

        overall_summary = pd.DataFrame([portfolio_metrics, benchmark_metrics, unconstrained_turnover_portfolio_metrics]).set_index('Type')

        # contribution to stats by group
        groupings = ['SECTOR', 'LIQUIDITY_SCORE', 'MATURITY_SCORE']
        grouped_summaries = []
        for grouping in groupings:
            dataset_group = dataset.copy()

            summary_groups_temp = []
            for i, group in enumerate(dataset_group[grouping].unique()):
                summary_temp = self.get_detailed_portfolio_metrics(dataset_group[dataset_group[grouping] == group])
                summary_temp['NAME'] = group
                summary_groups_temp.append(summary_temp)

            grouped_summary = pd.DataFrame(summary_groups_temp).sort_values('WEIGHT', ascending = False)
            grouped_summary['AGGREGATION'] = grouping
            grouped_summary = grouped_summary.drop(grouped_summary[(grouped_summary['WEIGHT'] == 0) & (grouped_summary['TURNOVER'] == 0)   ].index) # drop groups where weight = 0 and no turnover
            grouped_summaries.append(grouped_summary)

        grouped_summaries =  pd.concat(grouped_summaries)
        grouped_summaries = grouped_summaries[['AGGREGATION','NAME'] + [col for col in grouped_summaries.columns if col not in ['AGGREGATION','NAME']] ]

        # Stock level summary
        position_summary = dataset[['ISIN', 'ISSUER', 'SECTOR', 'WEIGHT','EXPECTED_RETURN','RETURN_STD_DEV', 'TURNOVER',
                                    'BENCHMARK_WEIGHT','BID_PRICE','ASK_PRICE' ]].sort_values('WEIGHT', ascending = False)

        # Optimization Parameters
        Params = self.optimization_configuration.copy()

        results = {'Params': Params, 'OverallStats': overall_summary, 'AggregatedContributions': grouped_summaries, 'PositionLevelSummary': position_summary, 'violated_constraints': violated_constraints}

        return results

    def get_detailed_portfolio_metrics(self,dataset):
        '''Helper function to return metrics of a given portfolio. Useful for getting final results.'''
        portfolio_metrics = {}
        portfolio_metrics['EXPECTED_RET']= (dataset['EXPECTED_RETURN'] * dataset['WEIGHT']).sum()
        portfolio_metrics['EXPECTED_RET_PEN_UNCERTAINTY'] = (dataset['WEIGHT'] * (dataset['EXPECTED_RETURN'] - dataset['RETURN_UNCERTAINTY_PENALTY'])).sum()
        portfolio_metrics['DURATION'] = (dataset['DURATION'] * dataset['WEIGHT']).sum()
        portfolio_metrics['DTS'] = (dataset['OAS'] * dataset['DURATION'] * dataset['WEIGHT']).sum()
        portfolio_metrics['LIQUIDITY_SCORE'] = (dataset['LIQUIDITY_SCORE'] * dataset['WEIGHT']).sum()
        portfolio_metrics['NUM_POSITIONS'] = (dataset['WEIGHT'].abs() > 0 ).sum()
        portfolio_metrics['TURNOVER'] = (dataset['TURNOVER'].abs()).sum()
        portfolio_metrics['NET_TURNOVER'] = (dataset['TURNOVER']).sum()
        portfolio_metrics['T_COST'] = (dataset['T_COST_COEFF']*(dataset['TURNOVER'].abs())).sum()
        portfolio_metrics['WEIGHT'] = dataset['WEIGHT'].sum()
        portfolio_metrics['POS_EXITED'] = ((dataset['WEIGHT'] == 0) &  (dataset['TURNOVER'] != 0) ).sum()
        portfolio_metrics['POS_ENTERED'] = ( (dataset['WEIGHT'] > 0) &  (dataset['TURNOVER'] ==  dataset['WEIGHT']   ) ).sum()

        return portfolio_metrics

    def export_to_xlsx(self, file_name='data/output.xlsx'):
        with pd.ExcelWriter(file_name) as writer:
            self._result['PositionLevelSummary'].to_excel(writer, sheet_name="PositionLevelSummary", index=False)
            self._result['OverallStats'].to_excel(writer, sheet_name="OverallStats", index=True)
            self._result['AggregatedContributions'].to_excel(writer, sheet_name="AggregatedContributions", index=False)


if __name__ == '__main__':
    optimizer = BondPortfolioOptimizer()
    optimizer.clear_data()
    # Run a test method
    optimizer.read_csv('data/bonds_w_exp_returns_new.csv')


