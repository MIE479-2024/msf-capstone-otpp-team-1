# MIE-479 Bond Portfolio Optimizer

Bond portfolio optimizer

## Initial Setup

### Step 1: Install dependencies

```
pip install -r requirements.txt
```

## Usage

Here's an example of using the library:

### Step 1: Create the object and import bonds data

```python
from bond_optimizer import BondPortfolioOptimizer
from bond_optimizer.BondPortfolioOptimizer import Constraint
from datetime import datetime

# Create optimizer object
optimizer = BondPortfolioOptimizer()
data_file_path = 'data/bonds_w_exp_returns_new.csv'
optimizer.read_csv(data_file_path)
```

### Step 2: Configure your optimizer based on the following allowed parameters + Add constraints:

Allower Parameters: ['constraints', 'benchmark', 'prev_portfolio', 'input_date', 'robust', 'allow_constraint_violation', 'enable_soft_constraints', 'transaction_cost_limit', 't_cost_func']

To understand in detail what each parameter does, refer to the extensive documentation [here](https://docs.google.com/document/d/11XTG5TBSoJqvDXXkQ3tMBu7nHyeJ-GbA0HAyOokLqt8/edit?tab=t.0).

#### 2.1 Automatic set-up using an Excel file of the format of [input_configuration_example.csv](https://github.com/YinanZhao/MIE479-Bond-Portfolio/blob/main/data/input_configuration_example.csv):

```
optimizer.read_in_configuration('data/input_configuration_example.csv')
```

#### 2.2 Manual set-up within library:

Set up configuration parameters

```python
# Can set all configurations into a large dictionary and then call set_optimization_configuration()
optimization_configuration = {
    'robust': True,
    'allow_constraint_violation': False,
    'enable_soft_constraints': False,
    'input_date': datetime(2023,1,2)
}

optimizer.set_optimization_configuration(optimization_configuration)

# Or can be done in-place
optimizer.set_optimization_configuration({'robust': False})
```

Add constraints

```python
# If the user adds a constraint which is trivial, an Exception will be raised
constraints = [Constraint('Cardinality', 'Sector', 'Technology', 5, 10,1), Constraint('WeightIfSelected', 'Security', 'ALL', 0.001, 1,1)]
optimizer.add_constraints(constraints)
```

### Step 3: Optimize and view results!

```python
optimize_results = optimizer.optimize()

# If desired, output can also be exported to a .xlsv file.
optimizer.export_to_xlsx()
```

## Results

After an optimization result, we return a dictionary. This section will go over what the values of the resulting dictionary.

Example output:

```
optimize_results = {
    'Params': {
        'constraints': [...],
        'benchmark': pd.DataFrame(...),
        'prev_portfolio': pd.DataFrame(...),
        'input_date': '2025-01-02',
        'robust': True,
        'allow_constraint_violation': True,
        'enable_soft_constraints': False,
        'transaction_cost_limit': 10000000,
        't_cost_func': compute_spread_t_cost
    },
    'OverallStats': pd.DataFrame([...]),
    'AggregatedContributions': pd.DataFrame([...]),
    'PositionLevelSummary': pd.DataFrame([...]),
    'violated_constraints': [
        Constraint(metric='Weight', scope='Sector', bucket='ALL', lowerbound=0.5, upperbound=1.5, constraint_priority=1)
    ]
}
```

\*NOTE: In the case where a solution is infeasible, the returned dictionary will contain values referring to the previous portfolio. Essentially, this indicates that when we are unable to find a solution, we stay with our portfolio.

Next, we will go over each key component in the dictionary:

### OptimizationStatus

Returns the resulting status of the optimization. One of:

- SUCCESS: The optimizer successfully found a portfolio which meets all constraints.
- FAILED (Reason - Infeasible, user constraint violated): The optimizer was unable to find a portfolio which meets all constraints.
- FAILED (Reason - Infeasible, inadequate turnover budget): The optimizer was able to find a portfolio which meets all user constraints within the allocated turnover budget.

### Params

Returns a dictionary of the user parameters used for optimization, [as in Step 2](#step-2-configure-your-optimizer-based-on-the-following-allowed-parameters).

### OverallStats

Returns a pandas DataFrame with 3 rows. Each row refers to a different portfolio. "ConstrainedTurnover Portfolio" refers to the optimal portfolio. "UnconstrainedTurnover Portfolio" refers to the optimized portfolio without the transaction cost constraint and "Benchmark" refers to the benchmark portfolio (Equally-weighted by default).

Each column of the dataframe represents a different metric of the portfolio. The following metrics are returned for each portfolio:

- EXPECTED_RET: Weighted expected return of the portfolio.
- EXPECTED_RET_PEN_UNCERTAINTY: Weighted expected return with adjustment for uncertainty of returns.
- DURATION: Weighted average duration.
- DTS: Duration times spread (DTS) for the portfolio.
- LIQUIDITY_SCORE: Weighted liquidity score.
- NUM_POSITIONS: Number of unique bonds in the portfolio.
- TURNOVER: Sum of absolute values of turnover for eeach bond in the portfolio.
- NET_TURNOVER: Net turnover in the portfolio.
- T_COST: Total transaction costs.
- WEIGHT: Total weight used in portfolio.
- POS_EXITED: Number of bonds in the prev_portfolio which we are no longer in.
- POS_ENTERED: Number of bonds we entered which were not in prev_portfolio.

### AggregatedContributions

Returns the same columns as for OverallStats. However, groups bonds by contributions of SECTOR, MATURITY and LIQUIDITY.

### PositionLevelSummary

Returns a DataFrame summary for the bonds in our portfolio. Each row refers to a different bond. Essentially, this returns the same columns as we had in the data, with the addition of the WEIGHT column, representing how much of our portfolio we are putting into the bond. The columns are: ['ISIN', 'ISSUER', 'SECTOR', 'WEIGHT', 'EXPECTED_RETURN', 'RETURN_STD_DEV', 'TURNOVER', 'BENCHMARK_WEIGHT', 'BID_PRICE', 'ASK_PRICE']

### violated_constraints

Returns a list of all violated constraints. This is for when we run the optimizer with allow_constraints_violation = True. For more details, refer to the [documentation](https://docs.google.com/document/d/11XTG5TBSoJqvDXXkQ3tMBu7nHyeJ-GbA0HAyOokLqt8/edit?tab=t.0).
