from bond_optimizer import BondPortfolioOptimizer
from bond_optimizer.BondPortfolioOptimizer import Constraint
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
from bond_optimizer.utils.helpers import get_sample_prev_data

class DefaultConstraints:
    """Store default constraint bound values for baseline model"""

    WEIGHT_IF_SELECTED = {
        'METRIC': 'WeightIfSelected',
        'LEVEL': 'Security',
        'NAME': 'ALL',
        'LOWER_BOUND': 0.005,
        'UPPER_BOUND': 0.03
    }

    PORTFOLIO_DTS = {
        'METRIC': 'DTS',
        'LEVEL': 'Portfolio',
        'NAME': None,
        'LOWER_BOUND': 0,
        'UPPER_BOUND': 700
    }

    SECTOR_CARDINALITY = {
        'METRIC': 'Cardinality',
        'LEVEL': 'Sector',
        'NAME': 'ALL',
        'LOWER_BOUND': 3,
        'UPPER_BOUND': 15
    }

    SECTOR_WEIGHT_DEV = {
        'METRIC': 'WeightDeviation',
        'LEVEL': 'Sector',
        'NAME': 'ALL',
        'LOWER_BOUND': 0,
        'UPPER_BOUND': 0.05
    }

    ISSUER_WEIGHT_DEV = {
        'METRIC': 'WeightDeviation',
        'LEVEL': 'Issuer',
        'NAME': 'ALL',
        'LOWER_BOUND': 0,
        'UPPER_BOUND': 0.02
    }

    SECTOR_DTS = {
        'METRIC': 'DTS',
        'LEVEL': 'Sector',
        'NAME': 'ALL',
        'LOWER_BOUND': 0,
        'UPPER_BOUND': 500
    }

    MATURITY_DURATION = {
        'METRIC': 'Duration',
        'LEVEL': 'Maturity',
        'NAME': 'ALL',
        'LOWER_BOUND': 0,
        'UPPER_BOUND': 1
    }

    LIQUIDITY_7 = {
        'METRIC': 'Weight',
        'LEVEL': 'Liquidity',
        'NAME': 7,
        'LOWER_BOUND': 0,
        'UPPER_BOUND': 0.02
    }

    LIQUIDITY_8 = {
        'METRIC': 'Weight',
        'LEVEL': 'Liquidity',
        'NAME': 8,
        'LOWER_BOUND': 0,
        'UPPER_BOUND': 0.05
    }

    LIQUIDITY_9 = {
        'METRIC': 'Weight',
        'LEVEL': 'Liquidity',
        'NAME': 9,
        'LOWER_BOUND': 0,
        'UPPER_BOUND': 0.05
    }

    PORTFOLIO_DURATION = {
        'METRIC': 'Duration',
        'LEVEL': 'Portfolio',
        'NAME': None,
        'LOWER_BOUND': 0,
        'UPPER_BOUND': 2
    }

def simulate_rebalancing(optimizer, start_date, end_date, rebalance_frequency, t_cost_limit):
    """
    Simulate portfolio rebalancing over a specified date range and output results as:
    - A dictionary of DataFrames. Each dataframe corresponds to a date, with weights for the portfolios
    - A summary DataFrame with high-level metrics for each date.

    Parameters:
    - optimizer object with constraints
    - start_date (datetime): Start date for the simulation.
    - end_date (datetime): End date for the simulation.
    - rebalance_frequency (timedelta): Frequency of rebalancing (e.g., 1 month).

    Returns:
    - dict: Dictionary where keys are rebalancing dates and values are DataFrames with portfolio details.
    - pd.DataFrame: Summary DataFrame with metrics for each rebalancing date.
    """

    results = {
        'ConstrainedPortfolio': [],
        'UnconstrainedPortfolio': [],
        'BenchmarkPortfolio': []
    }

    # Initialize the portfolio (empty at the beginning)
    prev_portfolio = pd.DataFrame(columns=['ISIN', 'PREV_WEIGHT', 'PREV_MIDPOINT'])

    # Iterate over the date range with the specified frequency
    current_date = start_date

    while current_date <= end_date:
        print(f"Running optimizer for {current_date.strftime('%Y-%m-%d')}")
        # Set the date and prepare working data
        optimizer.set_optimization_configuration({'input_date': current_date})

        # skip if date is not in the data
        if len(optimizer._working_data) == 0:
            current_date += rebalance_frequency
            continue

        # Update weights based on price changes (simulation)
        prices = optimizer._working_data[['ISIN','BID_PRICE','ASK_PRICE']].copy()
        prices['MIDPOINT'] = (prices['BID_PRICE']  + prices['ASK_PRICE'] ) / 2
        prev_portfolio = pd.merge(prev_portfolio, prices, how = 'left', on = 'ISIN')

        #update weights based on the changes to the midpoint
        if len(prev_portfolio) > 0:
            prev_portfolio['PREV_WEIGHT'] = prev_portfolio["PREV_WEIGHT"] * (prev_portfolio["MIDPOINT"]/prev_portfolio["PREV_MIDPOINT"])
            prev_portfolio['PREV_WEIGHT'] = prev_portfolio['PREV_WEIGHT'].fillna(0)
            # Normalize weights
            prev_portfolio['PREV_WEIGHT'] = prev_portfolio["PREV_WEIGHT"]/(prev_portfolio["PREV_WEIGHT"].sum() )

        prev_portfolio.drop(['BID_PRICE', 'ASK_PRICE', 'PREV_MIDPOINT'], axis = 1, inplace = True)

        # Update prev_portfolio
        optimizer.set_optimization_configuration({'prev_portfolio': prev_portfolio})
        # Run optimization
        optimization_result = optimizer.optimize()

        # Extract outputs
        result_dict = optimization_result.copy()

        # Unpack outputs
        # OptimizationResult field TBD

        OverallStats = result_dict['OverallStats']

        # Overall stats for each portfolio
        ConstrainedPortfolio = OverallStats.loc['ConstrainedTurnover']
        UnconstrainedPortfolio = OverallStats.loc['UnconstrainedTurnover']
        BenchmarkPortfolio = OverallStats.loc['Benchmark']
        results['ConstrainedPortfolio'].append(ConstrainedPortfolio)
        results['UnconstrainedPortfolio'].append(UnconstrainedPortfolio)
        results['BenchmarkPortfolio'].append(BenchmarkPortfolio)

        # Prepare the new portfolio as the previous portfolio for the next iteration
        prev_portfolio = result_dict['PositionLevelSummary'].copy()
        prev_portfolio['MIDPOINT'] = (prev_portfolio['BID_PRICE']  + prev_portfolio['ASK_PRICE'] ) / 2
        prev_portfolio = prev_portfolio[['ISIN', 'WEIGHT', 'MIDPOINT']].rename(columns={'WEIGHT': 'PREV_WEIGHT', 'ISIN': 'ISIN', 'MIDPOINT': 'PREV_MIDPOINT'})

        # first day is all cash
        optimizer.set_optimization_configuration({'transaction_cost_limit': t_cost_limit})

        # Advance to the next rebalancing date
        current_date += rebalance_frequency

    # Calculate averages for each portfolio type
    averages = {}
    for portfolio_type, portfolio_data in results.items():
        # Convert list of Series to a DataFrame
        portfolio_df = pd.DataFrame(portfolio_data)
        averages[portfolio_type] = portfolio_df.mean()

    # Display the averages
    for portfolio_type, avg_values in averages.items():
        print(f"\n{portfolio_type} Averages:\n\n{avg_values}")

    return results

def create_optimizer(start_date):
    optimization_configuration = {
        'robust': True,
        'allow_constraint_violation': False,
        'enable_soft_constraints': False,
    }

    # Initialize your library
    optimizer = BondPortfolioOptimizer()

    # Import data
    data_file_path = 'data/bonds_w_exp_returns_new.csv'
    optimizer.read_csv(data_file_path)

    # add constraints
    constraints = [
    DefaultConstraints.WEIGHT_IF_SELECTED,
    DefaultConstraints.PORTFOLIO_DTS,
    DefaultConstraints.SECTOR_CARDINALITY,
    DefaultConstraints.SECTOR_WEIGHT_DEV,
    DefaultConstraints.ISSUER_WEIGHT_DEV,
    DefaultConstraints.SECTOR_DTS,
    DefaultConstraints.MATURITY_DURATION,
    DefaultConstraints.LIQUIDITY_7,
    DefaultConstraints.LIQUIDITY_8,
    DefaultConstraints.LIQUIDITY_9,
    DefaultConstraints.PORTFOLIO_DURATION
    ]

    for c in constraints:
        optimizer.add_constraint(
            Constraint(
                c['METRIC'],
                c['LEVEL'],
                c['NAME'],
                c['LOWER_BOUND'],
                c['UPPER_BOUND'],
                1
            )
        )
    # set previous portfolio
    prev_portfolio_day = start_date - timedelta(days=1)
    prev_portfolio = get_sample_prev_data(optimizer._raw_data, prev_portfolio_day.year, prev_portfolio_day.month, prev_portfolio_day.day) # pass in the weights of the current portfolio, pre-optimization
    optimization_configuration['prev_portfolio'] = prev_portfolio
    optimizer.set_optimization_configuration(optimization_configuration)
    return optimizer

if __name__ == '__main__':
    # simulation parameters
    start_date = datetime(2025, 1, 2)
    end_date = datetime(2025, 6, 1)
    rebalance_frequency = timedelta(days=7) # weekly rebalancing

    # Run simulation
    optimizer = create_optimizer(start_date)
    summary_results = simulate_rebalancing(optimizer, start_date, end_date, rebalance_frequency, t_cost_limit = 0.15)

