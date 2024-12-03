from bond_optimizer import BondPortfolioOptimizer
from bond_optimizer.BondPortfolioOptimizer import Constraint
from datetime import datetime, timedelta
import pandas as pd
import seaborn as sns
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

    # Dictionary to store detailed results for each rebalancing date
    # Define an empty DataFrame for fields with results
    summary_df = pd.DataFrame(columns=[
        'Date',
        'Expected_Return',
        'Duration',
        'Turnover',
        'Turnover_Cost',
        'DTS',
        'Expected_Return_Benchmark',
        'DTS_Benchmark',
        'Status',
        'ViolatedConstraints'
    ])

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
        prev_portfolio = pd.merge(prev_portfolio,prices, how = 'left', on = 'ISIN')

        #update weights based on the changes to the midpoint between the previous optimization till now.
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
        Params = result_dict['Params']
        OptimizationResult = result_dict['OptimizationStatus']
        OverallStats = result_dict['OverallStats']
        AggregatedContributions = result_dict['AggregatedContributions']
        PositionLevelSummary = result_dict['PositionLevelSummary']
        ViolatedConstraints = result_dict['violated_constraints']

        # If optimization result is not success, rerun model with slacks to see which constraints are violated
        if OptimizationResult != 'SUCCESS':
        
            optimizer.set_optimization_configuration({
            'robust': True,
            'allow_constraint_violation': True,
            'enable_soft_constraints': True
            })
            optimizer2_results = optimizer.optimize()
            ViolatedConstraints = optimizer2_results['violated_constraints']
            # Revert to original settings
            optimizer.set_optimization_configuration({
            'robust': True,
            'allow_constraint_violation': False,
            'enable_soft_constraints': False
            })


        print('Result: ', OptimizationResult)
        # Overall stats for each portfolio
        ConstrainedPortfolio = OverallStats.loc['ConstrainedTurnover']
        UnconstrainedPortfolio = OverallStats.loc['UnconstrainedTurnover']
        BenchmarkPortfolio = OverallStats.loc['Benchmark']

        # Append metrics to the summary DataFrame
        # The following are examples: Sector1 expected return contribution

        # To access the output structure:

        # Output structure (preliminary)
# dict: A dictionary containing the following keys:
#             - 'OptimizationResult' (str): Description of success or failure of optimization and causes. (not implemented yet)
#             - 'OverallStats' (df): Dataframe with portfolio statistics. Each row is one of ConstrainedTurnover, UnconstrainedTurnover, Benchmark
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

        summary_df = pd.concat([
            summary_df,
            pd.DataFrame({
                'Date': [Params['input_date']],
                'Expected_Return': [ConstrainedPortfolio['EXPECTED_RET']],
                'Duration': [ConstrainedPortfolio['DURATION']],
                'Turnover': [ConstrainedPortfolio['TURNOVER']],
                'Turnover_Cost':[ConstrainedPortfolio['T_COST']],
                'DTS': [ConstrainedPortfolio['DTS']],
                'Expected_Return_Benchmark':[BenchmarkPortfolio['EXPECTED_RET']],
                'DTS_Benchmark': [BenchmarkPortfolio['DTS']],
                'Status': [OptimizationResult],
                'ViolatedConstraints': [ViolatedConstraints]

            })
        ], ignore_index=True)
#'Expected_Return_Contribution_Sector1': [AggregatedContributions[(AggregatedContributions['AGGREGATION']== 'SECTOR') & (AggregatedContributions['NAME']== 'Sector2')].iloc[0].EXPECTED_RET]


        print("Expected Return: ", ConstrainedPortfolio['EXPECTED_RET'])

        # Prepare the new portfolio as the previous portfolio for the next iteration
        prev_portfolio = result_dict['PositionLevelSummary'].copy()
        prev_portfolio['MIDPOINT'] = (prev_portfolio['BID_PRICE']  + prev_portfolio['ASK_PRICE'] ) / 2
        prev_portfolio = prev_portfolio[['ISIN', 'WEIGHT', 'MIDPOINT']].rename(columns={'WEIGHT': 'PREV_WEIGHT', 'ISIN': 'ISIN', 'MIDPOINT': 'PREV_MIDPOINT'})

        # Set the t-cost limit. We do not set the t-cost limit during the first optimization, as we started with 100% cash.
        optimizer.set_optimization_configuration({'transaction_cost_limit': t_cost_limit})

        # Advance to the next rebalancing date
        current_date += rebalance_frequency
        
    return summary_df

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
    end_date = datetime(2025, 12, 8)
    rebalance_frequency = timedelta(days=7) # daily rebalancing

    # Run simulation
 
    res = []
    for t_cost_limit in [0.1, 0.125, 0.15, 0.2]:
        optimizer = create_optimizer(start_date)
        summary_results = simulate_rebalancing(optimizer, start_date, end_date, rebalance_frequency, t_cost_limit = t_cost_limit)
        summary_results['t_cost_limit'] = t_cost_limit
        summary_results['Excess_Return'] = summary_results['Expected_Return'] - summary_results['Expected_Return_Benchmark']
        summary_results['Excess_DTS'] = summary_results['DTS'] - summary_results['DTS_Benchmark']
        summary_results['Excess Ret/DTS'] = (summary_results['Expected_Return'] / summary_results['DTS'] ) - (summary_results['Expected_Return_Benchmark'] / summary_results['DTS_Benchmark'] )
        res.append(summary_results)

    # print(summary_results)
    data = pd.concat(res)


metrics_to_plot = ['Excess_Return', 'Turnover', 'Excess Ret/DTS']

for metric in metrics_to_plot:
    # Set plot style
    sns.set(style="whitegrid")

    # Create  plot
    plt.figure(figsize=(10, 6))

    #  color palette for t_cost_limit
    palette = sns.color_palette("tab10", n_colors=data['t_cost_limit'].nunique())
    t_cost_colors = dict(zip(data['t_cost_limit'].unique(), palette))

    unique_statuses = data['Status'].unique()
    marker_shapes = ['.', 's', '*', 'X']  
    marker_styles = {status: marker_shapes[i % len(marker_shapes)] for i, status in enumerate(unique_statuses)}

    # Plot data with distinct black markers and lines by t_cost_limit
    for t_cost in data['t_cost_limit'].unique():
        subset = data[data['t_cost_limit'] == t_cost]
        for status in unique_statuses:
            status_subset = subset[subset['Status'] == status]
            plt.scatter(
                status_subset['Date'],
                status_subset[metric],
                marker=marker_styles[status],
                s=60,  
                color="black",  
                label="_nolegend_" 
            )
        plt.plot(subset['Date'], subset[metric], color=t_cost_colors[t_cost])

    #  legend for t_cost_limit 
    t_cost_legend = [
        plt.Line2D([0], [0], color=t_cost_colors[t_cost], lw=2, label=t_cost)
        for t_cost in t_cost_colors.keys()
    ]

    #   legend for Status 
    status_markers = [
        plt.Line2D([0], [0], marker=marker_styles[status], color='w', 
                markerfacecolor='black', markersize=7, label=status)
        for status in marker_styles.keys()
    ]

    plt.legend(
        handles=t_cost_legend + status_markers,
        title="Legend",
        loc='upper left',
        ncol=3 
    )

   
    plt.title(metric + ' and Opt. Status at Turnover Cost Budgets')
    plt.xlabel('Date')
    plt.ylabel(metric)
    plt.xticks(rotation=45)
    plt.tight_layout()  


    plt.show()