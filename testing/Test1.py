import sys
import os
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pandas as pd
import math

from bond_optimizer import BondPortfolioOptimizer
from bond_optimizer.BondPortfolioOptimizer import Constraint

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

def run_optimization_test(
    constraint_name: str,
    multiplier_range: tuple, 
    num_points: int,
    start_date: datetime,
    end_date: datetime,
    rebalance_freq: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Runs optimization testing different multiples of a constraint's default upper bound
    
    Args:
        constraint_name: constraint to be modified
        multiplier_range: (min, max) multipliers to apply to default upper bound, e.g. (0.2,2.0) means that we shock by 2%-200% of baseline
        num_points: Number of test points between min and max multiplier
        start_date: Start date for testing period
        end_date: End date for testing period
        rebalance_freq: Days between rebalances, 1 for daily rebalancing, 14 fo biweekly etc.
    
    Returns:
        Tuple of means and stds for plotting
    """
    dates = []
    current_date = start_date
    while current_date <= end_date:
        dates.append(current_date)
        current_date += timedelta(days=rebalance_freq)
        
    constraint_to_modify = getattr(DefaultConstraints, constraint_name)
    baseline = constraint_to_modify['UPPER_BOUND']
    test_values = np.linspace(baseline * multiplier_range[0], baseline * multiplier_range[1], num_points)
    
    results_list = []
    
    for test_value in test_values:
        print(f"\nTesting {constraint_name} upper bound: {test_value:.2f}")
        metrics_for_dates = []
        prev_portfolio = pd.DataFrame(columns=['ISIN', 'PREV_WEIGHT', 'PREV_MIDPOINT'])
        optimizer = BondPortfolioOptimizer()
        optimizer.read_csv('data/bonds_w_exp_returns_new.csv')
        for date in dates:
            temp_prev_portfolio = prev_portfolio.copy(deep=True)
            print(f"Processing date: {date.strftime('%Y-%m-%d')}")
            
            num_assets = len(optimizer._working_data)
            benchmark_weights = pd.DataFrame({
                'ISIN': optimizer._working_data['ISIN'],
                'WEIGHT': [1/num_assets] * num_assets
            })

            # Update weights based on price changes (simulation)
            prices = optimizer._working_data[['ISIN','BID_PRICE','ASK_PRICE']].copy()
            prices['MIDPOINT'] = (prices['BID_PRICE']  + prices['ASK_PRICE'] ) / 2
            prev_portfolio = pd.merge(prev_portfolio,prices, how = 'left', on = 'ISIN')


            if len(prev_portfolio) > 0:
                prev_portfolio['PREV_WEIGHT'] = prev_portfolio["PREV_WEIGHT"] * (prev_portfolio["MIDPOINT"]/prev_portfolio["PREV_MIDPOINT"])
                prev_portfolio['PREV_WEIGHT'] = prev_portfolio['PREV_WEIGHT'].fillna(0)
                # Normalize weights
                prev_portfolio['PREV_WEIGHT'] = prev_portfolio["PREV_WEIGHT"]/(prev_portfolio["PREV_WEIGHT"].sum() )
             
            prev_portfolio.drop(['BID_PRICE', 'ASK_PRICE', 'PREV_MIDPOINT'], axis = 1, inplace = True)
            
            optimizer.set_optimization_configuration({
                'benchmark': benchmark_weights,
                'prev_portfolio': prev_portfolio,
                'robust': True,
                'allow_constraint_violation': True,
                'enable_soft_constraints': True,
                'input_date': date
            })
            if len(optimizer._working_data) == 0:
                continue
            
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
                upper_bound = test_value if c == constraint_to_modify else c['UPPER_BOUND']
                optimizer.add_constraint(
                    Constraint(
                        c['METRIC'],
                        c['LEVEL'],
                        c['NAME'],
                        c['LOWER_BOUND'],
                        upper_bound,
                        1
                    )
                )
            
            try:
                result = optimizer.optimize()
                portfolio_stats = result['OverallStats'].loc['ConstrainedTurnover']
                benchmark_stats = result['OverallStats'].loc['Benchmark']

                metrics_for_dates.append({
                    'EXPECTED_RET': portfolio_stats['EXPECTED_RET'],
                    'BENCHMARK_EXPECTED_RET':benchmark_stats['EXPECTED_RET'],
                    'EXCESS_RET':portfolio_stats['EXPECTED_RET'] - benchmark_stats['EXPECTED_RET'],
                    'DURATION': portfolio_stats['DURATION'],
                    'TURNOVER': portfolio_stats['TURNOVER'],
                    'NUM_POSITIONS': round(portfolio_stats['NUM_POSITIONS']),
                    'POS_ENTERED': round(portfolio_stats['POS_ENTERED']),
                    'POS_EXITED': round(portfolio_stats['POS_EXITED']),
                    'T_COST': portfolio_stats['T_COST']
                })
                prev_portfolio = result['PositionLevelSummary'].copy()
                prev_portfolio['MIDPOINT'] = (prev_portfolio['BID_PRICE']  + prev_portfolio['ASK_PRICE'] ) / 2
                prev_portfolio = prev_portfolio[['ISIN', 'WEIGHT', 'MIDPOINT']].rename(columns={'WEIGHT': 'PREV_WEIGHT', 'ISIN': 'ISIN', 'MIDPOINT': 'PREV_MIDPOINT'})

                # update the t cost
                optimizer.set_optimization_configuration({'transaction_cost_limit': 0.15})
            
            except Exception as e:
                prev_portfolio = temp_prev_portfolio
                print(f"Error for date {date}: {str(e)}")
                metrics_for_dates.append({
                    'EXPECTED_RET': 0, 'BENCHMARK_EXPECTED_RET':0, 'EXCESS_RET':0, 'DURATION': 0, 'TURNOVER': 0,
                    'NUM_POSITIONS': 0, 'POS_ENTERED': 0, 'POS_EXITED': 0,
                    'T_COST': 0
                })
        
        results_list.append(pd.DataFrame(metrics_for_dates))
        
    
    # means and standard deviations
    means = pd.DataFrame([df.mean() for df in results_list], index=test_values)
    stds = pd.DataFrame([df.std() for df in results_list], index=test_values)

    # positions can only be whole numbers
    means['NUM_POSITIONS'] = means['NUM_POSITIONS'].round()
    means['POS_ENTERED'] = means['POS_ENTERED'].round()
    means['POS_EXITED'] = means['POS_EXITED'].round()
    
    return means, stds

def plot_shock_impact(means_df: pd.DataFrame, stds_df: pd.DataFrame):
    plt.style.use('seaborn')
    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(8, 12))
    fig.suptitle('Optimization Results', fontsize=14)

    test_values = means_df.index

    # Turnover plot
    ax1.errorbar(test_values, means_df['TURNOVER'],
                fmt='o', capsize=5, capthick=1, elinewidth=1)
    ax1.set_title('Portfolio Turnover')
    ax1.set_xlabel('Upper Bound')
    ax1.set_ylabel('Turnover')
    ax1.grid(True, alpha=0.3)

    # Number of positions
    ax2.errorbar(test_values, means_df['NUM_POSITIONS'],
                fmt='o', capsize=5, capthick=1, elinewidth=1)
    ax2.set_title('Number of Positions')
    ax2.set_xlabel('Upper Bound')
    ax2.set_ylabel('Count')
    ax2.grid(True, alpha=0.3)
    ax2.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # Positions entered/exited
    ax3.errorbar(test_values, means_df['POS_ENTERED'],
                fmt='o', label='Entered', capsize=5, capthick=1, elinewidth=1)
    ax3.errorbar(test_values, means_df['POS_EXITED'],
                fmt='o', label='Exited', capsize=5, capthick=1, elinewidth=1)
    ax3.set_title('Positions Entered/Exited')
    ax3.set_xlabel('Upper Bound')
    ax3.set_ylabel('Count')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # Excess return
    ax4.errorbar(test_values, means_df['EXCESS_RET'],
                fmt='o', capsize=5, capthick=1, elinewidth=1)
    ax4.set_title('Excess Return (rel. Benchmark)')
    ax4.set_xlabel('Upper Bound')
    ax4.set_ylabel('Return')
    ax4.grid(True, alpha=0.3)

    # Transaction cost
    ax5.errorbar(test_values, means_df['T_COST'],
                fmt='o', capsize=5, capthick=1, elinewidth=1)
    ax5.set_title('Transaction Cost')
    ax5.set_xlabel('Upper Bound')
    ax5.set_ylabel('Cost')
    ax5.grid(True, alpha=0.3)

    # Excess return
    # ax6.errorbar(test_values, means_df['EXCESS_RET'],
    #             fmt='o', capsize=5, capthick=1, elinewidth=1)
    # ax6.set_title('Excess Return (rel. Benchmark)')
    # ax6.set_xlabel('Upper Bound')
    # ax6.set_ylabel('Return')
    # ax6.grid(True, alpha=0.3)

    fig.delaxes(plt.subplot(3, 2, 6))
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":

    means_df, stds_df = run_optimization_test(
        constraint_name='LIQUIDITY_7',
        multiplier_range=(0.2, 2),
        num_points=30,
        start_date=datetime(2025, 1, 2),
        end_date=datetime(2025, 6, 1), # do weekly
        rebalance_freq=7
    )



    plot_shock_impact(means_df, stds_df)
