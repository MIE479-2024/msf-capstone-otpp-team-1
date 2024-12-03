import sys
import os
from typing import List, Dict, Any
from bond_optimizer.utils.helpers import get_sample_prev_data

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bond_optimizer import BondPortfolioOptimizer
from bond_optimizer.BondPortfolioOptimizer import Constraint
from datetime import datetime
import pandas as pd

def setup_optimizer():
    optimizer = BondPortfolioOptimizer()
    optimizer.clear_data()
    optimizer.read_csv(os.path.join('data', 'bond_data_rev1.csv'))
    optimizer.set_date(datetime(2023,1,2))
    optimizer.create_working_data()
    optimizer.define_prev_portfolio_weights(get_sample_prev_data(optimizer._raw_data, 2023,1,1))
    return optimizer

def run_test_case(test_constraints: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Runs a single test case for the given constraints
    """
    optimizer = setup_optimizer()
    
    # Add all constraints for test case
    for constraint in test_constraints:
        optimizer.add_constraint(Constraint(
            metric=constraint['Metric'],
            scope=constraint['Level'],
            bucket=constraint['Name'],
            lowerbound=constraint['Lowerbound'],
            upperbound=constraint['Upperbound'],
            constraint_priority=1
        ))
    
    result = optimizer.optimize()
    return result

def analyze_results(result: pd.DataFrame, test_num: int, constraints: List[Dict[str, Any]]):
    """Results for a given test case. Returns all weights and checks constraints"""
    print(f"\nTest {test_num} Results")
    print("=" * 50)
    
    # Get bonds with weights and print results
    positions = result[result['Weight'] > 0.0001]
    
    print(f"Total bonds selected: {len(positions)}")
    print(f"Total portfolio weight: {positions['Weight'].sum()}")
    
    # Analyze by sector
    sector_analysis = positions.groupby('Sector').agg({
        'Weight': ['sum', 'count']
    })
    print("\nSector Analysis:")
    print(sector_analysis)
    
    # Check constraints
    print("\nConstraint Verification:")
    for constraint in constraints:
        metric = constraint['Metric']
        level = constraint['Level']
        name = constraint['Name']
        lb = constraint['Lowerbound']
        ub = constraint['Upperbound']
        
        if metric == 'Cardinality': # Labeled for cardinality, but can just change depending on test needed
            if level == 'Portfolio':
                count = len(positions)
                print(f"Portfolio Cardinality: {count} (should be between {lb} and {ub})")
                print(f"Constraint met: {lb <= count <= ub}")
            elif level == 'Sector':
                if name == 'ALL':
                    sector_counts = positions.groupby('Sector').size()
                    print(f"Sector Cardinalities: {sector_counts.to_dict()}")
                    print(f"Constraints met: {all(lb <= count <= ub for count in sector_counts)}")
                else:
                    sector_count = len(positions[positions['Sector'] == name])
                    print(f"{name} Sector Cardinality: {sector_count} (should be between {lb} and {ub})")
                    print(f"Constraint met: {lb <= sector_count <= ub}")
        
        elif metric == 'Weight': # Ensure that weight constraint is also satisified
            if level == 'Sector':
                if name == 'ALL':
                    sector_weights = positions.groupby('Sector')['Weight'].sum()
                    print(f"Sector Weights: {sector_weights.to_dict()}")
                    print(f"Constraints met: {all(lb <= weight <= ub for weight in sector_weights)}")
                else:
                    sector_weight = positions[positions['Sector'] == name]['Weight'].sum()
                    print(f"{name} Sector Weight: {sector_weight:.4f} (should be between {lb} and {ub})")
                    print(f"Constraint met: {lb <= sector_weight <= ub}")
    
    print("\n")
    return positions

def load_test_cases() -> Dict[int, List[Dict[str, Any]]]:
    """
    Returns all test cases in format of our testing doc, modify this as needed for different metrics to be testsed
    """
    test_cases = {
        1: [
            {'Metric': 'WeightIfSelected', 'Level': 'Security', 'Name': 'ALL', 'Lowerbound': 0.01, 'Upperbound': 1},
            {'Metric': 'Cardinality', 'Level': 'Portfolio', 'Name': 'None', 'Lowerbound': 5, 'Upperbound': 10}
        ],
        2: [
            {'Metric': 'WeightIfSelected', 'Level': 'Security', 'Name': 'ALL', 'Lowerbound': 0.01, 'Upperbound': 1},
            {'Metric': 'Cardinality', 'Level': 'Sector', 'Name': 'ALL', 'Lowerbound': 1, 'Upperbound': 3}
        ],
        3: [
            {'Metric': 'WeightIfSelected', 'Level': 'Security', 'Name': 'ALL', 'Lowerbound': 0.01, 'Upperbound': 1},
            {'Metric': 'Cardinality', 'Level': 'Portfolio', 'Name': 'None', 'Lowerbound': 8, 'Upperbound': 15},
            {'Metric': 'Cardinality', 'Level': 'Sector', 'Name': 'ALL', 'Lowerbound': 1, 'Upperbound': 4}
        ],
        4: [
            {'Metric': 'WeightIfSelected', 'Level': 'Security', 'Name': 'ALL', 'Lowerbound': 0.01, 'Upperbound': 1},
            {'Metric': 'Cardinality', 'Level': 'Sector', 'Name': 'Technology', 'Lowerbound': 2, 'Upperbound': 4},
            {'Metric': 'Weight', 'Level': 'Sector', 'Name': 'Technology', 'Lowerbound': 0.1, 'Upperbound': 0.25}
        ],
        5: [
            {'Metric': 'WeightIfSelected', 'Level': 'Security', 'Name': 'ALL', 'Lowerbound': 0.01, 'Upperbound': 1},
            {'Metric': 'Cardinality', 'Level': 'Portfolio', 'Name': 'None', 'Lowerbound': 10, 'Upperbound': 20},
            {'Metric': 'Weight', 'Level': 'Sector', 'Name': 'ALL', 'Lowerbound': 0.05, 'Upperbound': 0.3}
        ],
        6: [
            {'Metric': 'WeightIfSelected', 'Level': 'Security', 'Name': 'ALL', 'Lowerbound': 0.01, 'Upperbound': 1},
            {'Metric': 'Cardinality', 'Level': 'Sector', 'Name': 'ALL', 'Lowerbound': 2, 'Upperbound': 5},
            {'Metric': 'Weight', 'Level': 'Issuer', 'Name': 'ALL', 'Lowerbound': 0, 'Upperbound': 0.1}
        ],
        7: [
            {'Metric': 'WeightIfSelected', 'Level': 'Security', 'Name': 'ALL', 'Lowerbound': 0.01, 'Upperbound': 1},
            {'Metric': 'Cardinality', 'Level': 'Portfolio', 'Name': 'None', 'Lowerbound': 15, 'Upperbound': 25},
            {'Metric': 'Cardinality', 'Level': 'Sector', 'Name': 'ALL', 'Lowerbound': 2, 'Upperbound': 8},
            {'Metric': 'Weight', 'Level': 'Sector', 'Name': 'ALL', 'Lowerbound': 0.05, 'Upperbound': 0.2}
        ],
        8: [
            {'Metric': 'WeightIfSelected', 'Level': 'Security', 'Name': 'ALL', 'Lowerbound': 0.02, 'Upperbound': 1},
            {'Metric': 'Cardinality', 'Level': 'Sector', 'Name': 'Consumer Goods', 'Lowerbound': 3, 'Upperbound': 5},
            {'Metric': 'Cardinality', 'Level': 'Sector', 'Name': 'Technology', 'Lowerbound': 2, 'Upperbound': 4}
        ],
        9: [
            {'Metric': 'WeightIfSelected', 'Level': 'Security', 'Name': 'ALL', 'Lowerbound': 0.01, 'Upperbound': 1},
            {'Metric': 'Cardinality', 'Level': 'Portfolio', 'Name': 'None', 'Lowerbound': 20, 'Upperbound': 30},
            {'Metric': 'Cardinality', 'Level': 'Sector', 'Name': 'ALL', 'Lowerbound': 3, 'Upperbound': 7},
            {'Metric': 'Duration', 'Level': 'Portfolio', 'Name': 'None', 'Lowerbound': None, 'Upperbound': 5}
        ]

        # Add or remove tests as needed here

    }
    return test_cases

def run_all_tests():
    """Runs all test cases and returns results"""
    test_cases = load_test_cases()
    results = {}
    
    for test_num in sorted(test_cases.keys()):  # Just to ensure that all test cases in loading function are outputted, had bug earlier
        print(f"\n{'='*20} Executing test {test_num} {'='*20}")
        constraints = test_cases[test_num]
        
        # Print test configuration
        # Provides details for each indivudal tests - refer to this if any test cases fail
        print("\nTest Constraints:")
        for c in constraints:
            print(f"  {c['Metric']:15} | {c['Level']:10} | {c['Name']:15} | LB: {c['Lowerbound']} | UB: {c['Upperbound']}")
        
        try:
            print("\nOptimizing portfolio")
            result = run_test_case(constraints)
            if result is not None:
                print("\nResults")
                analyzed_result = analyze_results(result, test_num, constraints)
                results[test_num] = analyzed_result
                print(f"\nTest {test_num} completed successfully")
            else:
                print(f"\nTest {test_num} failed - returned none")
        except Exception as e:
            print(f"\nError in Test Number {test_num}:")
            print(f"Error details: {str(e)}")
            results[test_num] = None
        
        print(f"{'='*60}\n")
    
    # Print summary of all tests at bottom
    print("\nTest Execution Summary:")
    print("-" * 30)
    for test_num in sorted(results.keys()):
        status = "Passed" if results[test_num] is not None else "Failed"
        print(f"Test {test_num}: {status}")
    
    return results

if __name__ == "__main__":
    print("Running Cardinality Constraint Tests")
    print("=" * 50)
    results = run_all_tests()