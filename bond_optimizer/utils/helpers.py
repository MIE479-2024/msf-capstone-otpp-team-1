from datetime import datetime
import pandas as pd

# Default t_cost function
def compute_spread_t_cost(data):
    '''Default function for computing t-cost coefficient. It is the spread/midpoint'''
    data['T_COST_COEFF'] = 100* (data['ASK_PRICE'] - data['BID_PRICE']) / ((data['ASK_PRICE'] + data['BID_PRICE'])/2)
    data['T_COST_COEFF'] = data['T_COST_COEFF'].clip(lower=0)
    data['T_COST_COEFF'] = data['T_COST_COEFF'].fillna(0) # removes NaN for cash
    return data

# Filter dataset methods
def filter_data_by_sector(data, category: str):
    data = data.loc[data['SECTOR'] == category].copy(deep=True)
    return data

def filter_data_by_liquidity(data, liquidity: int):
    data = data.loc[data['LIQUIDITY_SCORE'] >= liquidity].copy(deep=True)
    return data

def filter_data_by_negatives(data):
    data = data.loc[data['EXPECTED_RETURN'] >= 0].copy(deep=True)
    return data

def filter_dataset(dataset, scope, bucket):
    '''Helper function for forming constraints'''
    match scope:
        case 'Portfolio':
            return dataset
        case 'Sector':
            return dataset.loc[dataset['SECTOR'] == bucket]
        case 'Issuer':
            return dataset.loc[dataset['ISSUER'] == bucket]
        case 'Maturity':
            return dataset.loc[dataset['MATURITY_SCORE'] == bucket]
        case 'Liquidity':
            return dataset.loc[dataset['LIQUIDITY_SCORE'] == bucket]
        case 'Security':
            return dataset.loc[dataset['ISIN'] == bucket]

# Scope name mapping
def scope_map(scope):
    '''Helper function for mapping scope'''
    match scope:
        case 'Portfolio':
            return 'PORTFOLIO'
        case 'Sector':
            return 'SECTOR'
        case 'Issuer':
            return 'ISSUER'
        case 'Maturity':
            return 'MATURITY_SCORE'
        case 'Liquidity':
            return'LIQUIDITY_SCORE'
        case 'Security':
            return 'ISIN'

# Param mapping
def get_param_row(key, value):
    '''Helper function for mapping params'''
    match key:
        case 'date':
            date_obj = datetime.strptime(value, "%Y-%m-%d")
            return 'input_date', date_obj
        case 'transaction_cost_limit':
            return 'transaction_cost_limit', float(value)
        case 'robust':
            if value == 'TRUE':
                return 'robust', True 
            return 'robust', False
        case 'allow_constraint_violation':
            if value == 'TRUE':
                return 'allow_constraint_violation', True 
            return 'allow_constraint_violation', False
        case 'enable_soft_constraints':
            if value == 'TRUE':
                return 'enable_soft_constraints', True 
            return 'enable_soft_constraints', False

def get_sample_prev_data(raw_data, y, m, d):
    '''Helper function to set previous day’s data as simply an equally weighted portfolio.'''
    temp = raw_data.loc[raw_data['DATE'] == datetime(y, m, d)]
    temp = temp[['ISIN']]
    if len(temp) != 0:
        temp['PREV_WEIGHT'] = 1/len(temp)
    # if nothing on previous day... I guess we set 0 weights
    else:
        temp['PREV_WEIGHT'] = 0
    return temp