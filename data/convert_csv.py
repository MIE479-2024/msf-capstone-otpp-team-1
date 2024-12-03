import pandas as pd
import random

'''Converts data into a csv as expected by the library'''

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


# Step 1: Read in the CSV file
input_file = 'bonds_w_exp_returns.csv'
df = pd.read_csv(input_file)


# rename certain columns
df.rename(columns={
    'SecurityId': 'ISIN',
    'Maturity': 'maturity',
    'Coupon': 'coupon',
    'LiquidityScore': 'liquidity_score',
    'ExpectedReturn': 'expected_return',
    'StdDev': 'return_std_dev',
}, inplace=True)

# add existing columns via transformations
df['OAS'] = df['spread'] * df['ModifiedDuration']
df['bid_price'] = df['CleanPrice'] - df['BidAskSpread']/2
df['ask_price'] = df['CleanPrice'] + df['BidAskSpread']/2

# add duration
df['maturity'] = pd.to_datetime(df['maturity'])
df['Date'] = pd.to_datetime(df['Date'])
def calculate_duration(start, end):
    return (end - start).days / 365
df['duration'] = df.apply(lambda row: calculate_duration(row['Date'], row['maturity']), axis=1)

# randomly assign issuer and sector
issuers = ['Company1', 'Company2', 'Company3', 'Company4', 'Company5',
           'Company6', 'Company7', 'Company8', 'Company9', 'Company10']
sectors = ['Sector1', 'Sector2', 'Sector3', 'Sector4', 'Sector5']

# Add Issuer and Sector columns with random values
df['issuer'] = [random.choice(issuers) for _ in range(len(df))]
df['sector'] = [random.choice(sectors) for _ in range(len(df))]


# - Drop unneeded columns
df.drop(columns=['spread', 'CleanPrice', 'AccruedInterest', 'DirtyPrice', 'ModifiedDuration', 'Rating', 'BidAskSpread', ], errors='ignore', inplace=True)


output_file = 'bonds_w_exp_returns_new.csv'
df.to_csv(output_file, index=False)
