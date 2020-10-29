import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Import enriched data
enriched_df = pd.read_csv("../../data/processed/enriched_data.csv")[['date', 'TICKER', 'spread', 'volume%', '%K', 'cross', 'PRC']]
enriched_df.columns = ['date', 'TICKER', 'spread', 'vol', 'k', 'cross', 'price']
techanalysis = pd.read_csv("../../data/processed/enriched_data.csv")[['date', 'TICKER', 'macd','signal']]
techanalysis = techanalysis.set_index(pd.to_datetime(techanalysis['date'])).iloc[:, 1:]

len(enriched_df)

# Import sentiment data
sentiment_df = pd.read_csv("../../data/processed/sentiments.csv")
len(sentiment_df)

def make_dataset(df, sentiment_df):
    
    df.drop(columns=["TICKER"], inplace=True)
    df = df.set_index(pd.to_datetime(df['date'])).iloc[:, 1:]
    df.fillna(0, inplace = True)
    
    # Train
    df['spread1'] = df['spread'].shift(periods=1)
    df['spread2'] = df['spread'].shift(periods=2)
    df['spread3'] = df['spread'].shift(periods=3)
    
    df['vol1'] = df['vol'].shift(periods=1)
    df['vol2'] = df['vol'].shift(periods=2)
    df['vol3'] = df['vol'].shift(periods=3)
    
    df['k1'] = df['k'].shift(periods=1)
    df['k2'] = df['k'].shift(periods=2)
    df['k3'] = df['k'].shift(periods=3)
    
    df['cross1'] = df['cross'].shift(periods=1)
    df['cross2'] = df['cross'].shift(periods=2)
    df['cross3'] = df['cross'].shift(periods=3)
    
    df['price1'] = df['price'].shift(periods=1)
    df['price2'] = df['price'].shift(periods=2)
    df['price3'] = df['price'].shift(periods=3)
    
    # Test
    target_df = df[['price']].copy()
    target_df.columns = ['price4']

    target_df['price5'] = target_df['price4'].shift(periods=-1)
    target_df['price6'] = target_df['price4'].shift(periods=-2)
    
    # Drop past data nulls
    X_temp_1 = df.drop(df.head(3).index).drop(columns=["spread", "vol", "k", "cross", "price"])
    y = target_df.drop(target_df.head(3).index)
    
    # Drop future data null
    X_temp_2 = X_temp_1.drop(X_temp_1.tail(2).index)
    y = y.drop(y.tail(2).index)
    
    sentiment_df = sentiment_df.set_index(pd.to_datetime(sentiment_df['date'])).iloc[:, 1:]
    sentiment_avg = sentiment_df.rolling(window=3).mean().shift(periods=1).drop(sentiment_df.head(3).index)
    sentiment_avg.dropna(inplace=True)
    
    X = pd.merge(X_temp_2, sentiment_avg, left_index=True, right_index=True, how="left")
    full_X = pd.merge(X_temp_1, sentiment_avg, left_index=True, right_index=True, how="left")
    
#     print("Head")
#     print(X.head())
#     print(y.head())
    
#     print("Tail")
#     print(X.tail())
#     print(y.tail())
    
    return X, y, full_X

def get_preds(X, y):
    
    X_train, y_train = X.loc[:"2012-10"].copy(), y.loc[:"2012-10"].copy()
    X_test, y_test = X.loc["2012-11":].copy(), y.loc["2012-11":].copy()
    
    seed = 42

    rf = RandomForestRegressor(random_state=seed, oob_score=True, n_estimators=200, n_jobs=-1)
    rf.fit(X_train, y_train)

    preds = rf.predict(X_test)
    preds_df = pd.DataFrame(preds, columns=y_test.columns, index=y_test.index)
    
    rmse = mean_squared_error(y_test, preds, squared=False)
    
    return rmse, preds_df

def get_total_preds(X, y, full_X):
    
    seed = 42

    rf = RandomForestRegressor(random_state=seed, oob_score=True, n_estimators=100, n_jobs=-1)
    rf.fit(X.loc[:'2012-10'], y.loc[:'2012-10'])

    preds = rf.predict(full_X)
    preds_df = pd.DataFrame(preds, columns=y.columns, index=full_X.index)
    
    return preds_df

pd.options.mode.chained_assignment = None

sentiment_df = pd.read_csv("../../data/processed/sentiments.csv")
ticker_names = enriched_df.loc[enriched_df["date"] >= "2012-11-01"]["TICKER"].unique()
preds_dict = dict()
rmse_dict = dict()

for idx, ticker in enumerate(ticker_names):
    
    data = enriched_df.loc[enriched_df["TICKER"] == ticker]
    X, y = make_dataset(data, sentiment_df)
    rmse, preds = get_preds(X, y)

    preds_dict[ticker] = preds
    rmse_dict[ticker] = rmse
        
    if idx % 25 == 0:
        print("Completed:", idx)

# Get total preds for 2012 + 3 days
ticker = 'AAPL'
data = enriched_df.loc[enriched_df["TICKER"] == ticker]
X, y, full_X = make_dataset(data, sentiment_df)

preds = get_total_preds(X, y, full_X)

new_index = list(pd.DatetimeIndex(['2013-01-01', '2013-01-02']))

ori_index = list(preds['price4'].index)
ori_index.extend(new_index)

new_index = pd.DatetimeIndex(ori_index)

new_preds = preds['price4'].reindex(index=new_index)

new_vals = list(preds.iloc[-1][['price5', 'price6']].T)

new_preds[-2] = new_vals[0]
new_preds[-1] = new_vals[1]

new_y = y.iloc[-1][['price5', 'price6']]
actual_y = y.reindex(index=new_preds.index)['price4']
actual_y[-4] = new_y[0]
actual_y[-3] = new_y[1]

aapl_df = pd.DataFrame({"price": actual_y, "forecast": new_preds})

aapl_df.to_csv("../../data/processed/aapl.csv", index=True)

def check_ticker(ticker):
    
    import matplotlib.pyplot as plt
    
    
    test = base_df.loc["2012-11":][['TICKER', 'price']]
    support = techanalysis.loc['2012-11':]
    
    fig, ax1 = plt.subplots(figsize=(25,20))

    ax2 = ax1.twinx()

    ax1.plot(test.loc[test['TICKER'] == ticker]['price'])
#     ax1.plot(preds_dict[ticker]['price4'])
    ax1.plot(preds['price4'])
#     ax2.plot(support.loc[support['TICKER']==ticker]['macd'], color = 'g', linestyle = '--')
#     ax2.plot(support.loc[support['TICKER']==ticker]['signal'], color = 'y', linestyle='--')

    plt.show()
    
    return

check_ticker("AAPL")

from sklearn.metrics import r2_score
test = base_df["2012-11":'2012-12-27'][['TICKER', 'price']]
#print(test.loc[test['TICKER']=='AAPL']['price'])
#print(preds_dict['AAPL']['price4'])

r2_score(preds_dict['AAPL']['price4'],test.loc[test['TICKER']=='AAPL']['price'])