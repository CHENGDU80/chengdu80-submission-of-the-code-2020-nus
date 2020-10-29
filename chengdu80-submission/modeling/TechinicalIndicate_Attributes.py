import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('transaction_data_processed.csv')
df2 = df
#spread and volume
df2['spread'] = df2['ASKHI'] - df2['BIDLO']

df2['volume%'] = df2['VOL']/df2['SHROUT']/1000

#momentum
def STOK(close, low, high, n):
    STOK = ((close - low.rolling(window=n).min()) / (high.rolling(window=n).max() - low.rolling(window=n).min()))
    return STOK
df4 = pd.DataFrame()
ticker =df2.TICKER.unique()
for i in ticker:
    df3 = df2[df2['TICKER']==i]
    df3['%K'] = STOK(df3['PRC'], df3['BIDLO'], df3['ASKHI'], 14)
    df4 = pd.concat([df4,df3])

# MACD
df5 = pd.DataFrame()
for i in ticker:
    df3 = df4[df4['TICKER'] == i]
    df3['exp1'] = df3.PRC.ewm(span=12, adjust=False).mean()
    df3['exp2'] = df3.PRC.ewm(span=26, adjust=False).mean()
    df3['macd'] = df3['exp1'] - df3['exp2']
    df3['signal'] = df3.macd.ewm(span=9, adjust=False).mean()
    df3['macd-1'] = df3['macd'].shift(periods=1)
    df3['signal-1'] = df3['signal'].shift(periods=1)
    df3['cross'] = df3.apply(lambda x: 1 if (x['macd-1'] < x['signal-1']) & (x['macd'] > x['signal'])
    else -1 if (x['macd-1'] > x['signal-1']) & (x['macd'] < x['signal'])
    else 0, axis=1)

    df5 = pd.concat([df5, df3])

