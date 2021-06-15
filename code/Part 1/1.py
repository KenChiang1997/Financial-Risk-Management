import numpy as np 
import pandas as pd 
import datetime as dt 
from scipy import stats
import matplotlib.pyplot as plt 
from pandas_datareader import data  as pdr 

#-----------------------------Python for finance Part 1 ------------------------------
def get_price_df(ticker,start,end):
    """
    ticker -- > list 
    start , end --> datetime 
    """
    output_df = pd.DataFrame()
    for stock in ticker :
        df = pdr.get_data_yahoo(stock , start , end)
        price_data = df["Adj Close"].values
        output_df[str(stock)] = price_data
    output_df = output_df.fillna(value=0)
    return output_df

def compute_stock_df_statistic(df):
    df = df.pct_change().dropna()
    ss = pd.DataFrame(columns=df.columns,index=['Mean','Standard Deviation',"Minimum",'Maximum','Skewness','Kurtosis'])
    for i in range(df.shape[1]):
        stock_values = df[str(df.columns[i])].values
        ss[str(df.columns[i])]['Mean'] = np.mean(stock_values)
        ss[str(df.columns[i])]['Standard Deviation'] = np.std(stock_values)
        ss[str(df.columns[i])]['Minimum'] = np.min(stock_values)
        ss[str(df.columns[i])]['Maximum'] = np.max(stock_values)
        ss[str(df.columns[i])]['Skewness'] = stats.skew(stock_values)
        ss[str(df.columns[i])]['Kurtosis']= stats.kurtosis(stock_values)
    return ss 

def find_frequency(returns,res_min,res_max):
    pos = np.where( (returns>= res_min) & (returns < res_max) )
    return returns[pos]

def plot_PDF(df,step,ticker):
    returns = df[str(ticker)].pct_change().dropna()
    returns  = np.sort(returns.values)
    returns_min = np.min(returns)
    returns_max = np.max(returns)
    n = (returns_max - returns_min) / step
    index = []
    frequency = []
    for _ in range(step):
        returns_min+=n
        index.append(returns_min)
        frequency.append( len(find_frequency(returns,returns_min,returns_min+n) ))

    frequency_df = pd.DataFrame()
    frequency_df['index'] = index
    frequency_df["frequency"] = frequency
    frequency_df = frequency_df.sort_values(by='index').reset_index(drop=True)

    plt.figure()
    plt.bar( x=frequency_df.index , height=frequency_df['frequency'].values, align='center', alpha=0.5 , width=0.8)
    plt.xticks(frequency_df.index ,frequency_df["index"].to_list(),rotation=-60,fontsize=7)
    plt.ylabel('Frequency')
    plt.xlabel('returns')
    plt.title(str(ticker)+"'s Return Probability Distribution")
    plt.show()

    return frequency_df
#------------------------------- data -------------------------------------
start = dt.datetime(2011,3,22)
end = dt.datetime(2021,3,22)
ticker = ["aapl",'msft','sbux','mcd','tsla']
df = get_price_df(ticker,start,end)
stats_df = compute_stock_df_statistic(df)
print("----------------------------------------- history stock price -----------------------------------------------\n")
print(df)
print("----------------------------------------- stats data --------------------------------------------------------\n")
print(stats_df)
PDF = plot_PDF(df=df,step=20,ticker='tsla')
print("----------------------------------------- frequency data -----------------------------------------------------\n")
print(PDF)









