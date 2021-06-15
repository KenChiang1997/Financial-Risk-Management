import numpy as np 
import pandas as pd 
import datetime as dt 
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr 


def get_price_df(ticker,start,end):
    """
    ticker -- > list 
    start , end --> datetime 
    """
    output_df = pd.DataFrame()
    for stock in ticker :
        df = pdr.get_data_yahoo(stock , start , end)
        price_data = df["Adj Close"]
        output_df[str(stock)] = price_data
    return output_df

def calculation(df,ticker,window):
    u=[]
    var=[]
    drift=[]
    returns_df=df.iloc[::-1]
    returns_df=returns_df.reset_index()
    a=pd.Series(returns_df[str(ticker)]).rolling(window=window)
    for i in a :
        if(i.shape[0]==window):
            log_returns =np.log( 1+i.pct_change().dropna() )
            u .append(  log_returns.mean() )
            var .append( log_returns.var() )
            drift .append( log_returns.mean() - 0.5*( log_returns.var() ) )
        else:
            pass
    u   =  u[::-1]
    var =  var[::-1]
    drift =  drift[::-1]
    return u,var,drift

def MC_simulation_df(df,trial,u,var,drift,ticker,window):
    output_df=pd.DataFrame(index=df.index[window:])
    for i in range(trial) :
        price_path=[ df[str(ticker)][0] ]
        st =df[str(ticker)][0]
        for j in range ( len(u)-2 ) :
            st=st*np.exp( (u[j]-drift[j]) * np.sqrt(1) + np.sqrt(var[j]) * np.sqrt(1) * np.random.normal(0,1))
            price_path.append(st)
        output_df[str(i)]=price_path
    return output_df

def plot_MC_figure(output_df,ticker):
    plt.figure()
    plt.title("MC simulation for "+str(ticker))
    for i in range(output_df.shape[1]):
        plt.plot(output_df[str(i)].values)
    plt.show()

def compute_var(df,output_df,confidence_level,window):
    returns = output_df.pct_change().dropna()
    type_1 = 1 - confidence_level
    var_list=[]
    for i in range(returns.shape[0]):
        daliy_return = np.array( returns.iloc[i])
        var = np.quantile(daliy_return,type_1)
        var_list.append(var)
    df = df[window:]
    df['returns'] = df.pct_change()
    df = df.dropna()
    df = df.reset_index()
    df['var'] = var_list
    return  df

def hit(returns,var):
    if returns < var :
        return 1
    else : 
        return 0

def back_test(var_df_confidence,confidence_level):
    type_1=1-confidence_level
    observation = var_df_confidence.shape[0]
    Exception_in_theory=type_1 * observation
    real_exceptions = var_df_confidence['hit'].sum()

    output_df = pd.DataFrame({
        'confidence_level' : confidence_level,
        'observations' : observation , 
        'exceptions in theory' : Exception_in_theory,
        'real exceptions ' : real_exceptions
    },index=["back test"])
    
    return output_df.transpose()

#------------------------------------------ get data-----------------------------------------------------------------------
ticker='aapl'
start = dt.datetime(2011,3,22)
end = dt.datetime(2021,3,22)
df = get_price_df([ticker],start,end)
#------------------------------------------ calculate annual parametes ----------------------------------------------------------------------
u,var,drift = calculation(df,ticker,window=504)
#------------------------------------------ MC simulation ----------------------------------------------------------------------
output_df = MC_simulation_df(df,trial=10000,u=u,var=var,drift=drift,ticker=ticker,window=504)
print("-----------------------------------------  MC simulation DataFrame  -----------------------------------------\n")
# # #------------------------------------------ plot figure ----------------------------------------------------------------------
plot_MC_figure(output_df,ticker)
# # #------------------------------------------  compute var ----------------------------------------------------------------------
var_df_99 = compute_var(df,output_df,confidence_level=0.99,window=504)
var_df_95 = compute_var(df,output_df,confidence_level=0.95,window=504)
print("-----------------------------------------  VaR , confidence level = 0.99  ------------------------------------\n")
print(var_df_99)
print("-----------------------------------------  VaR , confidence level = 0.95  ------------------------------------\n")
print(var_df_95)
# #------------------------------------------  hit  ----------------------------------------------------------------------
var_df_99['hit']=var_df_99.apply(lambda x : hit(x['returns'],x['var']),axis=1)
var_df_95['hit']=var_df_95.apply(lambda x : hit(x['returns'],x['var']),axis=1)
print("-----------------------------------------  hit indicator , confidence level = 0.99  ---------------------------\n")
print(var_df_99)
print("-----------------------------------------  hit indicator , confidence level = 0.95  ---------------------------\n")
print(var_df_95)
#------------------------------------------  back_test  ----------------------------------------------------------------------
back_test_99 = back_test(var_df_99,confidence_level=0.99)
back_test_95 = back_test(var_df_95,confidence_level=0.95)
print("-----------------------------------------  backtest , confidence level = 0.99  ---------------------------\n")
print(back_test_99)
print("-----------------------------------------  backtest , confidence level = 0.95  ---------------------------\n")
print(back_test_95)





