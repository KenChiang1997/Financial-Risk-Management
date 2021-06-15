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

def history_simulation_method(returns_df,ticker,window,confidence_level):
    returns_df=returns_df.iloc[::-1]
    returns_df=returns_df.reset_index()
    type_1=1-confidence_level
    a=pd.Series( returns_df[str(ticker)] ) .rolling(window=window)
    var_list=[]

    for i in a :
        if i.shape[0] == window :
            returns = np.array( i.to_list() )
            var = np.quantile(returns,type_1)
            var_list.append(var)
        else:
            pass

    var = pd.DataFrame(var_list,columns=['var'])
    returns_df = pd.concat([returns_df,var],axis=1)
    returns_df=returns_df.iloc[::-1]
    return returns_df
    
def plot_figure(var_list_1,var_list_2,window,ticker):
    plt.figure()
    plt.title('daliy return and var for '+str(ticker))
    plt.plot(var_list_1[str(ticker)].values[int(window):],label='daliy return')
    plt.plot(var_list_1['var'],color='red',label='99 percent confidence level')
    plt.plot(var_list_2['var'],color='green',label='95 percent confidence level')
    plt.legend()
    plt.show()

def back_test(var,ticker,window,confidence_level):
    observation = var.shape[0]-window
    Exception_in_theory = observation * (1-confidence_level) 
    var_return = var[str(ticker)][window:]
    var_var = var['var'][window:]
    z=0
    for i in range(var_return.shape[0]):
        returns = var_return[i]
        var = var_var[i]
        if returns < var :
            z+=1 
        else:
            pass
    output_df = pd.DataFrame({
        'confidence_level' : confidence_level,
        'observations' : observation , 
        'exceptions in theory' : Exception_in_theory,
        'real exceptions ' : z
    },index=["back test"])
    return output_df.transpose()

def hit(var_list,ticker):
    var_list['hit']=0
    for i in range(var_list.shape[0]):
        try:
            if var_list[str(sticker)][i] < var_list['var'][i]:
                var_list['hit'][i]==1
            else:
                pass
        except:
            pass
    return var_list
#------------------------------------------ get data-----------------------------------------------------------------------
ticker='tsla'
start = dt.datetime(2011,3,22)
end = dt.datetime(2021,3,22)
df = get_price_df([str(ticker)],start,end)
aapl_returns = df[str(ticker)].pct_change().dropna()
#------------------------------------------ compute var by simulation method------------------------------------------------
var_list_1 = history_simulation_method(aapl_returns,ticker,window=504,confidence_level=0.99)
var_list_2 = history_simulation_method(aapl_returns,ticker,window=504,confidence_level=0.95)
print("----------------------------------------- Var For confidence level = 0.99  -----------------------------------------------\n")
print(var_list_1)
print("----------------------------------------- Var For confidence level = 0.95  -----------------------------------------------\n")
print(var_list_2)
#------------------------------------------ hit indicator -----------------------------------------------------------------------
hit_1 = hit(var_list_1,ticker)
hit_2 = hit(var_list_2,ticker)
print("----------------------------------------- hit indicator confidence level = 0.99  -----------------------------------------------\n")
print(hit_1)
print("----------------------------------------- hit indicator confidence level = 0.95  -----------------------------------------------\n")

print(hit_2)
#------------------------------------------ back test -----------------------------------------------------------------------
result_1=back_test(var_list_1,ticker,window=504,confidence_level=0.99)
result_2=back_test(var_list_2,ticker,window=504,confidence_level=0.95)
print("----------------------------------------- backtest confidence level = 0.99  -----------------------------------------------\n")
print(result_1)
print("----------------------------------------- backtest confidence level = 0.95  -----------------------------------------------\n")
print(result_2)
#------------------------------------------ plot figure -----------------------------------------------------------------------
plot_figure(var_list_1,var_list_2,window=504,ticker=ticker)
