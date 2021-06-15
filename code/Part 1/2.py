import numpy as np 
import pandas as pd 
import datetime as dt
from scipy import stats
import statsmodels.api as sm
import matplotlib.pyplot as plt 
from pandas_datareader import data as pdr 

#------------------------------------------------built function-------------------------------------------------------
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

def compute_beta(portfolio_df,market_df):
    portfolio_returns = portfolio_df.pct_change().dropna()
    market_returns = market_df.pct_change().dropna()
    market_returns = sm.add_constant(market_returns)
    model=sm.OLS(portfolio_returns,market_returns)
    res=model.fit()
    return res.params.values[1]

def compute_all_beta(portfolio_df,market_df):
    beta_list=[]
    assets = portfolio_df.columns
    for i in range(portfolio_df.shape[1]):
        beta = compute_beta(portfolio_df=portfolio_df[str(assets[i])],market_df=market_df)
        beta_list.append(beta)
    df = pd.DataFrame()
    df["Beta"] = beta_list 
    df.index = assets 
    return df.values

def compute_residual(asset_df,market_df,beta):
    asset_return = asset_df.pct_change().dropna()
    market_return = market_df.pct_change().dropna()
    asset_return_var = np.array(asset_return.var())
    market_return_var = np.array(market_return.var())
    return float(asset_return_var - beta**2*market_return_var )

def compute_residual_matrix(market_df,portfolio_df,beta):
    cols = portfolio_df.columns
    residual=[]
    for i in range(portfolio_df.shape[1]):
        res=compute_residual(asset_df=portfolio_df[str(cols[i])],market_df=market_df,beta=beta[i])
        residual.append( abs(res) )
    residual = np.diag(residual)  
    return residual

# VaR Function
def Portfolio_FC_VaR(df,confidence_level,delta_t,initial_weights):
    """
    confidence level --> Type I error 
    delta T          --> duration to matruratity 
    sigma            --> Portfolio return standard deviation
    corr             --> Portfolio return correlation matrix 
    """
    returns = df.pct_change().dropna()
    returns_corr = returns.corr()
    returns_std = returns.std()

    weights = initial_weights * returns_std  ## Dality Dollars volatility 
    n = stats.norm.ppf(confidence_level)
    time = np.sqrt(delta_t)

    k = np.sqrt( np.dot(np.dot(weights.transpose(),returns_corr),weights) )
    VaR = n*k*time
    print("Full Covariance Model : For over %i days , Under %.3f Confidence Level , VaR for this Portfolio is %.3f \n" %(delta_t ,confidence_level , VaR) )
    return VaR

def Assets_FC_VaR(df,asset,confidence_level,delta_t,initial_investment):
    """
    confidence level --> Type I error 
    delta T          --> duration to matruratity 
    sigma            --> Asset return standard deviation
    """
    df = df[str(asset)]

    returns = df.pct_change().dropna()
    sigma = returns.std() * initial_investment  # Daliy Dollars Volatility
    n = stats.norm.ppf(confidence_level)

    print(" For over %i days , Under %.3f Confidence Level , VaR for %s is %.3f " %( delta_t , confidence_level , str(asset) ,n*sigma*np.sqrt(delta_t)) )
    return  n*sigma*np.sqrt(delta_t) 

def Portfolio_Beta_VaR(beta,initial_weights,confidence_level,delta_t,market_df,portfolio_df):

    market_returns = market_df.pct_change().dropna()
    market_std = market_returns.std() 
    weights = np.array(initial_weights)

    portfolio_variance = np.dot(np.dot(np.dot(weights.transpose(),beta),beta.transpose()),weights) * market_std**2
    portfolio_std = np.sqrt(portfolio_variance)
    n = stats.norm.ppf(confidence_level)
    time = np.sqrt(delta_t)

    VaR = n*portfolio_std*time
    print("Beta Model : For over %i days , Under %.3f Confidence Level , VaR for this Portfolio is %.3f \n" %(delta_t , confidence_level , VaR) )
    return VaR

def Portfolio_Diaganol_VaR(beta,DM,initial_weights,confidence_level,delta_t,market_df,portfolio_df):
    market_returns = market_df.pct_change().dropna()
    market_std = market_returns.std() 
    weights = np.array(initial_weights)

    portfolio_variance = np.dot(np.dot(np.dot(weights.transpose(),beta),beta.transpose()),weights) * market_std**2
    portfolio_std = np.sqrt(portfolio_variance)
    n = stats.norm.ppf(confidence_level)
    time = np.sqrt(delta_t)

    VaR = n*portfolio_std*time + np.dot(np.dot(weights.transpose(),DM),weights)
    print("Diaganol Model : For over %i days , Under %.3f Confidence Level , VaR for this Portfolio is %.3f \n" %(delta_t ,confidence_level , VaR) )
    return VaR

def Portfolio_Undiversified_VaR(df,confidence_level,delta_t,investment_weight):
    VaR=0
    cols =  df.columns
    for i in range(df.shape[1]):
        k=Assets_FC_VaR(df,asset=str(cols[i]),confidence_level=confidence_level,delta_t=delta_t,initial_investment=investment_weight[i])
        VaR+=k
    print("Undiversified Model : For over %i days , Under %.3f Confidence Level , VaR for this Portfolio is %.3f \n" %( delta_t ,confidence_level , VaR) )
    return VaR
#------------------------------------------parameters setting -----------------------------------------------------------------
start = dt.datetime(2011,3,23)
end = dt.datetime(2021,3,22)
weight=[20,20,20,20,20]
ticker = ["aapl",'msft','sbux','mcd','tsla']
stock_market=['^GSPC']
#--------------------------------------call function  ------------------------------------------------------------------------
print("----------------------------------------- history stock price -----------------------------------------------\n")
portfolio_df = get_price_df(ticker,start,end)
market_df = get_price_df(stock_market,start,end)
beta = compute_all_beta(portfolio_df=portfolio_df,market_df=market_df)
print(portfolio_df)
# -------------------------------------Full Covariance VaR ---------------------------------------------------------------------

print("-----------------------------------------Full Covariance Model----------------------------------------------\n")

VaR_99 = Portfolio_FC_VaR(portfolio_df,initial_weights=weight,confidence_level=0.95,delta_t=5)
# --------------------------------------Diaganol Model VaR-------------------------------------------------------------------------------
print("-----------------------------------------Diaganol Model Model-----------------------------------------------\n")

DM=compute_residual_matrix(market_df=market_df,portfolio_df=portfolio_df,beta=beta)
Diaganol_VaR =Portfolio_Diaganol_VaR(beta=beta,DM=DM,initial_weights=weight,confidence_level=0.95,delta_t=5,market_df=market_df,portfolio_df=portfolio_df)
# --------------------------------------Beta Model VaR-------------------------------------------------------------------------------

print("------------------------------------------Beta Model--------------------------------------------------------\n")

Portfolio_Beta_VaR(beta=beta,initial_weights=weight,confidence_level=0.95,delta_t=5,market_df=market_df,portfolio_df=portfolio_df)
# --------------------------------------Undiversified Model VaR-------------------------------------------------------------------------------

print("------------------------------------------Undiversified Model-----------------------------------------------\n")

Portfolio_Undiversified_VaR(portfolio_df,confidence_level=0.95,delta_t=5,investment_weight=weight)
