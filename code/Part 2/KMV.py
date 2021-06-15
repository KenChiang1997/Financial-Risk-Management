import warnings
import numpy as np 
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt 
import matplotlib.dates as mdates

from scipy.stats import norm
from scipy.optimize import fsolve
warnings.filterwarnings("ignore")


class Merton_KMV():
    def __init__(self,D,E,t,r,sigma_e,x):
        self.D = D 
        self.E = E
        self.t = t
        self.r = r 
        self.sigma_e = sigma_e
        self.x = x
    def d1(self,A,D,sigma_a,r,t):
        self.d1_v = ( np.log(A/D) + ( r + sigma_a**2 /2 ) * t ) / ( sigma_a*np.sqrt(t) )
        return self.d1_v 
    def d2(self,A,D,sigma_a,r,t):
        self.d2_v = self.d1_v - sigma_a*np.sqrt(t)
        return self.d2_v 
    def func(self,x):
        sigma_a = x[0]
        a =  x[1] 
        d1_value = self.d1(A=a, D=self.D , sigma_a=sigma_a , r=self.r , t=self.t)
        d2_value = self.d2(A=a, D=self.D , sigma_a=sigma_a , r=self.r , t=self.t)
        return [ (a*norm.cdf(d1_value) - np.exp(-self.r*self.t)*self.D*norm.cdf(d2_value) - self.E) , ((a/self.E) * norm.cdf(d1_value) * sigma_a - self.sigma_e )]
    def fsolve(self):
        root = fsolve( self.func, x0=self.x)
        return root


def KMV_df(df,trading_days):
    """
    build a df that content the parameters for calculating firm asset value and sigma_a 
    """
    df['RET'] = np.log( df['RET'].values + 1)
    n=trading_days
    sigma_e = []
    df['KMV debt'] = df['DLC'] + 0.5 * df['DLTT']
    for i in range(df.shape[0]+1,n,-1):
        e_values = np.std( df['RET'][-(n-i):i].values ) * np.sqrt(n)
        sigma_e.append(e_values)

    df = df[n-1:]
    df['sigma_e'] = list(reversed(sigma_e))  

    df = df [['DATE','RET','me','ir','KMV debt','sigma_e']]
    df = df.reset_index(drop=True)
    return df 

def d2(A,D,sigma_a,r,t):
    d2_v =  ( np.log(A/D) + (r-0.5*sigma_a**2) * t ) / ( sigma_a*np.sqrt(t) )
    return d2_v

def d1(A,D,sigma_a,r,t):
    d1_v =  ( np.log(A/D) + (r+0.5*sigma_a**2) * t ) / ( sigma_a*np.sqrt(t) )
    return d1_v

def kmv(kmv_df):
    sigma_list = []
    A_list = []
    EDP_list = []
    d2_list = []
    d1_list = []
    for i in range(kmv_df.shape[0]):
        t = 1
        d = kmv_df['KMV debt'][i]
        e = kmv_df['me'][i]
        r = kmv_df['ir'][i]
        sigma_e = kmv_df['sigma_e'][i]

        model = Merton_KMV(D=d,E=e,t=t,r=r,sigma_e=sigma_e,x=[ sigma_e , e+d ])
        ans = model.fsolve()
        
        sigma_list.append(ans[0])
        A_list.append(ans[1])

        
        d2_vlaue = d2(A=ans[1],D=d, sigma_a=ans[0],r=r,t=t)
        d2_list.append(d2_vlaue)
        d1_vlaue = d1(A=ans[1],D=d, sigma_a=ans[0] ,r=r,t=t)
        d1_list.append(d1_vlaue)
        EDP_list.append( norm.cdf(-1*d2_vlaue) )
        
    kmv_df['sigma_a'] = sigma_list
    kmv_df['A'] = A_list
    kmv_df['d1'] = d1_list
    kmv_df['d2'] = d2_list
    kmv_df['EDP'] = EDP_list
    return kmv_df


# --------------------------------[ Part I: Python for finance -(1,2) ]---------------------------------------------------------------

df_1 = pd.read_csv(r'/Users/chen-lichiang/Desktop/data_20020701.csv')
df_2 = pd.read_csv(r'/Users/chen-lichiang/Desktop/data_20011001.csv')


kmv_df_1 = KMV_df(df=df_1,trading_days=df_1.shape[0])
kmv_df_1 = kmv(kmv_df=kmv_df_1)



kmv_df_2 = KMV_df(df=df_2,trading_days=df_2.shape[0])
kmv_df_2 = kmv(kmv_df=kmv_df_2)

print("----------------------------[ Part I: Python for finance -(1,2) ]----------------\n")

print("---------------------------- EDP on date :"+str(kmv_df_1['DATE'][0])+"------------------------\n")
print(kmv_df_1,"\n")
print("---------------------------- EDP on date :"+str(kmv_df_2['DATE'][0])+"------------------------\n")
print(kmv_df_2,'\n')


#--------------------------------[ Part I: Python for finance -( bonus ) ]---------------------------------------------------------------

print("----------------------------[ Part I: Python for finance -( bonus )]------------\n")

df_3 = pd.read_excel(r'/Users/chen-lichiang/Desktop/data_bonus.xls')


kmv_df = KMV_df(df=df_3,trading_days=df_1.shape[0])
kmv_df = kmv(kmv_df=kmv_df)
kmv_df = kmv_df[:-35]
days = kmv_df['DATE']
print(kmv_df)

plt.figure()
plt.title("WorldCom : EDF")
plt.plot(days,kmv_df['EDP'])
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=70))
plt.gcf().autofmt_xdate()
plt.xlabel('time')
plt.ylabel('%(log)')
plt.grid()
plt.show()


