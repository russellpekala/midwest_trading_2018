from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
'''
File:
    sample_strategy.py

Description:
    A sample implementation of a portfolio construction strategy
    that would qualify as a submission to the contest.

Questions:
    Please contact jigar@uchicago.edu or vtalasani@uchicago.edu
'''

from portfolio import PortfolioGenerator
import pandas as pd
import numpy as np
#import statsmodels.api as sm

class SampleStrategy(PortfolioGenerator):

    def __init__(self):
        pass

    def build_signal(self, stock_features):

        LR = LinearRegression()

        returns = stock_features.groupby(['ticker'])['returns'].mean()
        returnStd = stock_features.groupby(['ticker'])['returns'].std()

        #print returns

        #stock_features['momentumm'] = self.momentum(stock_features)
        #X = stock_features[['momentum', 'value']]

        f1 = self.value(stock_features)
        f2 = self.size(stock_features)
        f3 = self.vix(stock_features)
        sharpe_ratio=returns/returnStd
        X_train = np.column_stack((f1,f2,f3))
        # print len(f1)
        # print len(f2)
        # print len(f3)
        # print np.shape(X_train)
        # print len(returns)

        # df = pd.DataFrame(data = {})
        # df['values'] = f1
        # df['size'] = f2
        # df['momentum'] = f3
        # X_train = df.values

        # est = sm.OLS(returns, X_train).fit()
        LR.fit(X_train,sharpe_ratio)
        
        # print LR.coefs_
        

        # return np.dot(X_train,LR.coefs_)
        return self.value(stock_features)

    def momentum(self, stock_features):
        return stock_features.groupby(['ticker'])['returns'].mean()

    def value(self, stock_features):
        return -np.log(stock_features.groupby(['ticker'])['pb'].mean())

    def size(self,stock_features):
        return stock_features.groupby(['ticker'])['market_cap'].mean()

    def vix(self,stock_features):
        return stock_features.groupby(['ticker'])['VIX'].mean()




# Test out performance by running 'python sample_strategy.py'
if __name__ == "__main__":
    portfolio = SampleStrategy()
    sharpe = portfolio.simulate_portfolio()
    print("*** Strategy Sharpe is {} ***".format(sharpe))