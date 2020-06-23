import pandas as pd
import numpy as np
import math

#for dataframes 

def normality(df, x, dist_col):
    '''
    parameter: <pandas.dataframe> df, <string> x, independent variable, <string> y dependent variable
    returns: checks the skew and variance of y for each item in the column
    '''
    result = []
    unique_list = sorted(list(df[x].unique()))
    for i in range(len(unique_list)):
        data = df[df[x]== unique_list[i]]
        skew = data[dist_col].skew()
        variance = data[dist_col].var()
        result.append((skew,variance))
    return result


def checkRange(df, x, dist_col):
    '''
    parameters:<pandas.dataframe> df, <string> x, independent variable, <string> y dependent variable
    returns: the range of skews and variances of y for each item in column 
    '''
    result = normality(df, x, dist_col)
    skew = []
    var =[]
    for item in result:
        skew.append(item[0])
        var.append(item[1])
    range_var = max(clean(var)) - min(clean(var))
    range_skew = max(clean(skew)) - min(clean(skew))
    return (range_skew, range_var)

def checkAverage(df, x, dist_col):
    '''
    parameters:<pandas.dataframe> df, <string> x, independent variable, <string> y dependent variable
    returns: average of skew and var 
    '''
    result = normality(df,x,dist_col)
    skew = []
    var =[]
    for item in result:
        skew.append(item[0])
        var.append(item[1])
    avg_var = Mean(var)
    avg_skew = Mean(skew)
    return (avg_skew, avg_var)

#for lists 

def standardDev(data):
    Sum = 0
    for i in range(len(data)):
        Sum = Sum + (data[i] - mean(data))**2
    return np.sqrt((1/len(data)) * Sum) 

def clean(data):
    clean_data = [0 if math.isnan(x) else x for x in data]
    return clean_data

def Mean(data):
    return sum(clean(data))/len(data)

def Median(data):
    if len(data) %2 ==1:
        return data[int((len(data)+1)/2)]
    else:
        return data[int(len(data)/2)] + data[int((len(data)/2) + 1)]

def pearsonSkew(data):
    return 3*(mean(data) - median(data))/standardDev(data)

def zScore(value,data):
    return (value - mean(data))/standardDev(data)

#using stats model
import statsmodels.api as sm
import sklearn.metrics as metrics
from statsmodels.formula.api import ols
import pandas as pd

def statsModelSummary(df, y):
    '''
    parameters: <pandas.dataframe> df, <string> log dependent variable y
    returns: regression summary via statsmodels 
    '''
    X = df.drop(y, axis =1)
    X_matrix = sm.add_constant(np.asarray(X))
    Y = df[y]
    model = sm.OLS(Y,X_matrix).fit()
    print('independent variables:', X.columns)
    print(model.summary())
    
#using sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import sklearn.metrics  

lm = LinearRegression()

def sklearnModelSummary(df, y, testSize):
    '''
    parameters: <pandas.dataframe> df, <string> dependent variable y
    retturns: R-squared and error metrics via sklearn after scaling X by z-score
    '''
    X = df.drop(y, axis =1)
    Y = df[y]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=testSize, random_state=101)
    lm.fit(X_train,y_train)
    coefficients = lm.coef_
    constant_term = lm.intercept_
    r_2 = metrics.r2_score(y_test, lm.predict(X_test) )
    mae = sklearn.metrics.mean_absolute_error(y_test, lm.predict(X_test) )
    mse = sklearn.metrics.mean_squared_error(y_test, lm.predict(X_test) )
    adjusted_r_2 = 1 - (1-r_2)*(len(X)-1)/(len(X)-len(X.columns) -1)  
    
    print('coefficients:', pd.DataFrame(coefficients,X.columns,columns=['Coefficient']))
    #print('variables:', list(X.columns))
    print('R-squared:', r_2)
    print('adjusted R-squared:', adjusted_r_2)
    print('Mean Absolute Error (MAE):', mae)
    print('Mean Squared Error (MSE):' , mse)
    print('Root Mean Squared Error (RMSE):', np.sqrt(mse))

#k-fold cross validation

def sklearnModel(X_train, y_train, X_test, y_test):
    '''
    parameters: <list,array or dataframe> X_train training independent vairables 
                <list,array or dataframe> y_train log training dependent variable
                <list,array or dataframe> X_test test independent variables
                <list,array or dataframe> y_test log test dependent variable
    resturns: (R-squared, Mean absolute error, Mean squared error, root-mean squared error)
    '''
    lm.fit(X_train,y_train)
    coefficients = lm.coef_
    constant_term = lm.intercept_
    r_2 = metrics.r2_score(y_test, lm.predict(X_test) )
    mae = sklearn.metrics.mean_absolute_error(y_test, lm.predict(X_test))
    mse = sklearn.metrics.mean_squared_error(y_test, lm.predict(X_test))
    adjusted_r_2 = 1 - (1-r_2)*(len(X_train)-1)/(len(X_train)-len(X_train.columns) -1)
    return (r_2,adjusted_r_2, mae, mse, np.sqrt(mse)) 

def kFoldsCrossValidation(df,k, y):
    '''
    parameters: <pandas.dataframe> df, <int> k number of folds, <string> y is dependent variable
    returns: a dataframe containing R-squared, mae, mse, rmse 
    '''
    X = df.drop(y, axis =1)
    Y = df[y]
    kX_df = kFolds(X,k)
    kY_df = kFolds(Y,k)
    result = []
    for i in range(k):
        X_test = kX_df[i]
        X_train = pd.concat([fold for n, fold in enumerate(kX_df) if n!=i]) 
        Y_test = kY_df[i]
        Y_train = pd.concat([fold for n, fold in enumerate(kY_df) if n!=i])
        result.append(sklearnModel(X_train, Y_train, X_test, Y_test))
    return pd.DataFrame(result, columns = ['r_2','adjusted_r_2', 'mae', 'mse', 'rmse'])

def KfoldsTestTrainAverage(df,k,y):
    '''
    parameters: <pandas.dataframe> df, <int> k number of folds, <string> y dependent variable
    returns: Average testing and training errors
    '''
    X = df.drop(y, axis =1)
    Y = df[y]
    kX_df = kFolds(X,k)
    kY_df = kFolds(Y,k)
    
    test_errs = []
    train_errs = []

    for i in range(k):
        X_test = kX_df[i]
        X_train = pd.concat([fold for n, fold in enumerate(kX_df) if n!=i]) 
        Y_test = kY_df[i]
        Y_train = pd.concat([fold for n, fold in enumerate(kY_df) if n!=i])
        lm.fit(X_train, Y_train)
        
        y_hat_train = lm.predict(X_train)
        y_hat_test = lm.predict(X_test)
        train_residuals = y_hat_train - Y_train
        test_residuals = y_hat_test - Y_test
        train_errs.append(np.mean(train_residuals.astype(float)**2))
        test_errs.append(np.mean(test_residuals.astype(float)**2))
    
    return(np.mean(train_errs), np.mean(test_errs))
    
    
def kFolds(df,k):
    '''
    parameters: <pandas.datafram> df, <int> k, number of splits
    returns: split dataset into k sliced pieces 
    '''
    fold_size = len(df)//k
    remainder = len(df)%k
    folds = []
    start = 0
    for foldings in range(1,k+1):
        if foldings <= remainder:
            fold = df.iloc[start: start + fold_size + 1]
            folds.append(fold)
            start += fold_size + 1
        else:
            fold =  df.iloc[start : start + fold_size] 
            folds.append(fold)
            start +=  fold_size
    return folds
          
def kFoldEvaluation(df,k,y):
    '''
    parameters:<pandas.dataframe> df, <int> k number of folds, <string> y dependent variable
    returns: average R-squared, mae, mse, rmse  for k-fold cross validation
    '''
    result_df = kFoldsCrossValidation(df,k, y)
    r_2_mean = result_df.r_2.mean()
    adjusted_r_2_mean = result_df.adjusted_r_2.mean()
    mae_mean = result_df.mae.mean()
    mse_mean = result_df.mse.mean()
    rmse_mean = result_df.rmse.mean()
    return (r_2_mean,adjusted_r_2_mean, mae_mean, mse_mean, rmse_mean)


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

#different method of splitting into k-folds
def kFoldsSklearn(df, k):
    '''
    parameters:<pandas.dataframe> df, <int> k number of folds
    returns: splits into kfold for kfold testing
    '''
    kf = KFold(n_splits = k, shuffle = True, random_state = 2)
    result = next(kf.split(df), None)

    train = df.iloc[result[0]]
    test =  df.iloc[result[1]]
    return (train,test)

#kfold cross validation score
def kFoldCVS(df,k,y):
    '''
    parameters:<pandas.dataframe> df, <int> k number of folds, <string> y dependent variable
    returns: average cross validation score
    '''
    X = df.drop(y, axis =1)
    Y = df[y]
    kf = KFold(n_splits = 10, shuffle = True, random_state = 2)
    cvs = cross_val_score(lm, X, Y, cv = kf)
    return cvs.mean()
    
#Selecting optimal columns
from sklearn.feature_selection import RFE

def selectVariables(df, n, y):
    '''
    parameters: <pandas.dataframe> df, <int> n number of independent variables to be selected, <string> y dependent variable
    returns: independent variables with best fit
    '''
    X = df.drop(y, axis =1)
    Y = df[y]
    selector = RFE(lm, n_features_to_select = n)
    selector = selector.fit(X, Y.values.ravel())
    selected_columns = X.columns[selector.support_ ]
    lm.fit(X[selected_columns],Y)
    return selected_columns
    
def selectVariablePValues(df, y, alpha):
    X = df.drop(y, axis =1)
    Y = df[y]
    est = sm.OLS(Y, X).fit()
    pvalues = pd.DataFrame(est.pvalues, columns=['p_values'])
    selected = pvalues[pvalues['p_values']<alpha]
    return selected
