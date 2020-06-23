import numpy as np
import pandas as pd

def splitData(df, fraction):
    '''
    parameters: <pandas.dataframe> df, the percentage used for training
    returns: randomly split data frame of test and train data
    '''
    df_train = df.sample(frac = fraction) 
    df_test = pd.concat([df, df_train, df_train]).drop_duplicates(keep=False)
    return (df_train,df_test)

def predictions(X_train, Y_train, X_test):
    '''
    parameters:<array> X_train training independent variables 
                <array> Y_train trainingd dependent variables 
                <array> X_test testing independent variables
    returns: predictions for X_test
    '''
    X = fillMatrix(X_test)
    coeff = coefficients(X_train,Y_train)
    prediction = []
    for item in X:
        prediction.append(dotProduct(item,coeff))
    return np.asarray(prediction, dtype = float)

def coefficients(X_train,Y_train):
    '''
    parameters:<array> X_train training independent variables 
                <array> Y_train trainingd dependent variables 
    returns: coefficients of linear approximation
    '''
    X = fillMatrix(X_train)
    Y = np.asarray(Y_train)
    X_prime = matrixMult(np.linalg.inv(matrixMult(X.transpose(),X)), X.transpose())
    B = []
    for i in range(X_prime.shape[0]):
        B.append(dotProduct(X_prime[i], Y))
    return np.asarray(B)

def initializeMatrix(X_train):
    '''
    parameters: <array> X_train training independent variables
    returns: matrix with zeroes size: number of observations x number of variables + 1
    '''
    return np.zeros((len(X_train[0]), len(X_train) + 1))

def fillMatrix(X_train):
    '''
    parameters: <array> X_train training independent variables
    returns: matrix with filled data via X_train
    '''
    X = initializeMatrix(X_train)
    x = np.ones((len(X_train[0])))
    for i in range(x.shape[0]):
        X[:,0] = x[i]
    for i in range(len(X_train)):
        X[:,i+1] = np.array(X_train[i])
    return X

def matrixMult(X,Y):
    result = np.zeros((X.shape[0], Y.shape[1]))
    for i in range(X.shape[0]):
        for j in range(Y.shape[1]):
            for k in range(Y.shape[0]):
                result[i][j] += X[i][k] * Y[k][j]
    return result

def dotProduct(X,Y):
    result = 0
    for i in range(len(X)):
        result = result + (X[i]*Y[i])
    return result

def errors(predictions, Y_test):
    '''
    parameters: <array> predictions, predictions of X_test
                <array> Y_Test, test independent variable
    returns: mae, mse, rmse
    '''
    true_res = np.asarray(Y_test)
    mae_sum = 0
    mse_sum = 0
    for i in range(len(predictions)):
        mae_sum = mae_sum + np.abs(predictions[i] - Y_test[i])
        mse_sum = mse_sum + (predictions[i] - Y_test[i])**2
        
    MAE = (1/(len(predictions)))*mae_sum
    MSE = (1/(len(predictions)))*mse_sum
    RMSE = np.sqrt(MSE)
    print('MAE:', MAE)
    print('MSE:', MSE)
    print('RMSE:', RMSE)
    
def r_squared(predictions, Y_test, num_of_variables):
    num_sum = 0
    denom_sum = 0
    Mean = mean(Y_test)
    for i in range(len(predictions)):
        num_sum += (Y_test[i] - predictions[i])**2
        denom_sum += (Y_test[i] - Mean)**2
    r_2 = 1- (num_sum/denom_sum)
    adjusted_r_2 = 1 - (1-r_2)*(len(predictions)-1)/(len(predictions)- num_of_variables -1) 
    print('R squared:', r_2)
    print('adujusted R squared:', adjusted_r_2)
    
def mean(data):
    return sum(data)/len(data)   
    
   
    
    