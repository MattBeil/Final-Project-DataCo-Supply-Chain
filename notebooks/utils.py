# Mathematical import
import numpy as np

# Data analysis imports
import pandas as pd
import pandas_profiling as pp

# Visualization imports
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objs as go

# Import this to hide warning filters
import warnings
warnings.filterwarnings("ignore")

# Time series analysis import
from statsmodels.tsa.seasonal import seasonal_decompose

# Packages to explore data, estimate statistical models and perfomr tests.
from pmdarima.arima import auto_arima
import pmdarima as pm
from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
import xgboost as xgb
from xgboost import XGBRegressor

# Acuraccy metrics imports
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pmdarima.metrics import smape


################################################################################
################################################################################
################################################################################



def percent(os, feature):
    
    '''
    A function that shows percentages above the visualizations.

    '''


    total = len(feature)
    for p in os.patches:
        percentage = '{:.1f}%'.format(100 * p.get_height()/total)
        x = p.get_x() + p.get_width() / 2 - 0.05
        y = p.get_y() + p.get_height()
        os.annotate(percentage, (x, y), size=12)


################################################################################
################################################################################
################################################################################



def accuracy_metrics(actual, predict):
    mape = np.mean(np.abs(predict - actual)/np.abs(actual))
    rmse = np.sqrt(mean_squared_error(actual, predict))
    mae = np.mean(np.abs(predict - actual))
    mse = mean_squared_error(actual, predict)
    r2 = r2_score(actual, predict)\

    print("Test MAPE: %.3f" % mape)
    print("Test RMSE: %.3f" % rmse)
    print("Test MAE: %.3f" % mae)
    print("Test MSE: %.3f" % mse)
    print("Test R2: %.3f" % r2)
