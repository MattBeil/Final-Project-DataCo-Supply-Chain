# Mathematical import
import numpy as np

# Data analysis imports
import pandas as pd
import pandas_profiling as pp

# Visualization imports
import matplotlib.pyplot as plt
import seaborn as sns

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



