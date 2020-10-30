# Lecture #5: Time Series Analysis

_October 30 2020_

## Why?

* Time is a natural [dimension](https://en.wikipedia.org/wiki/Dimension_(data_warehouse)) of most data.
* Ability to visualize, understand, explain, and predict time series is a very valuable skill, but a non-trivial one.
* It can be much easier if you use the right tools. Let's see what the Python environment can offer...

## ...on your own!

1. Pick data set of your choice, some examples:
   1. [Monthly milk production](https://github.com/plotly/datasets/blob/master/monthly-milk-production-pounds.csv)
   2. [CO2 (ppm) mauna loa](https://datahub.io/core/co2-ppm)
   3. [Mean daily temperature, Fisher River near Dallas](https://github.com/PacktPublishing/Practical-Time-Series-Analysis/blob/master/Data%20Files/mean-daily-temperature-fisher-river.xlsx)
   4. [Monthly car sales in Quebec 1960-1968](https://github.com/jbrownlee/Datasets/blob/master/monthly-car-sales.csv)
2. Load and inspect the data. Remember to take care of missing data, measurements in non-fixes intervals, etc. Pandas provide robust resources for these tasks, see the [time series documentation](https://pandas.pydata.org/pandas-docs/stable/timeseries.html).
3. Using a model of your choice (e.g. regression analysis, ARIMA, neural networks, ... use Google, chances are it is already implemented in Python) predict a selected time frame and compare the result of the model with reality.

Example:
![Anomaly detection simulation example](files/anomaly_detection_example.png)

Last, but not least, *add your results to this repository* via [pull request](https://help.github.com/articles/about-pull-requests/). 
The pull request should consist of ideally two files:

1. `assignments\ts_<your_username>.ipynb` - standard Jupyter notebook with your solution
2. `assignments\ts_<your_username>.py` - plain text version of the notebook (File -> Download as Python), which will be useful in the "code review" phase

_Inspiration_: 
* [Time Series Forecast Study with Python: Monthly Sales of French Champagne](https://machinelearningmastery.com/time-series-forecast-study-python-monthly-sales-french-champagne/)
* [Time Series Prediction with LSTM Recurrent Neural Networks in Python with Keras](https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/)
* [Prophet](https://github.com/facebook/prophet)

Enjoy, but proceed with caution.
