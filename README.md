# Stock Price Prediction and Analysis using Machine Learning
## Overview
This repository contains code and analysis for predicting stock prices and trends using machine learning techniques, specifically focusing on Samsung Electronics Co., Ltd. (Stock symbol: 005930.KS). The analysis includes data preprocessing, exploratory data analysis (EDA), model selection, evaluation, and interpretation.

## Requirements
Make sure you have the following libraries installed in your r environment:

quantmod
tidyverse
dplyr
ggplot2
caret
randomForest
e1071
tidymodels
xts
gridExtra
Hmisc
caTools
forecast

## Usage
Open the Stock_Price_Prediction.R file in your R environment.

## Run the code blocks sequentially to execute data retrieval, preprocessing, model building, and evaluation.

## Description
Stock_Price_Prediction.R: Main R script containing the code for data retrieval, preprocessing, EDA, model building (ARIMA, logistic regression, XGBoost, random forest), model evaluation, and visualization.
README.md: This file providing an overview, usage instructions, and description of the repository.

## Data
The dataset used for analysis covers the stock information of Samsung Electronics Co., Ltd. from July 15, 2010, to the present. The dataset includes the following columns:

005930.KS.Open: Opening price of Samsung stock on a given day.
005930.KS.High: Highest price of Samsung stock reached during the trading day.
005930.KS.Low: Lowest price of Samsung stock reached during the trading day.
005930.KS.Close: Closing price of Samsung stock on a given day.
005930.KS.Volume: Volume of Samsung stock traded on a given day.
005930.KS.Adjusted: Adjusted closing price of Samsung stock on a given day.
Direction: Indicates if the stock price went higher or lower on a specific day.

## Findings
### Regression Model (ARIMA):
Mean Squared Error (MSE): 0.4622
Root Mean Squared Error (RMSE): 0.6798
Mean Absolute Error (MAE): 0.6718
Mean Absolute Percentage Error (MAPE): 6.04%

### Classification Models:
Random Forest (RF) Accuracy: 63.56%
Logistic Regression (LR) Accuracy: 78.51%
XGBoost (Boost) Accuracy: 58.74%

### The repository includes a detailed literature review on machine learning techniques for stock return forecasting, comparing methods such as support vector machines, neural networks, and ensemble methods.

## Conclusion
The analysis provides insights into the effectiveness of machine learning models for stock price prediction and classification, highlighting the importance of model selection, performance evaluation, and visualization.

For a more comprehensive understanding, refer to the detailed code and analysis in the Stock_Price_Prediction.R file.


