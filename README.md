# Bank Customer Churn Prediction
Bank customer churn prediction with CNN.

## Problem and Motivation
Banks often face the problem of customer churn. With the help of data collected by one of the banks of the United States (U.S. Bank) and Data Science methods, we can predict whether the client will leave this bank or not, which will help the bank to better assess the risks of providing services to the client and options for improving work.
The main task of forecasting is to classify a customer as one who will stop working with the bank or vice versa, using Supervised Machine Learning based on customer data provided by the bank.

## Solving Processes 
<ul>
* Collection of raw data from the kaggle.com open resource
* Data Processing
* Data Cleaning 
* EDA
* Building a predictive model based on the collected, processed, and cleaned data
* Presentation of results to interested parties
</ul>

## Resources
Dataset
<ul>
* https://www.kaggle.com/datasets/shantanudhakadd/bank-customer-churn-prediction
</ul>
Dataset manipulation Libraries
<ul>
* Pandas
* Numpy
</ul>
Data Visualisation Libraries
<ul>
* Seaborn
* Plotly
* MatPlotLib
</ul>
Model Building
<ul>
* TensorFlow
* Keras
* A one-dimensional convolutional neural network model with Binary Cross Entropy cost function and accuracy as a model performance evaluation metric
</ul>
Working with Files Libraries
<ul>
* os
* zipfile
</ul>
Interface
<ul>
* Plotly Dash
</ul>

## Method Implementation
### Component Diagram
img
### Component Functionality
<ol>
1. Data Storage. The local storage that houses the collected data is used to train machine learning models.
2. Data Preprocessing.
a. Data cleaning. Rows with missing values are removed from the data set.
b. Data normalization and coding. Categorical data are converted to numeric, redundant characteristics are removed, data are normalized, the maximum and minimum values for numeric columns and category values for categorical ones are stored.
c. Splitting the data into appropriate sets (training, test)
3. Model Training. Model Training component provides functionality:
a. Creation/initialization of the model with the architecture of a One-Dimencional Convolutional Neural Network.
b. Its (re)training with a potential change in the architecture and values of training hyperparameters.
c. Final qualitative and quantitative testing of forecasting effectiveness.
d. Saving the trained model using the version control algorithm for its further support and improvement of the model.
4. Predicting customer churn. A component of direct prediction of processed user input by feeding it to the input of the loaded latest most efficient version of the model.
5. User interface. A component that represents a graphical web user interface that has two main functions:
a. Enter all the necessary data about the client (this data will later be transferred to pre-processing, for use by the model)
b. View the prediction result
6. Processing of user data.
a. Dataframe formation. Data entered manually are collected in a dataframe, data entered in the form of files are converted into dataframes.
b. Data cleaning.
c. Data normalization and coding. With the help of the data saved during processing, the numerical and coding of categorical data is normalized.
</ol>
### Detailed Process Diagram
img

## Screenshots
img
