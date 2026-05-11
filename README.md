Forminifera Morphometric Data Science Report 

This repository contains the code for this report.This project explores morphometric measurements from foraminifera specimens and tests whether shell measurements can be used to predict specimen age (cm), which is treated as an age/depth proxy. The project includes exploratory data analysis, visualisation, PCA and regression modelling.

The main dataset used: data/raw_data.xlsx

Variables:
cm - age/depth proxy variable
area- shell area
feret - maximum shell diameter
perimeter- she ll perimeter

Scripts

This repository includes two scripts (EDA/MODELLING).
EDA script:
-basic data inspection
-missing value check
-histograms
-correlation heatmap
-scatterplots
-boxplots
-pairplots
-PCA plot
-Scree plot
-summary statistics table
-correlation matrix table
-dataset overview table

modelling script:
-train/test splits
-model evaluation
-model comparison table
-random forest feature importance
-observed vs predicted plot
-residual plot
-cross-validation
-hyperparameter tuning using GridSearchCV

Predictor variables: area, feret, perimeter
target variable: cm
models deployed: 
-Linear regression
-Random forest
-Gradient boosting




