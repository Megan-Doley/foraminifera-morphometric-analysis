#imported libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import os
#-------------------------------------------------------------------------------
# Load data
df = pd.read_excel( 
    "data/raw_data.xlsx",
    sheet_name="area"
)
os.makedirs("outputs/figures", exist_ok=True)
os.makedirs("outputs/model_results", exist_ok=True)
#----------------------------------------------------------
#select features and target variable
print(df.head())
X = df[['area','Feret','Perimeter']]
y = df['cm'] #age/depth proxy used as the target variable

#----------------------------------------------------------------

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining shape:", X_train.shape)
print("Testing shape:", X_test.shape)
#-----------------------------------------------------------

# Train linear regression model
lr = LinearRegression()
lr.fit(
    X_train,
    y_train
)

pred_lr = lr.predict(X_test)

print("\nLINEAR REGRESSION")
print("------------------")

print(
"R2:",
round(
r2_score(
y_test,
pred_lr
),3)
)

print(
"MAE:",
round(
mean_absolute_error(
y_test,
pred_lr
),3)
)

print(
"RMSE:",
round(
mean_squared_error(
y_test,
pred_lr
)**0.5,
3)
)

#------------------------------------------------------------

# Train random forest model
rf = RandomForestRegressor(
    n_estimators=500,
    random_state=42
)

rf.fit(
    X_train,
    y_train
)

pred_rf = rf.predict(X_test)


print("\nRANDOM FOREST")
print("------------------")

print(
"R2:",
round(
r2_score(
y_test,
pred_rf
),3)
)

print(
"MAE:",
round(
mean_absolute_error(
y_test,
pred_rf
),3)
)

print(
"RMSE:",
round(
mean_squared_error(
y_test,
pred_rf
)**0.5,
3)
)

#------------------------------------------------------------
# Cross-validation for linear regression
scores = cross_val_score(
    LinearRegression(),
    X,
    y,
    cv=5,
    scoring='r2'
)

print(scores)
print(scores.mean())

#--------------------------------------------------------------
# Train gradient boosting model
gbr = GradientBoostingRegressor(
random_state=42
)

gbr.fit(
X_train,
y_train
)

pred_gbr = gbr.predict(X_test)

print("\nGRADIENT BOOSTING")

print(
"R2:",
round(
r2_score(
y_test,
pred_gbr
),3)
)

print(
"MAE:",
round(
mean_absolute_error(
y_test,
pred_gbr
),3)
)

print(
"RMSE:",
round(
mean_squared_error(
y_test,
pred_gbr
)**0.5,
3)
)

#-----------------------------------------------------------------
# Feature importance
importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf.feature_importances_
}).sort_values("Importance", ascending=False)

print("\nFeature Importance")
print(importance)
#---------------------------------------------------------
# Feature importance plot
plt.figure()
plt.bar(importance["Feature"], importance["Importance"])
plt.title("Random Forest Feature Importance")
plt.ylabel("Importance")
plt.tight_layout()
plt.savefig("outputs/figures/feature_importance.png")
plt.close()

#---------------------------------------------------------------------------
# ovbserved and predicted plot
plt.figure()
plt.scatter(y_test, pred_lr)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    linestyle="--"
)

plt.xlabel("Observed Age (cm)")
plt.ylabel("Predicted Age (cm)")
plt.title("Observed vs Predicted Values")

plt.tight_layout()
plt.savefig("outputs/figures/predicted_vs_actual.png")
plt.close()

#---------------------------------------------------------------------
# residual plot
residuals = y_test - pred_lr

plt.figure()
plt.scatter(pred_lr, residuals)

plt.axhline(0, linestyle="--")

plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Residual Diagnostics")

plt.tight_layout()
plt.savefig("outputs/figures/residual_plot.png")
plt.close()

#---------------------------------------------------------------------
#cross validation scores
kf = KFold(n_splits=5, shuffle=True, random_state=42)

cv_scores = cross_val_score(
    rf,
    X,
    y,
    cv=kf,
    scoring='r2'
)

print("\nCross Validation R2 Scores:")
print(cv_scores)

print("Mean CV R2:", np.mean(cv_scores))
print("Standard Deviation CV R2:", np.std(cv_scores))

#-----------------------------------------------------
# Hyperparameter tuning with GridSearchCV

param_grid = {
    'n_estimators':[100,200],
    'max_depth':[3,5,10,None]
}

grid = GridSearchCV(
    RandomForestRegressor(random_state=42),
    param_grid,
    cv=5,
    scoring='r2'
)

grid.fit(X_train,y_train)

print(grid.best_params_)
print(grid.best_score_)

#---------------------------------------------------------
# Model comparison
results = pd.DataFrame({

"Model":[
"Linear Regression",
"Random Forest",
"Gradient Boosting"
],

"R2":[
r2_score(y_test,pred_lr),
r2_score(y_test,pred_rf),
r2_score(y_test,pred_gbr)
],

"MAE":[
mean_absolute_error(y_test,pred_lr),
mean_absolute_error(y_test,pred_rf),
mean_absolute_error(y_test,pred_gbr)
],

"RMSE":[
mean_squared_error(y_test,pred_lr)**0.5,
mean_squared_error(y_test,pred_rf)**0.5,
mean_squared_error(y_test,pred_gbr)**0.5
]

})

print("\nMODEL COMPARISON")
print(results.round(3))

results.to_csv(
"outputs/model_results/model_comparison.csv",
index=False
)

#--------------------------------------------------------