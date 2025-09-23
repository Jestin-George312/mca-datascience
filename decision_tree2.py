import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.tree import plot_tree


#load datasets
california=fetch_california_housing(as_frame=True)
X=california.data
Y=california.target

#Train-test split
X_train,X_test,Y_train,Y_test=train_test_split(
    X,Y,test_size=0.2,random_state=42
)
#Train Decision Tree Regressor
model=DecisionTreeRegressor(random_state=42,max_depth=5)
model.fit(X_train,Y_train)

#predictions

Y_pred=model.predict(X_test)

#Evaluation

print("Mean squared error : ",mean_squared_error(Y_test,Y_pred))
print("R2 Score : ",r2_score(Y_test,Y_pred))

#visualization

plt.figure(figsize=(20,10))
plot_tree(model,feature_names=X.columns,filled=True,rounded=True,fontsize=10)
plt.title("Decision Tree Regression-California Housing Dataset")
plt.show()

#visualization
plt.scatter(Y_test,Y_pred,alpha=0.5,color="purple")
plt.xlabel("Actual prices (in $1000s)")
plt.ylabel("Predicted prices (in $100,000s)")
plt.title("Decision Tree Regression-Predicted vs Actual ")
plt.plot([min(Y_test),max(Y_test)],[min(Y_test),max(Y_test)],color="red")
plt.show()

