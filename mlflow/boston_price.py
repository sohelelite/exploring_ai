import mlflow
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

X, y = fetch_california_housing(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

experiment_name = "Boston_Price"
try:
    exp_id = mlflow.create_experiment(name=experiment_name)
except:
    exp_id = mlflow.get_experiment_by_name(name=experiment_name).experiment_id


with mlflow.start_run(experiment_id=exp_id):
    mlflow.log_artifact('images')
    n_estimators=100
    learning_rate=1.3

    #build a Machine Learning model using sklearn library
    abr = AdaBoostRegressor(n_estimators=n_estimators,learning_rate=learning_rate,random_state=42).fit(X_train, y_train)
    y_pred = abr.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)

    #create a log for Hyperparameters
    mlflow.log_param('n_estimators', n_estimators)
    mlflow.log_param('learning_rate', learning_rate)

    #create log for evaluation metrics
    mlflow.log_metric('mae', mae)
    mlflow.sklearn.log_model(abr, "Boston_Price")

    mlflow.set_tracking_uri("https://localhost:7501")