import mlflow
import mlflow.sklearn

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# ==========================================
# 1. Load California Housing Dataset
# ==========================================

data = fetch_california_housing()

X = data.data
y = data.target


# ==========================================
# 2. Split Dataset
#    80% Training
#    20% Validation
# ==========================================

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# ==========================================
# 3. Create MLflow Experiment
# ==========================================

mlflow.set_experiment("California_Housing_Experiments")


# ==========================================
# 4. Define Experiments
# ==========================================

experiments = [
    {
        "run_name": "Run 1",
        "max_depth": 3,
        "learning_rate": 0.1
    },
    {
        "run_name": "Run 2",
        "max_depth": 5,
        "learning_rate": 0.05
    },
    {
        "run_name": "Run 3",
        "max_depth": 7,
        "learning_rate": 0.01
    }
]


# ==========================================
# 5. Train Each Experiment
# ==========================================

for experiment in experiments:

    with mlflow.start_run(run_name=experiment["run_name"]):

        # Get hyperparameters
        max_depth = experiment["max_depth"]
        learning_rate = experiment["learning_rate"]

        # ------------------------------------------
        # Create the model
        # ------------------------------------------

        model = GradientBoostingRegressor(
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=42
        )

        # ------------------------------------------
        # Train the model
        # ------------------------------------------

        model.fit(X_train, y_train)

        # ------------------------------------------
        # Make predictions on validation data
        # ------------------------------------------

        predictions = model.predict(X_val)

        # ------------------------------------------
        # Calculate metrics
        # ------------------------------------------

        rmse = mean_squared_error(
            y_val,
            predictions
        ) ** 0.5

        mae = mean_absolute_error(
            y_val,
            predictions
        )

        r2 = r2_score(
            y_val,
            predictions
        )

        # ------------------------------------------
        # Log parameters to MLflow
        # ------------------------------------------

        mlflow.log_param(
            "max_depth",
            max_depth
        )

        mlflow.log_param(
            "learning_rate",
            learning_rate
        )

        # ------------------------------------------
        # Log metrics to MLflow
        # ------------------------------------------

        mlflow.log_metric(
            "RMSE",
            rmse
        )

        mlflow.log_metric(
            "MAE",
            mae
        )

        mlflow.log_metric(
            "R2",
            r2
        )

        # ------------------------------------------
        # Log trained model to MLflow
        # ------------------------------------------
        # The trusted type is required by newer
        # MLflow/skops versions for tree-based models.

        mlflow.sklearn.log_model(
            model,
            name="model",
            skops_trusted_types=[
                "sklearn.tree._tree.Tree"
            ]
        )

        # ------------------------------------------
        # Print results
        # ------------------------------------------

        print()
        print("=" * 50)
        print(experiment["run_name"])
        print("=" * 50)

        print(f"Max Depth     : {max_depth}")
        print(f"Learning Rate : {learning_rate}")
        print(f"RMSE          : {rmse:.4f}")
        print(f"MAE           : {mae:.4f}")
        print(f"R²            : {r2:.4f}")
        print("=" * 50)


print()
print("All experiments completed successfully!")
print("Open the MLflow UI using:")
print("mlflow ui")