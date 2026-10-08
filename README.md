# MLflow Experiment Tracking — California Housing

## 1. Task Description

This project demonstrates **ML Experiment Tracking using MLflow**.

The goal is to train the same machine learning regression model using three different hyperparameter configurations, track the experiments using MLflow, compare their results, and select the best-performing model.

The **California Housing dataset** is used for the house price prediction problem.

The machine learning model used is:

* **Gradient Boosting Regressor**

MLflow is used to track:

* Model parameters
* Evaluation metrics
* Trained model artifacts
* Experiment results

The primary metric used to select the best model is **RMSE (Root Mean Squared Error)**. A lower RMSE indicates better performance.

---

## 2. Project Structure

```text
mlflow-task/
│
├── train.py
├── requirements.txt
└── README.md
```

### `train.py`

The Python script:

1. Loads the California Housing dataset.
2. Splits the dataset into 80% training and 20% validation data.
3. Uses a fixed random seed for reproducibility.
4. Trains three Gradient Boosting Regression models.
5. Logs the hyperparameters to MLflow.
6. Calculates RMSE, MAE, and R².
7. Logs the metrics to MLflow.
8. Saves each trained model as an MLflow artifact.

### `requirements.txt`

Contains the Python libraries required to run the project.

---

## 3. Dataset

The project uses the **California Housing dataset** provided by Scikit-learn.

The dataset is split into:

* **80% Training Data**
* **20% Validation Data**

A fixed random seed of `42` is used to make the split reproducible.

```python
train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)
```

---

## 4. Experiment Configurations

Three experiments are trained using different hyperparameters.

| Run   | Max Depth | Learning Rate |
| ----- | --------: | ------------: |
| Run 1 |         3 |          0.10 |
| Run 2 |         5 |          0.05 |
| Run 3 |         7 |          0.01 |

The same `GradientBoostingRegressor` model is used for all three experiments.

---

## 5. Metrics

The following validation metrics are recorded using MLflow:

### RMSE

Root Mean Squared Error measures the average prediction error, with larger errors receiving more weight.

**Lower RMSE is better.**

### MAE

Mean Absolute Error measures the average absolute difference between the predicted and actual values.

**Lower MAE is better.**

### R²

R² measures how well the model explains the variation in the target variable.

**Higher R² is better.**

For this task, **RMSE is the primary metric used for model selection**.

---

## 6. How to Run the Project

### Step 1 — Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
```

Navigate to the project directory:

```bash
cd mlflow-task
```

### Step 2 — Install the requirements

```bash
pip install -r requirements.txt
```

### Step 3 — Run the training script

```bash
python train.py
```

The script will train all three experiments and log their parameters, metrics, and models to MLflow.

### Step 4 — Start the MLflow UI

```bash
mlflow ui
```

Open the following address in a browser:

```text
http://127.0.0.1:5000
```

The experiment is named:

```text
California_Housing_Experiments
```

---

## 7. Experiment Results

The three runs can be compared in the MLflow UI.

| Run   | Max Depth | Learning Rate |            RMSE |             MAE |              R² |
| ----- | --------: | ------------: | --------------: | --------------: | --------------: |
| Run 1 |         3 |          0.10 | **[ADD VALUE]** | **[ADD VALUE]** | **[ADD VALUE]** |
| Run 2 |         5 |          0.05 | **[ADD VALUE]** | **[ADD VALUE]** | **[ADD VALUE]** |
| Run 3 |         7 |          0.01 | **[ADD VALUE]** | **[ADD VALUE]** | **[ADD VALUE]** |

---

## 8. MLflow Screenshot

The MLflow UI was used to compare the three experiments.

Add a screenshot of the MLflow experiment results below:

```text
![MLflow Experiment Results](mlflow-results.png)
```

Place the screenshot file in the project directory:

```text
mlflow-task/
│
├── train.py
├── requirements.txt
├── README.md
└── mlflow-results.png
```

The screenshot should show the three MLflow runs and their parameters and metrics.

---

## 9. Best Model

The best model is:

**[ADD BEST RUN — Run 1 / Run 2 / Run 3]**

Configuration:

* **Max Depth:** [ADD VALUE]
* **Learning Rate:** [ADD VALUE]
* **RMSE:** [ADD VALUE]
* **MAE:** [ADD VALUE]
* **R²:** [ADD VALUE]

### Why was this model selected?

The model was selected because it achieved the **lowest RMSE** among the three experiments.

Since RMSE is the primary model selection metric specified in this task, the configuration with the lowest validation RMSE is considered the best-performing model.

---

## 10. MLflow Tracking

For each experiment, MLflow tracks the following:

### Parameters

```text
max_depth
learning_rate
```

### Metrics

```text
RMSE
MAE
R2
```

### Artifacts

Each experiment also contains the trained machine learning model as an MLflow artifact.

This allows the experiments to be reproduced, compared, and the selected model to be saved for future use.

---

## 11. Conclusion

This project demonstrates the basic ML experiment lifecycle:

```text
Train → Track → Compare → Select
```

Three different Gradient Boosting Regression configurations were trained and tracked using MLflow.

The experiments were compared using RMSE, MAE, and R², with **RMSE used as the primary metric for selecting the best model**.

MLflow makes it easier to keep track of different experiments, compare model configurations, and store trained models as artifacts.
