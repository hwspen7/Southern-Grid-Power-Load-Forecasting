# Southern Grid Power Load Forecasting (XGBoost)

This project focuses on short-term power load forecasting based on historical electricity load data collected from the southern power grid. It aims to predict next-hour electricity demand using machine learning techniques, with XGBoost serving as the main forecasting model. On unseen test data, the model achieved an MAE of 58.51, an RMSE of 95.63, an R² score of 0.778, and a MAPE of 5.84%, indicating strong performance in capturing overall load trends and daily periodic behavior.

## Project Structure

```
.
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── figures/
│       └── predicted_load_and_real_load.png
├── model/
│   └── xgb.pkl
├── utils/
│   ├── log.py
│   └── common.py
├── predict.py
└── README.md
```

## Method Overview

The model used in this project is:

* XGBoost Regressor

The prediction task is framed as a supervised regression problem, where the model learns patterns from historical load data and predicts the load at the next timestamp.

## Feature Engineering

To capture temporal patterns and short-term dependencies, the following features are constructed:

### Time-based features

Time is encoded using one-hot representations:

* Hour of the day (hour_00 ~ hour_23)
* Month of the year (month_01 ~ month_12)

### Recent load features

Short term dependencise are modeled using lag features:

* Load at previous 1 hour
* Load at previous 2 hours
* Load at previous 3 hours

### Daily pattern feature

To capture periodic behavior:

* Load at the same time on the previous day (yesterday_load)

## Prediction Workflow

The prediction process is designed to closely mimic real-world deployment.

For each prediction timestamp:

1. Only historical data before the current time is kept
2. All future data is hidden to avoid data leakage
3. Features are generated using the available history
4. The trained model predicts the load for that timestamp

The step-by-step prediction strategy ensures that the model does not rely on future information.

## Results

The model is eavluated on unseen test data using multiple regression metrics:

```
MAE   : 58.51 (58.51350527527512)
RMSE  : 95.63 (95.62592906392777)
R²    : 0.778 (0.7783933825721643)
MAPE  : 5.84% (5.839583621186919)
```

Given that the load values typically range from approximately 700 to 1400, the relative prediction error remains low, indicating solid model performance.

## Visualization

### Load Distribution & Temporal Patterns

![power_load_analysis](../../Typora/images/power_load_analysis.png)

The dataset shows clear temporal patterns:

* The load distribution is slightly right-skewed, with occasional high-demand peaks
* Strong daily cycles are observed, with higher load during daytime
* Seasonal variation is visible across months
* Weekday and weekend patterns are generally similar

### Prediction vs Real Load

![predicted_load_and_real_load](../../Typora/images/predicted_load_and_real_load.png)

The model is able to follow the overall trend and periodic patterns closely. It performs well in most time periods and captures daily fluctuations effectively.

## Analysis and Design

* Adopted a step-by-step forecasting approach, where each timestamp is predicted sequentially to better reflect real-world deployment scenarios
* Ensured that only historical data prior to each prediction time is used, effectively preventing data leakage
* Used a dictionary-based structure to store historical load values, enabling efficient lookup instead of repeated DataFrame filtering
* Organized the pipeline into separate stages for data preprocessing, feature engineering, and model inference, improving code clarity and maintainability
* Observed that the model captures overall trends and daily periodic patterns well, with stable performance across most time periods
* Noted that prediction errors increase during sudden load spikes, mainly due to the absence of external features such as weather or unexpected demand factors

## How to Run

### 1. Install dependencies

```
pip install pandas numpy scikit-learn xgboost matplotlib joblib
```

### 2. Prepare data

Place your dataset under:

```
data/test.csv
```

* time
* Power_load

### 3. Run prediction

```
python predict.py
```





