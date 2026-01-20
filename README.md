# Recruit Restaurant Visitor Forecasting - LGBM + CatBoost Ensemble

This project implements an ensemble solution (LightGBM + CatBoost) for the [Recruit Restaurant Visitor Forecasting](https://www.kaggle.com/c/recruit-restaurant-visitor-forecasting) Kaggle competition, leveraging the 1st place feature engineering strategy with log-space ensemble to predict restaurant visitor counts.


## 📋Project Overview

### 1st solution (Original File)
**public:0.471, private:0.505**

https://www.kaggle.com/code/plantsgo/solution-public-0-471-private-0-505
-   Uses `air_visit_data` as the core signal.
    
-   Constructs a very large number of handcrafted aggregation features.
    
-   Relies heavily on a custom **kernelMedian** aggregation.
    
-   Generates training samples via sliding windows (`slip = [14, 28, 42]`).
    
-   Trains **LightGBM** models with KFold cross-validation and early stopping.
    
-   Outputs a submission file whose name includes CV score information.
    

This version closely follows early high-ranking public kernels, but is **hard to maintain and reuse**.


### catboost-lbsm (Modified Version)
**public:0.470, private:0.503**
-   Rebuilds the pipeline into a **clean, modular, and reproducible structure**.
    
-   Automatically detects the Kaggle dataset directory (no hard-coded paths).
    
-   Uses a **1st-place–style rolling feature framework**.
    
-   Trains **two different tree models**:
    
    -   LightGBM
        
    -   CatBoost
        
-   Performs **log-space weighted ensembling**.
    
-   Produces timestamped submission files.
    

This version is designed to be **engineering-friendly, extensible, and competition-ready**.

## 🛠️Requirements

Recommended environment: **Python 3.9+**

Required packages:

-   numpy
    
-   pandas
    
-   scikit-learn
    
-   lightgbm
    
-   catboost
    
-   python-dateutil
    

All dependencies are available by default in Kaggle Notebooks.
## 📁Dataset
This project targets the Kaggle competition **Recruit Restaurant Visitor Forecasting**.

https://www.kaggle.com/competitions/recruit-restaurant-visitor-forecasting/data

Required files (CSV or CSV.ZIP):

-   `air_visit_data`
    
-   `air_reserve`
    
-   `hpg_reserve`
    
-   `air_store_info`
    
-   `hpg_store_info`
    
-   `store_id_relation`
    
-   `date_info`
    
-   `sample_submission`

## 🚀 Usage


 **1. Download the file**
Download the file and upload it to kaggle
```python
catboost-lbsm.ipynb
```

 **2. Prepare Dataset**

Upload the competition dataset to `/kaggle/input/` (or adjust the `find_dataset_dir()` function to point to your local dataset path).

**3. Run the Script and Get the Output**
The script generates a timestamped submission file (e.g., `submission_lgb_cb_ens_20240520_143022.csv`) with predicted visitor counts.
![输入图片说明](/img/1.jpg)



## 🔧Key Differences Between catboost-lbsm and 1st solution

## 1 Data Loading

*Hard-coded paths → Automatic dataset discovery*


### 1st solution (Before)

```python
df_train = pd.read_csv('../input/air_visit_data.csv')
df_test = pd.read_csv("../input/sample_submission.csv")
store_id_relation = pd.read_csv("../input/store_id_relation.csv")
air_reserve = pd.read_csv("../input/air_reserve.csv")
hpg_reserve = pd.read_csv("../input/hpg_reserve.csv")

```
-   Strongly coupled to a specific directory layout
    
-   Breaks easily across environments
### catboost-lbsm (After)
```python
def find_dataset_dir():
    required = [
        "air_reserve", "hpg_reserve", "air_store_info",
        "hpg_store_info", "air_visit_data",
        "store_id_relation", "date_info", "sample_submission"
    ]
    roots = glob.glob("/kaggle/input/*")
    ...
    return os.path.dirname(cands[0])

data_path = find_dataset_dir()

def pick_file(prefix):
    cands = glob.glob(os.path.join(data_path, prefix + ".*"))
    ...
    return cands_sorted[0]

```
**Advantages**
-   No hard-coded paths
    
-   Works across Kaggle datasets and local mirrors
    
-   More robust and portable

## 2 Feature Engineering Philosophy

*kernelMedian-heavy aggregation → structured rolling feature system*

### 1st solution_public (Before)

date processing + multiple window sizes + custom `kernelMedian` weighting

example:
```python
def date_handle(df):
    df["weekday"] = pd.to_datetime(df["visit_date"]).dt.weekday
    df["day"] = pd.to_datetime(df["visit_date"]).dt.day
    ...
    df = df.merge(air_info, on="air_store_id", how="left").fillna(-1)
    return df

```
-   Very large code surface
    
-   Hard to debug and extend
    
-   Tight coupling between feature logic and training loop
### catboost-lbsm (After)

Centralized feature factory:
```python
def make_feats(end_date, n_day):
    label = get_label(end_date, n_day)
    parts = []
    parts.append(get_store_visitor_feat(label, key, 1000))
    parts.append(get_store_week_feat(label, key, 1000))
    parts.append(get_store_holiday_feat(label, key, 1000))
    parts.append(get_genre_visitor_feat(label, key, 1000))
    parts.append(get_reserve_feat(label, key))
    parts.append(get_first_last_time(label, key, 1000))
    parts.append(label)
    feat = concat(parts)
    feat = second_feat(feat)
    return feat

```
Rolling construction:
```python
for i in range(58):
    train_feat_sub = make_feats(date_add_days(start_date, i * (-7)), 39)
    train_feat = pd.concat([train_feat, train_feat_sub])

```
**Advantages**
-   Clear feature taxonomy
    
-   Easier tuning and experimentation
    
-   Matches proven 1st-place competition structure
## 3 Reserve Feature Handling

*Implicit reshaping → explicit aggregations*

### catboost-lbsm Example
```python
air_result = (
    air_tmp.groupby(["store_id", "visit_date"], as_index=False)
    .agg(
        air_reserve_visitors=("reserve_visitors", "sum"),
        air_reserve_count=("reserve_visitors", "count"),
    )
)

```
**Advantages**

-   More stable than `unstack/stack`
    
-   Easier to reason about joins
    
-   Fewer silent alignment bugs
## 4 Training Strategy

*Single LGBM with KFold → LGBM + CatBoost ensemble*

### 1st solution (Before)

-   LightGBM only
    
-   KFold CV
    
-   Early stopping
    
-   Long training cycles
### catboost-lbsm (After)

**LightGBM**
```python
lgb_model = lgb.train(
    lgb_params,
    lgb_train,
    num_boost_round=2300
)

```
**CatBoost**
```
cb = CatBoostRegressor(
    loss_function="RMSE",
    iterations=6000,
    depth=8,
    learning_rate=0.03,
    subsample=0.8,
    rsm=0.8
)
cb.fit(X_train, y_train)

```
**Log-space Ensembling**
```
pred_ens_log = 0.65 * pred_lgb + 0.35 * pred_cb
visitors = np.expm1(pred_ens_log)

```
**Advantages**

-   Captures complementary inductive biases
    
-   More stable leaderboard performance
    
-   Easy to tune ensemble weights
## 5 Output Engineering

*Fixed filenames → timestamped submissions*
```
out_name = (
    f"submission_lgb_cb_ens_"
    f"{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
)

```
**Advantages**
-   No accidental overwrites
    
-   Easier experiment tracking
## 📝 Notice
-   The feature engineering process requires a large amount of computation, and it is recommended to run it on a machine with at least 16GB of memory.

-   Hyperparameters such as increasing the number of rounds, learning rate, and ensemble weights can be further optimized to achieve better performance.
