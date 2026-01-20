#!/usr/bin/env python
# coding: utf-8

# In[1]:


# ============================================================
# Recruit Restaurant Visitor Forecasting
# 1st place features + LightGBM + CatBoost ensemble (log-space)
# Auto-detect Kaggle input directory
# ============================================================

import os
import glob
import time
import numpy as np
import pandas as pd
from dateutil.parser import parse
from datetime import timedelta
from sklearn.preprocessing import LabelEncoder

# -------------------------
# 0) Auto-detect dataset directory
# -------------------------
def find_dataset_dir():
    # look for a directory that contains air_reserve and air_visit_data etc.
    required = [
        "air_reserve", "hpg_reserve", "air_store_info", "hpg_store_info",
        "air_visit_data", "store_id_relation", "date_info", "sample_submission"
    ]
    roots = glob.glob("/kaggle/input/*")
    roots = [r for r in roots if os.path.isdir(r)]

    for r in roots:
        files = " ".join([f.lower() for f in os.listdir(r)])
        ok = all(any(req in f for f in files.split()) or (req in files) for req in required)
        # above is lenient; we will also try direct glob checks:
        glob_ok = (
            len(glob.glob(os.path.join(r, "air_reserve.*"))) > 0 and
            len(glob.glob(os.path.join(r, "air_visit_data.*"))) > 0 and
            len(glob.glob(os.path.join(r, "sample_submission.*"))) > 0
        )
        if ok or glob_ok:
            return r

    # fallback: search for air_reserve file and take its parent directory
    cands = glob.glob("/kaggle/input/**/air_reserve.*", recursive=True)
    if len(cands) == 0:
        raise FileNotFoundError("Cannot find air_reserve.* under /kaggle/input. Please add the competition dataset to your notebook.")
    return os.path.dirname(cands[0])

data_path = find_dataset_dir()
print("Using data_path:", data_path)
print("Files:", sorted(os.listdir(data_path))[:30])

# helper: pick a file by prefix (handles .csv or .csv.zip naming)
def pick_file(prefix):
    cands = glob.glob(os.path.join(data_path, prefix + ".*"))
    if len(cands) == 0:
        # try recursive within the dataset folder
        cands = glob.glob(os.path.join(data_path, "**", prefix + ".*"), recursive=True)
    if len(cands) == 0:
        raise FileNotFoundError(f"Cannot find file starting with {prefix} under {data_path}")
    # prefer .csv if present
    cands_sorted = sorted(cands, key=lambda p: (0 if p.endswith(".csv") else 1, len(p)))
    return cands_sorted[0]

# -------------------------
# 1) Load data
# -------------------------
air_reserve_path = pick_file("air_reserve")
hpg_reserve_path = pick_file("hpg_reserve")
air_store_path   = pick_file("air_store_info")
hpg_store_path   = pick_file("hpg_store_info")
air_visit_path   = pick_file("air_visit_data")
store_map_path   = pick_file("store_id_relation")
date_info_path   = pick_file("date_info")
sub_path         = pick_file("sample_submission")

print("Reading:")
print(air_reserve_path)
print(air_visit_path)
print(sub_path)

air_reserve = pd.read_csv(air_reserve_path).rename(columns={"air_store_id": "store_id"})
hpg_reserve = pd.read_csv(hpg_reserve_path).rename(columns={"hpg_store_id": "store_id"})
air_store   = pd.read_csv(air_store_path).rename(columns={"air_store_id": "store_id"})
hpg_store   = pd.read_csv(hpg_store_path).rename(columns={"hpg_store_id": "store_id"})
air_visit   = pd.read_csv(air_visit_path).rename(columns={"air_store_id": "store_id"})
store_id_map = pd.read_csv(store_map_path).set_index("hpg_store_id", drop=False)
date_info    = pd.read_csv(date_info_path).rename(columns={"calendar_date": "visit_date"}).drop("day_of_week", axis=1)
submission   = pd.read_csv(sub_path)

# -------------------------
# 2) Preprocess (same idea as 1st place)
# -------------------------
submission["visit_date"] = submission["id"].str[-10:]
submission["store_id"] = submission["id"].str[:-11]

air_reserve["visit_date"] = air_reserve["visit_datetime"].str[:10]
air_reserve["reserve_date"] = air_reserve["reserve_datetime"].str[:10]
air_reserve["dow"] = pd.to_datetime(air_reserve["visit_date"]).dt.dayofweek

hpg_reserve["visit_date"] = hpg_reserve["visit_datetime"].str[:10]
hpg_reserve["reserve_date"] = hpg_reserve["reserve_datetime"].str[:10]
hpg_reserve["dow"] = pd.to_datetime(hpg_reserve["visit_date"]).dt.dayofweek

air_visit["id"] = air_visit["store_id"] + "_" + air_visit["visit_date"]

hpg_reserve["store_id"] = hpg_reserve["store_id"].map(store_id_map["air_store_id"]).fillna(hpg_reserve["store_id"])
hpg_store["store_id"] = hpg_store["store_id"].map(store_id_map["air_store_id"]).fillna(hpg_store["store_id"])
hpg_store.rename(columns={"hpg_genre_name": "air_genre_name", "hpg_area_name": "air_area_name"}, inplace=True)

data = pd.concat([air_visit, submission], axis=0).copy()
data["dow"] = pd.to_datetime(data["visit_date"]).dt.dayofweek

date_info["holiday_flg2"] = pd.to_datetime(date_info["visit_date"]).dt.dayofweek
date_info["holiday_flg2"] = ((date_info["holiday_flg2"] > 4) | (date_info["holiday_flg"] == 1)).astype(int)

air_store["air_area_name0"] = air_store["air_area_name"].apply(lambda x: x.split(" ")[0])
lbl = LabelEncoder()
air_store["air_genre_name"] = lbl.fit_transform(air_store["air_genre_name"])
air_store["air_area_name0"] = lbl.fit_transform(air_store["air_area_name0"])

data["visitors"] = np.log1p(data["visitors"])
data = data.merge(air_store, on="store_id", how="left")
data = data.merge(date_info[["visit_date", "holiday_flg", "holiday_flg2"]], on=["visit_date"], how="left")

# -------------------------
# 3) Feature functions (same structure as 1st place)
# -------------------------
def concat(L):
    result = None
    for l in L:
        if result is None:
            result = l
        else:
            result[l.columns.tolist()] = l
    return result

def left_merge(data1, data2, on):
    if type(on) != list:
        on = [on]
    if (set(on) & set(data2.columns)) != set(on):
        data2_temp = data2.reset_index()
    else:
        data2_temp = data2.copy()
    columns = [f for f in data2.columns if f not in on]
    result = data1.merge(data2_temp, on=on, how="left")
    result = result[columns]
    return result

def diff_of_days(day1, day2):
    return (parse(day1[:10]) - parse(day2[:10])).days

def date_add_days(start_date, days):
    end_date = parse(start_date[:10]) + timedelta(days=days)
    return end_date.strftime("%Y-%m-%d")

def get_label(end_date, n_day):
    label_end_date = date_add_days(end_date, n_day)
    label = data[(data["visit_date"] < label_end_date) & (data["visit_date"] >= end_date)].copy()
    label["end_date"] = end_date
    label["diff_of_day"] = label["visit_date"].apply(lambda x: diff_of_days(x, end_date))
    label["month"] = label["visit_date"].str[5:7].astype(int)
    label["year"] = label["visit_date"].str[:4].astype(int)
    for i in [3, 2, 1, -1]:
        date_info_temp = date_info.copy()
        date_info_temp["visit_date"] = date_info_temp["visit_date"].apply(lambda x: date_add_days(x, i))
        date_info_temp.rename(columns={"holiday_flg": f"ahead_holiday_{i}", "holiday_flg2": f"ahead_holiday2_{i}"}, inplace=True)
        label = label.merge(date_info_temp, on=["visit_date"], how="left")
    return label.reset_index(drop=True)

def get_store_visitor_feat(label, key, n_day):
    start_date = date_add_days(key[0], -n_day)
    data_temp = data[(data.visit_date < key[0]) & (data.visit_date > start_date)].copy()
    result = data_temp.groupby(["store_id"], as_index=False)["visitors"].agg(
        {f"store_min{n_day}":"min", f"store_mean{n_day}":"mean", f"store_median{n_day}":"median",
         f"store_max{n_day}":"max", f"store_count{n_day}":"count", f"store_std{n_day}":"std", f"store_skew{n_day}":"skew"}
    )
    return left_merge(label, result, on=["store_id"]).fillna(0)

def get_store_exp_visitor_feat(label, key, n_day):
    start_date = date_add_days(key[0], -n_day)
    data_temp = data[(data.visit_date < key[0]) & (data.visit_date > start_date)].copy()
    d = data_temp["visit_date"].apply(lambda x: diff_of_days(key[0], x))
    w = d.apply(lambda x: 0.985 ** x)
    data_temp["vis_w"] = data_temp["visitors"] * w.values
    result = data_temp.groupby(["store_id"], as_index=False)["vis_w"].agg({f"store_exp_mean{n_day}":"sum"})
    return left_merge(label, result, on=["store_id"]).fillna(0)

def get_store_week_feat(label, key, n_day):
    start_date = date_add_days(key[0], -n_day)
    data_temp = data[(data.visit_date < key[0]) & (data.visit_date > start_date)].copy()
    result = data_temp.groupby(["store_id","dow"], as_index=False)["visitors"].agg(
        {f"store_dow_mean{n_day}":"mean", f"store_dow_median{n_day}":"median",
         f"store_dow_count{n_day}":"count", f"store_dow_std{n_day}":"std", f"store_dow_skew{n_day}":"skew"}
    )
    return left_merge(label, result, on=["store_id","dow"]).fillna(0)

def get_store_week_exp_feat(label, key, n_day):
    start_date = date_add_days(key[0], -n_day)
    data_temp = data[(data.visit_date < key[0]) & (data.visit_date > start_date)].copy()
    d = data_temp["visit_date"].apply(lambda x: diff_of_days(key[0], x))
    w = d.apply(lambda x: 0.985 ** x)
    data_temp["vis_w"] = data_temp["visitors"] * w.values
    result = data_temp.groupby(["store_id","dow"], as_index=False)["vis_w"].agg({f"store_dow_exp_mean{n_day}":"sum"})
    return left_merge(label, result, on=["store_id","dow"]).fillna(0)

def get_store_holiday_feat(label, key, n_day):
    start_date = date_add_days(key[0], -n_day)
    data_temp = data[(data.visit_date < key[0]) & (data.visit_date > start_date)].copy()
    result = data_temp.groupby(["store_id","holiday_flg"], as_index=False)["visitors"].agg(
        {f"store_holiday_mean{n_day}":"mean", f"store_holiday_count{n_day}":"count",
         f"store_holiday_std{n_day}":"std", f"store_holiday_skew{n_day}":"skew"}
    )
    return left_merge(label, result, on=["store_id","holiday_flg"]).fillna(0)

def get_genre_visitor_feat(label, key, n_day):
    start_date = date_add_days(key[0], -n_day)
    data_temp = data[(data.visit_date < key[0]) & (data.visit_date > start_date)].copy()
    result = data_temp.groupby(["air_genre_name"], as_index=False)["visitors"].agg(
        {f"genre_mean{n_day}":"mean", f"genre_median{n_day}":"median",
         f"genre_count{n_day}":"count", f"genre_std{n_day}":"std", f"genre_skew{n_day}":"skew"}
    )
    return left_merge(label, result, on=["air_genre_name"]).fillna(0)

def get_genre_week_feat(label, key, n_day):
    start_date = date_add_days(key[0], -n_day)
    data_temp = data[(data.visit_date < key[0]) & (data.visit_date > start_date)].copy()
    result = data_temp.groupby(["air_genre_name","dow"], as_index=False)["visitors"].agg(
        {f"genre_dow_mean{n_day}":"mean", f"genre_dow_median{n_day}":"median",
         f"genre_dow_count{n_day}":"count", f"genre_dow_std{n_day}":"std", f"genre_dow_skew{n_day}":"skew"}
    )
    return left_merge(label, result, on=["air_genre_name","dow"]).fillna(0)

# def get_reserve_feat(label, key):
#     label_end_date = date_add_days(key[0], key[1])

#     air_tmp = air_reserve[(air_reserve.visit_date >= key[0]) & (air_reserve.visit_date < label_end_date) & (air_reserve.reserve_date < key[0])].copy()
#     air_tmp["diff_time"] = (pd.to_datetime(air_tmp["visit_datetime"]) - pd.to_datetime(air_tmp["reserve_datetime"])).dt.days
#     air_result = air_tmp.groupby(["store_id","visit_date"])["reserve_visitors"].agg({"air_reserve_visitors":"sum","air_reserve_count":"count"}).unstack().fillna(0).stack()
#     air_date_result = air_tmp.groupby(["visit_date"])["reserve_visitors"].agg({"air_date_visitors":"sum","air_date_count":"count"})
#     air_diff_time = air_tmp.groupby(["visit_date"])["diff_time"].agg({"air_diff_time_mean":"mean"})
#     air_store_diff_time = air_tmp.groupby(["store_id","visit_date"])["diff_time"].agg({"air_store_diff_time_mean":"mean"})

#     hpg_tmp = hpg_reserve[(hpg_reserve.visit_date >= key[0]) & (hpg_reserve.visit_date < label_end_date) & (hpg_reserve.reserve_date < key[0])].copy()
#     hpg_tmp["diff_time"] = (pd.to_datetime(hpg_tmp["visit_datetime"]) - pd.to_datetime(hpg_tmp["reserve_datetime"])).dt.days
#     hpg_result = hpg_tmp.groupby(["store_id","visit_date"])["reserve_visitors"].agg({"hpg_reserve_visitors":"sum","hpg_reserve_count":"count"}).unstack().fillna(0).stack()
#     hpg_date_result = hpg_tmp.groupby(["visit_date"])["reserve_visitors"].agg({"hpg_date_visitors":"sum","hpg_date_count":"count"})
#     hpg_diff_time = hpg_tmp.groupby(["visit_date"])["diff_time"].agg({"hpg_diff_time_mean":"mean"})
#     hpg_store_diff_time = hpg_tmp.groupby(["store_id","visit_date"])["diff_time"].agg({"hpg_store_diff_time_mean":"mean"})

#     parts = [
#         left_merge(label, air_result, on=["store_id","visit_date"]).fillna(0),
#         left_merge(label, air_store_diff_time, on=["store_id","visit_date"]).fillna(0),
#         left_merge(label, air_date_result, on=["visit_date"]).fillna(0),
#         left_merge(label, air_diff_time, on=["visit_date"]).fillna(0),
#         left_merge(label, hpg_result, on=["store_id","visit_date"]).fillna(0),
#         left_merge(label, hpg_date_result, on=["visit_date"]).fillna(0),
#         left_merge(label, hpg_store_diff_time, on=["store_id","visit_date"]).fillna(0),
#         left_merge(label, hpg_diff_time, on=["visit_date"]).fillna(0),
#     ]
#     return pd.concat(parts, axis=1)
def get_reserve_feat(label, key):
    label_end_date = date_add_days(key[0], key[1])

    # ---------- AIR ----------
    air_tmp = air_reserve[
        (air_reserve.visit_date >= key[0]) &
        (air_reserve.visit_date < label_end_date) &
        (air_reserve.reserve_date < key[0])
    ].copy()

    air_tmp["diff_time"] = (
        pd.to_datetime(air_tmp["visit_datetime"]) - pd.to_datetime(air_tmp["reserve_datetime"])
    ).dt.days

    # store_id + visit_date level
    air_result = (
        air_tmp.groupby(["store_id", "visit_date"], as_index=False)
        .agg(
            air_reserve_visitors=("reserve_visitors", "sum"),
            air_reserve_count=("reserve_visitors", "count"),
        )
        .set_index(["store_id", "visit_date"])
    )

    air_store_diff_time = (
        air_tmp.groupby(["store_id", "visit_date"], as_index=False)
        .agg(air_store_diff_time_mean=("diff_time", "mean"))
        .set_index(["store_id", "visit_date"])
    )

    # visit_date level
    air_date_result = (
        air_tmp.groupby(["visit_date"], as_index=False)
        .agg(
            air_date_visitors=("reserve_visitors", "sum"),
            air_date_count=("reserve_visitors", "count"),
        )
        .set_index(["visit_date"])
    )

    air_diff_time = (
        air_tmp.groupby(["visit_date"], as_index=False)
        .agg(air_diff_time_mean=("diff_time", "mean"))
        .set_index(["visit_date"])
    )

    # ---------- HPG ----------
    hpg_tmp = hpg_reserve[
        (hpg_reserve.visit_date >= key[0]) &
        (hpg_reserve.visit_date < label_end_date) &
        (hpg_reserve.reserve_date < key[0])
    ].copy()

    hpg_tmp["diff_time"] = (
        pd.to_datetime(hpg_tmp["visit_datetime"]) - pd.to_datetime(hpg_tmp["reserve_datetime"])
    ).dt.days

    hpg_result = (
        hpg_tmp.groupby(["store_id", "visit_date"], as_index=False)
        .agg(
            hpg_reserve_visitors=("reserve_visitors", "sum"),
            hpg_reserve_count=("reserve_visitors", "count"),
        )
        .set_index(["store_id", "visit_date"])
    )

    hpg_store_diff_time = (
        hpg_tmp.groupby(["store_id", "visit_date"], as_index=False)
        .agg(hpg_store_diff_time_mean=("diff_time", "mean"))
        .set_index(["store_id", "visit_date"])
    )

    hpg_date_result = (
        hpg_tmp.groupby(["visit_date"], as_index=False)
        .agg(
            hpg_date_visitors=("reserve_visitors", "sum"),
            hpg_date_count=("reserve_visitors", "count"),
        )
        .set_index(["visit_date"])
    )

    hpg_diff_time = (
        hpg_tmp.groupby(["visit_date"], as_index=False)
        .agg(hpg_diff_time_mean=("diff_time", "mean"))
        .set_index(["visit_date"])
    )

    # ---------- merge back to label ----------
    # left_merge expects columns on index or reset_index, we can just reset_index before passing
    parts = [
        left_merge(label, air_result.reset_index(), on=["store_id", "visit_date"]).fillna(0),
        left_merge(label, air_store_diff_time.reset_index(), on=["store_id", "visit_date"]).fillna(0),
        left_merge(label, air_date_result.reset_index(), on=["visit_date"]).fillna(0),
        left_merge(label, air_diff_time.reset_index(), on=["visit_date"]).fillna(0),

        left_merge(label, hpg_result.reset_index(), on=["store_id", "visit_date"]).fillna(0),
        left_merge(label, hpg_store_diff_time.reset_index(), on=["store_id", "visit_date"]).fillna(0),
        left_merge(label, hpg_date_result.reset_index(), on=["visit_date"]).fillna(0),
        left_merge(label, hpg_diff_time.reset_index(), on=["visit_date"]).fillna(0),
    ]

    return pd.concat(parts, axis=1)


def get_first_last_time(label, key, n_day):
    start_date = date_add_days(key[0], -n_day)
    tmp = data[(data.visit_date < key[0]) & (data.visit_date > start_date)].copy().sort_values("visit_date")
    grp = tmp.groupby("store_id")["visit_date"].agg(
        first_time=lambda x: diff_of_days(key[0], np.min(x)),
        last_time=lambda x: diff_of_days(key[0], np.max(x)),
    )
    return left_merge(label, grp, on=["store_id"]).fillna(0)

def second_feat(result):
    for c1, c2 in [(14, 28), (28, 56), (56, 1000)]:
        a = f"store_mean{c1}"
        b = f"store_mean{c2}"
        if a in result.columns and b in result.columns:
            result[f"store_mean_ratio_{c1}_{c2}"] = (result[a] + 1e-6) / (result[b] + 1e-6)
    return result

def make_feats(end_date, n_day):
    t0 = time.time()
    key = (end_date, n_day)
    print("make_feats key:", key)

    label = get_label(end_date, n_day)

    parts = []
    parts.append(get_store_visitor_feat(label, key, 1000))
    parts.append(get_store_visitor_feat(label, key, 56))
    parts.append(get_store_visitor_feat(label, key, 28))
    parts.append(get_store_visitor_feat(label, key, 14))
    parts.append(get_store_exp_visitor_feat(label, key, 1000))

    parts.append(get_store_week_feat(label, key, 1000))
    parts.append(get_store_week_feat(label, key, 56))
    parts.append(get_store_week_feat(label, key, 28))
    parts.append(get_store_week_feat(label, key, 14))
    parts.append(get_store_week_exp_feat(label, key, 1000))

    parts.append(get_store_holiday_feat(label, key, 1000))

    parts.append(get_genre_visitor_feat(label, key, 1000))
    parts.append(get_genre_visitor_feat(label, key, 56))
    parts.append(get_genre_visitor_feat(label, key, 28))
    parts.append(get_genre_week_feat(label, key, 1000))
    parts.append(get_genre_week_feat(label, key, 56))
    parts.append(get_genre_week_feat(label, key, 28))

    parts.append(get_reserve_feat(label, key))
    parts.append(get_first_last_time(label, key, 1000))

    parts.append(label)

    feat = concat(parts)
    feat = second_feat(feat)
    print("shape:", feat.shape, "time:", round(time.time() - t0, 1), "s")
    return feat

# -------------------------
# 4) Build train/test feats (rolling)
# -------------------------
train_feat = pd.DataFrame()
start_date = "2017-03-12"

for i in range(58):
    train_feat_sub = make_feats(date_add_days(start_date, i * (-7)), 39)
    train_feat = pd.concat([train_feat, train_feat_sub], axis=0, ignore_index=True)

for i in range(1, 6):
    train_feat_sub = make_feats(date_add_days(start_date, i * 7), 42 - (i * 7))
    train_feat = pd.concat([train_feat, train_feat_sub], axis=0, ignore_index=True)

test_feat = make_feats(date_add_days(start_date, 42), 39)

drop_cols = ["id", "store_id", "visit_date", "end_date", "air_area_name", "visitors", "month"]
predictors = [c for c in test_feat.columns if c not in drop_cols]

X_train = train_feat[predictors]
y_train = train_feat["visitors"]   # already log1p
X_test  = test_feat[predictors]

# -------------------------
# 5) Train LGBM
# -------------------------
import lightgbm as lgb

lgb_params = {
    "learning_rate": 0.02,
    "boosting_type": "gbdt",
    "objective": "regression",
    "metric": "rmse",
    "sub_feature": 0.7,
    "num_leaves": 60,
    "min_data": 100,
    "min_hessian": 1,
    "verbose": -1,
}

t0 = time.time()
lgb_train = lgb.Dataset(X_train, y_train)
lgb_model = lgb.train(lgb_params, lgb_train, num_boost_round=2300)
pred_lgb = lgb_model.predict(X_test)
print("LGBM time:", round(time.time() - t0, 1), "s")

# -------------------------
# 6) Train CatBoost
# -------------------------
try:
    from catboost import CatBoostRegressor
except Exception:
    import sys, subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "catboost"])
    from catboost import CatBoostRegressor

t0 = time.time()
cb = CatBoostRegressor(
    loss_function="RMSE",
    iterations=6000,
    learning_rate=0.03,
    depth=8,
    l2_leaf_reg=6,
    subsample=0.8,
    rsm=0.8,
    random_seed=42,
    verbose=300
)
cb.fit(X_train, y_train)
pred_cb = cb.predict(X_test)
print("CatBoost time:", round(time.time() - t0, 1), "s")

# -------------------------
# 7) Ensemble in log space -> submission
# -------------------------
w_lgb, w_cb = 0.65, 0.35
pred_ens_log = w_lgb * pred_lgb + w_cb * pred_cb

sub = pd.DataFrame({
    "id": test_feat["store_id"] + "_" + test_feat["visit_date"],
    "visitors": np.expm1(pred_ens_log)
})
sub = submission[["id"]].merge(sub, on="id", how="left").fillna(0.0)
sub["visitors"] = sub["visitors"].clip(lower=0.0)

out_name = f"submission_lgb_cb_ens_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
sub.to_csv(out_name, index=False, float_format="%.4f")
print("Saved:", out_name)
print(sub.head())


# In[ ]:




