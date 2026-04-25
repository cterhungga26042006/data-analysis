"""
DATATHON 2026 — Round 1
Part 3: Sales Forecasting Pipeline
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Tuned to actual sales.csv observations:
  • 3,833 training rows (04/07/2012 – 31/12/2022)
  • Strong annual seasonality: peak Apr–Jun, trough Nov–Jan
  • Revenue decline from 2019 onward (structural break)
  • 382 days where COGS > Revenue (margin squeeze events)
  • COGS ratio mean ~0.87, can exceed 1.0

Install:
    pip install pandas numpy scikit-learn lightgbm xgboost shap matplotlib
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import lightgbm as lgb
import xgboost as xgb
import shap

SEED = 42
np.random.seed(SEED)
os.makedirs("plots", exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
print("━━━ Loading data ━━━")

sales      = pd.read_csv("sales.csv",             parse_dates=["Date"])
sample_sub = pd.read_csv("sample_submission.csv", parse_dates=["Date"])

# Load auxiliary tables if available (graceful fallback if missing)
def safe_read(path, **kwargs):
    try:
        return pd.read_csv(path, **kwargs)
    except FileNotFoundError:
        print(f"  [WARN] {path} not found — skipping")
        return None

orders      = safe_read("orders.csv",      parse_dates=["order_date"])
order_items = safe_read("order_items.csv")
promotions  = safe_read("promotions.csv",  parse_dates=["start_date", "end_date"])
web_traffic = safe_read("web_traffic.csv", parse_dates=["date"])
inventory   = safe_read("inventory.csv",   parse_dates=["snapshot_date"])
payments    = safe_read("payments.csv")
returns_df  = safe_read("returns.csv",     parse_dates=["return_date"])
customers   = safe_read("customers.csv",   parse_dates=["signup_date"])

sales = sales.sort_values("Date").reset_index(drop=True)
print(f"  Train rows : {len(sales)}")
print(f"  Test rows  : {len(sample_sub)}")
print(f"  Train range: {sales['Date'].min().date()} to {sales['Date'].max().date()}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. BUILD FULL DATE SPINE (train + test)
# ══════════════════════════════════════════════════════════════════════════════
train_df = sales[["Date", "Revenue", "COGS"]].copy()
test_stub = sample_sub[["Date"]].copy()
test_stub["Revenue"] = np.nan
test_stub["COGS"]    = np.nan

df = pd.concat([train_df, test_stub], ignore_index=True).sort_values("Date").reset_index(drop=True)
full_range = pd.date_range(df["Date"].min(), df["Date"].max(), freq="D")


# ══════════════════════════════════════════════════════════════════════════════
# 3. AUXILIARY FEATURES FROM OTHER TABLES
# ══════════════════════════════════════════════════════════════════════════════
print("\n━━━ Building auxiliary features ━━━")

aux_frames = []

if orders is not None:
    daily_orders = (
        orders.groupby("order_date").agg(
            n_orders      = ("order_id",     "count"),
            n_cancelled   = ("order_status", lambda x: (x == "cancelled").sum()),
            n_delivered   = ("order_status", lambda x: (x == "delivered").sum()),
            n_unique_cust = ("customer_id",  "nunique"),
        )
        .reset_index().rename(columns={"order_date": "Date"})
    )
    aux_frames.append(daily_orders)

if orders is not None and payments is not None:
    orders_pay = orders[["order_id", "order_date"]].merge(payments, on="order_id")
    daily_pay = (
        orders_pay.groupby("order_date").agg(
            avg_payment   = ("payment_value", "mean"),
            total_payment = ("payment_value", "sum"),
        )
        .reset_index().rename(columns={"order_date": "Date"})
    )
    aux_frames.append(daily_pay)

if returns_df is not None:
    daily_returns = (
        returns_df.groupby("return_date").agg(
            n_returns    = ("return_id",     "count"),
            total_refund = ("refund_amount", "sum"),
        )
        .reset_index().rename(columns={"return_date": "Date"})
    )
    aux_frames.append(daily_returns)

if web_traffic is not None:
    daily_web = (
        web_traffic.groupby("date").agg(
            total_sessions  = ("sessions",                "sum"),
            total_visitors  = ("unique_visitors",         "sum"),
            total_pageviews = ("page_views",              "sum"),
            avg_bounce      = ("bounce_rate",             "mean"),
            avg_duration    = ("avg_session_duration_sec","mean"),
        )
        .reset_index().rename(columns={"date": "Date"})
    )
    aux_frames.append(daily_web)

if promotions is not None:
    promo_list = []
    for d in full_range:
        active = int(((promotions["start_date"] <= d) & (promotions["end_date"] >= d)).sum())
        promo_list.append({"Date": d, "n_active_promos": active})
    aux_frames.append(pd.DataFrame(promo_list))

if inventory is not None:
    monthly_inv = (
        inventory.groupby("snapshot_date").agg(
            avg_fill_rate    = ("fill_rate",         "mean"),
            avg_sell_through = ("sell_through_rate", "mean"),
            pct_stockout     = ("stockout_flag",     "mean"),
            pct_overstock    = ("overstock_flag",    "mean"),
        )
        .reset_index().rename(columns={"snapshot_date": "Date"}).sort_values("Date")
    )
    monthly_inv_daily = (
        monthly_inv.set_index("Date")
        .reindex(full_range).ffill().bfill()
        .reset_index().rename(columns={"index": "Date"})
    )
    aux_frames.append(monthly_inv_daily)

if customers is not None:
    daily_signups = (
        customers.groupby("signup_date").size()
        .reset_index(name="n_new_customers")
        .rename(columns={"signup_date": "Date"})
    )
    aux_frames.append(daily_signups)

for tbl in aux_frames:
    df = df.merge(tbl, on="Date", how="left")

print(f"  Shape after aux merge: {df.shape}")


# ══════════════════════════════════════════════════════════════════════════════
# 4. FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════
print("\n━━━ Engineering features ━━━")

df = df.sort_values("Date").reset_index(drop=True)

# Calendar
df["year"]           = df["Date"].dt.year
df["month"]          = df["Date"].dt.month
df["day"]            = df["Date"].dt.day
df["dayofweek"]      = df["Date"].dt.dayofweek
df["dayofyear"]      = df["Date"].dt.dayofyear
df["weekofyear"]     = df["Date"].dt.isocalendar().week.astype(int)
df["quarter"]        = df["Date"].dt.quarter
df["is_weekend"]     = (df["dayofweek"] >= 5).astype(int)
df["is_month_start"] = df["Date"].dt.is_month_start.astype(int)
df["is_month_end"]   = df["Date"].dt.is_month_end.astype(int)
df["is_quarter_end"] = df["Date"].dt.is_quarter_end.astype(int)

# Fourier terms (annual + weekly seasonality)
for period, n_harm in [(365.25, 3), (7, 2)]:
    for k in range(1, n_harm + 1):
        df[f"sin_{int(period)}_{k}"] = np.sin(2 * np.pi * k * df["dayofyear"] / period)
        df[f"cos_{int(period)}_{k}"] = np.cos(2 * np.pi * k * df["dayofyear"] / period)

# Data-driven season flags (verified from actual sales.csv EDA)
df["is_peak_season"]   = df["month"].isin([4, 5, 6]).astype(int)   # avg >6M VND/day
df["is_trough_season"] = df["month"].isin([11, 12, 1]).astype(int)  # avg <2.7M VND/day
df["is_mid_season"]    = df["month"].isin([7, 8]).astype(int)        # shoulder

# Vietnamese events
df["is_tet_season"] = ((df["month"] == 1) & (df["day"] >= 20) |
                        (df["month"] == 2) & (df["day"] <= 15)).astype(int)
df["is_sale_event"] = (df["month"].isin([11]) |
                        ((df["month"] == 12) & (df["day"] <= 15)) |
                        ((df["month"] == 6)  & (df["day"] >= 15)) |
                        ((df["month"] == 7)  & (df["day"] <= 15))).astype(int)

# Structural break (revenue dropped ~40% after 2018)
df["post_2018"]  = (df["year"] >= 2019).astype(int)
df["year_trend"] = df["year"] - 2012

# Lag features
for lag in [1, 2, 3, 7, 14, 21, 28, 90, 180, 365]:
    df[f"rev_lag_{lag}"]  = df["Revenue"].shift(lag)
    df[f"cogs_lag_{lag}"] = df["COGS"].shift(lag)

# Rolling statistics
for w in [7, 14, 30, 90, 180, 365]:
    s = df["Revenue"].shift(1)
    df[f"rev_roll_mean_{w}"] = s.rolling(w, min_periods=1).mean()
    df[f"rev_roll_std_{w}"]  = s.rolling(w, min_periods=1).std().fillna(0)
    df[f"rev_roll_max_{w}"]  = s.rolling(w, min_periods=1).max()
    df[f"rev_roll_min_{w}"]  = s.rolling(w, min_periods=1).min()

# Year-over-year
df["rev_yoy"]        = df["Revenue"].shift(365)
df["rev_yoy_growth"] = (
    (df["Revenue"].shift(1) - df["Revenue"].shift(366))
    / (df["Revenue"].shift(366).abs() + 1e-9)
).clip(-2, 2)

# COGS ratio (lagged — no leakage)
df["cogs_ratio_lag1"] = (df["COGS"].shift(1) / (df["Revenue"].shift(1) + 1e-9)).clip(0, 2)

# Fill NaN in all feature columns
EXCLUDE  = {"Date", "Revenue", "COGS"}
num_cols = [c for c in df.columns if c not in EXCLUDE]
df[num_cols] = df[num_cols].fillna(0)

FEATURES = [c for c in df.columns if c not in EXCLUDE]
print(f"  Total features: {len(FEATURES)}")


# ══════════════════════════════════════════════════════════════════════════════
# 5. TRAIN / TEST SPLIT
# ══════════════════════════════════════════════════════════════════════════════
TRAIN_END  = pd.Timestamp("2022-12-31")
TEST_START = pd.Timestamp("2023-01-01")

train_df = df[(df["Date"] <= TRAIN_END) & df["Revenue"].notna()].copy()
# Drop first year where lag-365 is zero (would confuse the model)
train_df = train_df[train_df["Date"] >= pd.Timestamp("2013-07-04")].copy()

test_df = df[df["Date"] >= TEST_START].copy()
test_df = test_df.set_index("Date").reindex(sample_sub["Date"]).reset_index()

X_train = train_df[FEATURES]
y_train = train_df["Revenue"]
X_test  = test_df[FEATURES]

print(f"\n━━━ Dataset sizes ━━━")
print(f"  X_train: {X_train.shape}  (from {train_df['Date'].min().date()})")
print(f"  X_test : {X_test.shape}")


# ══════════════════════════════════════════════════════════════════════════════
# 6. WALK-FORWARD CROSS-VALIDATION
# ══════════════════════════════════════════════════════════════════════════════
print("\n━━━ Walk-forward cross-validation (5 folds x 1 year) ━━━")

fold_results = []
for fold in range(5):
    val_end   = len(train_df) - fold * 365
    val_start = val_end - 365
    if val_start < 365:
        break

    X_tr, y_tr   = X_train.iloc[:val_start],      y_train.iloc[:val_start]
    X_val, y_val = X_train.iloc[val_start:val_end], y_train.iloc[val_start:val_end]

    lgb_m = lgb.LGBMRegressor(
        n_estimators=1500, learning_rate=0.03, num_leaves=63, max_depth=7,
        min_child_samples=30, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=0.5, random_state=SEED, n_jobs=-1, verbose=-1
    )
    lgb_m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(-1)])

    xgb_m = xgb.XGBRegressor(
        n_estimators=1500, learning_rate=0.03, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.5,
        random_state=SEED, n_jobs=-1, verbosity=0,
        early_stopping_rounds=80, eval_metric="mae"
    )
    xgb_m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

    pred = 0.55 * lgb_m.predict(X_val) + 0.45 * xgb_m.predict(X_val)
    mae  = mean_absolute_error(y_val, pred)
    rmse = np.sqrt(mean_squared_error(y_val, pred))
    r2   = r2_score(y_val, pred)
    fold_results.append({"fold": fold + 1, "MAE": mae, "RMSE": rmse, "R2": r2})
    print(f"  Fold {fold+1}:  MAE={mae:>12,.0f}  RMSE={rmse:>12,.0f}  R²={r2:.4f}")

cv_df = pd.DataFrame(fold_results)
print(f"\n  CV Mean →  MAE={cv_df['MAE'].mean():>12,.0f}  RMSE={cv_df['RMSE'].mean():>12,.0f}  R²={cv_df['R2'].mean():.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# 7. FINAL MODELS
# ══════════════════════════════════════════════════════════════════════════════
print("\n━━━ Training final models on full training data ━━━")

final_lgb = lgb.LGBMRegressor(
    n_estimators=1000, learning_rate=0.03, num_leaves=63, max_depth=7,
    min_child_samples=30, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=0.5, random_state=SEED, n_jobs=-1, verbose=-1
)
final_lgb.fit(X_train, y_train)

final_xgb = xgb.XGBRegressor(
    n_estimators=1000, learning_rate=0.03, max_depth=6,
    subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.5,
    random_state=SEED, n_jobs=-1, verbosity=0
)
final_xgb.fit(X_train, y_train)
print("  Done.")


# ══════════════════════════════════════════════════════════════════════════════
# 8. PREDICT & SUBMISSION
# ══════════════════════════════════════════════════════════════════════════════
print("\n━━━ Generating predictions ━━━")

LGB_W, XGB_W = 0.55, 0.45
rev_pred = LGB_W * final_lgb.predict(X_test) + XGB_W * final_xgb.predict(X_test)
rev_pred = np.clip(rev_pred, 0, None)

# COGS: 180-day median ratio from end of training (stable, allows ratio > 1)
recent     = train_df.tail(180)
cogs_ratio = (recent["COGS"] / (recent["Revenue"] + 1e-9)).median()
cogs_pred  = np.clip(rev_pred * cogs_ratio, 0, None)

print(f"  Rev  min={rev_pred.min():>12,.0f}  max={rev_pred.max():>12,.0f}  mean={rev_pred.mean():>12,.0f}")
print(f"  COGS ratio (180-day median): {cogs_ratio:.4f}")

print("\n━━━ Writing submission.csv ━━━")
submission = sample_sub[["Date"]].copy()
submission["Revenue"] = np.round(rev_pred, 2)
submission["COGS"]    = np.round(cogs_pred, 2)
submission["Date"]    = submission["Date"].dt.strftime("%Y-%m-%d")
submission.to_csv("submission.csv", index=False)
print(f"  Saved {len(submission)} rows")
print(submission.head(5).to_string(index=False))


# ══════════════════════════════════════════════════════════════════════════════
# 9. PLOTS
# ══════════════════════════════════════════════════════════════════════════════
print("\n━━━ Generating plots ━━━")

# Plot 1: Raw training revenue
fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(train_df["Date"], train_df["Revenue"] / 1e6, lw=0.6, alpha=0.8)
ax.axvline(pd.Timestamp("2019-01-01"), color="red", lw=1.2, ls="--", label="Structural break (2019)")
ax.set_title("Training Revenue 2012–2022", fontsize=13)
ax.set_ylabel("Revenue (million VND)"); ax.legend(); plt.tight_layout()
plt.savefig("plots/01_train_revenue.png", dpi=150, bbox_inches="tight"); plt.close()

# Plot 2: Monthly seasonality
monthly_avg = train_df.copy()
monthly_avg["month"] = pd.to_datetime(train_df["Date"]).dt.month
monthly_avg = monthly_avg.groupby("month")["Revenue"].mean()
colors = ["#e74c3c" if m in [4,5,6] else "#3498db" if m in [11,12,1] else "#95a5a6"
          for m in monthly_avg.index]
fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(monthly_avg.index, monthly_avg.values / 1e6, color=colors)
ax.set_xticks(range(1, 13))
ax.set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"])
ax.set_title("Avg Daily Revenue by Month (Red=Peak, Blue=Trough)", fontsize=12)
ax.set_ylabel("million VND"); plt.tight_layout()
plt.savefig("plots/02_monthly_seasonality.png", dpi=150, bbox_inches="tight"); plt.close()

# Plot 3: Train fit
train_pred = LGB_W * final_lgb.predict(X_train) + XGB_W * final_xgb.predict(X_train)
fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(train_df["Date"], train_df["Revenue"] / 1e6, lw=0.5, alpha=0.7, label="Actual")
ax.plot(train_df["Date"], train_pred / 1e6,           lw=0.5, alpha=0.7, label="Predicted")
ax.set_title("Revenue: Actual vs Predicted — Training Period", fontsize=13)
ax.set_ylabel("million VND"); ax.legend(); plt.tight_layout()
plt.savefig("plots/03_train_fit.png", dpi=150, bbox_inches="tight"); plt.close()

# Plot 4: Test forecast with context
fig, ax = plt.subplots(figsize=(14, 4))
context = train_df[train_df["Date"] >= "2022-07-01"]
ax.plot(context["Date"], context["Revenue"] / 1e6, color="steelblue", lw=0.8, label="Actual (train tail)")
ax.plot(test_df["Date"], rev_pred / 1e6, color="orange", lw=0.8, label="Forecast")
ax.axvline(pd.Timestamp("2023-01-01"), color="gray", ls="--", lw=1)
ax.set_title("Revenue Forecast — Jan 2023 to Jul 2024", fontsize=13)
ax.set_ylabel("million VND"); ax.legend(); plt.tight_layout()
plt.savefig("plots/04_test_forecast.png", dpi=150, bbox_inches="tight"); plt.close()

# Plot 5 & 6: SHAP
print("  Computing SHAP (may take ~30s)...")
X_shap    = X_train.sample(min(600, len(X_train)), random_state=SEED)
explainer = shap.TreeExplainer(final_lgb)
shap_vals = explainer.shap_values(X_shap)

plt.figure(figsize=(10, 8))
shap.summary_plot(shap_vals, X_shap, show=False, max_display=20)
plt.title("SHAP Feature Importance (Beeswarm)", fontsize=12, pad=10)
plt.tight_layout(); plt.savefig("plots/05_shap_summary.png", dpi=150, bbox_inches="tight"); plt.close()

plt.figure(figsize=(10, 7))
shap.summary_plot(shap_vals, X_shap, plot_type="bar", show=False, max_display=20)
plt.title("Top 20 Features by Mean |SHAP|", fontsize=12, pad=10)
plt.tight_layout(); plt.savefig("plots/06_shap_bar.png", dpi=150, bbox_inches="tight"); plt.close()

# Plot 7: CV performance
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, col, color in zip(axes, ["MAE","RMSE","R2"], ["#e74c3c","#e67e22","#2ecc71"]):
    ax.bar(cv_df["fold"], cv_df[col], color=color)
    ax.set_title(f"CV {col}"); ax.set_xlabel("Fold")
plt.suptitle("Walk-Forward Cross-Validation Results", fontsize=13)
plt.tight_layout(); plt.savefig("plots/07_cv_results.png", dpi=150, bbox_inches="tight"); plt.close()

print("  Saved 7 plots to ./plots/")


# ══════════════════════════════════════════════════════════════════════════════
# 10. SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
train_mae  = mean_absolute_error(y_train, train_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
train_r2   = r2_score(y_train, train_pred)

print("\n━━━ Training set metrics (sanity check) ━━━")
print(f"  MAE  : {train_mae:>15,.2f}")
print(f"  RMSE : {train_rmse:>15,.2f}")
print(f"  R²   : {train_r2:.6f}")

print("\n━━━ Cross-validation summary ━━━")
print(cv_df.to_string(index=False))
print(f"\n  Mean   MAE={cv_df['MAE'].mean():>12,.0f}   RMSE={cv_df['RMSE'].mean():>12,.0f}   R²={cv_df['R2'].mean():.4f}")

print("\n" + "━" * 60)
print("✓  submission.csv   → upload to Kaggle")
print("✓  plots/           → 7 plots ready for report")
print("━" * 60)