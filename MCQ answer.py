"""
DATATHON 2026 — Round 1
Part 1: Multiple Choice Questions — Answer Script
Run this script from the folder containing all CSV files.
"""
 
import pandas as pd
import numpy as np
 
# ── Load data ────────────────────────────────────────────────────────────────
print("Loading data...")
orders       = pd.read_csv("orders.csv",       parse_dates=["order_date"])
order_items  = pd.read_csv("order_items.csv")
products     = pd.read_csv("products.csv")
customers    = pd.read_csv("customers.csv",    parse_dates=["signup_date"])
returns      = pd.read_csv("returns.csv",      parse_dates=["return_date"])
reviews      = pd.read_csv("reviews.csv")
payments     = pd.read_csv("payments.csv")
shipments    = pd.read_csv("shipments.csv")
geography    = pd.read_csv("geography.csv")
web_traffic  = pd.read_csv("web_traffic.csv",  parse_dates=["date"])
sales        = pd.read_csv("sales.csv",        parse_dates=["Date"])
 
print("=" * 60)
 
# ── Q1 ───────────────────────────────────────────────────────────────────────
# Median inter-order gap (days) for customers with more than 1 order
print("\nQ1: Median inter-order gap for customers with >1 order")
 
# Sort orders by customer and date
orders_sorted = orders.sort_values(["customer_id", "order_date"])
 
# Compute gap between consecutive orders per customer
orders_sorted["prev_date"] = orders_sorted.groupby("customer_id")["order_date"].shift(1)
orders_sorted["gap_days"]  = (orders_sorted["order_date"] - orders_sorted["prev_date"]).dt.days
 
# Keep only customers with >1 order (i.e., rows that have a previous order)
multi_order_gaps = orders_sorted.dropna(subset=["gap_days"])
 
median_gap = multi_order_gaps["gap_days"].median()
print(f"  Median inter-order gap: {median_gap:.1f} days")
print(f"  → Closest answer: ", end="")
for label, val in [("A) 30", 30), ("B) 90", 90), ("C) 180", 180), ("D) 365", 365)]:
    if abs(median_gap - val) == min(abs(median_gap - v) for v in [30, 90, 180, 365]):
        print(label)
        break
 
print("=" * 60)
 
# ── Q2 ───────────────────────────────────────────────────────────────────────
# Segment with highest average gross margin = (price - cogs) / price
print("\nQ2: Segment with highest average gross margin")
 
products["gross_margin"] = (products["price"] - products["cogs"]) / products["price"]
margin_by_segment = products.groupby("segment")["gross_margin"].mean().sort_values(ascending=False)
print(margin_by_segment.to_string())
print(f"  → Best segment: {margin_by_segment.idxmax()}")
 
print("=" * 60)
 
# ── Q3 ───────────────────────────────────────────────────────────────────────
# Most common return reason for Streetwear products
print("\nQ3: Most common return reason for Streetwear category")
 
streetwear_products = products[products["category"] == "Streetwear"][["product_id"]]
streetwear_returns  = returns.merge(streetwear_products, on="product_id")
reason_counts       = streetwear_returns["return_reason"].value_counts()
print(reason_counts.to_string())
print(f"  → Most common reason: {reason_counts.idxmax()}")
 
print("=" * 60)
 
# ── Q4 ───────────────────────────────────────────────────────────────────────
# Traffic source with lowest average bounce rate
print("\nQ4: Traffic source with lowest average bounce rate")
 
avg_bounce = web_traffic.groupby("traffic_source")["bounce_rate"].mean().sort_values()
print(avg_bounce.to_string())
print(f"  → Lowest bounce rate source: {avg_bounce.idxmin()}")
 
print("=" * 60)
 
# ── Q5 ───────────────────────────────────────────────────────────────────────
# % of order_items rows with a promo applied (promo_id not null)
print("\nQ5: % of order_items rows with promo_id not null")
 
pct_promo = order_items["promo_id"].notna().mean() * 100
print(f"  % with promo: {pct_promo:.1f}%")
print(f"  → Closest answer: ", end="")
for label, val in [("A) 12%", 12), ("B) 25%", 25), ("C) 39%", 39), ("D) 54%", 54)]:
    if abs(pct_promo - val) == min(abs(pct_promo - v) for v in [12, 25, 39, 54]):
        print(label)
        break
 
print("=" * 60)
 
# ── Q6 ───────────────────────────────────────────────────────────────────────
# Age group with highest average orders per customer (excluding null age_group)
print("\nQ6: Age group with highest avg orders per customer")
 
customers_with_age = customers[customers["age_group"].notna()]
order_counts       = orders.groupby("customer_id").size().reset_index(name="order_count")
 
merged = customers_with_age.merge(order_counts, on="customer_id", how="left")
merged["order_count"] = merged["order_count"].fillna(0)
 
avg_orders_by_age = merged.groupby("age_group")["order_count"].mean().sort_values(ascending=False)
print(avg_orders_by_age.to_string())
print(f"  → Best age group: {avg_orders_by_age.idxmax()}")
 
print("=" * 60)
 
# ── Q7 ───────────────────────────────────────────────────────────────────────
# Region with highest total revenue in sales_train.csv
# NOTE: sales.csv is aggregated daily (no region column).
# We need to attribute revenue to regions via orders → geography.
print("\nQ7: Region with highest total revenue (via orders → geography → sales)")
 
# Merge orders with geography to get region per order
orders_geo = orders.merge(geography[["zip", "region"]], on="zip", how="left")
 
# Merge order_items to get revenue per line: quantity * unit_price - discount_amount
order_items["line_revenue"] = order_items["quantity"] * order_items["unit_price"] - order_items["discount_amount"]
items_with_region = order_items.merge(orders_geo[["order_id", "region", "order_date"]], on="order_id")
 
# Filter to training period
train_end = pd.Timestamp("2022-12-31")
items_train = items_with_region[items_with_region["order_date"] <= train_end]
 
revenue_by_region = items_train.groupby("region")["line_revenue"].sum().sort_values(ascending=False)
print(revenue_by_region.to_string())
print(f"  → Highest revenue region: {revenue_by_region.idxmax()}")
 
print("=" * 60)
 
# ── Q8 ───────────────────────────────────────────────────────────────────────
# Most common payment method for cancelled orders
print("\nQ8: Most common payment method for cancelled orders")
 
cancelled = orders[orders["order_status"] == "cancelled"]
payment_counts = cancelled["payment_method"].value_counts()
print(payment_counts.to_string())
print(f"  → Most common method: {payment_counts.idxmax()}")
 
print("=" * 60)
 
# ── Q9 ───────────────────────────────────────────────────────────────────────
# Size with highest return rate = returns / order_items (joined with products)
print("\nQ9: Size with highest return rate")
 
items_with_size = order_items.merge(products[["product_id", "size"]], on="product_id")
 
# Count order lines per size
order_lines_by_size = items_with_size.groupby("size").size().rename("order_lines")
 
# Count return records per size
returns_with_size = returns.merge(products[["product_id", "size"]], on="product_id")
return_lines_by_size = returns_with_size.groupby("size").size().rename("return_lines")
 
size_df = pd.concat([order_lines_by_size, return_lines_by_size], axis=1).fillna(0)
size_df["return_rate"] = size_df["return_lines"] / size_df["order_lines"]
size_df = size_df.sort_values("return_rate", ascending=False)
 
print(size_df.to_string())
print(f"  → Highest return rate size: {size_df['return_rate'].idxmax()}")
 
print("=" * 60)
 
# ── Q10 ──────────────────────────────────────────────────────────────────────
# Installment plan with highest average payment value per order
print("\nQ10: Installment plan with highest avg payment value")
 
avg_payment_by_installment = payments.groupby("installments")["payment_value"].mean().sort_values(ascending=False)
print(avg_payment_by_installment.to_string())
print(f"  → Highest avg payment installment plan: {avg_payment_by_installment.idxmax()} kỳ")
 
print("=" * 60)
print("\nDone! Check answers above and match to MCQ options.")
 