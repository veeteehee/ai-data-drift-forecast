import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
import smtplib
from email.mime.text import MIMEText
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Data Drift Radar", page_icon="📊", layout="centered")
st.title("📊 Data Drift Radar (Multi-Metric + Forecast + Alerts)")

st.write("Upload a CSV with a `date` column and one or more numeric columns (e.g. `sales`, `price`, `visitors`).")

# ----------------------------------------------------------------------
# File upload / fallback to local CSV
uploaded = st.file_uploader("Upload CSV", type=["csv"])
if uploaded is not None:
    df = pd.read_csv(uploaded, parse_dates=["date"])
else:
    try:
        df = pd.read_csv("sales.csv", parse_dates=["date"])
        st.info("Using local sales.csv (no file uploaded).")
    except Exception:
        st.warning("No file uploaded and local sales.csv not found.")
        st.stop()

# ----------------------------------------------------------------------
# Clean & validate
if "date" not in df.columns:
    lower_map = {c.lower(): c for c in df.columns}
    if "date" in lower_map:
        df = df.rename(columns={lower_map["date"]: "date"})
    else:
        st.error("CSV must contain a 'date' column.")
        st.stop()

df = df.sort_values("date").drop_duplicates(subset=["date"])
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if not num_cols:
    st.error("Need at least one numeric column to monitor drift.")
    st.stop()

st.subheader("Raw Data (first 25 rows)")
st.dataframe(df.head(25))

if len(df) < 12:
    st.warning("Need at least 12 rows (more is better).")
    st.stop()

# ----------------------------------------------------------------------
# Sidebar controls
st.sidebar.header("Drift Settings")
baseline_days = st.sidebar.number_input("Baseline window (days)", min_value=7, max_value=180, value=14, step=1)
recent_days   = st.sidebar.number_input("Recent window (days)",   min_value=3, max_value=90,  value=7,  step=1)
alpha         = st.sidebar.slider("Significance level (α)", 0.001, 0.2, 0.05, 0.001)

# Email alert settings
st.sidebar.header("Email Alert (Optional)")
enable_email = st.sidebar.checkbox("Enable email alert when drift is detected", value=False)
smtp_server  = st.sidebar.text_input("SMTP server (e.g., smtp.gmail.com)", value="")
smtp_port    = st.sidebar.number_input("SMTP port", value=587, min_value=1, max_value=65535)
from_email   = st.sidebar.text_input("From email (sender)", value="")
app_password = st.sidebar.text_input("Email app password", value="", type="password")
to_email     = st.sidebar.text_input("To email (recipient)", value="")

# ----------------------------------------------------------------------
# Baseline / recent windows
last_date = df["date"].max()
baseline_start = last_date - pd.Timedelta(days=baseline_days + recent_days)
recent_start   = last_date - pd.Timedelta(days=recent_days)

base_df  = df[(df["date"] > baseline_start) & (df["date"] <= recent_start)]
recent_df= df[(df["date"] > recent_start) & (df["date"] <= last_date)]

st.subheader("Windows Used")
c1, c2 = st.columns(2)
with c1: st.metric("Baseline rows", len(base_df))
with c2: st.metric("Recent rows", len(recent_df))
st.caption(f"Baseline: ({baseline_start.date()} — {recent_start.date()}],  Recent: ({recent_start.date()} — {last_date.date()}]")

if len(base_df) < 5 or len(recent_df) < 3:
    st.warning("Not enough data in selected windows.")
    st.stop()

# ----------------------------------------------------------------------
# Per-feature drift (KS test)
st.subheader("Per-Feature Drift (KS Test)")
results = []
for col in num_cols:
    b = pd.to_numeric(base_df[col], errors="coerce").dropna()
    r = pd.to_numeric(recent_df[col], errors="coerce").dropna()
    if len(b) >= 5 and len(r) >= 3:
        stat, p = ks_2samp(b.values, r.values)
        drift = p < alpha
        results.append({"feature": col, "ks_stat": stat, "p_value": p, "drift": drift})
    else:
        results.append({"feature": col, "ks_stat": np.nan, "p_value": np.nan, "drift": False})

res_df = pd.DataFrame(results).sort_values(["drift","p_value"], ascending=[False, True])
st.dataframe(res_df)

overall_drift = res_df["drift"].any()

st.subheader("Overall Drift Status")
if overall_drift:
    st.error(f"⚠️ Drift detected in at least one feature (α = {alpha}).")
else:
    st.success(f"✅ No significant drift across monitored features (α = {alpha}).")

# ----------------------------------------------------------------------
# Time series chart
st.subheader("Time Series (All Numeric Columns)")
plot_df = df.set_index("date")[num_cols]
st.line_chart(plot_df)

# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# Forecast (Sales) — Regime-aware (uses recent regime after change-point if available)
st.subheader("Forecast (Sales)")
st.caption("Regime-aware forecast: if a change-point is detected, fit only on data after it.")

horizon_days = st.slider("Forecast horizon (days)", min_value=7, max_value=60, value=14, step=1)

if "sales" not in df.columns:
    st.warning("No 'sales' column found—cannot build forecast.")
else:
    # Try to detect a change-point first (fallback to full series if not available)
    recent_series = df.dropna(subset=["sales"]).copy()
    use_recent_from_idx = 0  # default: use entire series
    change_found = False
    try:
        import ruptures as rpt
        if len(recent_series) >= 10:
            algo = rpt.Pelt(model="rbf").fit(recent_series["sales"].values)
            cp = algo.predict(pen=5)
            # cp is a list of segment end indices; pick the first major one if inside range
            if cp and 0 < cp[0] < len(recent_series):
                use_recent_from_idx = cp[0] - 1  # 0-based
                change_found = True
    except ImportError:
        pass  # ruptures not installed; we’ll just use entire series

    # Build the training slice
    train_df = recent_series.iloc[use_recent_from_idx:].copy() if change_found else recent_series.copy()
    train_df["t"] = train_df["date"].map(pd.Timestamp.toordinal)

    # Simple train/test split (holdout = last 10% of training slice)
    if len(train_df) >= 20:
        split = int(len(train_df) * 0.9)
    else:
        split = max(2, len(train_df) - 3)

    X = train_df[["t"]].values
    y = train_df["sales"].values
    X_train, y_train = X[:split], y[:split]
    X_test,  y_test  = X[split:], y[split:]

    from sklearn.linear_model import LinearRegression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    r2 = (lr.score(X_test, y_test) if len(y_test) > 0 else lr.score(X_train, y_train))

    # Future dates
    last_date_train = train_df["date"].max()
    future_dates = pd.date_range(last_date_train + pd.Timedelta(days=1), periods=horizon_days, freq="D")
    future_ord = future_dates.map(pd.Timestamp.toordinal).to_numpy().reshape(-1, 1)
    future_sales = lr.predict(future_ord)

    # Combine actual + forecast
    forecast_df = pd.DataFrame({"date": future_dates, "forecast_sales": future_sales})
    show_df = (
        df[["date","sales"]].rename(columns={"sales": "actual_sales"})
        .merge(forecast_df, on="date", how="outer")
        .set_index("date")
        .sort_index()
    )

    # Display
    if change_found:
        st.info(f"Forecast trained on recent regime starting ~{train_df['date'].min().date()} (after change-point).")
    else:
        st.info("No change-point used (trained on entire series).")

    st.write(f"Forecast model R² (holdout if available): **{r2:.3f}**")
    st.line_chart(show_df)

    st.write("Forecast table")
    st.dataframe(forecast_df.set_index("date").round(2))
    st.download_button("Download forecast (CSV)",
                       data=forecast_df.to_csv(index=False).encode(),
                       file_name="forecast.csv",
                       mime="text/csv")
# ----------------------------------------------------------------------
# Trend vs Level breakdown (Sales)
st.subheader("Trend vs Level (Sales)")

if "sales" not in df.columns:
    st.info("No 'sales' column found — skipping Trend vs Level.")
else:
    # Build baseline/recent windows aligned with the drift section
    # (we already computed base_df and recent_df)
    # Compute level change
    baseline_mean = pd.to_numeric(base_df["sales"], errors="coerce").dropna().mean()
    recent_mean   = pd.to_numeric(recent_df["sales"], errors="coerce").dropna().mean()
    level_change  = recent_mean - baseline_mean

    # Compute linear trend (slope per day) on the full series
    df_tr = df.dropna(subset=["sales"]).copy()
    df_tr["t"] = df_tr["date"].map(pd.Timestamp.toordinal)
    X_tr = df_tr[["t"]].values
    y_tr = df_tr["sales"].values

    slope_per_day = np.nan
    intercept = np.nan
    if len(df_tr) >= 3:
        lr_tr = LinearRegression()
        lr_tr.fit(X_tr, y_tr)
        slope_per_day = float(lr_tr.coef_[0])
        intercept = float(lr_tr.intercept_)

    # Badges
    def badge_from_value(val, tol=1e-6):
        if np.isnan(val):
            return "—"
        if val > tol:
            return "⬆️ Up"
        if val < -tol:
            return "⬇️ Down"
        return "➖ Flat"

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Baseline mean", f"{baseline_mean:.2f}")
    with c2:
        st.metric("Recent mean", f"{recent_mean:.2f}")
    with c3:
        st.metric("Level change (recent - baseline)", f"{level_change:.2f}", delta=f"{level_change:.2f}")

    c4, c5 = st.columns(2)
    with c4:
        st.metric("Trend slope (per day)", f"{slope_per_day:.2f}" if not np.isnan(slope_per_day) else "—",
                  delta=badge_from_value(slope_per_day))
    with c5:
        st.caption("Interpretation")
        st.write(
            "- **Level change**: immediate jump or drop between recent and baseline windows.\n"
            "- **Trend slope**: steady growth/decline across the whole series."
        )

    # Simple visual comparing baseline vs recent means
    st.caption("Baseline vs Recent Level (Sales)")
    level_df = pd.DataFrame({
        "window": ["baseline", "recent"],
        "mean_sales": [baseline_mean, recent_mean]
    }).set_index("window")
    st.bar_chart(level_df)
# ----------------------------------------------------------------------
# Change-Point Detection (Sales)
st.subheader("Change-Point Detection (Sales)")
st.caption(
    "Identifies the most likely date where the underlying sales pattern shifted. "
    "Uses the 'ruptures' package with a simple mean-shift model."
)

try:
    import ruptures as rpt
except ImportError:
    st.warning("Install 'ruptures' (pip install ruptures) to enable this feature.")
else:
    if "sales" not in df.columns:
        st.info("No 'sales' column found — skipping change-point detection.")
    else:
        ts = df.dropna(subset=["sales"]).copy()
        if len(ts) < 10:
            st.info("Need at least 10 data points to run change-point detection.")
        else:
            algo = rpt.Pelt(model="rbf").fit(ts["sales"].values)
            # allow up to one major change-point for clarity
            result = algo.predict(pen=5)
            change_idx = result[0] if result else None

            if change_idx and change_idx < len(ts):
                change_date = ts.iloc[change_idx - 1]["date"]
                st.success(f"Most likely change-point detected around **{change_date.date()}**.")
                # Visualize with a simple vertical marker
                st.line_chart(
                    ts.set_index("date")[["sales"]],
                    height=300
                )
                st.caption("Change-point marked approximately on the chart above (vertical guide not shown interactively).")
            else:
                st.info("No strong change-point detected.")


# ----------------------------------------------------------------------
# Baseline vs recent distributions
st.subheader("Recent vs Baseline Distributions")
for col in num_cols:
    st.markdown(f"**{col}**")
    b = pd.to_numeric(base_df[col], errors="coerce").dropna()
    r = pd.to_numeric(recent_df[col], errors="coerce").dropna()
    if len(b) >= 5 and len(r) >= 3:
        c1, c2 = st.columns(2)
        with c1: st.caption("Baseline"); st.bar_chart(pd.DataFrame({col: b.values}))
        with c2: st.caption("Recent");  st.bar_chart(pd.DataFrame({col: r.values}))
    else:
        st.caption("Not enough data in one of the windows.")

# ----------------------------------------------------------------------
# Optional email alert
def send_email_alert(subject: str, body: str):
    if not (smtp_server and smtp_port and from_email and app_password and to_email):
        st.warning("Email settings incomplete; cannot send alert.")
        return False
    try:
        msg = MIMEText(body, "plain", "utf-8")
        msg["Subject"] = subject
        msg["From"] = from_email
        msg["To"] = to_email
        with smtplib.SMTP(smtp_server, int(smtp_port)) as server:
            server.starttls()
            server.login(from_email, app_password)
            server.send_message(msg)
        return True
    except Exception as e:
        st.error(f"Email failed: {e}")
        return False

if enable_email and overall_drift:
    drifted = res_df[res_df["drift"]]["feature"].tolist()
    body = (
        f"Drift detected in: {', '.join(drifted)}\n"
        f"Baseline window: {baseline_days} days\n"
        f"Recent window: {recent_days} days\n"
        f"Significance α = {alpha}\n"
        f"Last date in dataset: {last_date.date()}\n"
    )
    if send_email_alert("Data Drift Radar: Drift Detected", body):
        st.success("📧 Alert sent!")

st.caption("Tip: Tune baseline/recent windows in the sidebar. Any numeric column will be monitored for drift.")
