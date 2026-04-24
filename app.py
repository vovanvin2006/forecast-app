import io
import re
import csv
import math
import numpy as np
import pandas as pd
import requests
import cloudscraper
import plotly.graph_objects as go
import streamlit as st

from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from streamlit_autorefresh import st_autorefresh

st.set_page_config(page_title="Stock Forecast 30D App", layout="wide")
st.title("📈 App dự báo giá cổ phiếu 30 ngày tới")
st.caption("Dán link dữ liệu hoặc upload file CSV/XLSX → app tự làm sạch dữ liệu, huấn luyện mô hình và vẽ dashboard dự báo.")

with st.sidebar:
    st.header("⚙️ Cấu hình")
    source_url = st.text_input("Link dữ liệu", placeholder="Dán link CSV / XLSX / trang historical data")
    uploaded_file = st.file_uploader("Hoặc upload file CSV/XLSX", type=["csv", "xlsx", "xls"])
    forecast_horizon = st.slider("Số ngày giao dịch cần dự báo", min_value=5, max_value=60, value=30, step=1)
    backtest_size = st.slider("Số phiên dùng để backtest", min_value=20, max_value=120, value=30, step=5)
    auto_refresh = st.checkbox("Tự refresh app mỗi 6 giờ", value=False)
    run_btn = st.button("🚀 Chạy dự báo", use_container_width=True)

if auto_refresh:
    st_autorefresh(interval=6 * 60 * 60 * 1000, key="data_refresh_6h")


def parse_number(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip().replace(" ", "").replace('"', "").replace("'", "")
    if s == "":
        return np.nan
    if "," in s and "." in s:
        if s.find(",") < s.find("."):
            s = s.replace(",", "")
        else:
            s = s.replace(".", "").replace(",", ".")
    else:
        if s.count(",") == 1 and len(s.split(",")[-1]) <= 2:
            s = s.replace(",", ".")
        else:
            s = s.replace(",", "")
    try:
        return float(s)
    except Exception:
        return np.nan


def parse_volume(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip().upper().replace(",", "").replace('"', "").replace("'", "")
    if s == "":
        return np.nan
    mult = 1
    if s.endswith("K"):
        mult = 1_000
        s = s[:-1]
    elif s.endswith("M"):
        mult = 1_000_000
        s = s[:-1]
    elif s.endswith("B"):
        mult = 1_000_000_000
        s = s[:-1]
    try:
        return float(s) * mult
    except Exception:
        return np.nan


def parse_pct(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip().replace("%", "").replace(",", ".").replace('"', "").replace("'", "")
    if s == "":
        return np.nan
    try:
        return float(s) / 100.0
    except Exception:
        return np.nan


def normalize_columns(df):
    mapping = {}
    for c in df.columns:
        cl = str(c).strip().lower()
        if cl in ["ngày", "date"]:
            mapping[c] = "date"
        elif cl in ["lần cuối", "price", "close", "đóng cửa"]:
            mapping[c] = "close"
        elif cl in ["mở", "open"]:
            mapping[c] = "open"
        elif cl in ["cao", "high"]:
            mapping[c] = "high"
        elif cl in ["thấp", "low"]:
            mapping[c] = "low"
        elif cl in ["kl", "vol.", "vol", "volume"]:
            mapping[c] = "volume"
        elif "% thay đổi" in cl or "change %" in cl or cl == "change":
            mapping[c] = "change_pct"
        else:
            mapping[c] = re.sub(r"\W+", "_", cl)
    return df.rename(columns=mapping)


def looks_like_date_col(series):
    s = pd.to_datetime(series.astype(str).str.strip(), errors="coerce", dayfirst=True)
    return s.notna().sum() >= min(3, len(series))


def split_single_column_csv(df):
    if df.shape[1] != 1:
        return df
    header = str(df.columns[0])
    body = df.iloc[:, 0].astype(str).tolist()
    text = "\n".join([header] + body)
    for sep in [",", ";", "\t"]:
        try:
            rows = list(csv.reader(io.StringIO(text), delimiter=sep, quotechar='"'))
            max_cols = max(len(r) for r in rows)
            if max_cols >= 7:
                rows = [r + [""] * (max_cols - len(r)) for r in rows]
                return pd.DataFrame(rows[1:], columns=rows[0])
        except Exception:
            continue
    return df


def force_expected_columns(df):
    df = split_single_column_csv(df)
    if df.shape[1] >= 7 and looks_like_date_col(df.iloc[:10, 0]):
        df = df.iloc[:, :7].copy()
        df.columns = ["date", "close", "open", "high", "low", "volume", "change_pct"]
        return df
    return normalize_columns(df)


def read_csv_robust(file_bytes):
    encodings = ["utf-8", "utf-8-sig", "cp1258", "cp1252", "latin1", "utf-16"]
    separators = [None, ",", ";", "\t"]
    last_error = None
    for enc in encodings:
        for sep in separators:
            try:
                if sep is None:
                    df = pd.read_csv(io.BytesIO(file_bytes), encoding=enc, sep=None, engine="python")
                else:
                    df = pd.read_csv(io.BytesIO(file_bytes), encoding=enc, sep=sep)
                df = force_expected_columns(df)
                if df.shape[1] >= 7 or "date" in df.columns:
                    return df
            except Exception as e:
                last_error = e
                continue
    raise ValueError(f"Không đọc được CSV. Lỗi cuối: {last_error}")


def find_price_table(tables):
    best = None
    best_score = -1
    for t in tables:
        cols = [str(c).strip().lower() for c in t.columns]
        joined = " | ".join(cols)
        score = 0
        if "date" in joined or "ngày" in joined:
            score += 3
        if "price" in joined or "close" in joined or "lần cuối" in joined:
            score += 4
        if "open" in joined or "mở" in joined:
            score += 1
        if "high" in joined or "cao" in joined:
            score += 1
        if "low" in joined or "thấp" in joined:
            score += 1
        if "vol" in joined or "kl" in joined or "volume" in joined:
            score += 1
        if score > best_score:
            best = t.copy()
            best_score = score
    if best is None:
        raise ValueError("Không tìm thấy bảng dữ liệu giá phù hợp.")
    return best


@st.cache_data(ttl=3600, show_spinner=False)
def load_data_from_upload(file):
    name = file.name.lower()
    if name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(file)
        return force_expected_columns(df)
    if name.endswith(".csv"):
        return read_csv_robust(file.getvalue())
    raise ValueError("File không đúng định dạng CSV/XLSX.")


@st.cache_data(ttl=3600, show_spinner=False)
def load_data_from_url(url):
    url = url.strip()
    if url.lower().endswith(".csv"):
        content = requests.get(url, timeout=30).content
        return read_csv_robust(content)
    if url.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(url)

    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept-Language": "vi,en-US;q=0.9,en;q=0.8",
        "Referer": "https://www.google.com/",
    }
    html = None
    try:
        r = requests.get(url, headers=headers, timeout=30)
        r.raise_for_status()
        html = r.text
    except Exception:
        pass
    if html is None:
        try:
            scraper = cloudscraper.create_scraper(browser={"browser": "chrome", "platform": "windows", "mobile": False})
            r = scraper.get(url, headers=headers, timeout=30)
            r.raise_for_status()
            html = r.text
        except Exception as e:
            raise RuntimeError(f"Không đọc được dữ liệu từ link. Lỗi: {e}")
    try:
        tables = pd.read_html(io.StringIO(html))
        if len(tables) == 0:
            raise ValueError("Không tìm thấy bảng HTML.")
        return force_expected_columns(find_price_table(tables))
    except Exception as e:
        raise RuntimeError(
            "Đã tải được nội dung trang nhưng không đọc được bảng dữ liệu. "
            "Bạn hãy dùng link CSV/XLSX trực tiếp hoặc upload file. "
            f"Chi tiết: {e}"
        )


def clean_data(df):
    df = force_expected_columns(df)
    if "date" not in df.columns:
        raise ValueError(f"Thiếu cột ngày (date/ngày). Columns hiện có: {list(df.columns)}")
    if "close" not in df.columns:
        raise ValueError(f"Thiếu cột giá đóng cửa. Columns hiện có: {list(df.columns)}")

    keep = [c for c in ["date", "close", "open", "high", "low", "volume", "change_pct"] if c in df.columns]
    df = df[keep].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=True)

    for c in ["close", "open", "high", "low"]:
        if c in df.columns:
            df[c] = df[c].apply(parse_number)

    if "volume" in df.columns:
        df["volume"] = df["volume"].apply(parse_volume)
    else:
        df["volume"] = np.nan

    if "change_pct" in df.columns:
        df["change_pct"] = df["change_pct"].apply(parse_pct)
    else:
        df["change_pct"] = np.nan

    df = df.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
    for c in ["open", "high", "low"]:
        if c not in df.columns:
            df[c] = df["close"]
    df["volume"] = df["volume"].ffill().bfill()
    df["change_pct"] = df["change_pct"].fillna(df["close"].pct_change())
    return df


def add_features(df):
    d = df.copy()
    d["return_1d"] = d["close"].pct_change()
    d["range_pct"] = (d["high"] - d["low"]) / d["low"].replace(0, np.nan)
    d["oc_pct"] = (d["close"] - d["open"]) / d["open"].replace(0, np.nan)

    for w in [3, 5, 7, 14, 21]:
        d[f"ma_{w}"] = d["close"].rolling(w).mean()
        d[f"std_{w}"] = d["close"].rolling(w).std()
        d[f"vol_ma_{w}"] = d["volume"].rolling(w).mean()

    for lag in [1, 2, 3, 5, 7, 10, 14, 21]:
        d[f"close_lag_{lag}"] = d["close"].shift(lag)
        d[f"ret_lag_{lag}"] = d["return_1d"].shift(lag)

    d["dayofweek"] = d["date"].dt.dayofweek
    d["dayofmonth"] = d["date"].dt.day
    d["month"] = d["date"].dt.month
    d["weekofyear"] = d["date"].dt.isocalendar().week.astype(int)
    return d.dropna().reset_index(drop=True)


def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return mae, rmse, mape


def train_best_model(df_feat, backtest_size):
    feature_cols = [c for c in df_feat.columns if c not in ["date", "close"]]
    X = df_feat[feature_cols]
    y = df_feat["close"]
    split_idx = len(df_feat) - backtest_size
    if split_idx <= 30:
        raise ValueError("Dữ liệu quá ít. Bạn cần nhiều phiên giao dịch hơn để train/backtest.")

    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    candidates = {
        "ExtraTrees": ExtraTreesRegressor(n_estimators=500, max_depth=12, min_samples_leaf=2, random_state=42, n_jobs=-1),
        "RandomForest": RandomForestRegressor(n_estimators=400, max_depth=10, min_samples_leaf=2, random_state=42, n_jobs=-1),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=300, learning_rate=0.03, max_depth=3, random_state=42),
    }

    best_name = None
    best_model = None
    best_mape = 1e18
    best_pred = None
    best_resid_std = None
    rows = []

    for name, model in candidates.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        mae, rmse, mape = evaluate(y_test, pred)
        resid_std = float(np.std(y_test - pred))
        rows.append({"model": name, "MAE": round(mae, 4), "RMSE": round(rmse, 4), "MAPE_pct": round(mape, 4)})
        if mape < best_mape:
            best_name = name
            best_model = model
            best_mape = mape
            best_pred = pred
            best_resid_std = resid_std

    metrics_df = pd.DataFrame(rows).sort_values("MAPE_pct").reset_index(drop=True)
    return best_name, best_model, feature_cols, metrics_df, y_test, best_pred, best_resid_std


def next_business_days(last_date, periods):
    return pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=periods)


def build_next_row(history_df, future_date):
    hist = history_df.copy().sort_values("date").reset_index(drop=True)
    row = {
        "open": hist["close"].iloc[-1],
        "high": hist["close"].iloc[-1],
        "low": hist["close"].iloc[-1],
        "volume": hist["volume"].tail(5).mean(),
        "change_pct": hist["close"].pct_change().iloc[-1],
        "return_1d": hist["close"].pct_change().iloc[-1],
        "range_pct": hist["range_pct"].tail(5).mean(),
        "oc_pct": hist["oc_pct"].tail(5).mean(),
    }
    close_s = hist["close"]
    ret_s = close_s.pct_change()
    for w in [3, 5, 7, 14, 21]:
        row[f"ma_{w}"] = close_s.tail(w).mean()
        row[f"std_{w}"] = close_s.tail(w).std()
        row[f"vol_ma_{w}"] = hist["volume"].tail(w).mean()
    for lag in [1, 2, 3, 5, 7, 10, 14, 21]:
        row[f"close_lag_{lag}"] = close_s.iloc[-lag]
        row[f"ret_lag_{lag}"] = ret_s.iloc[-lag] if len(ret_s.dropna()) >= lag else 0.0
    row["dayofweek"] = pd.Timestamp(future_date).dayofweek
    row["dayofmonth"] = pd.Timestamp(future_date).day
    row["month"] = pd.Timestamp(future_date).month
    row["weekofyear"] = int(pd.Timestamp(future_date).isocalendar().week)
    return pd.DataFrame([row])


def recursive_forecast(model, df_feat, feature_cols, horizon, resid_std):
    hist = df_feat.copy().sort_values("date").reset_index(drop=True)
    future_dates = next_business_days(hist["date"].max(), horizon)
    out = []
    for d in future_dates:
        x_next = build_next_row(hist, d)
        for c in feature_cols:
            if c not in x_next.columns:
                x_next[c] = 0.0
        x_next = x_next[feature_cols].ffill().bfill().fillna(0.0)
        yhat = float(model.predict(x_next)[0])
        lower = yhat - 1.96 * resid_std
        upper = yhat + 1.96 * resid_std
        out.append({"date": pd.Timestamp(d), "forecast_close": round(yhat, 2), "lower_95": round(lower, 2), "upper_95": round(upper, 2)})
        new_row = {
            "date": pd.Timestamp(d),
            "close": yhat,
            "open": hist["close"].iloc[-1],
            "high": max(yhat, hist["close"].iloc[-1]),
            "low": min(yhat, hist["close"].iloc[-1]),
            "volume": hist["volume"].tail(5).mean(),
            "change_pct": (yhat - hist["close"].iloc[-1]) / hist["close"].iloc[-1] if hist["close"].iloc[-1] != 0 else 0.0,
        }
        hist = pd.concat([hist, pd.DataFrame([new_row])], ignore_index=True)
        hist["return_1d"] = hist["close"].pct_change()
        hist["range_pct"] = (hist["high"] - hist["low"]) / hist["low"].replace(0, np.nan)
        hist["oc_pct"] = (hist["close"] - hist["open"]) / hist["open"].replace(0, np.nan)
    return pd.DataFrame(out)


def make_main_chart(actual_df, forecast_df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=actual_df["date"].tail(120), y=actual_df["close"].tail(120), mode="lines", name="Giá thực tế"))
    fig.add_trace(go.Scatter(x=forecast_df["date"], y=forecast_df["upper_95"], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=forecast_df["date"], y=forecast_df["lower_95"], mode="lines", fill="tonexty", line=dict(width=0), name="Vùng tin cậy 95%", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=forecast_df["date"], y=forecast_df["forecast_close"], mode="lines+markers", name="Dự báo 30 ngày tới"))
    fig.update_layout(title="Giá lịch sử + Dự báo 30 ngày giao dịch tới", xaxis_title="Ngày", yaxis_title="Giá", height=520)
    return fig


def make_backtest_chart(actual_dates, y_test, y_pred):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=actual_dates, y=y_test, mode="lines", name="Actual"))
    fig.add_trace(go.Scatter(x=actual_dates, y=y_pred, mode="lines", name="Predicted"))
    fig.update_layout(title="Backtest trên tập kiểm tra", xaxis_title="Ngày", yaxis_title="Giá", height=420)
    return fig


def show_cards(latest_close, forecast_df, model_name):
    mean_fc = float(forecast_df["forecast_close"].mean())
    min_fc = float(forecast_df["forecast_close"].min())
    max_fc = float(forecast_df["forecast_close"].max())
    diff_pct = (mean_fc - latest_close) / latest_close * 100 if latest_close != 0 else 0
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Giá đóng cửa mới nhất", f"{latest_close:,.2f}")
    c2.metric("Mô hình chọn", model_name)
    c3.metric("Giá dự báo TB", f"{mean_fc:,.2f}")
    c4.metric("Giá dự báo thấp nhất", f"{min_fc:,.2f}")
    c5.metric("Chênh lệch TB", f"{diff_pct:,.2f}%")


if run_btn:
    try:
        with st.spinner("Đang tải dữ liệu, huấn luyện mô hình và tạo dashboard..."):
            if uploaded_file is not None:
                raw_df = load_data_from_upload(uploaded_file)
            elif source_url.strip():
                raw_df = load_data_from_url(source_url)
            else:
                st.warning("Bạn hãy nhập link dữ liệu hoặc upload file CSV/XLSX.")
                st.stop()

            clean_df = clean_data(raw_df)
            feat_df = add_features(clean_df)
            if len(feat_df) < 80:
                st.error("Dữ liệu quá ít để dự báo tốt. Bạn nên dùng lịch sử dài hơn.")
                st.stop()

            model_name, model, feature_cols, metrics_df, y_test, y_pred, resid_std = train_best_model(feat_df, backtest_size=backtest_size)
            forecast_df = recursive_forecast(model=model, df_feat=feat_df, feature_cols=feature_cols, horizon=forecast_horizon, resid_std=resid_std)
            forecast_df["run_timestamp"] = pd.Timestamp.now()
            metrics_df["selected_model"] = metrics_df["model"].eq(model_name).map({True: "YES", False: "NO"})
            latest_close = float(feat_df["close"].iloc[-1])

            st.success("Dự báo xong.")
            with st.expander("Xem dữ liệu sau khi app đã làm sạch", expanded=False):
                st.dataframe(clean_df.head(20), use_container_width=True, hide_index=True)

            show_cards(latest_close, forecast_df, model_name)
            c1, c2 = st.columns([2, 1])
            with c1:
                st.plotly_chart(make_main_chart(feat_df, forecast_df), use_container_width=True)
            with c2:
                st.subheader("Model metrics")
                st.dataframe(metrics_df, use_container_width=True, hide_index=True)

            actual_dates = feat_df["date"].iloc[-len(y_test):]
            st.plotly_chart(make_backtest_chart(actual_dates, y_test.values, y_pred), use_container_width=True)
            st.subheader("Bảng dự báo 30 ngày tới")
            st.dataframe(forecast_df, use_container_width=True, hide_index=True)

            csv_forecast = forecast_df.to_csv(index=False).encode("utf-8")
            csv_metrics = metrics_df.to_csv(index=False).encode("utf-8")
            csv_history = feat_df.to_csv(index=False).encode("utf-8")
            d1, d2, d3 = st.columns(3)
            d1.download_button("⬇️ Tải forecast CSV", csv_forecast, "forecast_30d.csv", "text/csv", use_container_width=True)
            d2.download_button("⬇️ Tải metrics CSV", csv_metrics, "model_metrics.csv", "text/csv", use_container_width=True)
            d3.download_button("⬇️ Tải history clean CSV", csv_history, "history_clean.csv", "text/csv", use_container_width=True)
            st.info("Nếu dữ liệu nguồn thay đổi theo ngày, forecast cũng sẽ thay đổi khi bạn mở lại app và bấm chạy lại.")
    except Exception as e:
        st.error(f"Lỗi: {e}")
        st.warning("Nếu link web bị chặn, hãy upload CSV/XLSX trực tiếp.")
else:
    st.markdown(
        """
        ### Cách dùng
        1. Dán link dữ liệu hoặc upload file CSV/XLSX.
        2. Bấm **Chạy dự báo**.
        3. App sẽ tự làm sạch dữ liệu, chọn mô hình tốt nhất, dự báo **30 ngày giao dịch tiếp theo** và vẽ dashboard.

        **App này đã tự xử lý thêm:**
        - file CSV lỗi encoding
        - file CSV bị dồn vào 1 cột
        - header tiếng Việt bị méo
        """
    )
