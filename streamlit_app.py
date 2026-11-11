import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# -----------------------------------------------------------------------------
# CONFIG & PAGE TITLE
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="XAUUSD BBMA Oma Ally Multi-TF Matrix",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("⚡ XAUUSD BBMA Oma Ally - Multi-Timeframe Screener & TP/SL Calculator")
st.markdown(
    "**Analisis Hirarki BBMA Oma Ally:** TF1 menentukan acuan Re-Entry Tren, "
    "sedangkan TF2 & TF3 memvalidasi struktur (Momentum, Extreme, MHV, CSAK, dan Re-Entry Eksekusi)."
)
st.markdown("---")

# -----------------------------------------------------------------------------
# SIDEBAR - RISK & MONEY MANAGEMENT CALCULATOR
# -----------------------------------------------------------------------------
st.sidebar.header("🛡️ Money Management Settings")
balance = st.sidebar.number_input(
    "Modal Akun ($)", min_value=10.0, value=1000.0, step=100.0
)
risk_pct = st.sidebar.slider(
    "Risiko per Trade (%)", min_value=0.5, max_value=5.0, value=1.0, step=0.5
)
risk_amount = balance * (risk_pct / 100.0)
st.sidebar.info(f"💵 Batas Toleransi Risiko: **${risk_amount:,.2f}**")


# -----------------------------------------------------------------------------
# DATA FETCHING & RESAMPLING
# -----------------------------------------------------------------------------
@st.cache_data(ttl=60)
def fetch_tf_data(interval, period):
  safe_periods = {
      "1m": "5d",
      "5m": "30d",
      "15m": "60d",
      "1h": "730d",
      "1d": "max",
  }
  actual_period = period
  if interval in ["1m", "5m"] and period in ["60d", "14d"]:
    actual_period = safe_periods.get(interval, "5d")

  tickers = ["GC=F", "XAUUSD=X", "GLD"]
  for ticker in tickers:
    try:
      df = yf.download(
          ticker, period=actual_period, interval=interval, progress=False
      )
      if not df.empty:
        if isinstance(df.columns, pd.MultiIndex):
          clean_df = pd.DataFrame()
          clean_df["Open"] = df["Open"].iloc[:, 0]
          clean_df["High"] = df["High"].iloc[:, 0]
          clean_df["Low"] = df["Low"].iloc[:, 0]
          clean_df["Close"] = df["Close"].iloc[:, 0]
          df = clean_df
        df = df.dropna(subset=["Open", "High", "Low", "Close"])
        if not df.empty:
          return df
    except Exception:
      continue
  return pd.DataFrame()


def resample_to_h4(df_h1):
  if df_h1.empty:
    return pd.DataFrame()
  df_h4 = df_h1.resample("4h").agg({
      "Open": "first",
      "High": "max",
      "Low": "min",
      "Close": "last",
  })
  return df_h4.dropna()


# -----------------------------------------------------------------------------
# BBMA OMA ALLY ANALYSIS LOGIC WITH STRICT TRIGGER FILTER
# -----------------------------------------------------------------------------
def check_historical_trigger(df, expected_bias):
  """Mendeteksi riwayat CSAK atau Momentum dalam 5 candle terakhir

  yang wajib searah dengan bias tren (mencegah salah baca arah).
  """
  if len(df) < 10:
    return "Data Terlalu Singkat"

  for i in range(1, 6):
    row = df.iloc[-i]
    prev_row = df.iloc[-(i + 1)]

    if expected_bias == "BUY":
      # Momentum Buy: Close/Low menembus Bottom BB
      if row["Close"] <= row["BottomBB"] or row["Low"] <= row["BottomBB"]:
        return "Valid (Didasari Momentum Buy)"
      # CSAK Buy: Close memotong ke atas MA20 (Mid BB)
      if row["Close"] > row["MA20"] and prev_row["Close"] <= prev_row["MA20"]:
        return "Valid (Didasari CSAK Buy)"

    elif expected_bias == "SELL":
      # Momentum Sell: Close/High menembus Top BB
      if row["Close"] >= row["TopBB"] or row["High"] >= row["TopBB"]:
        return "Valid (Didasari Momentum Sell)"
      # CSAK Sell: Close memotong ke bawah MA20 (Mid BB)
      if row["Close"] < row["MA20"] and prev_row["Close"] >= prev_row["MA20"]:
        return "Valid (Didasari CSAK Sell)"

  return "Peringatan: Tanpa Trigger Baku yang Searah"


def analyze_bbma_tf(df, is_tf1=False):
  if df.empty or len(df) < 20:
    return {
        "price": 0.0,
        "status": "-",
        "bias": "NEUTRAL",
        "top_bb": 0.0,
        "bot_bb": 0.0,
        "mid_bb": 0.0,
        "ma5_h": 0.0,
        "ma5_l": 0.0,
        "detail": "Data tidak mencukupi",
    }

  df = df.copy()
  df["MA20"] = df["Close"].rolling(window=20).mean()
  df["STD"] = df["Close"].rolling(window=20).std()
  df["TopBB"] = df["MA20"] + (df["STD"] * 2)
  df["BottomBB"] = df["MA20"] - (df["STD"] * 2)
  df["MA5_High"] = df["High"].rolling(window=5).mean()
  df["MA5_Low"] = df["Low"].rolling(window=5).mean()

  df = df.dropna()
  if df.empty or len(df) < 2:
    return {
        "price": 0.0,
        "status": "-",
        "bias": "NEUTRAL",
        "top_bb": 0.0,
        "bot_bb": 0.0,
        "mid_bb": 0.0,
        "ma5_h": 0.0,
        "ma5_l": 0.0,
        "detail": "Data kosong setelah indikator",
    }

  latest = df.iloc[-1]
  prev = df.iloc[-2]

  c_price = float(latest["Close"])
  top_bb = float(latest["TopBB"])
  bot_bb = float(latest["BottomBB"])
  mid_bb = float(latest["MA20"])
  ma5_h = float(latest["MA5_High"])
  ma5_l = float(latest["MA5_Low"])
  high = float(latest["High"])
  low = float(latest["Low"])

  status = "-"
  bias = "NEUTRAL"

  # Tentukan Preliminary Bias berdasarkan posisi harga terhadap Mid BB dan MA5
  if c_price >= mid_bb:
    prelim_bias = "BUY"
  else:
    prelim_bias = "SELL"

  # Cek Validasi Historis Trigger sesuai arah prelim bias
  trigger_status = check_historical_trigger(df, prelim_bias)

  if is_tf1:
    if (c_price <= ma5_h) and (c_price >= mid_bb or low >= mid_bb):
      status = "VALID RE-ENTRY BUY (TF1 Tren)"
      bias = "BUY"
      trigger_status = check_historical_trigger(df, "BUY")
      detail = f"Zona MA5/10 Low - Mid BB. {trigger_status}"
    elif (c_price >= ma5_l) and (c_price <= mid_bb or high <= mid_bb):
      status = "VALID RE-ENTRY SELL (TF1 Tren)"
      bias = "SELL"
      trigger_status = check_historical_trigger(df, "SELL")
      detail = f"Zona MA5/10 High - Mid BB. {trigger_status}"
    elif c_price > mid_bb:
      status = "BULLISH BIAS (Menunggu Re-Entry Buy)"
      bias = "BUY"
      detail = f"Harga di atas Mid BB. {trigger_status}"
    else:
      status = "BEARISH BIAS (Menunggu Re-Entry Sell)"
      bias = "SELL"
      detail = f"Harga di bawah Mid BB. {trigger_status}"
  else:
    if c_price >= top_bb or (high >= top_bb and prev["High"] >= prev["TopBB"]):
      status = "MOMENTUM / EXTREME SELL"
      bias = "SELL"
      trigger_status = check_historical_trigger(df, "SELL")
      detail = f"Terbentuk Extreme/Momentum di Top BB. {trigger_status}"
    elif c_price <= bot_bb or (
        low <= bot_bb and prev["Low"] <= prev["BottomBB"]
    ):
      status = "MOMENTUM / EXTREME BUY"
      bias = "BUY"
      trigger_status = check_historical_trigger(df, "BUY")
      detail = f"Terbentuk Extreme/Momentum di Bottom BB. {trigger_status}"
    elif c_price >= ma5_h and c_price > mid_bb:
      status = "RE-ENTRY / CSAK SELL"
      bias = "SELL"
      trigger_status = check_historical_trigger(df, "SELL")
      detail = f"Area Re-Entry/CSAK Sell. {trigger_status}"
    elif c_price <= ma5_l and c_price < mid_bb:
      status = "RE-ENTRY / CSAK BUY"
      bias = "BUY"
      trigger_status = check_historical_trigger(df, "BUY")
      detail = f"Area Re-Entry/CSAK Buy. {trigger_status}"
    elif c_price > mid_bb:
      status = "BULLISH CONTINUATION"
      bias = "BUY"
      detail = f"Struktur Bullish Berkelanjutan. {trigger_status}"
    else:
      status = "BEARISH CONTINUATION"
      bias = "SELL"
      detail = f"Struktur Bearish Berkelanjutan. {trigger_status}"

  return {
      "price": c_price,
      "status": status,
      "bias": bias,
      "top_bb": top_bb,
      "bot_bb": bot_bb,
      "mid_bb": mid_bb,
      "ma5_h": ma5_h,
      "ma5_l": ma5_l,
      "detail": detail,
  }


# -----------------------------------------------------------------------------
# FETCH DATA FEED
# -----------------------------------------------------------------------------
with st.spinner("Mengambil feed data multi-timeframe XAUUSD..."):
  df_1d = fetch_tf_data("1d", "60d")
  df_1h = fetch_tf_data("1h", "14d")
  df_15m = fetch_tf_data("15m", "5d")
  df_5m = fetch_tf_data("5m", "1d")
  df_1m = fetch_tf_data("1m", "1d")
  df_4h = resample_to_h4(df_1h)


def get_tf_dict(tf1_code):
  return {
      "1d": analyze_bbma_tf(df_1d, is_tf1=(tf1_code == "1d")),
      "4h": analyze_bbma_tf(df_4h, is_tf1=(tf1_code == "4h")),
      "1h": analyze_bbma_tf(df_1h, is_tf1=(tf1_code == "1h")),
      "15m": analyze_bbma_tf(df_15m, is_tf1=(tf1_code == "15m")),
      "5m": analyze_bbma_tf(df_5m, is_tf1=(tf1_code == "5m")),
      "1m": analyze_bbma_tf(df_1m, is_tf1=(tf1_code == "1m")),
  }


tf_labels = {
    "1d": "Daily (D1)",
    "4h": "H4 (Resampled)",
    "1h": "H1",
    "15m": "M15",
    "5m": "M5",
    "1m": "M1",
}


# -----------------------------------------------------------------------------
# TRADE LEVELS & RISK MANAGEMENT CALCULATOR
# -----------------------------------------------------------------------------
def calculate_trade_levels(
    bias, current_price, top_bb, mid_bb, bot_bb, account_risk
):
  if bias == "BUY":
    entry = current_price
    sl = round(bot_bb - 3.0, 2)
    tp1 = round(mid_bb, 2)
    tp2 = round(top_bb - 4.0, 2)
    tp3 = round(top_bb, 2)

    risk_pips = round((entry - sl) * 10, 1)
    reward_pips = round((tp1 - entry) * 10, 1)
    rrr = round(reward_pips / risk_pips, 2) if risk_pips > 0 else 0.0
    lot_size = (
        round(account_risk / (risk_pips * 10), 2) if risk_pips > 0 else 0.0
    )

  elif bias == "SELL":
    entry = current_price
    sl = round(top_bb + 3.0, 2)
    tp1 = round(mid_bb, 2)
    tp2 = round(bot_bb + 4.0, 2)
    tp3 = round(bot_bb, 2)

    risk_pips = round((sl - entry) * 10, 1)
    reward_pips = round((entry - tp1) * 10, 1)
    rrr = round(reward_pips / risk_pips, 2) if risk_pips > 0 else 0.0
    lot_size = (
        round(account_risk / (risk_pips * 10), 2) if risk_pips > 0 else 0.0
    )
  else:
    entry, sl, tp1, tp2, tp3, rrr, risk_pips, lot_size = (
        0,
        0,
        0,
        0,
        0,
        0.0,
        0.0,
        0.0,
    )

  return entry, sl, tp1, tp2, tp3, rrr, risk_pips, max(lot_size, 0.01)


def evaluate_combo(tf1_code, tf2_code, tf3_code):
  current_tf_data = get_tf_dict(tf1_code)
  d1 = current_tf_data[tf1_code]
  d2 = current_tf_data[tf2_code]
  d3 = current_tf_data[tf3_code]

  valid_signal = "-"
  kesimpulan = "-"
  dominant_bias = d1["bias"]

  if "BUY" in dominant_bias:
    if "BUY" in d2["bias"] and "BUY" in d3["bias"]:
      valid_signal = "VALID SETUP BBMA (BUY)"
      kesimpulan = (
          f"✅ **Alignment Sempurna!** {tf_labels[tf1_code]} acuan Re-Entry Buy,"
          f" divalidasi penuh oleh {tf_labels[tf2_code]} &"
          f" {tf_labels[tf3_code]}."
      )
    elif "BUY" in d2["bias"]:
      valid_signal = "MENUNGGU KONFIRMASI TF3"
      kesimpulan = (
          f"⏳ {tf_labels[tf1_code]} & {tf_labels[tf2_code]} searah Buy. Tunggu"
          f" konfirmasi trigger di {tf_labels[tf3_code]}."
      )
    else:
      valid_signal = "TIDAK VALID"
      kesimpulan = (
          f"❌ **Belum Valid.** {tf_labels[tf2_code]} belum mengonfirmasi arah"
          " Buy."
      )

  elif "SELL" in dominant_bias:
    if "SELL" in d2["bias"] and "SELL" in d3["bias"]:
      valid_signal = "VALID SETUP BBMA (SELL)"
      kesimpulan = (
          f"✅ **Alignment Sempurna!** {tf_labels[tf1_code]} acuan Re-Entry"
          f" Sell, divalidasi penuh oleh {tf_labels[tf2_code]} &"
          f" {tf_labels[tf3_code]}."
      )
    elif "SELL" in d2["bias"]:
      valid_signal = "MENUNGGU KONFIRMASI TF3"
      kesimpulan = (
          f"⏳ {tf_labels[tf1_code]} & {tf_labels[tf2_code]} searah Sell. Tunggu"
          f" konfirmasi trigger di {tf_labels[tf3_code]}."
      )
    else:
      valid_signal = "TIDAK VALID"
      kesimpulan = (
          f"❌ **Belum Valid.** {tf_labels[tf2_code]} belum mengonfirmasi arah"
          " Sell."
      )

  return {
      "valid_signal": valid_signal,
      "dominant_bias": dominant_bias,
      "kesimpulan": kesimpulan,
      "tf_data_dict": current_tf_data,
  }


# -----------------------------------------------------------------------------
# MAIN INTERFACE DISPLAY
# -----------------------------------------------------------------------------
live_price = df_1h["Close"].iloc[-1] if not df_1h.empty else 0.0
st.metric("Harga Live XAUUSD", f"${live_price:,.2f}")

st.markdown("---")
st.subheader("🎯 Matriks Hirarki BBMA & Kalkulator Ukuran Lot")

tab1, tab2, tab3, tab4 = st.tabs([
    "D1 ➔ H4 ➔ H1",
    "H4 ➔ H1 ➔ M15",
    "H1 ➔ M15 ➔ M5",
    "M15 ➔ M5 ➔ M1",
])


def render_tab_content(tf1, tf2, tf3):
  eval_res = evaluate_combo(tf1, tf2, tf3)
  current_tf_data = eval_res["tf_data_dict"]
  d_exec = current_tf_data[tf3]
  execution_bias = eval_res["dominant_bias"]

  entry, sl, tp1, tp2, tp3, rrr, risk_pips, lot_size = calculate_trade_levels(
      execution_bias,
      d_exec["price"],
      d_exec["top_bb"],
      d_exec["mid_bb"],
      d_exec["bot_bb"],
      risk_amount,
  )

  col1, col2, col3, col4 = st.columns(4)
  col1.metric("Status Sinyal Setup", eval_res["valid_signal"])
  col2.metric(
      "Risk-to-Reward (RRR)", f"1 : {rrr}" if rrr > 0 else "-"
  )
  col3.metric("Rekomendasi Lot", f"{lot_size} Lot" if lot_size > 0 else "-")
  col4.metric("Bias Acuan Tren (TF1)", execution_bias)

  if "VALID SETUP" in eval_res["valid_signal"]:
    st.success(eval_res["kesimpulan"])
  elif "MENUNGGU" in eval_res["valid_signal"]:
    st.warning(eval_res["kesimpulan"])
  else:
    st.error(eval_res["kesimpulan"])

  # TABEL MATRIKS TIMEFRAME
  st.markdown("#### 📌 Matriks Kondisi Indikator Hirarki & Validasi Trigger")
  matrix_rows = []
  for code in [tf1, tf2, tf3]:
    res = current_tf_data[code]
    role_label = (
        f"TF1 (Acuan Re-Entry - {tf_labels[code]})"
        if code == tf1
        else (
            f"TF2 (Validasi - {tf_labels[code]})"
            if code == tf2
            else f"TF3 (Eksekusi - {tf_labels[code]})"
        )
    )
    matrix_rows.append({
        "Peran & Timeframe": role_label,
        "Status / Kondisi": res["status"],
        "Arah Bias": res["bias"],
        "Top BB": f"${res['top_bb']:,.2f}" if res["top_bb"] > 0 else "-",
        "Mid BB (MA20)": f"${res['mid_bb']:,.2f}" if res["mid_bb"] > 0 else "-",
        "Bottom BB": f"${res['bot_bb']:,.2f}" if res["bot_bb"] > 0 else "-",
        "Keterangan & Riwayat Trigger": res["detail"],
    })
  st.dataframe(pd.DataFrame(matrix_rows), use_container_width=True)

  # TABEL KALKULATOR RENCANA PERDAGANGAN
  icon_label = "🟢" if execution_bias == "BUY" else "🔴"
  st.markdown(
      f"#### 📋 Rencana Eksekusi Trade & Manajemen Risiko ({icon_label}"
      f" {execution_bias})"
  )

  trade_table = pd.DataFrame([
      {
          "Parameter": "Harga Entry Saat Ini (TF3)",
          "Nilai Level Harga": f"${entry:,.2f}" if entry > 0 else "-",
      },
      {
          "Parameter": "Stop Loss (SL)",
          "Nilai Level Harga": f"${sl:,.2f}" if sl > 0 else "-",
      },
      {
          "Parameter": "Jarak Risiko (Pips / Points)",
          "Nilai Level Harga": (
              f"{risk_pips} Pips ({risk_pips*10:.0f} Points)"
              if risk_pips > 0
              else "-"
          ),
      },
      {
          "Parameter": "Take Profit 1 (Mid BB)",
          "Nilai Level Harga": f"${tp1:,.2f}" if tp1 > 0 else "-",
      },
      {
          "Parameter": "Take Profit 2 (Buffer Area)",
          "Nilai Level Harga": f"${tp2:,.2f}" if tp2 > 0 else "-",
      },
      {
          "Parameter": "Take Profit 3 (Extreme BB)",
          "Nilai Level Harga": f"${tp3:,.2f}" if tp3 > 0 else "-",
      },
  ])
  st.dataframe(trade_table, use_container_width=True)


with tab1:
  render_tab_content("1d", "4h", "1h")

with tab2:
  render_tab_content("4h", "1h", "15m")

with tab3:
  render_tab_content("1h", "15m", "5m")

with tab4:
  render_tab_content("15m", "5m", "1m")
