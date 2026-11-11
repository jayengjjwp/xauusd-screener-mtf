import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# -----------------------------------------------------------------------------
# CONFIG & PAGE TITLE
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="XAUUSD BBMA Oma Ally Pro Matrix",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title(
    "⚡ XAUUSD BBMA Oma Ally Pro - Advanced Multi-TF Screener & Risk Calculator"
)
st.markdown(
    "**Hirarki Ketat Berbasis Aturan Pro:** TF1 (Acuan Re-Entry) ➔ TF2 (Konfirmasi"
    " Extreme/CSAK/Reject BB & EMA 50) ➔ TF3 (Eksekusi Trigger Akhir)."
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
# DATA FETCHING & RESAMPLING (PERPANJANG PERIODE D1 AGAR EMA 50 AMAN)
# -----------------------------------------------------------------------------
@st.cache_data(ttl=60)
def fetch_tf_data(interval, period):
  safe_periods = {
      "1m": "5d",
      "5m": "30d",
      "15m": "60d",
      "1h": "730d",
      "1d": "1y",  # Diperpanjang ke 1 tahun agar data EMA 50 tidak kurang dari 55 baris
  }
  actual_period = safe_periods.get(interval, period)
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
# ADVANCED BBMA & EMA 50 ANALYSIS ENGINE
# -----------------------------------------------------------------------------
def analyze_bbma_tf(df, role_type="TF1"):
  if df.empty or len(df) < 55:
    return {
        "price": 0.0,
        "status": "DATA KURANG",
        "bias": "NEUTRAL",
        "top_bb": 0.0,
        "bot_bb": 0.0,
        "mid_bb": 0.0,
        "ema50": 0.0,
        "detail": (
            f"Data baris hanya {len(df)}, butuh min. 55 untuk kalkulasi EMA 50"
        ),
    }

  df = df.copy()
  df["MA20"] = df["Close"].rolling(window=20).mean()
  df["STD"] = df["Close"].rolling(window=20).std()
  df["TopBB"] = df["MA20"] + (df["STD"] * 2)
  df["BottomBB"] = df["MA20"] - (df["STD"] * 2)
  df["MA5_High"] = df["High"].rolling(window=5).mean()
  df["MA5_Low"] = df["High"].rolling(window=5).mean()  # Diperbaiki ke High/Low
  df["MA5_Low"] = df["Low"].rolling(window=5).mean()
  df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()

  df = df.dropna()
  if df.empty or len(df) < 2:
    return {
        "price": 0.0,
        "status": "KOSONG",
        "bias": "NEUTRAL",
        "top_bb": 0.0,
        "bot_bb": 0.0,
        "mid_bb": 0.0,
        "ema50": 0.0,
        "detail": "Data kosong setelah kalkulasi indikator",
    }

  latest = df.iloc[-1]
  prev = df.iloc[-2]

  c_price = float(latest["Close"])
  top_bb = float(latest["TopBB"])
  bot_bb = float(latest["BottomBB"])
  mid_bb = float(latest["MA20"])
  ema50 = float(latest["EMA50"])
  ma5_h = float(latest["MA5_High"])
  ma5_l = float(latest["MA5_Low"])
  high = float(latest["High"])
  low = float(latest["Low"])

  status = "-"
  bias = "NEUTRAL"
  detail = "-"

  # Deteksi Rejection terhadap EMA50 atau BB
  reject_top_bb = high >= top_bb and c_price < top_bb
  reject_bot_bb = low <= bot_bb and c_price > bot_bb
  reject_ema50_sell = high >= ema50 and c_price < ema50
  reject_ema50_buy = low <= ema50 and c_price > ema50

  # Deteksi Momentum
  momentum_sell = (
      c_price >= top_bb or (high >= top_bb and prev["High"] >= prev["TopBB"])
  )
  momentum_buy = (
      c_price <= bot_bb or (low <= bot_bb and prev["Low"] <= prev["BottomBB"])
  )

  # Deteksi CSAK
  body_size = abs(c_price - float(latest["Open"]))
  avg_body = abs(df["Close"] - df["Open"]).rolling(10).mean().iloc[-1]
  csak_buy = (
      body_size > (avg_body * 1.2)
      and c_price > mid_bb
      and prev["Close"] <= prev["MA20"]
  )
  csak_sell = (
      body_size > (avg_body * 1.2)
      and c_price < mid_bb
      and prev["Close"] >= prev["MA20"]
  )

  # LOGIKA BERDASARKAN PERAN TIMEFRAME (TF1, TF2, atau TF3)
  if role_type == "TF1":
    if (c_price <= ma5_h) and (c_price >= mid_bb or low >= mid_bb):
      status = "RE-ENTRY BUY (Acuan Tren)"
      bias = "BUY"
      detail = (
          "Zona Re-Entry Buy aktif di antara MA5/10 Low dan Mid BB (Didasari"
          " Tren Besar)"
      )
    elif (c_price >= ma5_l) and (c_price <= mid_bb or high <= mid_bb):
      status = "RE-ENTRY SELL (Acuan Tren)"
      bias = "SELL"
      detail = (
          "Zona Re-Entry Sell aktif di antara MA5/10 High dan Mid BB (Didasari"
          " Tren Besar)"
      )
    elif c_price > mid_bb:
      status = "BULLISH BIAS (Menunggu Re-Entry Buy)"
      bias = "BUY"
      detail = "Harga di atas Mid BB, menunggu pullback sehat ke zona Buy"
    else:
      status = "BEARISH BIAS (Menunggu Re-Entry Sell)"
      bias = "SELL"
      detail = "Harga di bawah Mid BB, menunggu pullback sehat ke zona Sell"

  elif role_type == "TF2":
    if momentum_buy or reject_bot_bb or reject_ema50_buy or c_price <= bot_bb:
      status = "KONFIRMASI BUY (Valid Reject/Extreme)"
      bias = "BUY"
      detail = "TF2 mengonfirmasi lewat Extreme/Momentum/Reject BB atau EMA 50"
    elif momentum_sell or reject_top_bb or reject_ema50_sell or c_price >= top_bb:
      status = "KONFIRMASI SELL (Valid Reject/Extreme)"
      bias = "SELL"
      detail = "TF2 mengonfirmasi lewat Extreme/Momentum/Reject BB atau EMA 50"
    elif csak_buy or (c_price >= ma5_h and c_price > mid_bb):
      status = "CSAK / RE-ENTRY BUY KONFIRMASI"
      bias = "BUY"
      detail = "TF2 membentuk CSAK Buy atau Re-Entry penunjang"
    elif csak_sell or (c_price <= ma5_l and c_price < mid_bb):
      status = "CSAK / RE-ENTRY SELL KONFIRMASI"
      bias = "SELL"
      detail = "TF2 membentuk CSAK Sell atau Re-Entry penunjang"
    elif c_price > mid_bb:
      status = "TF2 BULLISH CONTINUATION"
      bias = "BUY"
      detail = "Struktur TF2 mendukung kenaikan"
    else:
      status = "TF2 BEARISH CONTINUATION"
      bias = "SELL"
      detail = "Struktur TF2 mendukung penurunan"

  elif role_type == "TF3":
    if momentum_buy:
      status = "TRIGGER MOMENTUM BUY"
      bias = "BUY"
      detail = "TF3 membentuk Momentum Buy sebagai trigger eksekusi"
    elif momentum_sell:
      status = "TRIGGER MOMENTUM SELL"
      bias = "SELL"
      detail = "TF3 membentuk Momentum Sell sebagai trigger eksekusi"
    elif csak_buy:
      status = "TRIGGER CSAK BUY"
      bias = "BUY"
      detail = "TF3 membentuk CSAK Buy (Full Setup / Trigger)"
    elif csak_sell:
      status = "TRIGGER CSAK SELL"
      bias = "SELL"
      detail = "TF3 membentuk CSAK Sell (Full Setup / Trigger)"
    elif (c_price <= ma5_l and c_price < mid_bb) or reject_bot_bb:
      status = "TRIGGER RE-ENTRY / EXTREME BUY"
      bias = "BUY"
      detail = "TF3 siap eksekusi Re-Entry / Extreme Buy"
    elif (c_price >= ma5_h and c_price > mid_bb) or reject_top_bb:
      status = "TRIGGER RE-ENTRY / EXTREME SELL"
      bias = "SELL"
      detail = "TF3 siap eksekusi Re-Entry / Extreme Sell"
    elif c_price > mid_bb:
      status = "TF3 BULLISH MENGALIR"
      bias = "BUY"
      detail = "Kondisi TF3 mengalir searah Buy"
    else:
      status = "TF3 BEARISH MENGALIR"
      bias = "SELL"
      detail = "Kondisi TF3 mengalir searah Sell"

  return {
      "price": c_price,
      "status": status,
      "bias": bias,
      "top_bb": top_bb,
      "bot_bb": bot_bb,
      "mid_bb": mid_bb,
      "ema50": ema50,
      "detail": detail,
  }


# -----------------------------------------------------------------------------
# FETCH DATA FEED
# -----------------------------------------------------------------------------
with st.spinner("Mengambil feed data multi-timeframe XAUUSD..."):
  df_1d = fetch_tf_data("1d", "1y")
  df_1h = fetch_tf_data("1h", "14d")
  df_15m = fetch_tf_data("15m", "5d")
  df_5m = fetch_tf_data("5m", "1d")
  df_1m = fetch_tf_data("1m", "1d")
  df_4h = resample_to_h4(df_1h)


def get_tf_dict(tf1_code, tf2_code, tf3_code):
  return {
      "tf1": analyze_bbma_tf(
          globals()[f"df_{tf1_code}"], role_type="TF1"
      ),
      "tf2": analyze_bbma_tf(
          globals()[f"df_{tf2_code}"], role_type="TF2"
      ),
      "tf3": analyze_bbma_tf(
          globals()[f"df_{tf3_code}"], role_type="TF3"
      ),
  }


tf_labels = {
    "1d": "Daily (D1)",
    "4h": "H4",
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
  data = get_tf_dict(tf1_code, tf2_code, tf3_code)
  d1 = data["tf1"]
  d2 = data["tf2"]
  d3 = data["tf3"]

  dominant_bias = d1["bias"]
  valid_signal = "-"
  kesimpulan = "-"

  if "BUY" in dominant_bias:
    if "BUY" in d2["bias"] and "BUY" in d3["bias"]:
      valid_signal = "🔥 PRO SETUP VALID (BUY)"
      kesimpulan = (
          f"✅ **Alignment Sempurna!** {tf_labels[tf1_code]} memberi acuan Re-Entry"
          f" Buy, dikonfirmasi oleh {tf_labels[tf2_code]} (Reject/Extreme/CSAK),"
          f" dan dieksekusi lewat trigger di {tf_labels[tf3_code]}."
      )
    elif "BUY" in d2["bias"]:
      valid_signal = "⏳ MENUNGGU TRIGGER TF3"
      kesimpulan = (
          f"⏳ TF1 & TF2 searah Buy. Tunggu konfirmasi trigger eksekusi matang"
          f" di {tf_labels[tf3_code]}."
      )
    else:
      valid_signal = "❌ BELUM VALID"
      kesimpulan = (
          f"❌ TF2 belum mengonfirmasi arah Buy atau masih melawan arah tren"
          " besar."
      )

  elif "SELL" in dominant_bias:
    if "SELL" in d2["bias"] and "SELL" in d3["bias"]:
      valid_signal = "🔥 PRO SETUP VALID (SELL)"
      kesimpulan = (
          f"✅ **Alignment Sempurna!** {tf_labels[tf1_code]} memberi acuan"
          f" Re-Entry Sell, dikonfirmasi oleh {tf_labels[tf2_code]}"
          f" (Reject/Extreme/CSAK), dan dieksekusi lewat trigger di"
          f" {tf_labels[tf3_code]}."
      )
    elif "SELL" in d2["bias"]:
      valid_signal = "⏳ MENUNGGU TRIGGER TF3"
      kesimpulan = (
          f"⏳ TF1 & TF2 searah Sell. Tunggu konfirmasi trigger eksekusi matang"
          f" di {tf_labels[tf3_code]}."
      )
    else:
      valid_signal = "❌ BELUM VALID"
      kesimpulan = (
          f"❌ TF2 belum mengonfirmasi arah Sell atau masih melawan arah tren"
          " besar."
      )

  return {
      "valid_signal": valid_signal,
      "dominant_bias": dominant_bias,
      "kesimpulan": kesimpulan,
      "tf_data": data,
  }


# -----------------------------------------------------------------------------
# MAIN INTERFACE DISPLAY
# -----------------------------------------------------------------------------
live_price = df_1h["Close"].iloc[-1] if not df_1h.empty else 0.0
st.metric("Harga Live XAUUSD", f"${live_price:,.2f}")

st.markdown("---")
st.subheader(
    "🎯 Matriks Hirarki Pro BBMA (TF1 Re-Entry ➔ TF2 Konfirmasi ➔ TF3 Trigger)"
)

tab1, tab2, tab3, tab4 = st.tabs([
    "D1 ➔ H4 ➔ H1",
    "H4 ➔ H1 ➔ M15",
    "H1 ➔ M15 ➔ M5",
    "M15 ➔ M5 ➔ M1",
])


def render_tab_content(tf1, tf2, tf3):
  eval_res = evaluate_combo(tf1, tf2, tf3)
  data = eval_res["tf_data"]
  d_exec = data["tf3"]
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
  col1.metric("Status Setup Pro", eval_res["valid_signal"])
  col2.metric(
      "Risk-to-Reward (RRR)", f"1 : {rrr}" if rrr > 0 else "-"
  )
  col3.metric("Rekomendasi Lot", f"{lot_size} Lot" if lot_size > 0 else "-")
  col4.metric("Bias Acuan (TF1)", execution_bias)

  if "PRO SETUP VALID" in eval_res["valid_signal"]:
    st.success(eval_res["kesimpulan"])
  elif "MENUNGGU" in eval_res["valid_signal"]:
    st.warning(eval_res["kesimpulan"])
  else:
    st.error(eval_res["kesimpulan"])

  # TABEL MATRIKS TIMEFRAME
  st.markdown("#### 📌 Matriks Validasi Hirarki Pro (TF1 ➔ TF2 ➔ TF3)")
  matrix_rows = [
      {
          "Peran Hirarki": f"TF1 (Acuan Re-Entry - {tf_labels[tf1]})",
          "Status / Kondisi": data["tf1"]["status"],
          "Arah Bias": data["tf1"]["bias"],
          "Top BB": f"${data['tf1']['top_bb']:,.2f}",
          "Mid BB": f"${data['tf1']['mid_bb']:,.2f}",
          "EMA 50": f"${data['tf1']['ema50']:,.2f}",
          "Bottom BB": f"${data['tf1']['bot_bb']:,.2f}",
          "Keterangan Analisis": data["tf1"]["detail"],
      },
      {
          "Peran Hirarki": (
              f"TF2 (Konfirmasi Reject/Extreme/CSAK - {tf_labels[tf2]})"
          ),
          "Status / Kondisi": data["tf2"]["status"],
          "Arah Bias": data["tf2"]["bias"],
          "Top BB": f"${data['tf2']['top_bb']:,.2f}",
          "Mid BB": f"${data['tf2']['mid_bb']:,.2f}",
          "EMA 50": f"${data['tf2']['ema50']:,.2f}",
          "Bottom BB": f"${data['tf2']['bot_bb']:,.2f}",
          "Keterangan Analisis": data["tf2"]["detail"],
      },
      {
          "Peran Hirarki": (
              f"TF3 (Trigger Eksekusi Akhir - {tf_labels[tf3]})"
          ),
          "Status / Kondisi": data["tf3"]["status"],
          "Arah Bias": data["tf3"]["bias"],
          "Top BB": f"${data['tf3']['top_bb']:,.2f}",
          "Mid BB": f"${data['tf3']['mid_bb']:,.2f}",
          "EMA 50": f"${data['tf3']['ema50']:,.2f}",
          "Bottom BB": f"${data['tf3']['bot_bb']:,.2f}",
          "Keterangan Analisis": data["tf3"]["detail"],
      },
  ]
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
