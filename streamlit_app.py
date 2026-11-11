import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(
    page_title="XAUUSD BBMA Oma Ally Multi-TF Matrix", layout="wide"
)

st.title("⚡ XAUUSD BBMA Oma Ally - Multi-Timeframe Screener & TP/SL Calculator")
st.markdown(
    "Analisis hirarki ketat: TF1 khusus Re-Entry Tren, sedangkan TF2 & TF3 "
    "memvalidasi Momentum, Extreme, MHV, CSAK, dan Re-Entry."
)
st.markdown("---")


@st.cache_data(ttl=60)
def fetch_tf_data(interval, period):
  try:
    df = yf.download(
        "GC=F", period=period, interval=interval, progress=False
    )
    if df.empty:
      df = yf.download(
          "XAUUSD=X", period=period, interval=interval, progress=False
      )
    if isinstance(df.columns, pd.MultiIndex):
      clean_df = pd.DataFrame()
      clean_df["Open"] = df["Open"].iloc[:, 0]
      clean_df["High"] = df["High"].iloc[:, 0]
      clean_df["Low"] = df["Low"].iloc[:, 0]
      clean_df["Close"] = df["Close"].iloc[:, 0]
      df = clean_df
    return df
  except Exception:
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


def analyze_bbma_tf(df, is_tf1=False):
  if df.empty or len(df) < 20:
    return {
        "price": 0.0,
        "status": "-",
        "bias": "-",
        "top_bb": 0.0,
        "bot_bb": 0.0,
        "mid_bb": 0.0,
        "ma5_h": 0.0,
        "ma5_l": 0.0,
        "detail": "Data tidak mencukupi",
    }

  df["MA20"] = df["Close"].rolling(window=20).mean()
  df["STD"] = df["Close"].rolling(window=20).std()
  df["TopBB"] = df["MA20"] + (df["STD"] * 2)
  df["BottomBB"] = df["MA20"] - (df["STD"] * 2)
  df["MA5_High"] = df["High"].rolling(window=5).mean()
  df["MA5_Low"] = df["Low"].rolling(window=5).mean()

  df = df.dropna()
  if df.empty:
    return {
        "price": 0.0,
        "status": "-",
        "bias": "-",
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
  detail = "-"

  # JIKA TF1: HANYA FOKUS PADA RE-ENTRY SEBAGAI ACUAN TREN UTAMA
  if is_tf1:
    if c_price >= ma5_h and c_price > mid_bb:
      status = "RE-ENTRY SELL (TF1 Tren)"
      bias = "SELL"
      detail = "TF1 valid Re-Entry Sell di atas Mid BB (Area MA5/10 High)"
    elif c_price <= ma5_l and c_price < mid_bb:
      status = "RE-ENTRY BUY (TF1 Tren)"
      bias = "BUY"
      detail = "TF1 valid Re-Entry Buy di bawah Mid BB (Area MA5/10 Low)"
    elif c_price > mid_bb:
      status = "BULLISH BIAS (Menuju Top BB)"
      bias = "BUY"
      detail = "Harga di atas Mid BB, menunggu pullback ke area Re-Entry Buy"
    else:
      status = "BEARISH BIAS (Menuju Bottom BB)"
      bias = "SELL"
      detail = "Harga di bawah Mid BB, menunggu pullback ke area Re-Entry Sell"
  else:
    # UNTUK TF2 & TF3: DETEKSI MOMENTUM, EXTREME, MHV, CSAK, DAN RE-ENTRY
    if c_price >= top_bb or (high >= top_bb and prev["High"] >= prev["TopBB"]):
      status = "MOMENTUM / EXTREME SELL"
      bias = "SELL"
      detail = "Terbentuk Momentum / Extreme di Top BB (Valid Sell)"
    elif c_price <= bot_bb or (
        low <= bot_bb and prev["Low"] <= prev["BottomBB"]
    ):
      status = "MOMENTUM / EXTREME BUY"
      bias = "BUY"
      detail = "Terbentuk Momentum / Extreme di Bottom BB (Valid Buy)"
    elif c_price >= ma5_h and c_price > mid_bb:
      status = "RE-ENTRY / CSAK SELL"
      bias = "SELL"
      detail = "Konfirmasi area Re-Entry Sell / CSAK Sell"
    elif c_price <= ma5_l and c_price < mid_bb:
      status = "RE-ENTRY / CSAK BUY"
      bias = "BUY"
      detail = "Konfirmasi area Re-Entry Buy / CSAK Buy"
    elif c_price > mid_bb:
      status = "BULLISH CONTINUATION"
      bias = "BUY"
      detail = "Struktur harga mendukung kelanjutan naik"
    else:
      status = "BEARISH CONTINUATION"
      bias = "SELL"
      detail = "Struktur harga mendukung kelanjutan turun"

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


# Fetch Data Dasar
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


def calculate_trade_levels(bias, current_price, top_bb, mid_bb, bot_bb):
  if bias == "BUY":
    entry = current_price
    sl = round(bot_bb - 3.0, 2)
    tp1 = round(mid_bb, 2)
    tp2 = round(top_bb - 4.0, 2)
    tp3 = round(top_bb, 2)

    risk = entry - sl
    reward = tp1 - entry
    rrr = round(reward / risk, 2) if risk > 0 else 0.0

  elif bias == "SELL":
    entry = current_price
    sl = round(top_bb + 3.0, 2)
    tp1 = round(mid_bb, 2)
    tp2 = round(bot_bb + 4.0, 2)
    tp3 = round(bot_bb, 2)

    risk = sl - entry
    reward = entry - tp1
    rrr = round(reward / risk, 2) if risk > 0 else 0.0
  else:
    entry, sl, tp1, tp2, tp3, rrr = 0, 0, 0, 0, 0, 0.0

  return entry, sl, tp1, tp2, tp3, rrr


def evaluate_combo(tf1_code, tf2_code, tf3_code):
  current_tf_data = get_tf_dict(tf1_code)
  d1 = current_tf_data[tf1_code]
  d2 = current_tf_data[tf2_code]
  d3 = current_tf_data[tf3_code]

  valid_signal = "-"
  syarat_tf2 = "-"
  syarat_tf3 = "-"
  kesimpulan = "-"
  dominant_bias = d1["bias"]

  if "BUY" in dominant_bias:
    syarat_tf2 = (
        "Validasi TF2: Wajib Momentum/Extreme Buy, MHV, atau Re-Entry Buy valid"
    )
    syarat_tf3 = "Validasi TF3: Wajib CSAK Buy / Momentum / Re-Entry Eksekusi"

    if "BUY" in d2["bias"] and "BUY" in d3["bias"]:
      valid_signal = "VALID SETUP BBMA (BUY)"
      kesimpulan = (
          f"✅ Alignment Sempurna! {tf_labels[tf1_code]} memberi acuan Re-Entry"
          f" Buy, divalidasi oleh {tf_labels[tf2_code]} & {tf_labels[tf3_code]}."
      )
    elif "BUY" in d2["bias"]:
      valid_signal = "MENUNGGU KONFIRMASI TF3"
      kesimpulan = (
          f"⏳ {tf_labels[tf1_code]} & {tf_labels[tf2_code]} searah Buy. Tunggu"
          f" trigger konfirmasi di {tf_labels[tf3_code]}."
      )
    else:
      valid_signal = "-"
      kesimpulan = (
          f"❌ Belum Valid. {tf_labels[tf2_code]} belum mengonfirmasi arah"
          " Buy."
      )

  elif "SELL" in dominant_bias:
    syarat_tf2 = (
        "Validasi TF2: Wajib Momentum/Extreme Sell, MHV, atau Re-Entry Sell valid"
    )
    syarat_tf3 = "Validasi TF3: Wajib CSAK Sell / Momentum / Re-Entry Eksekusi"

    if "SELL" in d2["bias"] and "SELL" in d3["bias"]:
      valid_signal = "VALID SETUP BBMA (SELL)"
      kesimpulan = (
          f"✅ Alignment Sempurna! {tf_labels[tf1_code]} memberi acuan Re-Entry"
          f" Sell, divalidasi oleh {tf_labels[tf2_code]} & {tf_labels[tf3_code]}."
      )
    elif "SELL" in d2["bias"]:
      valid_signal = "MENUNGGU KONFIRMASI TF3"
      kesimpulan = (
          f"⏳ {tf_labels[tf1_code]} & {tf_labels[tf2_code]} searah Sell. Tunggu"
          f" trigger konfirmasi di {tf_labels[tf3_code]}."
      )
    else:
      valid_signal = "-"
      kesimpulan = (
          f"❌ Belum Valid. {tf_labels[tf2_code]} belum mengonfirmasi arah"
          " Sell."
      )

  return {
      "valid_signal": valid_signal,
      "dominant_bias": dominant_bias,
      "tf1_act": d1["status"],
      "tf1_detail": d1["detail"],
      "tf2_act": d2["status"],
      "tf2_req": syarat_tf2,
      "tf3_act": d3["status"],
      "tf3_req": syarat_tf3,
      "kesimpulan": kesimpulan,
      "tf_data_dict": current_tf_data,
  }


live_price = df_1h["Close"].iloc[-1] if not df_1h.empty else 0.0
st.metric("Live XAUUSD Price", f"${live_price:,.2f}")

st.markdown("---")
st.subheader(
    "🎯 Deteksi Validitas Re-Entry, Validasi TF2/TF3 & Manajemen Risiko"
)

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

  entry, sl, tp1, tp2, tp3, rrr = calculate_trade_levels(
      execution_bias,
      d_exec["price"],
      d_exec["top_bb"],
      d_exec["mid_bb"],
      d_exec["bot_bb"],
  )

  col1, col2, col3 = st.columns(3)
  col1.metric("Status Signal", eval_res["valid_signal"])
  col2.metric(
      "Risk-to-Reward Ratio (RRR)", f"1 : {rrr}" if rrr > 0 else "-"
  )
  col3.metric("Arah Bias Acuan (TF1)", execution_bias)

  st.info(eval_res["kesimpulan"])

  st.markdown("#### 📌 Matriks Status Timeframe Hirarki Ini")
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
        "Mid BB": f"${res['mid_bb']:,.2f}" if res["mid_bb"] > 0 else "-",
        "Bottom BB": f"${res['bot_bb']:,.2f}" if res["bot_bb"] > 0 else "-",
        "Keterangan": res["detail"],
    })
  st.dataframe(pd.DataFrame(matrix_rows), use_container_width=True)

  icon_label = "🟢" if execution_bias == "BUY" else "🔴"
  st.markdown(
      f"#### 📋 Kalkulator Manajemen Risiko & Target Harga ({icon_label}"
      f" {execution_bias})"
  )
  trade_table = pd.DataFrame([
      {
          "Parameter": "Zona Harga Entry (TF3)",
          "Nilai": f"${entry:,.2f}" if entry > 0 else "-",
      },
      {
          "Parameter": "Stop Loss (SL)",
          "Nilai": f"${sl:,.2f}" if sl > 0 else "-",
      },
      {
          "Parameter": "Take Profit 1 (TP1 - Mid BB)",
          "Nilai": f"${tp1:,.2f}" if tp1 > 0 else "-",
      },
      {
          "Parameter": "Take Profit 2 (TP2)",
          "Nilai": f"${tp2:,.2f}" if tp2 > 0 else "-",
      },
      {
          "Parameter": "Take Profit 3 (TP3 - Extreme BB)",
          "Nilai": f"${tp3:,.2f}" if tp3 > 0 else "-",
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
