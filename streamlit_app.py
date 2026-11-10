import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(
    page_title="XAUUSD BBMA Oma Ally Multi-TF Matrix", layout="wide"
)

st.title("⚡ XAUUSD BBMA Oma Ally - Multi-Timeframe Screener & TP/SL Calculator")
st.markdown(
    "Analisis hirarki multi-timeframe lengkap dengan proyeksi Target Profit"
    " (TP1, TP2, TP3), Stop Loss (SL), dan Risk-to-Reward Ratio."
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


def analyze_bbma_tf(df):
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

  if high >= top_bb or prev["High"] >= prev["TopBB"]:
    status = "EXTREME / MHV SELL"
    bias = "SELL"
    detail = "Harga menguji Top BB (Potensi Extreme / Re-Entry Sell)"
  elif c_price >= ma5_h and c_price > mid_bb:
    status = "RE-ENTRY SELL / RETRACEMENT HIGH"
    bias = "SELL"
    detail = "Harga berada di area MA5/10 High di atas Mid BB"
  elif low <= bot_bb or prev["Low"] <= prev["BottomBB"]:
    status = "EXTREME / MHV BUY"
    bias = "BUY"
    detail = "Harga menguji Bottom BB (Potensi Extreme / Re-Entry Buy)"
  elif c_price <= ma5_l and c_price < mid_bb:
    status = "RE-ENTRY BUY / RETRACEMENT LOW"
    bias = "BUY"
    detail = "Harga berada di area MA5/10 Low di bawah Mid BB"
  elif c_price > mid_bb:
    status = "BULLISH BIAS (Di atas Mid BB)"
    bias = "BUY"
    detail = "Harga bergerak di atas Mid BB menuju Top BB"
  elif c_price < mid_bb:
    status = "BEARISH BIAS (Di bawah Mid BB)"
    bias = "SELL"
    detail = "Harga bergerak di bawah Mid BB menuju Bottom BB"

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

tf_data = {
    "1d": analyze_bbma_tf(df_1d),
    "4h": analyze_bbma_tf(df_4h),
    "1h": analyze_bbma_tf(df_1h),
    "15m": analyze_bbma_tf(df_15m),
    "5m": analyze_bbma_tf(df_5m),
    "1m": analyze_bbma_tf(df_1m),
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
    sl = round(bot_bb - 3.0, 2)  # Di bawah Bottom BB
    tp1 = round(mid_bb, 2)  # Target 1 di Mid BB
    tp2 = round(top_bb - 4.0, 2)  # Target 2 mendekati Top BB
    tp3 = round(top_bb, 2)  # Target 3 di Top BB

    risk = entry - sl
    reward = tp1 - entry
    rrr = round(reward / risk, 2) if risk > 0 else 0.0

  elif bias == "SELL":
    entry = current_price
    sl = round(top_bb + 3.0, 2)  # Di atas Top BB
    tp1 = round(mid_bb, 2)  # Target 1 di Mid BB
    tp2 = round(bot_bb + 4.0, 2)  # Target 2 mendekati Bottom BB
    tp3 = round(bot_bb, 2)  # Target 3 di Bottom BB

    risk = sl - entry
    reward = entry - tp1
    rrr = round(reward / risk, 2) if risk > 0 else 0.0
  else:
    entry, sl, tp1, tp2, tp3, rrr = 0, 0, 0, 0, 0, 0.0

  return entry, sl, tp1, tp2, tp3, rrr


def evaluate_combo(tf1_code, tf2_code, tf3_code):
  d1 = tf_data[tf1_code]
  d2 = tf_data[tf2_code]
  d3 = tf_data[tf3_code]

  valid_signal = "-"
  syarat_tf2 = "-"
  syarat_tf3 = "-"
  kesimpulan = "-"

  if "BUY" in d1["bias"]:
    syarat_tf2 = "Wajib MHV / Extreme Buy (Penolakan Bottom BB)"
    syarat_tf3 = "Wajib CSAK Buy / Breakout MA5/10 High untuk Entry"

    if "BUY" in d2["bias"] and "BUY" in d3["bias"]:
      valid_signal = "VALID RE-ENTRY BUY"
      kesimpulan = (
          f"✅ Alignment Sempurna! {tf_labels[tf1_code]} Setup Buy,"
          f" {tf_labels[tf2_code]} Confirm Penolakan, {tf_labels[tf3_code]} Siap"
          " Eksekusi."
      )
    elif "BUY" in d2["bias"]:
      valid_signal = "MENUNGGU CONFIRMATION TF3"
      kesimpulan = (
          f"⏳ {tf_labels[tf1_code]} & {tf_labels[tf2_code]} Valid Buy. Tunggu"
          f" Trigger di {tf_labels[tf3_code]}."
      )
    else:
      valid_signal = "-"
      kesimpulan = (
          f"❌ Belum Valid. {tf_labels[tf2_code]} masih koreksi turun / belum"
          " ada penolakan."
      )

  elif "SELL" in d1["bias"]:
    syarat_tf2 = "Wajib MHV / Extreme Sell (Penolakan Top BB)"
    syarat_tf3 = "Wajib CSAK Sell / Breakout MA5/10 Low untuk Entry"

    if "SELL" in d2["bias"] and "SELL" in d3["bias"]:
      valid_signal = "VALID RE-ENTRY SELL"
      kesimpulan = (
          f"✅ Alignment Sempurna! {tf_labels[tf1_code]} Setup Sell,"
          f" {tf_labels[tf2_code]} Confirm Penolakan, {tf_labels[tf3_code]} Siap"
          " Eksekusi."
      )
    elif "SELL" in d2["bias"]:
      valid_signal = "MENUNGGU CONFIRMATION TF3"
      kesimpulan = (
          f"⏳ {tf_labels[tf1_code]} & {tf_labels[tf2_code]} Valid Sell. Tunggu"
          f" Trigger di {tf_labels[tf3_code]}."
      )
    else:
      valid_signal = "-"
      kesimpulan = (
          f"❌ Belum Valid. {tf_labels[tf2_code]} masih retracement naik / belum"
          " ada penolakan."
      )

  return {
      "valid_signal": valid_signal,
      "tf1_act": d1["status"],
      "tf1_detail": d1["detail"],
      "tf2_act": d2["status"],
      "tf2_req": syarat_tf2,
      "tf3_act": d3["status"],
      "tf3_req": syarat_tf3,
      "kesimpulan": kesimpulan,
  }


live_price = tf_data["1h"]["price"]
st.metric("Live XAUUSD Price", f"${live_price:,.2f}")

st.subheader("📌 Matriks Status Multi-Timeframe")
matrix_rows = []
for code, label in tf_labels.items():
  res = tf_data[code]
  matrix_rows.append({
      "Timeframe": label,
      "Status Saat Ini": res["status"],
      "Arah Bias": res["bias"],
      "Top BB": f"${res['top_bb']:,.2f}" if res["top_bb"] > 0 else "-",
      "Mid BB (MA20)": f"${res['mid_bb']:,.2f}" if res["mid_bb"] > 0 else "-",
      "Bottom BB": f"${res['bot_bb']:,.2f}" if res["bot_bb"] > 0 else "-",
      "Keterangan Aktivitas": res["detail"],
  })

st.dataframe(pd.DataFrame(matrix_rows), use_container_width=True)

st.markdown("---")
st.subheader("🎯 Deteksi Validitas Re-Entry & Manajemen Risiko (SL, TP, RRR)")

# Tab Hirarki
tab1, tab2, tab3, tab4 = st.tabs([
    "D1 ➔ H4 ➔ H1",
    "H4 ➔ H1 ➔ M15",
    "H1 ➔ M15 ➔ M5",
    "M15 ➔ M5 ➔ M1",
])


def render_tab_content(tf1, tf2, tf3):
  eval_res = evaluate_combo(tf1, tf2, tf3)
  d_exec = tf_data[tf3]  # Acuan TP/SL menggunakan TF Eksekusi (TF3)

  entry, sl, tp1, tp2, tp3, rrr = calculate_trade_levels(
      d_exec["bias"],
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
  col3.metric("Arah Bias Eksekusi", d_exec["bias"])

  st.info(eval_res["kesimpulan"])

  # Tabel Kalkulator TP & SL
  st.markdown("#### 📋 Kalkulator Manajemen Risiko & Target Harga")
  trade_table = pd.DataFrame([
      {
          "Parameter": "Zona Harga Entry",
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

  st.markdown("#### 🔍 Detail Hirarki Timeframe")
  st.table(
      pd.DataFrame([{
          "Peran": f"TF 1 (Setup Utama - {tf_labels[tf1]})",
          "Kondisi Saat Ini": eval_res["tf1_act"],
          "Detail Aktivitas": eval_res["tf1_detail"],
          "Syarat Ideal Re-Entry": "-",
      }, {
          "Peran": f"TF 2 (Konfirmasi - {tf_labels[tf2]})",
          "Kondisi Saat Ini": eval_res["tf2_act"],
          "Detail Aktivitas": "-",
          "Syarat Ideal Re-Entry": eval_res["tf2_req"],
      }, {
          "Peran": f"TF 3 (Eksekusi - {tf_labels[tf3]})",
          "Kondisi Saat Ini": eval_res["tf3_act"],
          "Detail Aktivitas": "-",
          "Syarat Ideal Re-Entry": eval_res["tf3_req"],
      }])
  )


with tab1:
  render_tab_content("1d", "4h", "1h")

with tab2:
  render_tab_content("4h", "1h", "15m")

with tab3:
  render_tab_content("1h", "15m", "5m")

with tab4:
  render_tab_content("15m", "5m", "1m")
