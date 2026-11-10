import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(
    page_title="XAUUSD BBMA Oma Ally Multi-TF Matrix", layout="wide"
)

st.title("⚡ XAUUSD BBMA Oma Ally - Multi-Timeframe Matrix Screener")
st.markdown(
    "Analisis hirarki multi-timeframe (Setup TF1 ➔ Confirm TF2 ➔ Entry TF3)"
    " untuk deteksi Re-Entry Buy & Sell."
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

  # 1. Extreme / Re-Entry Sell Zone (Atas)
  if high >= top_bb or prev["High"] >= prev["TopBB"]:
    status = "EXTREME / MHV SELL"
    bias = "SELL"
    detail = "Harga menguji Top BB (Potensi Extreme / Re-Entry Sell)"
  elif c_price >= ma5_h and c_price > mid_bb:
    status = "RE-ENTRY SELL / RETRACEMENT HIGH"
    bias = "SELL"
    detail = "Harga berada di area MA5/10 High di atas Mid BB"

  # 2. Extreme / Re-Entry Buy Zone (Bawah)
  elif low <= bot_bb or prev["Low"] <= prev["BottomBB"]:
    status = "EXTREME / MHV BUY"
    bias = "BUY"
    detail = "Harga menguji Bottom BB (Potensi Extreme / Re-Entry Buy)"
  elif c_price <= ma5_l and c_price < mid_bb:
    status = "RE-ENTRY BUY / RETRACEMENT LOW"
    bias = "BUY"
    detail = "Harga berada di area MA5/10 Low di bawah Mid BB"

  # 3. Mid BB / Sideway Transitions
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


# Mapping interval yfinance & period download
tf_config = {
    "1d": {"period": "60d", "label": "Daily (D1)"},
    "1h": {"period": "7d", "label": "H1"},
    "15m": {"period": "5d", "label": "M15"},
    "5m": {"period": "1d", "label": "M5"},
    "1m": {"period": "1d", "label": "M1"},
}

# Fetch Data per TF
tf_data = {}
with st.spinner("Mengambil feed data multi-timeframe XAUUSD..."):
  for tf_key, cfg in tf_config.items():
    raw_df = fetch_tf_data(tf_key, cfg["period"])
    tf_data[tf_key] = analyze_bbma_tf(raw_df)

# Karena yfinance tidak menyediakan interval native 4h, H4 diestimasi dari H1 atau ditandai
# Di sini kita menyajikan 4 Kombo Utama yang didukung yfinance data
combos = [
    {"name": "D1 - H1 - M15 (Swing / Long-term)", "tf1": "1d", "tf3": "15m"},
    {"name": "H1 - M15 - M5 (Intraday Scalp)", "tf1": "1h", "tf3": "5m"},
    {"name": "M15 - M5 - M1 (Micro Scalp)", "tf1": "15m", "tf3": "1m"},
]


def evaluate_combo(tf1_code, tf2_code, tf3_code):
  d1 = tf_data[tf1_code]
  d2 = tf_data[tf2_code]
  d3 = tf_data[tf3_code]

  valid_signal = "-"
  syarat_tf2 = "-"
  syarat_tf3 = "-"
  kesimpulan = "-"

  # Logika Re-Entry BUY (Top-Down Validation)
  if "BUY" in d1["bias"]:
    syarat_tf2 = "Wajib MHV / Extreme Buy (Penolakan Bottom BB)"
    syarat_tf3 = "Wajib CSAK Buy / Breakout MA5/10 High untuk Entry"

    if "BUY" in d2["bias"] and "BUY" in d3["bias"]:
      valid_signal = "VALID RE-ENTRY BUY"
      kesimpulan = (
          "✅ Alignment Sempurna! TF1 Setup Re-Entry Buy, TF2 Confirm CS"
          " Penolakan, TF3 Siap Eksekusi Buy."
      )
    elif "BUY" in d2["bias"]:
      valid_signal = "MENUNGGU CONFIRMATION TF3"
      kesimpulan = (
          "⏳ TF1 & TF2 Sudah Valid Buy. Tunggu Trigger CSAK/Momentum Buy di"
          " TF3."
      )
    else:
      valid_signal = "-"
      kesimpulan = (
          "❌ Belum Valid. TF2 masih mengalami koreksi turun/belum ada"
          " penolakan."
      )

  # Logika Re-Entry SELL (Top-Down Validation)
  elif "SELL" in d1["bias"]:
    syarat_tf2 = "Wajib MHV / Extreme Sell (Penolakan Top BB)"
    syarat_tf3 = "Wajib CSAK Sell / Breakout MA5/10 Low untuk Entry"

    if "SELL" in d2["bias"] and "SELL" in d3["bias"]:
      valid_signal = "VALID RE-ENTRY SELL"
      kesimpulan = (
          "✅ Alignment Sempurna! TF1 Setup Re-Entry Sell, TF2 Confirm CS"
          " Penolakan, TF3 Siap Eksekusi Sell."
      )
    elif "SELL" in d2["bias"]:
      valid_signal = "MENUNGGU CONFIRMATION TF3"
      kesimpulan = (
          "⏳ TF1 & TF2 Sudah Valid Sell. Tunggu Trigger CSAK/Momentum Sell di"
          " TF3."
      )
    else:
      valid_signal = "-"
      kesimpulan = (
          "❌ Belum Valid. TF2 masih mengalami retracement naik/belum ada"
          " penolakan."
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


# Live Price Overview
live_price = tf_data["1h"]["price"]
st.metric("Live XAUUSD Price", f"${live_price:,.2f}")

st.subheader("📌 Matriks Status Multi-Timeframe")

matrix_rows = []
for code, cfg in tf_config.items():
  res = tf_data[code]
  matrix_rows.append({
      "Timeframe": cfg["label"],
      "Status Saat Ini": res["status"],
      "Arah Bias": res["bias"],
      "Top BB": f"${res['top_bb']:,.2f}" if res["top_bb"] > 0 else "-",
      "Mid BB (MA20)": f"${res['mid_bb']:,.2f}" if res["mid_bb"] > 0 else "-",
      "Bottom BB": f"${res['bot_bb']:,.2f}" if res["bot_bb"] > 0 else "-",
      "Keterangan Aktivitas": res["detail"],
  })

st.dataframe(pd.DataFrame(matrix_rows), use_container_width=True)

st.markdown("---")
st.subheader("🎯 Deteksi Validitas Re-Entry Kombinasi TF")

# Tab Hirarki TF
tab1, tab2, tab3 = st.tabs(
    ["D1 ➔ H1 ➔ M15", "H1 ➔ M15 ➔ M5", "M15 ➔ M5 ➔ M1"]
)

with tab1:
  eval1 = evaluate_combo("1d", "1h", "15m")
  col1, col2 = st.columns([1, 2])
  col1.metric("Status Signal", eval1["valid_signal"])
  col2.info(eval1["kesimpulan"])

  st.table(
      pd.DataFrame([{
          "Peran": "TF 1 (Setup Utama - D1)",
          "Kondisi Saat Ini": eval1["tf1_act"],
          "Detail Aktivitas": eval1["tf1_detail"],
      }, {
          "Peran": "TF 2 (Konfirmasi - H1)",
          "Kondisi Saat Ini": eval1["tf2_act"],
          "Syarat Ideal Re-Entry": eval1["tf2_req"],
      }, {
          "Peran": "TF 3 (Eksekusi - M15)",
          "Kondisi Saat Ini": eval1["tf3_act"],
          "Syarat Ideal Re-Entry": eval1["tf3_req"],
      }])
  )

with tab2:
  eval2 = evaluate_combo("1h", "15m", "5m")
  col1, col2 = st.columns([1, 2])
  col1.metric("Status Signal", eval2["valid_signal"])
  col2.info(eval2["kesimpulan"])

  st.table(
      pd.DataFrame([{
          "Peran": "TF 1 (Setup Utama - H1)",
          "Kondisi Saat Ini": eval2["tf1_act"],
          "Detail Aktivitas": eval2["tf1_detail"],
      }, {
          "Peran": "TF 2 (Konfirmasi - M15)",
          "Kondisi Saat Ini": eval2["tf2_act"],
          "Syarat Ideal Re-Entry": eval2["tf2_req"],
      }, {
          "Peran": "TF 3 (Eksekusi - M5)",
          "Kondisi Saat Ini": eval2["tf3_act"],
          "Syarat Ideal Re-Entry": eval2["tf3_req"],
      }])
  )

with tab3:
  eval3 = evaluate_combo("15m", "5m", "1m")
  col1, col2 = st.columns([1, 2])
  col1.metric("Status Signal", eval3["valid_signal"])
  col2.info(eval3["kesimpulan"])

  st.table(
      pd.DataFrame([{
          "Peran": "TF 1 (Setup Utama - M15)",
          "Kondisi Saat Ini": eval3["tf1_act"],
          "Detail Aktivitas": eval3["tf1_detail"],
      }, {
          "Peran": "TF 2 (Konfirmasi - M5)",
          "Kondisi Saat Ini": eval3["tf2_act"],
          "Syarat Ideal Re-Entry": eval3["tf2_req"],
      }, {
          "Peran": "TF 3 (Eksekusi - M1)",
          "Kondisi Saat Ini": eval3["tf3_act"],
          "Syarat Ideal Re-Entry": eval3["tf3_req"],
      }])
  )