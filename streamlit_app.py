from datetime import datetime
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

# Konfigurasi Halaman
st.set_page_config(
    page_title="XAUUSD BBMA Pro Screener", page_icon="⚡", layout="wide"
)

st.title(
    "⚡ XAUUSD BBMA Oma Ally Pro - Multi-Timeframe Re-Entry Screener & Risk"
    " Calculator"
)
st.markdown(
    f"📅 **Waktu Akses:** {datetime.now().strftime('%A, %d %B %Y - %H:%M')} WIB | "
    "Sistem otomatis mendeteksi arah setup di **seluruh Timeframe** secara"
    " kronologis murni metodologi **BBMA Oma Ally**."
)


# -----------------------------------------------------------------
# 1. SIDEBAR PENGATURAN SISTEM
# -----------------------------------------------------------------
st.sidebar.header("⚙️ Konfigurasi Sistem Pro")

tf_package = st.sidebar.selectbox(
    "Pilih Paket Hierarki Timeframe",
    ["D1 - H4 - H1", "H4 - H1 - M15", "M15 - M5 - M1"],
    index=0,
)

if tf_package == "D1 - H4 - H1":
  tf_high, tf_mid, tf_exec = "1d", "4h", "1h"
elif tf_package == "H4 - H1 - M15":
  tf_high, tf_mid, tf_exec = "4h", "1h", "15m"
else:
  tf_high, tf_mid, tf_exec = "15m", "5m", "1m"

# Sidebar Risk Management
st.sidebar.markdown("---")
st.sidebar.header("🛡️ Money Management Settings")
balance = st.sidebar.number_input(
    "Modal Akun ($)", min_value=10.0, value=1000.0, step=100.0
)
risk_pct = st.sidebar.slider(
    "Risiko per Trade (%)", min_value=0.5, max_value=5.0, value=1.0, step=0.5
)
risk_amount = balance * (risk_pct / 100.0)
st.sidebar.info(f"💵 Batas Toleransi Risiko: **${risk_amount:,.2f}**")


# -----------------------------------------------------------------
# 2. FUNGSI TEKNIKAL & DATA FEED XAUUSD
# -----------------------------------------------------------------
@st.cache_data(ttl=60)
def fetch_tf_data(interval, period="1y"):
  tickers = ["GC=F", "XAUUSD=X", "GLD"]
  for ticker in tickers:
    try:
      df = yf.download(
          ticker, period=period, interval=interval, progress=False
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
          return calculate_bbma(df)
    except Exception:
      continue
  return pd.DataFrame()


def calculate_bbma(df):
  if df.empty or len(df) < 20:
    return df
  df = df.copy()
  df["MA5_High"] = df["High"].rolling(window=5).mean()
  df["MA5_Low"] = df["Low"].rolling(window=5).mean()
  df["MA10_High"] = df["High"].rolling(window=10).mean()
  df["MA10_Low"] = df["Low"].rolling(window=10).mean()
  df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()

  # Mid BB menggunakan EMA 20 (Standar BBMA Oma Ally)
  df["MA20"] = df["Close"].ewm(span=20, adjust=False).mean()
  df["BB_Middle"] = df["MA20"]

  std = df["Close"].rolling(window=20).std()
  df["BB_Upper"] = df["MA20"] + (std * 2)
  df["TopBB"] = df["BB_Upper"]

  df["BB_Lower"] = df["MA20"] - (std * 2)
  df["BottomBB"] = df["BB_Lower"]
  return df


def resample_to_h4(df_h1):
  if df_h1.empty:
    return pd.DataFrame()
  df_h4 = df_h1.resample("4h").agg({
      "Open": "first",
      "High": "max",
      "Low": "min",
      "Close": "last",
  })
  return calculate_bbma(df_h4.dropna())


def analyze_bbma_tf_auto(df, role_type="TF1"):
  """Analisis murni metodologi BBMA Oma Ally dengan filter Netral ketat"""
  if df.empty or len(df) < 55:
    return {
        "price": 0.0,
        "status": "DATA KURANG",
        "bias": "NEUTRAL",
        "top_bb": 0.0,
        "bot_bb": 0.0,
        "mid_bb": 0.0,
        "ema50": 0.0,
        "detail": "Data baris kurang dari minimum.",
    }

  latest = df.iloc[-1]
  c_price = float(latest["Close"])
  top_bb = float(latest["TopBB"])
  bot_bb = float(latest["BottomBB"])
  mid_bb = float(latest["MA20"])
  ema50 = float(latest["EMA50"])
  ma5_h = float(latest["MA5_High"])
  ma5_l = float(latest["MA5_Low"])
  ma10_h = float(latest["MA10_High"])
  ma10_l = float(latest["MA10_Low"])
  high = float(latest["High"])
  low = float(latest["Low"])

  recent_df = df.tail(8)
  avg_body = abs(df["Close"] - df["Open"]).rolling(10).mean().iloc[-1]

  # 1. Deteksi Momentum Murni (Close menembus Top/Bottom BB)[cite: 3]
  momentum_buy = any(recent_df["Close"] <= recent_df["BottomBB"])
  momentum_sell = any(recent_df["Close"] >= recent_df["TopBB"])

  # 2. Deteksi CSAK Murni (Close melewati Mid BB & MA 5/10)[cite: 3]
  csak_buy = False
  csak_sell = False
  for i in range(len(df) - 6, len(df)):
    if i > 0:
      curr_c = df.iloc[i]
      prev_c = df.iloc[i - 1]
      b_sz = abs(float(curr_c["Close"]) - float(curr_c["Open"]))
      if (
          float(curr_c["Close"]) > float(curr_c["MA20"])
          and float(prev_c["Close"]) <= float(prev_c["MA20"])
          and b_sz >= (avg_body * 0.70)
      ):
        csak_buy = True
      if (
          float(curr_c["Close"]) < float(curr_c["MA20"])
          and float(prev_c["Close"]) >= float(prev_c["MA20"])
          and b_sz >= (avg_body * 0.70)
      ):
        csak_sell = True

  # 3. Deteksi Re-Entry Spesifik (Pullback di area MA 5/10)
  is_re_buy = (low <= ma10_l * 1.025) and (c_price >= bot_bb)
  is_re_sell = (high >= ma10_h * 0.975) and (c_price <= top_bb)

  # 4. Penentuan Struktur & Netralisasi (100% menggunakan csak_buy dan csak_sell)
  structural_bias = "NEUTRAL"
  if momentum_buy or csak_buy or (c_price > mid_bb and is_re_buy):
    structural_bias = "BUY"
  elif momentum_sell or csak_sell or (c_price < mid_bb and is_re_sell):
    structural_bias = "SELL"
  else:
    structural_bias = "NEUTRAL"

  status = "NEUTRAL (Menunggu Setup)"
  bias = structural_bias
  detail = "Harga berada di area netral atau belum memenuhi syarat Re-Entry."

  if role_type == "TF1":
    if structural_bias == "BUY":
      if is_re_buy or (c_price <= ma5_h):
        status = "RE-ENTRY BUY VALID (MA 5/10 Low)"
        detail = "Didahului struktur Buy, harga pullback akurat di area MA 5/10 Low."
      else:
        status = "BULLISH BIAS (Menunggu Re-Entry Buy)"
        detail = "Struktur Buy aktif, menunggu koreksi sempurna ke area MA 5/10 Low."
    elif structural_bias == "SELL":
      if is_re_sell or (c_price >= ma5_l):
        status = "RE-ENTRY SELL VALID (MA 5/10 High)"
        detail = "Didahului struktur Sell, harga koreksi naik akurat di area MA 5/10 High."
      else:
        status = "BEARISH BIAS (Menunggu Re-Entry Sell)"
        detail = "Struktur Sell aktif, menunggu pantulan ke area MA 5/10 High."
    else:
      status = "MARKET NETRAL / KONSOLIDASI"
      detail = "Belum ada CSAK, Momentum, atau Re-Entry yang valid pada acuan TF1."

  elif role_type == "TF2":
    if structural_bias == "BUY":
      status = "TF2 KONFIRMASI BUY"
      detail = "TF2 mendukung struktur kenaikan."
    elif structural_bias == "SELL":
      status = "TF2 KONFIRMASI SELL"
      detail = "TF2 mendukung struktur penurunan."
    else:
      status = "TF2 NETRAL"
      detail = "Struktur TF2 dalam kondisi konsolidasi."

  elif role_type == "TF3":
    if structural_bias == "BUY":
      if momentum_buy:
        status = "TRIGGER MOMENTUM BUY"
        detail = "TF3 membentuk Momentum Buy di Bottom BB."[cite: 3]
      elif csak_buy:
        status = "TRIGGER CSAK BUY"
        detail = "TF3 memicu CSAK Buy menembus Mid BB."[cite: 3]
      else:
        status = "TRIGGER RE-ENTRY BUY"
        detail = "TF3 siap eksekusi Re-Entry Buy."
    elif structural_bias == "SELL":
      if momentum_sell:
        status = "TRIGGER MOMENTUM SELL"
        detail = "TF3 membentuk Momentum Sell di Top BB."[cite: 3]
      elif csak_sell:
        status = "TRIGGER CSAK SELL"
        detail = "TF3 memicu CSAK Sell menembus Mid BB."[cite: 3]
      else:
        status = "TRIGGER RE-ENTRY SELL"
        detail = "TF3 siap eksekusi Re-Entry Sell."
    else:
      status = "TF3 GOLL/WAITING"
      detail = "Tidak ada trigger eksekusi aktif."

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


# -----------------------------------------------------------------
# 3. KALKULATOR RISK MANAGEMENT XAUUSD
# -----------------------------------------------------------------
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


# -----------------------------------------------------------------
# 4. TARIK DATA & EVALUASI OTOMATIS XAUUSD
# -----------------------------------------------------------------
with st.spinner("Menganalisis struktur Multi-Timeframe live XAUUSD..."):
  df_1h = fetch_tf_data("1h", "30d")
  df_15m = fetch_tf_data("15m", "10d")
  df_5m = fetch_tf_data("5m", "5d")
  df_1m = fetch_tf_data("1m", "2d")
  df_1d = fetch_tf_data("1d", "1y")
  df_4h = resample_to_h4(df_1h)


def evaluate_xauusd_automatic(tf1_code, tf2_code, tf3_code):
  d1 = analyze_bbma_tf_auto(globals()[f"df_{tf1_code}"], "TF1")
  d2 = analyze_bbma_tf_auto(globals()[f"df_{tf2_code}"], "TF2")
  d3 = analyze_bbma_tf_auto(globals()[f"df_{tf3_code}"], "TF3")

  dominant_bias = d1["bias"]

  if dominant_bias == "NEUTRAL":
    valid_signal = "⏳ PASAR NETRAL / KONSOLIDASI"
    kesimpulan = (
        f"ℹ️ Acuan {tf1_code.upper()} tidak menunjukkan tren dominan atau re-entry"
        " yang valid. Disarankan untuk **Wait and See**."
    )
  else:
    valid_signal = f"🔥 SETUP AKTIF ({dominant_bias})"
    kesimpulan = (
        f"✅ **Setup Terdeteksi di {tf1_code.upper()}!** Timeframe utama"
        f" menunjukkan **{d1['status']}**."
    )

  return {
      "valid_signal": valid_signal,
      "dominant_bias": dominant_bias,
      "kesimpulan": kesimpulan,
      "tf_data": {"tf1": d1, "tf2": d2, "tf3": d3},
  }


tf_labels = {
    "1d": "Daily (D1)",
    "4h": "H4",
    "1h": "H1",
    "15m": "M15",
    "5m": "M5",
    "1m": "M1",
}

# -----------------------------------------------------------------
# 5. TAMPILAN ANTARMUKA UTAMA (UI DASHBOARD)
# -----------------------------------------------------------------
live_price = df_1h["Close"].iloc[-1] if not df_1h.empty else 0.0
st.metric("Harga Live XAUUSD", f"${live_price:,.2f}")
st.markdown("---")
st.subheader(
    f"🎯 Matriks Hirarki Otomatis BBMA XAUUSD | Paket: {tf_package}"
)

tab1, tab2, tab3, tab4 = st.tabs([
    "D1 ➔ H4 ➔ H1",
    "H4 ➔ H1 ➔ M15",
    "H1 ➔ M15 ➔ M5",
    "M15 ➔ M5 ➔ M1",
])


def render_tab_content(tf1, tf2, tf3):
  eval_res = evaluate_xauusd_automatic(tf1, tf2, tf3)
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
  col1.metric("Status Keputusan AI", eval_res["valid_signal"])
  col2.metric("Risk-to-Reward (RRR)", f"1 : {rrr}" if rrr > 0 else "-")
  col3.metric("Rekomendasi Lot", f"{lot_size} Lot" if lot_size > 0 else "-")
  col4.metric("Arah Keputusan", execution_bias)

  if "SETUP AKTIF" in eval_res["valid_signal"]:
    st.success(eval_res["kesimpulan"])
  else:
    st.info(eval_res["kesimpulan"])

  # Tabel Matriks Timeframe
  st.markdown("#### 📌 Matriks Deteksi Otomatis (TF1 ➔ TF2 ➔ TF3)")
  matrix_rows = [
      {
          "Peran Hirarki": f"TF1 (Acuan Utama - {tf_labels[tf1]})",
          "Status / Kondisi": data["tf1"]["status"],
          "Arah Bias": data["tf1"]["bias"],
          "Top BB": f"${data['tf1']['top_bb']:,.2f}",
          "Mid BB": f"${data['tf1']['mid_bb']:,.2f}",
          "EMA 50": f"${data['tf1']['ema50']:,.2f}",
          "Bottom BB": f"${data['tf1']['bot_bb']:,.2f}",
          "Keterangan Analisis": data["tf1"]["detail"],
      },
      {
          "Peran Hirarki": f"TF2 (Konfirmasi Fleksibel - {tf_labels[tf2]})",
          "Status / Kondisi": data["tf2"]["status"],
          "Arah Bias": data["tf2"]["bias"],
          "Top BB": f"${data['tf2']['top_bb']:,.2f}",
          "Mid BB": f"${data['tf2']['mid_bb']:,.2f}",
          "EMA 50": f"${data['tf2']['ema50']:,.2f}",
          "Bottom BB": f"${data['tf2']['bot_bb']:,.2f}",
          "Keterangan Analisis": data["tf2"]["detail"],
      },
      {
          "Peran Hirarki": f"TF3 (Eksekusi / Trigger - {tf_labels[tf3]})",
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

  # Tabel Rencana Perdagangan & Manajemen Risiko
  icon_label = "🟢" if execution_bias == "BUY" else ("🔴" if execution_bias == "SELL" else "⚪")
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
          "Parameter": "Jarak Risiko",
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
          "Nilai Level Harga": (
              f"${tp3:,.2f}" if (tp3 is not None and tp3 > 0) else "-"
          ),
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
