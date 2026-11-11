from datetime import datetime
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

# Konfigurasi Halaman
st.set_page_config(
    page_title="BBMA MTF Pro BEI Screener", page_icon="📈", layout="wide"
)

st.title(
    "📈 BBMA Oma Ally Pro - Multi-Timeframe Re-Entry Screener (944+ Saham BEI)"
)
st.markdown(
    f"📅 **Waktu Akses:** {datetime.now().strftime('%A, %d %B %Y - %H:%M')} WIB | "
    "Sistem validasi hierarki ketat: **TF1 (Re-Entry MA 5/10) ➔ TF2 (Konfirmasi)"
    " ➔ TF3 (Eksekusi)**."
)


# -----------------------------------------------------------------
# 1. LOAD OTOMATIS SELURUH SAHAM DARI EXCEL
# -----------------------------------------------------------------
@st.cache_data
def load_all_tickers_from_excel():
  try:
    excel_file = "Daftar Saham  - 20261111.xlsx"
    df_excel = pd.read_excel(excel_file)
    tickers = df_excel["Kode"].dropna().astype(str).str.strip().tolist()
    return sorted(list(set(tickers)))
  except Exception as e:
    st.warning(
        f"File Excel tidak ditemukan ({e}). Menggunakan cadangan emiten utama."
    )
    return ["BBCA", "BBRI", "BMRI", "BBNI", "ASII", "TLKM", "GOTO", "ADRO"]


all_stocks = load_all_tickers_from_excel()

# -----------------------------------------------------------------
# 2. SIDEBAR PENGATURAN MODE & ARAH SETUP
# -----------------------------------------------------------------
st.sidebar.header("⚙️ Konfigurasi Sistem Pro")
mode = st.sidebar.radio(
    "Pilih Mode Operasi",
    [
        "🚀 Auto Screener (Cari Sinyal Re-Entry)",
        "🔍 Cek Manual (Single Stock Pro)",
    ],
)

setup_direction = st.sidebar.selectbox(
    "Pilih Arah Sinyal Re-Entry", ["RE-ENTRY BUY", "RE-ENTRY SELL"]
)

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


# -----------------------------------------------------------------
# 3. FUNGSI TEKNIKAL & ANALISIS BBMA PRO DUA ARAH
# -----------------------------------------------------------------
@st.cache_data(ttl=0)
def load_stock_data(ticker_symbol, interval):
  period = "5d" if interval in ["1m", "5m", "15m"] else "1mo"
  if interval == "4h":
    period = "3mo"
  elif interval == "1d":
    period = "1y"

  try:
    df = yf.download(
        ticker_symbol, period=period, interval=interval, progress=False
    )
    if not df.empty:
      if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
      df = df.dropna(subset=["Open", "High", "Low", "Close"])
      if not df.empty:
        return calculate_bbma(df)
  except Exception:
    pass
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


def analyze_bbma_tf(df, role_type="TF1", target_bias="BUY"):
  if df.empty or len(df) < 55:
    return {
        "price": 0.0,
        "status": "DATA KURANG",
        "bias": "NEUTRAL",
        "detail": "Data tidak mencukupi untuk indikator lengkap",
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
  ma10_h = float(latest["MA10_High"])
  ma10_l = float(latest["MA10_Low"])
  high = float(latest["High"])
  low = float(latest["Low"])

  status = "-"
  bias = "NEUTRAL"
  detail = "-"

  # Komponen Deteksi
  reject_bot_bb = low <= bot_bb and c_price > bot_bb
  reject_top_bb = high >= top_bb and c_price < top_bb
  reject_ema50_buy = low <= ema50 and c_price > ema50
  reject_ema50_sell = high >= ema50 and c_price < ema50

  extreme_buy = low <= bot_bb and c_price > bot_bb
  extreme_sell = high >= top_bb and c_price < top_bb

  momentum_buy = c_price <= bot_bb or (
      low <= bot_bb and latest["Close"] < prev["Close"]
  )
  momentum_sell = c_price >= top_bb or (
      high >= top_bb and latest["Close"] > prev["Close"]
  )

  body_size = abs(c_price - float(latest["Open"]))
  avg_body = (
      abs(df["Close"] - df["Open"]).rolling(10).mean().iloc[-1]
      if len(df) >= 10
      else body_size
  )

  csak_buy = (
      c_price > mid_bb
      and prev["Close"] <= mid_bb
      and body_size >= (avg_body * 1.1)
  )
  csak_sell = (
      c_price < mid_bb
      and prev["Close"] >= mid_bb
      and body_size >= (avg_body * 1.1)
  )

  # ==========================================
  # 1. LOGIKA TF1 (ACUAN UTAMA RE-ENTRY DI MA 5/10)
  # ==========================================
  if role_type == "TF1":
    if target_bias == "BUY":
      # Re-Entry Buy berpatokan di area MA 5/10 Low
      if c_price <= ma5_h and low <= ma10_l * 1.02:
        status = "RE-ENTRY BUY VALID (MA 5/10 Low)"
        bias = "BUY"
        detail = "Harga pullback dan tertahan di area MA 5/10 Low."
      elif c_price > mid_bb:
        status = "BULLISH BIAS (Menunggu Re-Entry)"
        bias = "BUY"
        detail = "Tren Bullish, menunggu koreksi ke MA 5/10 Low."
      else:
        status = "TIDAK VALID"
        bias = "SELL"
        detail = "Belum memenuhi syarat Re-Entry Buy."
    else:
      # Re-Entry Sell berpatokan di area MA 5/10 High
      if c_price >= ma5_l and high >= ma10_h * 0.98:
        status = "RE-ENTRY SELL VALID (MA 5/10 High)"
        bias = "SELL"
        detail = "Harga koreksi naik dan tertahan di area MA 5/10 High."
      elif c_price < mid_bb:
        status = "BEARISH BIAS (Menunggu Re-Entry)"
        bias = "SELL"
        detail = "Tren Bearish, menunggu pantulan ke MA 5/10 High."
      else:
        status = "TIDAK VALID"
        bias = "BUY"
        detail = "Belum memenuhi syarat Re-Entry Sell."

  # ==========================================
  # 2. LOGIKA TF2 (KONFIRMASI: Extreme / Re-Entry Searah / Reject EMA 50 / Reject Top-Low BB)
  # ==========================================
  elif role_type == "TF2":
    if target_bias == "BUY":
      if (
          extreme_buy
          or reject_bot_bb
          or reject_ema50_buy
          or (c_price <= ma5_h and low <= ma10_l * 1.02)
      ):
        status = "TF2 KONFIRMASI BUY (Valid Extreme/Reject/Re-Entry)"
        bias = "BUY"
        detail = (
            "TF2 mengonfirmasi lewat Extreme, Reject BB/EMA 50, atau Re-Entry"
            " searah."
        )
      elif c_price > mid_bb:
        status = "TF2 BULLISH SEARAH"
        bias = "BUY"
        detail = "Struktur TF2 mendukung kenaikan."
      else:
        status = "TF2 BELUM KONFIRMASI"
        bias = "SELL"
        detail = "TF2 belum menunjukkan konfirmasi Buy."
    else:
      if (
          extreme_sell
          or reject_top_bb
          or reject_ema50_sell
          or (c_price >= ma5_l and high >= ma10_h * 0.98)
      ):
        status = "TF2 KONFIRMASI SELL (Valid Extreme/Reject/Re-Entry)"
        bias = "SELL"
        detail = (
            "TF2 mengonfirmasi lewat Extreme, Reject BB/EMA 50, atau Re-Entry"
            " searah."
        )
      elif c_price < mid_bb:
        status = "TF2 BEARISH SEARAH"
        bias = "SELL"
        detail = "Struktur TF2 mendukung penurunan."
      else:
        status = "TF2 BELUM KONFIRMASI"
        bias = "BUY"
        detail = "TF2 belum menunjukkan konfirmasi Sell."

  # ==========================================
  # 3. LOGIKA TF3 (EKSEKUSI / TRIGGER: Full Setup / Re-Entry / CSAK / Momentum)
  # ==========================================
  elif role_type == "TF3":
    if target_bias == "BUY":
      if momentum_buy:
        status = "TRIGGER MOMENTUM BUY"
        bias = "BUY"
        detail = "TF3 membentuk Momentum Buy sebagai pelatuk."
      elif csak_buy:
        status = "TRIGGER CSAK BUY"
        bias = "BUY"
        detail = "TF3 membentuk CSAK Buy."
      elif (c_price <= ma5_l and c_price < mid_bb) or reject_bot_bb:
        status = "TRIGGER RE-ENTRY / EXTREME BUY"
        bias = "BUY"
        detail = "TF3 siap eksekusi Re-Entry / Extreme Buy."
      elif c_price > mid_bb:
        status = "TF3 BULLISH MENGALIR"
        bias = "BUY"
        detail = "Kondisi TF3 mengalir searah Buy."
      else:
        status = "TF3 BELUM ADA TRIGGER"
        bias = "SELL"
        detail = "TF3 belum memunculkan trigger Buy."
    else:
      if momentum_sell:
        status = "TRIGGER MOMENTUM SELL"
        bias = "SELL"
        detail = "TF3 membentuk Momentum Sell sebagai pelatuk."
      elif jika_csak_sell := csak_sell:
        status = "TRIGGER CSAK SELL"
        bias = "SELL"
        detail = "TF3 membentuk CSAK Sell."
      elif (c_price >= ma5_h and c_price > mid_bb) or reject_top_bb:
        status = "TRIGGER RE-ENTRY / EXTREME SELL"
        bias = "SELL"
        detail = "TF3 siap eksekusi Re-Entry / Extreme Sell."
      elif c_price < mid_bb:
        status = "TF3 BEARISH MENGALIR"
        bias = "SELL"
        detail = "Kondisi TF3 mengalir searah Sell."
      else:
        status = "TF3 BELUM ADA TRIGGER"
        bias = "BUY"
        detail = "TF3 belum memunculkan trigger Sell."

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


def evaluate_pro_combo(
    code, tf_high_code, tf_mid_code, tf_exec_code, direction="BUY"
):
  try:
    if tf_high_code == "4h":
      df_h1 = load_stock_data(f"{code}.JK", "1h")
      d_high = resample_to_h4(df_h1)
    else:
      d_high = load_stock_data(f"{code}.JK", tf_high_code)

    d_mid = load_stock_data(f"{code}.JK", tf_mid_code)
    d_exec = load_stock_data(f"{code}.JK", tf_exec_code)

    if d_high.empty or d_mid.empty or d_exec.empty:
      return None

    target_bias = "BUY" if direction == "RE-ENTRY BUY" else "SELL"

    res_tf1 = analyze_bbma_tf(d_high, role_type="TF1", target_bias=target_bias)
    res_tf2 = analyze_bbma_tf(d_mid, role_type="TF2", target_bias=target_bias)
    res_tf3 = analyze_bbma_tf(d_exec, role_type="TF3", target_bias=target_bias)

    is_valid = (
        res_tf1["bias"] == target_bias
        and res_tf2["bias"] == target_bias
        and res_tf3["bias"] == target_bias
        and "VALID" in res_tf1["status"]
    )

    entry_zone = (
        (res_tf3["bot_bb"] + res_tf3["mid_bb"]) / 2
        if target_bias == "BUY"
        else (res_tf3["top_bb"] + res_tf3["mid_bb"]) / 2
    )

    if is_valid:
      status_label = f"🔥 PRO {direction} VALID"
    elif res_tf1["bias"] == target_bias and res_tf2["bias"] == target_bias:
      status_label = "⏳ Menunggu Trigger TF3"
    else:
      status_label = "❌ Belum Valid"

    return {
        "Emiten": code,
        "Harga Terkini": res_tf3["price"],
        "Estimasi Area Entry": f"Rp {entry_zone:,.0f}",
        "Status Sinyal": status_label,
        "Valid": is_valid,
        "TF1": res_tf1,
        "TF2": res_tf2,
        "TF3": res_tf3,
        "DF_Exec": d_exec,
    }
  except Exception:
    return None


# -----------------------------------------------------------------
# MODE 1: AUTO SCREENER
# -----------------------------------------------------------------
if mode == "🚀 Auto Screener (Cari Sinyal Re-Entry)":
  st.subheader(f"🚀 Auto Screener - Filter Ketat Pro ({setup_direction})")
  st.markdown(
      "Sistem memindai emiten BEI otomatis berdasarkan hirarki ketat: **TF1"
      f" (Re-Entry MA 5/10) ➔ TF2 (Konfirmasi) ➔ TF3 (Eksekusi)** menggunakan"
      f" paket **{tf_package}**."
  )

  scan_limit = st.sidebar.slider(
      "Jumlah Maksimal Saham yang Dipindai", 10, len(all_stocks), 60
  )
  run_scan = st.sidebar.button("⚡ MULAI SCANNING PASAR")

  if run_scan:
    stocks_to_scan = all_stocks[:scan_limit]
    prog_bar = st.progress(0)
    txt_status = st.empty()

    results = []
    total = len(stocks_to_scan)

    for i, code in enumerate(stocks_to_scan):
      txt_status.text(f"Memindai emiten [{i+1}/{total}]: {code}.JK ...")
      res = evaluate_pro_combo(
          code, tf_high, tf_mid, tf_exec, direction=setup_direction
      )
      if res:
        results.append(res)
      prog_bar.progress((i + 1) / total)

    txt_status.empty()
    prog_bar.empty()

    if results:
      df_res = pd.DataFrame(results)
      df_valid = df_res[df_res["Valid"] == True]

      st.markdown("---")
      st.subheader(f"🔥 HASIL FILTER: SAHAM VALID ({setup_direction})")

      if not df_valid.empty:
        st.success(
            f"Ditemukan **{len(df_valid)} emiten** dengan setup Pro"
            f" {setup_direction} yang valid!"
        )
        st.dataframe(
            df_valid[
                [
                    "Emiten",
                    "Harga Terkini",
                    "Estimasi Area Entry",
                    "Status Sinyal",
                ]
            ].style.format({"Harga Terkini": "Rp {:,.0f}"}),
            use_container_width=True,
        )

        selected_valid = st.selectbox(
            "Pilih emiten valid di atas untuk melihat detail chart:",
            df_valid["Emiten"].tolist(),
        )
        matched_row = df_valid[df_valid["Emiten"] == selected_valid].iloc[0]
        d_target = matched_row["DF_Exec"]

        fig = go.Figure()
        fig.add_trace(
            go.Candlestick(
                x=d_target.index,
                open=d_target["Open"],
                high=d_target["High"],
                low=d_target["Low"],
                close=d_target["Close"],
                name="Harga",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=d_target.index,
                y=d_target["TopBB"],
                name="BB Upper",
                line=dict(color="gray", width=1, dash="dot"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=d_target.index,
                y=d_target["MA20"],
                name="Mid BB",
                line=dict(color="orange", width=1.5),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=d_target.index,
                y=d_target["BottomBB"],
                name="BB Lower",
                line=dict(color="gray", width=1, dash="dot"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=d_target.index,
                y=d_target["EMA50"],
                name="EMA 50",
                line=dict(color="magenta", width=1.2),
            )
        )

        fig.update_layout(
            title=f"Chart Eksekusi Pro - {selected_valid}.JK",
            yaxis_title="Harga (IDR)",
            template="plotly_dark",
            height=500,
            xaxis_rangeslider_visible=False,
        )
        st.plotly_chart(fig, use_container_width=True)

      else:
        st.warning(
            f"Belum ada emiten yang memenuhi kriteria mutlak {setup_direction}"
            " saat ini."
        )

      st.markdown("---")
      st.subheader("📋 Seluruh Hasil Pemantauan Pasar")
      st.dataframe(
          df_res[
              ["Emiten", "Harga Terkini", "Estimasi Area Entry", "Status Sinyal"]
          ].style.format({"Harga Terkini": "Rp {:,.0f}"}),
          use_container_width=True,
      )
    else:
      st.error("Gagal menarik data pemindaian pasar.")
  else:
    st.info(
        "👈 Atur batasan scan di sidebar, lalu klik **⚡ MULAI SCANNING PASAR**"
        " untuk memfilter otomatis."
    )

# -----------------------------------------------------------------
# MODE 2: CEK MANUAL
# -----------------------------------------------------------------
else:
  st.subheader(f"🔍 Cek Manual Pro - Analisis Hirarki ({setup_direction})")
  st.markdown(
      "Pilih emiten apa saja untuk melihat rincian validasi per timeframe"
      f" berdasarkan aturan ketat {setup_direction}."
  )

  selected_stock = st.sidebar.selectbox(
      f"Pilih Emiten (Total {len(all_stocks)} Saham)", all_stocks, index=0
  )

  with st.spinner(f"Memuat data analisis pro untuk {selected_stock}.JK ..."):
    res = evaluate_pro_combo(
        selected_stock, tf_high, tf_mid, tf_exec, direction=setup_direction
    )

  if not res:
    st.warning(
        f"⚠️ Gagal memuat data untuk **{selected_stock}.JK**. Periksa likuiditas"
        " saham tersebut."
    )
  else:
    d_exec = res["DF_Exec"]
    latest_price = res["Harga Terkini"]
    prev_price = d_exec["Close"].iloc[-2] if len(d_exec) > 1 else latest_price
    chg = latest_price - prev_price
    pct = (chg / prev_price) * 100 if prev_price > 0 else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        f"Harga {selected_stock}",
        f"Rp {latest_price:,.0f}",
        f"{chg:+,.0f} ({pct:+.2f}%)",
    )
    c2.metric("TF Eksekusi (TF3)", tf_exec.upper())
    c3.metric("TF Konfirmasi (TF2)", tf_mid.upper())
    c4.metric("TF Acuan Re-Entry (TF1)", tf_high.upper())

    st.markdown("---")
    st.subheader(f"🎯 Status Sinyal Pro: {res['Status Sinyal']}")
    st.markdown(f"📍 **Estimasi Area Titik Entry:** **{res['Estimasi Area Entry']}**")

    if res["Valid"]:
      st.success(
          f"Saham **{selected_stock}** saat ini memenuhi **Pro Setup Valid"
          f" ({setup_direction})** secara sempurna berdasarkan hirarki"
          f" {tf_package}!"
      )
    else:
      st.info(
          f"Saham **{selected_stock}** belum memenuhi seluruh kriteria hirarki"
          f" {setup_direction} yang ketat pada saat ini."
      )

    st.markdown("---")
    st.subheader("📌 Matriks Validasi Hirarki Pro (TF1 ➔ TF2 ➔ TF3)")

    matrix_rows = [
        {
            "Peran Hirarki": f"TF1 (Acuan Re-Entry MA 5/10 - {tf_high.upper()})",
            "Status / Kondisi": res["TF1"]["status"],
            "Arah Bias": res["TF1"]["bias"],
            "Keterangan Analisis": res["TF1"]["detail"],
        },
        {
            "Peran Hirarki": (
                f"TF2 (Konfirmasi Extreme/Reject/Re-Entry - {tf_mid.upper()})"
            ),
            "Status / Kondisi": res["TF2"]["status"],
            "Arah Bias": res["TF2"]["bias"],
            "Keterangan Analisis": res["TF2"]["detail"],
        },
        {
            "Peran Hirarki": (
                f"TF3 (Eksekusi/Trigger Full Setup/CSAK/Momentum -"
                f" {tf_exec.upper()})"
            ),
            "Status / Kondisi": res["TF3"]["status"],
            "Arah Bias": res["TF3"]["bias"],
            "Keterangan Analisis": res["TF3"]["detail"],
        },
    ]
    st.dataframe(pd.DataFrame(matrix_rows), use_container_width=True)

    st.markdown("---")

    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=d_exec.index,
            open=d_exec["Open"],
            high=d_exec["High"],
            low=d_exec["Low"],
            close=d_exec["Close"],
            name="Harga",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=d_exec.index,
            y=d_exec["TopBB"],
            name="BB Upper",
            line=dict(color="gray", width=1, dash="dot"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=d_exec.index,
            y=d_exec["MA20"],
            name="Mid BB",
            line=dict(color="orange", width=1.5),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=d_exec.index,
            y=d_exec["BottomBB"],
            name="BB Lower",
            line=dict(color="gray", width=1, dash="dot"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=d_exec.index,
            y=d_exec["EMA50"],
            name="EMA 50",
            line=dict(color="magenta", width=1.2),
        )
    )

    fig.update_layout(
        title=f"Analisis Detail Pro BBMA - {selected_stock}.JK ({tf_exec})",
        yaxis_title="Harga (IDR)",
        template="plotly_dark",
        height=550,
        xaxis_rangeslider_visible=False,
    )
    st.plotly_chart(fig, use_container_width=True)
