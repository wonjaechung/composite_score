# 최신 상태로 정리된 전체 코드 (기본 전략 + 최적 전략 + 시그널 비교 탭)

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objs as go
from sklearn.decomposition import PCA

st.set_page_config(layout="wide")
st.title("🎯 코인맵 투자 적합도 스코어")

@st.cache_data
def load_data():
    df = pd.read_csv("glassnode_data.csv", parse_dates=["Date"])
    df.set_index("Date", inplace=True)
    return df

@st.cache_data
def load_score_table():
    return pd.read_csv("optimized_score_table.csv")

df = load_data()
score_table = load_score_table()

def score_mapper_factory(indicator):
    table = score_table[score_table["Indicator"] == indicator]
    def mapper(val):
        for _, row in table.iterrows():
            if row["Lower"] <= val < row["Upper"]:
                return row["Score"]
        return np.nan
    return mapper

# 📌 Tab Layout 설정
tabs = st.tabs(["📈 Score 1 (STH SOPR + Netflow)", "🧠 Score 2 (SOPR + LTH/STH)", "🔍 스코어 비교"])

# 📈 기본 전략 탭
with tabs[0]:
    left, right = st.columns([2, 1])
    indicators = ['STH_SOPR', 'Exchange_Netflow']
    weights = [0.6, 0.4]

    for ind in indicators:
        df[ind] = df[ind].fillna(0)

    df_weekly = df[['BTC_Price'] + indicators].copy()
    for ind, w in zip(indicators, weights):
        mapper = score_mapper_factory(ind)
        df_weekly[f'{ind}_Score'] = df_weekly[ind].apply(mapper).fillna(0)

    df_weekly['Composite_Score'] = df_weekly[[f'{ind}_Score' for ind in indicators]].mul(weights).sum(axis=1)
    df_weekly = df_weekly.resample("W-MON").last().interpolate()
    df_weekly["Weekly_Change"] = df_weekly["Composite_Score"].diff()

    def buy_signal_only(score, change):
        if pd.isna(score): return 0
        if score >= 72 and change > 0: return 1
        return 0

    df_weekly["Signal"] = df_weekly.apply(lambda row: buy_signal_only(row["Composite_Score"], row["Weekly_Change"]), axis=1)
    buy_signals = df_weekly[df_weekly["Signal"] == 1]

    latest = df_weekly.iloc[-1]
    previous = df_weekly.iloc[-2]
    delta = latest["Composite_Score"] - previous["Composite_Score"]
    arrow = "🔼" if delta > 0 else "🔽" if delta < 0 else "⏺"

    with left:
        st.markdown(f"""
        **Composite Score 요약:**
        - 이번주 Score: **{latest["Composite_Score"]:.2f}점**
        - 전주 대비 변화: **{delta:+.2f}** {arrow}
        
        
        **개별 지표 값:**
        - STH_SOPR: **{latest['STH_SOPR']:.4f}**
        - Exchange_Netflow: **{latest['Exchange_Netflow']:.4f}**
        """)

        st.markdown("**최근 4주 추이:**")
        st.dataframe(
            df_weekly[["Composite_Score", "STH_SOPR", "Exchange_Netflow"]]
            .dropna()
            .tail(4)
            .iloc[::-1]
            .round(4)
            .rename(columns={
                "Composite_Score": "복합 점수",
                "STH_SOPR": "STH_SOPR",
                "Exchange_Netflow": "Netflow"
            }),
            use_container_width=True
        )

    with right:
        st.markdown(f"""
        #### 🧮 점수 계산 방식
        - 각 지표는 정규화 후 0~100점으로 환산
        - `STH_SOPR`와 `Exchange_Netflow`를 각각 60%, 40% 가중치로 조합해 복합 점수 산출

        #### 📘 지표 설명
        - `STH_SOPR`: 단기 보유자의 손익 실현 비율 (1 이상: 수익 실현)
        - `Exchange_Netflow`: 거래소로 유입되는 비트코인 수량 (많으면 매도 압력)

        #### 📊 해석 기준
        - Score가 **72 이상**이고 상승 전환 시: ✅ 강한 매수 신호 **(buy)**
        - Score가 **하락세로 전환**될 경우 주의
        """)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_weekly.index, y=df_weekly["BTC_Price"], mode='lines', name="BTC 가격"))
    for idx in buy_signals.index:
        fig.add_trace(go.Scatter(x=[idx], y=[buy_signals.loc[idx, "BTC_Price"]], mode="markers+text", name="BUY",
                                 text=["BUY"], textposition="top center", marker=dict(color="green", size=8)))
    fig.update_layout(title="과거 코인맵 buy 시그널", height=500, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("🎮 과거 buy 시그널 시점 수익률 계산기")
    selected_date = st.selectbox("BUY 시그널 날짜 선택", buy_signals.index.strftime("%Y-%m-%d")[::-1])
    selected_dt = pd.to_datetime(selected_date)
    price_now = df_weekly.loc[selected_dt]["BTC_Price"]

    def calculate_return(date, days):
        try:
            future_date = date + pd.Timedelta(days=days)
            future_price = df_weekly[df_weekly.index >= future_date]["BTC_Price"].iloc[0]
            return_pct = (future_price / price_now - 1) * 100
            return round(return_pct, 2)
        except:
            return None

    cols = st.columns(4)
    for i, d in enumerate([7, 30, 180, 365]):
        pct = calculate_return(selected_dt, d)
        if pct is not None:
            cols[i].metric(label=f"{d}일 뒤 수익률", value=f"{pct:+.2f}%")
        else:
            cols[i].write(f"{d}일 뒤 데이터 없음")

# 🧠 최적 전략 탭
with tabs[1]:
    left, right = st.columns([2, 1])

    indicators = ['SOPR', 'LTH_STH_Supply_Ratio']
    normalized_df = df[indicators].copy()
    for col in normalized_df.columns:
        min_val = normalized_df[col].min()
        max_val = normalized_df[col].max()
        normalized_df[col] = (normalized_df[col] - min_val) / (max_val - min_val)

    pca = PCA(n_components=1)
    composite_score = pca.fit_transform(normalized_df)
    composite_score_scaled = 100 * (composite_score - composite_score.min()) / (composite_score.max() - composite_score.min())
    df['Composite_Score_PCA'] = composite_score_scaled.flatten()

    df_weekly_pca = df[["BTC_Price", "SOPR", "LTH_STH_Supply_Ratio", "Composite_Score_PCA"]].resample("W-MON").first().dropna()
    df_weekly_pca["Weekly_Change"] = df_weekly_pca["Composite_Score_PCA"].diff()

    def buy_signal_pca(score, change):
        if pd.isna(score): return 0
        if score >= 20 and change > 0: return 1
        return 0

    df_weekly_pca["Signal"] = df_weekly_pca.apply(lambda row: buy_signal_pca(row["Composite_Score_PCA"], row["Weekly_Change"]), axis=1)
    buy_signals_pca = df_weekly_pca[df_weekly_pca["Signal"] == 1]

    latest = df_weekly_pca.iloc[-1]
    previous = df_weekly_pca.iloc[-2]
    delta = latest["Composite_Score_PCA"] - previous["Composite_Score_PCA"]
    arrow = "🔼" if delta > 0 else "🔽" if delta < 0 else "⏺"

    with left:
        st.markdown(f"""
        **Composite Score 요약:**
        - 이번주 Score: **{latest["Composite_Score_PCA"]:.2f}점**
        - 전주 대비 변화: **{delta:+.2f}** {arrow}


        **개별 지표 값:**
        - SOPR: **{latest['SOPR']:.4f}**
        - LTH/STH Supply Ratio: **{latest['LTH_STH_Supply_Ratio']:.4f}**
        """)

        st.markdown("**최근 4주 추이:**")
        st.dataframe(
            df_weekly_pca[["Composite_Score_PCA", "SOPR", "LTH_STH_Supply_Ratio"]]
            .dropna()
            .tail(4)
            .iloc[::-1]
            .round(4)
            .rename(columns={
                "Composite_Score_PCA": "복합 점수 (PCA)",
                "SOPR": "SOPR",
                "LTH_STH_Supply_Ratio": "LTH/STH 비율"
            }),
            use_container_width=True
        )

    with right:
        st.markdown("""
        #### 🧮 점수 계산 방식
        - PCA(주성분 분석)를 활용해 `SOPR`, `LTH/STH Supply Ratio` 두 지표를 단일 스코어로 결합합니다.
        - 이 스코어는 0~100 스케일로 정규화되어, 점수가 높을수록 '공포 국면'에 가까움을 의미합니다.

        #### 📘 지표 설명
        - `SOPR`: 전체 시장의 손익 상태. 1보다 작으면 많은 투자자들이 손실 실현 중이라는 뜻.
        - `LTH/STH Supply Ratio`: 장기 보유자와 단기 보유자의 공급 비율. 높을수록 장기 보유자가 시장을 지배 중.

        #### 📊 해석 기준
        - Score가 **20 이하이고 상승 전환 시**: ✅ 강한 매수 신호 **(buy)**
        - Score가 20~60 사이이고 상승세일 경우: ⚠️ 단기 반등 가능성 **(buy)**
        - Score가 80 이상이고 하락 전환 시: 🔥 과열 경고
        """)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_weekly_pca.index, y=df_weekly_pca["BTC_Price"], mode='lines', name="BTC 가격"))
    for idx in buy_signals_pca.index:
        fig.add_trace(go.Scatter(x=[idx], y=[buy_signals_pca.loc[idx, "BTC_Price"]],
                                 mode="markers+text", name="BUY", text=["BUY"],
                                 textposition="top center", marker=dict(color="green", size=8)))
    fig.update_layout(title="과거 코인맵 buy 시그널", height=500, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("🎮 과거 buy 시그널 시점 수익률 계산기")
    selected_date = st.selectbox("BUY 시그널 날짜 선택)", buy_signals_pca.index.strftime("%Y-%m-%d")[::-1], key="pca")
    selected_dt = pd.to_datetime(selected_date)
    price_now = df_weekly_pca.loc[selected_dt]["BTC_Price"]

    def calculate_return_pca(date, days):
        try:
            future_date = date + pd.Timedelta(days=days)
            future_price = df_weekly_pca[df_weekly_pca.index >= future_date]["BTC_Price"].iloc[0]
            return_pct = (future_price / price_now - 1) * 100
            return round(return_pct, 2)
        except:
            return None

    cols = st.columns(4)
    for i, d in enumerate([7, 30, 180, 365]):
        pct = calculate_return_pca(selected_dt, d)
        if pct is not None:
            cols[i].metric(label=f"{d}일 뒤 수익률", value=f"{pct:+.2f}%")
        else:
            cols[i].write(f"{d}일 뒤 데이터 없음")

# 🔍 시그널 비교 탭
with tabs[2]:
    st.subheader("🆚 최신 시그널 비교")
    basic_signal = df_weekly.iloc[-1]["Signal"]
    pca_signal = df_weekly_pca.iloc[-1]["Signal"]
    st.markdown(f"""
    - 📈 Score 1 (STH SOPR + Netflow) **{'BUY' if basic_signal == 1 else '❌ 없음'}**  
    - 🧠 Score 2 (SOPR + LTH/STH): **{'BUY' if pca_signal == 1 else '❌ 없음'}**
    """)
    if basic_signal != pca_signal:
        st.warning("⚠️ 두 스코어간 시그널이 다릅니다. 신중한 해석이 필요합니다.")
    else:
        st.success("✅ 두 스코어 모두 같은 시그널을 주고 있습니다.")

    # 전략 비교 해석 추가 (원문 그대로 사용)
    st.markdown("---")
    st.subheader("🧠 왜 첫 번째 전략은 단기, 두 번째 전략은 중장기 전략인가?")

    st.markdown("""
✅ **첫 번째 전략: STH SOPR + Exchange Netflow (단기 민감형 전략)**

**1. 사용 지표의 특성**  
- STH_SOPR: 1~3개월 이내 보유자의 평균 실현 손익을 반영 → 단기 투자자 심리 변화에 매우 민감함  
- Exchange Netflow: 비트코인 거래소 유입량 → 실시간 매도 압력의 변화 신호  
👉 두 지표 모두 뉴스, 가격 급등락 같은 외부 이벤트에 즉각 반응하며, 단기 추세 전환 포착에 적합함

**2. 스코어링 방식**  
- 수치 기반 절대 구간 점수화 (0~100점) 후, 가중 평균 (STH SOPR 60%, Netflow 40%)  
- 추가적으로 최근 1주일 Composite Score 변화율을 반영하여 즉각적 시그널 생성  
👉 노이즈(단기 변동성)에 빠르게 반응하도록 설계됨

**3. 결과적 특성**  
- 짧은 호흡의 가격 반등을 빠르게 포착 가능  
- 시그널 발생 빈도가 높아 스윙 트레이딩, 단타 매매 전략에 유리  
- 시장 구조 변화보다는 당장의 수급 흐름을 민감하게 반영
    """)

    st.markdown("""
✅ **두 번째 전략: SOPR + LTH/STH Supply Ratio → PCA 기반 (중장기 구조 반영 전략)**

**1. 사용 지표의 특성**  
- SOPR: 전체 시장의 평균 손익 실현 상태 → 중장기 추세를 반영하는 특성이 강함  
- LTH/STH Supply Ratio: 장기 보유자 대비 단기 보유자 비율 → 시장 포지션 구조의 근본적인 변화를 반영  
👉 이 지표들은 단기 수급보다는 시장 참여자들의 포지션 전환 흐름을 나타냄 → 변화가 발생하면 수 주~수 개월 동안 지속되는 경향이 강함

**2. 스코어링 방식**  
- 지표들을 0~1로 정규화한 후, PCA(주성분 분석)로 주요 흐름을 단일 스코어로 압축  
- 생성된 스코어를 0~100 스케일로 재조정하여, 점수가 낮을수록 공포(저점 신호)를 의미  
👉 한 번 형성된 흐름이 단기 급등락에 쉽게 흔들리지 않도록 설계됨

**3. 결과적 특성**  
- 중장기 시장 추세의 전환 지점을 포착 ("이제 상승이 시작됐다"를 탐지)  
- 시그널 빈도는 적지만, 성공 확률이 높고 신호의 질이 우수  
- 단기 노이즈에는 민감하지 않아 포지션 트레이딩 및 중장기 투자 전략에 적합
    """)

    st.markdown("### 🎯 결론 비교 요약")

    st.markdown("""
| 항목 | Score 1 (STH + Netflow) | Score 2 (PCA: SOPR + LTH/STH) |
|:--|:--|:--|
| **목표** | 빠른 타이밍 포착 | 구조적 전환 탐지 |
| **지표 특성** | 단기 심리, 수급 반영 | 보유 구조, 손익 상태 |
| **변화 속도** | 민감, 급격히 움직임 | 완만, 안정적 |
| **시그널 빈도** | 높음 | 낮음 |
| **전략 성격** | 스윙 or 단타 | 포지션 트레이딩 |
| **노이즈 민감도** | 높음 | 낮음 |
    """)

