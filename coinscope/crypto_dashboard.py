import pandas as pd
import requests
from datetime import datetime
import streamlit as st
import plotly.graph_objects as go
import numpy as np

# إعداد الصفحة
st.set_page_config(page_title="Cryptocurrency Analysis", layout="wide")

# تعديل الخلفية والألوان
st.markdown(
    """
    <style>
    .stApp {
        background-color: #394a51;
    }
    h1, h2, h3 {
        color: #7fa99b;
    }
    body, .stMarkdown {
        color: #fdc57b;
        font-family: 'Arial', sans-serif;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Fetch USDT trading pairs from Binance
def get_available_symbols():
    url = "https://api.binance.com/api/v3/exchangeInfo"
    response = requests.get(url)
    if response.status_code != 200:
        st.error("Error fetching data from API.")
        return []
    data = response.json()
    symbols = [symbol['symbol'] for symbol in data['symbols'] if
               symbol['status'] == 'TRADING' and 'USDT' in symbol['symbol']]
    return symbols

# Fetch historical crypto data
def get_crypto_data(symbol, interval, start_date, end_date):
    # تحويل تاريخ البداية إلى منتصف الليل
    start_datetime = datetime.combine(start_date, datetime.min.time())

    # إذا كان تاريخ النهاية هو اليوم الحالي، نضبطه للحظة الحالية
    if end_date == datetime.now().date():
        end_datetime = datetime.now()  # حتى اللحظة الحالية
    else:
        end_datetime = datetime.combine(end_date, datetime.max.time())

    # إعداد المعاملات لجلب البيانات
    params = {
        'symbol': symbol,
        'interval': interval,
        'startTime': int(start_datetime.timestamp() * 1000),
        'endTime': int(end_datetime.timestamp() * 1000),
    }

    response = requests.get("https://api.binance.com/api/v3/klines", params=params)
    if response.status_code != 200:
        st.error(f"Error fetching data: {response.status_code}")
        return pd.DataFrame()

    data = response.json()
    df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume", "close_time",
                                     "quote_asset_volume", "number_of_trades", "taker_buy_base_asset_volume",
                                     "taker_buy_quote_asset_volume", "ignore"])

    # تحويل التوقيت إلى صيغة مناسبة
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['timestamp'] = df['timestamp'].dt.strftime('%H:%M')  # عرض الساعة والدقيقة فقط
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
    df = df.apply(pd.to_numeric, errors='ignore')
    return df


# حساب المؤشرات الفنية
def calculate_indicators(df):
    df['SMA_20'] = df['close'].rolling(window=20).mean()
    df['SMA_50'] = df['close'].rolling(window=50).mean()

    # حساب RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # حساب MACD
    df['EMA_12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']

    # مؤشر Super K
    df['SuperK'] = ((df['close'] - df['low']) / (df['high'] - df['low'])) * 100

    # مؤشر ICT (نسخة مبسطة)
    df['Fair Value Gap'] = df['high'] - df['low']
    df['Order Block'] = df['close'].rolling(window=5).mean()

    return df

# حساب Bollinger Bands
def calculate_bollinger_bands(df, window=20):
    df['SMA'] = df['close'].rolling(window=window).mean()
    df['std_dev'] = df['close'].rolling(window=window).std()
    df['Upper Band'] = df['SMA'] + (2 * df['std_dev'])
    df['Lower Band'] = df['SMA'] - (2 * df['std_dev'])
    return df

# حساب النسب المئوية لتغير الأسعار
def price_percentage_change(df):
    df['price_change_pct'] = df['close'].pct_change() * 100
    positive_changes = df[df['price_change_pct'] > 0]
    negative_changes = df[df['price_change_pct'] < 0]
    return positive_changes, negative_changes

# جلب بيانات العمق السوقي (Market Depth)
def get_market_depth(symbol):
    url = f"https://api.binance.com/api/v3/depth?symbol={symbol}&limit=5"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return data
    else:
        return None

# حساب التحليلات المتقدمة
def advanced_analysis(df):
    min_price = df['low'].min()
    max_price = df['high'].max()
    current_range = max_price - min_price

    min_price_count = len(df[df['low'] == min_price])
    max_price_count = len(df[df['high'] == max_price])

    price_moves_up = len(df[df['close'].diff() > 0])
    price_moves_down = len(df[df['close'].diff() < 0])

    # إضافة تحليل Bollinger Bands
    df = calculate_bollinger_bands(df)

    # إضافة تحليل النسب المئوية لتغير الأسعار
    positive_changes, negative_changes = price_percentage_change(df)

    # إضافة تحليل العمق السوقي (Market Depth)
    market_depth = get_market_depth(symbol)

    return min_price, max_price, current_range, min_price_count, max_price_count, price_moves_up, price_moves_down, df, positive_changes, negative_changes, market_depth

# جلب آخر الأخبار من NewsAPI
def get_crypto_news(symbol):
    # استخدم NewsAPI لجلب الأخبار المتعلقة بالعملات المشفرة
    api_key = "dff4a21d41174869a719bdb2b9214d95"  # استبدل بـ مفتاح الـ API الخاص بك
    url = f'https://newsapi.org/v2/everything?q={symbol}&apiKey={api_key}&language=en'
    response = requests.get(url)
    if response.status_code != 200:
        st.error("Error fetching news.")
        return []
    data = response.json()
    articles = data.get('articles', [])
    return articles

# User Interface
st.title('🔍 Cryptocurrency Analysis Using Binance')

# Sidebar inputs
with st.sidebar:
    st.header("📊 Analysis Settings")
    symbols = get_available_symbols()
    if symbols:
        symbol = st.selectbox('Select Cryptocurrency:', symbols, index=symbols.index('BTCUSDT'))
        interval = st.selectbox('Select Interval:', ['1m', '5m', '15m', '1h', '4h', '1d'], index=4)
        start_date = st.date_input('Start Date', value=datetime.now().date())
        end_date = st.date_input('End Date', value=datetime.now().date())
        chart_type = st.selectbox("Select Chart Type:", ['Candlestick', 'Line', 'Bar'])
        deep_analysis = st.checkbox("Enable Deep Dive Analysis")
        if start_date > end_date:
            st.error("Start Date must be before End Date.")
    else:
        st.warning("No available symbols currently.")

# Load data
if st.sidebar.button('📥 Load Data'):
    df = get_crypto_data(symbol, interval, start_date, end_date)

    if not df.empty:
        df = calculate_indicators(df)
        min_price, max_price, current_range, min_price_count, max_price_count, price_moves_up, price_moves_down, df, positive_changes, negative_changes, market_depth = advanced_analysis(df)

        # Tabs for visualization
        tab1, tab2, tab3 = st.tabs(["📈 Chart", "📊 Data Analysis", "💡 Recommendations"])

        with tab1:
            st.subheader(f"Price Chart for {symbol}")

            # رسم الشارت (في حال اخترت "Candlestick" أو "Line")
            if chart_type == 'Candlestick':
                fig = go.Figure(data=[go.Candlestick(x=df['timestamp'],
                                                    open=df['open'], high=df['high'], low=df['low'], close=df['close'])])
                fig.update_layout(title=f'{symbol} Candlestick Chart', xaxis_title='Time', yaxis_title='Price (USDT)')
                st.plotly_chart(fig)

            elif chart_type == 'Line':
                fig = go.Figure(data=[go.Scatter(x=df['timestamp'], y=df['close'], mode='lines')])
                fig.update_layout(title=f'{symbol} Line Chart', xaxis_title='Time', yaxis_title='Price (USDT)')
                st.plotly_chart(fig)

            elif chart_type == 'Bar':
                fig = go.Figure(data=[go.Bar(x=df['timestamp'], y=df['close'])])
                fig.update_layout(title=f'{symbol} Bar Chart', xaxis_title='Time', yaxis_title='Price (USDT)')
                st.plotly_chart(fig)

        with tab2:
            st.subheader("Market Data and Analysis")
            # جدول البيانات الماركت داتا
            st.write(df)

            # عرض التحليلات المتقدمة
            st.write(f"Min Price: {min_price} - Max Price: {max_price}")
            st.write(f"Price Range: {current_range}")
            st.write(f"Frequency of Min Price: {min_price_count} times")
            st.write(f"Frequency of Max Price: {max_price_count} times")
            st.write(f"Number of Price Movements Up: {price_moves_up}")
            st.write(f"Number of Price Movements Down: {price_moves_down}")

            # عرض تغييرات الأسعار الإيجابية والسلبية
            st.write("Positive Price Changes:")
            st.write(positive_changes)
            st.write("Negative Price Changes:")
            st.write(negative_changes)

        with tab3:
            st.subheader("Recommendations and Market Depth")
            if market_depth:
                st.write("Top 5 Market Depth (Bids & Asks):")
                st.write(market_depth)
            else:
                st.warning("Market Depth data not available.")

            st.write(f"Deep Dive Analysis enabled: {deep_analysis}")
            st.write("Consider the analysis for deeper insights!")
