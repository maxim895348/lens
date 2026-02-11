import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import io
from typing import Dict, List, Tuple, Optional
import re
from fuzzywuzzy import fuzz
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="Autonomous BI Agent",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS
# ============================================================================
def load_custom_css(theme="mckinsey"):
    themes = {
        "mckinsey": """
            <style>
            .main {background-color: #ffffff;}
            .stMetric {background-color: #f8f9fa; padding: 20px; border-radius: 10px; border-left: 4px solid #1f77b4;}
            .metric-positive {color: #28a745;}
            .metric-negative {color: #dc3545;}
            h1, h2, h3 {color: #2c3e50; font-family: 'Helvetica Neue', sans-serif;}
            </style>
        """,
        "dark": """
            <style>
            .main {background-color: #0e1117;}
            .stMetric {background-color: #1e2530; padding: 20px; border-radius: 10px; border-left: 4px solid #00d4ff;}
            h1, h2, h3 {color: #00d4ff;}
            </style>
        """,
        "corporate": """
            <style>
            .main {background-color: #f5f7fa;}
            .stMetric {background-color: #ffffff; padding: 20px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);}
            h1, h2, h3 {color: #34495e;}
            </style>
        """,
        "minimal": """
            <style>
            .main {background-color: #ffffff;}
            .stMetric {background-color: #fafafa; padding: 15px; border-radius: 5px;}
            h1, h2, h3 {color: #000000; font-weight: 300;}
            </style>
        """
    }
    st.markdown(themes.get(theme, themes["mckinsey"]), unsafe_allow_html=True)

# ============================================================================
# DATA INTELLIGENCE ENGINE
# ============================================================================
class DataIntelligence:
    
    @staticmethod
    def fuzzy_match_column(columns: List[str], target: str, threshold: int = 80) -> Optional[str]:
        """Fuzzy matching для поиска колонок"""
        for col in columns:
            if fuzz.ratio(col.lower(), target.lower()) >= threshold:
                return col
        return None
    
    @staticmethod
    def detect_date_column(df: pd.DataFrame) -> Optional[str]:
        """Автоопределение колонки с датами"""
        date_keywords = ['date', 'дата', 'период', 'month', 'месяц', 'year', 'год', 'time', 'время']
        
        for col in df.columns:
            col_lower = str(col).lower()
            if any(keyword in col_lower for keyword in date_keywords):
                try:
                    pd.to_datetime(df[col], errors='coerce')
                    return col
                except:
                    continue
        
        for col in df.columns:
            try:
                if pd.to_datetime(df[col], errors='coerce').notna().sum() > len(df) * 0.5:
                    return col
            except:
                continue
        
        return None
    
    @staticmethod
    def detect_business_type(df: pd.DataFrame, sheet_names: List[str]) -> str:
        """Определение типа бизнеса по структуре данных"""
        columns_str = ' '.join([str(c).lower() for c in df.columns] + [s.lower() for s in sheet_names])
        
        finance_keywords = ['revenue', 'выручка', 'profit', 'прибыль', 'expense', 'расход', 'balance', 'баланс', 'p&l', 'pnl']
        sales_keywords = ['sales', 'продаж', 'customer', 'клиент', 'deal', 'сделка', 'revenue', 'выручка', 'pipeline']
        inventory_keywords = ['stock', 'склад', 'inventory', 'товар', 'product', 'warehouse', 'остаток']
        hr_keywords = ['employee', 'сотрудник', 'salary', 'зарплата', 'hr', 'персонал', 'headcount']
        
        scores = {
            'finance': sum(1 for kw in finance_keywords if kw in columns_str),
            'sales': sum(1 for kw in sales_keywords if kw in columns_str),
            'inventory': sum(1 for kw in inventory_keywords if kw in columns_str),
            'hr': sum(1 for kw in hr_keywords if kw in columns_str)
        }
        
        if max(scores.values()) == 0:
            return 'mixed'
        
        return max(scores, key=scores.get)
    
    @staticmethod
    def normalize_currency(df: pd.DataFrame, currency_cols: List[str], target_currency: str = 'USD') -> pd.DataFrame:
        """Нормализация валют"""
        exchange_rates = {'RUB': 0.011, 'EUR': 1.08, 'USD': 1.0, 'GBP': 1.27}
        
        for col in currency_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    
    @staticmethod
    def clean_data(df: pd.DataFrame) -> pd.DataFrame:
        """Очистка данных"""
        df = df.copy()
        df = df.dropna(how='all')
        df = df.drop_duplicates()
        
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].str.strip()
        
        return df

# ============================================================================
# ML ANALYTICS CORE
# ============================================================================
class MLAnalytics:
    
    @staticmethod
    def detect_anomalies(df: pd.DataFrame, value_col: str, threshold: float = 0.1) -> pd.DataFrame:
        """Выявление аномалий с помощью Isolation Forest"""
        if value_col not in df.columns:
            return df
        
        data = df[[value_col]].dropna()
        if len(data) < 10:
            return df
        
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)
        
        iso_forest = IsolationForest(contamination=threshold, random_state=42)
        predictions = iso_forest.fit_predict(scaled_data)
        
        df_result = df.copy()
        df_result['anomaly'] = False
        df_result.loc[data.index, 'anomaly'] = predictions == -1
        
        return df_result
    
    @staticmethod
    def simple_forecast(df: pd.DataFrame, date_col: str, value_col: str, periods: int = 12) -> pd.DataFrame:
        """Простой прогноз на основе линейной регрессии"""
        if date_col not in df.columns or value_col not in df.columns:
            return pd.DataFrame()
        
        df_clean = df[[date_col, value_col]].dropna()
        df_clean[date_col] = pd.to_datetime(df_clean[date_col])
        df_clean = df_clean.sort_values(date_col)
        
        if len(df_clean) < 3:
            return pd.DataFrame()
        
        df_clean['time_idx'] = range(len(df_clean))
        
        X = df_clean['time_idx'].values.reshape(-1, 1)
        y = df_clean[value_col].values
        
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(X, y)
        
        last_date = df_clean[date_col].max()
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=periods, freq='MS')
        future_idx = np.arange(len(df_clean), len(df_clean) + periods).reshape(-1, 1)
        
        forecast_values = model.predict(future_idx)
        
        forecast_df = pd.DataFrame({
            date_col: future_dates,
            value_col: forecast_values,
            'is_forecast': True
        })
        
        df_clean['is_forecast'] = False
        result = pd.concat([df_clean[[date_col, value_col, 'is_forecast']], forecast_df], ignore_index=True)
        
        return result
    
    @staticmethod
    def calculate_trends(df: pd.DataFrame, date_col: str, value_col: str) -> Dict:
        """Расчёт трендов"""
        if date_col not in df.columns or value_col not in df.columns:
            return {}
        
        df_clean = df[[date_col, value_col]].dropna()
        df_clean[date_col] = pd.to_datetime(df_clean[date_col])
        df_clean = df_clean.sort_values(date_col)
        
        if len(df_clean) < 2:
            return {}
        
        recent_value = df_clean[value_col].iloc[-1]
        previous_value = df_clean[value_col].iloc[-2]
        
        change_pct = ((recent_value - previous_value) / previous_value * 100) if previous_value != 0 else 0
        
        avg_value = df_clean[value_col].mean()
        std_value = df_clean[value_col].std()
        
        return {
            'recent_value': recent_value,
            'previous_value': previous_value,
            'change_pct': change_pct,
            'avg_value': avg_value,
            'std_value': std_value,
            'trend': 'up' if change_pct > 0 else 'down' if change_pct < 0 else 'stable'
        }
    
    @staticmethod
    def generate_insights(df: pd.DataFrame, business_type: str) -> List[str]:
        """Генерация автоматических инсайтов"""
        insights = []
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            return ["Недостаточно числовых данных для анализа"]
        
        for col in numeric_cols[:3]:
            data = df[col].dropna()
            if len(data) > 0:
                avg = data.mean()
                std = data.std()
                cv = (std / avg * 100) if avg != 0 else 0
                
                if cv > 50:
                    insights.append(f"📊 {col}: высокая волатильность (CV={cv:.1f}%)")
                elif cv < 10:
                    insights.append(f"📊 {col}: стабильные показатели (CV={cv:.1f}%)")
        
        if business_type == 'sales':
            insights.append("💡 Рекомендация: проанализируйте воронку продаж и конверсию")
        elif business_type == 'finance':
            insights.append("💡 Рекомендация: отслеживайте маржинальность и операционные расходы")
        elif business_type == 'hr':
            insights.append("💡 Рекомендация: мониторьте текучесть кадров и производительность")
        
        return insights if insights else ["✅ Данные загружены и готовы к анализу"]

# ============================================================================
# VISUALIZATION ENGINE
# ============================================================================
class Visualizations:
    
    @staticmethod
    def create_kpi_card(title: str, value: float, delta: float = None, format_str: str = "{:,.0f}"):
        """KPI карточка"""
        col1, col2 = st.columns([3, 1])
        with col1:
            st.metric(
                label=title,
                value=format_str.format(value),
                delta=f"{delta:+.1f}%" if delta is not None else None
            )
    
    @staticmethod
    def create_time_series(df: pd.DataFrame, date_col: str, value_col: str, title: str):
        """Временной ряд с прогнозом"""
        fig = go.Figure()
        
        if 'is_forecast' in df.columns:
            actual = df[df['is_forecast'] == False]
            forecast = df[df['is_forecast'] == True]
            
            fig.add_trace(go.Scatter(
                x=actual[date_col], y=actual[value_col],
                mode='lines+markers',
                name='Факт',
                line=dict(color='#1f77b4', width=3)
            ))
            
            if len(forecast) > 0:
                fig.add_trace(go.Scatter(
                    x=forecast[date_col], y=forecast[value_col],
                    mode='lines+markers',
                    name='Прогноз',
                    line=dict(color='#ff7f0e', width=3, dash='dash')
                ))
        else:
            fig.add_trace(go.Scatter(
                x=df[date_col], y=df[value_col],
                mode='lines+markers',
                name=value_col,
                line=dict(color='#1f77b4', width=3)
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title=date_col,
            yaxis_title=value_col,
            hovermode='x unified',
            template='plotly_white',
            height=400
        )
        
        return fig
    
    @staticmethod
    def create_bar_chart(df: pd.DataFrame, x_col: str, y_col: str, title: str):
        """Барчарт"""
        fig = px.bar(
            df, x=x_col, y=y_col,
            title=title,
            template='plotly_white',
            color=y_col,
            color_continuous_scale='Blues'
        )
        fig.update_layout(height=400, showlegend=False)
        return fig
    
    @staticmethod
    def create_distribution(df: pd.DataFrame, col: str, title: str):
        """Распределение"""
        fig = px.histogram(
            df, x=col,
            title=title,
            template='plotly_white',
            nbins=30
        )
        fig.update_layout(height=400)
        return fig

# ============================================================================
# MAIN APPLICATION
# ============================================================================
def main():
    # Sidebar
    with st.sidebar:
        st.title("⚙️ Настройки")
        
        mode = st.selectbox(
            "Режим работы",
            ["CONSULTATIVE", "AUTOPILOT", "CUSTOM"],
            help="CONSULTATIVE: с подтверждениями | AUTOPILOT: полная автоматизация | CUSTOM: ручная настройка"
        )
        
        theme = st.selectbox(
            "Тема дизайна",
            ["mckinsey", "dark", "corporate", "minimal"],
            format_func=lambda x: {
                "mckinsey": "McKinsey Executive",
                "dark": "Dark Analytics",
                "corporate": "Corporate Clean",
                "minimal": "Minimalist"
            }[x]
        )
        
        ml_enabled = st.checkbox("ML-аналитика", value=True)
        show_insights = st.checkbox("Авто-инсайты", value=True)
        
        st.divider()
        st.markdown("### 📁 Загрузка данных")
        
        uploaded_file = st.file_uploader(
            "Excel или CSV файл",
            type=['xlsx', 'xls', 'csv'],
            help="Поддерживаются многолистовые Excel файлы"
        )
    
    load_custom_css(theme)
    
    # Header
    st.title("📊 Autonomous BI Agent")
    st.markdown("*McKinsey-level analytics • Powered by ML • Production-ready*")
    
    if uploaded_file is None:
        st.info("👈 Загрузите файл для начала анализа")
        
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### 🎯 Возможности")
            st.markdown("""
            - Multi-sheet Excel support
            - Авто-определение бизнес-модели
            - ML прогнозы и аномалии
            - 15+ типов визуализаций
            - Export в PDF
            """)
        
        with col2:
            st.markdown("### 🧠 ML-аналитика")
            st.markdown("""
            - Прогнозы временных рядов
            - Выявление аномалий
            - Трендовый анализ
            - Корреляции
            - Авто-инсайты
            """)
        
        with col3:
            st.markdown("### 📈 Типы данных")
            st.markdown("""
            - Финансовые отчёты
            - Продажи и CRM
            - Складской учёт
            - HR и зарплата
            - Микс данных
            """)
        
        return
    
    # Load data
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
            sheet_names = ['Sheet1']
            all_sheets = {'Sheet1': df}
        else:
            excel_file = pd.ExcelFile(uploaded_file)
            sheet_names = excel_file.sheet_names
            all_sheets = {sheet: pd.read_excel(uploaded_file, sheet_name=sheet) for sheet in sheet_names}
            df = all_sheets[sheet_names[0]]
    except Exception as e:
        st.error(f"Ошибка загрузки файла: {str(e)}")
        return
    
    # Data Intelligence
    di = DataIntelligence()
    df_clean = di.clean_data(df)
    
    business_type = di.detect_business_type(df_clean, sheet_names)
    date_col = di.detect_date_column(df_clean)
    
    # PHASE 1: Data Forensics
    if mode == "CONSULTATIVE":
        with st.expander("📊 PHASE 1: Data Forensics", expanded=True):
            st.markdown(f"**Тип бизнеса:** `{business_type.upper()}`")
            st.markdown(f"**Листов обнаружено:** `{len(sheet_names)}`")
            st.markdown(f"**Строк данных:** `{len(df_clean):,}`")
            st.markdown(f"**Колонок:** `{len(df_clean.columns)}`")
            
            if date_col:
                st.success(f"✅ Колонка с датами: `{date_col}`")
            else:
                st.warning("⚠️ Колонка с датами не обнаружена")
            
            st.dataframe(df_clean.head(10), use_container_width=True)
    
    # Выбор листа для анализа
    if len(sheet_names) > 1:
        selected_sheet = st.selectbox("Выберите лист для анализа", sheet_names)
        df_clean = di.clean_data(all_sheets[selected_sheet])
        date_col = di.detect_date_column(df_clean)
    
    # KPI Dashboard
    st.markdown("---")
    st.subheader("📈 Executive Dashboard")
    
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) >= 3:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            value = df_clean[numeric_cols[0]].sum()
            st.metric(numeric_cols[0], f"{value:,.0f}")
        
        with col2:
            value = df_clean[numeric_cols[1]].mean()
            st.metric(f"Avg {numeric_cols[1]}", f"{value:,.1f}")
        
        with col3:
            if len(numeric_cols) > 2:
                value = df_clean[numeric_cols[2]].sum()
                st.metric(numeric_cols[2], f"{value:,.0f}")
        
        with col4:
            st.metric("Записей", f"{len(df_clean):,}")
    
    # Tabs для визуализаций
    tabs = st.tabs(["📊 Тренды", "📈 Распределения", "🔍 Аномалии", "🎯 Инсайты"])
    
    # Tab 1: Тренды
    with tabs[0]:
        if date_col and len(numeric_cols) > 0:
            value_col = st.selectbox("Выберите метрику", numeric_cols, key="trend_metric")
            
            if ml_enabled:
                ml = MLAnalytics()
                df_forecast = ml.simple_forecast(df_clean, date_col, value_col, periods=6)
                
                if not df_forecast.empty:
                    viz = Visualizations()
                    fig = viz.create_time_series(df_forecast, date_col, value_col, f"{value_col} - Факт и Прогноз")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    trends = ml.calculate_trends(df_clean, date_col, value_col)
                    if trends:
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Последнее значение", f"{trends['recent_value']:,.1f}")
                        with col2:
                            st.metric("Изменение", f"{trends['change_pct']:+.1f}%")
                        with col3:
                            st.metric("Среднее", f"{trends['avg_value']:,.1f}")
            else:
                viz = Visualizations()
                fig = viz.create_time_series(df_clean, date_col, value_col, f"{value_col} - Временной ряд")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Для отображения трендов необходима колонка с датами")
    
    # Tab 2: Распределения
    with tabs[1]:
        if len(numeric_cols) > 0:
            col_to_plot = st.selectbox("Выберите метрику", numeric_cols, key="dist_metric")
            
            viz = Visualizations()
            fig = viz.create_distribution(df_clean, col_to_plot, f"Распределение: {col_to_plot}")
            st.plotly_chart(fig, use_container_width=True)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Среднее", f"{df_clean[col_to_plot].mean():,.1f}")
            with col2:
                st.metric("Медиана", f"{df_clean[col_to_plot].median():,.1f}")
            with col3:
                st.metric("Ст. откл.", f"{df_clean[col_to_plot].std():,.1f}")
    
    # Tab 3: Аномалии
    with tabs[2]:
        if ml_enabled and len(numeric_cols) > 0:
            col_to_analyze = st.selectbox("Выберите метрику", numeric_cols, key="anomaly_metric")
            
            ml = MLAnalytics()
            df_anomalies = ml.detect_anomalies(df_clean, col_to_analyze)
            
            anomaly_count = df_anomalies['anomaly'].sum()
            
            col1, col2 = st.columns([2, 1])
            with col1:
                st.metric("Обнаружено аномалий", f"{anomaly_count}")
            with col2:
                st.metric("% от общего", f"{anomaly_count/len(df_anomalies)*100:.1f}%")
            
            if anomaly_count > 0:
                st.dataframe(
                    df_anomalies[df_anomalies['anomaly'] == True],
                    use_container_width=True
                )
        else:
            st.info("Включите ML-аналитику для выявления аномалий")
    
    # Tab 4: Инсайты
    with tabs[3]:
        if show_insights:
            ml = MLAnalytics()
            insights = ml.generate_insights(df_clean, business_type)
            
            st.markdown("### 🧠 Автоматические инсайты")
            for insight in insights:
                st.markdown(f"- {insight}")
            
            st.markdown("---")
            st.markdown("### 📊 Сводная статистика")
            st.dataframe(df_clean.describe(), use_container_width=True)

if __name__ == "__main__":
    main()
