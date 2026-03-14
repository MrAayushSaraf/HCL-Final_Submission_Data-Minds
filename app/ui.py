import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from datetime import datetime, timedelta
import calendar

# API setup
PREDICT_URL = "http://localhost:8000/predict"
FORECAST_URL = "http://localhost:8000/forecast"

st.set_page_config(page_title="Store Sales Forecaster", page_icon="📈", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .main .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    h1 { color: #1E3A8A; font-family: 'Helvetica Neue', sans-serif; font-weight: 700; }
    h3 { color: #374151; }
    .stButton>button {
        background-color: #2563EB; color: white; border-radius: 8px;
        padding: 0.5rem 1rem; font-weight: 600; transition: all 0.3s;
        width: 100%;
    }
    .stButton>button:hover { background-color: #1D4ED8; }
    .metric-card {
        background-color: white; padding: 1.2rem; border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); text-align: center; margin-bottom: 1rem;
    }
    .period-badge {
        display: inline-block; background: #EFF6FF; color: #1D4ED8;
        border-radius: 20px; padding: 0.2rem 0.8rem; font-size: 0.85rem;
        font-weight: 600; margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

st.title("📈 Store Sales Forecaster")
st.markdown("Predict **daily**, **weekly**, or **monthly** sales for a product family at any store — powered by an optimized Machine Learning pipeline.")

col1, col2 = st.columns([1, 2])

with col1:
    st.header("Input Parameters")

    with st.form("prediction_form"):
        st.subheader("Forecast Period")
        period = st.radio("Select Forecast Period", ["Daily", "Weekly", "Monthly"], horizontal=True)
        date = st.date_input("Reference Date", datetime.today())

        st.subheader("Store Details")
        
        # 54 Unique store names inspired by Corporacion Favorita (Ecuador)
        STORE_NAMES_LIST = [
            "Megamaxi 6 de Diciembre", "Supermaxi El Bosque", "Aki San Rafael", "Gran Aki Molinero", "Megamaxi Mall del Sol",
            "Supermaxi Policentro", "Aki Mapasingue", "Super Aki Via a la Costa", "Megamaxi Scala", "Supermaxi San Luis",
            "Aki Chillogallo", "Gran Aki Carapungo", "Megamaxi San Francisco", "Supermaxi Americas", "Aki Guamani",
            "Supermaxi Los Chillos", "Megamaxi City Mall", "Gran Aki Tarqui", "Super Aki Mucho Lote", "Aki Duran",
            "Megamaxi Quicentro Sur", "Supermaxi Cumbaya", "Aki Tumbaco", "Gran Aki Centro", "Megamaxi Ceibos",
            "Supermaxi San Marino", "Aki Alborada", "Super Aki Garzota", "Megamaxi San Luis Shopping", "Supermaxi Condado",
            "Aki Calderon", "Gran Aki Pomasqui", "Megamaxi Pacifico", "Supermaxi Manta", "Aki Portoviejo",
            "Super Aki Chone", "Megamaxi Mall del Rio", "Supermaxi Cuenca Centro", "Aki Azogues", "Gran Aki Gualaceo",
            "Megamaxi Paseo Ambato", "Supermaxi Riobamba", "Aki Latacunga Centro", "Super Aki Salcedo", "Megamaxi Machala Sur",
            "Supermaxi El Oro", "Aki Pasaje", "Gran Aki Santa Rosa", "Megamaxi Loja Central", "Supermaxi Zamora",
            "Aki Catamayo", "Supermaxi Playas", "Aki Salinas", "Super Aki Santa Elena"
        ]
        
        store_mapping = {name: i+1 for i, name in enumerate(STORE_NAMES_LIST)}
        selected_store_name = st.selectbox("Store Location", list(store_mapping.keys()))
        store_nbr = store_mapping[selected_store_name]
        city = st.selectbox("Store City", ["Quito", "Guayaquil", "Cuenca", "Ambato", "Santo Domingo",
                                           "Machala", "Latacunga", "Manta", "Riobamba", "Loja",
                                           "Guaranda", "Puyo", "Salinas", "Babahoyo", "Quevedo", "Playas"])
        state = st.selectbox("Store State", ["Pichincha", "Guayas", "Azuay", "Tungurahua",
                                             "Santo Domingo de los Tsachilas", "El Oro", "Cotopaxi",
                                             "Manabi", "Chimborazo", "Loja", "Bolivar", "Pastaza",
                                             "Santa Elena", "Los Rios"])
        STORE_TYPE_MAP = {
            "Hypermarket (Type A)": "A",
            "Supermarket (Type B)": "B",
            "Discount Store (Type C)": "C",
            "Convenience Store (Type D)": "D",
            "Neighborhood Store (Type E)": "E"
        }
        store_type_label = st.selectbox("Store Format / Type", list(STORE_TYPE_MAP.keys()))
        store_type = STORE_TYPE_MAP[store_type_label]
        cluster = st.number_input("Store Cluster", min_value=1, max_value=17, value=1)

        st.subheader("Product & Promotion")
        family = st.selectbox("Product Family", [
            "AUTOMOTIVE", "BABY CARE", "BEAUTY", "BEVERAGES", "BOOKS", "BREAD/BAKERY",
            "CELEBRATION", "CLEANING", "DAIRY", "DELI", "EGGS", "FROZEN FOODS",
            "GROCERY I", "GROCERY II", "HARDWARE", "HOME AND KITCHEN I", "HOME AND KITCHEN II",
            "HOME APPLIANCES", "HOME CARE", "LADIESWEAR", "LAWN AND GARDEN", "LINGERIE",
            "LIQUOR,WINE,BEER", "MAGAZINES", "MEATS", "PERSONAL CARE", "PET SUPPLIES",
            "PLAYERS AND ELECTRONICS", "POULTRY", "PREPARED FOODS", "PRODUCE",
            "SCHOOL AND OFFICE SUPPLIES", "SEAFOOD"
        ])
        onpromotion = st.number_input("Items on Promotion", min_value=0, value=0)

        submit_button = st.form_submit_button(label=f"🔮 Generate {period} Forecast")

# Compute date range based on selected period
def get_date_range(period, ref_date):
    if period == "Daily":
        return ref_date, ref_date
    elif period == "Weekly":
        # Week starting from the reference date
        start = ref_date
        end = ref_date + timedelta(days=6)
        return start, end
    elif period == "Monthly":
        start = ref_date.replace(day=1)
        last_day = calendar.monthrange(ref_date.year, ref_date.month)[1]
        end = ref_date.replace(day=last_day)
        return start, end

with col2:
    # Plausible average prices per unit for each product family (USD)
    FAMILY_PRICES = {
        "AUTOMOTIVE": 15.00, "BABY CARE": 8.50, "BEAUTY": 6.00, "BEVERAGES": 2.50, "BOOKS": 12.00,
        "BREAD/BAKERY": 1.50, "CELEBRATION": 4.00, "CLEANING": 5.00, "DAIRY": 3.00, "DELI": 4.50,
        "EGGS": 2.20, "FROZEN FOODS": 6.50, "GROCERY I": 3.50, "GROCERY II": 8.00, "HARDWARE": 10.00,
        "HOME AND KITCHEN I": 20.00, "HOME AND KITCHEN II": 25.00, "HOME APPLIANCES": 150.00,
        "HOME CARE": 7.50, "LADIESWEAR": 25.00, "LAWN AND GARDEN": 18.00, "LINGERIE": 12.00,
        "LIQUOR,WINE,BEER": 14.00, "MAGAZINES": 5.00, "MEATS": 8.50, "PERSONAL CARE": 4.00,
        "PET SUPPLIES": 11.00, "PLAYERS AND ELECTRONICS": 80.00, "POULTRY": 6.00, "PREPARED FOODS": 5.50,
        "PRODUCE": 2.00, "SCHOOL AND OFFICE SUPPLIES": 3.00, "SEAFOOD": 9.00
    }

    if submit_button:
        start_date, end_date = get_date_range(period, date)

        base_payload = {
            "store_nbr": store_nbr,
            "family": family,
            "onpromotion": onpromotion,
            "city": city,
            "state": state,
            "type": store_type,
            "cluster": cluster
        }

        with st.spinner(f"Generating {period} forecast..."):
            try:
                if period == "Daily":
                    # Use single /predict endpoint
                    payload = {**base_payload, "date": start_date.strftime("%Y-%m-%d")}
                    resp = requests.post(PREDICT_URL, json=payload)

                    if resp.status_code == 200:
                        result = resp.json()
                        predicted = result['predicted_sales']
                        model_used = result['model_used']

                        st.success("Forecast generated successfully!")
                        st.markdown(f'<span class="period-badge">📅 Daily — {start_date.strftime("%d %b %Y")}</span>', unsafe_allow_html=True)
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3 style="color:#4B5563;margin-bottom:0;font-size:1rem;">Predicted Revenue</h3>
                            <h1 style="color:#2563EB;font-size:3.5rem;margin:0.5rem 0;">${(predicted * FAMILY_PRICES.get(family, 5.0)):,.2f}</h1>
                            <p style="color:#6B7280;font-size:0.9rem;">est. revenue for <strong>{family}</strong></p>
                            <hr style="border:1px solid #E5E7EB;margin:0.8rem 0;">
                            <p style="color:#9CA3AF;font-size:0.75rem;">Model: {model_used}</p>
                        </div>""", unsafe_allow_html=True)
                    else:
                        st.error(f"API Error: {resp.text}")

                else:
                    # Use /forecast endpoint for weekly/monthly
                    payload = {
                        **base_payload,
                        "start_date": start_date.strftime("%Y-%m-%d"),
                        "end_date": end_date.strftime("%Y-%m-%d")
                    }
                    resp = requests.post(FORECAST_URL, json=payload)

                    if resp.status_code == 200:
                        result = resp.json()
                        forecasts = result['forecasts']
                        total_sales = result['total_sales']
                        model_used = result['model_used']

                        df_fc = pd.DataFrame(forecasts)
                        df_fc['date'] = pd.to_datetime(df_fc['date'])
                        df_fc = df_fc.sort_values('date')

                        st.success("Forecast generated successfully!")

                        period_label = f"{'Week' if period == 'Weekly' else 'Month'} of {start_date.strftime('%d %b %Y')} → {end_date.strftime('%d %b %Y')}"
                        st.markdown(f'<span class="period-badge">📅 {period} — {period_label}</span>', unsafe_allow_html=True)

                        # Convert predicted units to predicted revenue
                        price_per_unit = FAMILY_PRICES.get(family, 5.0)
                        df_fc['predicted_revenue'] = df_fc['predicted_sales'] * price_per_unit
                        total_revenue = total_sales * price_per_unit

                        # Summary metrics
                        c1, c2, c3 = st.columns(3)
                        with c1:
                            st.markdown(f"""<div class="metric-card">
                                <p style="color:#6B7280;font-size:0.85rem;margin:0">Total {period} Revenue</p>
                                <h2 style="color:#2563EB;font-size:2rem;margin:0.3rem 0">${total_revenue:,.0f}</h2>
                                <p style="color:#9CA3AF;font-size:0.75rem;margin:0">Est. USD</p>
                            </div>""", unsafe_allow_html=True)
                        with c2:
                            avg = df_fc['predicted_revenue'].mean()
                            st.markdown(f"""<div class="metric-card">
                                <p style="color:#6B7280;font-size:0.85rem;margin:0">Daily Avg Revenue</p>
                                <h2 style="color:#16A34A;font-size:2rem;margin:0.3rem 0">${avg:,.0f}</h2>
                                <p style="color:#9CA3AF;font-size:0.75rem;margin:0">per day</p>
                            </div>""", unsafe_allow_html=True)
                        with c3:
                            peak = df_fc.loc[df_fc['predicted_revenue'].idxmax()]
                            st.markdown(f"""<div class="metric-card">
                                <p style="color:#6B7280;font-size:0.85rem;margin:0">Peak Revenue Day</p>
                                <h2 style="color:#D97706;font-size:2rem;margin:0.3rem 0">${peak['predicted_revenue']:,.0f}</h2>
                                <p style="color:#9CA3AF;font-size:0.75rem;margin:0">{peak['date'].strftime('%d %b')}</p>
                            </div>""", unsafe_allow_html=True)

                        # Time-series bar chart
                        st.subheader(f"📊 {period} Revenue Forecast Chart")
                        fig, ax = plt.subplots(figsize=(10, 4))
                        colors = ['#2563EB' if v < df_fc['predicted_revenue'].max() * 0.9 else '#D97706'
                                  for v in df_fc['predicted_revenue']]
                        bars = ax.bar(df_fc['date'], df_fc['predicted_revenue'], color=colors, width=0.7, edgecolor='white')
                        ax.set_xlabel("Date", fontsize=10)
                        ax.set_ylabel("Predicted Revenue ($)", fontsize=10)
                        ax.set_title(f"{family} — {period} Revenue Forecast", fontsize=13, fontweight='bold')
                        ax.tick_params(axis='x', rotation=45, labelsize=7)
                        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))
                        ax.spines['top'].set_visible(False)
                        ax.spines['right'].set_visible(False)
                        ax.grid(axis='y', alpha=0.3, linestyle='--')
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()

                        # Data table
                        with st.expander("View Full Daily Breakdown"):
                            df_display = df_fc[['date', 'predicted_revenue']].copy()
                            df_display['date'] = df_display['date'].dt.strftime('%A, %d %b %Y')
                            df_display.columns = ['Date', 'Predicted Revenue ($)']
                            df_display['Predicted Revenue ($)'] = df_display['Predicted Revenue ($)'].round(2)
                            st.dataframe(df_display, use_container_width=True)

                    else:
                        st.error(f"API Error: {resp.text}")

            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot connect to the API. Please ensure the FastAPI server is running on port 8000.")

    else:
        st.info("👈 Configure your forecast parameters on the left and click 'Generate Forecast'.")
        # Feature importance plot
        try:
            st.subheader("🔍 Model Feature Importance")
            st.caption("These are the top factors driving the model's sales predictions.")
            st.image("reports/figures/feature_importance.png", use_column_width=True)
        except:
            st.write("Train the model to see feature importances here.")
