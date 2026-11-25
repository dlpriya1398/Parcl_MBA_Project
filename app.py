import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import glob
import os

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="Parcl Real Estate Intelligence | Buyer Segmentation & Market Analytics",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling adopting the "Purple Orange Turquoise" Color Palette (from cp.jpeg)
st.markdown("""
<style>
    /* Main App Background & Base Fonts */
    .stApp {
        background: linear-gradient(180deg, #151A1D 0%, #20262A 100%);
        font-family: 'Inter', sans-serif;
        color: #F8FAFC;
    }
    
    /* Hero Header Banner with Purple-Magenta-to-Coral Gradient (cp.jpeg palette) */
    .hero-banner {
        background: linear-gradient(135deg, #5C113A 0%, #8D1C5C 45%, #F2822E 85%, #FF8B15 100%);
        padding: 26px 34px;
        border-radius: 16px;
        color: white;
        border: 2px solid #F2822E;
        box-shadow: 0 12px 28px -5px rgba(141, 28, 92, 0.45);
        margin-bottom: 25px;
    }
    .hero-title {
        font-size: 2.4rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
        color: #FFFFFF !important;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    .hero-subtitle {
        font-size: 1.1rem;
        font-weight: 500;
        opacity: 0.95;
        margin-top: 6px;
        color: #FFE6D5 !important;
    }
    
    /* KPI Metric Cards with Turquoise & Coral Accent Borders */
    div[data-testid="stMetric"] {
        background: linear-gradient(145deg, #1A2024 0%, #252D32 100%);
        border-radius: 14px;
        padding: 18px 22px;
        border-left: 6px solid #FF8B15;
        border-top: 1px solid rgba(242, 130, 46, 0.3);
        border-right: 1px solid rgba(53, 101, 124, 0.3);
        border-bottom: 1px solid rgba(133, 166, 17, 0.3);
        box-shadow: 0 6px 18px rgba(0, 0, 0, 0.25);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 25px rgba(242, 130, 46, 0.35);
    }
    div[data-testid="stMetricLabel"] {
        color: #A0AEC0 !important;
        font-size: 0.9rem !important;
        font-weight: 600 !important;
    }
    div[data-testid="stMetricValue"] {
        color: #FF8B15 !important;
        font-weight: 800 !important;
        font-size: 1.8rem !important;
    }
    
    /* Custom Sidebar Styling */
    section[data-testid="stSidebar"] {
        background: #0E1215;
        border-right: 1px solid #35657C;
    }
    section[data-testid="stSidebar"] .stMarkdown h1, 
    section[data-testid="stSidebar"] .stMarkdown h2, 
    section[data-testid="stSidebar"] .stMarkdown h3,
    section[data-testid="stSidebar"] label {
        color: #F8FAFC !important;
    }
    
    /* Tab Styling with Purple-to-Orange Active Pill */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background-color: #0E1215;
        padding: 8px;
        border-radius: 14px;
        border: 1px solid #35657C;
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        border-radius: 10px;
        color: #A0AEC0;
        font-weight: 600;
        border: none;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #8D1C5C 0%, #FF8B15 100%) !important;
        color: #FFFFFF !important;
        font-weight: 700;
        box-shadow: 0 4px 15px rgba(242, 130, 46, 0.4);
    }
    
    /* Strategy Recommendation Cards (cp.jpeg theme colors) */
    .rec-card {
        background-color: #1A2024;
        border-radius: 14px;
        padding: 22px;
        border: 1px solid #35657C;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        margin-bottom: 18px;
    }
    .rec-card-header-1 { color: #D946EF; font-weight: 700; font-size: 1.2rem; }
    .rec-card-header-2 { color: #FF8B15; font-weight: 700; font-size: 1.2rem; }
    .rec-card-header-3 { color: #00E5FF; font-weight: 700; font-size: 1.2rem; }
    .rec-card-header-4 { color: #A3E635; font-weight: 700; font-size: 1.2rem; }
</style>
""", unsafe_allow_html=True)

# Purple-Orange-Turquoise Palette Definitions (from cp.jpeg)
CP_PERSONA_MAP = {
    'C1: Global Investors': '#8D1C5C',   # Deep Purple / Magenta
    'C2: First-Time Buyers': '#FF8B15',  # Vivid Sunset Orange
    'C3: Corporate Buyers': '#35657C',   # Deep Turquoise / Teal
    'C4: Luxury Investors': '#85A611'    # Chartreuse / Olive Lime
}

# Data Loading Function (Clients + Properties Merged)
@st.cache_data
def load_unified_data():
    # 1. Load Clients Dataset
    client_file = "segmented_clients.csv" if os.path.exists("segmented_clients.csv") else "clients.csv"
    if os.path.exists(client_file):
        df_clients = pd.read_csv(client_file)
    else:
        np.random.seed(42)
        n = 2000
        client_types = np.random.choice(['Individual', 'Company'], size=n, p=[0.75, 0.25])
        genders = np.random.choice(['Male', 'Female', 'Other'], size=n, p=[0.48, 0.48, 0.04])
        countries = np.random.choice(['United States', 'United Kingdom', 'UAE', 'Germany', 'Singapore', 'Canada'], size=n)
        regions = np.random.choice(['North America', 'Europe', 'Middle East', 'Asia Pacific'], size=n)
        dobs = pd.to_datetime(np.random.randint(pd.to_datetime('1955-01-01').value, pd.to_datetime('2002-12-31').value, size=n))
        acq_purposes = np.random.choice(['Investment', 'Personal use'], size=n, p=[0.55, 0.45])
        loans = np.random.choice(['Yes', 'No'], size=n, p=[0.40, 0.60])
        referrals = np.random.choice(['Website', 'Agency', 'Client Referral', 'Social Media'], size=n)
        satisfaction = np.random.choice([1, 2, 3, 4, 5], size=n, p=[0.05, 0.10, 0.20, 0.40, 0.25])
        
        df_clients = pd.DataFrame({
            'client_id': [f'CL_{i:04d}' for i in range(1, n + 1)],
            'client_type': client_types,
            'gender': genders,
            'country': countries,
            'region': regions,
            'date_of_birth': dobs.strftime('%Y-%m-%d'),
            'acquisition_purpose': acq_purposes,
            'loan_applied': loans,
            'referral_channel': referrals,
            'satisfaction_score': satisfaction
        })
        
    if 'age' not in df_clients.columns:
        df_clients['age'] = 2026 - pd.to_datetime(df_clients['date_of_birth'], errors='coerce').dt.year
        df_clients['age'] = df_clients['age'].fillna(35)
        
    if 'buyer_persona' not in df_clients.columns:
        def assign_persona(row):
            c_type = str(row['client_type']).lower()
            purpose = str(row['acquisition_purpose']).lower()
            loan = str(row['loan_applied']).lower()
            sat = row['satisfaction_score']
            age = row['age']
            
            if c_type == 'company' or c_type == 'corporate':
                return 'C3: Corporate Buyers'
            elif sat >= 4.5 and purpose == 'investment':
                return 'C4: Luxury Investors'
            elif loan == 'yes' and age <= 38:
                return 'C2: First-Time Buyers'
            elif purpose == 'investment':
                return 'C1: Global Investors'
            else:
                return 'C2: First-Time Buyers'
        df_clients['buyer_persona'] = df_clients.apply(assign_persona, axis=1)

    # 2. Load Split Property Data Parts
    prop_files = sorted(glob.glob("properties_part_*.csv")) or sorted(glob.glob("/workspace/knowledge/properties_part_*.csv"))
    if prop_files:
        df_props = pd.concat([pd.read_csv(f) for f in prop_files], ignore_index=True)
    elif os.path.exists("properties.csv"):
        df_props = pd.read_csv("properties.csv")
    else:
        # Fallback benchmark property generator
        np.random.seed(42)
        m = 5000
        df_props = pd.DataFrame({
            'listing_id': np.arange(1000, 1000 + m),
            'tower_number': np.random.randint(1, 21, size=m),
            'unit_category': np.random.choice(['Apartment', 'Office'], size=m, p=[0.85, 0.15]),
            'floor_area_sqft': np.random.uniform(400, 2000, size=m).round(2),
            'listing_status': np.random.choice(['Sold', 'Available'], size=m, p=[0.73, 0.27]),
            'client_ref': [f'CL_{i:04d}' for i in np.random.randint(1, 2001, size=m)]
        })
        base_rate = np.where(df_props['unit_category'] == 'Office', 340, 290)
        df_props['price_per_sqft'] = base_rate + np.random.normal(0, 15, size=m)
        df_props['sale_price'] = (df_props['floor_area_sqft'] * df_props['price_per_sqft']).round(2)
        df_props['sale_price'] = df_props['sale_price'].apply(lambda x: f"${x:,.2f}")

    # Process Numeric Sale Price
    if 'sale_price' in df_props.columns:
        df_props['sale_price_clean'] = df_props['sale_price'].astype(str).str.replace('$', '', regex=False).str.replace(',', '', regex=False).astype(float)
    else:
        df_props['sale_price_clean'] = df_props['floor_area_sqft'] * 300.0

    # Calculate Price / SqFt
    df_props['price_per_sqft'] = (df_props['sale_price_clean'] / df_props['floor_area_sqft']).round(2)

    # Normalize client reference key format safely
    def normalize_client_ref(val):
        if pd.isna(val) or val is None:
            return ""
        s = str(val).strip().upper()
        if s.startswith('C') and not s.startswith('CL_') and len(s) > 1 and s[1:].isdigit():
            try:
                return f"CL_{int(s[1:]):04d}"
            except ValueError:
                return s
        return s

    if 'client_ref' in df_props.columns:
        df_props['client_ref_norm'] = df_props['client_ref'].apply(normalize_client_ref)
        df_merged = pd.merge(
            df_props,
            df_clients,
            left_on='client_ref_norm',
            right_on='client_id',
            how='left'
        )
    else:
        df_merged = df_props.copy()
        
    df_merged['buyer_persona'] = df_merged['buyer_persona'].fillna('C1: Global Investors')
    df_merged['unit_category'] = df_merged['unit_category'].fillna('Apartment')

    return df_clients, df_merged

df_clients_raw, df_props_raw = load_unified_data()

# Hero Banner Header Section
st.markdown("""
<div class='hero-banner'>
    <div class='hero-title'>🏢 Parcl Real Estate Market Intelligence</div>
    <div class='hero-subtitle'>Machine Learning Based Buyer Segmentation, Investment Profiling & Property Analytics Dashboard</div>
</div>
""", unsafe_allow_html=True)

# Sidebar Interactive Filters
st.sidebar.markdown("<h2 style='color: #FF8B15;'>🔍 Filter Options</h2>", unsafe_allow_html=True)

all_countries = ['All'] + sorted(list(df_clients_raw['country'].dropna().unique()))
selected_country = st.sidebar.selectbox("Country of Residence", all_countries)

all_regions = ['All'] + sorted(list(df_clients_raw['region'].dropna().unique()))
selected_region = st.sidebar.selectbox("Geographic Region", all_regions)

all_purposes = ['All'] + sorted(list(df_clients_raw['acquisition_purpose'].dropna().unique()))
selected_purpose = st.sidebar.selectbox("Acquisition Purpose", all_purposes)

all_types = ['All'] + sorted(list(df_clients_raw['client_type'].dropna().unique()))
selected_client_type = st.sidebar.selectbox("Client Type", all_types)

all_categories = ['All'] + sorted(list(df_props_raw['unit_category'].dropna().unique()))
selected_category = st.sidebar.selectbox("Unit Category", all_categories)

all_statuses = ['All'] + sorted(list(df_props_raw['listing_status'].dropna().unique()))
selected_status = st.sidebar.selectbox("Listing Status", all_statuses)

# Filter Logic Application
df_clients_filtered = df_clients_raw.copy()
df_props_filtered = df_props_raw.copy()

if selected_country != 'All':
    df_clients_filtered = df_clients_filtered[df_clients_filtered['country'] == selected_country]
    df_props_filtered = df_props_filtered[df_props_filtered['country'] == selected_country]
if selected_region != 'All':
    df_clients_filtered = df_clients_filtered[df_clients_filtered['region'] == selected_region]
    df_props_filtered = df_props_filtered[df_props_filtered['region'] == selected_region]
if selected_purpose != 'All':
    df_clients_filtered = df_clients_filtered[df_clients_filtered['acquisition_purpose'] == selected_purpose]
    df_props_filtered = df_props_filtered[df_props_filtered['acquisition_purpose'] == selected_purpose]
if selected_client_type != 'All':
    df_clients_filtered = df_clients_filtered[df_clients_filtered['client_type'] == selected_client_type]
    df_props_filtered = df_props_filtered[df_props_filtered['client_type'] == selected_client_type]
if selected_category != 'All':
    df_props_filtered = df_props_filtered[df_props_filtered['unit_category'] == selected_category]
if selected_status != 'All':
    df_props_filtered = df_props_filtered[df_props_filtered['listing_status'] == selected_status]

# 5 Top Summary Metric Cards
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Total Clients", f"{len(df_clients_filtered):,}")
with col2:
    st.metric("Total Listings", f"{len(df_props_filtered):,}")
with col3:
    avg_price = df_props_filtered['sale_price_clean'].mean() if len(df_props_filtered) > 0 else 0
    st.metric("Avg Sale Price", f"${avg_price:,.0f}")
with col4:
    avg_sqft_price = df_props_filtered['price_per_sqft'].mean() if len(df_props_filtered) > 0 else 0
    st.metric("Avg Price / sqft", f"${avg_sqft_price:,.1f}")
with col5:
    sold_pct = (df_props_filtered['listing_status'] == 'Sold').mean() * 100 if len(df_props_filtered) > 0 else 0
    st.metric("Sold Ratio %", f"{sold_pct:.1f}%")

st.markdown("<br>", unsafe_allow_html=True)

# Main Navigation Tabs (5 Modules)
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Buyer Segmentation Overview",
    "💼 Investor Behavior Dashboard",
    "🏢 Property Portfolio Analytics",
    "🌐 Geographic Buyer Analysis",
    "🔍 Segment Insights Panel"
])

# ---------------------------------------------------------
# MODULE 1: BUYER SEGMENTATION OVERVIEW
# ---------------------------------------------------------
with tab1:
    st.subheader("Buyer Segment Distribution & Clustering Breakdown")
    st.write("Overview of client proportions classified across the 4 ML-derived buyer personas.")
    
    if len(df_clients_filtered) == 0:
        st.warning("No client records match the current sidebar filter criteria.")
    else:
        m1_col1, m1_col2 = st.columns(2)
        
        with m1_col1:
            segment_counts = df_clients_filtered['buyer_persona'].value_counts().reset_index()
            segment_counts.columns = ['Persona', 'Count']
            
            fig_pie = px.pie(
                segment_counts, 
                values='Count', 
                names='Persona',
                title="Proportion of Buyer Personas",
                color='Persona',
                color_discrete_map=CP_PERSONA_MAP,
                hole=0.45
            )
            fig_pie.update_traces(textinfo='percent+label', textfont_size=13, marker=dict(line=dict(color='#20262A', width=2)))
            fig_pie.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(family="Inter", color="#F8FAFC"))
            st.plotly_chart(fig_pie, use_container_width=True)
            
        with m1_col2:
            fig_bar = px.bar(
                segment_counts,
                x='Persona',
                y='Count',
                color='Persona',
                title="Client Count by Persona Segment",
                text='Count',
                color_discrete_map=CP_PERSONA_MAP
            )
            fig_bar.update_traces(textposition='outside')
            fig_bar.update_layout(
                xaxis_title="", 
                yaxis_title="Number of Clients", 
                showlegend=False,
                paper_bgcolor='rgba(0,0,0,0)', 
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(family="Inter", color="#F8FAFC")
            )
            st.plotly_chart(fig_bar, use_container_width=True)

# ---------------------------------------------------------
# MODULE 2: INVESTOR BEHAVIOR DASHBOARD
# ---------------------------------------------------------
with tab2:
    st.subheader("Financial & Behavioral Patterns by Persona")
    st.write("Examines debt dependency (loan reliance) and purchasing motivations across client cohorts.")
    
    if len(df_clients_filtered) == 0:
        st.warning("No client records match the current sidebar filter criteria.")
    else:
        m2_col1, m2_col2 = st.columns(2)
        
        with m2_col1:
            loan_df = df_clients_filtered.groupby(['buyer_persona', 'loan_applied']).size().reset_index(name='count')
            fig_loan = px.bar(
                loan_df,
                x='buyer_persona',
                y='count',
                color='loan_applied',
                barmode='group',
                title="Mortgage Loan Reliance (Loan Applied)",
                labels={'buyer_persona': 'Persona Segment', 'count': 'Client Count', 'loan_applied': 'Applied for Loan'},
                color_discrete_map={'Yes': '#FF8B15', 'No': '#35657C'}
            )
            fig_loan.update_layout(xaxis_title="", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(family="Inter", color="#F8FAFC"))
            st.plotly_chart(fig_loan, use_container_width=True)
            
        with m2_col2:
            purpose_df = df_clients_filtered.groupby(['buyer_persona', 'acquisition_purpose']).size().reset_index(name='count')
            fig_purpose = px.bar(
                purpose_df,
                x='buyer_persona',
                y='count',
                color='acquisition_purpose',
                barmode='stack',
                title="Acquisition Purpose Intent Breakdown",
                labels={'buyer_persona': 'Persona Segment', 'count': 'Client Count', 'acquisition_purpose': 'Intent'},
                color_discrete_map={'Investment': '#8D1C5C', 'Personal use': '#85A611'}
            )
            fig_purpose.update_layout(xaxis_title="", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(family="Inter", color="#F8FAFC"))
            st.plotly_chart(fig_purpose, use_container_width=True)

# ---------------------------------------------------------
# MODULE 3: PROPERTY PORTFOLIO ANALYTICS
# ---------------------------------------------------------
with tab3:
    st.subheader("🏢 Real Estate Property Portfolio & Price Analytics")
    st.write("Analyzes property valuation distributions, price per square foot, and capital investment per buyer segment.")
    
    if len(df_props_filtered) == 0:
        st.warning("No property listing records match the selected filter criteria.")
    else:
        p_col1, p_col2 = st.columns(2)
        
        with p_col1:
            fig_box = px.box(
                df_props_filtered,
                x='unit_category',
                y='sale_price_clean',
                color='unit_category',
                title="Property Sale Price Distribution by Category",
                labels={'unit_category': 'Category', 'sale_price_clean': 'Sale Price ($)'},
                color_discrete_map={'Apartment': '#8D1C5C', 'Office': '#FF8B15'}
            )
            fig_box.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(family="Inter", color="#F8FAFC"))
            st.plotly_chart(fig_box, use_container_width=True)
            
        with p_col2:
            persona_vol = df_props_filtered.groupby('buyer_persona')['sale_price_clean'].sum().reset_index()
            persona_vol['sale_price_millions'] = (persona_vol['sale_price_clean'] / 1e6).round(2)
            fig_vol = px.bar(
                persona_vol,
                x='buyer_persona',
                y='sale_price_millions',
                color='buyer_persona',
                title="Total Capital Volume Transacted ($ Millions)",
                text='sale_price_millions',
                color_discrete_map=CP_PERSONA_MAP
            )
            fig_vol.update_traces(textposition='outside')
            fig_vol.update_layout(xaxis_title="", yaxis_title="Volume ($M)", showlegend=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(family="Inter", color="#F8FAFC"))
            st.plotly_chart(fig_vol, use_container_width=True)
            
        p_col3, p_col4 = st.columns(2)
        with p_col3:
            tower_sqft = df_props_filtered.groupby('tower_number')['price_per_sqft'].mean().reset_index()
            fig_tower = px.bar(
                tower_sqft,
                x='tower_number',
                y='price_per_sqft',
                title="Average Price per SqFt ($/sqft) by Tower",
                labels={'tower_number': 'Tower #', 'price_per_sqft': 'Avg $/sqft'},
                color_discrete_sequence=['#35657C']
            )
            fig_tower.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(family="Inter", color="#F8FAFC"))
            st.plotly_chart(fig_tower, use_container_width=True)
            
        with p_col4:
            status_df = df_props_filtered['listing_status'].value_counts().reset_index()
            status_df.columns = ['Status', 'Count']
            fig_status = px.pie(
                status_df,
                values='Count',
                names='Status',
                title="Property Listing Inventory Status",
                color='Status',
                color_discrete_map={'Sold': '#85A611', 'Available': '#FF8B15'},
                hole=0.45
            )
            fig_status.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(family="Inter", color="#F8FAFC"))
            st.plotly_chart(fig_status, use_container_width=True)

# ---------------------------------------------------------
# MODULE 4: GEOGRAPHIC BUYER ANALYSIS
# ---------------------------------------------------------
with tab4:
    st.subheader("Regional Capital Distribution & Geographic Origin")
    st.write("Spatial tracking of domestic vs. cross-border real estate investor activity.")
    
    if len(df_clients_filtered) == 0:
        st.warning("No client records match the current sidebar filter criteria.")
    else:
        m3_col1, m3_col2 = st.columns(2)
        
        with m3_col1:
            geo_region = df_clients_filtered.groupby(['region', 'buyer_persona']).size().reset_index(name='count')
            fig_geo = px.bar(
                geo_region,
                x='region',
                y='count',
                color='buyer_persona',
                title="Regional Buyer Concentration by Persona",
                labels={'region': 'Geographic Region', 'count': 'Client Count'},
                color_discrete_map=CP_PERSONA_MAP
            )
            fig_geo.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(family="Inter", color="#F8FAFC"))
            st.plotly_chart(fig_geo, use_container_width=True)
            
        with m3_col2:
            country_df = df_clients_filtered['country'].value_counts().reset_index()
            country_df.columns = ['Country', 'Client Count']
            fig_country = px.choropleth(
                country_df,
                locations='Country',
                locationmode='country names',
                color='Client Count',
                title="Global Buyer Origin Density Map",
                color_continuous_scale='Plasma'
            )
            fig_country.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(family="Inter", color="#F8FAFC"))
            st.plotly_chart(fig_country, use_container_width=True)

# ---------------------------------------------------------
# MODULE 5: SEGMENT INSIGHTS PANEL
# ---------------------------------------------------------
with tab5:
    st.subheader("Descriptive Metrics & Strategic Recommendations")
    st.write("Statistical profiling and targeted business strategies for Parcl platform decision-makers.")
    
    if len(df_clients_filtered) == 0:
        st.warning("No client records match the current sidebar filter criteria.")
    else:
        stats_df = df_clients_filtered.groupby('buyer_persona').agg(
            Total_Clients=('client_id', 'count'),
            Avg_Age=('age', 'mean'),
            Avg_Satisfaction=('satisfaction_score', 'mean'),
            Loan_Applied_Pct=('loan_applied', lambda x: (x == 'Yes').mean() * 100),
            Investment_Intent_Pct=('acquisition_purpose', lambda x: (x == 'Investment').mean() * 100)
        ).reset_index()
        
        stats_df['Avg_Age'] = stats_df['Avg_Age'].round(1)
        stats_df['Avg_Satisfaction'] = stats_df['Avg_Satisfaction'].round(2)
        stats_df['Loan_Applied_Pct'] = stats_df['Loan_Applied_Pct'].round(1).astype(str) + "%"
        stats_df['Investment_Intent_Pct'] = stats_df['Investment_Intent_Pct'].round(1).astype(str) + "%"
        
        st.dataframe(stats_df, use_container_width=True)
        
        st.divider()
        st.subheader("💡 Strategic Recommendations for Parcl")
        
        rec_col1, rec_col2 = st.columns(2)
        with rec_col1:
            st.markdown("""
            <div class='rec-card'>
                <div class='rec-card-header-1'>C1: Global Investors</div>
                <p><b>Strategy</b>: Cross-border yield syndication, tax-optimized currency escrow, high-cap commercial listings.</p>
                <p><b>Targeting</b>: International digital marketing in primary global financial hubs.</p>
            </div>
            
            <div class='rec-card'>
                <div class='rec-card-header-2'>C2: First-Time Buyers</div>
                <p><b>Strategy</b>: Embedded mortgage comparison tools, introductory down-payment assistance calculators.</p>
                <p><b>Targeting</b>: Social media referral programs and localized housing affordability guides.</p>
            </div>
            """, unsafe_allow_html=True)
            
        with rec_col2:
            st.markdown("""
            <div class='rec-card'>
                <div class='rec-card-header-3'>C3: Corporate Buyers</div>
                <p><b>Strategy</b>: Institutional API access, bulk multi-unit acquisition discounts, custom compliance reporting.</p>
                <p><b>Targeting</b>: Direct B2B enterprise sales and institutional real estate advisory.</p>
            </div>
            
            <div class='rec-card'>
                <div class='rec-card-header-4'>C4: Luxury Investors</div>
                <p><b>Strategy</b>: White-glove private wealth concierge, off-market premier listings, priority investor support.</p>
                <p><b>Targeting</b>: Exclusive investor summits and private wealth manager partnerships.</p>
            </div>
            """, unsafe_allow_html=True)

st.sidebar.divider()
st.sidebar.caption("Parcl Real Estate ML Engine v1.0 | Unified Mentor Capstone")
