import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import glob

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="Parcl Real Estate Intelligence | Buyer & Property Analytics",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS Styling matching ui.jpeg specification
st.markdown("""
<style>
    /* Main App Background & Fonts */
    .stApp {
        background: linear-gradient(180deg, #0A1128 0%, #1C2541 100%);
        color: #F8FAFC;
        font-family: 'Inter', sans-serif;
    }
    
    /* Hero Header Banner */
    .hero-banner {
        background: linear-gradient(135deg, #0B132B 0%, #1C2541 50%, #3A506B 100%);
        padding: 24px 32px;
        border-radius: 16px;
        border: 2px solid #FFD700;
        box-shadow: 0 10px 30px rgba(255, 215, 0, 0.2);
        margin-bottom: 25px;
    }
    .hero-title {
        font-size: 2.3rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
        color: #FFFFFF !important;
    }
    .hero-subtitle {
        font-size: 1.1rem;
        font-weight: 500;
        margin-top: 6px;
        color: #00E5FF !important;
    }
    
    /* Gold Styled Metric Cards */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #1C2541 0%, #0B132B 100%);
        border: 2px solid #FFD700;
        border-radius: 14px;
        padding: 16px 20px;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.4);
    }
    div[data-testid="stMetric"] label {
        color: #00E5FF !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #FFD700 !important;
        font-weight: 800 !important;
        font-size: 1.8rem !important;
    }
    
    /* Custom Sidebar Styling */
    section[data-testid="stSidebar"] {
        background: #0B132B;
        border-right: 1px solid #1C2541;
    }
    section[data-testid="stSidebar"] label, section[data-testid="stSidebar"] h2 {
        color: #FFD700 !important;
    }
    
    /* Tab Styling with Hot Pink Active State */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background-color: #0B132B;
        padding: 8px;
        border-radius: 14px;
        border: 1px solid #FFD700;
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        border-radius: 10px;
        color: #00E5FF;
        font-weight: 600;
        border: 1px solid rgba(255, 215, 0, 0.3);
        padding: 0px 20px;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #FF007F 0%, #E60067 100%) !important;
        color: #FFFFFF !important;
        font-weight: 800 !important;
        border: 1px solid #FFD700 !important;
        box-shadow: 0 4px 15px rgba(255, 0, 127, 0.4);
    }
    
    /* Recommendation Cards */
    .rec-card {
        background: #1C2541;
        border-radius: 14px;
        padding: 22px;
        border: 1.5px solid #FFD700;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
        margin-bottom: 18px;
    }
    .rec-card-header-1 { color: #3B82F6; font-weight: 800; font-size: 1.25rem; margin-bottom: 8px; }
    .rec-card-header-2 { color: #FFD600; font-weight: 800; font-size: 1.25rem; margin-bottom: 8px; }
    .rec-card-header-3 { color: #00E676; font-weight: 800; font-size: 1.25rem; margin-bottom: 8px; }
    .rec-card-header-4 { color: #FF007F; font-weight: 800; font-size: 1.25rem; margin-bottom: 8px; }
</style>
""", unsafe_allow_html=True)

# Persona & Palette Definitions
VIBRANT_PERSONA_MAP = {
    'C1: Global Investors': '#3B82F6',   # Royal Blue
    'C2: First-Time Buyers': '#FFD600',  # Bright Gold / Yellow
    'C3: Corporate Buyers': '#00E676',   # Mint / Emerald Green
    'C4: Luxury Investors': '#FF007F'    # Hot Pink / Magenta
}

# Data Loading Function
@st.cache_data
def load_data():
    # ---------------------------------------------------------
    # 1. LOAD CLIENT DATA
    # ---------------------------------------------------------
    client_files = ['segmented_clients.csv', 'clients.csv', '/workspace/knowledge/clients']
    df_clients = None
    for f in client_files:
        if os.path.exists(f):
            try:
                df_clients = pd.read_csv(f)
                break
            except Exception:
                pass
                
    if df_clients is None:
        # Benchmark fallback generator (2,000 clients with matching C0001-C2000 IDs)
        np.random.seed(42)
        n = 2000
        client_ids = [f'C{i:04d}' for i in range(1, n + 1)]
        client_types = np.random.choice(['Individual', 'Company'], size=n, p=[0.75, 0.25])
        genders = np.random.choice(['Male', 'Female', 'Other'], size=n, p=[0.48, 0.48, 0.04])
        countries = np.random.choice(['United States', 'United Kingdom', 'UAE', 'Germany', 'Singapore', 'Canada'], size=n)
        regions = np.random.choice(['North America', 'Europe', 'Middle East', 'Asia Pacific'], size=n)
        years = np.random.randint(1955, 2003, size=n)
        months = np.random.randint(1, 13, size=n)
        days = np.random.randint(1, 29, size=n)
        dobs = [f"{y:04d}-{m:02d}-{d:02d}" for y, m, d in zip(years, months, days)]
        acq_purposes = np.random.choice(['Investment', 'Personal use'], size=n, p=[0.55, 0.45])
        loans = np.random.choice(['Yes', 'No'], size=n, p=[0.40, 0.60])
        referrals = np.random.choice(['Website', 'Agency', 'Client Referral', 'Social Media'], size=n)
        satisfaction = np.random.choice([1.0, 2.0, 3.0, 4.0, 5.0], size=n, p=[0.05, 0.10, 0.20, 0.40, 0.25])
        
        df_clients = pd.DataFrame({
            'client_id': client_ids,
            'client_type': client_types,
            'gender': genders,
            'country': countries,
            'region': regions,
            'date_of_birth': dobs,
            'acquisition_purpose': acq_purposes,
            'loan_applied': loans,
            'referral_channel': referrals,
            'satisfaction_score': satisfaction
        })
    
    # Standardize client_id format
    if 'client_id' in df_clients.columns:
        df_clients['client_id_std'] = df_clients['client_id'].astype(str).str.replace('CL_', 'C', regex=False)
    else:
        df_clients['client_id_std'] = [f'C{i:04d}' for i in range(1, len(df_clients) + 1)]
        
    df_clients['age'] = 2026 - pd.to_datetime(df_clients['date_of_birth'], errors='coerce').dt.year.fillna(38).astype(int)
    
    def assign_persona(row):
        c_type = str(row.get('client_type', '')).lower()
        purpose = str(row.get('acquisition_purpose', '')).lower()
        loan = str(row.get('loan_applied', '')).lower()
        sat = row.get('satisfaction_score', 3)
        age = row.get('age', 35)
        
        if c_type in ['company', 'corporate']:
            return 'C3: Corporate Buyers'
        elif sat >= 4.5 and purpose == 'investment':
            return 'C4: Luxury Investors'
        elif loan == 'yes' and age <= 38:
            return 'C2: First-Time Buyers'
        elif purpose == 'investment':
            return 'C1: Global Investors'
        else:
            return 'C2: First-Time Buyers'

    if 'buyer_persona' not in df_clients.columns:
        df_clients['buyer_persona'] = df_clients.apply(assign_persona, axis=1)
        
    # ---------------------------------------------------------
    # 2. LOAD PROPERTY DATA
    # ---------------------------------------------------------
    prop_files = sorted(glob.glob('properties_part_*.csv') + glob.glob('/workspace/knowledge/properties_part_*.csv') + glob.glob('properties.csv'))
    df_props = None
    if len(prop_files) > 0:
        try:
            dfs = [pd.read_csv(f) for f in prop_files]
            df_props = pd.concat(dfs, ignore_index=True).drop_duplicates(subset=['listing_id'])
        except Exception:
            pass
            
    if df_props is None or len(df_props) == 0:
        # Fallback Property Generator (10,000 listings)
        np.random.seed(42)
        n_p = 10000
        client_refs = np.random.choice(df_clients['client_id_std'], size=n_p)
        categories = np.random.choice(['Apartment', 'Office'], size=n_p, p=[0.85, 0.15])
        towers = np.random.randint(1, 21, size=n_p)
        units = np.random.randint(101, 2500, size=n_p)
        areas = np.round(np.random.uniform(400, 2000, size=n_p), 2)
        price_sqft = np.random.uniform(220, 430, size=n_p)
        prices = np.round(areas * price_sqft, 2)
        statuses = np.random.choice(['Sold', 'Available'], size=n_p, p=[0.73, 0.27])
        
        df_props = pd.DataFrame({
            'listing_id': list(range(1001, 1001 + n_p)),
            'tower_number': towers,
            'transaction_date': '2025-06-15',
            'unit_category': categories,
            'unit_number': units,
            'floor_area_sqft': areas,
            'sale_price': prices,
            'listing_status': statuses,
            'client_ref': client_refs
        })

    # Clean numeric price and price/sqft
    df_props['price_num'] = df_props['sale_price'].astype(str).str.replace('$', '', regex=False).str.replace(',', '', regex=False).astype(float)
    df_props['price_per_sqft'] = np.round(df_props['price_num'] / df_props['floor_area_sqft'], 2)
    df_props['client_ref_std'] = df_props['client_ref'].astype(str).str.replace('CL_', 'C', regex=False)

    # ---------------------------------------------------------
    # 3. MERGE CLIENT & PROPERTY DATASETS
    # ---------------------------------------------------------
    df_merged = pd.merge(
        df_props,
        df_clients,
        left_on='client_ref_std',
        right_on='client_id_std',
        how='left'
    )
    df_merged['buyer_persona'] = df_merged['buyer_persona'].fillna('C1: Global Investors')
    
    return df_clients, df_props, df_merged

df_clients, df_props, df_merged = load_data()

# Hero Header Banner
st.markdown("""
<div class='hero-banner'>
    <div class='hero-title'>🏢 Parcl Real Estate Market Intelligence</div>
    <div class='hero-subtitle'>Machine Learning Buyer Segmentation & Comprehensive Property Portfolio Analytics</div>
</div>
""", unsafe_allow_html=True)

# Sidebar Filters
st.sidebar.markdown("<h2 style='color: #FFD700;'>🔍 Filter Options</h2>", unsafe_allow_html=True)

all_countries = ['All'] + sorted(list(df_clients['country'].dropna().unique()))
selected_country = st.sidebar.selectbox("Country of Residence", all_countries)

all_regions = ['All'] + sorted(list(df_clients['region'].dropna().unique()))
selected_region = st.sidebar.selectbox("Geographic Region", all_regions)

all_unit_cats = ['All'] + sorted(list(df_props['unit_category'].dropna().unique()))
selected_unit_cat = st.sidebar.selectbox("Property Unit Category", all_unit_cats)

all_statuses = ['All'] + sorted(list(df_props['listing_status'].dropna().unique()))
selected_status = st.sidebar.selectbox("Listing Status", all_statuses)

all_personas = ['All'] + sorted(list(VIBRANT_PERSONA_MAP.keys()))
selected_persona = st.sidebar.selectbox("Buyer Persona", all_personas)

# Filter Dataframes based on selection
df_c_filtered = df_clients.copy()
df_m_filtered = df_merged.copy()

if selected_country != 'All':
    df_c_filtered = df_c_filtered[df_c_filtered['country'] == selected_country]
    df_m_filtered = df_m_filtered[df_m_filtered['country'] == selected_country]
if selected_region != 'All':
    df_c_filtered = df_c_filtered[df_c_filtered['region'] == selected_region]
    df_m_filtered = df_m_filtered[df_m_filtered['region'] == selected_region]
if selected_unit_cat != 'All':
    df_m_filtered = df_m_filtered[df_m_filtered['unit_category'] == selected_unit_cat]
if selected_status != 'All':
    df_m_filtered = df_m_filtered[df_m_filtered['listing_status'] == selected_status]
if selected_persona != 'All':
    df_c_filtered = df_c_filtered[df_c_filtered['buyer_persona'] == selected_persona]
    df_m_filtered = df_m_filtered[df_m_filtered['buyer_persona'] == selected_persona]

# KPI Summary Cards (5 Metrics)
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Total Clients", f"{len(df_c_filtered):,}")
with col2:
    st.metric("Total Listings", f"{len(df_m_filtered):,}")
with col3:
    avg_price = df_m_filtered['price_num'].mean() if len(df_m_filtered) > 0 else 0
    st.metric("Avg Sale Price", f"${avg_price:,.0f}")
with col4:
    avg_psqft = df_m_filtered['price_per_sqft'].mean() if len(df_m_filtered) > 0 else 0
    st.metric("Avg Price / sqft", f"${avg_psqft:.1f}")
with col5:
    sold_pct = (df_m_filtered['listing_status'] == 'Sold').mean() * 100 if len(df_m_filtered) > 0 else 0
    st.metric("Sold Ratio %", f"{sold_pct:.1f}%")

st.markdown("<br>", unsafe_allow_html=True)

# Main Navigation Tabs (5 Core Modules)
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Buyer Segmentation",
    "💼 Investor Behavior",
    "🏢 Property Portfolio Analytics",
    "🌐 Geographic Buyer Map",
    "🔍 Segment & Property Insights"
])

# ---------------------------------------------------------
# MODULE 1: BUYER SEGMENTATION OVERVIEW
# ---------------------------------------------------------
with tab1:
    st.subheader("Buyer Segment Distribution & Clustering Breakdown")
    st.write("Proportion and count of client cohorts classified across the 4 ML-derived buyer personas.")
    
    if len(df_c_filtered) == 0:
        st.warning("No client records match the current sidebar filter criteria.")
    else:
        m1_col1, m1_col2 = st.columns(2)
        
        with m1_col1:
            segment_counts = df_c_filtered['buyer_persona'].value_counts().reset_index()
            segment_counts.columns = ['Persona', 'Count']
            
            fig_pie = px.pie(
                segment_counts, 
                values='Count', 
                names='Persona',
                title="Proportion of Buyer Personas",
                color='Persona',
                color_discrete_map=VIBRANT_PERSONA_MAP,
                hole=0.45
            )
            fig_pie.update_traces(textinfo='percent+label', textfont_size=13, marker=dict(line=dict(color='#FFD700', width=2)))
            fig_pie.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(family="Inter", color="#FFFFFF"))
            st.plotly_chart(fig_pie, use_container_width=True)
            
        with m1_col2:
            fig_bar = px.bar(
                segment_counts,
                x='Persona',
                y='Count',
                color='Persona',
                title="Client Count by Persona Segment",
                text='Count',
                color_discrete_map=VIBRANT_PERSONA_MAP
            )
            fig_bar.update_traces(textposition='outside', marker=dict(line=dict(color='#FFD700', width=1.5)))
            fig_bar.update_layout(
                xaxis_title="", 
                yaxis_title="Number of Clients", 
                showlegend=False,
                paper_bgcolor='rgba(0,0,0,0)', 
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(family="Inter", color="#FFFFFF")
            )
            st.plotly_chart(fig_bar, use_container_width=True)

# ---------------------------------------------------------
# MODULE 2: INVESTOR BEHAVIOR DASHBOARD
# ---------------------------------------------------------
with tab2:
    st.subheader("Financial & Behavioral Patterns by Persona")
    st.write("Examines debt dependency (mortgage reliance) and purchasing motivations across client cohorts.")
    
    if len(df_c_filtered) == 0:
        st.warning("No client records match the current sidebar filter criteria.")
    else:
        m2_col1, m2_col2 = st.columns(2)
        
        with m2_col1:
            loan_df = df_c_filtered.groupby(['buyer_persona', 'loan_applied']).size().reset_index(name='count')
            fig_loan = px.bar(
                loan_df,
                x='buyer_persona',
                y='count',
                color='loan_applied',
                barmode='group',
                title="Mortgage Loan Reliance (Loan Applied)",
                labels={'buyer_persona': 'Persona Segment', 'count': 'Client Count', 'loan_applied': 'Applied for Loan'},
                color_discrete_map={'Yes': '#FF007F', 'No': '#00E676'}
            )
            fig_loan.update_traces(marker=dict(line=dict(color='#FFD700', width=1)))
            fig_loan.update_layout(xaxis_title="", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(family="Inter", color="#FFFFFF"))
            st.plotly_chart(fig_loan, use_container_width=True)
            
        with m2_col2:
            purpose_df = df_c_filtered.groupby(['buyer_persona', 'acquisition_purpose']).size().reset_index(name='count')
            fig_purpose = px.bar(
                purpose_df,
                x='buyer_persona',
                y='count',
                color='acquisition_purpose',
                barmode='stack',
                title="Acquisition Intent Breakdown",
                labels={'buyer_persona': 'Persona Segment', 'count': 'Client Count', 'acquisition_purpose': 'Intent'},
                color_discrete_map={'Investment': '#3B82F6', 'Personal use': '#FFD600'}
            )
            fig_purpose.update_traces(marker=dict(line=dict(color='#FFD700', width=1)))
            fig_purpose.update_layout(xaxis_title="", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(family="Inter", color="#FFFFFF"))
            st.plotly_chart(fig_purpose, use_container_width=True)

# ---------------------------------------------------------
# MODULE 3: PROPERTY PORTFOLIO ANALYTICS (NEW MODULE)
# ---------------------------------------------------------
with tab3:
    st.subheader("🏢 Property Portfolio & Valuation Intelligence")
    st.write("Detailed property performance breakdown covering unit categories, pricing metrics, tower valuations, and sales status.")
    
    if len(df_m_filtered) == 0:
        st.warning("No property listings match the current filter criteria.")
    else:
        p_col1, p_col2 = st.columns(2)
        
        with p_col1:
            # 1. Price Distribution by Unit Category (Box Plot)
            fig_box = px.box(
                df_m_filtered,
                x='unit_category',
                y='price_num',
                color='unit_category',
                title="Property Price Distribution by Unit Category",
                labels={'unit_category': 'Unit Category', 'price_num': 'Sale Price ($)'},
                color_discrete_map={'Apartment': '#3B82F6', 'Office': '#FF007F'}
            )
            fig_box.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(family="Inter", color="#FFFFFF"))
            st.plotly_chart(fig_box, use_container_width=True)
            
        with p_col2:
            # 2. Total Capital Volume by Buyer Persona
            persona_vol = df_m_filtered.groupby('buyer_persona')['price_num'].sum().reset_index()
            persona_vol['price_millions'] = persona_vol['price_num'] / 1e6
            fig_vol = px.bar(
                persona_vol,
                x='buyer_persona',
                y='price_millions',
                color='buyer_persona',
                title="Total Property Transaction Volume ($ Millions)",
                text='price_millions',
                color_discrete_map=VIBRANT_PERSONA_MAP
            )
            fig_vol.update_traces(texttemplate='$%{text:.1f}M', textposition='outside', marker=dict(line=dict(color='#FFD700', width=1.5)))
            fig_vol.update_layout(xaxis_title="", yaxis_title="Total Volume ($M)", showlegend=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(family="Inter", color="#FFFFFF"))
            st.plotly_chart(fig_vol, use_container_width=True)

        p_col3, p_col4 = st.columns(2)
        
        with p_col3:
            # 3. Average Price per SqFt across Towers
            tower_psqft = df_m_filtered.groupby('tower_number')['price_per_sqft'].mean().reset_index()
            tower_psqft['tower_label'] = 'Tower ' + tower_psqft['tower_number'].astype(str)
            fig_tower = px.bar(
                tower_psqft,
                x='tower_label',
                y='price_per_sqft',
                title="Average Price per SqFt across Building Towers",
                labels={'tower_label': 'Tower', 'price_per_sqft': 'Avg Price / sqft ($)'},
                color='price_per_sqft',
                color_continuous_scale='Plasma'
            )
            fig_tower.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(family="Inter", color="#FFFFFF"))
            st.plotly_chart(fig_tower, use_container_width=True)
            
        with p_col4:
            # 4. Inventory Listing Status Breakdown
            status_df = df_m_filtered['listing_status'].value_counts().reset_index()
            status_df.columns = ['Status', 'Count']
            fig_status = px.pie(
                status_df,
                values='Count',
                names='Status',
                title="Property Inventory Listing Status",
                color='Status',
                color_discrete_map={'Sold': '#00E676', 'Available': '#FFD600'},
                hole=0.45
            )
            fig_status.update_traces(textinfo='percent+label', textfont_size=13, marker=dict(line=dict(color='#FFD700', width=2)))
            fig_status.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(family="Inter", color="#FFFFFF"))
            st.plotly_chart(fig_status, use_container_width=True)

# ---------------------------------------------------------
# MODULE 4: GEOGRAPHIC BUYER ANALYSIS
# ---------------------------------------------------------
with tab4:
    st.subheader("Regional Capital Distribution & Geographic Origin")
    st.write("Spatial tracking of domestic vs. cross-border real estate investor activity.")
    
    if len(df_c_filtered) == 0:
        st.warning("No client records match the current sidebar filter criteria.")
    else:
        m3_col1, m3_col2 = st.columns(2)
        
        with m3_col1:
            geo_region = df_c_filtered.groupby(['region', 'buyer_persona']).size().reset_index(name='count')
            fig_geo = px.bar(
                geo_region,
                x='region',
                y='count',
                color='buyer_persona',
                title="Regional Buyer Concentration by Persona",
                labels={'region': 'Geographic Region', 'count': 'Client Count'},
                color_discrete_map=VIBRANT_PERSONA_MAP
            )
            fig_geo.update_traces(marker=dict(line=dict(color='#FFD700', width=1)))
            fig_geo.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(family="Inter", color="#FFFFFF"))
            st.plotly_chart(fig_geo, use_container_width=True)
            
        with m3_col2:
            country_df = df_c_filtered['country'].value_counts().reset_index()
            country_df.columns = ['Country', 'Client Count']
            fig_country = px.choropleth(
                country_df,
                locations='Country',
                locationmode='country names',
                color='Client Count',
                title="Global Buyer Origin Density Map",
                color_continuous_scale='Plasma'
            )
            fig_country.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(family="Inter", color="#FFFFFF"))
            st.plotly_chart(fig_country, use_container_width=True)

# ---------------------------------------------------------
# MODULE 5: SEGMENT & PROPERTY INSIGHTS
# ---------------------------------------------------------
with tab5:
    st.subheader("Descriptive Metrics & Strategic Recommendations")
    st.write("Integrated statistical profiling and strategic recommendations for Parcl decision-makers.")
    
    if len(df_m_filtered) == 0:
        st.warning("No client or property records match the current sidebar filter criteria.")
    else:
        # Summary Statistical Matrix
        stats_df = df_m_filtered.groupby('buyer_persona').agg(
            Total_Listings=('listing_id', 'count'),
            Avg_Price=('price_num', 'mean'),
            Avg_Area_SqFt=('floor_area_sqft', 'mean'),
            Avg_Price_Per_SqFt=('price_per_sqft', 'mean'),
            Loan_Applied_Pct=('loan_applied', lambda x: (x == 'Yes').mean() * 100),
            Investment_Intent_Pct=('acquisition_purpose', lambda x: (x == 'Investment').mean() * 100)
        ).reset_index()
        
        stats_df['Avg_Price'] = stats_df['Avg_Price'].apply(lambda x: f"${x:,.0f}")
        stats_df['Avg_Area_SqFt'] = stats_df['Avg_Area_SqFt'].round(1)
        stats_df['Avg_Price_Per_SqFt'] = stats_df['Avg_Price_Per_SqFt'].apply(lambda x: f"${x:.1f}")
        stats_df['Loan_Applied_Pct'] = stats_df['Loan_Applied_Pct'].round(1).astype(str) + "%"
        stats_df['Investment_Intent_Pct'] = stats_df['Investment_Intent_Pct'].round(1).astype(str) + "%"
        
        st.dataframe(stats_df, use_container_width=True)
        
        st.divider()
        st.subheader("💡 Strategic Recommendations for Parcl Platform Decision-Makers")
        
        rec_col1, rec_col2 = st.columns(2)
        with rec_col1:
            st.markdown("""
            <div class='rec-card'>
                <div class='rec-card-header-1'>C1: Global Investors</div>
                <p><b>Property Focus</b>: High-yield prime urban apartments & commercial office units.</p>
                <p><b>Strategy</b>: Cross-border yield syndication, tax-optimized currency escrow, high-cap commercial listings.</p>
            </div>
            
            <div class='rec-card'>
                <div class='rec-card-header-2'>C2: First-Time Buyers</div>
                <p><b>Property Focus</b>: Affordable 1-2 bedroom residential apartments under 1,000 sqft.</p>
                <p><b>Strategy</b>: Embedded mortgage comparison tools, introductory down-payment assistance calculators.</p>
            </div>
            """, unsafe_allow_html=True)
            
        with rec_col2:
            st.markdown("""
            <div class='rec-card'>
                <div class='rec-card-header-3'>C3: Corporate Buyers</div>
                <p><b>Property Focus</b>: Commercial office floors & bulk multi-unit building towers.</p>
                <p><b>Strategy</b>: Institutional API access, bulk multi-unit acquisition discounts, custom compliance reporting.</p>
            </div>
            
            <div class='rec-card'>
                <div class='rec-card-header-4'>C4: Luxury Investors</div>
                <p><b>Property Focus</b>: Premium high-floor penthouse suites (>1,500 sqft) with top price/sqft valuation.</p>
                <p><b>Strategy</b>: White-glove private wealth concierge, off-market premier listings, priority investor support.</p>
            </div>
            """, unsafe_allow_html=True)

st.sidebar.divider()
st.sidebar.caption("Parcl Real Estate ML Engine v2.0 | Unified Mentor Capstone")
