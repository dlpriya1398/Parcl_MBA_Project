import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import glob

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="Parcl Real Estate Intelligence | Market & Buyer Dashboard",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS — Light Theme strictly grounded in cp.jpeg palette (Zero Black / Dark Mode)
# Palette: Magenta (#861645), Sunset Orange (#F2822E), Deep Teal (#354A57), Olive Lime (#85A611), Berry (#9B3245), Off-White (#FDFBF7)
st.markdown("""
<style>
    /* Main app light background */
    .stApp {
        background-color: #FDFBF7;
        color: #354A57;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Custom Header Banner */
    .header-container {
        background: linear-gradient(135deg, #861645 0%, #9B3245 45%, #F2822E 100%);
        padding: 24px 30px;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(134, 22, 69, 0.15);
        margin-bottom: 25px;
        border-left: 6px solid #FF8B15;
    }
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #FFFFFF !important;
        margin-bottom: 4px;
        letter-spacing: -0.5px;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #FDFBF7 !important;
        font-weight: 400;
        opacity: 0.95;
    }
    
    /* Light Metric Cards with cp.jpeg Accent Tops */
    .metric-card-1 {
        background: #FFFFFF;
        border: 1px solid #EAE5DC;
        border-top: 4px solid #861645;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(53, 74, 87, 0.05);
    }
    .metric-card-2 {
        background: #FFFFFF;
        border: 1px solid #EAE5DC;
        border-top: 4px solid #F2822E;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(53, 74, 87, 0.05);
    }
    .metric-card-3 {
        background: #FFFFFF;
        border: 1px solid #EAE5DC;
        border-top: 4px solid #354A57;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(53, 74, 87, 0.05);
    }
    .metric-card-4 {
        background: #FFFFFF;
        border: 1px solid #EAE5DC;
        border-top: 4px solid #85A611;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(53, 74, 87, 0.05);
    }
    .metric-card-5 {
        background: #FFFFFF;
        border: 1px solid #EAE5DC;
        border-top: 4px solid #9B3245;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(53, 74, 87, 0.05);
    }
    .metric-label {
        font-size: 0.85rem;
        font-weight: 600;
        color: #354A57;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 800;
        color: #861645;
        margin-top: 4px;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #F4F0E8 !important;
        border-right: 1px solid #E2DCD0;
    }
    section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {
        color: #354A57 !important;
    }
    
    /* Tab Styling with Palette Pill Highlight */
    button[data-baseweb="tab"] {
        background-color: #F4F0E8 !important;
        color: #354A57 !important;
        border-radius: 20px !important;
        padding: 8px 18px !important;
        font-weight: 600 !important;
        border: 1px solid #E2DCD0 !important;
        margin-right: 6px !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(135deg, #861645 0%, #F2822E 100%) !important;
        color: #FFFFFF !important;
        border: none !important;
        box-shadow: 0 3px 10px rgba(134, 22, 69, 0.2) !important;
    }
    
    /* Card/Container Boxes */
    .stPlotlyChart {
        background-color: #FFFFFF;
        border-radius: 10px;
        padding: 10px;
        border: 1px solid #EAE5DC;
        box-shadow: 0 2px 6px rgba(53, 74, 87, 0.04);
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# DATA LOADING ENGINE (Clients + 5 Property CSV Parts)
# ---------------------------------------------------------
@st.cache_data
def load_combined_data():
    # 1. Load Clients Dataset
    client_path = "segmented_clients.csv"
    if not os.path.exists(client_path):
        client_path = "clients.csv"
    
    if os.path.exists(client_path):
        df_clients = pd.read_csv(client_path)
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
    
    if 'date_of_birth' in df_clients.columns:
        df_clients['age'] = 2026 - pd.to_datetime(df_clients['date_of_birth'], errors='coerce').dt.year
        df_clients['age'] = df_clients['age'].fillna(df_clients['age'].median())
    elif 'age' not in df_clients.columns:
        df_clients['age'] = 35.0

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

    # 2. Load Properties Parts (properties_part_1.csv to properties_part_5.csv)
    prop_files = sorted(glob.glob("properties_part_*.csv")) or sorted(glob.glob("/workspace/knowledge/properties_part_*.csv"))
    if prop_files:
        df_props = pd.concat([pd.read_csv(f) for f in prop_files], ignore_index=True)
    elif os.path.exists("properties.csv"):
        df_props = pd.read_csv("properties.csv")
    else:
        # Benchmark properties dataset fallback
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

    # Process Clean Numeric Price & Price per SqFt
    if 'sale_price' in df_props.columns:
        df_props['sale_price_clean'] = df_props['sale_price'].astype(str).str.replace('$', '').str.replace(',', '').astype(float)
    else:
        df_props['sale_price_clean'] = df_props['floor_area_sqft'] * 300.0

    if 'floor_area_sqft' in df_props.columns:
        df_props['price_per_sqft'] = (df_props['sale_price_clean'] / df_props['floor_area_sqft']).round(2)

    # Normalize client_ref key for clean inner join
    def normalize_ref(val):
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
        df_props['client_ref_norm'] = df_props['client_ref'].apply(normalize_ref)
        df_merged = pd.merge(
            df_props,
            df_clients[['client_id', 'buyer_persona', 'country', 'region', 'acquisition_purpose', 'loan_applied', 'client_type', 'satisfaction_score']],
            left_on='client_ref_norm',
            right_on='client_id',
            how='left'
        )
    else:
        df_merged = df_props.copy()

    df_merged['buyer_persona'] = df_merged['buyer_persona'].fillna('C1: Global Investors')
    
    return df_clients, df_merged

df_clients, df_merged = load_combined_data()

# ---------------------------------------------------------
# PALETTE COLOR CONFIGURATION (cp.jpeg Inspired)
# ---------------------------------------------------------
PALETTE = {
    'magenta': '#861645',
    'orange': '#F2822E',
    'teal': '#354A57',
    'lime': '#85A611',
    'berry': '#9B3245',
    'terracotta': '#885344',
    'off_white': '#FDFBF7',
    'card_bg': '#FFFFFF'
}

PERSONA_COLORS = {
    'C1: Global Investors': '#861645',   # Magenta
    'C2: First-Time Buyers': '#F2822E',  # Sunset Orange
    'C3: Corporate Buyers': '#354A57',   # Deep Teal
    'C4: Luxury Investors': '#85A611'    # Olive Lime
}

COLOR_SEQ = ['#861645', '#F2822E', '#354A57', '#85A611', '#9B3245', '#885344']

# ---------------------------------------------------------
# HEADER BANNER
# ---------------------------------------------------------
st.markdown("""
<div class="header-container">
    <div class="main-header">🏢 Parcl Real Estate Market Intelligence</div>
    <div class="sub-header">Machine Learning Based Buyer Segmentation & Investment Profiling Dashboard</div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# SIDEBAR FILTERS
# ---------------------------------------------------------
st.sidebar.header("🔍 Interactive Filter Panel")

all_countries = ['All'] + sorted(list(df_clients['country'].dropna().unique()))
selected_country = st.sidebar.selectbox("Country of Residence", all_countries)

all_regions = ['All'] + sorted(list(df_clients['region'].dropna().unique()))
selected_region = st.sidebar.selectbox("Geographic Region", all_regions)

all_purposes = ['All'] + sorted(list(df_clients['acquisition_purpose'].dropna().unique()))
selected_purpose = st.sidebar.selectbox("Acquisition Purpose", all_purposes)

all_types = ['All'] + sorted(list(df_clients['client_type'].dropna().unique()))
selected_client_type = st.sidebar.selectbox("Client Type", all_types)

if 'unit_category' in df_merged.columns:
    all_units = ['All'] + sorted(list(df_merged['unit_category'].dropna().unique()))
    selected_unit = st.sidebar.selectbox("Unit Category", all_units)
else:
    selected_unit = 'All'

# Apply Filter Logic
df_filtered_clients = df_clients.copy()
df_filtered_props = df_merged.copy()

if selected_country != 'All':
    df_filtered_clients = df_filtered_clients[df_filtered_clients['country'] == selected_country]
    df_filtered_props = df_filtered_props[df_filtered_props['country'] == selected_country]
if selected_region != 'All':
    df_filtered_clients = df_filtered_clients[df_filtered_clients['region'] == selected_region]
    df_filtered_props = df_filtered_props[df_filtered_props['region'] == selected_region]
if selected_purpose != 'All':
    df_filtered_clients = df_filtered_clients[df_filtered_clients['acquisition_purpose'] == selected_purpose]
    df_filtered_props = df_filtered_props[df_filtered_props['acquisition_purpose'] == selected_purpose]
if selected_client_type != 'All':
    df_filtered_clients = df_filtered_clients[df_filtered_clients['client_type'] == selected_client_type]
    df_filtered_props = df_filtered_props[df_filtered_props['client_type'] == selected_client_type]
if selected_unit != 'All' and 'unit_category' in df_filtered_props.columns:
    df_filtered_props = df_filtered_props[df_filtered_props['unit_category'] == selected_unit]

# ---------------------------------------------------------
# 5 SUMMARY METRIC CARDS (LIGHT THEME WITH PALETTE ACCENTS)
# ---------------------------------------------------------
c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.markdown(f"""
    <div class="metric-card-1">
        <div class="metric-label">👥 Total Clients</div>
        <div class="metric-value">{len(df_filtered_clients):,}</div>
    </div>
    """, unsafe_allow_html=True)
with c2:
    st.markdown(f"""
    <div class="metric-card-2">
        <div class="metric-label">🏢 Total Listings</div>
        <div class="metric-value">{len(df_filtered_props):,}</div>
    </div>
    """, unsafe_allow_html=True)
with c3:
    avg_price = df_filtered_props['sale_price_clean'].mean() if len(df_filtered_props) > 0 else 0
    st.markdown(f"""
    <div class="metric-card-3">
        <div class="metric-label">💰 Avg Sale Price</div>
        <div class="metric-value">${avg_price:,.0f}</div>
    </div>
    """, unsafe_allow_html=True)
with c4:
    avg_sqft = df_filtered_props['price_per_sqft'].mean() if len(df_filtered_props) > 0 else 0
    st.markdown(f"""
    <div class="metric-card-4">
        <div class="metric-label">📐 Avg Price / sqft</div>
        <div class="metric-value">${avg_sqft:.1f}</div>
    </div>
    """, unsafe_allow_html=True)
with c5:
    sold_pct = (df_filtered_props['listing_status'] == 'Sold').mean() * 100 if len(df_filtered_props) > 0 and 'listing_status' in df_filtered_props.columns else 0
    st.markdown(f"""
    <div class="metric-card-5">
        <div class="metric-label">🎯 Sold Units %</div>
        <div class="metric-value">{sold_pct:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------
# CORE NAVIGATION TABS (5 MODULES)
# ---------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Buyer Segmentation Overview",
    "💼 Investor Behavior Dashboard",
    "🏢 Property Portfolio Analytics",
    "🌐 Geographic Buyer Analysis",
    "🔍 Segment Insights Panel"
])

# Helper function for Plotly layout consistency (Light Theme)
def apply_light_layout(fig, title_text=""):
    fig.update_layout(
        title=dict(text=title_text, font=dict(color="#354A57", size=16, family="Inter")),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(color="#354A57", family="Inter"),
        margin=dict(l=30, r=30, t=50, b=30),
        xaxis=dict(gridcolor="#F0ECE1", zerolinecolor="#E2DCD0"),
        yaxis=dict(gridcolor="#F0ECE1", zerolinecolor="#E2DCD0")
    )
    return fig

# ---------------------------------------------------------
# MODULE 1: BUYER SEGMENTATION OVERVIEW
# ---------------------------------------------------------
with tab1:
    st.subheader("Buyer Segment Distribution & Clustering Breakdown")
    st.write("Overview of client proportions classified across the 4 ML-derived buyer personas.")
    
    if len(df_filtered_clients) == 0:
        st.warning("No client records match the current sidebar filter criteria.")
    else:
        m1_col1, m1_col2 = st.columns(2)
        
        with m1_col1:
            segment_counts = df_filtered_clients['buyer_persona'].value_counts().reset_index()
            segment_counts.columns = ['Persona', 'Count']
            
            fig_pie = px.pie(
                segment_counts, 
                values='Count', 
                names='Persona',
                title="Proportion of Buyer Personas",
                color='Persona',
                color_discrete_map=PERSONA_COLORS,
                hole=0.45
            )
            fig_pie.update_traces(textinfo='percent+label', marker=dict(line=dict(color='#FFFFFF', width=2)))
            apply_light_layout(fig_pie, "Proportion of Buyer Personas")
            st.plotly_chart(fig_pie, use_container_width=True)
            
        with m1_col2:
            fig_bar = px.bar(
                segment_counts,
                x='Persona',
                y='Count',
                color='Persona',
                title="Client Count by Persona Segment",
                text='Count',
                color_discrete_map=PERSONA_COLORS
            )
            fig_bar.update_traces(marker=dict(line=dict(color='#354A57', width=1)))
            apply_light_layout(fig_bar, "Client Count by Persona Segment")
            fig_bar.update_layout(xaxis_title="", yaxis_title="Number of Clients", showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)

# ---------------------------------------------------------
# MODULE 2: INVESTOR BEHAVIOR DASHBOARD
# ---------------------------------------------------------
with tab2:
    st.subheader("Financial & Behavioral Patterns by Persona")
    st.write("Examines debt dependency (loan reliance) and purchasing motivations across client cohorts.")
    
    if len(df_filtered_clients) == 0:
        st.warning("No client records match the current sidebar filter criteria.")
    else:
        m2_col1, m2_col2 = st.columns(2)
        
        with m2_col1:
            loan_df = df_filtered_clients.groupby(['buyer_persona', 'loan_applied']).size().reset_index(name='count')
            fig_loan = px.bar(
                loan_df,
                x='buyer_persona',
                y='count',
                color='loan_applied',
                barmode='group',
                title="Mortgage Loan Reliance (Loan Applied)",
                labels={'buyer_persona': 'Persona Segment', 'count': 'Client Count', 'loan_applied': 'Applied for Loan'},
                color_discrete_map={'Yes': '#9B3245', 'No': '#354A57'}
            )
            apply_light_layout(fig_loan, "Mortgage Loan Reliance")
            fig_loan.update_layout(xaxis_title="")
            st.plotly_chart(fig_loan, use_container_width=True)
            
        with m2_col2:
            purpose_df = df_filtered_clients.groupby(['buyer_persona', 'acquisition_purpose']).size().reset_index(name='count')
            fig_purpose = px.bar(
                purpose_df,
                x='buyer_persona',
                y='count',
                color='acquisition_purpose',
                barmode='stack',
                title="Acquisition Purpose Intent Breakdown",
                labels={'buyer_persona': 'Persona Segment', 'count': 'Client Count', 'acquisition_purpose': 'Intent'},
                color_discrete_map={'Investment': '#861645', 'Personal use': '#F2822E'}
            )
            apply_light_layout(fig_purpose, "Acquisition Purpose Intent Breakdown")
            fig_purpose.update_layout(xaxis_title="")
            st.plotly_chart(fig_purpose, use_container_width=True)

# ---------------------------------------------------------
# MODULE 3: PROPERTY PORTFOLIO ANALYTICS
# ---------------------------------------------------------
with tab3:
    st.subheader("🏢 Property Portfolio & Valuation Metrics")
    st.write("Analyzes property transaction values, unit categories, and tower price distributions.")
    
    if len(df_filtered_props) == 0:
        st.warning("No property listings match the current filter criteria.")
    else:
        m3_col1, m3_col2 = st.columns(2)
        
        with m3_col1:
            fig_box = px.box(
                df_filtered_props,
                x='unit_category',
                y='sale_price_clean',
                color='unit_category',
                title="Price Distribution by Unit Category",
                labels={'unit_category': 'Category', 'sale_price_clean': 'Sale Price ($)'},
                color_discrete_map={'Apartment': '#861645', 'Office': '#F2822E'}
            )
            apply_light_layout(fig_box, "Price Distribution by Unit Category")
            st.plotly_chart(fig_box, use_container_width=True)
            
        with m3_col2:
            persona_vol = df_filtered_props.groupby('buyer_persona')['sale_price_clean'].sum().reset_index()
            persona_vol['Volume_M'] = persona_vol['sale_price_clean'] / 1e6
            fig_vol = px.bar(
                persona_vol,
                x='buyer_persona',
                y='Volume_M',
                color='buyer_persona',
                title="Total Capital Volume ($ Millions) by Persona",
                text=persona_vol['Volume_M'].apply(lambda x: f"${x:.1f}M"),
                color_discrete_map=PERSONA_COLORS
            )
            apply_light_layout(fig_vol, "Total Capital Volume ($ Millions) by Persona")
            fig_vol.update_layout(xaxis_title="", yaxis_title="Volume ($ Millions)", showlegend=False)
            st.plotly_chart(fig_vol, use_container_width=True)
            
        st.divider()
        m3_col3, m3_col4 = st.columns(2)
        
        with m3_col3:
            tower_price = df_filtered_props.groupby('tower_number')['price_per_sqft'].mean().reset_index()
            fig_tower = px.line(
                tower_price,
                x='tower_number',
                y='price_per_sqft',
                title="Average Price / SqFt Across Building Towers",
                markers=True,
                color_discrete_sequence=['#354A57']
            )
            fig_tower.update_traces(line=dict(width=3, color='#354A57'), marker=dict(size=8, color='#F2822E'))
            apply_light_layout(fig_tower, "Average Price / SqFt Across Building Towers")
            fig_tower.update_layout(xaxis=dict(dtick=1), yaxis_title="Price / SqFt ($)")
            st.plotly_chart(fig_tower, use_container_width=True)
            
        with m3_col4:
            if 'listing_status' in df_filtered_props.columns:
                status_df = df_filtered_props['listing_status'].value_counts().reset_index()
                status_df.columns = ['Status', 'Count']
                fig_status = px.pie(
                    status_df,
                    values='Count',
                    names='Status',
                    title="Property Inventory Listing Status",
                    hole=0.5,
                    color='Status',
                    color_discrete_map={'Sold': '#85A611', 'Available': '#9B3245'}
                )
                apply_light_layout(fig_status, "Property Inventory Listing Status")
                st.plotly_chart(fig_status, use_container_width=True)

# ---------------------------------------------------------
# MODULE 4: GEOGRAPHIC BUYER ANALYSIS
# ---------------------------------------------------------
with tab4:
    st.subheader("Regional Capital Distribution & Geographic Origin")
    st.write("Spatial tracking of domestic vs. cross-border real estate investor activity.")
    
    if len(df_filtered_clients) == 0:
        st.warning("No client records match the current sidebar filter criteria.")
    else:
        m4_col1, m4_col2 = st.columns(2)
        
        with m4_col1:
            geo_region = df_filtered_clients.groupby(['region', 'buyer_persona']).size().reset_index(name='count')
            fig_geo = px.bar(
                geo_region,
                x='region',
                y='count',
                color='buyer_persona',
                title="Regional Buyer Concentration by Persona",
                labels={'region': 'Geographic Region', 'count': 'Client Count'},
                color_discrete_map=PERSONA_COLORS
            )
            apply_light_layout(fig_geo, "Regional Buyer Concentration by Persona")
            st.plotly_chart(fig_geo, use_container_width=True)
            
        with m4_col2:
            country_df = df_filtered_clients['country'].value_counts().reset_index()
            country_df.columns = ['Country', 'Client Count']
            fig_country = px.choropleth(
                country_df,
                locations='Country',
                locationmode='country names',
                color='Client Count',
                title="Global Buyer Origin Density Map",
                color_continuous_scale=[[0, '#FDFBF7'], [0.5, '#F2822E'], [1, '#861645']]
            )
            apply_light_layout(fig_country, "Global Buyer Origin Density Map")
            st.plotly_chart(fig_country, use_container_width=True)

# ---------------------------------------------------------
# MODULE 5: SEGMENT INSIGHTS PANEL
# ---------------------------------------------------------
with tab5:
    st.subheader("Descriptive Metrics & Strategic Recommendations")
    st.write("Statistical profiling and targeted business strategies for Parcl platform decision-makers.")
    
    if len(df_filtered_clients) == 0:
        st.warning("No client records match the current sidebar filter criteria.")
    else:
        stats_df = df_filtered_clients.groupby('buyer_persona').agg(
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
            **C1: Global Investors** (Magenta #861645)
            - *Strategy*: Cross-border yield syndication, tax-optimized currency escrow, high-cap commercial listings.
            - *Targeting*: International digital marketing in primary global financial hubs.
            
            **C2: First-Time Buyers** (Sunset Orange #F2822E)
            - *Strategy*: Embedded mortgage comparison tools, introductory down-payment assistance calculators.
            - *Targeting*: Social media referral programs and localized housing affordability guides.
            """)
        with rec_col2:
            st.markdown("""
            **C3: Corporate Buyers** (Deep Teal #354A57)
            - *Strategy*: Institutional API access, bulk multi-unit acquisition discounts, custom compliance reporting.
            - *Targeting*: Direct B2B enterprise sales and institutional real estate advisory.
            
            **C4: Luxury Investors** (Olive Lime #85A611)
            - *Strategy*: White-glove private wealth concierge, off-market premier listings, priority investor support.
            - *Targeting*: Exclusive investor summits and private wealth manager partnerships.
            """)

st.sidebar.divider()
st.sidebar.caption("Parcl Real Estate ML Engine v2.0 | Unified Mentor Capstone")
