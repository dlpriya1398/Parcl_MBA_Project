import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import glob

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="Parcl Real Estate Intelligence | Buyer Segmentation & Market Analytics",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling using cp.jpeg palette with High-Visibility Tabs & Icons
st.markdown("""
<style>
    /* Main App Light Background */
    .stApp {
        background-color: #FDFBF7 !important;
        color: #354A57 !important;
    }
    
    /* Header Section Banner */
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #861645 0%, #9B3245 40%, #F2822E 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
        padding-bottom: 5px;
    }
    .sub-header {
        font-size: 1.05rem;
        font-weight: 600;
        color: #354A57;
        margin-bottom: 1.5rem;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #F4F0E8 !important;
        border-right: 1px solid #E2DED6 !important;
    }
    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] label {
        color: #861645 !important;
        font-weight: 700 !important;
    }
    
    /* KPI Metric Cards - High Visibility White Container with Accent Borders */
    div[data-testid="stMetric"] {
        background-color: #FFFFFF !important;
        border: 1px solid #E2DED6 !important;
        border-top: 4px solid #F2822E !important;
        border-radius: 10px !important;
        padding: 16px 20px !important;
        box-shadow: 0 4px 10px rgba(53, 74, 87, 0.05) !important;
    }
    div[data-testid="stMetricLabel"] {
        color: #354A57 !important;
        font-size: 0.95rem !important;
        font-weight: 700 !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    div[data-testid="stMetricValue"] {
        color: #861645 !important;
        font-size: 1.85rem !important;
        font-weight: 800 !important;
    }

    /* STREAMLIT TABS & TAB ICONS / EMOJIS VISIBILITY FIX */
    div[data-baseweb="tab-highlight"] {
        background-color: #F2822E !important;
    }
    
    /* Inactive Tabs */
    button[data-baseweb="tab"] {
        background-color: #EAE6DF !important;
        color: #354A57 !important;
        border-radius: 8px 8px 0px 0px !important;
        border: 1px solid #D6CEC4 !important;
        border-bottom: none !important;
        padding: 12px 24px !important;
        margin-right: 6px !important;
        font-weight: 700 !important;
        font-size: 1.05rem !important;
        opacity: 1 !important;
        transition: all 0.2s ease-in-out !important;
    }
    
    /* Ensure Emojis, SVGs, and Icons inside tabs are 100% opaque and sharp */
    button[data-baseweb="tab"] * {
        color: #354A57 !important;
        fill: #354A57 !important;
        opacity: 1 !important;
        font-weight: 800 !important;
    }
    
    /* Active Selected Tab */
    button[data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(135deg, #861645 0%, #F2822E 100%) !important;
        color: #FFFFFF !important;
        border: 1px solid #861645 !important;
        box-shadow: 0px 4px 14px rgba(134, 22, 69, 0.25) !important;
    }
    
    /* Active Selected Tab Emojis, Icons & Text */
    button[data-baseweb="tab"][aria-selected="true"] * {
        color: #FFFFFF !important;
        fill: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important;
        opacity: 1 !important;
    }
    
    /* Tab Hover state */
    button[data-baseweb="tab"]:hover {
        background-color: #D6CEC4 !important;
        color: #861645 !important;
    }
</style>
""", unsafe_allow_html=True)

# Define cp.jpeg Persona Palette Constants
PERSONA_COLORS = {
    'C1: Global Investors': '#861645',  # Rich Magenta Purple
    'C2: First-Time Buyers': '#F2822E', # Sunset Orange
    'C3: Corporate Buyers': '#354A57',  # Deep Slate Teal
    'C4: Luxury Investors': '#85A611'   # Chartreuse / Olive Lime
}

# Data Loading Function (Clients + Merged Property Portfolio)
@st.cache_data
def load_data():
    client_path = "segmented_clients.csv"
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
        df_clients['age'] = 2026 - pd.to_datetime(df_clients['date_of_birth']).dt.year
        
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

    # Load Property Datasets
    prop_files = sorted(glob.glob("properties_part_*.csv"))
    if prop_files:
        df_props = pd.concat([pd.read_csv(f) for f in prop_files], ignore_index=True)
    elif os.path.exists("properties.csv"):
        df_props = pd.read_csv("properties.csv")
    else:
        df_props = pd.DataFrame()

    if not df_props.empty:
        if 'sale_price' in df_props.columns:
            df_props['price_num'] = df_props['sale_price'].astype(str).str.replace('$', '').str.replace(',', '').astype(float)
        else:
            df_props['price_num'] = df_props['floor_area_sqft'] * 300.0
            
        df_props['price_per_sqft'] = (df_props['price_num'] / df_props['floor_area_sqft']).round(2)
        
        def norm_ref(val):
            if pd.isna(val) or val is None:
                return ""
            s = str(val).strip().upper()
            if s.startswith('C') and not s.startswith('CL_') and len(s) > 1 and s[1:].isdigit():
                try:
                    return f"CL_{int(s[1:]):04d}"
                except ValueError:
                    return s
            return s

        df_props['client_ref_norm'] = df_props['client_ref'].apply(norm_ref) if 'client_ref' in df_props.columns else ""
        
        merged_df = pd.merge(
            df_props,
            df_clients[['client_id', 'buyer_persona', 'country', 'region']],
            left_on='client_ref_norm',
            right_on='client_id',
            how='left'
        )
        merged_df['buyer_persona'] = merged_df['buyer_persona'].fillna('C1: Global Investors')
    else:
        merged_df = pd.DataFrame()

    return df_clients, merged_df

df_clients, df_props = load_data()

# Header Section
st.markdown("<div class='main-header'>🏢 Parcl Real Estate Market Intelligence</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>Machine Learning Based Buyer Segmentation & Property Portfolio Intelligence</div>", unsafe_allow_html=True)
st.divider()

# Sidebar Interactive Filters
st.sidebar.header("🔍 Filter Controls")

all_countries = ['All'] + sorted(list(df_clients['country'].unique()))
selected_country = st.sidebar.selectbox("Country of Residence", all_countries)

all_regions = ['All'] + sorted(list(df_clients['region'].unique()))
selected_region = st.sidebar.selectbox("Geographic Region", all_regions)

all_purposes = ['All'] + sorted(list(df_clients['acquisition_purpose'].unique()))
selected_purpose = st.sidebar.selectbox("Acquisition Purpose", all_purposes)

all_types = ['All'] + sorted(list(df_clients['client_type'].unique()))
selected_client_type = st.sidebar.selectbox("Client Type", all_types)

if not df_props.empty:
    all_categories = ['All'] + sorted(list(df_props['unit_category'].dropna().unique()))
    selected_category = st.sidebar.selectbox("Property Unit Category", all_categories)
else:
    selected_category = 'All'

# Apply Filter Logic
df_filtered_clients = df_clients.copy()
if selected_country != 'All':
    df_filtered_clients = df_filtered_clients[df_filtered_clients['country'] == selected_country]
if selected_region != 'All':
    df_filtered_clients = df_filtered_clients[df_filtered_clients['region'] == selected_region]
if selected_purpose != 'All':
    df_filtered_clients = df_filtered_clients[df_filtered_clients['acquisition_purpose'] == selected_purpose]
if selected_client_type != 'All':
    df_filtered_clients = df_filtered_clients[df_filtered_clients['client_type'] == selected_client_type]

if not df_props.empty:
    df_filtered_props = df_props.copy()
    if selected_country != 'All':
        df_filtered_props = df_filtered_props[df_filtered_props['country'] == selected_country]
    if selected_region != 'All':
        df_filtered_props = df_filtered_props[df_filtered_props['region'] == selected_region]
    if selected_category != 'All':
        df_filtered_props = df_filtered_props[df_filtered_props['unit_category'] == selected_category]
else:
    df_filtered_props = pd.DataFrame()

# KPI Summary Cards (5 Cards Layout)
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Total Clients", len(df_filtered_clients))
with col2:
    total_listings = len(df_filtered_props) if not df_filtered_props.empty else 0
    st.metric("Total Property Listings", f"{total_listings:,}")
with col3:
    avg_price = df_filtered_props['price_num'].mean() if not df_filtered_props.empty else 0
    st.metric("Avg Sale Price", f"${avg_price:,.0f}")
with col4:
    avg_sqft_price = df_filtered_props['price_per_sqft'].mean() if not df_filtered_props.empty else 0
    st.metric("Avg Price / SqFt", f"${avg_sqft_price:.1f}")
with col5:
    sold_ratio = (df_filtered_props['listing_status'] == 'Sold').mean() * 100 if not df_filtered_props.empty else 0
    st.metric("Listing Sold Ratio", f"{sold_ratio:.1f}%")

st.markdown("<br>", unsafe_allow_html=True)

# Main Navigation Tabs (High Visibility Tab Header Icons & Text)
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Buyer Segmentation",
    "💼 Investor Behavior",
    "🏢 Property Portfolio",
    "🌐 Geographic Origin",
    "🔍 Strategic Insights"
])

# ---------------------------------------------------------
# MODULE 1: BUYER SEGMENTATION OVERVIEW
# ---------------------------------------------------------
with tab1:
    st.subheader("Buyer Segment Distribution & Persona Breakdown")
    st.write("Distribution of real estate purchasers categorized into ML-derived personas.")
    
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
            fig_pie.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#354A57'))
            st.plotly_chart(fig_pie, use_container_width=True)
            
        with m1_col2:
            fig_bar = px.bar(
                segment_counts,
                x='Persona',
                y='Count',
                color='Persona',
                title="Client Count by Buyer Persona Segment",
                text='Count',
                color_discrete_map=PERSONA_COLORS
            )
            fig_bar.update_layout(
                xaxis_title="", yaxis_title="Number of Clients", showlegend=False,
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#354A57')
            )
            st.plotly_chart(fig_bar, use_container_width=True)

# ---------------------------------------------------------
# MODULE 2: INVESTOR BEHAVIOR DASHBOARD
# ---------------------------------------------------------
with tab2:
    st.subheader("Financial Reliance & Purchasing Motivations")
    st.write("Debt dependency (mortgage loan application) and acquisition purpose per persona.")
    
    if len(df_filtered_clients) == 0:
        st.warning("No client records match the current filter criteria.")
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
                labels={'buyer_persona': 'Persona Segment', 'count': 'Client Count', 'loan_applied': 'Loan Applied'},
                color_discrete_map={'Yes': '#9B3245', 'No': '#354A57'}
            )
            fig_loan.update_layout(xaxis_title="", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#354A57'))
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
            fig_purpose.update_layout(xaxis_title="", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#354A57'))
            st.plotly_chart(fig_purpose, use_container_width=True)

# ---------------------------------------------------------
# MODULE 3: PROPERTY PORTFOLIO ANALYTICS
# ---------------------------------------------------------
with tab3:
    st.subheader("Property Unit Category & Valuation Metrics")
    st.write("Real estate transaction pricing, floor area metrics, and capital volume by persona.")
    
    if df_filtered_props.empty:
        st.warning("No property listings match the current filter criteria.")
    else:
        m3_col1, m3_col2 = st.columns(2)
        
        with m3_col1:
            fig_box = px.box(
                df_filtered_props,
                x='unit_category',
                y='price_num',
                color='unit_category',
                title="Sale Price Distribution by Unit Category (Apartment vs. Office)",
                labels={'unit_category': 'Unit Category', 'price_num': 'Sale Price ($)'},
                color_discrete_map={'Apartment': '#861645', 'Office': '#F2822E'}
            )
            fig_box.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#354A57'))
            st.plotly_chart(fig_box, use_container_width=True)
            
        with m3_col2:
            vol_df = df_filtered_props.groupby('buyer_persona')['price_num'].sum().reset_index()
            vol_df['price_millions'] = vol_df['price_num'] / 1e6
            fig_vol = px.bar(
                vol_df,
                x='buyer_persona',
                y='price_millions',
                color='buyer_persona',
                title="Total Capital Volume ($ Millions) Transacted by Persona",
                text=vol_df['price_millions'].apply(lambda x: f"${x:.1f}M"),
                color_discrete_map=PERSONA_COLORS
            )
            fig_vol.update_layout(
                xaxis_title="", yaxis_title="Capital Volume ($ Millions)", showlegend=False,
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#354A57')
            )
            st.plotly_chart(fig_vol, use_container_width=True)

# ---------------------------------------------------------
# MODULE 4: GEOGRAPHIC ORIGIN ANALYSIS
# ---------------------------------------------------------
with tab4:
    st.subheader("Regional Buyer Concentration & Global Map")
    st.write("Spatial mapping of domestic vs. cross-border real estate investor activity.")
    
    if len(df_filtered_clients) == 0:
        st.warning("No client records match the current filter criteria.")
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
            fig_geo.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#354A57'))
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
                color_continuous_scale=['#F4F0E8', '#F2822E', '#861645']
            )
            fig_country.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#354A57'))
            st.plotly_chart(fig_country, use_container_width=True)

# ---------------------------------------------------------
# MODULE 5: STRATEGIC INSIGHTS PANEL
# ---------------------------------------------------------
with tab5:
    st.subheader("Cohort Metrics & Business Recommendations")
    st.write("Statistical profiles and strategic recommendations for Parcl decision-makers.")
    
    if len(df_filtered_clients) == 0:
        st.warning("No client records match the current filter criteria.")
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
            **C1: Global Investors**
            - *Strategy*: Cross-border yield syndication, tax-optimized currency escrow, high-cap commercial listings.
            - *Targeting*: International digital marketing in primary global financial hubs.
            
            **C2: First-Time Buyers**
            - *Strategy*: Embedded mortgage comparison tools, introductory down-payment assistance calculators.
            - *Targeting*: Social media referral programs and localized housing affordability guides.
            """)
        with rec_col2:
            st.markdown("""
            **C3: Corporate Buyers**
            - *Strategy*: Institutional API access, bulk multi-unit acquisition discounts, custom compliance reporting.
            - *Targeting*: Direct B2B enterprise sales and institutional real estate advisory.
            
            **C4: Luxury Investors**
            - *Strategy*: White-glove private wealth concierge, off-market premier listings, priority investor support.
            - *Targeting*: Exclusive investor summits and private wealth manager partnerships.
            """)

st.sidebar.divider()
st.sidebar.caption("Parcl Real Estate ML Engine v1.0 | Unified Mentor Capstone")
