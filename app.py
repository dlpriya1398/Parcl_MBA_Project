import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import glob

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="Parcl Real Estate Market Intelligence",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Theme CSS based on ui_ref.jpeg color palette:
# Background: Soft Pastel Rose (#FFBBC6 / #FDF0F3)
# Primary Accent: Deep Violet / Purple (#2C005A)
# Secondary Accent: Dark Teal (#008C8D)
# Highlight Accent: Vivid Magenta (#820C6D)
# Cards: Pure White (#FFFFFF) with rounded corners & soft drop shadow
st.markdown("""
<style>
    /* Global App Background */
    .stApp {
        background-color: #FDF0F3;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #FCE4EC;
        border-right: 1px solid #F8BBD0;
    }
    section[data-testid="stSidebar"] .stMarkdown h1, 
    section[data-testid="stSidebar"] .stMarkdown h2, 
    section[data-testid="stSidebar"] .stMarkdown h3,
    section[data-testid="stSidebar"] label {
        color: #2C005A !important;
        font-weight: 600;
    }
    
    /* Header Section */
    .header-banner {
        background: linear-gradient(135deg, #2C005A 0%, #820C6D 100%);
        padding: 24px 32px;
        border-radius: 16px;
        color: #FFFFFF;
        box-shadow: 0 8px 24px rgba(44, 0, 90, 0.15);
        margin-bottom: 24px;
    }
    .header-title {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        margin: 0;
        color: #FFFFFF;
    }
    .header-subtitle {
        font-size: 1.05rem;
        color: #FCE4EC;
        margin-top: 6px;
        font-weight: 400;
    }
    
    /* KPI Metric Cards */
    .metric-card {
        background-color: #FFFFFF;
        border-radius: 14px;
        padding: 20px 16px;
        text-align: center;
        border: 1px solid #F8BBD0;
        box-shadow: 0 4px 12px rgba(44, 0, 90, 0.05);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(44, 0, 90, 0.1);
    }
    .metric-label {
        font-size: 0.88rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 8px;
    }
    .metric-value {
        font-size: 1.85rem;
        font-weight: 800;
        color: #2C005A;
    }
    .metric-sub {
        font-size: 0.8rem;
        color: #008C8D;
        font-weight: 600;
        margin-top: 4px;
    }

    /* Tab Styling */
    div[data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
        padding-bottom: 8px;
    }
    button[data-baseweb="tab"] {
        background-color: #FFFFFF !important;
        border: 1px solid #F8BBD0 !important;
        border-radius: 10px !important;
        padding: 10px 20px !important;
        color: #2C005A !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        box-shadow: 0 2px 6px rgba(44, 0, 90, 0.04) !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(135deg, #2C005A 0%, #008C8D 100%) !important;
        color: #FFFFFF !important;
        border: none !important;
        box-shadow: 0 4px 12px rgba(44, 0, 90, 0.2) !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] * {
        color: #FFFFFF !important;
    }
    
    /* Card Container */
    .content-card {
        background-color: #FFFFFF;
        border-radius: 14px;
        padding: 24px;
        border: 1px solid #F8BBD0;
        box-shadow: 0 4px 12px rgba(44, 0, 90, 0.05);
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Data Loading Function (Includes properties_part_1..5 join fallback)
@st.cache_data
def load_data():
    file_path = "segmented_clients.csv"
    if os.path.exists(file_path):
        df_clients = pd.read_csv(file_path)
    else:
        # Fallback generator if segmented_clients.csv is missing
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
        df_clients['kmeans_cluster'] = df_clients['buyer_persona'].map({
            'C1: Global Investors': 0,
            'C2: First-Time Buyers': 1,
            'C3: Corporate Buyers': 2,
            'C4: Luxury Investors': 3
        })
        
    # Load properties data if available
    prop_files = sorted(glob.glob('properties_part_*.csv')) or sorted(glob.glob('/workspace/knowledge/properties_part_*.csv'))
    if prop_files:
        df_props = pd.concat([pd.read_csv(f) for f in prop_files], ignore_index=True)
        if 'sale_price' in df_props.columns:
            df_props['price_clean'] = df_props['sale_price'].astype(str).str.replace('$', '').str.replace(',', '').astype(float)
        else:
            df_props['price_clean'] = df_props['floor_area_sqft'] * 300.0
        
        if 'floor_area_sqft' in df_props.columns:
            df_props['price_per_sqft'] = df_props['price_clean'] / df_props['floor_area_sqft']
            
        def norm_key(x):
            if pd.isna(x): return ""
            s = str(x).strip().upper()
            if s.startswith('C') and not s.startswith('CL_') and s[1:].isdigit():
                return f"CL_{int(s[1:]):04d}"
            return s
            
        if 'client_ref' in df_props.columns:
            df_props['client_ref_norm'] = df_props['client_ref'].apply(norm_key)
            df_merged = pd.merge(df_props, df_clients, left_on='client_ref_norm', right_on='client_id', how='left')
        else:
            df_merged = df_props.copy()
            df_merged['buyer_persona'] = 'C1: Global Investors'
    else:
        df_merged = df_clients.copy()
        
    return df_clients, df_merged

df_clients, df_merged = load_data()

# Header Section
st.markdown("""
<div class='header-banner'>
    <div class='header-title'>🏢 Parcl Real Estate Market Intelligence</div>
    <div class='header-subtitle'>Machine Learning Based Buyer Segmentation & Investment Profiling Dashboard</div>
</div>
""", unsafe_allow_html=True)

# Sidebar Filters
st.sidebar.header("🔍 Filter Options")

all_countries = ['All'] + sorted(list(df_clients['country'].unique()))
selected_country = st.sidebar.selectbox("Country of Residence", all_countries)

all_regions = ['All'] + sorted(list(df_clients['region'].unique()))
selected_region = st.sidebar.selectbox("Geographic Region", all_regions)

all_purposes = ['All'] + sorted(list(df_clients['acquisition_purpose'].unique()))
selected_purpose = st.sidebar.selectbox("Acquisition Purpose", all_purposes)

all_types = ['All'] + sorted(list(df_clients['client_type'].unique()))
selected_client_type = st.sidebar.selectbox("Client Type", all_types)

# Filter Dataset
df_filtered = df_clients.copy()
if selected_country != 'All':
    df_filtered = df_filtered[df_filtered['country'] == selected_country]
if selected_region != 'All':
    df_filtered = df_filtered[df_filtered['region'] == selected_region]
if selected_purpose != 'All':
    df_filtered = df_filtered[df_filtered['acquisition_purpose'] == selected_purpose]
if selected_client_type != 'All':
    df_filtered = df_filtered[df_filtered['client_type'] == selected_client_type]

# KPI Top Bar
kpi_col1, kpi_col2, kpi_col3, kpi_col4, kpi_col5 = st.columns(5)

inv_pct = (df_filtered['acquisition_purpose'] == 'Investment').mean() * 100 if len(df_filtered) > 0 else 0
loan_pct = (df_filtered['loan_applied'] == 'Yes').mean() * 100 if len(df_filtered) > 0 else 0
avg_sat = df_filtered['satisfaction_score'].mean() if len(df_filtered) > 0 else 0

total_props = len(df_merged)
avg_price = df_merged['price_clean'].mean() if 'price_clean' in df_merged.columns else 344375

with kpi_col1:
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-label'>Total Clients</div>
        <div class='metric-value'>{len(df_filtered):,}</div>
        <div class='metric-sub'>Active Profiles</div>
    </div>
    """, unsafe_allow_html=True)

with kpi_col2:
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-label'>Total Listings</div>
        <div class='metric-value'>{total_props:,}</div>
        <div class='metric-sub'>Property Units</div>
    </div>
    """, unsafe_allow_html=True)

with kpi_col3:
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-label'>Investment Intent</div>
        <div class='metric-value'>{inv_pct:.1f}%</div>
        <div class='metric-sub'>Capital Allocators</div>
    </div>
    """, unsafe_allow_html=True)

with kpi_col4:
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-label'>Loan Dependency</div>
        <div class='metric-value'>{loan_pct:.1f}%</div>
        <div class='metric-sub'>Mortgage Applicants</div>
    </div>
    """, unsafe_allow_html=True)

with kpi_col5:
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-label'>Avg Satisfaction</div>
        <div class='metric-value'>{avg_sat:.2f}</div>
        <div class='metric-sub'>Out of 5.0 Rating</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Main Navigation Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Buyer Segmentation",
    "💼 Investor Behavior",
    "🏢 Property Portfolio",
    "🌐 Geographic Buyer Origin",
    "🔍 Strategic Insights"
])

# Color Palette derived from ui_ref.jpeg:
# Primary: #2C005A (Deep Purple), #008C8D (Dark Teal), #820C6D (Magenta), #FF758F (Rose Pink)
palette_map = {
    'C1: Global Investors': '#2C005A',
    'C2: First-Time Buyers': '#008C8D',
    'C3: Corporate Buyers': '#820C6D',
    'C4: Luxury Investors': '#FF758F'
}
palette_seq = ['#2C005A', '#008C8D', '#820C6D', '#FF758F']

# ---------------------------------------------------------
# MODULE 1: BUYER SEGMENTATION OVERVIEW
# ---------------------------------------------------------
with tab1:
    st.subheader("Buyer Segment Distribution & Clustering Breakdown")
    st.write("Proportion of clients classified across the 4 ML-derived buyer personas.")
    
    if len(df_filtered) == 0:
        st.warning("No records match the current filter selection.")
    else:
        m1_col1, m1_col2 = st.columns(2)
        segment_counts = df_filtered['buyer_persona'].value_counts().reset_index()
        segment_counts.columns = ['Persona', 'Count']
        
        with m1_col1:
            fig_pie = px.pie(
                segment_counts,
                values='Count',
                names='Persona',
                title="Proportion of Buyer Personas",
                color='Persona',
                color_discrete_map=palette_map,
                hole=0.45
            )
            fig_pie.update_traces(textinfo='percent+label', marker=dict(line=dict(color='#FFFFFF', width=2)))
            fig_pie.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#2C005A'))
            st.plotly_chart(fig_pie, use_container_width=True)
            
        with m1_col2:
            fig_bar = px.bar(
                segment_counts,
                x='Persona',
                y='Count',
                color='Persona',
                title="Client Count by Persona Segment",
                text='Count',
                color_discrete_map=palette_map
            )
            fig_bar.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis_title="",
                yaxis_title="Number of Clients",
                showlegend=False,
                font=dict(color='#2C005A')
            )
            st.plotly_chart(fig_bar, use_container_width=True)

# ---------------------------------------------------------
# MODULE 2: INVESTOR BEHAVIOR DASHBOARD
# ---------------------------------------------------------
with tab2:
    st.subheader("Financial & Behavioral Patterns by Persona")
    st.write("Examines debt dependency (loan reliance) and purchasing motivations across client cohorts.")
    
    if len(df_filtered) == 0:
        st.warning("No records match the current filter selection.")
    else:
        m2_col1, m2_col2 = st.columns(2)
        
        with m2_col1:
            loan_df = df_filtered.groupby(['buyer_persona', 'loan_applied']).size().reset_index(name='count')
            fig_loan = px.bar(
                loan_df,
                x='buyer_persona',
                y='count',
                color='loan_applied',
                barmode='group',
                title="Mortgage Loan Reliance (Loan Applied)",
                color_discrete_map={'Yes': '#820C6D', 'No': '#008C8D'}
            )
            fig_loan.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#2C005A'))
            st.plotly_chart(fig_loan, use_container_width=True)
            
        with m2_col2:
            purpose_df = df_filtered.groupby(['buyer_persona', 'acquisition_purpose']).size().reset_index(name='count')
            fig_purpose = px.bar(
                purpose_df,
                x='buyer_persona',
                y='count',
                color='acquisition_purpose',
                barmode='stack',
                title="Acquisition Purpose Intent Breakdown",
                color_discrete_map={'Investment': '#2C005A', 'Personal use': '#FF758F'}
            )
            fig_purpose.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#2C005A'))
            st.plotly_chart(fig_purpose, use_container_width=True)

# ---------------------------------------------------------
# MODULE 3: PROPERTY PORTFOLIO ANALYTICS
# ---------------------------------------------------------
with tab3:
    st.subheader("Property Transaction & Valuation Analytics")
    st.write("Investment capital flow, unit categories, and price distributions across property listings.")
    
    if len(df_merged) == 0:
        st.warning("No property listings available.")
    else:
        m3_col1, m3_col2 = st.columns(2)
        
        with m3_col1:
            if 'price_clean' in df_merged.columns and 'unit_category' in df_merged.columns:
                fig_box = px.box(
                    df_merged,
                    x='unit_category',
                    y='price_clean',
                    color='unit_category',
                    title="Property Sale Price Range by Unit Category",
                    color_discrete_map={'Apartment': '#2C005A', 'Office': '#008C8D'}
                )
                fig_box.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', yaxis_title="Price ($)")
                st.plotly_chart(fig_box, use_container_width=True)
                
        with m3_col2:
            if 'buyer_persona' in df_merged.columns and 'price_clean' in df_merged.columns:
                vol_df = df_merged.groupby('buyer_persona')['price_clean'].sum().reset_index()
                vol_df['price_millions'] = vol_df['price_clean'] / 1e6
                fig_vol = px.bar(
                    vol_df,
                    x='buyer_persona',
                    y='price_millions',
                    color='buyer_persona',
                    title="Total Investment Volume ($ Millions) by Persona",
                    text='price_millions',
                    color_discrete_map=palette_map
                )
                fig_vol.update_traces(texttemplate='$%{text:.1f}M', textposition='outside')
                fig_vol.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', showlegend=False)
                st.plotly_chart(fig_vol, use_container_width=True)

# ---------------------------------------------------------
# MODULE 4: GEOGRAPHIC BUYER ORIGIN
# ---------------------------------------------------------
with tab4:
    st.subheader("Regional Capital Distribution & Geographic Origin")
    st.write("Spatial tracking of domestic vs. cross-border real estate investor activity.")
    
    if len(df_filtered) == 0:
        st.warning("No records match the current filter selection.")
    else:
        m4_col1, m4_col2 = st.columns(2)
        
        with m4_col1:
            geo_region = df_filtered.groupby(['region', 'buyer_persona']).size().reset_index(name='count')
            fig_geo = px.bar(
                geo_region,
                x='region',
                y='count',
                color='buyer_persona',
                title="Regional Buyer Concentration by Persona",
                color_discrete_map=palette_map
            )
            fig_geo.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_geo, use_container_width=True)
            
        with m4_col2:
            country_df = df_filtered['country'].value_counts().reset_index()
            country_df.columns = ['Country', 'Client Count']
            fig_country = px.choropleth(
                country_df,
                locations='Country',
                locationmode='country names',
                color='Client Count',
                title="Global Buyer Origin Density Map",
                color_continuous_scale=['#FDF0F3', '#820C6D', '#2C005A']
            )
            fig_country.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_country, use_container_width=True)

# ---------------------------------------------------------
# MODULE 5: STRATEGIC INSIGHTS PANEL
# ---------------------------------------------------------
with tab5:
    st.subheader("Descriptive Metrics & Strategic Recommendations")
    st.write("Statistical profiling and targeted business strategies for Parcl platform decision-makers.")
    
    if len(df_filtered) == 0:
        st.warning("No records match the current filter selection.")
    else:
        stats_df = df_filtered.groupby('buyer_persona').agg(
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
            <div class='content-card'>
                <h4 style='color: #2C005A; margin-top:0;'>C1: Global Investors</h4>
                <ul>
                    <li><b>Strategy</b>: Cross-border yield syndication, tax-optimized currency escrow, high-cap commercial listings.</li>
                    <li><b>Targeting</b>: International digital marketing in primary global financial hubs.</li>
                </ul>
                <h4 style='color: #008C8D; margin-top:16px;'>C2: First-Time Buyers</h4>
                <ul>
                    <li><b>Strategy</b>: Embedded mortgage comparison tools, introductory down-payment assistance calculators.</li>
                    <li><b>Targeting</b>: Social media referral programs and localized housing affordability guides.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
        with rec_col2:
            st.markdown("""
            <div class='content-card'>
                <h4 style='color: #820C6D; margin-top:0;'>C3: Corporate Buyers</h4>
                <ul>
                    <li><b>Strategy</b>: Institutional API access, bulk multi-unit acquisition discounts, custom compliance reporting.</li>
                    <li><b>Targeting</b>: Direct B2B enterprise sales and institutional real estate advisory.</li>
                </ul>
                <h4 style='color: #FF758F; margin-top:16px;'>C4: Luxury Investors</h4>
                <ul>
                    <li><b>Strategy</b>: White-glove private wealth concierge, off-market premier listings, priority investor support.</li>
                    <li><b>Targeting</b>: Exclusive investor summits and private wealth manager partnerships.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

st.sidebar.divider()
st.sidebar.caption("Parcl Real Estate ML Engine v1.0 | Unified Mentor Capstone")
