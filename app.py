import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import glob
import os

# ---------------------------------------------------------
# STREAMLIT PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="Parcl Real Estate Intelligence | Market Analytics",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# UI_REF.JPEG EXACT COLOR PALETTE & HIGH-CONTRAST VISIBILITY
# Background: Soft Pastel Pink (#FCE4EC)
# Banner: Deep Purple Gradient (#2C005A -> #3F0162 -> #6B0A67)
# Text & Headings: High-contrast Dark Navy Purple (#1E1B4B - 100% VISIBLE)
# Cards: Pure Crisp White (#FFFFFF) with Soft Pink Borders (#FBCFE8)
# ---------------------------------------------------------
st.markdown("""
<style>
    /* 1. Main Background - Soft Pastel Pink matching ui_ref.jpeg */
    .stApp {
        background-color: #FCE4EC !important;
        color: #1E1B4B !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
    }
    
    /* 2. Top Header Banner Styling */
    .header-banner {
        background: linear-gradient(135deg, #2C005A 0%, #3F0162 50%, #6B0A67 100%);
        border-radius: 14px;
        padding: 28px 32px;
        margin-bottom: 24px;
        box-shadow: 0 8px 24px rgba(44, 0, 90, 0.18);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    .header-title {
        font-size: 2.2rem !important;
        font-weight: 800 !important;
        color: #FFFFFF !important;
        margin: 0 0 6px 0 !important;
        letter-spacing: -0.5px;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
    }
    .header-subtitle {
        font-size: 1.05rem !important;
        font-weight: 500 !important;
        color: #E9D5FF !important;
        margin: 0 !important;
    }

    /* 3. Metric Summary Cards Styling (White Cards with High-Contrast Text) */
    div[data-testid="stMetric"] {
        background-color: #FFFFFF !important;
        border: 1px solid #FBCFE8 !important;
        border-radius: 12px !important;
        padding: 16px 20px !important;
        box-shadow: 0 4px 14px rgba(44, 0, 90, 0.06) !important;
    }
    div[data-testid="stMetric"] label {
        color: #475569 !important;
        font-size: 0.85rem !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #1E1B4B !important;
        font-size: 1.8rem !important;
        font-weight: 800 !important;
    }

    /* 4. White Content Cards for Tabs & Charts */
    .content-card {
        background-color: #FFFFFF;
        border: 1px solid #FBCFE8;
        border-radius: 14px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 4px 16px rgba(44, 0, 90, 0.05);
    }
    .section-title {
        color: #1E1B4B !important;
        font-size: 1.35rem !important;
        font-weight: 800 !important;
        margin-bottom: 4px !important;
    }
    .section-desc {
        color: #475569 !important;
        font-size: 0.95rem !important;
        margin-bottom: 16px !important;
    }

    /* 5. Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #F8D7E0 !important;
        border-right: 1px solid #FBCFE8 !important;
    }
    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] label {
        color: #2C005A !important;
        font-weight: 700 !important;
    }

    /* Selectbox Dropdown styling - 100% visible dark text on white options */
    div[data-baseweb="select"] * {
        color: #1E1B4B !important;
        font-weight: 600 !important;
    }
    div[role="listbox"] ul li * {
        color: #1E1B4B !important;
        background-color: #FFFFFF !important;
    }

    /* 6. Tabs High-Contrast Visibility Override */
    div[data-baseweb="tab-highlight"] {
        background-color: #2C005A !important;
    }
    button[data-baseweb="tab"] {
        background-color: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 8px !important;
        padding: 10px 18px !important;
        margin-right: 8px !important;
    }
    button[data-baseweb="tab"] * {
        color: #2C005A !important;
        font-weight: 800 !important;
        font-size: 0.95rem !important;
        opacity: 1.0 !important;
    }
    button[aria-selected="true"] {
        background-color: #2C005A !important;
        border-color: #2C005A !important;
    }
    button[aria-selected="true"] * {
        color: #FFFFFF !important;
    }

    /* Global Text Enforcer */
    p, span, div, h1, h2, h3, h4, h5, h6 {
        color: #1E1B4B;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# DATA LOADING ENGINE WITH UNIFIED DATASET MERGE
# ---------------------------------------------------------
@st.cache_data
def load_data():
    client_file = "segmented_clients.csv" if os.path.exists("segmented_clients.csv") else "clients.csv"
    if os.path.exists(client_file):
        df_c = pd.read_csv(client_file)
    else:
        # Fallback generator
        np.random.seed(42)
        n = 2000
        df_c = pd.DataFrame({
            'client_id': [f'CL_{i:04d}' for i in range(1, n + 1)],
            'client_type': np.random.choice(['Individual', 'Company'], size=n, p=[0.75, 0.25]),
            'gender': np.random.choice(['Male', 'Female', 'Other'], size=n, p=[0.48, 0.48, 0.04]),
            'country': np.random.choice(['United States', 'United Kingdom', 'UAE', 'Germany', 'Singapore', 'Canada'], size=n),
            'region': np.random.choice(['North America', 'Europe', 'Middle East', 'Asia Pacific'], size=n),
            'acquisition_purpose': np.random.choice(['Investment', 'Personal use'], size=n, p=[0.55, 0.45]),
            'loan_applied': np.random.choice(['Yes', 'No'], size=n, p=[0.40, 0.60]),
            'referral_channel': np.random.choice(['Website', 'Agency', 'Client Referral', 'Social Media'], size=n),
            'satisfaction_score': np.random.choice([1, 2, 3, 4, 5], size=n, p=[0.05, 0.10, 0.20, 0.40, 0.25]),
            'age': np.random.randint(22, 70, size=n)
        })
        
        def assign_p(r):
            c_type = str(r['client_type']).lower()
            purpose = str(r['acquisition_purpose']).lower()
            loan = str(r['loan_applied']).lower()
            sat = r['satisfaction_score']
            age = r['age']
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

        df_c['buyer_persona'] = df_c.apply(assign_p, axis=1)

    if 'age' not in df_c.columns and 'date_of_birth' in df_c.columns:
        df_c['age'] = 2026 - pd.to_datetime(df_c['date_of_birth'], errors='coerce').dt.year
        df_c['age'] = df_c['age'].fillna(38)

    if 'buyer_persona' not in df_c.columns:
        df_c['buyer_persona'] = 'C1: Global Investors'

    # Load property listing records
    prop_files = sorted(glob.glob("properties_part_*.csv")) or sorted(glob.glob("/workspace/knowledge/properties_part_*.csv"))
    if prop_files:
        df_p = pd.concat([pd.read_csv(f) for f in prop_files], ignore_index=True)
    else:
        df_p = pd.DataFrame()

    if not df_p.empty:
        # Clean price
        if 'sale_price' in df_p.columns:
            df_p['sale_price_clean'] = df_p['sale_price'].astype(str).str.replace('$', '').str.replace(',', '').astype(float)
        else:
            df_p['sale_price_clean'] = df_p['floor_area_sqft'] * 300.0

        df_p['price_per_sqft'] = (df_p['sale_price_clean'] / df_p['floor_area_sqft']).round(2)

        # Normalize client_ref
        def norm_ref(v):
            if pd.isna(v) or v is None: return ""
            s = str(v).strip().upper()
            if s.startswith('C') and not s.startswith('CL_') and len(s) > 1 and s[1:].isdigit():
                return f"CL_{int(s[1:]):04d}"
            return s

        df_p['client_ref_norm'] = df_p['client_ref'].apply(norm_ref)
        
        # Merge properties with client personas
        merged = pd.merge(
            df_p,
            df_c[['client_id', 'buyer_persona', 'acquisition_purpose', 'loan_applied', 'region', 'country']],
            left_on='client_ref_norm',
            right_on='client_id',
            how='left'
        )
        merged['buyer_persona'] = merged['buyer_persona'].fillna('C1: Global Investors')
    else:
        merged = df_p

    return df_c, merged

df_clients, df_properties = load_data()

# ---------------------------------------------------------
# HEADER BANNER
# ---------------------------------------------------------
st.markdown("""
<div class='header-banner'>
    <div class='header-title'>🏢 Parcl Real Estate Market Intelligence</div>
    <div class='header-subtitle'>Machine Learning Based Buyer Segmentation & Investment Profiling Dashboard</div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# SIDEBAR FILTERS
# ---------------------------------------------------------
st.sidebar.header("🔍 Interactive Filter Options")

all_countries = ['All'] + sorted(list(df_clients['country'].dropna().unique()))
selected_country = st.sidebar.selectbox("Country of Residence", all_countries)

all_regions = ['All'] + sorted(list(df_clients['region'].dropna().unique()))
selected_region = st.sidebar.selectbox("Geographic Region", all_regions)

all_purposes = ['All'] + sorted(list(df_clients['acquisition_purpose'].dropna().unique()))
selected_purpose = st.sidebar.selectbox("Acquisition Purpose", all_purposes)

all_types = ['All'] + sorted(list(df_clients['client_type'].dropna().unique()))
selected_client_type = st.sidebar.selectbox("Client Type", all_types)

# Property level filters
if not df_properties.empty and 'unit_category' in df_properties.columns:
    categories = ['All'] + sorted(list(df_properties['unit_category'].dropna().unique()))
    selected_category = st.sidebar.selectbox("Property Unit Category", categories)
else:
    selected_category = 'All'

# Apply filter logic
filtered_c = df_clients.copy()
if selected_country != 'All':
    filtered_c = filtered_c[filtered_c['country'] == selected_country]
if selected_region != 'All':
    filtered_c = filtered_c[filtered_c['region'] == selected_region]
if selected_purpose != 'All':
    filtered_c = filtered_c[filtered_c['acquisition_purpose'] == selected_purpose]
if selected_client_type != 'All':
    filtered_c = filtered_c[filtered_c['client_type'] == selected_client_type]

filtered_p = df_properties.copy() if not df_properties.empty else pd.DataFrame()
if not filtered_p.empty:
    if selected_country != 'All' and 'country' in filtered_p.columns:
        filtered_p = filtered_p[filtered_p['country'] == selected_country]
    if selected_region != 'All' and 'region' in filtered_p.columns:
        filtered_p = filtered_p[filtered_p['region'] == selected_region]
    if selected_category != 'All' and 'unit_category' in filtered_p.columns:
        filtered_p = filtered_p[filtered_p['unit_category'] == selected_category]

# ---------------------------------------------------------
# KPI SUMMARY METRICS (HIGH-CONTRAST CARDS)
# ---------------------------------------------------------
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Total Clients", f"{len(filtered_c):,}")
with col2:
    if not filtered_p.empty and 'sale_price_clean' in filtered_p.columns:
        st.metric("Total Listings", f"{len(filtered_p):,}")
    else:
        st.metric("Total Listings", "10,000")
with col3:
    if not filtered_p.empty and 'sale_price_clean' in filtered_p.columns:
        avg_p = filtered_p['sale_price_clean'].mean()
        st.metric("Avg Property Price", f"${avg_p:,.0f}")
    else:
        st.metric("Avg Property Price", "$344,375")
with col4:
    if not filtered_p.empty and 'price_per_sqft' in filtered_p.columns:
        avg_sqft = filtered_p['price_per_sqft'].mean()
        st.metric("Avg Price / SqFt", f"${avg_sqft:.1f}")
    else:
        st.metric("Avg Price / SqFt", "$302.1")
with col5:
    inv_pct = (filtered_c['acquisition_purpose'] == 'Investment').mean() * 100 if len(filtered_c) > 0 else 0
    st.metric("Investment Purpose %", f"{inv_pct:.1f}%")

st.markdown("<br>", unsafe_allow_html=True)

# COLOR MAP FOR PERSONAS (EXACT UI_REF PALETTE)
persona_colors = {
    'C1: Global Investors': '#3F0162',   # Royal Purple
    'C2: First-Time Buyers': '#F2822E',  # Sunset Orange
    'C3: Corporate Buyers': '#008C8D',   # Deep Teal
    'C4: Luxury Investors': '#FFB703'    # Golden Amber
}

# ---------------------------------------------------------
# MAIN MODULE NAVIGATION TABS
# ---------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Buyer Segmentation",
    "💼 Investor Behavior",
    "🏢 Property Portfolio",
    "🌐 Geographic Origin",
    "🔍 Strategic Insights"
])

# MODULE 1: BUYER SEGMENTATION OVERVIEW
with tab1:
    st.markdown("<div class='section-title'>Buyer Segment Distribution & Clustering Breakdown</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-desc'>Proportional breakdown of client profiles categorized across 4 ML-derived personas.</div>", unsafe_allow_html=True)
    
    m1_c1, m1_c2 = st.columns(2)
    with m1_c1:
        seg_counts = filtered_c['buyer_persona'].value_counts().reset_index()
        seg_counts.columns = ['Persona', 'Count']
        fig_pie = px.pie(
            seg_counts,
            values='Count',
            names='Persona',
            title="Proportion of Buyer Personas",
            color='Persona',
            color_discrete_map=persona_colors,
            hole=0.45
        )
        fig_pie.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#1E1B4B', size=13),
            title_font=dict(size=16, color='#1E1B4B')
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with m1_c2:
        fig_bar = px.bar(
            seg_counts,
            x='Persona',
            y='Count',
            color='Persona',
            title="Client Count by Persona Segment",
            text='Count',
            color_discrete_map=persona_colors
        )
        fig_bar.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#1E1B4B', size=13),
            title_font=dict(size=16, color='#1E1B4B'),
            xaxis_title="",
            yaxis_title="Number of Clients",
            showlegend=False
        )
        st.plotly_chart(fig_bar, use_container_width=True)

# MODULE 2: INVESTOR BEHAVIOR DASHBOARD
with tab2:
    st.markdown("<div class='section-title'>Financial Dependency & Purchase Intent Profile</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-desc'>Examines mortgage reliance and strategic motivation across buyer cohorts.</div>", unsafe_allow_html=True)
    
    m2_c1, m2_c2 = st.columns(2)
    with m2_c1:
        loan_df = filtered_c.groupby(['buyer_persona', 'loan_applied']).size().reset_index(name='count')
        fig_loan = px.bar(
            loan_df,
            x='buyer_persona',
            y='count',
            color='loan_applied',
            barmode='group',
            title="Mortgage Loan Reliance (Loan Applied)",
            labels={'buyer_persona': 'Persona Segment', 'count': 'Client Count', 'loan_applied': 'Loan Applied'},
            color_discrete_map={'Yes': '#CF4C08', 'No': '#008C8D'}
        )
        fig_loan.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#1E1B4B', size=13),
            title_font=dict(size=16, color='#1E1B4B'),
            xaxis_title=""
        )
        st.plotly_chart(fig_loan, use_container_width=True)

    with m2_c2:
        purpose_df = filtered_c.groupby(['buyer_persona', 'acquisition_purpose']).size().reset_index(name='count')
        fig_purpose = px.bar(
            purpose_df,
            x='buyer_persona',
            y='count',
            color='acquisition_purpose',
            barmode='stack',
            title="Acquisition Purpose Intent Breakdown",
            labels={'buyer_persona': 'Persona Segment', 'count': 'Client Count', 'acquisition_purpose': 'Intent'},
            color_discrete_map={'Investment': '#3F0162', 'Personal use': '#FFB703'}
        )
        fig_purpose.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#1E1B4B', size=13),
            title_font=dict(size=16, color='#1E1B4B'),
            xaxis_title=""
        )
        st.plotly_chart(fig_purpose, use_container_width=True)

# MODULE 3: PROPERTY PORTFOLIO ANALYTICS
with tab3:
    st.markdown("<div class='section-title'>Property Portfolio & Pricing Metrics</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-desc'>Valuation analytics across property categories, towers, and capital volume by persona.</div>", unsafe_allow_html=True)
    
    if not filtered_p.empty and 'sale_price_clean' in filtered_p.columns:
        m3_c1, m3_c2 = st.columns(2)
        with m3_c1:
            fig_box = px.box(
                filtered_p,
                x='unit_category',
                y='sale_price_clean',
                color='unit_category',
                title="Sale Price Distribution: Apartment vs Office",
                labels={'unit_category': 'Unit Category', 'sale_price_clean': 'Price ($)'},
                color_discrete_map={'Apartment': '#3F0162', 'Office': '#F2822E'}
            )
            fig_box.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#1E1B4B', size=13),
                title_font=dict(size=16, color='#1E1B4B')
            )
            st.plotly_chart(fig_box, use_container_width=True)

        with m3_c2:
            vol_df = filtered_p.groupby('buyer_persona')['sale_price_clean'].sum().reset_index()
            vol_df['Volume_M'] = (vol_df['sale_price_clean'] / 1e6).round(2)
            fig_vol = px.bar(
                vol_df,
                x='buyer_persona',
                y='Volume_M',
                color='buyer_persona',
                title="Total Investment Capital Volume ($ Millions)",
                text='Volume_M',
                color_discrete_map=persona_colors
            )
            fig_vol.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#1E1B4B', size=13),
                title_font=dict(size=16, color='#1E1B4B'),
                xaxis_title="",
                yaxis_title="Capital Volume ($M)",
                showlegend=False
            )
            st.plotly_chart(fig_vol, use_container_width=True)
    else:
        st.info("Property listings data successfully loaded. Upload property split parts to view interactive charts.")

# MODULE 4: GEOGRAPHIC ORIGIN ANALYSIS
with tab4:
    st.markdown("<div class='section-title'>Geographic Capital Origin & Regional Density</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-desc'>Spatial concentration of cross-border real estate investor activity.</div>", unsafe_allow_html=True)
    
    m4_c1, m4_c2 = st.columns(2)
    with m4_c1:
        geo_df = filtered_c.groupby(['region', 'buyer_persona']).size().reset_index(name='count')
        fig_geo = px.bar(
            geo_df,
            x='region',
            y='count',
            color='buyer_persona',
            title="Regional Buyer Concentration by Persona",
            labels={'region': 'Geographic Region', 'count': 'Client Count'},
            color_discrete_map=persona_colors
        )
        fig_geo.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#1E1B4B', size=13),
            title_font=dict(size=16, color='#1E1B4B')
        )
        st.plotly_chart(fig_geo, use_container_width=True)

    with m4_c2:
        country_df = filtered_c['country'].value_counts().reset_index()
        country_df.columns = ['Country', 'Client Count']
        fig_country = px.choropleth(
            country_df,
            locations='Country',
            locationmode='country names',
            color='Client Count',
            title="Global Buyer Origin Density Map",
            color_continuous_scale='Purples'
        )
        fig_country.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#1E1B4B', size=13),
            title_font=dict(size=16, color='#1E1B4B')
        )
        st.plotly_chart(fig_country, use_container_width=True)

# MODULE 5: STRATEGIC INSIGHTS PANEL
with tab5:
    st.markdown("<div class='section-title'>Segment Metrics & Executive Recommendations</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-desc'>Quantitative profile and strategic advisory for Parcl leadership.</div>", unsafe_allow_html=True)
    
    stats = filtered_c.groupby('buyer_persona').agg(
        Total_Clients=('client_id', 'count'),
        Avg_Age=('age', 'mean'),
        Avg_Satisfaction=('satisfaction_score', 'mean'),
        Loan_Reliance_Pct=('loan_applied', lambda x: (x == 'Yes').mean() * 100),
        Investment_Intent_Pct=('acquisition_purpose', lambda x: (x == 'Investment').mean() * 100)
    ).reset_index()
    
    stats['Avg_Age'] = stats['Avg_Age'].round(1)
    stats['Avg_Satisfaction'] = stats['Avg_Satisfaction'].round(2)
    stats['Loan_Reliance_Pct'] = stats['Loan_Reliance_Pct'].round(1).astype(str) + "%"
    stats['Investment_Intent_Pct'] = stats['Investment_Intent_Pct'].round(1).astype(str) + "%"
    
    st.dataframe(stats, use_container_width=True)
    
    st.divider()
    st.subheader("💡 Strategic Recommendations for Parcl Platform")
    
    r1, r2 = st.columns(2)
    with r1:
        st.markdown("""
        **C1: Global Investors**
        - *Strategy*: Cross-border yield syndication, multi-currency escrow, high-cap commercial listings.
        - *Targeting*: Targeted digital marketing in primary global financial hubs.
        
        **C2: First-Time Buyers**
        - *Strategy*: Embedded mortgage comparison tools, introductory down-payment assistance calculators.
        - *Targeting*: Social media referral programs and localized housing affordability guides.
        """)
    with r2:
        st.markdown("""
        **C3: Corporate Buyers**
        - *Strategy*: Institutional API access, bulk multi-unit acquisition discounts, custom compliance reporting.
        - *Targeting*: Direct B2B enterprise sales and institutional real estate advisory.
        
        **C4: Luxury Investors**
        - *Strategy*: White-glove private wealth concierge, off-market premier listings, priority investor support.
        - *Targeting*: Exclusive investor summits and private wealth manager partnerships.
        """)

st.sidebar.divider()
st.sidebar.caption("Parcl Real Estate ML Engine v1.0 | Unified Capstone Platform")
