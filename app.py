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

# Custom High-Contrast CSS Styling (Clean & Accessible - No Hacky Overrides)
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #861645;
        margin-bottom: 0.1rem;
    }
    .sub-title {
        font-size: 1.05rem;
        font-weight: 500;
        color: #354A57;
        margin-bottom: 1.2rem;
    }
    div[data-testid="stMetricValue"] {
        font-weight: 700;
        color: #861645;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# DATA LOADING & MERGING PIPELINE
# ---------------------------------------------------------
@st.cache_data
def load_data():
    # 1. Load Clients / Segmented Clients
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
        
    if 'age' not in df_clients.columns and 'date_of_birth' in df_clients.columns:
        df_clients['age'] = 2026 - pd.to_datetime(df_clients['date_of_birth'], errors='coerce').dt.year
        df_clients['age'] = df_clients['age'].fillna(35)
        
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

    # 2. Load Properties Dataset (Parts 1-5 or single properties.csv)
    prop_files = sorted(glob.glob("properties_part_*.csv"))
    if prop_files:
        df_props = pd.concat([pd.read_csv(f) for f in prop_files], ignore_index=True)
    elif os.path.exists("properties.csv"):
        df_props = pd.read_csv("properties.csv")
    else:
        df_props = pd.DataFrame()

    if not df_props.empty:
        # Clean Sale Price
        if 'sale_price' in df_props.columns:
            df_props['sale_price_clean'] = df_props['sale_price'].astype(str).str.replace('$', '', regex=False).str.replace(',', '', regex=False).astype(float)
        else:
            df_props['sale_price_clean'] = df_props['floor_area_sqft'] * 300.0

        # Calculate Price / sqft
        if 'floor_area_sqft' in df_props.columns and 'sale_price_clean' in df_props.columns:
            df_props['price_per_sqft'] = (df_props['sale_price_clean'] / df_props['floor_area_sqft']).round(2)

        # Normalize client_ref key for merging
        if 'client_ref' in df_props.columns:
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
            df_props['client_ref_norm'] = df_props['client_ref'].apply(norm_ref)

            merged = pd.merge(
                df_props,
                df_clients[['client_id', 'buyer_persona', 'acquisition_purpose', 'loan_applied', 'country', 'region']],
                left_on='client_ref_norm',
                right_on='client_id',
                how='left'
            )
            merged['buyer_persona'] = merged['buyer_persona'].fillna('C1: Global Investors')
        else:
            merged = df_props.copy()
            merged['buyer_persona'] = 'C1: Global Investors'
    else:
        merged = pd.DataFrame()

    return df_clients, merged

df_clients, df_props = load_data()

# Palette Definition from cp.jpeg
COLOR_MAP = {
    'C1: Global Investors': '#861645',   # Rich Magenta Purple
    'C2: First-Time Buyers': '#F2822E',  # Sunset Orange
    'C3: Corporate Buyers': '#354A57',   # Slate Teal
    'C4: Luxury Investors': '#85A611'    # Olive Lime
}
COLOR_SEQ = ['#861645', '#F2822E', '#354A57', '#85A611', '#9B3245']

# ---------------------------------------------------------
# HEADER & SIDEBAR FILTERS
# ---------------------------------------------------------
st.markdown("<div class='main-title'>🏢 Parcl Real Estate Market Intelligence</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Machine Learning Buyer Segmentation & Property Portfolio Analytics</div>", unsafe_allow_html=True)
st.divider()

st.sidebar.header("🔍 Filter Options")

all_countries = ['All'] + sorted(list(df_clients['country'].dropna().unique()))
selected_country = st.sidebar.selectbox("Country of Residence", all_countries)

all_regions = ['All'] + sorted(list(df_clients['region'].dropna().unique()))
selected_region = st.sidebar.selectbox("Geographic Region", all_regions)

all_purposes = ['All'] + sorted(list(df_clients['acquisition_purpose'].dropna().unique()))
selected_purpose = st.sidebar.selectbox("Acquisition Purpose", all_purposes)

all_types = ['All'] + sorted(list(df_clients['client_type'].dropna().unique()))
selected_client_type = st.sidebar.selectbox("Client Type", all_types)

# Filter Datasets
df_c_filtered = df_clients.copy()
if selected_country != 'All':
    df_c_filtered = df_c_filtered[df_c_filtered['country'] == selected_country]
if selected_region != 'All':
    df_c_filtered = df_c_filtered[df_c_filtered['region'] == selected_region]
if selected_purpose != 'All':
    df_c_filtered = df_c_filtered[df_c_filtered['acquisition_purpose'] == selected_purpose]
if selected_client_type != 'All':
    df_c_filtered = df_c_filtered[df_c_filtered['client_type'] == selected_client_type]

df_p_filtered = df_props.copy() if not df_props.empty else pd.DataFrame()
if not df_p_filtered.empty and 'country' in df_p_filtered.columns:
    if selected_country != 'All':
        df_p_filtered = df_p_filtered[df_p_filtered['country'] == selected_country]
    if selected_region != 'All':
        df_p_filtered = df_p_filtered[df_p_filtered['region'] == selected_region]

# TOP KPI METRICS
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
with kpi1:
    st.metric("Total Clients", f"{len(df_c_filtered):,}")
with kpi2:
    total_props = len(df_p_filtered) if not df_p_filtered.empty else 0
    st.metric("Total Properties", f"{total_props:,}")
with kpi3:
    avg_price = df_p_filtered['sale_price_clean'].mean() if not df_p_filtered.empty else 0
    st.metric("Avg Sale Price", f"${avg_price:,.0f}")
with kpi4:
    avg_psqft = df_p_filtered['price_per_sqft'].mean() if not df_p_filtered.empty else 0
    st.metric("Avg Price / SqFt", f"${avg_psqft:.1f}")
with kpi5:
    sold_ratio = (df_p_filtered['listing_status'] == 'Sold').mean() * 100 if not df_p_filtered.empty and 'listing_status' in df_p_filtered.columns else 0
    st.metric("Sold Inventory %", f"{sold_ratio:.1f}%")

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------
# MAIN NAVIGATION TABS
# ---------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Buyer Segmentation",
    "💼 Investor Behavior",
    "🏢 Property Portfolio",
    "🌐 Geographic Origin",
    "🔍 Strategic Insights"
])

# MODULE 1: BUYER SEGMENTATION
with tab1:
    st.subheader("Buyer Segment Distribution & Clustering Breakdown")
    st.write("Overview of client proportions classified across the 4 ML-derived buyer personas.")
    
    if len(df_c_filtered) == 0:
        st.warning("No client records match the current sidebar filter criteria.")
    else:
        col_a, col_b = st.columns(2)
        with col_a:
            seg_counts = df_c_filtered['buyer_persona'].value_counts().reset_index()
            seg_counts.columns = ['Persona', 'Count']
            
            fig_pie = px.pie(
                seg_counts, 
                values='Count', 
                names='Persona',
                title="Proportion of Buyer Personas",
                color='Persona',
                color_discrete_map=COLOR_MAP,
                hole=0.4
            )
            fig_pie.update_traces(textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)
            
        with col_b:
            fig_bar = px.bar(
                seg_counts,
                x='Persona',
                y='Count',
                color='Persona',
                title="Client Count by Persona Segment",
                text='Count',
                color_discrete_map=COLOR_MAP
            )
            fig_bar.update_layout(xaxis_title="", yaxis_title="Number of Clients", showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)

# MODULE 2: INVESTOR BEHAVIOR
with tab2:
    st.subheader("Financial & Behavioral Patterns by Persona")
    st.write("Examines debt dependency (loan reliance) and purchasing motivations across client cohorts.")
    
    if len(df_c_filtered) == 0:
        st.warning("No client records match the current sidebar filter criteria.")
    else:
        col_c, col_d = st.columns(2)
        with col_c:
            loan_df = df_c_filtered.groupby(['buyer_persona', 'loan_applied']).size().reset_index(name='count')
            fig_loan = px.bar(
                loan_df,
                x='buyer_persona',
                y='count',
                color='loan_applied',
                barmode='group',
                title="Mortgage Loan Reliance (Loan Applied)",
                labels={'buyer_persona': 'Persona Segment', 'count': 'Client Count', 'loan_applied': 'Loan Applied'},
                color_discrete_map={'Yes': '#F2822E', 'No': '#354A57'}
            )
            fig_loan.update_layout(xaxis_title="")
            st.plotly_chart(fig_loan, use_container_width=True)
            
        with col_d:
            purpose_df = df_c_filtered.groupby(['buyer_persona', 'acquisition_purpose']).size().reset_index(name='count')
            fig_purpose = px.bar(
                purpose_df,
                x='buyer_persona',
                y='count',
                color='acquisition_purpose',
                barmode='stack',
                title="Acquisition Purpose Intent Breakdown",
                labels={'buyer_persona': 'Persona Segment', 'count': 'Client Count', 'acquisition_purpose': 'Intent'},
                color_discrete_map={'Investment': '#861645', 'Personal use': '#85A611'}
            )
            fig_purpose.update_layout(xaxis_title="")
            st.plotly_chart(fig_purpose, use_container_width=True)

# MODULE 3: PROPERTY PORTFOLIO
with tab3:
    st.subheader("Property Transaction & Portfolio Analytics")
    st.write("Valuation distribution, tower pricing metrics, and sales volume by buyer persona.")
    
    if df_p_filtered.empty:
        st.info("Property listings data is loading or missing.")
    else:
        col_e, col_f = st.columns(2)
        with col_e:
            fig_box = px.box(
                df_p_filtered,
                x='unit_category',
                y='sale_price_clean',
                color='unit_category',
                title="Property Price Distribution by Unit Category",
                labels={'unit_category': 'Unit Category', 'sale_price_clean': 'Sale Price ($)'},
                color_discrete_sequence=['#861645', '#F2822E']
            )
            st.plotly_chart(fig_box, use_container_width=True)
            
        with col_f:
            vol_df = df_p_filtered.groupby('buyer_persona')['sale_price_clean'].sum().reset_index()
            vol_df['volume_millions'] = vol_df['sale_price_clean'] / 1e6
            fig_vol = px.bar(
                vol_df,
                x='buyer_persona',
                y='volume_millions',
                color='buyer_persona',
                title="Total Capital Volume ($ Millions) by Persona",
                labels={'buyer_persona': '', 'volume_millions': 'Volume ($M)'},
                color_discrete_map=COLOR_MAP,
                text_auto='.1f'
            )
            fig_vol.update_layout(showlegend=False)
            st.plotly_chart(fig_vol, use_container_width=True)

        col_g, col_h = st.columns(2)
        with col_g:
            tower_df = df_p_filtered.groupby('tower_number')['price_per_sqft'].mean().reset_index()
            fig_tower = px.line(
                tower_df,
                x='tower_number',
                y='price_per_sqft',
                title="Average Valuation ($ / SqFt) Across Towers 1 - 20",
                labels={'tower_number': 'Tower Number', 'price_per_sqft': 'Avg Price / SqFt ($)'},
                markers=True
            )
            fig_tower.update_traces(line_color='#861645', marker_color='#F2822E', marker_size=8)
            st.plotly_chart(fig_tower, use_container_width=True)
            
        with col_h:
            status_df = df_p_filtered['listing_status'].value_counts().reset_index()
            status_df.columns = ['Status', 'Count']
            fig_status = px.pie(
                status_df,
                values='Count',
                names='Status',
                title="Inventory Status: Sold vs. Available",
                color='Status',
                color_discrete_map={'Sold': '#861645', 'Available': '#85A611'},
                hole=0.4
            )
            st.plotly_chart(fig_status, use_container_width=True)

# MODULE 4: GEOGRAPHIC ORIGIN
with tab4:
    st.subheader("Regional Capital Distribution & Geographic Origin")
    st.write("Spatial tracking of domestic vs. cross-border real estate investor activity.")
    
    if len(df_c_filtered) == 0:
        st.warning("No client records match the current sidebar filter criteria.")
    else:
        col_i, col_j = st.columns(2)
        with col_i:
            geo_region = df_c_filtered.groupby(['region', 'buyer_persona']).size().reset_index(name='count')
            fig_geo = px.bar(
                geo_region,
                x='region',
                y='count',
                color='buyer_persona',
                title="Regional Buyer Concentration by Persona",
                labels={'region': 'Geographic Region', 'count': 'Client Count'},
                color_discrete_map=COLOR_MAP
            )
            st.plotly_chart(fig_geo, use_container_width=True)
            
        with col_j:
            country_df = df_c_filtered['country'].value_counts().reset_index()
            country_df.columns = ['Country', 'Client Count']
            fig_country = px.choropleth(
                country_df,
                locations='Country',
                locationmode='country names',
                color='Client Count',
                title="Global Buyer Origin Density Map",
                color_continuous_scale=['#354A57', '#861645', '#F2822E']
            )
            st.plotly_chart(fig_country, use_container_width=True)

# MODULE 5: STRATEGIC INSIGHTS
with tab5:
    st.subheader("Descriptive Metrics & Strategic Recommendations")
    st.write("Statistical profiling and targeted business strategies for Parcl platform decision-makers.")
    
    if len(df_c_filtered) == 0:
        st.warning("No client records match the current sidebar filter criteria.")
    else:
        stats_df = df_c_filtered.groupby('buyer_persona').agg(
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
        st.subheader("💡 Strategic Recommendations for Parcl Platform")
        
        rec1, rec2 = st.columns(2)
        with rec1:
            st.markdown("""
            **C1: Global Investors** (Magenta `#861645`)
            - *Strategy*: Cross-border yield syndication, tax-optimized currency escrow, high-cap commercial listings.
            - *Targeting*: International digital marketing in primary global financial hubs.
            
            **C2: First-Time Buyers** (Sunset Orange `#F2822E`)
            - *Strategy*: Embedded mortgage comparison tools, introductory down-payment assistance calculators.
            - *Targeting*: Social media referral programs and localized housing affordability guides.
            """)
        with rec2:
            st.markdown("""
            **C3: Corporate Buyers** (Slate Teal `#354A57`)
            - *Strategy*: Institutional API access, bulk multi-unit acquisition discounts, custom compliance reporting.
            - *Targeting*: Direct B2B enterprise sales and institutional real estate advisory.
            
            **C4: Luxury Investors** (Olive Lime `#85A611`)
            - *Strategy*: White-glove private wealth concierge, off-market premier listings, priority investor support.
            - *Targeting*: Exclusive investor summits and private wealth manager partnerships.
            """)

st.sidebar.divider()
st.sidebar.caption("Parcl Real Estate ML Engine v1.0 | Capstone Intelligence")
