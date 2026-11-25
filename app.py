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
# CUSTOM STYLING (LIGHT BLUE BACKGROUND & HIGH CONTRAST UI)
# ---------------------------------------------------------
st.markdown("""
<style>
    /* Main App Light Blue Background */
    .stApp {
        background-color: #EBF5FF !important;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #D6E8FA !important;
        border-right: 1px solid #BFDBFE !important;
    }
    
    /* Headers & Text */
    .main-header {
        font-size: 2.3rem;
        font-weight: 800;
        color: #0F2942;
        margin-bottom: 0.2rem;
        letter-spacing: -0.5px;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #335C81;
        margin-bottom: 1.5rem;
        font-weight: 600;
    }
    
    /* Custom Card Containers */
    .custom-card {
        background-color: #FFFFFF;
        border: 2px solid #BEE3F8;
        border-radius: 12px;
        padding: 18px;
        box-shadow: 0 4px 12px rgba(15, 41, 66, 0.06);
        text-align: center;
        margin-bottom: 15px;
    }
    .card-title {
        font-size: 0.9rem;
        font-weight: 700;
        color: #4A5568;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .card-value {
        font-size: 1.8rem;
        font-weight: 800;
        color: #0F2942;
        margin-top: 5px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# DATA LOADING & PREPROCESSING
# ---------------------------------------------------------
@st.cache_data
def load_all_data():
    # 1. Load Client Dataset
    client_file = "segmented_clients.csv"
    if os.path.exists(client_file):
        df_clients = pd.read_csv(client_file)
    elif os.path.exists("clients.csv"):
        df_clients = pd.read_csv("clients.csv")
    else:
        # Fallback benchmark generator
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

    # Clean client_id format
    df_clients['client_id_norm'] = df_clients['client_id'].astype(str).str.strip().str.upper()

    # 2. Load Property Files
    prop_files = sorted(glob.glob("properties_part_*.csv")) or sorted(glob.glob("/workspace/knowledge/properties_part_*.csv"))
    if prop_files:
        df_props = pd.concat([pd.read_csv(f) for f in prop_files], ignore_index=True)
    elif os.path.exists("properties.csv"):
        df_props = pd.read_csv("properties.csv")
    else:
        # Fallback properties dataset
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
        df_props['sale_price_clean'] = (
            df_props['sale_price']
            .astype(str)
            .str.replace('$', '', regex=False)
            .str.replace(',', '', regex=False)
            .astype(float)
        )
    else:
        df_props['sale_price_clean'] = df_props['floor_area_sqft'] * 300.0

    # Calculate Price per SqFt
    df_props['price_per_sqft'] = (df_props['sale_price_clean'] / df_props['floor_area_sqft']).round(2)

    # Normalize client_ref for joining
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

    df_props['client_ref_norm'] = df_props['client_ref'].apply(normalize_ref)

    # Merge Properties with Client Profiles
    df_merged = pd.merge(
        df_props,
        df_clients[['client_id_norm', 'client_type', 'country', 'region', 'acquisition_purpose', 'loan_applied', 'satisfaction_score', 'buyer_persona']],
        left_on='client_ref_norm',
        right_on='client_id_norm',
        how='left'
    )
    
    df_merged['buyer_persona'] = df_merged['buyer_persona'].fillna('C1: Global Investors')
    df_merged['unit_category'] = df_merged['unit_category'].fillna('Apartment')
    df_merged['listing_status'] = df_merged['listing_status'].fillna('Sold')

    return df_clients, df_merged

df_clients_raw, df_merged_raw = load_all_data()

# ---------------------------------------------------------
# COLOR PALETTE MAPPING (FROM cp.jpeg)
# ---------------------------------------------------------
COLOR_PALETTE = {
    'C1: Global Investors': '#861645',   # Rich Magenta Purple
    'C2: First-Time Buyers': '#F2822E',  # Vibrant Sunset Orange
    'C3: Corporate Buyers': '#354A57',   # Deep Slate Teal
    'C4: Luxury Investors': '#85A611',   # Chartreuse / Olive Lime
    'Apartment': '#354A57',
    'Office': '#861645',
    'Sold': '#85A611',
    'Available': '#F2822E'
}

# ---------------------------------------------------------
# HEADER SECTION
# ---------------------------------------------------------
st.markdown("<div class='main-header'>🏢 Parcl Real Estate Market Intelligence</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>Machine Learning Buyer Segmentation & Property Portfolio Intelligence Dashboard</div>", unsafe_allow_html=True)

# ---------------------------------------------------------
# SIDEBAR FILTERS
# ---------------------------------------------------------
st.sidebar.header("🔍 Filter Options")

all_countries = ['All'] + sorted(list(df_clients_raw['country'].dropna().unique()))
selected_country = st.sidebar.selectbox("Country of Residence", all_countries)

all_regions = ['All'] + sorted(list(df_clients_raw['region'].dropna().unique()))
selected_region = st.sidebar.selectbox("Geographic Region", all_regions)

all_purposes = ['All'] + sorted(list(df_clients_raw['acquisition_purpose'].dropna().unique()))
selected_purpose = st.sidebar.selectbox("Acquisition Purpose", all_purposes)

all_categories = ['All'] + sorted(list(df_merged_raw['unit_category'].dropna().unique()))
selected_category = st.sidebar.selectbox("Unit Category", all_categories)

# Apply Filter Logic
df_clients_filtered = df_clients_raw.copy()
df_merged_filtered = df_merged_raw.copy()

if selected_country != 'All':
    df_clients_filtered = df_clients_filtered[df_clients_filtered['country'] == selected_country]
    df_merged_filtered = df_merged_filtered[df_merged_filtered['country'] == selected_country]

if selected_region != 'All':
    df_clients_filtered = df_clients_filtered[df_clients_filtered['region'] == selected_region]
    df_merged_filtered = df_merged_filtered[df_merged_filtered['region'] == selected_region]

if selected_purpose != 'All':
    df_clients_filtered = df_clients_filtered[df_clients_filtered['acquisition_purpose'] == selected_purpose]
    df_merged_filtered = df_merged_filtered[df_merged_filtered['acquisition_purpose'] == selected_purpose]

if selected_category != 'All':
    df_merged_filtered = df_merged_filtered[df_merged_filtered['unit_category'] == selected_category]

# ---------------------------------------------------------
# TOP KPI SUMMARY METRICS
# ---------------------------------------------------------
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

with kpi1:
    st.markdown(f"""
    <div class='custom-card' style='border-top: 4px solid #354A57;'>
        <div class='card-title'>Total Clients</div>
        <div class='card-value'>{len(df_clients_filtered):,}</div>
    </div>
    """, unsafe_allow_html=True)

with kpi2:
    st.markdown(f"""
    <div class='custom-card' style='border-top: 4px solid #861645;'>
        <div class='card-title'>Total Properties</div>
        <div class='card-value'>{len(df_merged_filtered):,}</div>
    </div>
    """, unsafe_allow_html=True)

with kpi3:
    avg_price = df_merged_filtered['sale_price_clean'].mean() if len(df_merged_filtered) > 0 else 0
    st.markdown(f"""
    <div class='custom-card' style='border-top: 4px solid #F2822E;'>
        <div class='card-title'>Avg Sale Price</div>
        <div class='card-value'>${avg_price:,.0f}</div>
    </div>
    """, unsafe_allow_html=True)

with kpi4:
    avg_sqft_price = df_merged_filtered['price_per_sqft'].mean() if len(df_merged_filtered) > 0 else 0
    st.markdown(f"""
    <div class='custom-card' style='border-top: 4px solid #85A611;'>
        <div class='card-title'>Avg Price / SqFt</div>
        <div class='card-value'>${avg_sqft_price:.1f}</div>
    </div>
    """, unsafe_allow_html=True)

with kpi5:
    sold_ratio = (df_merged_filtered['listing_status'] == 'Sold').mean() * 100 if len(df_merged_filtered) > 0 else 0
    st.markdown(f"""
    <div class='custom-card' style='border-top: 4px solid #354A57;'>
        <div class='card-title'>Sold Ratio</div>
        <div class='card-value'>{sold_ratio:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

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

# MODULE 1: BUYER SEGMENTATION OVERVIEW
with tab1:
    st.subheader("Buyer Segment Distribution & Persona Breakdown")
    if len(df_clients_filtered) == 0:
        st.warning("No client records match the selected filter criteria.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            segment_counts = df_clients_filtered['buyer_persona'].value_counts().reset_index()
            segment_counts.columns = ['Persona', 'Count']
            
            fig_pie = px.pie(
                segment_counts,
                values='Count',
                names='Persona',
                title="Proportion of Buyer Personas",
                color='Persona',
                color_discrete_map=COLOR_PALETTE,
                hole=0.4
            )
            fig_pie.update_traces(textinfo='percent+label', marker=dict(line=dict(color='#FFFFFF', width=2)))
            fig_pie.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_pie, use_container_width=True)

        with c2:
            fig_bar = px.bar(
                segment_counts,
                x='Persona',
                y='Count',
                color='Persona',
                title="Client Count by Persona Cohort",
                text='Count',
                color_discrete_map=COLOR_PALETTE
            )
            fig_bar.update_layout(xaxis_title="", yaxis_title="Number of Clients", showlegend=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_bar, use_container_width=True)

# MODULE 2: INVESTOR BEHAVIOR
with tab2:
    st.subheader("Financing Dependency & Acquisition Intent")
    if len(df_clients_filtered) == 0:
        st.warning("No client records match the selected filter criteria.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            loan_df = df_clients_filtered.groupby(['buyer_persona', 'loan_applied']).size().reset_index(name='count')
            fig_loan = px.bar(
                loan_df,
                x='buyer_persona',
                y='count',
                color='loan_applied',
                barmode='group',
                title="Mortgage Loan Reliance",
                color_discrete_map={'Yes': '#F2822E', 'No': '#354A57'}
            )
            fig_loan.update_layout(xaxis_title="", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_loan, use_container_width=True)

        with c2:
            purpose_df = df_clients_filtered.groupby(['buyer_persona', 'acquisition_purpose']).size().reset_index(name='count')
            fig_purpose = px.bar(
                purpose_df,
                x='buyer_persona',
                y='count',
                color='acquisition_purpose',
                barmode='stack',
                title="Acquisition Purpose Breakdown",
                color_discrete_map={'Investment': '#861645', 'Personal use': '#85A611'}
            )
            fig_purpose.update_layout(xaxis_title="", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_purpose, use_container_width=True)

# MODULE 3: PROPERTY PORTFOLIO ANALYTICS
with tab3:
    st.subheader("Property Valuation & Capital Distribution")
    if len(df_merged_filtered) == 0:
        st.warning("No property records match the selected filter criteria.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            fig_box = px.box(
                df_merged_filtered,
                x='unit_category',
                y='sale_price_clean',
                color='unit_category',
                title="Property Price Range by Unit Category",
                color_discrete_map=COLOR_PALETTE,
                labels={'sale_price_clean': 'Sale Price ($)', 'unit_category': 'Category'}
            )
            fig_box.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_box, use_container_width=True)

        with c2:
            persona_vol = df_merged_filtered.groupby('buyer_persona')['sale_price_clean'].sum().reset_index()
            persona_vol['Volume_M'] = (persona_vol['sale_price_clean'] / 1e6).round(2)
            fig_vol = px.bar(
                persona_vol,
                x='buyer_persona',
                y='Volume_M',
                color='buyer_persona',
                title="Total Investment Volume by Persona ($ Millions)",
                text='Volume_M',
                color_discrete_map=COLOR_PALETTE
            )
            fig_vol.update_layout(xaxis_title="", yaxis_title="Capital Volume ($M)", showlegend=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_vol, use_container_width=True)

# MODULE 4: GEOGRAPHIC ORIGIN
with tab4:
    st.subheader("Geographic Origin & Regional Concentration")
    if len(df_clients_filtered) == 0:
        st.warning("No client records match the selected filter criteria.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            geo_region = df_clients_filtered.groupby(['region', 'buyer_persona']).size().reset_index(name='count')
            fig_geo = px.bar(
                geo_region,
                x='region',
                y='count',
                color='buyer_persona',
                title="Regional Buyer Concentration",
                color_discrete_map=COLOR_PALETTE
            )
            fig_geo.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_geo, use_container_width=True)

        with c2:
            country_df = df_clients_filtered['country'].value_counts().reset_index()
            country_df.columns = ['Country', 'Count']
            fig_country = px.choropleth(
                country_df,
                locations='Country',
                locationmode='country names',
                color='Count',
                title="Global Buyer Origin Density Map",
                color_continuous_scale=['#BEE3F8', '#354A57', '#861645']
            )
            fig_country.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_country, use_container_width=True)

# MODULE 5: STRATEGIC INSIGHTS
with tab5:
    st.subheader("Cohort Metrics & Strategic Recommendations")
    if len(df_clients_filtered) == 0:
        st.warning("No client records match the selected filter criteria.")
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
        st.subheader("💡 Strategic Recommendations")

        r1, r2 = st.columns(2)
        with r1:
            st.markdown("""
            **C1: Global Investors**
            - *Strategy*: Cross-border yield syndication, tax-optimized currency escrow, high-cap commercial listings.
            - *Targeting*: International digital marketing in primary global financial hubs.
            
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
st.sidebar.caption("Parcl Real Estate Intelligence | Unified Mentor Capstone")
