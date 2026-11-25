import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="Parcl Real Estate Market Intelligence",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS matching ui.jpeg exact visual interface
st.markdown("""
<style>
    /* Main App Background */
    .stApp {
        background-color: #F4F6F9;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    /* Top Hero Header Banner - Dark Space Blue with Gold Border */
    .hero-banner {
        background: linear-gradient(135deg, #0A1128 0%, #101F42 50%, #0A1128 100%);
        border: 2px solid #FFD700;
        border-radius: 15px;
        padding: 22px 30px;
        color: white;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.25);
        margin-bottom: 25px;
    }
    .hero-title {
        font-size: 2.3rem;
        font-weight: 800;
        color: #FFFFFF !important;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .hero-subtitle {
        font-size: 1.1rem;
        font-weight: 600;
        color: #00E5FF !important;
        margin-top: 6px;
    }

    /* KPI Summary Cards Container Grid */
    .kpi-container {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 18px;
        margin-bottom: 25px;
    }

    /* Individual KPI Metric Card (Vivid Sky Blue with Gold Border) */
    .kpi-card {
        background: linear-gradient(135deg, #00C6FF 0%, #0072FF 100%);
        border: 2px solid #FFD700;
        border-radius: 14px;
        padding: 16px 20px;
        box-shadow: 0 4px 12px rgba(0, 114, 255, 0.3);
        color: #0A1128;
    }
    .kpi-header {
        font-size: 1.05rem;
        font-weight: 700;
        color: #0A1128;
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 8px;
    }
    .kpi-value {
        font-size: 2.2rem;
        font-weight: 900;
        color: #FFD700;
        text-shadow: 1px 2px 4px rgba(0, 0, 0, 0.4);
        margin: 0;
    }

    /* Navigation Tabs - Pill Shaped with Gold Border */
    .stTabs [data-baseweb="tab-list"] {
        gap: 15px;
        background-color: transparent;
        padding-bottom: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        background: linear-gradient(135deg, #00C6FF 0%, #0072FF 100%) !important;
        border: 2px solid #FFD700 !important;
        border-radius: 25px !important;
        padding: 10px 24px !important;
        color: #0A1128 !important;
        font-weight: 700 !important;
        font-size: 1.0rem !important;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.15);
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #FF007F 0%, #E91E63 100%) !important;
        border: 2px solid #FFD700 !important;
        color: #FFFFFF !important;
        box-shadow: 0 6px 16px rgba(255, 0, 127, 0.4) !important;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0F172A;
    }
    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] h3, 
    section[data-testid="stSidebar"] label {
        color: #F8FAFC !important;
    }

    /* Section Header */
    .section-title {
        font-size: 1.8rem;
        font-weight: 800;
        color: #0A1128;
        margin-top: 15px;
        margin-bottom: 4px;
    }
    .section-desc {
        font-size: 1.0rem;
        color: #475569;
        margin-bottom: 20px;
    }

    /* Recommendation Cards */
    .rec-card {
        background-color: #FFFFFF;
        border-left: 5px solid #FF007F;
        border-radius: 12px;
        padding: 18px 22px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.06);
        margin-bottom: 18px;
    }
    .rec-title-1 { color: #3B82F6; font-weight: 800; font-size: 1.2rem; }
    .rec-title-2 { color: #10B981; font-weight: 800; font-size: 1.2rem; }
    .rec-title-3 { color: #F97316; font-weight: 800; font-size: 1.2rem; }
    .rec-title-4 { color: #FF007F; font-weight: 800; font-size: 1.2rem; }
</style>
""", unsafe_allow_html=True)

# Colors matching ui.jpeg
UI_PERSONA_MAP = {
    'C1: Global Investors': '#3B82F6',     # Royal Blue
    'C2: First-Time Buyers': '#FFD600',    # Bright Gold / Yellow
    'C3: Corporate Buyers': '#00E676',     # Mint / Turquoise Green
    'C4: Luxury Investors': '#FF007F'      # Magenta / Hot Pink
}

# Data Loading Function
@st.cache_data
def load_data():
    file_path = "segmented_clients.csv"
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
    else:
        np.random.seed(42)
        n = 2000
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
        
        df = pd.DataFrame({
            'client_id': [f'CL_{i:04d}' for i in range(1, n + 1)],
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
        
        df['age'] = 2026 - pd.to_datetime(df['date_of_birth']).dt.year
        
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

        df['buyer_persona'] = df.apply(assign_persona, axis=1)
        df['kmeans_cluster'] = df['buyer_persona'].map({
            'C1: Global Investors': 0,
            'C2: First-Time Buyers': 1,
            'C3: Corporate Buyers': 2,
            'C4: Luxury Investors': 3
        })
    return df

df_raw = load_data()

# Hero Banner Header Section
st.markdown("""
<div class='hero-banner'>
    <div class='hero-title'>🏢 Parcl Real Estate Market Intelligence</div>
    <div class='hero-subtitle'>Machine Learning Based Buyer Segmentation & Investment Profiling Dashboard</div>
</div>
""", unsafe_allow_html=True)

# Sidebar Interactive Filters
st.sidebar.markdown("<h2 style='color: #00E5FF;'>🔍 Filter Options</h2>", unsafe_allow_html=True)

all_countries = ['All'] + sorted(list(df_raw['country'].unique()))
selected_country = st.sidebar.selectbox("Country of Residence", all_countries)

all_regions = ['All'] + sorted(list(df_raw['region'].unique()))
selected_region = st.sidebar.selectbox("Geographic Region", all_regions)

all_purposes = ['All'] + sorted(list(df_raw['acquisition_purpose'].unique()))
selected_purpose = st.sidebar.selectbox("Acquisition Purpose", all_purposes)

all_types = ['All'] + sorted(list(df_raw['client_type'].unique()))
selected_client_type = st.sidebar.selectbox("Client Type", all_types)

# Filter Logic
df_filtered = df_raw.copy()

if selected_country != 'All':
    df_filtered = df_filtered[df_filtered['country'] == selected_country]
if selected_region != 'All':
    df_filtered = df_filtered[df_filtered['region'] == selected_region]
if selected_purpose != 'All':
    df_filtered = df_filtered[df_filtered['acquisition_purpose'] == selected_purpose]
if selected_client_type != 'All':
    df_filtered = df_filtered[df_filtered['client_type'] == selected_client_type]

# KPI Metric Cards (Matching ui.jpeg)
total_clients_val = f"{len(df_filtered):,}"
inv_ratio_val = f"{(df_filtered['acquisition_purpose'] == 'Investment').mean() * 100:.1f}%" if len(df_filtered) > 0 else "0%"
loan_ratio_val = f"{(df_filtered['loan_applied'] == 'Yes').mean() * 100:.1f}%" if len(df_filtered) > 0 else "0%"
avg_sat_val = f"{df_filtered['satisfaction_score'].mean():.2f} / 5.0" if len(df_filtered) > 0 else "0.0 / 5.0"

st.markdown(f"""
<div class='kpi-container'>
    <div class='kpi-card'>
        <div class='kpi-header'>👤 Total Clients</div>
        <div class='kpi-value'>{total_clients_val}</div>
    </div>
    <div class='kpi-card'>
        <div class='kpi-header'>🎯 Investment Intent</div>
        <div class='kpi-value'>{inv_ratio_val}</div>
    </div>
    <div class='kpi-card'>
        <div class='kpi-header'>🏛️ Loan Dependency</div>
        <div class='kpi-value'>{loan_ratio_val}</div>
    </div>
    <div class='kpi-card'>
        <div class='kpi-header'>⭐ Avg Satisfaction</div>
        <div class='kpi-value'>{avg_sat_val}</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Navigation Tabs (Pill Buttons)
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Buyer Segmentation Overview",
    "💼 Investor Behavior Dashboard",
    "🌐 Geographic Buyer Analysis",
    "🔍 Segment Insights Panel"
])

# MODULE 1: BUYER SEGMENTATION OVERVIEW
with tab1:
    st.markdown("<div class='section-title'>Buyer Segment Distribution & Clustering Breakdown</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='section-desc'>Overview of {len(df_filtered):,} participants classified across the 4 ML-derived buyer personas.</div>", unsafe_allow_html=True)
    
    if len(df_filtered) == 0:
        st.warning("No client records match the current sidebar filter criteria.")
    else:
        m1_col1, m1_col2 = st.columns(2)
        
        with m1_col1:
            segment_counts = df_filtered['buyer_persona'].value_counts().reset_index()
            segment_counts.columns = ['Persona', 'Count']
            
            # Pie Chart matching ui.jpeg
            fig_pie = px.pie(
                segment_counts, 
                values='Count', 
                names='Persona',
                title="Proportion of Buyer Personas",
                color='Persona',
                color_discrete_map=UI_PERSONA_MAP
            )
            fig_pie.update_traces(
                textinfo='percent', 
                textfont=dict(size=15, color='black', family='Segoe UI'),
                marker=dict(line=dict(color='#FFD700', width=2))
            )
            fig_pie.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', 
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(size=14, color='#0A1128', family='Segoe UI'),
                title_font=dict(size=18, color='#0A1128', family='Segoe UI')
            )
            st.plotly_chart(fig_pie, use_container_width=True)
            
        with m1_col2:
            # Bar Chart matching ui.jpeg
            fig_bar = px.bar(
                segment_counts,
                x='Persona',
                y='Count',
                color='Persona',
                title="Client Count by Persona Segment",
                text='Count',
                color_discrete_map=UI_PERSONA_MAP
            )
            fig_bar.update_traces(
                textposition='outside',
                textfont=dict(size=14, color='#0A1128', family='Segoe UI'),
                marker=dict(line=dict(color='#FFD700', width=2))
            )
            fig_bar.update_layout(
                xaxis_title="", 
                yaxis_title="", 
                showlegend=False,
                paper_bgcolor='rgba(0,0,0,0)', 
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(size=13, color='#0A1128', family='Segoe UI'),
                title_font=dict(size=18, color='#0A1128', family='Segoe UI')
            )
            st.plotly_chart(fig_bar, use_container_width=True)

# MODULE 2: INVESTOR BEHAVIOR DASHBOARD
with tab2:
    st.markdown("<div class='section-title'>Financial & Behavioral Patterns by Persona</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-desc'>Examines debt dependency (loan reliance) and purchasing motivations across client cohorts.</div>", unsafe_allow_html=True)
    
    if len(df_filtered) == 0:
        st.warning("No client records match the current sidebar filter criteria.")
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
                labels={'buyer_persona': 'Persona Segment', 'count': 'Client Count', 'loan_applied': 'Applied for Loan'},
                color_discrete_map={'Yes': '#FF007F', 'No': '#00E676'}
            )
            fig_loan.update_traces(marker=dict(line=dict(color='#FFD700', width=1.5)))
            fig_loan.update_layout(xaxis_title="", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
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
                labels={'buyer_persona': 'Persona Segment', 'count': 'Client Count', 'acquisition_purpose': 'Intent'},
                color_discrete_map={'Investment': '#00C6FF', 'Personal use': '#FFD600'}
            )
            fig_purpose.update_traces(marker=dict(line=dict(color='#FFD700', width=1.5)))
            fig_purpose.update_layout(xaxis_title="", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_purpose, use_container_width=True)

# MODULE 3: GEOGRAPHIC BUYER ANALYSIS
with tab3:
    st.markdown("<div class='section-title'>Regional Capital Distribution & Geographic Origin</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-desc'>Spatial tracking of domestic vs. cross-border real estate investor activity.</div>", unsafe_allow_html=True)
    
    if len(df_filtered) == 0:
        st.warning("No client records match the current sidebar filter criteria.")
    else:
        m3_col1, m3_col2 = st.columns(2)
        
        with m3_col1:
            geo_region = df_filtered.groupby(['region', 'buyer_persona']).size().reset_index(name='count')
            fig_geo = px.bar(
                geo_region,
                x='region',
                y='count',
                color='buyer_persona',
                title="Regional Buyer Concentration by Persona",
                labels={'region': 'Geographic Region', 'count': 'Client Count'},
                color_discrete_map=UI_PERSONA_MAP
            )
            fig_geo.update_traces(marker=dict(line=dict(color='#FFD700', width=1.5)))
            fig_geo.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_geo, use_container_width=True)
            
        with m3_col2:
            country_df = df_filtered['country'].value_counts().reset_index()
            country_df.columns = ['Country', 'Client Count']
            fig_country = px.choropleth(
                country_df,
                locations='Country',
                locationmode='country names',
                color='Client Count',
                title="Global Buyer Origin Density Map",
                color_continuous_scale='Plasma'
            )
            fig_country.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_country, use_container_width=True)

# MODULE 4: SEGMENT INSIGHTS PANEL
with tab4:
    st.markdown("<div class='section-title'>Descriptive Metrics & Strategic Recommendations</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-desc'>Statistical profiling and targeted business strategies for Parcl platform decision-makers.</div>", unsafe_allow_html=True)
    
    if len(df_filtered) == 0:
        st.warning("No client records match the current sidebar filter criteria.")
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
        st.markdown("<h3 style='color: #0A1128;'>💡 Strategic Recommendations for Parcl</h3>", unsafe_allow_html=True)
        
        rec_col1, rec_col2 = st.columns(2)
        with rec_col1:
            st.markdown("""
            <div class='rec-card'>
                <div class='rec-title-1'>C1: Global Investors</div>
                <p><b>Strategy</b>: Cross-border yield syndication, tax-optimized currency escrow, high-cap commercial listings.</p>
                <p><b>Targeting</b>: International digital marketing in primary global financial hubs.</p>
            </div>
            <div class='rec-card'>
                <div class='rec-title-2'>C2: First-Time Buyers</div>
                <p><b>Strategy</b>: Embedded mortgage comparison tools, introductory down-payment assistance calculators.</p>
                <p><b>Targeting</b>: Social media referral programs and localized housing affordability guides.</p>
            </div>
            """, unsafe_allow_html=True)
            
        with rec_col2:
            st.markdown("""
            <div class='rec-card'>
                <div class='rec-title-3'>C3: Corporate Buyers</div>
                <p><b>Strategy</b>: Institutional API access, bulk multi-unit acquisition discounts, custom compliance reporting.</p>
                <p><b>Targeting</b>: Direct B2B enterprise sales and institutional real estate advisory.</p>
            </div>
            <div class='rec-card'>
                <div class='rec-title-4'>C4: Luxury Investors</div>
                <p><b>Strategy</b>: White-glove private wealth concierge, off-market premier listings, priority investor support.</p>
                <p><b>Targeting</b>: Exclusive investor summits and private wealth manager partnerships.</p>
            </div>
            """, unsafe_allow_html=True)

st.sidebar.divider()
st.sidebar.caption("Parcl Real Estate ML Engine v1.0 | Unified Mentor Capstone")
