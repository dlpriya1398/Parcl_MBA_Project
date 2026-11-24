import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="Parcl Real Estate Intelligence | Buyer Segmentation",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling for polished UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.0rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Data Loading Function
@st.cache_data
def load_data():
    file_path = "segmented_clients.csv"
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
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
        
        df = pd.DataFrame({
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
        
        # Derived attributes
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

# Header Section
st.markdown("<div class='main-header'>🏢 Parcl Real Estate Market Intelligence</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>Machine Learning Based Buyer Segmentation & Investment Profiling Dashboard</div>", unsafe_allow_html=True)
st.divider()

# Sidebar Interactive Filters (As per Slide 13 specifications)
st.sidebar.header("🔍 Filter Options")

all_countries = ['All'] + sorted(list(df_raw['country'].unique()))
selected_country = st.sidebar.selectbox("Country of Residence", all_countries)

all_regions = ['All'] + sorted(list(df_raw['region'].unique()))
selected_region = st.sidebar.selectbox("Geographic Region", all_regions)

all_purposes = ['All'] + sorted(list(df_raw['acquisition_purpose'].unique()))
selected_purpose = st.sidebar.selectbox("Acquisition Purpose", all_purposes)

all_types = ['All'] + sorted(list(df_raw['client_type'].unique()))
selected_client_type = st.sidebar.selectbox("Client Type", all_types)

# Apply Filter Logic
df_filtered = df_raw.copy()

if selected_country != 'All':
    df_filtered = df_filtered[df_filtered['country'] == selected_country]
if selected_region != 'All':
    df_filtered = df_filtered[df_filtered['region'] == selected_region]
if selected_purpose != 'All':
    df_filtered = df_filtered[df_filtered['acquisition_purpose'] == selected_purpose]
if selected_client_type != 'All':
    df_filtered = df_filtered[df_filtered['client_type'] == selected_client_type]

# KPI Summary Cards
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Clients", len(df_filtered))
with col2:
    inv_ratio = (df_filtered['acquisition_purpose'] == 'Investment').mean() * 100 if len(df_filtered) > 0 else 0
    st.metric("Investment Purpose %", f"{inv_ratio:.1f}%")
with col3:
    loan_ratio = (df_filtered['loan_applied'] == 'Yes').mean() * 100 if len(df_filtered) > 0 else 0
    st.metric("Loan Reliance %", f"{loan_ratio:.1f}%")
with col4:
    avg_sat = df_filtered['satisfaction_score'].mean() if len(df_filtered) > 0 else 0
    st.metric("Avg Satisfaction", f"{avg_sat:.2f} / 5.0")

st.markdown("<br>", unsafe_allow_html=True)

# Main Navigation Tabs (4 Core Modules as per Slide 12)
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Buyer Segmentation Overview",
    "💼 Investor Behavior Dashboard",
    "🌐 Geographic Buyer Analysis",
    "🔍 Segment Insights Panel"
])

# ---------------------------------------------------------
# MODULE 1: BUYER SEGMENTATION OVERVIEW
# ---------------------------------------------------------
with tab1:
    st.subheader("Buyer Segment Distribution & Clustering Breakdown")
    st.write("Overview of client proportions classified across the 4 ML-derived buyer personas.")
    
    if len(df_filtered) == 0:
        st.warning("No client records match the current sidebar filter criteria.")
    else:
        m1_col1, m1_col2 = st.columns(2)
        
        with m1_col1:
            segment_counts = df_filtered['buyer_persona'].value_counts().reset_index()
            segment_counts.columns = ['Persona', 'Count']
            
            fig_pie = px.pie(
                segment_counts, 
                values='Count', 
                names='Persona',
                title="Proportion of Buyer Personas",
                color='Persona',
                color_discrete_sequence=px.colors.qualitative.Bold,
                hole=0.4
            )
            fig_pie.update_traces(textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)
            
        with m1_col2:
            fig_bar = px.bar(
                segment_counts,
                x='Persona',
                y='Count',
                color='Persona',
                title="Client Count by Persona Segment",
                text='Count',
                color_discrete_sequence=px.colors.qualitative.Bold
            )
            fig_bar.update_layout(xaxis_title="", yaxis_title="Number of Clients", showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)

# ---------------------------------------------------------
# MODULE 2: INVESTOR BEHAVIOR DASHBOARD
# ---------------------------------------------------------
with tab2:
    st.subheader("Financial & Behavioral Patterns by Persona")
    st.write("Examines debt dependency (loan reliance) and purchasing motivations across client cohorts.")
    
    if len(df_filtered) == 0:
        st.warning("No client records match the current sidebar filter criteria.")
    else:
        m2_col1, m2_col2 = st.columns(2)
        
        with m2_col1:
            # Financing Reliance: Loan Applied by Persona
            loan_df = df_filtered.groupby(['buyer_persona', 'loan_applied']).size().reset_index(name='count')
            fig_loan = px.bar(
                loan_df,
                x='buyer_persona',
                y='count',
                color='loan_applied',
                barmode='group',
                title="Mortgage Loan Reliance (Loan Applied)",
                labels={'buyer_persona': 'Persona Segment', 'count': 'Client Count', 'loan_applied': 'Applied for Loan'},
                color_discrete_map={'Yes': '#EF4444', 'No': '#10B981'}
            )
            fig_loan.update_layout(xaxis_title="")
            st.plotly_chart(fig_loan, use_container_width=True)
            
        with m2_col2:
            # Acquisition Purpose by Persona
            purpose_df = df_filtered.groupby(['buyer_persona', 'acquisition_purpose']).size().reset_index(name='count')
            fig_purpose = px.bar(
                purpose_df,
                x='buyer_persona',
                y='count',
                color='acquisition_purpose',
                barmode='stack',
                title="Acquisition Purpose Intent Breakdown",
                labels={'buyer_persona': 'Persona Segment', 'count': 'Client Count', 'acquisition_purpose': 'Intent'},
                color_discrete_map={'Investment': '#3B82F6', 'Personal use': '#F59E0B'}
            )
            fig_purpose.update_layout(xaxis_title="")
            st.plotly_chart(fig_purpose, use_container_width=True)

# ---------------------------------------------------------
# MODULE 3: GEOGRAPHIC BUYER ANALYSIS
# ---------------------------------------------------------
with tab3:
    st.subheader("Regional Capital Distribution & Geographic Origin")
    st.write("Spatial tracking of domestic vs. cross-border real estate investor activity.")
    
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
                color_discrete_sequence=px.colors.qualitative.Vivid
            )
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
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig_country, use_container_width=True)

# ---------------------------------------------------------
# MODULE 4: SEGMENT INSIGHTS PANEL
# ---------------------------------------------------------
with tab4:
    st.subheader("Descriptive Metrics & Strategic Recommendations")
    st.write("Statistical profiling and targeted business strategies for Parcl platform decision-makers.")
    
    if len(df_filtered) == 0:
        st.warning("No client records match the current sidebar filter criteria.")
    else:
        # Summary Metrics Table
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
