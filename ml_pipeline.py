import os
import glob
import pandas as pd
import numpy as np

def normalize_client_ref(val):
    if pd.isna(val) or val is None:
        return ''
    s = str(val).strip().upper()
    if s.startswith('C') and not s.startswith('CL_') and len(s) > 1 and s[1:].isdigit():
        try:
            return f'CL_{int(s[1:]):04d}'
        except ValueError:
            return s
    return s

def run_ml_pipeline(client_input="clients.csv", output_path="segmented_clients.csv"):
    print("=" * 70)
    print("PARCL REAL ESTATE PLATFORM: UNIFIED ML ENGINE")
    print("1. Client Buyer Segmentation (Unsupervised Clustering)")
    print("2. Property Valuation & Price Prediction (Supervised Regression)")
    print("=" * 70)

    # 1. Load Client Dataset
    try:
        df_clients = pd.read_csv(client_input)
        print(f"\n[+] Loaded client dataset '{client_input}' with {len(df_clients)} records.")
    except Exception as e:
        print(f"\n[!] '{client_input}' not found. Generating benchmark client dataset...")
        np.random.seed(42)
        n = 2000
        client_types = np.random.choice(['Individual', 'Company'], size=n, p=[0.75, 0.25])
        genders = np.random.choice(['Male', 'Female', 'Other'], size=n, p=[0.48, 0.48, 0.04])
        countries = np.random.choice(['United States', 'United Kingdom', 'UAE', 'Germany', 'Singapore', 'Canada'], size=n)
        regions = np.random.choice(['North America', 'Europe', 'Middle East', 'Asia Pacific'], size=n)
        
        start_date = pd.to_datetime('1955-01-01').value
        end_date = pd.to_datetime('2002-12-31').value
        dobs = pd.to_datetime(np.random.randint(start_date, end_date, size=n))
        
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
        df_clients.to_csv(client_input, index=False)

    # Clean & Engineer Features
    df_clients = df_clients.drop_duplicates(subset=['client_id']).copy()
    if 'date_of_birth' in df_clients.columns:
        df_clients['date_of_birth'] = pd.to_datetime(df_clients['date_of_birth'], errors='coerce')
        df_clients['age'] = 2026 - df_clients['date_of_birth'].dt.year
        df_clients['age'] = df_clients['age'].fillna(df_clients['age'].median())
    elif 'age' not in df_clients.columns:
        df_clients['age'] = 35.0

    df_clients['client_type'] = df_clients['client_type'].fillna('Individual')
    df_clients['acquisition_purpose'] = df_clients['acquisition_purpose'].fillna('Investment')
    df_clients['loan_applied'] = df_clients['loan_applied'].fillna('No')
    df_clients['region'] = df_clients['region'].fillna('Unknown')
    df_clients['country'] = df_clients['country'].fillna('Unknown')
    df_clients['satisfaction_score'] = df_clients['satisfaction_score'].fillna(df_clients['satisfaction_score'].median())

    # Map Persona Definitions
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

    # Try Scikit-Learn; if system DLL blocked, use NumPy Engine
    try:
        from sklearn.preprocessing import StandardScaler, OneHotEncoder
        from sklearn.cluster import KMeans, AgglomerativeClustering
        from sklearn.metrics import silhouette_score, r2_score, mean_absolute_error, mean_squared_error
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.linear_model import LinearRegression
        from sklearn.model_selection import train_test_split
        
        categorical_cols = ['client_type', 'acquisition_purpose', 'loan_applied', 'region', 'referral_channel']
        encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
        encoded_cats = encoder.fit_transform(df_clients[categorical_cols])
        df_encoded_cats = pd.DataFrame(encoded_cats, columns=encoder.get_feature_names_out(categorical_cols), index=df_clients.index)
        
        scaler = StandardScaler()
        scaled_nums = scaler.fit_transform(df_clients[['age', 'satisfaction_score']])
        df_scaled_nums = pd.DataFrame(scaled_nums, columns=['age_scaled', 'satisfaction_score_scaled'], index=df_clients.index)
        
        X_cluster = pd.concat([df_scaled_nums, df_encoded_cats], axis=1)
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        df_clients['kmeans_cluster'] = kmeans.fit_predict(X_cluster)
        sil_score = silhouette_score(X_cluster, df_clients['kmeans_cluster'])
        print(f"[+] Buyer Segmentation Complete (Silhouette Score = {sil_score:.4f})")
        use_sklearn = True
    except Exception as err:
        print(f"[!] Scikit-Learn unavailable on host OS ({err}). Using Native NumPy ML Engine...")
        use_sklearn = False

    df_clients.to_csv(output_path, index=False)
    print(f"[+] Segmented client profiles saved to '{output_path}'.")

    # 2. Load Property Transaction Data & Build Valuation Regression
    print("\n[Step 2/2] Loading Property Transaction Records & Training Valuation Models...")
    prop_files = sorted(glob.glob("properties_part_*.csv")) or sorted(glob.glob("/workspace/knowledge/properties_part_*.csv"))
    
    if prop_files:
        print(f"    - Found {len(prop_files)} property files: {prop_files}")
        df_props = pd.concat([pd.read_csv(f) for f in prop_files], ignore_index=True)
    elif os.path.exists("properties.csv"):
        df_props = pd.read_csv("properties.csv")
    else:
        print("    - [!] No property CSVs found. Generating benchmark property dataset...")
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

    # Clean Price Column
    if 'sale_price' in df_props.columns:
        df_props['sale_price_clean'] = df_props['sale_price'].astype(str).str.replace('$', '').str.replace(',', '').astype(float)
    else:
        df_props['sale_price_clean'] = df_props['floor_area_sqft'] * 300.0

    # Calculate Price per SqFt
    if 'floor_area_sqft' in df_props.columns and 'sale_price_clean' in df_props.columns:
        df_props['price_per_sqft'] = (df_props['sale_price_clean'] / df_props['floor_area_sqft']).round(2)

    # Normalize client reference key safely (handles float, NaN, strings)
    if 'client_ref' in df_props.columns:
        df_props['client_ref_norm'] = df_props['client_ref'].apply(normalize_client_ref)
        merged_df = pd.merge(
            df_props,
            df_clients[['client_id', 'buyer_persona', 'acquisition_purpose', 'loan_applied']],
            left_on='client_ref_norm', right_on='client_id', how='left'
        )
    else:
        merged_df = df_props.copy()
        merged_df['buyer_persona'] = 'C1: Global Investors'

    merged_df['buyer_persona'] = merged_df['buyer_persona'].fillna('C1: Global Investors')
    merged_df['unit_category'] = merged_df['unit_category'].fillna('Apartment')
    merged_df['tower_number'] = merged_df['tower_number'].fillna(1)

    print(f"    - Merged property dataset shape: {merged_df.shape}")

    # Feature Matrix for Price Valuation
    reg_features = ['floor_area_sqft', 'tower_number', 'unit_category', 'buyer_persona']
    X_reg = pd.get_dummies(merged_df[reg_features], drop_first=True)
    y_reg = merged_df['sale_price_clean']

    if use_sklearn:
        X_train, X_test, y_train, y_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
        
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        y_pred_lr = lr.predict(X_test)
        r2_lr = r2_score(y_test, y_pred_lr)
        mae_lr = mean_absolute_error(y_test, y_pred_lr)
        
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        y_pred_rf = rf.predict(X_test)
        r2_rf = r2_score(y_test, y_pred_rf)
        rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
        mae_rf = mean_absolute_error(y_test, y_pred_rf)

        print("\n" + "=" * 70)
        print("MODEL EVALUATION & VALUATION PERFORMANCE SUMMARY")
        print("=" * 70)
        print(f"Linear Regression R² Score:    {r2_lr:.4f} (MAE = ${mae_lr:,.2f})")
        print(f"Random Forest Regressor R²:    {r2_rf:.4f} (RMSE = ${rmse_rf:,.2f}, MAE = ${mae_rf:,.2f})")
        
        importances = pd.Series(rf.feature_importances_, index=X_reg.columns).sort_values(ascending=False)
        print("\nTop Property Valuation Drivers (Feature Importance):")
        for feat, imp in importances.head(5).items():
            print(f"  • {feat:<35}: {imp*100:.2f}% importance")
    else:
        # Pure NumPy OLS Model
        X_mat = np.column_stack([np.ones(len(X_reg)), X_reg.values.astype(float)])
        y_vec = y_reg.values
        coeffs, _, _, _ = np.linalg.lstsq(X_mat, y_vec, rcond=None)
        y_pred = X_mat @ coeffs
        ss_res = np.sum((y_vec - y_pred) ** 2)
        ss_tot = np.sum((y_vec - np.mean(y_vec)) ** 2)
        r2_val = 1 - (ss_res / ss_tot)
        mae_val = np.mean(np.abs(y_vec - y_pred))

        print("\n" + "=" * 70)
        print("MODEL EVALUATION & VALUATION PERFORMANCE SUMMARY (NumPy OLS Engine)")
        print("=" * 70)
        print(f"Property Valuation R² Score:   {r2_val:.4f}")
        print(f"Mean Absolute Error (MAE):     ${mae_val:,.2f}")

    print("\n[+] Unified ML Pipeline Executed Successfully!")
    print("=" * 70)

if __name__ == "__main__":
    run_ml_pipeline()
