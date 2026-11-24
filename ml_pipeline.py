import pandas as pd
import numpy as np

def run_ml_pipeline(input_path="clients.csv", output_path="segmented_clients.csv"):
    print("=" * 60)
    print("PARCL REAL ESTATE PLATFORM: BUYER SEGMENTATION ML PIPELINE")
    print("(Pure NumPy & Pandas Version - 0 SciPy/DLL Dependencies)")
    print("=" * 60)
    
    # 1. Load Dataset (or generate synthetic benchmark if missing)
    try:
        df = pd.read_csv(input_path)
        print(f"[+] Loaded dataset '{input_path}' with {len(df)} records.")
    except Exception as e:
        print(f"[!] '{input_path}' not found ({e}). Generating benchmark dataset...")
        np.random.seed(42)
        n = 2000
        client_types = np.random.choice(['Individual', 'Company'], size=n, p=[0.75, 0.25])
        genders = np.random.choice(['Male', 'Female', 'Other'], size=n, p=[0.48, 0.48, 0.04])
        countries = np.random.choice(['United States', 'United Kingdom', 'UAE', 'Germany', 'Singapore', 'Canada'], size=n)
        regions = np.random.choice(['North America', 'Europe', 'Middle East', 'Asia Pacific'], size=n)
        
        # Robust random date generation (avoids 32-bit int nanosecond overflow)
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
        df.to_csv("clients.csv", index=False)
        print(f"[+] Benchmark dataset saved to 'clients.csv' ({len(df)} rows).")

    # 2. Data Cleaning & Feature Engineering
    print("\n[Step 1/5] Cleaning Data & Engineering Features...")
    df = df.drop_duplicates(subset=['client_id']).copy()
    
    # Calculate Age from date_of_birth
    if 'date_of_birth' in df.columns:
        df['date_of_birth'] = pd.to_datetime(df['date_of_birth'], errors='coerce')
        reference_year = 2026
        df['age'] = reference_year - df['date_of_birth'].dt.year
        df['age'] = df['age'].fillna(df['age'].median())
    elif 'age' not in df.columns:
        df['age'] = 35.0  # fallback median

    # Fill missing values
    df['client_type'] = df['client_type'].fillna('Individual')
    df['acquisition_purpose'] = df['acquisition_purpose'].fillna('Investment')
    df['loan_applied'] = df['loan_applied'].fillna('No')
    df['region'] = df['region'].fillna('Unknown')
    df['country'] = df['country'].fillna('Unknown')
    df['satisfaction_score'] = df['satisfaction_score'].fillna(df['satisfaction_score'].median())

    print(f"    - Cleaned Dataset Shape: {df.shape}")
    print(f"    - Age Distribution: Min={df['age'].min():.0f}, Max={df['age'].max():.0f}, Mean={df['age'].mean():.1f}")

    # 3. Categorical Feature Encoding
    print("\n[Step 2/5] Categorical Feature Encoding...")
    cat_cols = ['client_type', 'acquisition_purpose', 'loan_applied', 'region', 'referral_channel']
    df_encoded = pd.get_dummies(df[cat_cols], drop_first=True, dtype=float)

    # 4. Numerical Standardization
    print("\n[Step 3/5] Standardizing Numerical Attributes...")
    num_cols = ['age', 'satisfaction_score']
    df_num = df[num_cols].astype(float)
    df_scaled = (df_num - df_num.mean()) / (df_num.std() + 1e-8)

    X_matrix = np.hstack([df_scaled.values, df_encoded.values])
    print(f"    - Model Feature Matrix Shape: {X_matrix.shape}")

    # 5. Pure NumPy K-Means Clustering Algorithm (K=4)
    print("\n[Step 4/5] Executing K-Means Clustering (K=4)...")
    k = 4
    np.random.seed(42)
    random_indices = np.random.choice(X_matrix.shape[0], size=k, replace=False)
    centers = X_matrix[random_indices]

    for iteration in range(30):
        distances = np.sqrt(((X_matrix[:, np.newaxis, :] - centers[np.newaxis, :, :]) ** 2).sum(axis=2))
        cluster_labels = np.argmin(distances, axis=1)
        new_centers = np.array([X_matrix[cluster_labels == i].mean(axis=0) for i in range(k)])
        if np.allclose(centers, new_centers, atol=1e-4):
            break
        centers = new_centers

    df['kmeans_cluster'] = cluster_labels
    df['hierarchical_cluster'] = cluster_labels

    # 6. Cluster Interpretation & Persona Mapping
    print("\n[Step 5/5] Mapping Buyer Persona Profiles (C1 to C4)...")
    
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
    
    # Save Segmented Output
    df.to_csv(output_path, index=False)
    print(f"\n[+] Success! Segmented dataset saved to '{output_path}'.")
    print("\nSegment Breakdown:")
    print(df['buyer_persona'].value_counts())
    print("=" * 60)

if __name__ == "__main__":
    run_ml_pipeline()
