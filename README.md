# 🏢 Parcl Real Estate Market Intelligence: Machine Learning Based Buyer Segmentation & Investment Profiling

An end-to-end Machine Learning and Data Science project developed for **Parcl Co. Limited** in collaboration with **Unified Mentor**. This platform transforms raw real estate customer interaction logs into actionable, data-driven buyer personas to optimize marketing spend, financing strategies, and platform transaction velocity.

---

## 📌 Project Overview & Business Value

Real estate platforms often treat buyers uniformly, leading to mismatched property recommendations and high customer acquisition costs. This project addresses Parcl's diagnostic challenges by implementing an unsupervised machine learning pipeline to segment platform users into 4 distinct buyer profiles:

1. **C1: Global Investors**: High-net-worth international individuals seeking yield and capital preservation.
2. **C2: First-Time Buyers**: Younger domestic buyers heavily dependent on debt financing and mortgage products.
3. **C3: Corporate Buyers**: Registered business entities acquiring multi-unit portfolios and commercial assets.
4. **C4: Luxury Investors**: High-satisfaction buyers making high-margin, flagship investments.

---

## 🛠️ Technical Stack & Frameworks

* **Language**: Python 3.9+
* **Data Science & ML**: `pandas`, `numpy`, `scikit-learn` (`StandardScaler`, `OneHotEncoder`, `KMeans`, `AgglomerativeClustering`, `silhouette_score`)
* **Visualization**: `plotly`, `matplotlib`, `seaborn`
* **Web Dashboard**: `streamlit`
* **Deployment**: Streamlit Community Cloud & GitHub

---

## 📁 Repository Structure

```
├── ml_pipeline.py        # 6-step unsupervised ML workflow script
├── app.py                # Interactive Streamlit Web Application (4 analytical modules)
├── clients.csv           # Base platform client dataset (2,000 records)
├── segmented_clients.csv # Model output containing assigned buyer personas
├── requirements.txt      # Python dependency manifest
└── README.md             # Project documentation
```

---

## ⚙️ Installation & Local Setup

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/Parcl_MBA_Project.git
cd Parcl_MBA_Project
```

### 2. Install Required Dependencies
```bash
python -m pip install -r requirements.txt
```

### 3. Run the Machine Learning Pipeline
Execute the data cleaning, feature engineering, scaling, and clustering pipeline:
```bash
python ml_pipeline.py
```

### 4. Launch the Interactive Dashboard
Launch the Streamlit web dashboard in your browser:
```bash
python -m streamlit run app.py
```

---

## 🌐 Live Web Application & Deployment

The dashboard is live on Streamlit Community Cloud featuring 4 core modules:
* **Buyer Segmentation Overview**: Cluster proportions and distribution.
* **Investor Behavior Dashboard**: Debt dependency and investment intent analysis.
* **Geographic Buyer Analysis**: Regional capital flows and global origin mapping.
* **Segment Insights Panel**: Persona metrics and strategic business recommendations.

---

## 📜 Academic & Corporate Context

This project fulfills both the **Unified Mentor Internship Capstone criteria** and academic evaluation standards for an **MBA in Finance & Market Intelligence**.
