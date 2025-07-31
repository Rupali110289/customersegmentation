import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import joblib
from sklearn.preprocessing import StandardScaler
import os

# --- User credentials ---
users = {
    "store_owner": {"password": "owner123", "role": "Store Owner"},
    "customer_01": {"password": "cust123", "role": "Customer"},
    "Mukundhan": {"password": "MLUswag257", "role": "Customer"}
}

# --- Session State Setup ---
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
if "role" not in st.session_state:
    st.session_state["role"] = None
if "login_success" not in st.session_state:
    st.session_state["login_success"] = False

# --- Login UI ---
if not st.session_state["authenticated"]:
    st.title("🔐 Login Page")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username in users and users[username]["password"] == password:
            st.session_state["authenticated"] = True
            st.session_state["role"] = users[username]["role"]
            st.session_state["login_success"] = True
            st.success("Login successful!")
            st.experimental_rerun()
        else:
            st.error("Invalid username or password")

# --- Post-login Content ---
else:
    role = st.session_state["role"]

    if role == "Store Owner":
        st.success("Logged in as Store Owner ✅")

        @st.cache_resource
        def load_models():
            pca_path = "/Users/mukundhanmlu/Downloads/FINAL_streamlit/models_4_cluster/pca_model (1).pkl"
            kmeans_path = "/Users/mukundhanmlu/Downloads/FINAL_streamlit/models_4_cluster/kmeans_model (1).pkl"
            if not os.path.exists(pca_path) or not os.path.exists(kmeans_path):
                st.error("Model files not found. Please train and save PCA and KMeans models first.")
                st.stop()
            return joblib.load(pca_path), joblib.load(kmeans_path)

        @st.cache_data
        def load_data():
            data_path = "/Users/mukundhanmlu/Downloads/FINAL_streamlit/data/final.csv"
            if not os.path.exists(data_path):
                st.error("Data file not found. Please ensure 'final.csv' is available.")
                st.stop()
            return pd.read_csv(data_path)

        pca, kmeans = load_models()
        df = load_data()

        features = ['Income', 'Recency', 'Kidhome', 'Teenhome', 'Age', 'Children', 'TotalMembers', 'TotalSpent']
        scaler = StandardScaler()
        X = df[features]
        X_scaled = scaler.fit_transform(X)
        pca_result = pca.transform(X_scaled)

        df['PCA_KMeans_Cluster'] = kmeans.predict(pca_result)
        df['PCA1'] = pca_result[:, 0]
        df['PCA2'] = pca_result[:, 1]

        st.sidebar.title("📚 Navigation")
        page = st.sidebar.radio("Go to", ["🏠 Home", "📊 Dashboard", "🛍️ Store Guide", "📢 Channel Effectiveness"])

        if page == "🏠 Home":
            st.title("🧑‍💼 Customer Segmentation Dashboard")
            st.markdown("""
            Welcome to the interactive **Customer Segmentation Dashboard**!

            ### 📌 Project Overview:
            - **Objective:** To segment customers using **PCA** and **KMeans**.
            - **Goal:** Targeted marketing and personalization.
            - **Dataset:** Includes income, age, family size, etc.
            - **Tech Stack:** Python, Streamlit, Plotly, Seaborn.
            - **ML Models:** PCA and KMeans.

            ### 🧭 How to Use:
            1. **Dashboard:** View cluster metrics and insights.
            2. **Store Guide:** Explore customer personas.
            3. **Channel Effectiveness:** Analyze marketing channels.
            """)

        elif page == "📊 Dashboard":
            st.title("📊 Customer Segmentation Dashboard")

            col1, col2, col3 = st.columns(3)
            col1.metric("Total Customers", len(df))
            col2.metric("Average Income", f"${df['Income'].mean():.0f}")
            col3.metric("Average Total Spent", f"${df['TotalSpent'].mean():.0f}")

            cluster_counts = df['PCA_KMeans_Cluster'].value_counts().sort_index()
            fig_pie = px.pie(values=cluster_counts, names=[f"Cluster {i}" for i in cluster_counts.index], title="Customer Distribution by Cluster")
            st.plotly_chart(fig_pie)

            fig_scatter = px.scatter(df, x='PCA1', y='PCA2', color=df['PCA_KMeans_Cluster'].astype(str), title="Cluster Visualization (PCA)")
            st.plotly_chart(fig_scatter)

            st.subheader("📦 Interactive Boxplot: Feature Distribution by Segment")

            mock_df = pd.DataFrame({
                "Customer_ID": range(1, 101),
                "Total_Spend": np.random.normal(1500, 400, 100).astype(int),
                "Visit_Frequency": np.random.randint(1, 10, 100),
                "Loyalty_Score": np.random.uniform(0, 1, 100),
                "Uses_Discount": np.random.choice([0, 1], size=100),
                "Segment": np.random.choice(["🛍️ Frequent Shoppers", "💸 Budget Buyers", "🧊 Inactive Users"], 100)
            })

            view_option = st.radio("Choose your view:", ["🧠 Business View", "📈 Technical View"], horizontal=True)

            if view_option == "🧠 Business View":
                st.subheader("🔍 Smart Correlation Insights")
                insights = [
                    {"title": "High Spenders Are Frequent Visitors", "desc": "Customers who visit more frequently also tend to spend more per visit.", "score": "+0.76"},
                    {"title": "Loyalty Score vs Spending", "desc": "Higher loyalty scores are associated with higher total spending.", "score": "+0.68"},
                    {"title": "Discount Usage Increases Visits", "desc": "Customers who use discounts tend to visit more often.", "score": "+0.54"}
                ]
                for insight in insights:
                    with st.container():
                        st.markdown(f"### 🧠 {insight['title']}")
                        st.write(insight["desc"])
                        st.markdown(f"📊 **Correlation Strength:** `{insight['score']}`")
                        st.markdown("---")

                st.subheader("👥 Customer Segment Summary")
                segment_data = {
                    "🛍️ Frequent Shoppers": {
                        "Avg Purchase": "₹1500",
                        "Visits/Month": "7",
                        "Preferred Day": "Friday",
                        "Action": "Recommend premium offers"
                    },
                    "💰 Budget Buyers": {
                        "Avg Purchase": "₹600",
                        "Visits/Month": "3",
                        "Preferred Day": "Tuesday",
                        "Action": "Push combo deals"
                    },
                    "😴 Inactive Users": {
                        "Avg Purchase": "₹300",
                        "Visits/Month": "1",
                        "Preferred Day": "Sunday",
                        "Action": "Send SMS campaigns"
                    }
                }
                segment_choice = st.selectbox("Select a customer segment", list(segment_data.keys()))
                seg = segment_data[segment_choice]

                with st.container():
                    st.markdown(f"### {segment_choice} Summary")
                    st.write(f"**Average Purchase:** {seg['Avg Purchase']}")
                    st.write(f"**Visits per Month:** {seg['Visits/Month']}")
                    st.write(f"**Preferred Day:** {seg['Preferred Day']}")
                    st.success(f"📢 Recommended Strategy: {seg['Action']}")

            elif view_option == "📈 Technical View":
                st.subheader("📦 Interactive Boxplot: Total Spend by Segment")
                fig_box = px.box(
                    mock_df,
                    x="Segment",
                    y="Total_Spend",
                    color="Segment",
                    title="Total Spend Distribution per Segment (Interactive)",
                    points="all",
                    template="plotly",
                    width=900,
                    height=500
                )
                st.plotly_chart(fig_box, use_container_width=True)

            with st.expander("📈 Campaign Trend"):
                if 'Dt_Customer' in df.columns:
                    df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'], errors='coerce')
                    df['YearMonth'] = df['Dt_Customer'].dt.to_period('M')
                    campaign_cols = ['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5']
                    df['TotalAccepted'] = df[campaign_cols].sum(axis=1)
                    monthly_trend = df.groupby('YearMonth')['TotalAccepted'].sum().reset_index()
                    monthly_trend['YearMonth'] = monthly_trend['YearMonth'].astype(str)
                    fig_trend = px.line(monthly_trend, x='YearMonth', y='TotalAccepted', title="Campaign Responses Over Time")
                    st.plotly_chart(fig_trend)
                else:
                    st.warning("Dt_Customer column is missing or not parseable.")

            with st.expander("🔄 Campaign Journey Sankey"):
                campaign_cols = ['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5']
                if all(col in df.columns for col in campaign_cols):
                    df['Journey'] = df[campaign_cols].astype(str).agg('-'.join, axis=1)
                    journey_counts = df['Journey'].value_counts().reset_index()
                    journey_counts.columns = ['Journey', 'Count']
                    label_list = ["Start"] + campaign_cols
                    source, target, value = [], [], []

                    for _, row in journey_counts.iterrows():
                        steps = row['Journey'].split('-')
                        last_label = "Start"
                        for i, step in enumerate(steps):
                            if step == '1':
                                source.append(label_list.index(last_label))
                                last_label = campaign_cols[i]
                                target.append(label_list.index(last_label))
                                value.append(row['Count'])

                    fig = go.Figure(data=[go.Sankey(
                        node=dict(label=label_list, pad=15, thickness=20),
                        link=dict(source=source, target=target, value=value)
                    )])
                    fig.update_layout(title_text="Customer Campaign Journey", font_size=10)
                    st.plotly_chart(fig)
                else:
                    st.warning("Required campaign columns are missing.")


        # --- Store Guide ---
        elif page == "🛍️ Store Guide":
            st.title("🛍️ Store & Customer Personas")
            st.subheader("🧠 AI-Powered Customer Personas")
            for cluster in sorted(df['PCA_KMeans_Cluster'].unique()):
                st.markdown(f"### Cluster {cluster}")
                sub_df = df[df['PCA_KMeans_Cluster'] == cluster]
                mean_income = sub_df['Income'].mean()
                spent = sub_df['TotalSpent'].mean()
                kids = sub_df['Children'].mean()
                campaign_accepts = sub_df[['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5']].sum(axis=1).mean()

                if mean_income > 60000 and spent > 800:
                    persona = "Affluent Professionals — High earners who spend generously"
                elif spent > 700 and kids > 1:
                    persona = "Savvy Parents — Mid-income families hunting for value"
                elif spent < 500:
                    persona = "Low Engagement Group — Minimal purchase activity"
                else:
                    persona = "Budget-Conscious Shoppers — Selective spenders with occasional purchases"

                st.write(f"**Persona:** {persona}")
                st.write(f"Avg Income: ${mean_income:.0f}, Avg Spent: ${spent:.0f}, Avg Kids: {kids:.1f}, Avg Campaigns Accepted: {campaign_accepts:.1f}")

            st.subheader("📍 New Customer Cluster Predictor")
            with st.form("predict_form"):
                income = st.number_input("Income", 0, 200000, 50000)
                recency = st.slider("Recency", 0, 100, 30)
                kidhome = st.slider("Kids at Home", 0, 3, 0)
                teenhome = st.slider("Teens at Home", 0, 3, 0)
                age = st.slider("Age", 18, 90, 40)
                children = kidhome + teenhome
                totalmembers = children + 2
                totalspent = st.number_input("Total Spent", 0, 5000, 800)
                submitted = st.form_submit_button("Predict")

                if submitted:
                    new_data = np.array([[income, recency, kidhome, teenhome, age, children, totalmembers, totalspent]])
                    new_scaled = scaler.transform(new_data)
                    new_pca = pca.transform(new_scaled)
                    prediction = kmeans.predict(new_pca)[0]

                    # Get cluster-specific recommendations
                    cluster_info = {
                        0: {
                            "persona": "Affluent Professionals",
                            "promotion": "Promote premium wines, gold gifts, and exclusive offers on luxury products."
                        },
                        1: {
                            "persona": "Savvy Parents",
                            "promotion": "Highlight meat, fruit combos, and family bundles. Offer loyalty points and kids' deals."
                        },
                        2: {
                            "persona": "Low Engagement Group",
                            "promotion": "Send re-engagement emails, offer steep discounts on popular items, and free delivery."
                        },
                        3: {
                            "persona": "Budget-Conscious Shoppers",
                            "promotion": "Focus on value packs, sweet deals, and daily essentials with clear savings."
                        }
                        # Add more clusters as needed
                    }

                    info = cluster_info.get(prediction, {
                        "persona": "Unknown",
                        "promotion": "No specific recommendation available for this cluster."
                    })

                    st.success(f"Predicted Cluster: {prediction} — {info['persona']}")
                    st.markdown(f"**📢 Recommended Promotion Strategy:** {info['promotion']}")


            # --- Cluster Deep Dive ---
            st.subheader("🔍 Cluster Deep Dive")

            selected = st.selectbox("Select a Cluster", sorted(df['PCA_KMeans_Cluster'].unique()), key="deep_dive_cluster")
            sub_df = df[df['PCA_KMeans_Cluster'] == selected]

            st.write(f"**Cluster {selected} - Summary**")
            st.dataframe(sub_df[features].describe().T.round(2))

            st.write("### Product Category Insights")
            product_cols = ['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
            mean_purchases = sub_df[product_cols].mean().sort_values(ascending=False)
            st.bar_chart(mean_purchases)

            if st.checkbox("Compare with another cluster"):
                other = st.selectbox("Select comparison cluster", [c for c in df['PCA_KMeans_Cluster'].unique() if c != selected], key="compare_cluster")
                other_df = df[df['PCA_KMeans_Cluster'] == other]
                compare_df = pd.DataFrame({
                    f"Cluster {selected}": sub_df[features].mean(),
                    f"Cluster {other}": other_df[features].mean()
                })
                st.write("### Comparison Table")
                st.dataframe(compare_df.round(2))


        # --- Channel Effectiveness ---
        elif page == "📢 Channel Effectiveness":
            st.title("📢 Marketing Channel Effectiveness")
            channel_cols = ['NumWebPurchases', 'NumStorePurchases', 'NumWebVisitsMonth', 'Response']
            if not all(col in df.columns for col in channel_cols):
                st.error("Required marketing channel columns are missing in the dataset.")
                st.stop()

            st.subheader("📈 Purchase Channel Breakdown")
            purchase_means = df[['NumWebPurchases', 'NumStorePurchases']].mean().sort_values(ascending=False)
            st.bar_chart(purchase_means)

            st.subheader("🌐 Web Visit vs. Web Purchase")
            web_conversion_rate = (df['NumWebPurchases'].sum() / df['NumWebVisitsMonth'].sum()) * 100
            st.metric("Web Conversion Rate", f"{web_conversion_rate:.2f}%")

            st.subheader("✅ Campaign Response Rate")
            response_rate = (df['Response'].sum() / len(df)) * 100
            st.metric("Campaign Conversion Rate", f"{response_rate:.2f}%")

            st.subheader("📊 Conversion Funnel")
            funnel_data = pd.DataFrame({
                "Stage": ["Web Visits", "Web Purchases", "Campaign Responses"],
                "Count": [
                    df['NumWebVisitsMonth'].sum(),
                    df['NumWebPurchases'].sum(),
                    df['Response'].sum()
                ]
            })
            fig_funnel = px.funnel(funnel_data, x="Count", y="Stage", title="Marketing Funnel")
            st.plotly_chart(fig_funnel)

            st.subheader("🔎 Channel Activity by Cluster")
            channel_cluster_avg = df.groupby('PCA_KMeans_Cluster')[channel_cols].mean()
            st.dataframe(channel_cluster_avg.round(2))


    elif role == "Customer":
        st.success("Logged in as Customer 🛍️")

        # Store Description
        st.header("🏪 Store Description")
        st.markdown("""
        Our store offers a wide variety of high-quality products curated to meet the needs of modern shoppers.  
        With a focus on convenience, quality, and affordability, we ensure that every customer enjoys a seamless shopping experience.
        """)

        # Product Details
        st.header("📦 Products Available")
        product_data = {
            "Category": ["Wines", "Fruits", "Meat Products", "Fish Products", "Sweet Products", "Gold Products"],
            "Description": [
                "A variety of premium wines from around the world",
                "Fresh and organic fruits sourced locally",
                "High-quality meat products for daily cooking",
                "Freshwater and saltwater fish products",
                "Delicious sweets and confectioneries",
                "Exclusive gold gift items and jewelry"
            ],
            "Price Range": ["₹500-₹3000", "₹50-₹500", "₹200-₹1500", "₹150-₹1200", "₹100-₹800", "₹1000-₹5000"]
        }
        st.dataframe(pd.DataFrame(product_data))

        # Store Offers
        st.header("🎉 Current Offers")
        st.markdown("""
        - 💥 **Buy 2 Get 1 Free** on Sweet Products  
        - 🥂 **10% Discount** on Wine purchases above ₹2000  
        - 🐟 **Flat ₹100 OFF** on Fish Products every Friday  
        - 🛒 **Free Delivery** on orders above ₹1500  
        """)
