import streamlit as st
import pandas as pd
import mysql.connector
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from PIL import Image
import os

# --------------------------
# DATABASE UTILITY FUNCTIONS
# --------------------------
@st.cache_resource
def get_connection():
    """Return a cached MySQL connection"""
    conn = mysql.connector.connect(
        host="localhost",
        user="root",             
        password="GeoRunz",  
        database="sakila"
    )
    return conn

@st.cache_data
def run_query(query: str) -> pd.DataFrame:
    """Execute query using cached connection and return DataFrame"""
    conn = get_connection()
    df = pd.read_sql(query, conn)
    return df

# --------------------------
# SIDEBAR NAVIGATION
# --------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "EDA", "Prediction"])

# --------------------------
# PAGE 1: HOME
# --------------------------
if page == "Home":
    st.title("🎬 Welcome to the Sakila Movie Rental Dashboard! Grab some popcorn!")
    
    img_path = os.path.join(os.path.dirname(__file__), "metropolis.jpeg")
    try:
        img = Image.open(img_path)
        st.image(img, use_container_width=True)
    except FileNotFoundError:
        st.error(f"Image 'metropolis.jpeg' not found at {img_path}")
    
    st.markdown("""
    Explore rental trends, revenue insights, and customer segmentation.  
    Navigate through the pages to perform **interactive EDA** and **customer prediction simulations**!  
    """)

# --------------------------
# PAGE 2: EDA
# --------------------------
elif page == "EDA":
    st.title("📊 Exploratory Data Analysis (EDA)")

    # --- Fetch Data ---
    movies_query = "SELECT f.film_id, f.title, f.rating, f.length FROM film f;"
    movies_df = run_query(movies_query)

    rental_query = """
    SELECT r.rental_id, r.customer_id, i.film_id
    FROM rental r
    JOIN inventory i ON r.inventory_id = i.inventory_id
    """
    rentals_df = run_query(rental_query)

    payment_query = """
    SELECT c.customer_id, SUM(p.amount) AS total_spent
    FROM customer c
    JOIN payment p ON c.customer_id = p.customer_id
    GROUP BY c.customer_id
    """
    payments_df = run_query(payment_query)

    # Merge rentals with movie info
    rental_movies = rentals_df.merge(movies_df, on="film_id", how="left")

    # --- KPIs ---
    st.subheader("Key Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Rentals", rentals_df.shape[0])
    col2.metric("Total Revenue ($)", payments_df['total_spent'].sum())
    col3.metric("Average Rental per Customer ($)", round(payments_df['total_spent'].mean(),2))

    # --- Filters ---
    st.subheader("Filters")
    top_n = st.slider("Select Top N Movies", min_value=5, max_value=20, value=10)
    ratings_filter = st.multiselect(
        "Select Ratings",
        options=movies_df['rating'].unique().tolist(),
        default=movies_df['rating'].unique().tolist()
    )

    filtered_movies = movies_df[movies_df['rating'].isin(ratings_filter)]
    rental_filtered = rental_movies[rental_movies['rating'].isin(ratings_filter)]

    # --- Top Movies Chart ---
    st.subheader(f"Top {top_n} Movies by Rentals")
    top_movies_count = rental_filtered.groupby('title').size().reset_index(name='times_rented')
    top_movies_count = top_movies_count.sort_values('times_rented', ascending=False).head(top_n)
    st.bar_chart(top_movies_count.set_index('title'))

    # --- Rentals by Rating ---
    st.subheader("Rentals by Rating")
    rating_counts = rental_filtered.groupby('rating').size().reset_index(name='count')
    st.bar_chart(rating_counts.set_index('rating'))

    # --- Revenue per Customer ---
    st.subheader("Revenue per Customer")
    st.dataframe(payments_df.sort_values('total_spent', ascending=False).head(10))

# --------------------------
# PAGE 3: PREDICTION
# --------------------------
elif page == "Prediction":
    st.title("🤖 Customer Segmentation Prediction")

    # --- Load Customer Features ---
    customer_query = """
    SELECT c.customer_id, COUNT(r.rental_id) AS total_rentals, AVG(f.length) AS avg_movie_length, SUM(p.amount) AS total_spent
    FROM customer c
    JOIN rental r ON c.customer_id = r.customer_id
    JOIN inventory i ON r.inventory_id = i.inventory_id
    JOIN film f ON i.film_id = f.film_id
    JOIN payment p ON r.rental_id = p.rental_id
    GROUP BY c.customer_id;
    """
    customer_features = run_query(customer_query)

    # --- Train KMeans ---
    features = customer_features[['total_rentals', 'avg_movie_length', 'total_spent']]
    kmeans = KMeans(n_clusters=3, random_state=42)
    customer_features['cluster'] = kmeans.fit_predict(features)

    # --- Customer Selector ---
    st.subheader("Select a Customer to Simulate")
    customer_id = st.selectbox("Customer ID", customer_features['customer_id'].tolist())
    cust_row = customer_features[customer_features['customer_id'] == customer_id].copy()

    # --- Interactive Feature Sliders ---
    st.markdown("#### Adjust Features to See Cluster Prediction Change")
    total_rentals = st.slider("Total Rentals", 0, int(customer_features['total_rentals'].max()), int(cust_row['total_rentals'].values[0]))
    avg_length = st.slider("Average Movie Length (mins)", 0, int(customer_features['avg_movie_length'].max())+10, int(cust_row['avg_movie_length'].values[0]))
    total_spent = st.slider("Total Spent ($)", 0, int(customer_features['total_spent'].max())+50, int(cust_row['total_spent'].values[0]))

    # Predict cluster for adjusted values
    import numpy as np
    adjusted_features = np.array([[total_rentals, avg_length, total_spent]])
    cluster = kmeans.predict(adjusted_features)[0]
    st.success(f"Predicted Customer Segment: Cluster {cluster}")

    # --- Cluster Visualization ---
    st.subheader("Customer Segmentation Plot")
    fig, ax = plt.subplots()
    sns.scatterplot(
        data=customer_features,
        x='total_rentals',
        y='total_spent',
        hue='cluster',
        palette='Set2',
        ax=ax,
        legend='full'
    )
    ax.scatter(total_rentals, total_spent, color='red', s=150, label='Selected Customer / Adjusted')
    ax.set_xlabel("Total Rentals")
    ax.set_ylabel("Total Spent ($)")
    ax.legend()
    st.pyplot(fig)
