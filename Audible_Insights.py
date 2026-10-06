
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity


# 1. PAGE CONFIGURATION

st.set_page_config(
    page_title="Audible Insights",
    page_icon="📚",
    layout="wide"
)


# 2. PROJECT PATHS


# Current file:
# Audible_Insights/app.py/Audible_Insights.py

APP_DIR = Path(__file__).resolve().parent

# Main project folder:
# Audible_Insights/
PROJECT_DIR = APP_DIR.parent

# Models folder:
MODELS_DIR = PROJECT_DIR / "models"


# ============================================================
# 3. LOAD SAVED MODELS AND DATA
# ============================================================

@st.cache_resource
def load_models():

    tfidf = joblib.load(
        MODELS_DIR / "tfidf_vectorizer.pkl"
    )

    tfidf_matrix = joblib.load(
        MODELS_DIR / "tfidf_matrix.pkl"
    )

    kmeans = joblib.load(
        MODELS_DIR / "kmeans_model.pkl"
    )

    books_df = pd.read_pickle(
        MODELS_DIR / "books_data.pkl"
    )

    return tfidf, tfidf_matrix, kmeans, books_df


try:

    tfidf, tfidf_matrix, kmeans, books_df = load_models()

except FileNotFoundError as e:

    st.error("Model files were not found.")

    st.write("Expected models folder:")
    st.code(str(MODELS_DIR))

    st.write(
        "Please make sure these files are available:"
    )

    st.code(
        """
tfidf_vectorizer.pkl
tfidf_matrix.pkl
kmeans_model.pkl
books_data.pkl
        """
    )

    st.stop()


# ============================================================
# 4. DATA PREPARATION
# ============================================================

# Make sure index matches TF-IDF matrix
books_df = books_df.reset_index(drop=True)

# Numeric columns
books_df["Rating"] = pd.to_numeric(
    books_df["Rating"],
    errors="coerce"
)

books_df["Number of Reviews"] = pd.to_numeric(
    books_df["Number of Reviews"],
    errors="coerce"
)

books_df["Price"] = pd.to_numeric(
    books_df["Price"],
    errors="coerce"
)

books_df["Rating"] = books_df["Rating"].fillna(0)

books_df["Number of Reviews"] = (
    books_df["Number of Reviews"].fillna(0)
)

books_df["Price"] = books_df["Price"].fillna(0)

books_df["Genre"] = books_df["Genre"].fillna("Unknown")

books_df["Author"] = books_df["Author"].fillna("Unknown")

books_df["Description"] = (
    books_df["Description"].fillna("")
)


# ============================================================
# 5. CONTENT-BASED RECOMMENDATION FUNCTION
# ============================================================

def content_recommendations(book_name, n=5):

    matches = books_df[
        books_df["Book Name"].str.contains(
            book_name,
            case=False,
            na=False,
            regex=False
        )
    ]

    if matches.empty:
        return None, None

    # First matching book
    idx = matches.index[0]

    actual_book = books_df.loc[idx, "Book Name"]

    # Calculate cosine similarity
    similarity_scores = cosine_similarity(
        tfidf_matrix[idx],
        tfidf_matrix
    ).flatten()

    # Sort highest similarity first
    similar_indices = similarity_scores.argsort()[::-1]

    # Remove selected book itself
    similar_indices = [
        i
        for i in similar_indices
        if i != idx
    ][:n]

    result = books_df.iloc[similar_indices][
        [
            "Book Name",
            "Author",
            "Genre",
            "Rating",
            "Number of Reviews"
        ]
    ].copy()

    result["Similarity Score"] = (
        similarity_scores[similar_indices]
    )

    result["Similarity Score"] = (
        result["Similarity Score"].round(3)
    )

    return actual_book, result


# ============================================================
# 6. CLUSTER-BASED RECOMMENDATION FUNCTION
# ============================================================

def cluster_recommendations(book_name, n=5):

    matches = books_df[
        books_df["Book Name"].str.contains(
            book_name,
            case=False,
            na=False,
            regex=False
        )
    ]

    if matches.empty:
        return None, None, None

    idx = matches.index[0]

    actual_book = books_df.loc[idx, "Book Name"]

    cluster_id = books_df.loc[idx, "Cluster"]

    recommendations = books_df[
        (books_df["Cluster"] == cluster_id) &
        (books_df.index != idx)
    ].copy()

    recommendations = recommendations.sort_values(
        by=[
            "Rating",
            "Number of Reviews"
        ],
        ascending=[
            False,
            False
        ]
    )

    result = recommendations[
        [
            "Book Name",
            "Author",
            "Genre",
            "Rating",
            "Number of Reviews",
            "Cluster"
        ]
    ].head(n)

    return actual_book, cluster_id, result


# ============================================================
# 7. HYBRID RECOMMENDATION FUNCTION
# ============================================================

def hybrid_recommendations(book_name, n=5):

    matches = books_df[
        books_df["Book Name"].str.contains(
            book_name,
            case=False,
            na=False,
            regex=False
        )
    ]

    if matches.empty:
        return None, None

    idx = matches.index[0]

    actual_book = books_df.loc[idx, "Book Name"]

    cluster_id = books_df.loc[idx, "Cluster"]

    # Cosine similarity
    similarity_scores = cosine_similarity(
        tfidf_matrix[idx],
        tfidf_matrix
    ).flatten()

    result = books_df.copy()

    result["Similarity Score"] = similarity_scores

    # Same cluster
    result = result[
        (result["Cluster"] == cluster_id) &
        (result.index != idx)
    ].copy()

    # Normalize rating
    result["Rating Score"] = (
        result["Rating"].clip(0, 5) / 5
    )

    # Hybrid score
    result["Hybrid Score"] = (
        0.80 * result["Similarity Score"] +
        0.20 * result["Rating Score"]
    )

    result = result.sort_values(
        "Hybrid Score",
        ascending=False
    )

    final_result = result[
        [
            "Book Name",
            "Author",
            "Genre",
            "Rating",
            "Similarity Score",
            "Hybrid Score"
        ]
    ].head(n).copy()

    final_result["Similarity Score"] = (
        final_result["Similarity Score"].round(3)
    )

    final_result["Hybrid Score"] = (
        final_result["Hybrid Score"].round(3)
    )

    return actual_book, final_result


# ============================================================
# 8. SIDEBAR
# ============================================================

st.sidebar.title("📚 Audible Insights")

page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Book Recommendation",
        "Genre Recommendation",
        "Book Search",
        "EDA Dashboard"
    ]
)


# ============================================================
# 9. HOME PAGE
# ============================================================

if page == "Home":

    st.title("📚 Audible Insights")

    st.subheader(
        "Intelligent Book Recommendation System"
    )

    st.write(
        """
        Audible Insights uses Natural Language Processing,
        TF-IDF, Cosine Similarity and K-Means Clustering
        to recommend books based on their content and
        characteristics.
        """
    )

    st.divider()

    # Metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "📚 Total Books",
            f"{len(books_df):,}"
        )

    with col2:

        st.metric(
            "✍️ Total Authors",
            f"{books_df['Author'].nunique():,}"
        )

    with col3:

        st.metric(
            "⭐ Average Rating",
            round(
                books_df["Rating"].mean(),
                2
            )
        )

    with col4:

        genre_count = books_df[
            books_df["Genre"] != "Unknown"
        ]["Genre"].nunique()

        st.metric(
            "📖 Genres",
            genre_count
        )

    st.divider()

    st.subheader("📖 Sample Books")

    st.dataframe(
        books_df[
            [
                "Book Name",
                "Author",
                "Genre",
                "Rating",
                "Price"
            ]
        ].head(20),
        use_container_width=True
    )


# ============================================================
# 10. BOOK RECOMMENDATION PAGE
# ============================================================

elif page == "Book Recommendation":

    st.title("🔍 Book Recommendation")

    st.write(
        "Select a book and discover similar books."
    )

    # Select book
    book_list = sorted(
        books_df["Book Name"]
        .dropna()
        .unique()
        .tolist()
    )

    selected_book = st.selectbox(
        "Select a Book",
        book_list
    )

    recommendation_type = st.radio(
        "Recommendation Method",
        [
            "Content-Based",
            "Cluster-Based",
            "Hybrid"
        ],
        horizontal=True
    )

    number_of_books = st.slider(
        "Number of Recommendations",
        min_value=3,
        max_value=10,
        value=5
    )

    if st.button(
        "Get Recommendations",
        type="primary"
    ):

        # ---------------------------------
        # Content-Based
        # ---------------------------------

        if recommendation_type == "Content-Based":

            actual_book, recommendations = (
                content_recommendations(
                    selected_book,
                    number_of_books
                )
            )

            if recommendations is None:

                st.warning(
                    "Book not found."
                )

            else:

                st.success(
                    f"Recommendations for: {actual_book}"
                )

                st.dataframe(
                    recommendations,
                    use_container_width=True,
                    hide_index=True
                )

        # ---------------------------------
        # Cluster-Based
        # ---------------------------------

        elif recommendation_type == "Cluster-Based":

            actual_book, cluster_id, recommendations = (
                cluster_recommendations(
                    selected_book,
                    number_of_books
                )
            )

            if recommendations is None:

                st.warning(
                    "Book not found."
                )

            else:

                st.success(
                    f"Recommendations for: {actual_book}"
                )

                st.info(
                    f"Book Cluster: {cluster_id}"
                )

                st.dataframe(
                    recommendations,
                    use_container_width=True,
                    hide_index=True
                )

        # ---------------------------------
        # Hybrid
        # ---------------------------------

        else:

            actual_book, recommendations = (
                hybrid_recommendations(
                    selected_book,
                    number_of_books
                )
            )

            if recommendations is None:

                st.warning(
                    "Book not found."
                )

            else:

                st.success(
                    f"Hybrid Recommendations for: {actual_book}"
                )

                st.dataframe(
                    recommendations,
                    use_container_width=True,
                    hide_index=True
                )


# ============================================================
# 11. GENRE RECOMMENDATION PAGE
# ============================================================

elif page == "Genre Recommendation":

    st.title("📖 Genre Recommendations")

    valid_genres = books_df[
        books_df["Genre"] != "Unknown"
    ]["Genre"].dropna().unique()

    valid_genres = sorted(valid_genres.tolist())

    if len(valid_genres) == 0:

        st.warning(
            "No genre information is available."
        )

    else:

        selected_genre = st.selectbox(
            "Select Genre",
            valid_genres
        )

        number_of_books = st.slider(
            "Number of Books",
            min_value=5,
            max_value=20,
            value=10
        )

        genre_books = books_df[
            books_df["Genre"] == selected_genre
        ].copy()

        genre_books = genre_books.sort_values(
            by=[
                "Rating",
                "Number of Reviews"
            ],
            ascending=[
                False,
                False
            ]
        )

        st.subheader(
            f"Top Books in {selected_genre}"
        )

        st.dataframe(
            genre_books[
                [
                    "Book Name",
                    "Author",
                    "Rating",
                    "Number of Reviews",
                    "Price"
                ]
            ].head(number_of_books),
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# 12. BOOK SEARCH PAGE
# ============================================================

# ============================================================
# BOOK SEARCH PAGE
# ============================================================

elif page == "Book Search":

    st.title("🔎 Search & Explore Books")

    st.write(
        "Search or select any book available in the Audible dataset."
    )

    # --------------------------------------------------------
    # Prepare all books
    # --------------------------------------------------------

    available_books = (
        books_df["Book Name"]
        .dropna()
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    # --------------------------------------------------------
    # Searchable dropdown
    # --------------------------------------------------------

    selected_book = st.selectbox(
        "Search or Select a Book",
        options=["Select a book..."] + available_books,
        index=0,
        help="Type a book name to search or choose from the list."
    )

    # --------------------------------------------------------
    # Show selected book details
    # --------------------------------------------------------

    if selected_book != "Select a book...":

        selected_data = books_df[
            books_df["Book Name"] == selected_book
        ]

        if not selected_data.empty:

            book = selected_data.iloc[0]

            st.success(f"📚 Selected Book: {selected_book}")

            st.divider()

            # ------------------------------------------------
            # Main Book Information
            # ------------------------------------------------

            st.subheader("📖 Book Details")

            col1, col2 = st.columns(2)

            with col1:

                st.write("**Book Name:**")
                st.write(book["Book Name"])

                st.write("**Author:**")
                st.write(book["Author"])

                st.write("**Genre:**")
                st.write(book["Genre"])

            with col2:

                st.write("**Rating:**")
                st.write(f"⭐ {book['Rating']} / 5")

                st.write("**Number of Reviews:**")
                st.write(f"{book['Number of Reviews']:,.0f}")

                st.write("**Price:**")
                st.write(f"₹ {book['Price']:,.2f}")

            # ------------------------------------------------
            # Listening Time
            # ------------------------------------------------

            if "Listening Time" in books_df.columns:

                st.write("**🎧 Listening Time:**")

                st.write(
                    book["Listening Time"]
                )

            # ------------------------------------------------
            # Rank
            # ------------------------------------------------

            if "Rank" in books_df.columns:

                st.write("**🏆 Rank:**")

                st.write(
                    book["Rank"]
                )

            # ------------------------------------------------
            # Description
            # ------------------------------------------------

            st.divider()

            st.subheader("📝 Description")

            description = book["Description"]

            if (
                pd.notna(description)
                and str(description).strip() != ""
            ):

                st.write(description)

            else:

                st.info(
                    "Description is not available for this book."
                )

            # ------------------------------------------------
            # Similar Books
            # ------------------------------------------------

            st.divider()

            st.subheader("📚 Similar Books")

            st.write(
                "You can also find books similar to the selected book."
            )

            if st.button(
                "Find Similar Books",
                type="primary"
            ):

                actual_book, recommendations = (
                    content_recommendations(
                        selected_book,
                        5
                    )
                )

                if recommendations is not None:

                    st.success(
                        f"Books similar to {actual_book}"
                    )

                    st.dataframe(
                        recommendations,
                        use_container_width=True,
                        hide_index=True
                    )

                else:

                    st.warning(
                        "Similar books could not be found."
                    )

# ============================================================
# EDA DASHBOARD - 10 GRAPHS
# ============================================================

# ============================================================
# EDA DASHBOARD
# ============================================================

elif page == "EDA Dashboard":

    st.title("📊 EDA Dashboard")
    st.write("Exploratory Data Analysis of the Audible Books Dataset")

    # ========================================================
    # 1. RATING DISTRIBUTION
    # ========================================================

    st.subheader("1. Book Rating Distribution")

    fig, ax = plt.subplots(figsize=(10, 5))

    sns.histplot(
        books_df["Rating"],
        bins=20,
        kde=True,
        ax=ax
    )

    ax.set_title("Book Rating Distribution")
    ax.set_xlabel("Rating")
    ax.set_ylabel("Number of Books")

    st.pyplot(fig)
    plt.close(fig)


    # ========================================================
    # 2. TOP 10 AUTHORS BY NUMBER OF BOOKS
    # ========================================================

    st.subheader("2. Top 10 Authors by Number of Books")

    top_authors = (
        books_df["Author"]
        .value_counts()
        .head(10)
    )

    fig, ax = plt.subplots(figsize=(10, 5))

    sns.barplot(
        x=top_authors.values,
        y=top_authors.index,
        ax=ax
    )

    ax.set_title("Top 10 Authors by Number of Books")
    ax.set_xlabel("Number of Books")
    ax.set_ylabel("Author")

    plt.tight_layout()

    st.pyplot(fig)
    plt.close(fig)


    # ========================================================
    # 3. AVERAGE RATING BY AUTHOR
    # ========================================================

    st.subheader("3. Top 10 Authors by Average Rating")

    author_stats = (
        books_df.groupby("Author")
        .agg(
            Average_Rating=("Rating", "mean"),
            Book_Count=("Book Name", "count")
        )
    )

    top_author_ratings = (
        author_stats[
            author_stats["Book_Count"] >= 3
        ]
        .sort_values(
            "Average_Rating",
            ascending=False
        )
        .head(10)
    )

    fig, ax = plt.subplots(figsize=(10, 5))

    sns.barplot(
        x=top_author_ratings["Average_Rating"],
        y=top_author_ratings.index,
        ax=ax
    )

    ax.set_title("Top 10 Authors by Average Rating")
    ax.set_xlabel("Average Rating")
    ax.set_ylabel("Author")

    plt.tight_layout()

    st.pyplot(fig)
    plt.close(fig)


    # ========================================================
    # 4. TOP 10 GENRES
    # ========================================================

    st.subheader("4. Top 10 Genres")

    genre_data = books_df[
        books_df["Genre"] != "Unknown"
    ]

    top_genres = (
        genre_data["Genre"]
        .value_counts()
        .head(10)
    )

    fig, ax = plt.subplots(figsize=(10, 5))

    sns.barplot(
        x=top_genres.values,
        y=top_genres.index,
        ax=ax
    )

    ax.set_title("Top 10 Genres")
    ax.set_xlabel("Number of Books")
    ax.set_ylabel("Genre")

    plt.tight_layout()

    st.pyplot(fig)
    plt.close(fig)


    # ========================================================
    # 5. RATING VS NUMBER OF REVIEWS
    # ========================================================

    st.subheader("5. Rating vs Number of Reviews")

    fig, ax = plt.subplots(figsize=(10, 5))

    sns.scatterplot(
        data=books_df,
        x="Rating",
        y="Number of Reviews",
        alpha=0.5,
        ax=ax
    )

    ax.set_title("Rating vs Number of Reviews")
    ax.set_xlabel("Rating")
    ax.set_ylabel("Number of Reviews")

    plt.tight_layout()

    st.pyplot(fig)
    plt.close(fig)


    # ========================================================
    # 6. TOP RATED BOOKS
    # ========================================================

    st.subheader("6. Top 10 Rated Books")

    top_rated_books = (
        books_df[
            books_df["Number of Reviews"] >= 100
        ]
        .sort_values(
            by=["Rating", "Number of Reviews"],
            ascending=[False, False]
        )
        [
            [
                "Book Name",
                "Author",
                "Rating",
                "Number of Reviews"
            ]
        ]
        .head(10)
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.barplot(
        data=top_rated_books,
        x="Rating",
        y="Book Name",
        ax=ax
    )

    ax.set_title("Top 10 Rated Books")
    ax.set_xlabel("Rating")
    ax.set_ylabel("Book Name")
    ax.set_xlim(0, 5)

    plt.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

    st.dataframe(
        top_rated_books,
        use_container_width=True,
        hide_index=True
    )


    # ========================================================
    # 7. HIDDEN GEMS
    # ========================================================

    st.subheader("7. Top 10 Hidden Gem Books")

    high_rating = (
        books_df["Rating"]
        .quantile(0.75)
    )

    positive_reviews = books_df[
        books_df["Number of Reviews"] > 0
    ]["Number of Reviews"]

    low_reviews = (
        positive_reviews
        .quantile(0.25)
    )

    hidden_gems = books_df[
        (books_df["Rating"] >= high_rating)
        &
        (books_df["Number of Reviews"] > 0)
        &
        (books_df["Number of Reviews"] <= low_reviews)
    ]

    hidden_gems = (
        hidden_gems
        .sort_values(
            ["Rating", "Number of Reviews"],
            ascending=[False, False]
        )
        [
            [
                "Book Name",
                "Author",
                "Rating",
                "Number of Reviews",
                "Genre"
            ]
        ]
        .head(10)
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.barplot(
        data=hidden_gems,
        x="Rating",
        y="Book Name",
        ax=ax
    )

    ax.set_title("Top 10 Hidden Gem Books")
    ax.set_xlabel("Rating")
    ax.set_ylabel("Book Name")
    ax.set_xlim(0, 5)

    plt.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

    st.dataframe(
        hidden_gems,
        use_container_width=True,
        hide_index=True
    )


    # ========================================================
    # 8. PRICE DISTRIBUTION
    # ========================================================

    st.subheader("8. Book Price Distribution")

    fig, ax = plt.subplots(figsize=(10, 5))

    sns.histplot(
        books_df["Price"],
        bins=30,
        kde=True,
        ax=ax
    )

    ax.set_title("Book Price Distribution")
    ax.set_xlabel("Price")
    ax.set_ylabel("Number of Books")

    plt.tight_layout()

    st.pyplot(fig)
    plt.close(fig)


    # ========================================================
    # 9. PRICE VS RATING
    # ========================================================

    st.subheader("9. Price vs Rating")

    fig, ax = plt.subplots(figsize=(10, 5))

    sns.scatterplot(
        data=books_df,
        x="Price",
        y="Rating",
        alpha=0.5,
        ax=ax
    )

    ax.set_title("Price vs Rating")
    ax.set_xlabel("Price")
    ax.set_ylabel("Rating")

    plt.tight_layout()

    st.pyplot(fig)
    plt.close(fig)
# ============================================================
# 14. FOOTER
# ============================================================

st.divider()

st.caption(
    "📚 Audible Insights | Intelligent Book Recommendation System"
)