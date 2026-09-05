import streamlit as st
import pandas as pd
import ast

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity


# -----------------------------
# Load Dataset
# -----------------------------

data = pd.read_csv("audible_cleaned_data.csv")

data["Genres"] = data["Genres"].apply(
    lambda x: ast.literal_eval(x) if pd.notna(x) else []
)


# -----------------------------
# Sidebar Navigation
# -----------------------------

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go to",
    ["Home", "Recommendations", "EDA"]
)


# -----------------------------
# Home Page
# -----------------------------

if page == "Home":

    st.title("Audible Insights: Intelligent Book Recommendations")

    st.write(
        "This application recommends books based on user preferences."
    )

    st.write("Number of books:", len(data))

    st.subheader("Dataset Preview")

    st.dataframe(data.head())


# -----------------------------
# Recommendations Page
# -----------------------------

elif page == "Recommendations":

    st.title("Book Recommendation System")

    st.write(
        "Select your preferences to discover similar books."
    )


    # -----------------------------
    # Genre Preference
    # -----------------------------

    all_genres = sorted(
        set(
            genre
            for genres in data["Genres"]
            for genre in genres
        )
    )

    selected_genre = st.selectbox(
        "Choose your favourite genre:",
        all_genres
    )

    st.write(
        "You selected:",
        selected_genre
    )


    # -----------------------------
    # Filter Books by Genre
    # -----------------------------

    genre_books = data[
        data["Genres"].apply(
            lambda genres: selected_genre in genres
        )
    ].copy()

    genre_books = genre_books.sort_values(
        by="Rating",
        ascending=False
    )

    st.write(
        "Books found in this genre:",
        len(genre_books)
    )

    st.subheader("Recommended Books")

    st.dataframe(
        genre_books[
            [
                "Book Name",
                "Author",
                "Rating",
                "Number of Reviews"
            ]
        ].head(10)
    )


    # -----------------------------
    # Book-Based Recommendations
    # -----------------------------

    st.subheader("Find Similar Books")

    book_data = data[
        data["Genres"].apply(len) > 0
    ].reset_index(drop=True)

    book_names = sorted(
        book_data["Book Name"].dropna().unique()
    )

    selected_book = st.selectbox(
        "Choose a book you like:",
        book_names
    )

    st.write(
        "Selected book:",
        selected_book
    )


    # -----------------------------
    # Convert Genres to Numerical Form
    # -----------------------------

    mlb = MultiLabelBinarizer()

    genre_matrix = mlb.fit_transform(
        book_data["Genres"]
    )


    # -----------------------------
    # Calculate Cosine Similarity
    # -----------------------------

    similarity_matrix = cosine_similarity(
        genre_matrix
    )


    selected_index = book_data[
        book_data["Book Name"] == selected_book
    ].index[0]


    similarity_scores = list(
        enumerate(
            similarity_matrix[selected_index]
        )
    )

    similarity_scores = sorted(
        similarity_scores,
        key=lambda x: x[1],
        reverse=True
    )


    # -----------------------------
    # Top 5 Similar Books
    # -----------------------------

    top_books = similarity_scores[1:6]

    recommended_indices = [
        index
        for index, score in top_books
    ]

    recommendations = book_data.iloc[
        recommended_indices
    ].copy()

    recommendations["Similarity Score"] = [
        round(score, 2)
        for index, score in top_books
    ]


    # -----------------------------
    # Display Similar Books
    # -----------------------------

    st.subheader(
        "Similar Book Recommendations"
    )

    st.dataframe(
        recommendations[
            [
                "Book Name",
                "Author",
                "Rating",
                "Genres",
                "Similarity Score"
            ]
        ]
    )


# -----------------------------
# EDA Page
# -----------------------------

elif page == "EDA":

    st.title("Exploratory Data Analysis")

    st.write(
        "Select a question level to explore the analysis."
    )

    eda_level = st.selectbox(
        "Choose analysis level:",
        [
            "Easy Level",
            "Medium Level",
            "Scenario Based"
        ]
    )


    # =========================================================
    # EASY LEVEL
    # =========================================================

    if eda_level == "Easy Level":

        st.header("Easy Level Questions")


        # -----------------------------------------------------
        # Question 1
        # -----------------------------------------------------

        st.subheader(
            "1. What are the most popular genres in the dataset?"
        )

        genre_counts = (
            data.explode("Genres")["Genres"]
            .value_counts()
            .head(10)
        )

        st.bar_chart(genre_counts)

        st.write(
            "Answer:",
            genre_counts.index[0],
            "is the most popular genre with",
            genre_counts.iloc[0],
            "books."
        )


        # -----------------------------------------------------
        # Question 2
        # -----------------------------------------------------

        st.subheader(
            "2. Which authors have the highest-rated books?"
        )

        author_ratings = (
            data.dropna(subset=["Rating"])
            .groupby("Author")
            .agg(
                Average_Rating=("Rating", "mean"),
                Number_of_Books=("Book Name", "count")
            )
            .reset_index()
        )

        # Keep authors with at least 5 books
        author_ratings = author_ratings[
            author_ratings["Number_of_Books"] >= 5
        ]

        top_authors = (
            author_ratings
            .sort_values(
                "Average_Rating",
                ascending=False
            )
            .head(10)
        )

        st.bar_chart(
            top_authors.set_index("Author")["Average_Rating"]
        )

        st.dataframe(top_authors)

        best_author = top_authors.iloc[0]

        st.write(
            "Answer:",
            best_author["Author"],
            "has the highest average rating of",
            round(best_author["Average_Rating"], 2),
            "among authors with at least 5 books."
        )


        # -----------------------------------------------------
        # Question 3
        # -----------------------------------------------------

        st.subheader(
            "3. What is the average rating distribution across books?"
        )

        rating_counts = (
            data["Rating"]
            .dropna()
            .value_counts()
            .sort_index()
        )

        st.bar_chart(rating_counts)

        average_rating = data["Rating"].mean()

        st.write(
            "Average book rating:",
            round(average_rating, 2)
        )

        st.write(
            "Answer: The chart shows how ratings are distributed "
            "across the books. The overall average rating is",
            round(average_rating, 2)
        )


        # -----------------------------------------------------
        # Question 4
        # -----------------------------------------------------

        st.subheader(
            "4. Are there trends in publication years for popular books?"
        )

        st.warning(
            "Publication year is not available in the supplied Audible dataset."
        )

        st.write(
            "Answer: This question cannot be analysed reliably because "
            "the dataset does not contain a publication year column."
        )


        # -----------------------------------------------------
        # Question 5
        # -----------------------------------------------------

        st.subheader(
            "5. How do ratings vary between books with different review counts?"
        )

        review_data = data[
            ["Rating", "Number of Reviews"]
        ].dropna().copy()

        review_data["Review Group"] = pd.cut(
            review_data["Number of Reviews"],
            bins=[-1, 50, 200, 1000, float("inf")],
            labels=[
                "0-50 Reviews",
                "51-200 Reviews",
                "201-1000 Reviews",
                "1000+ Reviews"
            ]
        )

        review_rating = (
            review_data.groupby(
                "Review Group",
                observed=False
            )["Rating"]
            .mean()
        )

        st.bar_chart(review_rating)

        st.dataframe(
            review_rating
            .reset_index()
            .rename(
                columns={"Rating": "Average Rating"}
            )
        )

        highest_review_group = review_rating.idxmax()

        st.write(
            "Answer:",
            highest_review_group,
            "has the highest average rating among the review-count groups."
        )


    # =========================================================
    # MEDIUM LEVEL
    # =========================================================

    elif eda_level == "Medium Level":

        st.header("Medium Level Questions")


        # -----------------------------------------------------
        # Question 1
        # -----------------------------------------------------

        st.subheader(
            "1. Which books are frequently clustered together based on descriptions?"
        )

        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.cluster import KMeans

        description_data = data[
            [
                "Book Name",
                "Author",
                "Description",
                "Rating"
            ]
        ].copy()

        description_data = description_data[
            description_data["Description"].notna()
        ]

        description_data = description_data[
            description_data["Description"]
            .astype(str)
            .str.strip()
            .ne("")
        ].reset_index(drop=True)

        vectorizer = CountVectorizer(
            max_features=5000,
            stop_words="english"
        )

        description_matrix = vectorizer.fit_transform(
            description_data["Description"]
        )

        kmeans = KMeans(
            n_clusters=8,
            random_state=42,
            n_init=10
        )

        description_data["Cluster"] = kmeans.fit_predict(
            description_matrix
        )

        cluster_counts = (
            description_data["Cluster"]
            .value_counts()
            .sort_index()
        )

        st.bar_chart(cluster_counts)

        selected_cluster = st.selectbox(
            "Choose a cluster to view books:",
            sorted(description_data["Cluster"].unique()),
            key="cluster_selector"
        )

        cluster_books = description_data[
            description_data["Cluster"] == selected_cluster
        ]

        st.dataframe(
            cluster_books[
                [
                    "Book Name",
                    "Author",
                    "Rating"
                ]
            ].head(10)
        )

        st.write(
            "Answer: Books in the same cluster contain similar "
            "words and patterns in their descriptions. "
            "The cluster chart also shows that some clusters are "
            "much larger than others."
        )


        # -----------------------------------------------------
        # Question 2
        # -----------------------------------------------------

        st.subheader(
            "2. How does genre similarity affect book recommendations?"
        )

        genre_book_data = data[
            data["Genres"].apply(len) > 0
        ].reset_index(drop=True)

        mlb_medium = MultiLabelBinarizer()

        medium_genre_matrix = mlb_medium.fit_transform(
            genre_book_data["Genres"]
        )

        medium_similarity = cosine_similarity(
            medium_genre_matrix
        )

        medium_book_names = sorted(
            genre_book_data["Book Name"]
            .dropna()
            .unique()
        )

        medium_selected_book = st.selectbox(
            "Choose a book to test genre similarity:",
            medium_book_names,
            key="medium_genre_book"
        )

        medium_index = genre_book_data[
            genre_book_data["Book Name"]
            == medium_selected_book
        ].index[0]

        medium_scores = list(
            enumerate(
                medium_similarity[medium_index]
            )
        )

        medium_scores = sorted(
            medium_scores,
            key=lambda x: x[1],
            reverse=True
        )[1:6]

        medium_recommendations = genre_book_data.iloc[
            [index for index, score in medium_scores]
        ].copy()

        medium_recommendations["Similarity Score"] = [
            round(score, 2)
            for index, score in medium_scores
        ]

        st.dataframe(
            medium_recommendations[
                [
                    "Book Name",
                    "Author",
                    "Rating",
                    "Genres",
                    "Similarity Score"
                ]
            ]
        )

        average_genre_similarity = (
            medium_recommendations[
                "Similarity Score"
            ].mean()
        )

        st.write(
            "Answer: Books with more shared genres receive higher "
            "cosine similarity scores and are therefore ranked higher "
            "by the genre-based recommendation system."
        )


        # -----------------------------------------------------
        # Question 3
        # -----------------------------------------------------

        st.subheader(
            "3. What is the effect of author popularity on book ratings?"
        )

        author_book_counts = (
            data.groupby("Author")["Book Name"]
            .count()
            .rename("Author Book Count")
        )

        author_popularity_data = data.merge(
            author_book_counts,
            on="Author",
            how="left"
        )

        def popularity_group(count):

            if count == 1:
                return "Low"

            elif count <= 4:
                return "Medium"

            elif count <= 9:
                return "High"

            else:
                return "Very High"


        author_popularity_data["Author Popularity"] = (
            author_popularity_data[
                "Author Book Count"
            ].apply(popularity_group)
        )

        popularity_ratings = (
            author_popularity_data
            .groupby(
                "Author Popularity"
            )["Rating"]
            .mean()
            .reindex(
                [
                    "Low",
                    "Medium",
                    "High",
                    "Very High"
                ]
            )
        )

        st.bar_chart(popularity_ratings)

        st.dataframe(
            popularity_ratings
            .reset_index()
            .rename(
                columns={"Rating": "Average Rating"}
            )
        )

        popularity_correlation = (
            author_popularity_data[
                [
                    "Author Book Count",
                    "Rating"
                ]
            ]
            .dropna()
            .corr()
            .iloc[0, 1]
        )

        st.write(
            "Correlation between author popularity and rating:",
            round(popularity_correlation, 3)
        )

        st.write(
            "Answer: The correlation is very small, indicating that "
            "authors with more books do not necessarily receive higher ratings."
        )


        # -----------------------------------------------------
        # Question 4
        # -----------------------------------------------------

        st.subheader(
            "4. Which combination of features provides the most accurate recommendations?"
        )

        model_results = pd.DataFrame(
            {
                "Model": [
                    "Content-Based",
                    "Clustering-Based",
                    "Hybrid"
                ],
                "Precision@5": [
                    1.0,
                    0.0,
                    1.0
                ],
                "Recall@5": [
                    0.068,
                    0.0,
                    0.068
                ],
                "Average Similarity": [
                    0.667,
                    0.237,
                    0.344
                ]
            }
        )

        st.dataframe(model_results)

        st.bar_chart(
            model_results.set_index("Model")[
                [
                    "Precision@5",
                    "Recall@5"
                ]
            ]
        )

        st.write(
            "Answer: The Content-Based model produced the strongest "
            "overall results in our evaluation. The Hybrid model achieved "
            "the same Precision@5 and Recall@5, but the Content-Based model "
            "had a higher average similarity score."
        )

        st.caption(
            "Precision and recall use genre overlap as a proxy for relevance "
            "because the dataset does not contain real user interaction labels."
        )


    # =========================================================
    # SCENARIO BASED
    # =========================================================

    elif eda_level == "Scenario Based":

        st.header("Scenario Based Questions")


        # -----------------------------------------------------
        # Scenario 1
        # -----------------------------------------------------

        st.subheader(
            "1. A new user likes science fiction books. Which top 5 books should be recommended?"
        )

        science_fiction_data = data[
            data["Genres"].apply(
                lambda genres:
                any(
                    "science fiction" in genre.lower()
                    for genre in genres
                )
            )
        ].copy()

        science_fiction_data[
            "Science Fiction Relevance"
        ] = science_fiction_data["Genres"].apply(
            lambda genres:
            sum(
                "science fiction" in genre.lower()
                for genre in genres
            )
        )

        top_science_fiction = (
            science_fiction_data
            .sort_values(
                by=[
                    "Science Fiction Relevance",
                    "Rating",
                    "Number of Reviews"
                ],
                ascending=[
                    False,
                    False,
                    False
                ]
            )
            .head(5)
        )

        st.dataframe(
            top_science_fiction[
                [
                    "Book Name",
                    "Author",
                    "Rating",
                    "Number of Reviews",
                    "Genres"
                ]
            ]
        )

        st.bar_chart(
            top_science_fiction
            .set_index("Book Name")["Rating"]
        )

        st.write(
            "Answer: These five books are recommended because they "
            "have science-fiction-related genres and strong ratings."
        )


        # -----------------------------------------------------
        # Scenario 2
        # -----------------------------------------------------

        st.subheader(
            "2. For a user who has previously rated thrillers highly, recommend similar books."
        )

        thriller_data = data[
            data["Genres"].apply(
                lambda genres:
                any(
                    "thriller" in genre.lower()
                    for genre in genres
                )
            )
        ].copy()

        thriller_data[
            "Thriller Relevance"
        ] = thriller_data["Genres"].apply(
            lambda genres:
            sum(
                "thriller" in genre.lower()
                for genre in genres
            )
        )

        top_thrillers = (
            thriller_data
            .sort_values(
                by=[
                    "Thriller Relevance",
                    "Rating",
                    "Number of Reviews"
                ],
                ascending=[
                    False,
                    False,
                    False
                ]
            )
            .head(5)
        )

        st.dataframe(
            top_thrillers[
                [
                    "Book Name",
                    "Author",
                    "Rating",
                    "Number of Reviews",
                    "Genres"
                ]
            ]
        )

        st.bar_chart(
            top_thrillers
            .set_index("Book Name")["Rating"]
        )

        st.write(
            "Answer: These books are selected because their genres "
            "contain thriller-related categories and they have strong ratings."
        )


        # -----------------------------------------------------
        # Scenario 3
        # -----------------------------------------------------

        st.subheader(
            "3. Identify books that are highly rated but have low popularity to recommend hidden gems."
        )

        author_counts_hidden = (
            data.groupby("Author")["Book Name"]
            .count()
            .rename("Author Book Count")
        )

        hidden_data = data.merge(
            author_counts_hidden,
            on="Author",
            how="left"
        )

        hidden_gems = hidden_data[
            (hidden_data["Rating"] >= 4.7)
            &
            (hidden_data["Author Book Count"] <= 1)
            &
            (hidden_data["Number of Reviews"] >= 3)
            &
            (hidden_data["Number of Reviews"] <= 20)
        ].copy()

        hidden_gems = (
            hidden_gems
            .sort_values(
                by=[
                    "Rating",
                    "Number of Reviews"
                ],
                ascending=[
                    False,
                    True
                ]
            )
            .head(5)
        )

        st.dataframe(
            hidden_gems[
                [
                    "Book Name",
                    "Author",
                    "Rating",
                    "Number of Reviews"
                ]
            ]
        )

        st.bar_chart(
            hidden_gems
            .set_index("Book Name")[
                "Number of Reviews"
            ]
        )

        st.write(
            "Answer: These books are considered hidden gems because "
            "they have very high ratings but relatively few reviews and "
            "come from authors with low representation in the dataset."
        )