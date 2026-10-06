import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="CineMatch",
    page_icon="🎬",
    layout="wide"
)


# =========================================================
# CSS
# =========================================================

st.markdown("""
<style>

.stApp {
    background-color: #0b0d12;
}

.block-container {
    max-width: 1400px;
    padding-top: 2rem;
}

/* Header */

.main-title {
    font-size: 42px;
    font-weight: 800;
    color: white;
}

.subtitle {
    color: #9ca3af;
    font-size: 16px;
    margin-bottom: 25px;
}

/* User selection box */

.selection-box {
    background: #151820;
    border: 1px solid #292d38;
    border-radius: 16px;
    padding: 20px;
    margin-bottom: 25px;
}

/* Metrics */

[data-testid="stMetric"] {
    background-color: #151820;
    border: 1px solid #292d38;
    border-radius: 15px;
    padding: 15px;
}

/* Section */

.section-title {
    font-size: 25px;
    font-weight: 700;
    margin-top: 30px;
    margin-bottom: 15px;
}

/* Footer */

.footer {
    text-align: center;
    color: #6b7280;
    padding: 35px;
    font-size: 13px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():

    df = pd.read_excel(
        "data/Movie_Recommendation_System.xlsx",
        sheet_name="Movie_Data"
    )

    df["Genre"] = df["Genre"].fillna("Unknown")
    df["Language"] = df["Language"].fillna("Unknown")

    df = df.dropna(subset=["Rating"])

    df = df.drop_duplicates(
        subset=["User_ID", "Movie_Title"]
    )

    return df


df = load_data()


# =========================================================
# CREATE MATRIX
# =========================================================

@st.cache_data
def create_model(df):

    user_movie_matrix = df.pivot_table(
        index="User_ID",
        columns="Movie_Title",
        values="Rating",
        aggfunc="mean"
    )

    filled_matrix = user_movie_matrix.fillna(0)

    similarity = cosine_similarity(
        filled_matrix
    )

    similarity_df = pd.DataFrame(
        similarity,
        index=user_movie_matrix.index,
        columns=user_movie_matrix.index
    )

    return user_movie_matrix, similarity_df


user_movie_matrix, similarity_df = create_model(df)


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">🎬 CineMatch</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Personalized movie recommendations using collaborative filtering'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# USER SELECTION
# =========================================================

st.markdown(
    '<div class="selection-box">',
    unsafe_allow_html=True
)

st.markdown("### 👤 Choose a User")

st.write(
    "Select a User ID to generate personalized movie recommendations."
)


# Make User IDs clean
user_ids = sorted(
    user_movie_matrix.index.tolist()
)


# Store selected user in session state
if "selected_user" not in st.session_state:

    st.session_state.selected_user = user_ids[0]


selected_user = st.selectbox(
    "User ID",
    options=user_ids,
    index=user_ids.index(
        st.session_state.selected_user
    ),
    format_func=lambda x: f"User {x}",
    key="user_selector"
)


# Apply button
apply_button = st.button(
    "🎯 Generate Recommendations",
    use_container_width=True
)


if apply_button:

    st.session_state.selected_user = selected_user


else:

    selected_user = st.session_state.selected_user


st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# USER DATA
# =========================================================

user_movies = user_movie_matrix.loc[
    selected_user
].dropna()


# =========================================================
# SIMILAR USERS
# =========================================================

similar_users = similarity_df[
    selected_user
].sort_values(
    ascending=False
)

similar_users = similar_users.drop(
    selected_user,
    errors="ignore"
)

top_users = similar_users.head(5).index


# =========================================================
# RECOMMENDATIONS
# =========================================================

similar_users_ratings = user_movie_matrix.loc[
    top_users
]

movie_scores = similar_users_ratings.mean(
    axis=0
)

# Remove already rated movies
movie_scores = movie_scores.drop(
    user_movies.index,
    errors="ignore"
)

# Remove invalid values
movie_scores = movie_scores.dropna()

# Sort
movie_scores = movie_scores.sort_values(
    ascending=False
)

recommendations = movie_scores.head(5)


# =========================================================
# METRICS
# =========================================================

st.markdown(
    '<div class="section-title">📊 User Overview</div>',
    unsafe_allow_html=True
)

col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "Selected User",
        f"User {selected_user}"
    )


with col2:

    st.metric(
        "Movies Rated",
        len(user_movies)
    )


with col3:

    st.metric(
        "Similar Users",
        len(top_users)
    )


with col4:

    if len(user_movies) > 0:

        avg_rating = user_movies.mean()

    else:

        avg_rating = 0

    st.metric(
        "Average Rating",
        f"{avg_rating:.2f} ⭐"
    )


# =========================================================
# RECOMMENDED MOVIES
# =========================================================

st.markdown(
    '<div class="section-title">✨ Recommended For You</div>',
    unsafe_allow_html=True
)


if recommendations.empty:

    st.warning(
        "No recommendations available for this user."
    )

else:

    columns = st.columns(
        len(recommendations)
    )

    for i, (movie, rating) in enumerate(
        recommendations.items()
    ):

        with columns[i]:

            with st.container(border=True):

                st.caption(
                    f"#{i + 1} RECOMMENDATION"
                )

                st.markdown(
                    f"### 🎬 {movie}"
                )

                st.markdown(
                    f"**⭐ {float(rating):.2f}**"
                )

                st.caption(
                    "Based on similar users"
                )


# =========================================================
# SIMILAR USERS
# =========================================================

left, right = st.columns(2)


with left:

    st.markdown(
        '<div class="section-title">🤝 Similar Users</div>',
        unsafe_allow_html=True
    )

    similar_data = []

    for user in top_users:

        similarity_percentage = (
            similarity_df.loc[
                selected_user,
                user
            ] * 100
        )

        similar_data.append({
            "User ID": user,
            "Similarity": round(
                similarity_percentage,
                2
            )
        })

    similar_df = pd.DataFrame(
        similar_data
    )

    st.dataframe(
        similar_df,
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# RATING HISTORY
# =========================================================

with right:

    st.markdown(
        '<div class="section-title">⭐ Rating History</div>',
        unsafe_allow_html=True
    )

    if not user_movies.empty:

        rating_distribution = (
            user_movies
            .round()
            .astype(int)
            .value_counts()
            .sort_index()
        )

        st.bar_chart(
            rating_distribution,
            height=250
        )

    else:

        st.info(
            "No rating history available."
        )


# =========================================================
# MOVIES RATED BY USER
# =========================================================

st.markdown(
    '<div class="section-title">🎞️ Movies You Have Rated</div>',
    unsafe_allow_html=True
)


if not user_movies.empty:

    watched_df = pd.DataFrame({
        "Movie": user_movies.index,
        "Your Rating": user_movies.values
    })

    watched_df = watched_df.sort_values(
        "Your Rating",
        ascending=False
    )

    watched_df["Your Rating"] = (
        watched_df["Your Rating"].round(2)
    )

    st.dataframe(
        watched_df,
        use_container_width=True,
        hide_index=True
    )

else:

    st.info(
        "This user has not rated any movies."
    )


# =========================================================
# RECOMMENDATION DETAILS
# =========================================================

st.markdown(
    '<div class="section-title">🎥 Recommendation Details</div>',
    unsafe_allow_html=True
)


if not recommendations.empty:

    details = df[
        df["Movie_Title"].isin(
            recommendations.index
        )
    ].copy()

    if not details.empty:

        details = (
            details
            .groupby("Movie_Title")
            .agg({
                "Genre": "first",
                "Language": "first"
            })
        )

        details = (
            details
            .reindex(recommendations.index)
            .reset_index()
        )

        details.columns = [
            "Movie",
            "Genre",
            "Language"
        ]

        details["Predicted Rating"] = (
            details["Movie"]
            .map(recommendations)
            .round(2)
        )

        st.dataframe(
            details,
            use_container_width=True,
            hide_index=True
        )


# =========================================================
# DATASET OVERVIEW
# =========================================================

st.markdown(
    '<div class="section-title">📚 Dataset Overview</div>',
    unsafe_allow_html=True
)

c1, c2, c3 = st.columns(3)

with c1:

    st.metric(
        "Total Users",
        df["User_ID"].nunique()
    )

with c2:

    st.metric(
        "Total Movies",
        df["Movie_Title"].nunique()
    )

with c3:

    st.metric(
        "Total Ratings",
        len(df)
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer">
        CineMatch • User-Based Collaborative Filtering<br>
        Built with Python • Pandas • Scikit-learn • Streamlit
    </div>
    """,
    unsafe_allow_html=True
)