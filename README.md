
# Audible Insights: Intelligent Book Recommendations

An intelligent audiobook recommendation system built using Python, NLP, clustering, content-based filtering, and hybrid recommendation techniques. The project also includes an interactive Streamlit application for exploring audiobook data and receiving recommendations.

## Project Overview

The objective of this project is to analyze Audible audiobook data and build a recommendation system that helps users discover relevant audiobooks based on their interests.

The project includes data cleaning, exploratory data analysis, natural language processing, clustering, recommendation systems, model evaluation, and an interactive Streamlit application.

## Dataset

The project uses Audible audiobook catalog data containing information such as:

- Book Name
- Author
- Rating
- Number of Reviews
- Price
- Description
- Listening Time
- Genres

The supplied datasets were cleaned, deduplicated, transformed, and merged to create the final dataset used for analysis and recommendation.

## Project Workflow

1. Data Loading and Inspection
2. Data Cleaning and Preprocessing
3. Dataset Merging
4. Exploratory Data Analysis
5. NLP on Book Descriptions
6. K-Means Clustering
7. Content-Based Recommendation
8. Clustering-Based Recommendation
9. Hybrid Recommendation
10. Recommendation Model Evaluation
11. Streamlit Application Development

## Exploratory Data Analysis

The EDA investigates questions such as:

- What are the most popular audiobook genres?
- Which authors have the highest average ratings?
- How are audiobook ratings distributed?
- How do ratings vary with the number of reviews?
- Does author popularity influence average ratings?
- Which highly rated books can be considered hidden gems?

## NLP and Clustering

Book descriptions were transformed using `CountVectorizer`.

K-Means clustering was then applied to group books according to similarities in their description text.

The clustering analysis also demonstrated some limitations of applying K-Means to sparse text data, including a dominant general cluster and several small outlier clusters.

## Recommendation Systems

Three recommendation approaches were explored.

### Content-Based Recommendation

Uses audiobook genres and cosine similarity to identify books with similar genre characteristics.

### Clustering-Based Recommendation

Uses NLP features from book descriptions and K-Means clusters to identify books with similar textual characteristics.

### Hybrid Recommendation

Combines genre similarity and description similarity to provide more balanced recommendations.

## Model Evaluation

The recommendation approaches were compared using:

- Precision@5
- Recall@5
- Average Similarity

Because the dataset does not contain actual user interaction histories or user-level relevance labels, genre overlap was used as a proxy definition of relevance for Precision@5 and Recall@5.

The content-based and hybrid approaches performed better than the clustering-only approach under this evaluation method.

## Streamlit Application

The project includes an interactive Streamlit application with three main sections:

### Home
Displays an overview and preview of the cleaned audiobook dataset.

### Recommendations
Allows users to:

- Browse highly rated books by genre
- Select a book and receive similar audiobook recommendations

### EDA
Provides interactive analysis for:

- Easy-level analytical questions
- Medium-level NLP and recommendation analysis
- Scenario-based business questions

## Technologies Used

- Python
- Pandas
- Scikit-learn
- CountVectorizer
- K-Means Clustering
- Cosine Similarity
- MultiLabelBinarizer
- Streamlit
- Jupyter Notebook
- GitHub

## Repository Files

| File | Description |
|---|---|
| `Audible project.ipynb` | Complete data analysis and recommendation-system notebook |
| `Audible.py` | Streamlit application |
| `audible_cleaned_data.csv` | Cleaned dataset used by the Streamlit application |
| `requirements.txt` | Required Python libraries |
| `README.md` | Project documentation |

## Installation

Clone the repository and install the required packages:

```bash
pip install -r requirements.txt
```

## Run the Streamlit Application

Run the following command from the project directory:

```bash
streamlit run Audible.py
```

Streamlit will start the application and provide a local URL that can be opened in a web browser.

## Key Features

- Audible audiobook data cleaning and preprocessing
- Genre analysis
- Rating and review analysis
- NLP-based description analysis
- K-Means book clustering
- Content-based recommendation
- Clustering-based recommendation
- Hybrid recommendation
- Recommendation model evaluation
- Hidden-gem discovery
- Interactive Streamlit dashboard

## Future Improvements

Future versions of the project could include:

- User-specific ratings and interaction history
- Collaborative filtering
- Improved NLP embeddings for book descriptions
- More advanced hybrid recommendation techniques
- Cloud deployment

