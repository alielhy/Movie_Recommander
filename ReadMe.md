# 🎬 Movie Recommendation System

A content-based movie recommendation system using machine learning, featuring a Flask API backend and a modern web interface.

## 📊 Methodology

This system follows a structured 5-step approach:

1. **Data Collection**: Combines MovieLens and TMDB datasets
2. **Preprocessing**: Cleans and normalizes movie metadata
3. **Feature Extraction**: Extracts genres, keywords, cast, and overview
4. **Vectorization**: Uses TF-IDF to transform text features
5. **Similarity Computation**: Calculates cosine similarity between movies

## 📁 Project Structure

```
movie-recommendation-system/
├── recommendation_api.py      # Flask API server
├── index.html                 # Web interface
├── requirements.txt           # Python dependencies
├── README.md                  # This file
└── datasets/
    ├── tmdb_5000_movies.csv
    ├── tmdb_5000_credits.csv
    ├── movies.csv
    └── ratings.csv
```

## 🚀 Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Datasets

You need these datasets (place them in the same directory as the API):

- `tmdb_5000_movies.csv`
- `tmdb_5000_credits.csv`
- `movies.csv` (from MovieLens)
- `ratings.csv` (from MovieLens)

**Download sources:**
- TMDB: https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata
- MovieLens: https://grouplens.org/datasets/movielens/

### 3. Run the API Server

```bash
python recommendation_api.py
```

The API will:
- Automatically train the model on first run (takes 5-10 minutes)
- Create cache files for faster subsequent loads
- Start the Flask server on `http://localhost:5000`

### 4. Open the Web Interface

Simply open `index.html` in your browser, or serve it with:

```bash
# Python 3
python -m http.server 8000

# Then open: http://localhost:8000
```

## 🎯 Usage

### Web Interface

1. Type a movie title in the search box
2. Select from autocomplete suggestions or press Enter
3. View 12 personalized recommendations with:
   - Similarity scores
   - Ratings and vote counts
   - Genres and release dates
   - Movie overviews

### API Endpoints

#### Get Recommendations
```bash
POST http://localhost:5000/api/recommend
Content-Type: application/json

{
  "title": "Inception",
  "top_n": 10
}
```

#### Search Movies
```bash
GET http://localhost:5000/api/search?q=inception&limit=10
```

#### Get All Movies
```bash
GET http://localhost:5000/api/movies?limit=100
```

#### Check Status
```bash
GET http://localhost:5000/api/status
```

## 🔧 Features

### Content-Based Filtering
- Uses movie features (genres, keywords, overview, ratings)
- No user data required (solves cold-start problem)
- Privacy-friendly (no user tracking)

### Smart Fuzzy Matching
- Handles typos and partial titles
- Removes common articles (the, a, an)
- Strips years from titles

### Feature Weighting
- Overview: 1.0
- Keywords: 0.8
- Genres: 1.2
- Ratings: 1.5

### Caching System
- Merged dataset cached for instant loading
- Pre-trained model cached
- Reduces startup time from 10min to <10sec

## 🛠️ Troubleshooting

### API won't start
- Check if port 5000 is available
- Ensure all CSV files are in the correct location
- Verify Python version (3.8+)

### Model training is slow
- Normal on first run (5-10 minutes)
- Creates `merged_movies_cache.pkl` and `recommendation_model.pkl`
- Subsequent runs load instantly from cache

### Web interface shows connection error
- Make sure API is running on port 5000
- Check browser console for CORS errors
- Try refreshing the page

### No recommendations found
- Verify movie title exists in dataset
- Try different spelling or year
- Use autocomplete suggestions

## 📈 Performance

- **Dataset**: ~3,700 movies
- **Training time**: 5-10 minutes (first run)
- **Loading time**: <10 seconds (cached)
- **Recommendation time**: <100ms per query
- **Feature dimensions**: ~1,500 features per movie

## 🎓 Academic Context

This system is based on research from:
- Giridharan N., Nathan K. S., Swetha M. K. (2022)
- Nirbhay Singh, Sanket Bhokare, Mayur Shah (2023)

**Key advantages:**
- Solves cold-start problem
- Explainable recommendations
- No user data dependency
- Scalable architecture

## 📝 License

Educational project for machine learning research.

## 👥 Authors

- EL-HIYANI ALI
- ELKYOUD MOHAMMED

---

**Note**: Make sure all datasets are downloaded before running the system. The first run will take longer as it builds the recommendation model.