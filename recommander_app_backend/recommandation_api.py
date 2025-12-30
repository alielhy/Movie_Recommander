"""
Movie Recommendation System API
Based on Content-Based Filtering with Feature-Based Sub-Approach
Methodology: Data Collection → Preprocessing → Feature Extraction → 
              Vectorization (TF-IDF) → Similarity Computation (Cosine Similarity)
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import ast
import re
import pickle
import os
from nltk.corpus import stopwords
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler
from scipy.sparse import hstack, csr_matrix
from fuzzywuzzy import process, fuzz

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Download NLTK stopwords
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

# Cache files
CACHE_FILE = 'merged_movies_cache.pkl'
MODEL_CACHE = 'recommendation_model.pkl'

class MovieRecommender:
    def __init__(self):
        self.merged = None
        self.feature_matrix = None
        self.cosine_sim = None
        self.indices = None
        self.tfidf_overview = None
        self.tfidf_keywords = None
        
    def clean_title(self, t):
        """Clean movie title for better matching"""
        t = re.sub(r'\s*\(\d{4}\)', '', t)
        t = re.sub(r'^(the|a|an)\s+', '', t, flags=re.IGNORECASE)
        return t.strip().lower()
    
    def safe_list(self, col, key='name'):
        """Safely extract list of names from JSON string"""
        try:
            return ' '.join([i[key].lower().replace(' ', '') for i in ast.literal_eval(col)])
        except:
            return ''
    
    def clean_overview(self, txt):
        """Remove stopwords from overview"""
        if pd.isna(txt):
            return ''
        return ' '.join([w.lower() for w in str(txt).split() if w.lower() not in stop_words])
    
    def load_and_preprocess_data(self):
        """
        STEP 1: DATA COLLECTION
        Load datasets from MovieLens and TMDB
        """
        print("=" * 60)
        print("STEP 1: DATA COLLECTION")
        print("=" * 60)
        
        # Load datasets
        tmdb_movies = pd.read_csv('tmdb_5000_movies.csv')
        tmdb_credits = pd.read_csv('tmdb_5000_credits.csv')
        ml_movies = pd.read_csv('movies.csv')
        ml_ratings = pd.read_csv('ratings.csv')
        
        print(f"✓ Loaded {len(tmdb_movies)} TMDB movies")
        print(f"✓ Loaded {len(ml_movies)} MovieLens movies")
        print(f"✓ Loaded {len(ml_ratings)} ratings")
        
        # Compute average ratings
        avg_stats = ml_ratings.groupby('movieId')['rating'].agg(['mean', 'count']).reset_index()
        avg_stats = avg_stats.rename(columns={'mean': 'avg_rating'})
        avg_stats['avg_rating'] = avg_stats['avg_rating'].round(2)
        
        # Merge with ml_movies
        ml_movies = ml_movies.merge(avg_stats, on='movieId', how='left')
        ml_movies['avg_rating'] = ml_movies['avg_rating'].fillna(3.0)
        ml_movies['count'] = ml_movies['count'].fillna(0).astype(int)
        
        # Check if cached merge exists
        if os.path.exists(CACHE_FILE):
            print(f"\n✓ Loading cached merged data from {CACHE_FILE}")
            with open(CACHE_FILE, 'rb') as f:
                merged = pickle.load(f)
        else:
            print("\n⚙ Running fuzzy merge (this may take time)...")
            merged = self.fuzzy_merge(tmdb_movies, ml_movies)
            
            # Save cache
            with open(CACHE_FILE, 'wb') as f:
                pickle.dump(merged, f)
            print(f"✓ Cache saved to {CACHE_FILE}")
        
        print(f"✓ Merged dataset: {len(merged)} movies")
        return merged
    
    def fuzzy_merge(self, tmdb_movies, ml_movies):
        """Fuzzy match TMDB and MovieLens titles"""
        tmdb_movies['title_clean'] = tmdb_movies['title'].apply(self.clean_title)
        ml_movies['title_clean'] = ml_movies['title'].apply(self.clean_title)
        
        matches = []
        for i, tmdb_title in enumerate(tmdb_movies['title_clean']):
            if i % 100 == 0:
                print(f"  Progress: {i}/{len(tmdb_movies)} ({i/len(tmdb_movies)*100:.1f}%)")
            
            best = process.extractOne(
                tmdb_title,
                ml_movies['title_clean'].tolist(),
                scorer=fuzz.ratio,
                score_cutoff=85
            )
            
            if best:
                ml_idx = ml_movies[ml_movies['title_clean'] == best[0]].index[0]
                tmdb_idx = tmdb_movies[tmdb_movies['title_clean'] == tmdb_title].index[0]
                matches.append((tmdb_idx, ml_idx))
        
        if matches:
            tmdb_part = tmdb_movies.iloc[[m[0] for m in matches]].reset_index(drop=True)
            ml_part = ml_movies.iloc[[m[1] for m in matches]][['genres', 'avg_rating', 'count']].reset_index(drop=True)
            ml_part = ml_part.rename(columns={
                'genres': 'genres_ml',
                'avg_rating': 'avg_rating_ml',
                'count': 'rating_count'
            })
            merged = pd.concat([tmdb_part, ml_part], axis=1)
        else:
            merged = tmdb_movies.copy()
            merged['avg_rating_ml'] = 3.5
            merged['rating_count'] = 0
            merged['genres_ml'] = ''
        
        return merged
    
    def preprocess_features(self, merged):
        """
        STEP 2: PREPROCESSING
        Clean and normalize movie metadata
        """
        print("\n" + "=" * 60)
        print("STEP 2: PREPROCESSING")
        print("=" * 60)
        
        # Extract genres and keywords
        merged['genres_tmdb'] = merged['genres'].apply(lambda x: self.safe_list(x, 'name'))
        merged['keywords'] = merged['keywords'].apply(lambda x: self.safe_list(x, 'name'))
        merged['genres_ml'] = merged['genres_ml'].apply(lambda x: x.lower().replace('|', ' '))
        
        # Clean overview
        merged['overview'] = merged['overview'].apply(self.clean_overview)
        
        print("✓ Extracted genres from TMDB and MovieLens")
        print("✓ Extracted keywords")
        print("✓ Cleaned overview text (removed stopwords)")
        
        return merged
    
    def extract_features(self, merged):
        """
        STEP 3: FEATURE EXTRACTION
        Combine movie features (genres, keywords, overview, ratings)
        """
        print("\n" + "=" * 60)
        print("STEP 3: FEATURE EXTRACTION")
        print("=" * 60)
        
        # Overview TF-IDF
        overview_text = merged['overview'].fillna('').replace('', 'plot unknown')
        self.tfidf_overview = TfidfVectorizer(max_features=1000, stop_words='english', min_df=1)
        overview_matrix = self.tfidf_overview.fit_transform(overview_text)
        print(f"✓ Overview features: {overview_matrix.shape[1]} dimensions")
        
        # Keywords TF-IDF
        keywords_text = merged['keywords'].fillna('').replace('', 'none')
        self.tfidf_keywords = TfidfVectorizer(max_features=500)
        keywords_matrix = self.tfidf_keywords.fit_transform(keywords_text)
        print(f"✓ Keyword features: {keywords_matrix.shape[1]} dimensions")
        
        # Genres (multi-label binarization)
        genre_list = (
            merged['genres_tmdb'].fillna('').str.split() +
            merged['genres_ml'].fillna('').str.split()
        )
        mlb_genre = MultiLabelBinarizer()
        genres_matrix = mlb_genre.fit_transform(genre_list)
        print(f"✓ Genre features: {genres_matrix.shape[1]} dimensions")
        
        # Rating boost
        scaler = MinMaxScaler()
        rating_boost = scaler.fit_transform(merged[['avg_rating_ml']]) * 2.0
        print(f"✓ Rating boost applied")
        
        return overview_matrix, keywords_matrix, genres_matrix, rating_boost
    
    def vectorize_features(self, overview_matrix, keywords_matrix, genres_matrix, rating_boost):
        """
        STEP 4: VECTORIZATION
        Combine all features into a single feature matrix with weights
        """
        print("\n" + "=" * 60)
        print("STEP 4: VECTORIZATION (TF-IDF)")
        print("=" * 60)
        
        feature_matrix = hstack([
            overview_matrix * 1.0,      # Overview weight
            keywords_matrix * 0.8,       # Keywords weight
            csr_matrix(genres_matrix) * 1.2,  # Genres weight
            csr_matrix(rating_boost) * 1.5    # Rating weight
        ])
        
        print(f"✓ Combined feature matrix: {feature_matrix.shape}")
        print(f"  - Total features: {feature_matrix.shape[1]}")
        print(f"  - Feature weights applied:")
        print(f"    · Overview: 1.0")
        print(f"    · Keywords: 0.8")
        print(f"    · Genres: 1.2")
        print(f"    · Ratings: 1.5")
        
        return feature_matrix
    
    def compute_similarity(self, feature_matrix):
        """
        STEP 5: SIMILARITY COMPUTATION
        Calculate cosine similarity between all movies
        """
        print("\n" + "=" * 60)
        print("STEP 5: SIMILARITY COMPUTATION (Cosine Similarity)")
        print("=" * 60)
        
        cosine_sim = cosine_similarity(feature_matrix, feature_matrix)
        print(f"✓ Similarity matrix computed: {cosine_sim.shape}")
        print(f"  - Each movie compared with {cosine_sim.shape[1]} movies")
        print(f"  - Similarity range: [{cosine_sim.min():.3f}, {cosine_sim.max():.3f}]")
        
        return cosine_sim
    
    def build_index(self, merged):
        """Build title-to-index mapping"""
        indices = pd.Series(
            merged.index,
            index=merged['title'].str.lower().str.strip()
        ).drop_duplicates()
        
        return indices
    
    def train(self):
        """Train the recommendation model"""
        print("\n" + "🎬" * 30)
        print("MOVIE RECOMMENDATION SYSTEM - TRAINING")
        print("🎬" * 30)
        
        # Check if model exists
        if os.path.exists(MODEL_CACHE):
            print(f"\n✓ Loading pre-trained model from {MODEL_CACHE}")
            with open(MODEL_CACHE, 'rb') as f:
                cache = pickle.load(f)
                self.merged = cache['merged']
                self.feature_matrix = cache['feature_matrix']
                self.cosine_sim = cache['cosine_sim']
                self.indices = cache['indices']
            print("✓ Model loaded successfully!")
            return
        
        # Step 1: Load data
        self.merged = self.load_and_preprocess_data()
        
        # Step 2: Preprocess
        self.merged = self.preprocess_features(self.merged)
        
        # Step 3-4: Extract and vectorize features
        overview_matrix, keywords_matrix, genres_matrix, rating_boost = self.extract_features(self.merged)
        self.feature_matrix = self.vectorize_features(overview_matrix, keywords_matrix, genres_matrix, rating_boost)
        
        # Step 5: Compute similarity
        self.cosine_sim = self.compute_similarity(self.feature_matrix)
        
        # Build index
        self.indices = self.build_index(self.merged)
        
        # Save model
        print("\n💾 Saving trained model...")
        with open(MODEL_CACHE, 'wb') as f:
            pickle.dump({
                'merged': self.merged,
                'feature_matrix': self.feature_matrix,
                'cosine_sim': self.cosine_sim,
                'indices': self.indices
            }, f)
        print(f"✓ Model saved to {MODEL_CACHE}")
        
        print("\n" + "✅" * 30)
        print("TRAINING COMPLETE!")
        print("✅" * 30 + "\n")
    
    def recommend(self, title, top_n=10):
        """
        Get movie recommendations
        Returns: list of recommended movies with details
        """
        title = title.strip().lower()
        
        # Find movie index
        if title not in self.indices:
            # Fuzzy search fallback
            matches = process.extract(title, self.indices.index.tolist(), limit=5, scorer=fuzz.ratio)
            if not matches or matches[0][1] < 60:
                return None
            title = matches[0][0]
        
        idx = self.indices[title]
        
        # Get similarity scores
        sim_scores = sorted(enumerate(self.cosine_sim[idx]), key=lambda x: x[1], reverse=True)[1:top_n+1]
        
        # Build recommendations
        recommendations = []
        for i, score in sim_scores:
            movie = self.merged.iloc[i]
            recommendations.append({
                'title': movie['title'],
                'rating': float(movie.get('avg_rating_ml', 3.0)),
                'votes': int(movie.get('rating_count', 0)),
                'similarity': float(score),
                'overview': movie.get('overview', '')[:200] + '...',
                'genres': movie.get('genres_tmdb', '').replace(' ', ', ').title(),
                'release_date': movie.get('release_date', 'N/A')
            })
        
        return {
            'query_movie': self.merged.iloc[idx]['title'],
            'recommendations': recommendations
        }
    
    def search_movies(self, query, limit=10):
        """Search for movies by partial title"""
        query = query.lower().strip()
        matches = process.extract(query, self.indices.index.tolist(), limit=limit, scorer=fuzz.partial_ratio)
        
        results = []
        for match, score in matches:
            if score > 50:
                idx = self.indices[match]
                movie = self.merged.iloc[idx]
                results.append({
                    'title': movie['title'],
                    'rating': float(movie.get('avg_rating_ml', 3.0)),
                    'votes': int(movie.get('rating_count', 0)),
                    'genres': movie.get('genres_tmdb', '').replace(' ', ', ').title(),
                    'release_date': movie.get('release_date', 'N/A')
                })
        
        return results
    
    def get_all_movies(self, limit=100):
        """Get list of all movies (for autocomplete)"""
        movies = self.merged.head(limit)[['title', 'avg_rating_ml', 'genres_tmdb']].copy()
        movies['genres'] = movies['genres_tmdb'].str.replace(' ', ', ').str.title()
        return movies[['title', 'avg_rating_ml', 'genres']].to_dict('records')

# Initialize recommender
recommender = MovieRecommender()

# API Routes
@app.route('/api/train', methods=['POST'])
def train_model():
    """Train the recommendation model"""
    try:
        recommender.train()
        return jsonify({
            'status': 'success',
            'message': 'Model trained successfully',
            'total_movies': len(recommender.merged)
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/recommend', methods=['POST'])
def get_recommendations():
    """Get movie recommendations"""
    data = request.json
    title = data.get('title', '')
    top_n = data.get('top_n', 10)
    
    if not title:
        return jsonify({'status': 'error', 'message': 'Movie title is required'}), 400
    
    try:
        result = recommender.recommend(title, top_n)
        if result is None:
            return jsonify({'status': 'error', 'message': f'Movie "{title}" not found'}), 404
        
        return jsonify({'status': 'success', 'data': result})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/search', methods=['GET'])
def search():
    """Search for movies"""
    query = request.args.get('q', '')
    limit = int(request.args.get('limit', 10))
    
    if not query:
        return jsonify({'status': 'error', 'message': 'Query parameter is required'}), 400
    
    try:
        results = recommender.search_movies(query, limit)
        return jsonify({'status': 'success', 'data': results})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/movies', methods=['GET'])
def get_movies():
    """Get all movies"""
    limit = int(request.args.get('limit', 100))
    
    try:
        movies = recommender.get_all_movies(limit)
        return jsonify({'status': 'success', 'data': movies})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/status', methods=['GET'])
def status():
    """Check API status"""
    return jsonify({
        'status': 'success',
        'message': 'Movie Recommendation API is running',
        'model_loaded': recommender.merged is not None,
        'total_movies': len(recommender.merged) if recommender.merged is not None else 0
    })

if __name__ == '__main__':
    # Train model on startup
    recommender.train()
    
    # Run Flask app
    print("\n🚀 Starting Flask API server...")
    print("📍 API will be available at: http://localhost:5000")
    print("\nAvailable endpoints:")
    print("  POST /api/recommend - Get recommendations")
    print("  GET  /api/search - Search movies")
    print("  GET  /api/movies - Get all movies")
    print("  GET  /api/status - Check API status")
    
    app.run(debug=True, port=5000)