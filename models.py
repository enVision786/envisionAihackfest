import sqlite3
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import re
import nltk
import emoji
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

class DatabaseManager:
    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = None
        self.cursor = None
    
    def connect(self):
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.cursor = self.conn.cursor()
    
    def disconnect(self):
        if self.conn:
            self.conn.close()
    
    def setup_database(self):
        """Create necessary tables if they don't exist"""
        self.connect()
        
        # Create entries table
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS entries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL,
            emotion_scores TEXT NOT NULL,
            timestamp DATETIME NOT NULL
        )
        ''')
        
        # Create life_events table
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS life_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            description TEXT,
            event_date DATETIME NOT NULL,
            importance INTEGER DEFAULT 1
        )
        ''')
        
        self.conn.commit()
        self.disconnect()
    
    def save_entry(self, text, emotion_scores, timestamp):
        """Save a new entry to the database"""
        self.connect()
        
        # Convert emotion_scores dict to JSON string
        emotion_scores_json = json.dumps(emotion_scores)
        
        self.cursor.execute(
            "INSERT INTO entries (text, emotion_scores, timestamp) VALUES (?, ?, ?)",
            (text, emotion_scores_json, timestamp)
        )
        
        self.conn.commit()
        self.disconnect()
    
    def get_latest_entry(self):
        """Get the most recent entry"""
        self.connect()
        
        self.cursor.execute(
            "SELECT * FROM entries ORDER BY timestamp DESC LIMIT 1"
        )
        
        row = self.cursor.fetchone()
        self.disconnect()
        
        if row:
            return {
                'id': row['id'],
                'text': row['text'],
                'emotion_scores': json.loads(row['emotion_scores']),
                'timestamp': row['timestamp']
            }
        return None
    
    def get_mood_data(self, start_date, end_date):
        """Get mood data for a specific date range"""
        self.connect()
        
        # Convert dates to datetime strings
        start_datetime = datetime.combine(start_date, datetime.min.time())
        end_datetime = datetime.combine(end_date, datetime.max.time())
        
        self.cursor.execute(
            "SELECT * FROM entries WHERE timestamp BETWEEN ? AND ? ORDER BY timestamp",
            (start_datetime, end_datetime)
        )
        
        rows = self.cursor.fetchall()
        self.disconnect()
        
        if not rows:
            return pd.DataFrame()
        
        # Process the data
        data = []
        for row in rows:
            entry = {
                'id': row['id'],
                'text': row['text'],
                'timestamp': row['timestamp']
            }
            
            # Add emotion scores as separate columns
            emotion_scores = json.loads(row['emotion_scores'])
            for emotion, score in emotion_scores.items():
                entry[emotion] = score
            
            data.append(entry)
        
        return pd.DataFrame(data)
    
    def get_all_mood_data(self):
        """Get all mood data"""
        self.connect()
        
        self.cursor.execute("SELECT * FROM entries ORDER BY timestamp")
        
        rows = self.cursor.fetchall()
        self.disconnect()
        
        if not rows:
            return pd.DataFrame()
        
        # Process the data
        data = []
        for row in rows:
            entry = {
                'id': row['id'],
                'text': row['text'],
                'timestamp': row['timestamp']
            }
            
            # Add emotion scores as separate columns
            emotion_scores = json.loads(row['emotion_scores'])
            for emotion, score in emotion_scores.items():
                entry[emotion] = score
            
            data.append(entry)
        
        return pd.DataFrame(data)
    
    def has_sufficient_data(self, min_entries=5):
        """Check if there's enough data for predictive analytics"""
        self.connect()
        
        self.cursor.execute("SELECT COUNT(*) as count FROM entries")
        count = self.cursor.fetchone()['count']
        
        self.disconnect()
        return count >= min_entries
    
    def clear_all_data(self):
        """Delete all data from the database"""
        self.connect()
        
        self.cursor.execute("DELETE FROM entries")
        self.cursor.execute("DELETE FROM life_events")
        
        self.conn.commit()
        self.disconnect()

class TextProcessor:
    def __init__(self):
        # Download NLTK resources if not already downloaded
        try:
            nltk.data.find('punkt')
            nltk.data.find('stopwords')
        except LookupError:
            nltk.download('punkt')
            nltk.download('stopwords')
        
        self.stop_words = set(stopwords.words('english'))
    
    def preprocess(self, text):
        """
        Preprocess text for analysis
        """
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Extract emojis (we'll keep them for analysis)
        emojis = ''.join(c for c in text if c in emoji.EMOJI_DATA)
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        
        # Remove special characters but keep emojis
        text_without_emojis = ''.join(c for c in text if c not in emoji.EMOJI_DATA)
        text_without_emojis = re.sub(r'[^\w\s]', '', text_without_emojis)
        
        # Tokenize
        tokens = word_tokenize(text_without_emojis)
        
        # Remove stopwords (optional, might want to keep for emotion analysis)
        # tokens = [word for word in tokens if word not in self.stop_words]
        
        # Reconstruct text
        processed_text = ' '.join(tokens)
        
        # Add emojis back
        if emojis:
            processed_text += ' ' + emojis
        
        return processed_text

class EmotionAnalyzer:
    def __init__(self):
        # Download NLTK resources if not already downloaded
        try:
            nltk.data.find('vader_lexicon')
        except LookupError:
            nltk.download('vader_lexicon')
        
        # Initialize VADER sentiment analyzer
        self.vader = SentimentIntensityAnalyzer()
        
        # For demo purposes, we'll simulate the model
        self.emotions = ["Valence", "Arousal", "Dominance", "Openness", 
                        "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]
    
    def analyze(self, text):
        """
        Analyze text and return emotion scores
        
        In a production environment, this would use actual models.
        For demo purposes, we're using VADER for sentiment and simulating the rest.
        """
        # Get VADER sentiment scores
        vader_scores = self.vader.polarity_scores(text)
        
        # Simulate emotion scores for demo
        # In a real implementation, these would come from actual models
        emotion_scores = {
            "Valence": vader_scores["compound"] * 0.5 + 0.5,  # Scale from [-1,1] to [0,1]
            "Arousal": self._simulate_score(text, "Arousal"),
            "Dominance": self._simulate_score(text, "Dominance"),
            "Openness": self._simulate_score(text, "Openness"),
            "Conscientiousness": self._simulate_score(text, "Conscientiousness"),
            "Extraversion": self._simulate_score(text, "Extraversion"),
            "Agreeableness": self._simulate_score(text, "Agreeableness"),
            "Neuroticism": self._simulate_score(text, "Neuroticism")
        }
        
        # Add confidence intervals
        for emotion in self.emotions:
            # Simulate confidence interval (would be model-derived in production)
            confidence = random.uniform(0.05, 0.2)
            emotion_scores[f"{emotion}_lower"] = max(0, emotion_scores[emotion] - confidence)
            emotion_scores[f"{emotion}_upper"] = min(1, emotion_scores[emotion] + confidence)
        
        return emotion_scores
    
    def _simulate_score(self, text, emotion):
        """
        Simulate an emotion score based on text characteristics
        This is just for demo purposes and would be replaced with actual model predictions
        """
        # Use text length, word count, etc. to create somewhat meaningful simulated scores
        text_length = len(text)
        word_count = len(text.split())
        
        # Create a hash-like value from the text and emotion name for consistency
        seed = sum(ord(c) for c in text) + sum(ord(c) for c in emotion)
        random.seed(seed)
        
        # Generate a score between 0 and 1
        base_score = random.uniform(0.3, 0.7)
        
        # Adjust based on text characteristics
        if "!" in text:
            base_score += 0.1
        if "?" in text:
            base_score -= 0.05
        if text_length > 100:
            base_score += 0.05
        
        # Ensure score is between 0 and 1
        return max(0, min(1, base_score))

class MoodPredictor:
    """
    Model for predicting future mood states based on historical data
    """
    def __init__(self):
        self.model = None
        self.poly = None
        self.first_day = None
    
    def train(self, mood_data, emotion='Valence'):
        """
        Train the model on historical mood data
        """
        if len(mood_data) < 5:
            return False
        
        # Ensure timestamp is datetime
        mood_data['timestamp'] = pd.to_datetime(mood_data['timestamp'])
        
        # Store the first day for reference
        self.first_day = mood_data['timestamp'].min()
        
        # Convert timestamps to numeric (days since first entry)
        mood_data['days'] = (mood_data['timestamp'] - self.first_day).dt.total_seconds() / (24 * 3600)
        
        # Prepare data for polynomial regression
        X = mood_data['days'].values.reshape(-1, 1)
        y = mood_data[emotion].values
        
        # Create polynomial features
        self.poly = PolynomialFeatures(degree=2)
        X_poly = self.poly.fit_transform(X)
        
        # Fit polynomial regression model
        self.model = LinearRegression()
        self.model.fit(X_poly, y)
        
        return True
    
    def predict(self, days_ahead=7, num_points=100):
        """
        Predict mood for the specified number of days ahead
        """
        if self.model is None or self.poly is None or self.first_day is None:
            return None
        
        # Create a range of days from 0 to days_ahead
        future_days = np.linspace(0, days_ahead, num_points)
        
        # Transform using polynomial features
        future_X = future_days.reshape(-1, 1)
        future_X_poly = self.poly.transform(future_X)
        
        # Make predictions
        predictions = self.model.predict(future_X_poly)
        
        # Ensure predictions are between 0 and 1
        predictions = np.clip(predictions, 0, 1)
        
        # Convert days back to timestamps
        future_dates = [self.first_day + timedelta(days=float(d)) for d in future_days]
        
        # Create a dataframe with the predictions
        prediction_df = pd.DataFrame({
            'timestamp': future_dates,
            'prediction': predictions
        })
        
        return prediction_df
    
    def get_confidence_intervals(self, prediction_df, confidence=0.95):
        """
        Add confidence intervals to the predictions
        In a real implementation, this would use proper statistical methods
        """
        # For demo purposes, we'll use a simple approach
        # In a real implementation, this would use proper statistical methods
        
        # Add confidence intervals (simple approach for demo)
        z_score = 1.96  # 95% confidence
        std_dev = 0.1  # Arbitrary for demo
        
        prediction_df['lower_bound'] = prediction_df['prediction'] - z_score * std_dev
        prediction_df['upper_bound'] = prediction_df['prediction'] + z_score * std_dev
        
        # Ensure bounds are between 0 and 1
        prediction_df['lower_bound'] = np.clip(prediction_df['lower_bound'], 0, 1)
        prediction_df['upper_bound'] = np.clip(prediction_df['upper_bound'], 0, 1)
        
        return prediction_df

class CalendarSync:
    """
    Utility for synchronizing with external calendar APIs
    In a production environment, this would use actual API integrations
    """
    def __init__(self, calendar_type="Google"):
        self.calendar_type = calendar_type
        self.authenticated = False
        self.events = []
    
    def authenticate(self, credentials=None):
        """
        Authenticate with the calendar service
        In a demo, we'll simulate this
        """
        # In a real implementation, this would use OAuth2 or similar
        self.authenticated = True
        return self.authenticated
    
    def fetch_events(self, start_date, end_date):
        """
        Fetch events from the calendar service
        For demo purposes, we'll return simulated events
        """
        if not self.authenticated:
            return []
        
        # In a real implementation, this would call the calendar API
        # For demo, we'll generate some sample events
        sample_events = [
            {
                "title": "Team Meeting",
                "description": "Weekly team sync",
                "start": start_date + timedelta(days=1, hours=10),
                "end": start_date + timedelta(days=1, hours=11),
                "importance": 2
            },
            {
                "title": "Doctor Appointment",
                "description": "Annual checkup",
                "start": start_date + timedelta(days=3, hours=14),
                "end": start_date + timedelta(days=3, hours=15),
                "importance": 3
            },
            {
                "title": "Coffee with Friend",
                "description": "Catching up",
                "start": start_date + timedelta(days=5, hours=16),
                "end": start_date + timedelta(days=5, hours=17),
                "importance": 1
            }
        ]
        
        self.events = sample_events
        return sample_events
    
    def save_events_to_database(self, db_manager):
        """
        Save fetched events to the database
        """
        if not self.events:
            return False
        
        # In a real implementation, this would save to the database
        # For demo, we'll just return True
        return True