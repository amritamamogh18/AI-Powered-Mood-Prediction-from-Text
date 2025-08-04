import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from datetime import datetime
import seaborn as sns
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud
import json
import os

# Download NLTK resources
nltk.download('vader_lexicon')
nltk.download('stopwords')

# Initialize sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Enhanced dataset (still small - in practice use thousands of samples)
data = {
    'text': [
        "I'm feeling great today! The sun is shining.",
        "This is the worst day ever, everything went wrong.",
        "I'm okay, just a regular day nothing special.",
        "I'm so excited for the weekend!",
        "I feel anxious about the upcoming exam.",
        "The news today made me really sad.",
        "I'm in love with this beautiful weather!",
        "That movie scared me so much!",
        "Work is stressing me out lately.",
        "I accomplished so much today, feeling proud!",
        "My dog just died, I'm devastated.",
        "Got promoted at work today! Best day of my life!",
        "Not sure how I feel, just numb I guess.",
        "The concert was amazing, I'm still buzzing!",
        "Another boring day in quarantine..."
    ],
    'mood': [
        'happy', 'sad', 'neutral', 'excited', 'anxious', 
        'sad', 'happy', 'fear', 'stressed', 'proud',
        'grief', 'ecstatic', 'numb', 'energized', 'bored'
    ]
}

df = pd.DataFrame(data)

# Create and train model
model = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('clf', LinearSVC())
])
model.fit(df['text'], df['mood'])

# Mood history storage
MOOD_HISTORY_FILE = 'mood_history.json'

def load_mood_history():
    if os.path.exists(MOOD_HISTORY_FILE):
        with open(MOOD_HISTORY_FILE, 'r') as f:
            return json.load(f)
    return []

def save_mood_history(history):
    with open(MOOD_HISTORY_FILE, 'w') as f:
        json.dump(history, f)

def predict_mood(text):
    # Get model prediction
    prediction = model.predict([text])[0]
    
    # Get sentiment scores
    sentiment = sia.polarity_scores(text)
    
    return {
        'mood': prediction,
        'confidence': np.max(model.decision_function([text])),
        'sentiment': sentiment,
        'timestamp': str(datetime.now()),
        'text': text
    }

def show_mood_history(history):
    if not history:
        print("\nNo mood history yet!")
        return
    
    # Convert to DataFrame for easier analysis
    history_df = pd.DataFrame(history)
    history_df['date'] = pd.to_datetime(history_df['timestamp']).dt.date
    
    # Mood frequency
    print("\n=== Your Mood Distribution ===")
    mood_counts = history_df['mood'].value_counts()
    print(mood_counts)
    
    # Plot mood over time
    plt.figure(figsize=(12, 6))
    mood_over_time = history_df.groupby('date')['mood'].value_counts().unstack().fillna(0)
    mood_over_time.plot(kind='area', stacked=True, alpha=0.5)
    plt.title('Your Mood Over Time')
    plt.ylabel('Mood Intensity')
    plt.xlabel('Date')
    plt.tight_layout()
    plt.show()
    
    # Sentiment trend
    sentiment_df = pd.json_normalize(history_df['sentiment'])
    sentiment_df['date'] = history_df['date']
    weekly_sentiment = sentiment_df.groupby('date').mean()
    
    plt.figure(figsize=(12, 6))
    plt.plot(weekly_sentiment.index, weekly_sentiment['compound'], marker='o')
    plt.title('Your Overall Sentiment Trend')
    plt.ylabel('Sentiment Score (-1 to 1)')
    plt.xlabel('Date')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.grid()
    plt.show()
    
    # Word cloud of your journal entries
    all_text = " ".join(history_df['text'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)
    
    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud of Your Journal Entries')
    plt.show()

def mood_analysis_dashboard(history):
    if not history:
        print("\nNo data for dashboard yet!")
        return
    
    history_df = pd.DataFrame(history)
    history_df['date'] = pd.to_datetime(history_df['timestamp'])
    
    # Mood distribution pie chart
    plt.figure(figsize=(18, 12))
    
    plt.subplot(2, 2, 1)
    mood_counts = history_df['mood'].value_counts()
    mood_counts.plot.pie(autopct='%1.1f%%', startangle=90)
    plt.title('Mood Distribution')
    plt.ylabel('')
    
    # Daily mood heatmap
    plt.subplot(2, 2, 2)
    history_df['day_of_week'] = history_df['date'].dt.day_name()
    history_df['hour'] = history_df['date'].dt.hour
    mood_hour = pd.crosstab(history_df['day_of_week'], history_df['hour'])
    
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    mood_hour = mood_hour.reindex(days_order)
    
    sns.heatmap(mood_hour, cmap='YlOrRd')
    plt.title('When Do You Typically Log Moods?')
    plt.xlabel('Hour of Day')
    plt.ylabel('Day of Week')
    
    # Sentiment distribution
    plt.subplot(2, 2, 3)
    sentiment_df = pd.json_normalize(history_df['sentiment'])
    sns.kdeplot(data=sentiment_df[['neg', 'neu', 'pos']], fill=True)
    plt.title('Sentiment Distribution')
    plt.xlabel('Sentiment Score')
    
    # Mood over time (small multiples)
    plt.subplot(2, 2, 4)
    top_moods = mood_counts.index[:4]  # Show top 4 moods
    for mood in top_moods:
        mood_data = history_df[history_df['mood'] == mood]
        mood_daily = mood_data.resample('D', on='date').count()
        plt.plot(mood_daily.index, mood_daily['mood'], label=mood)
    
    plt.title('Top Moods Over Time')
    plt.ylabel('Entries per Day')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def suggest_activities(mood):
    suggestions = {
        'happy': [
            "Keep spreading positivity!",
            "Try a new hobby to channel this energy",
            "Share your happiness with someone"
        ],
        'sad': [
            "Practice self-care with a warm bath",
            "Call a friend or loved one",
            "Write about what's bothering you"
        ],
        'anxious': [
            "Try 5 minutes of deep breathing",
            "Go for a walk in nature",
            "Practice grounding techniques"
        ],
        'stressed': [
            "Do 10 minutes of stretching",
            "Try a mindfulness meditation",
            "Prioritize your tasks and take breaks"
        ],
        'excited': [
            "Channel this energy into a creative project",
            "Plan something fun to look forward to",
            "Share your excitement with others"
        ]
    }
    
    default_suggestions = [
        "Journal about how you're feeling",
        "Take 5 deep breaths",
        "Drink some water and stretch"
    ]
    
    return suggestions.get(mood.lower(), default_suggestions)

def main():
    mood_history = load_mood_history()
    
    print("\n" + "="*50)
    print(" AI MOOD TRACKER & PREDICTOR ".center(50, "="))
    print("="*50)
    
    while True:
        print("\nMAIN MENU:")
        print("1. Log how you're feeling")
        print("2. View mood history")
        print("3. Mood analysis dashboard")
        print("4. Get mood insights")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ")
        
        if choice == '1':
            print("\nHow are you feeling today? (Type your thoughts)")
            print("Example: 'I'm feeling optimistic about my new project'")
            print("Or just describe your day in a few sentences.")
            user_input = input("\nYour entry: ")
            
            if user_input.lower() in ['quit', 'exit']:
                continue
                
            analysis = predict_mood(user_input)
            mood_history.append(analysis)
            save_mood_history(mood_history)
            
            print(f"\nAI Analysis:")
            print(f"- Predicted Mood: {analysis['mood'].upper()}")
            print(f"- Confidence: {analysis['confidence']:.2f}")
            print(f"- Sentiment: {analysis['sentiment']}")
            
            print("\nSuggested Activities:")
            for i, suggestion in enumerate(suggest_activities(analysis['mood']), 1):
                print(f"{i}. {suggestion}")
                
        elif choice == '2':
            show_mood_history(mood_history)
            
        elif choice == '3':
            mood_analysis_dashboard(mood_history)
            
        elif choice == '4':
            if not mood_history:
                print("\nNot enough data yet. Keep logging your moods!")
                continue
                
            print("\nMOOD INSIGHTS:")
            # Simple insights based on history
            history_df = pd.DataFrame(mood_history)
            most_common_mood = history_df['mood'].mode()[0]
            sentiment_trend = pd.json_normalize(history_df['sentiment'])['compound'].mean()
            
            print(f"- Your most common mood: {most_common_mood}")
            print(f"- Your overall sentiment trend: {'positive' if sentiment_trend > 0.05 else 'negative' if sentiment_trend < -0.05 else 'neutral'}")
            
            # Detect recent changes
            if len(mood_history) > 5:
                recent = mood_history[-5:]
                recent_moods = [entry['mood'] for entry in recent]
                if all(m == 'happy' for m in recent_moods):
                    print("- You've been consistently happy recently! 😊")
                elif all(m in ['sad', 'anxious'] for m in recent_moods):
                    print("- You've been feeling down recently. Consider reaching out for support.")
            
        elif choice == '5':
            print("\nGoodbye! Your mood history has been saved.")
            break
            
        else:
            print("\nInvalid choice. Please enter 1-5.")

if __name__ == "__main__":
    main()