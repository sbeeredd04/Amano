import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import numpy as np
from sqlalchemy import select
from models.song_model import Song, UserHistory, UserMood
from utils.db import get_session
import threading

# DQN Model Definition
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Data Preprocessing Helper Function
def preprocess_data(df):
    """
    Preprocess the song dataset, scale features, and prepare the final DataFrame for filtering.
    """
    df = df.dropna()
    df = df.drop(['duration_ms', 'explicit', 'mode', 'liveness', 'loudness', 'time_signature', 'key'], axis=1)
    df.rename(columns={'Unnamed: 0': 'song_id'}, inplace=True)

    features_to_scale = ['popularity', 'danceability', 'energy', 'acousticness', 'valence', 'tempo', 'speechiness', 'instrumentalness']
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(df[features_to_scale])
    df_scaled = pd.DataFrame(scaled_features, columns=features_to_scale)

    df_scaled['song_id'] = df['song_id']
    df_scaled['track_id'] = df['track_id']
    df_scaled['artist_name'] = df['artists'].fillna('')
    df_scaled['track_name'] = df['track_name']
    df_scaled['album_name'] = df['album_name'].fillna('')
    df_scaled['track_genre'] = df['track_genre'].fillna('')

    return df_scaled.drop_duplicates()

# Helper Function to Get User's Songs from Database and Initialize Playlist
def get_user_playlist_from_db(user_id, df_scaled):
    """
    Fetch the user's liked songs (reward = 1) from the UserHistory table and initialize a playlist by looking up the songs in the dataset.
    """
    session = get_session()
    with session:
        statement = select(UserHistory).filter_by(user_id=user_id, reward=1)  # Only select liked songs (reward=1)
        user_history = session.scalars(statement).all()

    # Initialize user playlist based on the song_id from the dataset
    user_playlist = []
    for record in user_history:
        song_record = df_scaled[df_scaled['song_id'] == record.song_id]
        if not song_record.empty:
            user_playlist.append(song_record['song_id'].values[0])
    
    return user_playlist, len(user_history)


def update_user_feedback(user_id, feedback):
    """
    Updates the user's feedback (reward) for songs in the UserHistory table.
    :param user_id: The user's ID
    :param feedback: A list of song feedback, each containing song_id, reward (1 or -1), and mood
    """
    session = get_session()
    with session:
        for item in feedback:
            song_id = item['song_id']
            reward = item['reward']  # Use 'reward' instead of 'liked'
            mood = item['mood']

            # Check if the song already exists in the history
            existing_record = session.query(UserHistory).filter_by(user_id=user_id, song_id=song_id).first()

            if existing_record:
                # Update existing record
                existing_record.reward = reward  # Update 'reward'
                existing_record.mood = mood
            else:
                # Insert a new record
                new_history = UserHistory(user_id=user_id, song_id=song_id, reward=reward, mood=mood)
                session.add(new_history)
        
        session.commit()

# Filter Songs Based on Similarity
def recommend_songs_filtered(user_songs, df, features, feature_weights, top_n=0):
    """
    Filter and recommend songs based on cosine similarity, weighted features, and the user's interaction history.
    """
    df_copy = df.copy()
    for feature in features:
        df_copy[feature] = df_copy[feature] * feature_weights.get(feature, 1.0)

    user_songs_df = df_copy[df_copy['song_id'].isin(user_songs)]
    filtered_df = df_copy[df_copy['track_genre'].isin(user_songs_df['track_genre'].unique())]
    
    similarity_matrix = cosine_similarity(filtered_df[features], user_songs_df[features])
    aggregated_similarities = similarity_matrix.mean(axis=1)
    filtered_df['similarity'] = aggregated_similarities
    recommendations = filtered_df[~filtered_df['song_id'].isin(user_songs)].sort_values(by=['similarity', 'popularity'], ascending=[False, False])
    
    if top_n > 0:
        recommendations = recommendations.head(top_n)
    
    recommendations = pd.concat([recommendations, user_songs_df])

    
    return recommendations

# Function to Fetch User Interaction History and Generate Immediate Recommendations
def fetch_user_history_and_recommend(user_id, df_scaled, features, feature_weights, default_mood='Calm'):
    """
    Serve immediate recommendations without retraining. Use the latest trained model to serve recommendations.
    The mood will default to 'Calm' if it has not been set by the user.
    """
    # Get user mood from the database or set to default mood ('Calm')
    user_mood = get_user_mood(user_id)
    if not user_mood:
        user_mood = default_mood

    # Initialize user's playlist from the song database and get the length of user history
    user_songs, history_length = get_user_playlist_from_db(user_id, df_scaled)

    # If no user songs are found, recommend popular tracks as a fallback
    if not user_songs:
        user_songs = df_scaled['song_id'].sample(10).tolist()  # Pick 10 random songs for initial recommendation

    # If enough user history exists, start DQN model
    if history_length >= 10:
        run_background_training(user_id, df_scaled, features, feature_weights)

    # Filter recommendations based on the user's history or default mood
    recommended_songs_df = recommend_songs_filtered(user_songs, df_scaled, features, feature_weights, top_n=10)

    return recommended_songs_df

# DQN Initialization Function
def init_dqn_model(state_size, action_size):
    policy_net = DQN(state_size, action_size)
    target_net = DQN(state_size, action_size)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.Adam(policy_net.parameters())
    memory = deque(maxlen=10000)
    return policy_net, target_net, optimizer, memory

# Background Training Function
def background_train_dqn(user_id, df_scaled, features, feature_weights):
    """
    Train the DQN model in the background based on user feedback.
    """
    print(f"Starting background training for user {user_id}...")
    
    # Initialize user's playlist from the song database
    user_songs, _ = get_user_playlist_from_db(user_id, df_scaled)
    recommended_songs_df = recommend_songs_filtered(user_songs, df_scaled, features, feature_weights, top_n=200)
    action_size = len(recommended_songs_df)

    policy_net, target_net, optimizer, memory = init_dqn_model(9, action_size)
    
    eps_start = 1.0  # Initial exploration rate
    eps_end = 0.1  # Minimum exploration rate
    eps_decay = 0.995  # Decay rate for exploration-exploitation balance
    batch_size = 32  # Mini-batch size
    gamma = 0.99  # Discount factor
    target_update = 10  # Frequency of updating the target network

    eps_threshold = eps_start  # Initialize epsilon threshold
    num_episodes = 10  # Number of training episodes

    # Get the user's current mood as one of the 9 moods
    user_mood = get_user_mood(user_id)
    mood_state = convert_mood_to_state(user_mood)  # Convert mood to an integer state

    for episode in range(num_episodes):
        state = get_initial_state(mood_state)  # Initialize state with user mood and song features
        
        for t in range(10):  # Generate 10 recommendations per episode
            action = select_action(state, eps_threshold, action_size)  # Select a song to recommend
            next_state, reward = get_user_feedback(action, recommended_songs_df, mood_state)

            if next_state == 'exit':
                break  # Exit training early
            
            memory.append((state, action, reward, next_state))
            optimize_model(policy_net, target_net, memory, optimizer, batch_size, gamma)
            state = next_state
        
        eps_threshold = max(eps_end, eps_threshold * eps_decay)
        
        if episode % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())
    
    print(f"Background training completed for user {user_id}.")

# Function to run training in the background
def run_background_training(user_id, df_scaled, features, feature_weights):
    """
    Run the DQN training process in a separate thread, allowing recommendations to be served immediately.
    """
    training_thread = threading.Thread(target=background_train_dqn, args=(user_id, df_scaled, features, feature_weights))
    training_thread.start()

# Convert mood to an integer state for DQN model
def convert_mood_to_state(mood):
    """
    Convert mood to an integer state.
    Moods: Angry, Content, Happy, Delighted, Calm, Sleepy, Sad, Depressed, Excited
    """
    mood_mapping = {
        'Angry': 1,
        'Content': 2,
        'Happy': 3,
        'Delighted': 4,
        'Calm': 5,
        'Sleepy': 6,
        'Sad': 7,
        'Depressed': 8,
        'Excited': 9
    }
    return mood_mapping.get(mood, 5)  # Default to 'Calm' if mood not found

# DQN Model Optimization Function
def optimize_model(policy_net, target_net, memory, optimizer, batch_size, gamma):
    if len(memory) < batch_size:
        return
    transitions = random.sample(memory, batch_size)
    batch_state, batch_action, batch_reward, batch_next_state = zip(*transitions)

    batch_state = torch.tensor(batch_state).float()
    batch_action = torch.tensor(batch_action).long()
    batch_reward = torch.tensor(batch_reward).float()
    batch_next_state = torch.tensor(batch_next_state).float()

    q_values = policy_net(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
    next_q_values = target_net(batch_next_state).max(1)[0]
    expected_q_values = batch_reward + (gamma * next_q_values)

    loss = nn.functional.mse_loss(q_values, expected_q_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Action Selection Based on Exploration-Exploitation Tradeoff
def select_action(state, eps_threshold, action_size):
    if random.random() > eps_threshold:
        with torch.no_grad():
            return policy_net(torch.tensor(state).float()).argmax().item()
    else:
        return random.randrange(action_size)

# Function to update user mood in the database
def update_user_mood(user_id, mood):
    """
    Update the user's current mood.
    :param user_id: The user's ID
    :param mood: The current mood (e.g., 'Happy', 'Sad', etc.)
    """
    session = get_session()
    with session:
        existing_mood = session.query(UserMood).filter_by(user_id=user_id).first()

        if existing_mood:
            existing_mood.mood = mood
        else:
            new_mood = UserMood(user_id=user_id, mood=mood)
            session.add(new_mood)
        
        session.commit()

# Function to get user mood from the database
def get_user_mood(user_id):
    """
    Fetch the user's current mood.
    :param user_id: The user's ID
    :return: String representing the current mood (e.g., 'Happy', 'Sad', etc.)
    """
    session = get_session()
    with session:
        existing_mood = session.query(UserMood).filter_by(user_id=user_id).first()
        if existing_mood:
            return existing_mood.mood  # Return mood as a string
        else:
            return 'Calm'  # Default to 'Calm' if no mood is found