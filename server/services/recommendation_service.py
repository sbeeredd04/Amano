from venv import logger
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
from server.routes.playlist import get_user_songs
import logging
import os

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Define the global list of fallback user songs
fallback_user_songs = [67016, 91000, 81004, 17000, 20414, 81000, 81074, 81109, 20652, 
                        91016, 91017, 91018, 51150, 51503, 56064, 33012, 57162, 53050, 
                        67351, 51450, 94632, 51500, 53055]

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
    logging.debug("Preprocessing data...")

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

    logging.debug(f"Data preprocessing complete. Shape of scaled data: {df_scaled.shape}")
    return df_scaled.drop_duplicates()

# Helper Function to Fetch User Songs Dynamically
def get_user_playlist_from_db(user_id, df_scaled):
    logging.debug(f"Fetching user {user_id}'s playlist from database...")

    user_songs = get_user_songs(user_id)
    if not user_songs:
        logging.warning(f"No playlists found for user {user_id}. Falling back to global song list.")
        user_songs = fallback_user_songs

    # Filter the songs that exist in the dataframe
    valid_user_songs = [song_id for song_id in user_songs if not df_scaled[df_scaled['song_id'] == song_id].empty]

    logging.debug(f"User {user_id}'s playlist: {valid_user_songs}")
    return valid_user_songs, len(valid_user_songs)

# Update User Feedback
def update_user_feedback(user_id, feedback):
    """Update user feedback and track changes."""
    logger.info(f"=== Updating Feedback for User {user_id} ===")
    
    session = get_session()
    try:
        for item in feedback:
            song_id = item['song_id']
            reward = item['reward']
            mood = item['mood']
            
            logger.info(f"Processing feedback item:")
            logger.info(f"- Song ID: {song_id}")
            logger.info(f"- Reward: {reward}")
            logger.info(f"- Mood: {mood}")

            # Update or create user history
            history = session.query(UserHistory).filter_by(
                user_id=user_id,
                song_id=song_id
            ).first()

            if history:
                logger.info(f"Updating existing feedback - Old reward: {history.reward}, New reward: {reward}")
                history.reward = reward
                history.mood = mood
            else:
                logger.info("Creating new feedback entry")
                new_history = UserHistory(
                    user_id=user_id,
                    song_id=song_id,
                    reward=reward,
                    mood=mood
                )
                session.add(new_history)

        session.commit()
        logger.info("Feedback update committed successfully")

    except Exception as e:
        session.rollback()
        logger.error(f"Error updating user feedback: {e}", exc_info=True)
        raise
    finally:
        session.close()

# Filter Songs Based on Similarity and Append User Playlist Songs
def recommend_songs_filtered(user_songs, df, features, feature_weights, top_n=0):
    logging.debug("Filtering and recommending songs based on similarity...")

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
        recommendations = recommendations.head(top_n // 2)  # 50% new songs
        user_songs_df = user_songs_df.sample(min(len(user_songs_df), top_n // 2))  # 50% user playlist songs

    final_recommendations = pd.concat([user_songs_df, recommendations]).drop_duplicates()

    logging.debug(f"Generated {len(final_recommendations)} song recommendations.")
    logging.debug(f"Recommendations: \n{final_recommendations[['song_id', 'track_name', 'similarity']].head(10)}")

    return final_recommendations

# Fetch User History and Generate Immediate Recommendations
def fetch_user_history_and_recommend(user_id, mood=None, use_user_songs=True, df_scaled=None):
    """
    Fetch user history and generate recommendations.
    """
    logger.info("=== Starting Recommendation Generation ===")
    logger.info(f"Parameters: user_id={user_id}, mood={mood}, use_user_songs={use_user_songs}")
    
    if df_scaled is None:
        logger.info("Loading dataset as it wasn't provided")
        df_scaled = load_and_get_dataset()
    
    try:
        session = get_session()
        
        if use_user_songs:
            logger.info("Using user's song history for recommendations")
            history = session.query(UserHistory).filter_by(user_id=user_id).all()
            
            if not history:
                logger.info("No user history found, falling back to default recommendations")
                return get_default_recommendations(df_scaled)
            
            logger.info(f"Found {len(history)} historical interactions")
            
            # Try DQN model first
            try:
                logger.info("Attempting to use DQN model for recommendations")
                model_path = f"models/dqn_{user_id}.pth"
                
                if os.path.exists(model_path):
                    logger.info("Found existing DQN model")
                    # Convert mood to state
                    mood_state = convert_mood_to_state(mood)
                    logger.info(f"Converted mood '{mood}' to state: {mood_state}")
                    
                    recommendations = get_dqn_recommendations(user_id, mood_state, df_scaled)
                    if recommendations is not None and len(recommendations) > 0:
                        logger.info("Successfully generated DQN recommendations")
                        logger.debug(f"First 3 DQN recommendations: {recommendations[:3]}")
                        return recommendations
                else:
                    logger.info("No DQN model found for user")
            except Exception as e:
                logger.warning(f"DQN recommendation failed: {e}", exc_info=True)
            
            # Fallback to similarity-based recommendations
            logger.info("Falling back to similarity-based recommendations")
            recommendations = process_user_history(history, mood)
            logger.info(f"Generated {len(recommendations)} similarity-based recommendations")
            return recommendations
        else:
            logger.info("Using default recommendations")
            recommendations = get_default_recommendations(df_scaled)
            logger.info(f"Generated {len(recommendations)} default recommendations")
            return recommendations
            
    except Exception as e:
        logger.error(f"Error in fetch_user_history_and_recommend: {e}", exc_info=True)
        raise

def process_user_history(history, mood=None):
    """Process user history to generate recommendations."""
    try:
        # Get positive feedback songs
        liked_songs = [h.song_id for h in history if h.reward > 0]
        if not liked_songs:
            return get_default_recommendations()
        
        # Get similar songs based on liked songs
        similar_songs = df_scaled[df_scaled['song_id'].isin(liked_songs)]
        if similar_songs.empty:
            return get_default_recommendations()
            
        # Sample recommendations
        recommendations = df_scaled.sample(n=min(10, len(df_scaled)))
        return recommendations.to_dict('records')
        
    except Exception as e:
        logger.error(f"Error processing user history: {e}")
        return get_default_recommendations()

# Background Training Function
def background_train_dqn(user_id, df_scaled, features, feature_weights):
    """Train DQN model with detailed logging."""
    logger.info(f"=== Starting DQN Training for User {user_id} ===")
    
    try:
        user_songs, count = get_user_playlist_from_db(user_id, df_scaled)
        logger.info(f"Retrieved {count} songs from user's playlist")
        
        recommended_songs_df = recommend_songs_filtered(user_songs, df_scaled, features, feature_weights, top_n=200)
        action_size = len(recommended_songs_df)
        logger.info(f"Action space size: {action_size}")

        # Initialize DQN components
        policy_net, target_net, optimizer, memory = init_dqn_model(1, action_size)
        logger.info("DQN model initialized")

        # Training parameters
        eps_start = 1.0
        eps_end = 0.1
        eps_decay = 0.995
        batch_size = 32
        gamma = 0.99
        target_update = 10
        eps_threshold = eps_start
        num_episodes = 10

        # Get initial state
        user_mood = get_user_mood(user_id)
        state = convert_mood_to_state(user_mood)
        logger.info(f"Initial state (mood): {user_mood} -> {state}")

        # Training loop
        total_reward = 0
        for episode in range(num_episodes):
            episode_reward = 0
            logger.info(f"\nStarting Episode {episode + 1}/{num_episodes}")
            logger.info(f"Epsilon: {eps_threshold:.4f}")

            for t in range(10):
                # Select and perform action
                action = select_action([state], eps_threshold, action_size, policy_net)
                next_state, reward = get_next_state_and_reward(action, recommended_songs_df)
                episode_reward += reward

                logger.debug(f"Step {t + 1}: Action={action}, Reward={reward}")

                # Store transition and optimize
                memory.append(([state], action, reward, [next_state]))
                optimize_model(policy_net, target_net, memory, optimizer, batch_size, gamma)

                state = next_state

            total_reward += episode_reward
            avg_reward = episode_reward / 10
            logger.info(f"Episode {episode + 1} complete:")
            logger.info(f"- Average Reward: {avg_reward:.4f}")
            logger.info(f"- Total Reward: {episode_reward}")

            # Update epsilon and target network
            eps_threshold = max(eps_end, eps_threshold * eps_decay)
            if episode % target_update == 0:
                target_net.load_state_dict(policy_net.state_dict())
                logger.info("Target network updated")

        # Save model and log final metrics
        torch.save(policy_net.state_dict(), f"models/dqn_{user_id}.pth")
        logger.info("\n=== Training Complete ===")
        logger.info(f"Final Metrics:")
        logger.info(f"- Total Reward: {total_reward}")
        logger.info(f"- Average Reward per Episode: {total_reward/num_episodes:.4f}")
        logger.info(f"- Final Epsilon: {eps_threshold:.4f}")
        logger.info(f"Model saved to models/dqn_{user_id}.pth")

    except Exception as e:
        logger.error(f"Error in DQN training: {e}", exc_info=True)
        raise

# DQN Model Optimization Function
def optimize_model(policy_net, target_net, memory, optimizer, batch_size, gamma):
    if len(memory) < batch_size:
        return
    transitions = random.sample(memory, batch_size)
    batch_state, batch_action, batch_reward, batch_next_state = zip(*transitions)

    # Convert the batch to tensors (now using scalar states)
    batch_state = torch.tensor(batch_state).float()  # Ensure states are 1D tensors
    batch_action = torch.tensor(batch_action).long()
    batch_reward = torch.tensor(batch_reward).float()
    batch_next_state = torch.tensor(batch_next_state).float()

    # Compute Q-values
    q_values = policy_net(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
    next_q_values = target_net(batch_next_state).max(1)[0]
    expected_q_values = batch_reward + (gamma * next_q_values)

    # Optimize the model
    loss = nn.functional.mse_loss(q_values, expected_q_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def select_action(state, eps_threshold, action_size, policy_net):
    if random.random() > eps_threshold:
        with torch.no_grad():
            return policy_net(torch.tensor(state).float()).argmax().item()  # Use policy_net to predict action
    else:
        return random.randrange(action_size)  # Exploration: return random action

# Function to update user mood in the database
def update_user_mood(user_id, mood):
    """Update user's current mood."""
    logger.info(f"=== Updating Mood for User {user_id} to {mood} ===")
    
    session = get_session()
    try:
        # Find existing mood entry
        existing_mood = session.query(UserMood).filter_by(user_id=user_id).first()
        
        if existing_mood:
            logger.info(f"Updating existing mood from {existing_mood.mood} to {mood}")
            existing_mood.mood = mood
        else:
            logger.info(f"Creating new mood entry for user")
            new_mood = UserMood(user_id=user_id, mood=mood)
            session.add(new_mood)
        
        session.commit()
        logger.info("Mood update successful")
        return True
        
    except Exception as e:
        logger.error(f"Error updating mood: {e}", exc_info=True)
        session.rollback()
        raise
    finally:
        session.close()

# Function to get user mood from the database
def get_user_mood(user_id):
    """Get user's current mood."""
    logger.info(f"=== Fetching Mood for User {user_id} ===")
    
    session = get_session()
    try:
        user_mood = session.query(UserMood).filter_by(user_id=user_id).first()
        mood = user_mood.mood if user_mood else "Calm"
        logger.info(f"Current mood: {mood}")
        return mood
    except Exception as e:
        logger.error(f"Error getting user mood: {e}", exc_info=True)
        return "Calm"  # Default mood
    finally:
        session.close()

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

import numpy as np

def get_initial_state(mood_state):
    """
    Return a one-hot encoded vector representing the user's mood state.
    Mood state is expected to be an integer in the range [1, 9].
    """
    state_size = 9
    state = np.zeros(state_size)
    state[mood_state - 1] = 1  # Set the corresponding index for the mood state to 1
    return state  # Return the one-hot encoded state

def get_next_state_and_reward(action, recommended_songs_df):
    """
    Simulate the next state and reward based on the action.
    Here, the action corresponds to a recommended song, and the reward is based on user feedback or random values.
    """
    song = recommended_songs_df.iloc[action]  # Get the song from the recommended songs
    
    # For now, let's assume reward is randomly assigned for simulation purposes
    reward = random.choice([-1, 0, 1])  # -1: dislike, 0: neutral, 1: like
    
    # For the next state, return the current song's features (to simulate mood state transition)
    next_state = convert_mood_to_state('Calm')  # Replace with actual logic if necessary

    return next_state, reward

def get_initial_recommendations(user_id, use_user_songs=True, df_scaled=None):
    """
    Get initial recommendations based on user's playlist history or default songs.
    Args:
        user_id: The user's ID
        use_user_songs: Whether to use user's song history
        df_scaled: Pre-loaded scaled dataset (optional)
    """
    logger.info(f"Getting initial recommendations for user {user_id}")
    try:
        session = get_session()
        
        if use_user_songs:
            # Check if user has any history
            user_history = session.query(UserHistory).filter_by(user_id=user_id).all()
            if user_history:
                logger.info("User has history, using personalized recommendations")
                # Use user's history for recommendations
                return fetch_user_history_and_recommend(
                    user_id=user_id,
                    use_user_songs=True,
                    df_scaled=df_scaled
                )
        
        # If no user history or use_user_songs is False, use default recommendations
        logger.info("Using default recommendations")
        return get_default_recommendations(df_scaled)
        
    except Exception as e:
        logger.error(f"Error getting initial recommendations: {e}")
        raise
    finally:
        session.close()

def get_default_recommendations(df_scaled):
    """Get default recommendations."""
    logger.info("Generating default recommendations")
    try:
        # Sample random songs
        recommendations = df_scaled.sample(n=10).to_dict('records')
        logger.info(f"Generated {len(recommendations)} default recommendations")
        logger.debug(f"First 3 default recommendations: {recommendations[:3]}")
        return recommendations
    except Exception as e:
        logger.error(f"Error getting default recommendations: {e}", exc_info=True)
        raise

def get_dqn_recommendations(user_id, state, df_scaled, top_n=10):
    """Get recommendations using the trained DQN model."""
    try:
        model = DQN(state_size=1, action_size=len(df_scaled))
        model.load_state_dict(torch.load(f"models/dqn_{user_id}.pth"))
        model.eval()

        with torch.no_grad():
            state_tensor = torch.tensor([[state]], dtype=torch.float32)
            q_values = model(state_tensor)
            
            # Get top N actions (song indices) based on Q-values
            top_actions = q_values.squeeze().argsort(descending=True)[:top_n]
            
            # Convert actions to song recommendations
            recommended_songs = df_scaled[df_scaled['song_id'].isin([df_scaled['song_id'][i] for i in top_actions])]
            
            return recommended_songs

    except Exception as e:
        logging.error(f"Error getting DQN recommendations: {e}")
        raise

def load_and_get_dataset():
    """Load and return the preprocessed dataset."""
    try:
        dataset_path = "/Users/sriujjwalreddyb/Amano/spotify_Song_Dataset/final_dataset.csv"
        logger.info(f"Loading dataset from {dataset_path}")
        
        if not os.path.exists(dataset_path):
            logger.error(f"Dataset file not found at {dataset_path}")
            raise FileNotFoundError(f"Dataset not found at {dataset_path}")
            
        df = pd.read_csv(dataset_path)
        logger.info(f"Successfully loaded dataset with {len(df)} records")
        return df
        
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise

def run_background_training(user_id, df_scaled=None, features=None, feature_weights=None):
    """
    Run background DQN training for a user based on their feedback history.
    """
    logger.info(f"=== Starting Background Training for User {user_id} ===")
    
    try:
        if df_scaled is None:
            logger.info("Loading dataset as it wasn't provided")
            df_scaled = load_and_get_dataset()
            
        if features is None:
            features = ['energy', 'acousticness', 'valence', 'tempo', 'speechiness', 'instrumentalness']
            logger.info(f"Using default features: {features}")
            
        if feature_weights is None:
            feature_weights = {
                'energy': 1.0,
                'acousticness': 5.0,
                'valence': 5.0,
                'tempo': 5.0,
                'instrumentalness': 5.0,
                'speechiness': 5.0
            }
            logger.info("Using default feature weights")

        # Get user's current mood
        user_mood = get_user_mood(user_id)
        logger.info(f"Current user mood: {user_mood}")

        # Train the DQN model
        logger.info("Starting DQN training")
        background_train_dqn(
            user_id=user_id,
            df_scaled=df_scaled,
            features=features,
            feature_weights=feature_weights
        )
        
        logger.info("Background training completed successfully")
        return True

    except Exception as e:
        logger.error(f"Error in background training: {e}", exc_info=True)
        raise
