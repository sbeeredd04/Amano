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
from models.song_model import Song, UserHistory, UserMood, Playlist, PlaylistSong
from utils.db import get_session, get_dataset
import logging
import os
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from collections import Counter

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
def fetch_user_history_and_recommend(user_id, mood=None, use_user_songs=True):
    """Updated recommendation pipeline with clustering."""
    logger.info(f"\n=== Generating Recommendations for User {user_id} ===")
    
    try:
        df_scaled = get_dataset()
        session = get_session()
        
        try:
            # Check for trained model
            model_path = f"models/user_{user_id}_dqn.pth"
            has_trained_model = os.path.exists(model_path)
            logger.info(f"Trained model exists: {has_trained_model}")
            
            if use_user_songs:
                user_songs = get_all_user_playlist_songs(user_id)
                logger.info(f"Found {len(user_songs)} songs in user's playlists")
                
                if user_songs:
                    if has_trained_model:
                        # Get both types of recommendations
                        cluster_recommendations = get_cluster_based_recommendations(
                            user_songs=user_songs,
                            df_scaled=df_scaled
                        )
                        
                        dqn_recommendations = get_dqn_recommendations(
                            user_id=user_id,
                            df_scaled=df_scaled,
                            user_songs=user_songs,
                            mood=mood
                        )
                        
                        # Combine recommendations
                        recommendations = combine_recommendations(
                            dqn_recommendations,
                            cluster_recommendations
                        )
                    else:
                        # Use only cluster-based recommendations
                        recommendations = get_cluster_based_recommendations(
                            user_songs=user_songs,
                            df_scaled=df_scaled
                        )
                else:
                    recommendations = get_default_recommendations(df_scaled)
            else:
                recommendations = get_default_recommendations(df_scaled)
                
            return recommendations
            
        finally:
            session.close()
            
    except Exception as e:
        logger.error(f"Error generating recommendations: {str(e)}", exc_info=True)
        raise

def combine_recommendations(dqn_recs, similarity_recs, dqn_weight=0.7):
    """Combine DQN and similarity-based recommendations."""
    logger.info("\n=== Combining Recommendations ===")
    logger.info(f"DQN recommendations: {len(dqn_recs)}")
    logger.info(f"Similarity recommendations: {len(similarity_recs)}")
    
    try:
        # Combine and weight recommendations
        combined = []
        seen_songs = set()
        
        # Add DQN recommendations first
        for rec in dqn_recs:
            if rec['song_id'] not in seen_songs:
                rec['score'] = rec.get('score', 0) * dqn_weight
                combined.append(rec)
                seen_songs.add(rec['song_id'])
        
        # Add similarity recommendations
        for rec in similarity_recs:
            if rec['song_id'] not in seen_songs:
                rec['score'] = rec.get('similarity', 0) * (1 - dqn_weight)
                combined.append(rec)
                seen_songs.add(rec['song_id'])
        
        # Sort by score
        combined.sort(key=lambda x: x.get('score', 0), reverse=True)
        
        logger.info(f"Combined {len(combined)} unique recommendations")
        return combined
            
    except Exception as e:
        logger.error(f"Error combining recommendations: {str(e)}", exc_info=True)
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
    Get initial recommendations based on either user's playlist songs or default songs.
    """
    logger.info(f"=== Getting Initial Recommendations for User {user_id} ===")
    logger.info(f"Using user songs: {use_user_songs}")
    
    try:
        # Check for existing DQN model
        model_path = f"models/dqn_{user_id}.pth"
        if os.path.exists(model_path):
            logger.info("Found existing DQN model - using it for recommendations")
            return get_dqn_based_recommendations(user_id, df_scaled)
            
        # If no model exists, proceed with initial recommendation logic
        if use_user_songs:
            # Get all songs from user's playlists
            user_songs = get_all_user_playlist_songs(user_id)
            if not user_songs:
                logger.info("No user playlist songs found - using defaults")
                return get_default_recommendations(df_scaled)
                
            logger.info(f"Found {len(user_songs)} songs in user's playlists")
            return get_similarity_based_recommendations(user_songs, df_scaled)
        else:
            logger.info("Using default recommendations")
        return get_default_recommendations(df_scaled)
        
    except Exception as e:
        logger.error(f"Error in initial recommendations: {e}")
        return get_default_recommendations(df_scaled)

def get_similarity_based_recommendations(user_songs, df_scaled=None, n_new_recommendations=10, n_familiar_recommendations=5):
    """Get recommendations based on cosine similarity with user's playlist songs."""
    if df_scaled is None:
        df_scaled = get_dataset()
    
    logger.info("\n=== Generating Similarity-based Recommendations ===")
    logger.info(f"User songs count: {len(user_songs)}")
    
    try:
        features = ['energy', 'acousticness', 'valence', 'tempo', 'speechiness', 'instrumentalness']
        
        # Log the dataset info
        logger.info("\nDataset Information:")
        logger.info(f"Total songs in dataset: {len(df_scaled)}")
        logger.info(f"Dataset columns: {df_scaled.columns.tolist()}")
        
        # Get user songs features and log details
        user_songs_df = df_scaled[df_scaled['song_id'].isin(user_songs)]
        logger.info("\nUser's playlist songs from dataset:")
        for _, song in user_songs_df.iterrows():
            logger.info(f"- Song ID: {song['song_id']}")
            logger.info(f"  Track: {song['track_name']}")
            logger.info(f"  Artist: {song['artist_name']}")
            logger.info(f"  Original ID in user_songs list: {[id for id in user_songs if id == song['song_id']]}")
        
        # Log any missing songs
        missing_songs = set(user_songs) - set(user_songs_df['song_id'].tolist())
        if missing_songs:
            logger.warning(f"\nSongs in user's playlist not found in dataset: {missing_songs}")
        
        if user_songs_df.empty:
            logger.warning("No user songs found in dataset, falling back to defaults")
            return get_default_recommendations(df_scaled)
            
        # Add verification step
        logger.info("\nVerifying song details between DB and dataset:")
        session = get_session()
        try:
            for song_id in user_songs:
                db_song = session.query(Song).filter_by(song_id=song_id).first()
                dataset_song = df_scaled[df_scaled['song_id'] == song_id]
                
                if db_song and not dataset_song.empty:
                    logger.info(f"\nSong ID: {song_id}")
                    logger.info(f"Database:")
                    logger.info(f"  Track: {db_song.track_name}")
                    logger.info(f"  Artist: {db_song.artists}")
                    logger.info(f"Dataset:")
                    logger.info(f"  Track: {dataset_song.iloc[0]['track_name']}")
                    logger.info(f"  Artist: {dataset_song.iloc[0]['artist_name']}")
        finally:
            session.close()

        # Calculate similarity
        similarity_matrix = cosine_similarity(
            df_scaled[features],
            user_songs_df[features]
        )
        
        # Get recommendations
        similarity_scores = np.mean(similarity_matrix, axis=1)
        df_scaled['similarity'] = similarity_scores
        
        new_recommendations = df_scaled[~df_scaled['song_id'].isin(user_songs)]\
            .nlargest(n_new_recommendations, 'similarity')
            
        logger.info("\nNew recommended songs:")
        for _, song in new_recommendations.iterrows():
            logger.info(f"- Song ID: {song['song_id']}")
            logger.info(f"  Track: {song['track_name']}")
            logger.info(f"  Artist: {song['artist_name']}")
            logger.info(f"  Similarity: {song['similarity']:.3f}")

        # Get familiar recommendations
        familiar_recommendations = user_songs_df.sample(
            n=min(n_familiar_recommendations, len(user_songs_df))
        ) if not user_songs_df.empty else df_scaled[df_scaled['song_id'].isin(fallback_user_songs)]\
            .sample(n=n_familiar_recommendations)
            
        logger.info("\nFamiliar recommended songs:")
        for _, song in familiar_recommendations.iterrows():
            logger.info(f"- Song ID: {song['song_id']}")
            logger.info(f"  Track: {song['track_name']}")
            logger.info(f"  Artist: {song['artist_name']}")

        recommendations = {
            'new_songs': new_recommendations.to_dict('records'),
            'user_songs': familiar_recommendations.to_dict('records')
        }
        
        return recommendations
        
    except Exception as e:
        logger.error(f"Error in similarity recommendations: {str(e)}", exc_info=True)
        return get_default_recommendations(df_scaled)

def get_dqn_based_recommendations(user_id, df_scaled, current_mood=None):
    """Get recommendations using the trained DQN model."""
    try:
        # Get all user playlist songs and similar songs
        user_songs = get_all_user_playlist_songs(user_id)
        similar_songs = get_similarity_based_recommendations(user_songs, df_scaled, n_recommendations=20)
        
        # Prepare candidate songs
        user_songs_df = df_scaled[df_scaled['song_id'].isin(user_songs)]
        new_songs_df = pd.DataFrame(similar_songs['new_songs'])
        
        # Use DQN to predict rewards for all songs
        model = DQN(state_size=9, action_size=len(df_scaled))
        model.load_state_dict(torch.load(f"models/dqn_{user_id}.pth"))
        model.eval()
        
        mood_state = convert_mood_to_state(current_mood)
        state_tensor = torch.tensor(get_initial_state(mood_state), dtype=torch.float32)
        
        with torch.no_grad():
            # Predict for new songs
            new_predictions = model(state_tensor)
            new_songs_df['predicted_reward'] = new_predictions[:len(new_songs_df)].numpy()
            
            # Predict for user songs
            user_predictions = model(state_tensor)
            user_songs_df = pd.DataFrame(similar_songs['user_songs'])
            user_songs_df['predicted_reward'] = user_predictions[:len(user_songs_df)].numpy()
        
        # Get top recommendations from both sets
        new_recommendations = new_songs_df.nlargest(n_new_recommendations, 'predicted_reward')
        familiar_recommendations = user_songs_df.nlargest(n_familiar_recommendations, 'predicted_reward')
        
        # Combine recommendations
        recommendations = pd.concat([new_recommendations, familiar_recommendations])
        
        # Structure the recommendations
        recommendations = {
            'new_songs': new_recommendations.to_dict('records'),
            'user_songs': familiar_recommendations.to_dict('records')
        }
        
        logger.info(f"Generated {len(new_recommendations)} new and {len(familiar_recommendations)} user playlist songs")
        return recommendations
        
    except Exception as e:
        logger.error(f"Error in DQN recommendations: {e}")
        return get_similarity_based_recommendations(user_songs, df_scaled)

def get_default_recommendations(df_scaled, n_total=15):
    """Get recommendations from default song list."""
    logger.info("Getting recommendations from default songs list")
    try:
        default_songs = df_scaled[df_scaled['song_id'].isin(fallback_user_songs)]
        
        # Structure as new songs and default songs
        return {
            'new_songs': default_songs.head(10).to_dict('records'),
            'user_songs': default_songs.tail(5).to_dict('records'),
            'source': 'default'
        }
        
    except Exception as e:
        logger.error(f"Error getting default recommendations: {e}")
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

def run_background_training(user_id, df_scaled=None, features=None, feature_weights=None):
    """
    Run background DQN training for a user based on their feedback history.
    """
    logger.info(f"=== Starting Background Training for User {user_id} ===")
    
    try:
        if df_scaled is None:
            logger.info("Loading dataset as it wasn't provided")
            df_scaled = get_dataset()
            
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

def get_weighted_recommendations(df, similarity_scores, popularity_weight=0.3, n_recommendations=10):
    """Get recommendations with weighted combination of similarity and popularity."""
    try:
        # Normalize similarity scores and popularity
        similarity_norm = (similarity_scores - similarity_scores.min()) / (similarity_scores.max() - similarity_scores.min())
        popularity_norm = (df['popularity'] - df['popularity'].min()) / (df['popularity'].max() - df['popularity'].min())
        
        # Combine scores with weighting
        df['combined_score'] = (1 - popularity_weight) * similarity_norm + popularity_weight * popularity_norm
        
        # Sort by combined score
        return df.nlargest(n_recommendations, 'combined_score')
    except Exception as e:
        print(f"Error in weighted recommendations: {str(e)}")
        return df.nlargest(n_recommendations, 'popularity')

def get_cluster_based_recommendations(user_songs, df_scaled, n_recommendations=10, features=None, exclude_songs=None):
    """Get recommendations using clustering and similarity with popularity weighting."""
    print("\n=== Generating Cluster-Based Recommendations ===")
    
    try:
        if features is None:
            features = ['energy', 'acousticness', 'valence', 'tempo', 'speechiness', 'instrumentalness']
        
        if exclude_songs is None:
            exclude_songs = []
            
        user_songs_df = df_scaled[df_scaled['song_id'].isin(user_songs)]
        print(f"User songs count: {len(user_songs_df)}")
        
        if len(user_songs_df) == 0:
            return get_default_recommendations(df_scaled)
            
        X = StandardScaler().fit_transform(user_songs_df[features])
        dbscan = DBSCAN(eps=0.5, min_samples=2)
        clusters = dbscan.fit_predict(X)
        
        cluster_counts = Counter(clusters)
        print(f"Found {len(cluster_counts)} clusters")
        
        all_recommendations = []
        
        for cluster_id in cluster_counts:
            cluster_mask = clusters == cluster_id
            cluster_songs_df = user_songs_df[cluster_mask]
            
            # Calculate similarity
            similarity_matrix = cosine_similarity(
                df_scaled[features],
                cluster_songs_df[features]
            )
            
            similarity_scores = np.mean(similarity_matrix, axis=1)
            
            cluster_df = df_scaled.copy()
            cluster_df['similarity'] = similarity_scores
            
            # Filter out excluded songs
            exclude_ids = user_songs + [s.get('song_id') for s in exclude_songs]
            filtered_df = cluster_df[~cluster_df['song_id'].isin(exclude_ids)]
            
            # Get weighted recommendations for this cluster
            cluster_recs = get_weighted_recommendations(
                df=filtered_df,
                similarity_scores=similarity_scores,
                popularity_weight=0.3,  # Adjust this weight to favor popularity more/less
                n_recommendations=max(2, int(n_recommendations * (len(cluster_songs_df) / len(user_songs_df))))
            )
            
            all_recommendations.append(cluster_recs)
            
        # Combine and get final recommendations
        final_recommendations = pd.concat(all_recommendations)
        final_recommendations = final_recommendations.nlargest(n_recommendations, 'combined_score')
        
        # Get popular songs from user's playlists
        familiar_recommendations = user_songs_df.nlargest(5, 'popularity')
        
        print(f"\nRecommendation counts:")
        print(f"New songs: {len(final_recommendations)}")
        print(f"Familiar songs: {len(familiar_recommendations)}")
        
        return {
            'new_songs': final_recommendations.to_dict('records'),
            'user_songs': familiar_recommendations.to_dict('records')
        }
        
    except Exception as e:
        print(f"Error in cluster-based recommendations: {str(e)}")
        return get_default_recommendations(df_scaled)

def get_user_history(user_id):
    """Get user's listening history from UserHistory table."""
    session = get_session()
    try:
        history = session.query(UserHistory).filter_by(user_id=user_id).all()
        return [h.song_id for h in history]
    finally:
        session.close()

def get_novel_recommendations(user_id, genres, mood, limit, exclude_songs):
    """Get novel/discovery recommendations."""
    try:
        df_scaled = get_dataset()
        
        # Get both user history and playlist songs
        user_history = get_user_history(user_id)
        playlist_songs = get_all_user_playlist_songs(user_id)
        
        # Combine all user's songs
        all_user_songs = list(set(user_history + playlist_songs))
        
        if not all_user_songs:
            print("No user history or playlist songs found")
            return []
            
        # Get recommendations using clustering
        recommendations = get_cluster_based_recommendations(
            user_songs=all_user_songs,
            df_scaled=df_scaled,
            n_recommendations=limit,
            features=['energy', 'acousticness', 'valence', 'tempo', 'speechiness', 'instrumentalness'],
            exclude_songs=exclude_songs
        )
        
        # Extract just the new songs from the recommendations
        return recommendations.get('new_songs', [])
        
    except Exception as e:
        print(f"Error in novel recommendations: {str(e)}")
        return []

def refresh_recommendations(user_id, previous_recs=None, refresh_type='smart', use_user_songs=True):
    """Smart recommendation refresh with multiple strategies."""
    try:
        print(f"\n=== Starting Recommendation Refresh ===")
        print(f"User ID: {user_id}")
        print(f"Refresh Type: {refresh_type}")
        print(f"Use User Songs: {use_user_songs}")
        
        df_scaled = get_dataset()
        n_recommendations = 15  # Increase total recommendations
        
        if use_user_songs:
            user_songs = get_all_user_playlist_songs(user_id)
            if not user_songs:
                return get_default_recommendations(df_scaled)
                
            user_mood = get_user_mood(user_id)
            user_genres = get_user_preferred_genres(user_id)
            
            if refresh_type == 'smart':
                # Adjust ratios to ensure more recommendations
                keep_ratio = 0.3  # 30% previous
                popular_ratio = 0.4  # 40% popular
                novel_ratio = 0.3  # 30% novel
                
                n_keep = int(n_recommendations * keep_ratio)
                n_popular = int(n_recommendations * popular_ratio)
                n_novel = n_recommendations - n_keep - n_popular
                
                print(f"\nRefresh Distribution:")
                print(f"Keep: {n_keep}")
                print(f"Popular: {n_popular}")
                print(f"Novel: {n_novel}")
                
                # Get recommendations
                kept_recs = filter_best_recommendations(previous_recs, n_keep) if previous_recs else []
                
                popular_recs = get_popular_recommendations(
                    user_id=user_id,
                    genres=user_genres,
                    mood=user_mood,
                    limit=n_popular,
                    exclude_songs=kept_recs
                )
                
                novel_recs = get_novel_recommendations(
                    user_id=user_id,
                    genres=user_genres,
                    mood=user_mood,
                    limit=n_novel,
                    exclude_songs=kept_recs + popular_recs
                )
                
                # Combine all recommendations
                final_recs = shuffle_recommendations(kept_recs + popular_recs + novel_recs)
                user_playlist_songs = get_user_playlist_songs(user_id, limit=5)
                
                print(f"\nFinal recommendation counts:")
                print(f"New songs: {len(final_recs)}")
                print(f"User songs: {len(user_playlist_songs)}")
                
                return {
                    'new_songs': final_recs,
                    'user_songs': user_playlist_songs,
                    'source': 'user_playlist'
                }
            else:
                return get_similarity_based_recommendations(
                    user_songs=user_songs,
                    df_scaled=df_scaled,
                    n_new_recommendations=15,  # Increase count
                    n_familiar_recommendations=5
                )
        else:
            return get_default_recommendations(df_scaled)
            
    except Exception as e:
        print(f"\nError in refresh recommendations: {str(e)}")
        return get_default_recommendations(df_scaled)

def get_user_playlist_songs(user_id, limit=5):
    """Get a sample of songs from user's playlists."""
    session = get_session()
    try:
        user_songs = get_all_user_playlist_songs(user_id)
        if not user_songs:
            return []
            
        songs = session.query(Song)\
            .filter(Song.song_id.in_(user_songs))\
            .limit(limit)\
            .all()
            
        return [song.serialize() for song in songs]
    finally:
        session.close()

def get_user_preferred_genres(user_id):
    """Get user's preferred genres based on playlist and history."""
    session = get_session()
    try:
        # Get genres from user's playlist songs
        playlist_songs = get_all_user_playlist_songs(user_id)
        if playlist_songs:
            genres = session.query(Song.track_genre)\
                .filter(Song.song_id.in_(playlist_songs))\
                .distinct()\
                .all()
            return [genre[0] for genre in genres]
        return []
    finally:
        session.close()

def filter_best_recommendations(recommendations, n_keep):
    """Filter the best recommendations to keep."""
    if not recommendations:
        return []
        
    # Sort by score/similarity if available, otherwise keep random
    sorted_recs = sorted(
        recommendations, 
        key=lambda x: x.get('score', x.get('similarity', 0)), 
        reverse=True
    )
    return sorted_recs[:n_keep]

def get_popular_recommendations(user_id, genres, mood, limit, exclude_songs):
    """Get popular songs matching user preferences with genre weighting."""
    session = get_session()
    try:
        # Get base query for popular songs in the given genres
        base_query = session.query(Song)\
            .filter(Song.track_genre.in_(genres))\
            .filter(Song.song_id.notin_([s['song_id'] for s in exclude_songs]))
        
        # Get top songs by popularity
        popular_songs = base_query.order_by(Song.popularity.desc()).limit(limit * 2).all()
        
        # Convert to DataFrame for weighted selection
        songs_df = pd.DataFrame([song.serialize() for song in popular_songs])
        if songs_df.empty:
            return []
            
        # Normalize popularity scores
        songs_df['popularity_norm'] = (songs_df['popularity'] - songs_df['popularity'].min()) / \
                                    (songs_df['popularity'].max() - songs_df['popularity'].min())
        
        # Add random variation to avoid always selecting the same songs
        songs_df['random_weight'] = np.random.uniform(0.8, 1.0, len(songs_df))
        songs_df['final_score'] = songs_df['popularity_norm'] * songs_df['random_weight']
        
        # Select final recommendations
        final_recommendations = songs_df.nlargest(limit, 'final_score')
        return final_recommendations.to_dict('records')
        
    finally:
        session.close()

def shuffle_recommendations(recommendations):
    """Shuffle recommendations while maintaining some structure."""
    # Shuffle within groups to maintain some ordering
    chunks = np.array_split(recommendations, 3)
    shuffled_chunks = [np.random.permutation(chunk).tolist() for chunk in chunks]
    return [rec for chunk in shuffled_chunks for rec in chunk]

def get_all_user_playlist_songs(user_id):
    """Get all unique songs from all playlists of a user."""
    print(f"\n=== Fetching User Playlist Songs ===")
    print(f"User ID: {user_id}")
    
    session = get_session()
    try:
        # Get all playlists for the user
        playlists = session.query(Playlist).filter_by(user_id=user_id).all()
        print(f"Found {len(playlists)} playlists")
        
        # Get all unique songs from all playlists
        all_songs = set()
        for playlist in playlists:
            songs = session.query(Song)\
                .join(PlaylistSong)\
                .filter(PlaylistSong.playlist_id == playlist.playlist_id)\
                .all()
            all_songs.update(song.song_id for song in songs)
        
        print(f"Total unique songs: {len(all_songs)}")
        return list(all_songs)
        
    except Exception as e:
        print(f"Error fetching playlist songs: {str(e)}")
        return []
    finally:
        session.close()
