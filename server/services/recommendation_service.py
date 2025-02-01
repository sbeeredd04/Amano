import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque, Counter
import numpy as np
from sqlalchemy import select
from models.song_model import Song, UserHistory, UserMood, Playlist, PlaylistSong
from utils.db import get_session, get_dataset
import os
from sklearn.cluster import DBSCAN
from routes.playlist import get_user_songs, get_all_user_playlist_songs
from datetime import datetime
from threading import Thread

# Define the global list of fallback user songs
fallback_user_songs = [67016, 91000, 81004, 17000, 20414, 81000, 81074, 81109, 20652, 
                        91016, 91017, 91018, 51150, 51503, 56064, 33012, 57162, 53050, 
                        67351, 51450, 94632, 51500, 53055]

# Global feature definitions
FEATURES = ['energy', 'acousticness', 'valence', 'tempo', 'speechiness', 'instrumentalness']
FEATURE_WEIGHTS = {
    'energy': 1.0,
    'acousticness': 5.0,
    'valence': 5.0,
    'tempo': 5.0,
    'instrumentalness': 5.0,
    'speechiness': 5.0
}

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
    """Preprocess the dataset for recommendations."""
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

# Add these constants at the top with other globals
MODEL_DIR = 'models/dqn'
os.makedirs(MODEL_DIR, exist_ok=True)

def get_model_path(user_id):
    """Get path to user's DQN model file."""
    return os.path.join(MODEL_DIR, f'dqn_user_{user_id}.pth')

def save_dqn_model(model, user_id):
    """Save DQN model for a user."""
    try:
        model_path = get_model_path(user_id)
        torch.save({
            'model_state_dict': model.state_dict(),
            'features': FEATURES,
            'timestamp': datetime.utcnow().isoformat()
        }, model_path)
        logger.info(f"Saved DQN model for user {user_id} at {model_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving DQN model: {str(e)}", exc_info=True)
        return False

def load_dqn_model(user_id, state_size, action_size):
    """Load DQN model for a user if it exists."""
    try:
        model_path = get_model_path(user_id)
        if not os.path.exists(model_path):
            logger.warning(f"No existing model found for user {user_id}")
            return None
            
        # Load the saved model
        checkpoint = torch.load(model_path)
        model = DQN(state_size, action_size)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        logger.info(f"Loaded DQN model for user {user_id}")
        logger.debug(f"Model features: {checkpoint['features']}")
        logger.debug(f"Model timestamp: {checkpoint['timestamp']}")
        
        return model
    except Exception as e:
        logger.error(f"Error loading DQN model: {str(e)}", exc_info=True)
        return None

#Background Training Function
def background_train_dqn(user_id, df_scaled, features, feature_weights):
    """Train DQN model using user feedback history and cluster-based candidates."""
    logger.info(f"=== Starting DQN Training for User {user_id} ===")
    
    try:
        # Get user's feedback history
        session = get_session()
        user_history = session.query(UserHistory).filter_by(user_id=user_id).all()
        user_songs = get_all_user_playlist_songs(user_id)
        
        logger.info(f"Retrieved {len(user_history)} feedback entries")
        logger.info(f"User has {len(user_songs)} songs in playlists")
        
        # Get candidate songs using clustering
        cluster_recommendations = get_cluster_weighted_recommendations(
                            user_songs=user_songs,
            df_scaled=df_scaled,
            n_recommendations=400  # Larger pool for training
        )
        
        if not cluster_recommendations:
            logger.error("No recommendations generated for training")
            return
            
        recommended_songs_df = pd.DataFrame(cluster_recommendations)
        logger.debug(f"Recommendation pool columns: {recommended_songs_df.columns}")
        
        # Verify required columns exist
        required_columns = ['song_id', 'popularity', 'weighted_score']
        missing_columns = [col for col in required_columns if col not in recommended_songs_df.columns]
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return
            
        action_size = len(recommended_songs_df)
        logger.info(f"Action space size: {action_size}")

        # Initialize or load existing model
        policy_net, target_net, optimizer, memory = init_dqn_model(9, action_size)
        
        # Load existing model if available
        existing_model = load_dqn_model(user_id, 9, action_size)
        if existing_model is not None:
            policy_net.load_state_dict(existing_model.state_dict())
            target_net.load_state_dict(existing_model.state_dict())
            logger.info("Loaded existing model for continued training")

        # Training parameters
        eps_start = 1.0
        eps_end = 0.1
        eps_decay = 0.995
        batch_size = 32
        gamma = 0.99
        target_update = 10
        eps_threshold = eps_start
        num_episodes = 20  # Increased episodes for better learning

        # Pre-fill replay memory with actual user feedback
        for feedback in user_history:
            state = convert_mood_to_state(feedback.mood)
            state_tensor = get_initial_state(state)
            
            # Find song index in recommended_songs_df
            song_idx = recommended_songs_df[
                recommended_songs_df['song_id'] == feedback.song_id
            ].index
            
            if len(song_idx) > 0:
                action = song_idx[0]
                reward = 1 if feedback.reward > 0 else -1
                next_state = state_tensor  # Simplified state transition
                
                memory.append((state_tensor, action, reward, next_state))
        
        logger.info(f"Pre-filled memory with {len(memory)} real feedback entries")

        # Training loop
        total_reward = 0
        for episode in range(num_episodes):
            episode_reward = 0
            
            # Get current user mood
            current_mood = get_user_mood(user_id)
            state = convert_mood_to_state(current_mood)
            state_tensor = get_initial_state(state)
            
            logger.info(f"\nEpisode {episode + 1}/{num_episodes}")
            logger.info(f"Current mood: {current_mood}")
            logger.info(f"Epsilon: {eps_threshold:.4f}")

            for t in range(20):  # More steps per episode
                # Select action using epsilon-greedy
                action = select_action(state_tensor, eps_threshold, action_size, policy_net)
                
                # Get song details
                song = recommended_songs_df.iloc[action]
                
                # Get reward from history if available, otherwise estimate
                historical_feedback = next(
                    (f for f in user_history if f.song_id == song['song_id']), 
                    None
                )
                
                if historical_feedback:
                    reward = 1 if historical_feedback.reward > 0 else -1
                    logger.debug(f"Using historical reward: {reward}")
                else:
                    # Use weighted_score instead of similarity
                    reward = 0.6 * song['weighted_score'] + 0.4 * song['popularity']
                    reward = (reward - 0.5) * 2  # Scale to [-1, 1]
                    logger.debug(f"Estimated reward: {reward}")
                
                # Get next state (simplified transition)
                next_state = state_tensor
                
                # Store transition
                memory.append((state_tensor, action, reward, next_state))
                
                # Train on batch
                if len(memory) >= batch_size:
                    optimize_model(policy_net, target_net, memory, optimizer, batch_size, gamma)

                episode_reward += reward
                state_tensor = next_state

            # Update metrics
            total_reward += episode_reward
            eps_threshold = max(eps_end, eps_threshold * eps_decay)
            
            # Update target network
            if episode % target_update == 0:
                target_net.load_state_dict(policy_net.state_dict())
                logger.info("Target network updated")

            logger.info(f"Episode {episode + 1} complete:")
            logger.info(f"Average Reward: {episode_reward/20:.4f}")

        # After training, save the model
        if save_dqn_model(policy_net, user_id):
            logger.info("Successfully saved trained model")
        else:
            logger.warning("Failed to save trained model")

    except Exception as e:
        logger.error(f"Error in DQN training: {str(e)}", exc_info=True)
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


def get_dqn_recommendations(user_id, df_scaled, current_mood, n_recommendations=200):
    """Get recommendations using trained DQN model."""
    logger.info("\n=== Generating DQN Recommendations ===")
    
    try:
        # Get user's playlist and liked songs
        session = get_session()
        user_songs = get_all_user_playlist_songs(user_id)
        
        # Get liked songs from history
        liked_songs = session.query(UserHistory)\
            .filter_by(user_id=user_id)\
            .filter(UserHistory.reward > 0)\
            .all()
        liked_song_ids = [h.song_id for h in liked_songs]
        
        logger.info(f"Found {len(user_songs)} playlist songs and {len(liked_song_ids)} liked songs")
        
        # Combine unique song IDs
        all_user_songs = list(set(user_songs + liked_song_ids))
        
        # Get candidate songs using clustering
        candidate_songs = get_cluster_weighted_recommendations(
            user_songs=all_user_songs,
            df_scaled=df_scaled,
            n_recommendations=400  # Get larger pool for DQN
        )
        
        if not candidate_songs:
            logger.error("No candidate songs generated")
            return []
            
        # Add user's playlist and liked songs to potential recommendations
        user_song_data = df_scaled[df_scaled['song_id'].isin(all_user_songs)].to_dict('records')
        all_candidates = candidate_songs + user_song_data
        
        logger.info(f"Total candidate pool size: {len(all_candidates)}")
        
        # Convert mood to state
        mood_state = get_mood_state(current_mood)
        
        # Load DQN model
        state_size = 9  # mood encoding size
        action_size = len(all_candidates)
        
        model = load_dqn_model(user_id, state_size, action_size)
        if model is None:
            logger.warning("No trained model found, falling back to clustering")
            return []
            
        # Get recommendations using loaded model
        model.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(mood_state).unsqueeze(0)
            q_values = model(state_tensor)
            
            # Get top recommendations
            _, top_actions = q_values.topk(n_recommendations)
            recommendations = [all_candidates[i] for i in top_actions[0].tolist()]
            
        logger.info(f"Generated {len(recommendations)} recommendations using DQN")
        return recommendations

    except Exception as e:
        logger.error(f"Error in DQN recommendations: {str(e)}", exc_info=True)
        return []

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
            features = FEATURES
            logger.info(f"Using default features: {features}")
            
        if feature_weights is None:
            feature_weights = FEATURE_WEIGHTS
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

def get_weighted_recommendations(df, similarity_scores, popularity_weight=0.4, n_recommendations=10):
    """Get recommendations weighted by similarity and popularity."""
    try:
        # Calculate weighted score
        df['combined_score'] = (
            (1 - popularity_weight) * similarity_scores + 
            popularity_weight * df['popularity']
        )
        
        # Get top recommendations
        recommendations = df.nlargest(n_recommendations, 'combined_score')
        return recommendations
        
    except Exception as e:
        logger.error(f"Error in weighted recommendations: {str(e)}")
        return pd.DataFrame()

def get_cluster_based_recommendations(user_songs, df_scaled, n_recommendations=10, features=None, exclude_songs=None):
    """Get recommendations using clustering and similarity with popularity weighting."""
    logger.info("\n=== Generating Cluster-Based Recommendations ===")
    
    try:
        if features is None:
            features = FEATURES
        
        if exclude_songs is None:
            exclude_songs = []
            
        user_songs_df = df_scaled[df_scaled['song_id'].isin(user_songs)]
        logger.info(f"User songs count: {len(user_songs_df)}")
        
        if len(user_songs_df) == 0:
            return get_default_recommendations(df_scaled)
            
        X = StandardScaler().fit_transform(user_songs_df[features])
        dbscan = DBSCAN(eps=0.5, min_samples=2)
        clusters = dbscan.fit_predict(X)
        
        cluster_counts = Counter(clusters)
        logger.info(f"Found {len(cluster_counts)} clusters")
        
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
                popularity_weight=0.4,  # Adjust this weight to favor popularity more/less
                n_recommendations=max(2, int(n_recommendations * (len(cluster_songs_df) / len(user_songs_df))))
            )
            all_recommendations.append(cluster_recs)
            
        # Combine and get final recommendations
        final_recommendations = pd.concat(all_recommendations)
        final_recommendations = final_recommendations.nlargest(n_recommendations, 'combined_score')
        
        # Get popular songs from user's playlists
        familiar_recommendations = user_songs_df.nlargest(5, 'popularity')
        
        logger.info(f"\nRecommendation counts:")
        logger.info(f"New songs: {len(final_recommendations)}")
        logger.info(f"Familiar songs: {len(familiar_recommendations)}")
        
        return {
            'new_songs': final_recommendations.to_dict('records'),
            'user_songs': familiar_recommendations.to_dict('records')
        }
        
    except Exception as e:
        logger.error(f"Error in cluster-based recommendations: {str(e)}")
        return get_default_recommendations(df_scaled)

def get_popular_recommendations(user_id, genres=None, mood=None, limit=10, exclude_songs=None):
    """Get popular song recommendations with optional genre and mood filtering."""
    try:
        logger.info(f"Getting popular recommendations:")
        logger.info(f"- User ID: {user_id}")
        logger.info(f"- Genres: {genres}")
        logger.info(f"- Mood: {mood}")
        logger.info(f"- Limit: {limit}")
        
        session = get_session()
        query = session.query(Song)
        
        # Filter by genres if provided
        if genres:
            query = query.filter(Song.track_genre.in_(genres))
            
        # Exclude songs if provided
        if exclude_songs:
            exclude_ids = [song.get('song_id') if isinstance(song, dict) else song 
                         for song in exclude_songs]
            query = query.filter(~Song.song_id.in_(exclude_ids))
            
        # Order by popularity and get recommendations
        recommendations = query.order_by(Song.popularity.desc())\
            .limit(limit)\
            .all()
            
        # Convert to dictionary format
        recommendations = [{
            'song_id': song.song_id,
            'track_name': song.track_name,
            'artist_name': song.artists,
            'track_genre': song.track_genre,
            'popularity': song.popularity,
            'similarity': 0.5  # Default similarity for popular recommendations
        } for song in recommendations]
        
        logger.info(f"Found {len(recommendations)} popular recommendations")
        return recommendations
        
    except Exception as e:
        logger.error(f"Error getting popular recommendations: {str(e)}", exc_info=True)
        return []
    finally:
        session.close()

def get_all_user_playlist_songs(user_id):
    """Get all unique songs from all playlists of a user."""
    logger.info(f"\n=== Fetching User Playlist Songs ===")
    logger.info(f"User ID: {user_id}")
    
    session = get_session()
    try:
        # Get all playlists for the user
        playlists = session.query(Playlist).filter_by(user_id=user_id).all()
        logger.info(f"Found {len(playlists)} playlists")
        
        # Get all unique songs from all playlists
        all_songs = set()
        for playlist in playlists:
            songs = session.query(Song)\
                .join(PlaylistSong)\
                .filter(PlaylistSong.playlist_id == playlist.playlist_id)\
                .all()
            all_songs.update(song.song_id for song in songs)
        
        logger.info(f"Total unique songs: {len(all_songs)}")
        return list(all_songs)
        
    except Exception as e:
        logger.error(f"Error fetching playlist songs: {str(e)}")
        return []
    finally:
        session.close()

def generate_recommendation_pool(user_id, current_mood, df_scaled):
    """Generate comprehensive recommendation pool using available methods."""
    logger.info("\n=== Generating Recommendation Pool ===")
    logger.info(f"User ID: {user_id}")
    logger.info(f"Current Mood: {current_mood}")
    
    try:
        # Get user's songs
        user_songs = get_all_user_playlist_songs(user_id)
        logger.info(f"Found {len(user_songs)} user songs")
        
        if not user_songs:
            logger.warning("No user songs found, using fallback songs")
            user_songs = fallback_user_songs
        
        # Generate initial recommendations using clustering
        logger.info("Generating initial cluster-based recommendations")
        cluster_recommendations = get_cluster_weighted_recommendations(
            user_songs=user_songs,
            df_scaled=df_scaled,
            n_recommendations=200  # Get a larger initial pool
        )
        
        if not cluster_recommendations:
            logger.error("Failed to generate cluster recommendations")
            # Fallback to simple popularity-based recommendations
            logger.info("Falling back to popularity-based recommendations")
            popular_recs = get_popular_recommendations(
                user_id=user_id,
                limit=200,  # Get more recommendations for the pool
                exclude_songs=user_songs  # Exclude user's songs
            )
            if not popular_recs:
                raise Exception("Failed to generate any recommendations")
            cluster_recommendations = popular_recs
            
        logger.info(f"Generated {len(cluster_recommendations)} initial recommendations")
        
        # Check if DQN model exists and has enough feedback
        session = get_session()
        try:
            history_count = session.query(UserHistory)\
                .filter_by(user_id=user_id)\
                .count()
            model_exists = os.path.exists(f'models/dqn/dqn_user_{user_id}.pth')
            
            logger.info(f"User has {history_count} feedback entries")
            logger.info(f"DQN model exists: {model_exists}")
            
            final_recommendations = cluster_recommendations
            source = 'clustering'
            
            if model_exists:
                # Try DQN recommendations
                try:
                    dqn_recommendations = get_dqn_recommendations(
                        user_id=user_id,
                        df_scaled=df_scaled,
                        current_mood=current_mood
                    )
                    if dqn_recommendations:
                        final_recommendations = dqn_recommendations
                        source = 'dqn'
                        logger.info("Successfully generated DQN recommendations")
                except Exception as e:
                    logger.error(f"Error getting DQN recommendations: {e}")
                    # Continue with cluster recommendations
            
            # Start background training if enough feedback
            if history_count >= 5:
                logger.info("Feedback threshold reached - initiating background DQN training")
                Thread(target=run_background_training, 
                       args=(user_id, df_scaled, FEATURES, FEATURE_WEIGHTS)).start()
            
            logger.info(f"Final recommendation pool size: {len(final_recommendations)}")
            
            return {
                'recommendation_pool': final_recommendations,
                'user_songs': user_songs,
                'has_dqn_model': model_exists,
                'source': source
            }
            
        finally:
            session.close()
            
    except Exception as e:
        logger.error(f"Error generating recommendation pool: {str(e)}", exc_info=True)
        raise

def get_mood_state(mood):
    """
    Convert mood string to state vector.
    Returns a one-hot encoded vector for the mood.
    """
    logger.debug(f"Converting mood '{mood}' to state vector")
    
    # Define mood mapping (same as convert_mood_to_state but returns vector)
    mood_mapping = {
        'Angry': 0,
        'Content': 1,
        'Happy': 2,
        'Delighted': 3,
        'Calm': 4,
        'Sleepy': 5,
        'Sad': 6,
        'Depressed': 7,
        'Excited': 8
    }
    
    # Create one-hot encoded vector
    state = np.zeros(9)  # 9 possible moods
    mood_idx = mood_mapping.get(mood, 4)  # Default to Calm (idx 4) if mood not found
    state[mood_idx] = 1
    
    logger.debug(f"Mood state vector: {state}")
    return state

def refresh_from_pool(pool, previous_recs, refresh_type='smart'):
    """
    Get refreshed recommendations from the stored pool with smart weighting.
    Includes popular recommendations and user playlist songs.
    """
    try:
        logger.info("\n=== Refreshing from Pool ===")
        
        # Get the pools
        recommendation_pool = pool['recommendation_pool']
        user_songs_pool = pool['user_songs_pool']
        
        if refresh_type == 'smart':
            # Define ratios for different types of recommendations
            keep_ratio = 0.3  # Keep 30% of previous recommendations
            popular_ratio = 0.4  # 40% popular songs from pool
            novel_ratio = 0.3  # 30% novel/discovery songs from pool
            
            total_new_recs = 20  # Base number of recommendations from pool
            
            # Calculate counts for each category
            n_keep = int(total_new_recs * keep_ratio)
            n_popular = int(total_new_recs * popular_ratio)
            n_novel = total_new_recs - n_keep - n_popular
            
            logger.info(f"Distribution - Keep: {n_keep}, Popular: {n_popular}, Novel: {n_novel}")
            
            # Keep best previous recommendations
            kept_recs = []
            if previous_recs:
                # Sort by similarity and popularity
                sorted_prev = sorted(
                    previous_recs,
                    key=lambda x: (x.get('similarity', 0) * 0.7 + x.get('popularity', 0) * 0.3),
                    reverse=True
                )
                kept_recs = sorted_prev[:n_keep]
            
            # Get popular recommendations from pool
            kept_ids = [rec['song_id'] for rec in kept_recs]
            available_pool = [rec for rec in recommendation_pool if rec['song_id'] not in kept_ids]
            
            # Sort pool by popularity and similarity
            popular_pool = sorted(
                available_pool,
                key=lambda x: (x.get('popularity', 0) * 0.8 + x.get('similarity', 0) * 0.2),
                reverse=True
            )
            popular_recs = popular_pool[:n_popular]
            
            # Get novel recommendations (lower similarity/popularity for discovery)
            used_ids = kept_ids + [rec['song_id'] for rec in popular_recs]
            novel_pool = [rec for rec in recommendation_pool if rec['song_id'] not in used_ids]
            novel_recs = random.sample(novel_pool, min(n_novel, len(novel_pool)))
            
            # Get extra popular recommendations from the database
            try:
                user_id = pool.get('user_id')
                if user_id:
                    extra_popular = get_popular_recommendations(
                        user_id=user_id,
                        genres=None,  # Get from all genres
                        mood=None,    # No mood filtering
                        limit=10,     # Get 10 extra recommendations
                        exclude_songs=kept_recs + popular_recs + novel_recs
                    )
                else:
                    extra_popular = []
                    logger.warning("No user_id in pool, skipping extra popular recommendations")
            except Exception as e:
                logger.error(f"Error getting extra popular recommendations: {e}")
                extra_popular = []
            
            # Combine all recommendations
            new_recommendations = kept_recs + popular_recs + novel_recs + extra_popular
            random.shuffle(new_recommendations)  # Shuffle to mix different types
            
            # Always include 5 random user playlist songs
            if user_songs_pool:
                user_recs = random.sample(user_songs_pool, min(5, len(user_songs_pool)))
            else:
                logger.warning("No user songs in pool")
                user_recs = []
            
            logger.info(f"Generated recommendations:")
            logger.info(f"- Kept: {len(kept_recs)}")
            logger.info(f"- Popular from pool: {len(popular_recs)}")
            logger.info(f"- Novel: {len(novel_recs)}")
            logger.info(f"- Extra popular: {len(extra_popular)}")
            logger.info(f"- User playlist songs: {len(user_recs)}")
            
            return {
                'new_songs': new_recommendations,
                'user_songs': user_recs,
                'source': 'smart_pool_refresh'
            }
            
        else:
            # Simple refresh - get random selections plus extra popular
            new_recs = random.sample(recommendation_pool, min(20, len(recommendation_pool)))
            
            # Get extra popular recommendations
            try:
                user_id = pool.get('user_id')
                if user_id:
                    extra_popular = get_popular_recommendations(
                        user_id=user_id,
                        genres=None,
                        mood=None,
                        limit=10,
                        exclude_songs=new_recs
                    )
                    new_recs.extend(extra_popular)
                    random.shuffle(new_recs)
            except Exception as e:
                logger.error(f"Error getting extra popular recommendations: {e}")
            
            # Always include 5 random user playlist songs
            if user_songs_pool:
                user_recs = random.sample(user_songs_pool, min(5, len(user_songs_pool)))
            else:
                logger.warning("No user songs in pool")
                user_recs = []
            
            return {
                'new_songs': new_recs,
                'user_songs': user_recs,
                'source': 'simple_pool_refresh'
            }
            
    except Exception as e:
        logger.error(f"Error refreshing from pool: {str(e)}", exc_info=True)
        raise

def get_cluster_weighted_recommendations(user_songs, df_scaled, n_recommendations=200):
    """
    Get recommendations using clustering and weighted similarity scores.
    """
    logger.info("\n=== Generating Cluster-Weighted Recommendations ===")
    logger.info(f"User songs count: {len(user_songs)}")
    
    try:
        features = FEATURES
        
        # Verify all features exist in df_scaled
        missing_features = [f for f in features if f not in df_scaled.columns]
        if missing_features:
            logger.error(f"Missing features in dataset: {missing_features}")
            return []
            
        # Get user songs dataframe
        user_songs_df = df_scaled[df_scaled['song_id'].isin(user_songs)]
        
        if user_songs_df.empty:
            logger.warning("No user songs found in dataset")
        return []
            
        # Log feature statistics
        logger.debug("Feature statistics for user songs:")
        for feature in features:
            stats = user_songs_df[feature].describe()
            logger.debug(f"{feature}: mean={stats['mean']:.2f}, std={stats['std']:.2f}")
            
        # Try clustering with increasingly relaxed parameters
        eps_values = [2, 4, 6, 8]
        min_samples_values = [2, 1]
        
        clusters = None
        final_eps = None
        final_min_samples = None
        
        for eps in eps_values:
            for min_samples in min_samples_values:
                logger.info(f"Trying clustering with eps={eps}, min_samples={min_samples}")
                
                X = StandardScaler().fit_transform(user_songs_df[features])
                dbscan = DBSCAN(
                    eps=eps,
                    min_samples=min_samples,
                    metric='euclidean'
                )
                current_clusters = dbscan.fit_predict(X)
                cluster_counts = Counter(current_clusters)
                
                # Count non-noise clusters
                non_noise_clusters = sum(1 for c in cluster_counts if c != -1)
                logger.info(f"Found {non_noise_clusters} non-noise clusters")
                
                if non_noise_clusters > 0:
                    clusters = current_clusters
                    final_eps = eps
                    final_min_samples = min_samples
                    break
            
            if clusters is not None:
                break
        
        # If still no clusters found, treat each noise point as its own cluster
        if clusters is None or len(set(clusters)) == 1:
            logger.info("Creating individual clusters for each song")
            clusters = np.arange(len(user_songs_df))
            final_eps = eps_values[-1]
            final_min_samples = 1
        
        cluster_counts = Counter(clusters)
        logger.info(f"Final clustering parameters: eps={final_eps}, min_samples={final_min_samples}")
        logger.info(f"Total clusters: {len(cluster_counts)}")
        logger.debug(f"Cluster sizes: {dict(cluster_counts)}")
        
        recommendation_pool = []
        
        # Process each cluster (including noise points as individual clusters)
        for cluster_id in cluster_counts:
            cluster_size = cluster_counts[cluster_id]
            logger.debug(f"Processing cluster {cluster_id} with {cluster_size} songs")
            
            # Get songs for this cluster
            if cluster_id == -1:
                # Process each noise point as individual cluster
                noise_mask = clusters == -1
                noise_songs = user_songs_df[noise_mask]
                for _, noise_song in noise_songs.iterrows():
                    cluster_songs_df = pd.DataFrame([noise_song])
                    cluster_weight = 1.0 / len(user_songs_df)
                    cluster_recs = get_recommendations_for_cluster(
                        cluster_songs_df,
                        df_scaled,
                        features,
                        cluster_weight,
                        n_recommendations,
                        user_songs,
                        recommendation_pool
                    )
                    recommendation_pool.extend(cluster_recs)
            else:
                # Process regular cluster
                cluster_mask = clusters == cluster_id
                cluster_songs_df = user_songs_df[cluster_mask]
                cluster_weight = len(cluster_songs_df) / len(user_songs_df)
                cluster_recs = get_recommendations_for_cluster(
                    cluster_songs_df,
                    df_scaled,
                    features,
                    cluster_weight,
                    n_recommendations,
                    user_songs,
                    recommendation_pool
                )
                recommendation_pool.extend(cluster_recs)
        
        logger.info(f"Generated {len(recommendation_pool)} total recommendations")
        
        # Verify recommendation format
        if recommendation_pool:
            sample_rec = recommendation_pool[0]
            logger.debug(f"Sample recommendation keys: {sample_rec.keys()}")
            logger.debug(f"Sample recommendation: {sample_rec}")
        
        return recommendation_pool
        
    except Exception as e:
        logger.error(f"Error in cluster recommendations: {str(e)}", exc_info=True)
        raise

def get_recommendations_for_cluster(cluster_songs_df, df_scaled, features, cluster_weight, n_recommendations, user_songs, existing_recommendations):
    """Helper function to get recommendations for a cluster or individual song."""
    
    # Calculate similarity
    similarity_matrix = cosine_similarity(
        df_scaled[features],
        cluster_songs_df[features]
    )
    
    similarity_scores = np.mean(similarity_matrix, axis=1)
    
    # Get recommendations for this cluster
    cluster_df = df_scaled.copy()
    cluster_df['similarity'] = similarity_scores
    cluster_df['weighted_score'] = (
        cluster_df['similarity'] * 0.2 +
        cluster_df['popularity'] * 0.8
    )
    
    # Filter out user songs and existing recommendations
    exclude_ids = user_songs + [rec['song_id'] for rec in existing_recommendations]
    filtered_df = cluster_df[~cluster_df['song_id'].isin(exclude_ids)]
    
    # Calculate number of recommendations for this cluster
    cluster_recs_count = max(5, int(n_recommendations * cluster_weight))
    
    # Get top recommendations for this cluster
    cluster_recs = filtered_df.nlargest(
        cluster_recs_count, 
        'weighted_score'
    ).to_dict('records')
    
    return cluster_recs

def init_dqn_model(state_size, action_size):
    """
    Initialize DQN model components.
    
    Args:
        state_size (int): Size of the state space (mood encoding size)
        action_size (int): Size of the action space (number of possible songs)
    
    Returns:
        tuple: (policy_net, target_net, optimizer, memory)
    """
    logger.info("Initializing DQN model components...")
    
    try:
        # Initialize networks
        policy_net = DQN(state_size, action_size)
        target_net = DQN(state_size, action_size)
        target_net.load_state_dict(policy_net.state_dict())
        target_net.eval()  # Target network is only used for evaluation
        
        # Initialize optimizer
        optimizer = optim.Adam(policy_net.parameters(), lr=0.001)
        
        # Initialize replay memory
        memory = deque(maxlen=10000)  # Store up to 10000 transitions
        
        logger.info(f"DQN initialized with state_size={state_size}, action_size={action_size}")
        return policy_net, target_net, optimizer, memory
        
    except Exception as e:
        logger.error(f"Error initializing DQN model: {str(e)}", exc_info=True)
        raise