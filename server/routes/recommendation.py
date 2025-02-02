from flask import Blueprint, request, jsonify
from services.recommendation_service import (
    generate_recommendation_pool,
    refresh_from_pool,
    update_user_feedback, 
    run_background_training, 
    update_user_mood,
    get_user_mood,
    get_all_user_playlist_songs
)
from utils.db import get_dataset
import pandas as pd
import os
from utils.db import get_session
from models.song_model import UserMood, UserHistory, Base, Playlist, PlaylistSong
import logging
from flask_cors import CORS
from datetime import datetime
from models.recommendation_model import RecommendationPool
from sqlalchemy import inspect
from threading import Thread
from matplotlib import pyplot as plt
import io
import base64
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from collections import Counter
import matplotlib
matplotlib.use('Agg')  # Add this at the top after imports
import numpy as np

logger = logging.getLogger(__name__)

recommendation_bp = Blueprint('recommendation', __name__)

# Configure CORS for the blueprint
CORS(recommendation_bp, 
     origins=["http://localhost:3000"],
     methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
     allow_headers=["Content-Type", "Authorization"],
     supports_credentials=True)

# Load the preprocessed dataset
try:
    logger.info("Loading dataset...")
    df_scaled = get_dataset()
    logger.info(f"Dataset loaded successfully with {len(df_scaled)} records")
except Exception as e:
    logger.error(f"Error loading dataset: {e}")
    raise

features = ['energy', 'acousticness', 'valence', 'tempo', 'speechiness', 'instrumentalness']
feature_weights = {
    'energy': 1.0, 
    'acousticness': 5.0, 
    'valence': 5.0, 
    'tempo': 5.0, 
    'instrumentalness': 5.0, 
    'speechiness': 5.0
}

def ensure_tables_exist(engine):
    """Ensure all required tables exist in the database."""
    inspector = inspect(engine)
    existing_tables = inspector.get_table_names()
    
    logger.info("Checking database tables...")
    logger.debug(f"Existing tables: {existing_tables}")
    
    if 'recommendation_pools' not in existing_tables:
        logger.warning("recommendation_pools table not found - creating it")
        Base.metadata.create_all(engine)
        logger.info("Created missing tables")

# Add this after creating the blueprint
engine = get_session().get_bind()
ensure_tables_exist(engine)

@recommendation_bp.route('/refresh', methods=['POST'])
def refresh_recommendations():
    """
    Endpoint to refresh recommendations from the stored pool with smart weighting.
    """
    try:
        data = request.json
        user_id = data.get('user_id')
        previous_recs = data.get('previous_recommendations', [])
        refresh_type = data.get('refresh_type', 'smart')
        
        logger.info(f"\n=== Refreshing Recommendations from Pool ===")
        logger.info(f"User ID: {user_id}")
        logger.info(f"Previous Recs Count: {len(previous_recs)}")
        logger.info(f"Refresh Type: {refresh_type}")
        
        if not user_id:
            return jsonify({"error": "User ID is required"}), 400
            
        session = get_session()
        try:
            pool = session.query(RecommendationPool)\
                .filter_by(user_id=user_id)\
                .first()
                
            if not pool:
                logger.error("No recommendation pool found for refresh")
                return jsonify({"error": "No recommendation pool found"}), 404
            
            logger.debug(f"Found pool with {len(pool.recommendation_pool)} recommendations")
            logger.debug(f"User songs in pool: {len(pool.user_songs_pool)}")
            
            refreshed_data = refresh_from_pool(
                pool={
                    'recommendation_pool': pool.recommendation_pool,
                    'user_songs_pool': pool.user_songs_pool,
                    'popular_songs_pool': pool.popular_songs_pool,
                    'user_id': user_id
                },
                previous_recs=previous_recs,
                refresh_type=refresh_type
            )
            
            logger.debug("Refreshed data received:")
            logger.debug(f"- New songs: {len(refreshed_data['new_songs'])}")
            logger.debug(f"- User songs: {len(refreshed_data['user_songs'])}")
            logger.debug(f"- Source: {refreshed_data['source']}")
            
            return jsonify({
                "recommendations": {
                    'recommendation_pool': refreshed_data['new_songs'],
                    'user_songs': refreshed_data['user_songs'],
                    'popular_songs': refreshed_data.get('popular_songs', []),
                    'has_dqn_model': os.path.exists(f'models/dqn/dqn_user_{user_id}.pth'),
                    'source': refreshed_data['source']
                },
                "source": refreshed_data['source']
            }), 200
            
        except Exception as e:
            logger.error(f"Error in refresh: {str(e)}", exc_info=True)
            session.rollback()
            return jsonify({"error": str(e)}), 500
        finally:
            session.close()
            
    except Exception as e:
        logger.error(f"Error in refresh endpoint: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@recommendation_bp.route('/feedback', methods=['POST'])
def handle_feedback():
    """
    Endpoint to handle user feedback (likes/dislikes) for songs.
    """
    logger.info("=== Processing User Feedback ===")
    data = request.json
    user_id = data.get('user_id')
    song_id = data.get('song_id')
    is_liked = data.get('is_liked')
    current_mood = data.get('mood')

    logger.info(f"Feedback Details:")
    logger.info(f"- User ID: {user_id}")
    logger.info(f"- Song ID: {song_id}")
    logger.info(f"- Liked: {is_liked}")
    logger.info(f"- Current Mood: {current_mood}")

    if not all([user_id, song_id, is_liked is not None]):
        logger.error("Missing required fields in feedback request")
        return jsonify({"error": "Missing required fields"}), 400

    try:
        session = get_session()
        reward = 1 if is_liked else -1
        
        feedback = [{
            'song_id': song_id,
            'reward': reward,
            'mood': current_mood
        }]
        update_user_feedback(user_id, feedback)

        history_count = session.query(UserHistory).filter_by(user_id=user_id).count()
        logger.info(f"User now has {history_count} feedback entries")

        # Increased threshold to 100
        if history_count >= 100:  # Changed from 5 to 100
            logger.info("Feedback threshold reached - initiating DQN training")
            run_background_training(
                user_id=user_id,
                df_scaled=df_scaled,
                features=features,
                feature_weights=feature_weights
            )
            return jsonify({
                "message": "Feedback recorded successfully and model trained",
                "training_triggered": True,
                "feedback_count": history_count
            }), 200
        else:
            logger.info(f"Need {100 - history_count} more feedback entries before training")
            return jsonify({
                "message": "Feedback recorded successfully",
                "training_triggered": False,
                "feedback_count": history_count
            }), 200

    except Exception as e:
        logger.error(f"Error recording feedback: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500
    finally:
        session.close()

@recommendation_bp.route('/mood', methods=['GET', 'POST'])
def manage_mood():
    """Endpoint to get or update user's current mood."""
    logger.info(f"=== Mood Management Endpoint ===")
    logger.info(f"Method: {request.method}")
    
    try:
        user_id = request.args.get('user_id') if request.method == 'GET' else request.json.get('user_id')
        
        if not user_id:
            logger.error("No user_id provided")
            return jsonify({"error": "user_id is required"}), 400
            
        if request.method == 'GET':
            logger.info(f"Getting mood for user {user_id}")
            mood = get_user_mood(user_id)
            return jsonify({"mood": mood})
        else:
            mood = request.json.get('mood')
            if not mood:
                logger.error("No mood provided in POST request")
                return jsonify({"error": "mood is required"}), 400
                
            logger.info(f"Updating mood for user {user_id} to {mood}")
            try:
                update_user_mood(user_id, mood)
                return jsonify({"message": "Mood updated successfully", "mood": mood})
            except Exception as e:
                logger.error(f"Failed to update mood: {str(e)}")
                return jsonify({"error": str(e)}), 500
            
    except Exception as e:
        logger.error(f"Error managing mood: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@recommendation_bp.route('/recs', methods=['POST'])
def generate_recommendations():
    """
    Comprehensive recommendation pipeline endpoint.
    Always generates fresh recommendations after clearing existing ones.
    """
    try:
        data = request.json
        logger.info("\n=== Recommendation Request ===")
        logger.info(f"Request data: {data}")
        
        user_id = data.get('user_id')
        current_mood = data.get('mood', 'Calm')
        
        if not user_id:
            logger.error("Missing user_id in request")
            return jsonify({"error": "User ID is required"}), 400
            
        session = get_session()
        try:
            # Clear existing recommendation pools for this user
            logger.info(f"Clearing existing recommendation pools for user {user_id}")
            existing_pools = session.query(RecommendationPool)\
                .filter_by(user_id=user_id)\
                .all()
            
            if existing_pools:
                for pool in existing_pools:
                    session.delete(pool)
                session.commit()
                logger.info(f"Cleared {len(existing_pools)} existing recommendation pools")
            
            # Debug user's playlists
            playlists = session.query(Playlist).filter_by(user_id=user_id).all()
            logger.debug(f"Found {len(playlists)} playlists for user {user_id}")
            for playlist in playlists:
                song_count = session.query(PlaylistSong).filter_by(playlist_id=playlist.playlist_id).count()
                logger.debug(f"Playlist {playlist.name}: {song_count} songs")
            
            # Generate new recommendations
            logger.info("Generating fresh recommendations")
            new_recs = generate_recommendation_pool(
                user_id=user_id,
                current_mood=current_mood,
                df_scaled=df_scaled
            )
            
            if not new_recs:
                logger.error("Failed to generate recommendations")
                return jsonify({"error": "Failed to generate recommendations"}), 500
            
            source = new_recs.get('source', 'new')
            
            # Store new pool in database
            new_pool = RecommendationPool(
                user_id=user_id,
                recommendation_pool=new_recs['recommendation_pool'],
                user_songs_pool=new_recs['user_songs'],
                popular_songs_pool=new_recs.get('popular_songs', []),
                created_at=datetime.utcnow()
            )
            session.add(new_pool)
            session.commit()
            logger.info("Stored new recommendation pool in database")
            
            # Send limited set to frontend
            recommendations = {
                'recommendation_pool': new_recs['recommendation_pool'][:30],  # Limit to 30
                'user_songs': new_recs['user_songs'][:30],  # Limit to 30
                'popular_songs': new_recs.get('popular_songs', [])[:20],  # Limit to 20
                'has_dqn_model': os.path.exists(f'models/dqn/dqn_user_{user_id}.pth'),
                'source': source
            }
            
            # Debug final response
            logger.info("\n=== Final Response Content ===")
            logger.info(f"Recommendation pool size (limited): {len(recommendations['recommendation_pool'])}")
            logger.info(f"User songs size (limited): {len(recommendations['user_songs'])}")
            logger.info(f"Popular songs size (limited): {len(recommendations.get('popular_songs', []))}")
            logger.info(f"Source: {source}")
            
            return jsonify({
                "recommendations": recommendations,
                "source": source
            }), 200
            
        except Exception as e:
            logger.error(f"Error in recommendation generation: {str(e)}", exc_info=True)
            session.rollback()
            return jsonify({"error": str(e)}), 500
        finally:
            session.close()
            
    except Exception as e:
        logger.error(f"Error in recommendation endpoint: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

def background_pool_refresh(user_id, current_mood, existing_pool):
    """Background task to refresh recommendation pool."""
    try:
        logger.info(f"Starting background pool refresh for user {user_id}")
        refreshed_pool = refresh_from_pool(
            pool=existing_pool,
            previous_recs=existing_pool,
            refresh_type='smart'
        )
        
        session = get_session()
        try:
            pool_record = session.query(RecommendationPool)\
                .filter_by(user_id=user_id)\
                .order_by(RecommendationPool.created_at.desc())\
                .first()
                
            if pool_record:
                pool_record.recommendation_pool = refreshed_pool
                pool_record.updated_at = datetime.utcnow()
                session.commit()
                logger.info("Background pool refresh completed")
        finally:
            session.close()
            
    except Exception as e:
        logger.error(f"Error in background pool refresh: {str(e)}", exc_info=True)

def background_pool_generation(user_id, current_mood):
    """Background task to generate new recommendation pool."""
    try:
        logger.info(f"Starting background pool generation for user {user_id}")
        new_recommendations = generate_recommendation_pool(
            user_id=user_id,
            current_mood=current_mood,
            df_scaled=df_scaled
        )
        
        session = get_session()
        try:
            new_pool = RecommendationPool(
                user_id=user_id,
                user_songs_pool=new_recommendations['user_songs'],
                recommendation_pool=new_recommendations['recommendation_pool'],
                created_at=datetime.utcnow()
            )
            session.add(new_pool)
            session.commit()
            logger.info("Background pool generation completed")
        finally:
            session.close()
            
    except Exception as e:
        logger.error(f"Error in background pool generation: {str(e)}", exc_info=True)

@recommendation_bp.route('/visualize', methods=['POST'])
def visualize_clusters():
    """Generate cluster visualization for user songs."""
    try:
        data = request.json
        user_id = data.get('user_id')
        
        if not user_id:
            return jsonify({"error": "User ID is required"}), 400
            
        # Get user songs
        user_songs = get_all_user_playlist_songs(user_id)
        if not user_songs:
            return jsonify({"error": "No user songs found"}), 404
            
        # Get visualization data
        user_songs_df = df_scaled[df_scaled['song_id'].isin(user_songs)].copy()
        
        # Generate clusters and visualization
        X = StandardScaler().fit_transform(user_songs_df[features])
        eps = 2 # DBSCAN eps parameter
        dbscan = DBSCAN(eps=eps, min_samples=2)
        clusters = dbscan.fit_predict(X)
        
        # Reduce dimensions for visualization
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        
        # Create plot with dark theme
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(10, 8))
        fig.patch.set_facecolor('#00000000')  # Transparent background
        ax.set_facecolor('#00000000')  # Transparent plot background
        
        # Plot each cluster with different color
        unique_clusters = set(clusters)
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_clusters)))
        
        for cluster, color in zip(unique_clusters, colors):
            mask = clusters == cluster
            ax.scatter(
                X_pca[mask, 0],
                X_pca[mask, 1],
                c=[color],
                label=f'Cluster {cluster}',
                alpha=0.6,
                s=100  # Increase point size
            )
            
            # Draw eps radius circles around points
            for x, y in zip(X_pca[mask, 0], X_pca[mask, 1]):
                circle = plt.Circle(
                    (x, y), 
                    eps/4,  # Scale down eps for better visualization
                    color=color,
                    fill=False,
                    alpha=0.2,
                    linestyle='--'
                )
                ax.add_artist(circle)
        
        # Customize plot
        ax.set_title('Your Music Clusters', color='white', pad=20)
        ax.set_xlabel('Principal Component 1', color='white')
        ax.set_ylabel('Principal Component 2', color='white')
        ax.tick_params(colors='white')
        ax.spines['bottom'].set_color('white')
        ax.spines['top'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['right'].set_color('white')
        
        # Adjust legend
        legend = ax.legend(
            bbox_to_anchor=(1.05, 1),
            loc='upper left',
            frameon=False
        )
        for text in legend.get_texts():
            text.set_color('white')
        
        # Save plot to bytes
        buf = io.BytesIO()
        plt.savefig(buf, format='png', transparent=True, bbox_inches='tight', dpi=300)
        buf.seek(0)
        plt.close()
        
        # Convert to base64
        plot_data = base64.b64encode(buf.getvalue()).decode('utf-8')
        
        # Convert numpy types to Python native types for JSON serialization
        cluster_counts = Counter(clusters)
        cluster_counts_dict = {int(k): int(v) for k, v in cluster_counts.items()}
        
        return jsonify({
            "plot": plot_data,
            "cluster_count": len(unique_clusters),
            "songs_per_cluster": cluster_counts_dict,
            "eps_radius": float(eps)
        }), 200
        
    except Exception as e:
        logger.error(f"Error generating visualization: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

