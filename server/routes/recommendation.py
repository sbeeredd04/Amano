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
from models.song_model import UserMood, UserHistory, Base
import logging
from flask_cors import CORS
from datetime import datetime
from models.recommendation_model import RecommendationPool
from sqlalchemy import inspect
from threading import Thread

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
            
        # Get stored recommendation pool
        session = get_session()
        try:
            pool = session.query(RecommendationPool)\
                .filter_by(user_id=user_id)\
                .first()
                
            if not pool:
                logger.info("No recommendation pool found - generating new pool")
                # Generate new pool if none exists
                pool_data = generate_recommendation_pool(
                    user_id=user_id,
                    current_mood=data.get('mood', 'Calm'),
                    df_scaled=df_scaled
                )
                
                # Store the new pool
                new_pool = RecommendationPool(
                    user_id=user_id,
                    recommendation_pool=pool_data['recommendation_pool'],
                    user_songs_pool=pool_data['user_songs'],
                    created_at=datetime.utcnow()
                )
                session.add(new_pool)
                session.commit()
                
                recommendations = pool_data
                source = 'new'
            else:
                logger.info("Found existing recommendation pool")
                # Get refreshed recommendations from existing pool
                refreshed_data = refresh_from_pool(
                    pool=pool.serialize(),
                    previous_recs=previous_recs,
                    refresh_type=refresh_type
                )
                recommendations = {
                    'recommendation_pool': refreshed_data['new_songs'],
                    'user_songs': refreshed_data['user_songs'],
                    'has_dqn_model': os.path.exists(f'models/dqn/dqn_user_{user_id}.pth'),
                    'source': refreshed_data['source']
                }
                source = refreshed_data['source']
            
            return jsonify({
                "recommendations": recommendations,
                "source": source
            }), 200
            
        except Exception as e:
            logger.error(f"Error refreshing recommendations: {str(e)}", exc_info=True)
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
        # Convert like/dislike to reward value
        reward = 1 if is_liked else -1
        logger.info(f"Converted feedback to reward value: {reward}")
        
        # Update user history with feedback
        feedback = [{
            'song_id': song_id,
            'reward': reward,
            'mood': current_mood
        }]
        update_user_feedback(user_id, feedback)

        # Get current feedback count
        history_count = session.query(UserHistory).filter_by(user_id=user_id).count()
        logger.info(f"User now has {history_count} feedback entries")

        # Trigger background training if enough feedback is collected
        if history_count >= 5:  # Threshold for training
            logger.info("Feedback threshold reached - initiating DQN training")
            run_background_training(
                user_id=user_id,
                df_scaled=df_scaled,
                features=features,
                feature_weights=feature_weights
            )
            logger.info("DQN training completed")
            return jsonify({
                "message": "Feedback recorded successfully and model trained",
                "training_triggered": True,
                "feedback_count": history_count
            }), 200
        else:
            logger.info(f"Need {5 - history_count} more feedback entries before training")
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
    Uses cached recommendation pools and background updates for better performance.
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
            # Check for existing recommendation pool
            existing_pool = session.query(RecommendationPool)\
                .filter_by(user_id=user_id)\
                .order_by(RecommendationPool.created_at.desc())\
                .first()
                
            recommendations = None
            source = 'new'
            
            if existing_pool and existing_pool.recommendation_pool:
                pool_age = datetime.utcnow() - existing_pool.created_at
                logger.info(f"Found existing pool (age: {pool_age.total_seconds()/60:.1f} minutes)")
                
                # Validate pool data
                has_valid_pool = (
                    existing_pool.recommendation_pool and 
                    len(existing_pool.recommendation_pool) > 0
                )
                
                if has_valid_pool:
                    logger.info(f"Using existing pool with {len(existing_pool.recommendation_pool)} recommendations")
                    if pool_age.total_seconds() < 1800:  # 30 minutes
                        recommendations = {
                            'recommendation_pool': existing_pool.recommendation_pool,
                            'user_songs': existing_pool.user_songs_pool,
                            'has_dqn_model': os.path.exists(f'models/dqn/dqn_user_{user_id}.pth'),
                            'source': 'cached'
                        }
                        source = 'cached'
                    else:
                        # Refresh old but valid pool
                        logger.info("Refreshing old pool")
                        refreshed_pool = refresh_from_pool(
                            pool={
                                'recommendation_pool': existing_pool.recommendation_pool,
                                'user_songs_pool': existing_pool.user_songs_pool,
                                'user_id': user_id
                            },
                            previous_recs=existing_pool.recommendation_pool,
                            refresh_type='smart'
                        )
                        recommendations = {
                            'recommendation_pool': refreshed_pool['new_songs'],
                            'user_songs': refreshed_pool['user_songs'],
                            'has_dqn_model': os.path.exists(f'models/dqn/dqn_user_{user_id}.pth'),
                            'source': 'refreshed'
                        }
                        source = 'refreshed'
            
            # Generate new recommendations if needed
            if not recommendations:
                logger.info("Generating new recommendation pool")
                recommendations = generate_recommendation_pool(
                    user_id=user_id,
                    current_mood=current_mood,
                    df_scaled=df_scaled
                )
                source = recommendations.get('source', 'new')
                
                if not recommendations.get('recommendation_pool'):
                    logger.error("Failed to generate recommendations")
                    return jsonify({"error": "Failed to generate recommendations"}), 500
                
                # Store the new pool
                new_pool = RecommendationPool(
                    user_id=user_id,
                    user_songs_pool=recommendations['user_songs'],
                    recommendation_pool=recommendations['recommendation_pool'],
                    created_at=datetime.utcnow()
                )
                session.add(new_pool)
                session.commit()
                logger.info("Stored new recommendation pool")
            
            logger.info(f"Returning {len(recommendations['recommendation_pool'])} recommendations")
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

