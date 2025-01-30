from flask import Blueprint, request, jsonify
from services.recommendation_service import (
    update_user_feedback, 
    run_background_training, 
    fetch_user_history_and_recommend, 
    update_user_mood,
    get_user_mood,
    get_initial_recommendations,
    load_and_get_dataset
)
import pandas as pd
import os
from utils.db import get_session
from models.song_model import UserMood, UserHistory
import logging
from flask_cors import CORS, cross_origin

logger = logging.getLogger(__name__)

recommendation_bp = Blueprint('recommendation', __name__)
CORS(recommendation_bp, 
     resources={
         r"/*": {
             "origins": ["http://localhost:3000"],
             "methods": ["GET", "POST", "OPTIONS"],
             "allow_headers": ["Content-Type"],
             "supports_credentials": True
         }
     })

# Load the preprocessed dataset
try:
    logger.info("Loading dataset...")
    df_scaled = load_and_get_dataset()
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

@recommendation_bp.route('/initial', methods=['GET'])
@cross_origin(supports_credentials=True)
def get_initial_recommendations_route():
    """
    Endpoint to get initial recommendations based on user's playlist history or default songs.
    """
    try:
        user_id = request.args.get('user_id')
        use_user_songs = request.args.get('use_user_songs', 'true').lower() == 'true'
        
        logger.info(f"Getting initial recommendations for user {user_id}")
        logger.debug(f"Using user songs: {use_user_songs}")
        
        if not user_id:
            return jsonify({"error": "User ID is required"}), 400
            
        recommendations = get_initial_recommendations(
            user_id=user_id,
            use_user_songs=use_user_songs,
            df_scaled=df_scaled
        )
        
        logger.info(f"Generated {len(recommendations)} initial recommendations")
        
        return jsonify({
            "recommendations": recommendations,
            "source": "user_playlist" if use_user_songs else "default_songs"
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting initial recommendations: {e}")
        return jsonify({"error": str(e)}), 500

@recommendation_bp.route('/refresh', methods=['POST'])
@cross_origin(supports_credentials=True)
def refresh_recommendations():
    """
    Endpoint to refresh recommendations based on user feedback and mood.
    """
    logger.info("=== Starting Recommendation Refresh Process ===")
    try:
        data = request.json
        user_id = data.get('user_id')
        mood = data.get('mood')
        use_user_songs = data.get('use_user_songs', True)
        
        logger.info(f"Request Parameters:")
        logger.info(f"- User ID: {user_id}")
        logger.info(f"- Mood: {mood}")
        logger.info(f"- Using User Songs: {use_user_songs}")
        
        if not user_id:
            logger.error("Missing user_id in request")
            return jsonify({"error": "Invalid request, missing user_id"}), 400

        # Check if user exists and has history
        session = get_session()
        user_history_count = session.query(UserHistory).filter_by(user_id=user_id).count()
        logger.info(f"User has {user_history_count} historical interactions")

        # Get new recommendations
        logger.info("Calling fetch_user_history_and_recommend")
        recommendations = fetch_user_history_and_recommend(
            user_id=user_id,
            mood=mood,
            use_user_songs=use_user_songs,
            df_scaled=df_scaled
        )
        
        logger.info(f"Received {len(recommendations)} recommendations")
        logger.debug(f"First 3 recommendations: {recommendations[:3]}")

        return jsonify({
            "recommendations": recommendations,
            "source": "user_playlist" if use_user_songs else "default_songs"
        }), 200

    except Exception as e:
        logger.error(f"Error in refresh_recommendations: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500
    finally:
        if 'session' in locals():
            session.close()

@recommendation_bp.route('/feedback', methods=['POST'])
def handle_feedback():
    """
    Endpoint to handle user feedback (likes/dislikes) for songs.
    """
    data = request.json
    user_id = data.get('user_id')
    song_id = data.get('song_id')
    is_liked = data.get('is_liked')
    current_mood = data.get('mood')

    if not all([user_id, song_id, is_liked is not None]):
        return jsonify({"error": "Missing required fields"}), 400

    try:
        session = get_session()
        # Convert like/dislike to reward value
        reward = 1 if is_liked else -1
        
        # Update user history with feedback
        feedback = [{
            'song_id': song_id,
            'reward': reward,
            'mood': current_mood
        }]
        update_user_feedback(user_id, feedback)

        # Trigger background training if enough feedback is collected
        history_count = session.query(UserHistory).filter_by(user_id=user_id).count()
        if history_count >= 5:  # Threshold for training
            run_background_training(user_id)

        return jsonify({"message": "Feedback recorded successfully"}), 200

    except Exception as e:
        logging.error(f"Error recording feedback: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        session.close()

@recommendation_bp.route('/mood', methods=['GET', 'POST'])
@cross_origin(supports_credentials=True)
def manage_mood():
    """
    Endpoint to get or update user's current mood.
    """
    logger.debug(f"Received {request.method} request to /mood endpoint")
    logger.debug(f"Request headers: {dict(request.headers)}")
    logger.debug(f"Request args: {dict(request.args)}")
    
    try:
        user_id = request.args.get('user_id') if request.method == 'GET' else request.json.get('user_id')
        
        if not user_id:
            logger.error("No user_id provided")
            return jsonify({"error": "user_id is required"}), 400
            
        if request.method == 'GET':
            logger.debug(f"Getting mood for user {user_id}")
            mood = get_user_mood(user_id)
            logger.debug(f"Retrieved mood: {mood}")
            return jsonify({"mood": mood})
        else:
            mood = request.json.get('mood')
            if not mood:
                logger.error("No mood provided in POST request")
                return jsonify({"error": "Mood is required"}), 400
                
            logger.debug(f"Updating mood for user {user_id} to {mood}")
            update_user_mood(user_id, mood)
            return jsonify({"message": "Mood updated successfully"})
            
    except Exception as e:
        logger.error(f"Error managing mood: {e}")
        return jsonify({"error": str(e)}), 500

