from flask import Blueprint, request, jsonify
from services.recommendation_service import update_user_feedback,  run_background_training, fetch_user_history_and_recommend, update_user_mood
import pandas as pd
import os


recommendation_bp = Blueprint('recommendation', __name__)

# Load the preprocessed dataset
df_scaled = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(__name__)), "../spotify_Song_Dataset/final_dataset.csv"))  # Preprocessed CSV
print(os.path.curdir)
features = ['energy', 'acousticness', 'valence', 'tempo', 'speechiness', 'instrumentalness']
feature_weights = {'energy': 1.0, 'acousticness': 5.0, 'valence': 5.0, 'tempo': 5.0, 'instrumentalness': 5.0, 'speechiness': 5.0}

@recommendation_bp.route('/initial', methods=['POST'])
def get_initial_recommendations():
    """
    Endpoint to get initial recommendations when no feedback is provided yet.
    Accepts user_id.
    """
    data = request.json
    user_id = data.get('user_id')

    if not user_id:
        return jsonify({"error": "Invalid request, missing user_id"}), 400

    # Fetch the initial recommendations based on user's mood or general recommendations
    initial_recommendations = fetch_user_history_and_recommend(user_id, df_scaled, features, feature_weights)

    # Truncate the recommendations to include only necessary fields
    truncated_recommendations = [
        {
            'song_id': rec['song_id'],
            'track_id': rec['track_id'],
            'track_name': rec['track_name'],
            'artist_name': rec['artist_name'],
            'album_name': rec['album_name']
        } 
        for rec in initial_recommendations.to_dict(orient='records')
    ]

    return jsonify({
        "initial_recommendations": truncated_recommendations
    })


@recommendation_bp.route('/refresh', methods=['POST'])
def refresh_recommendations():
    """
    Endpoint to refresh recommendations based on user feedback.
    Accepts a list of songs and feedback (liked/disliked) along with the mood.
    """
    data = request.json
    user_id = data.get('user_id')
    feedback = data.get('feedback')  # List of {'song_id': ..., 'reward': 1/-1, 'mood': 'Calm'}

    if not user_id or not feedback:
        return jsonify({"error": "Invalid request"}), 400

    # Update the user feedback and mood in the database
    update_user_feedback(user_id, feedback)

    # Fetch the new recommendations based on user history and feedback
    new_recommendations = fetch_user_history_and_recommend(user_id, df_scaled, features, feature_weights)

    # Truncate the recommendations to include only necessary fields (similar to /initial)
    truncated_recommendations = [
        {
            'song_id': rec['song_id'],
            'track_id': rec['track_id'],
            'track_name': rec['track_name'],
            'artist_name': rec['artist_name'],
            'album_name': rec['album_name']
        }
        for rec in new_recommendations.to_dict(orient='records')
    ]

    # Return the new truncated recommendations
    return jsonify({
        "new_recommendations": truncated_recommendations
    })


@recommendation_bp.route('/update_mood', methods=['POST'])
def update_user_mood_and_recommend():
    """
    Endpoint to update the user's mood based on chatbot input and refresh recommendations.
    Accepts user_id and mood (as a string representing the mood).
    """
    data = request.json
    user_id = data.get('user_id')
    mood = data.get('mood')  # String representing the mood (e.g., 'Happy', 'Sad')

    if not user_id or not mood:
        return jsonify({"error": "Invalid request"}), 400

    # Update the user's mood in the database
    update_user_mood(user_id, mood)

    # Fetch new recommendations based on updated mood
    new_recommendations = fetch_user_history_and_recommend(user_id, df_scaled, features, feature_weights)

    # Return the new recommendations
    return jsonify({"new_recommendations": new_recommendations.to_dict(orient='records')})