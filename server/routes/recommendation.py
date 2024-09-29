from flask import Blueprint, request, jsonify
from services.recommendation_service import update_user_feedback, fetch_user_history_and_train_with_mood, update_user_mood
from utils.db import get_session

recommendation_bp = Blueprint('recommendation', __name__)

@recommendation_bp.route('/refresh', methods=['POST'])
def refresh_recommendations():
    """
    Endpoint to refresh recommendations based on user feedback.
    Accepts a list of songs and feedback (liked/disliked) along with the mood.
    """
    data = request.json
    user_id = data.get('user_id')
    feedback = data.get('feedback')  # List of {'song_id': ..., 'liked': 1/-1, 'mood': 'Happy'}

    if not user_id or not feedback:
        return jsonify({"error": "Invalid request"}), 400

    # Update the user feedback and mood in the database
    update_user_feedback(user_id, feedback)

    # Load the preprocessed dataset
    df_scaled = pd.read_csv('spotify_Song_Dataset/final_dataset.csv')  # Preprocessed CSV
    features = ['energy', 'acousticness', 'valence', 'tempo', 'speechiness', 'instrumentalness']
    feature_weights = {'energy': 1.0, 'acousticness': 5.0, 'valence': 5.0, 'tempo': 5.0, 'instrumentalness': 5.0, 'speechiness': 5.0}

    # Fetch the new recommendations after updating the feedback
    new_recommendations = fetch_user_history_and_train_with_mood(user_id, df_scaled, features, feature_weights)

    # Return the new recommendations to the frontend
    return jsonify({"new_recommendations": new_recommendations.to_dict(orient='records')})

@recommendation_bp.route('/update_mood', methods=['POST'])
def update_user_mood_and_recommend():
    """
    Endpoint to update the user's mood based on chatbot input and refresh recommendations.
    Accepts user_id and mood (as a string representing the mood).
    """
    data = request.json
    user_id = data.get('user_id')
    mood = data.get('mood')  # Mood is now a string (e.g., 'Happy')

    if not user_id or not mood:
        return jsonify({"error": "Invalid request"}), 400

    # Update the user's mood in the database
    update_user_mood(user_id, mood)

    # Load the preprocessed dataset
    df_scaled = pd.read_csv('spotify_Song_Dataset/final_dataset.csv')  # Preprocessed CSV
    features = ['energy', 'acousticness', 'valence', 'tempo', 'speechiness', 'instrumentalness']
    feature_weights = {'energy': 1.0, 'acousticness': 5.0, 'valence': 5.0, 'tempo': 5.0, 'instrumentalness': 5.0, 'speechiness': 5.0}

    # Fetch new recommendations based on updated mood
    new_recommendations = fetch_user_history_and_train_with_mood(user_id, df_scaled, features, feature_weights)

    # Return the new recommendations
    return jsonify({"new_recommendations": new_recommendations.to_dict(orient='records')})
