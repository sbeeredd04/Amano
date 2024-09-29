from flask import Blueprint, request, jsonify
from services.llm_service import invoke_llm
from services.recommendation_service import update_user_mood

chatbot_bp = Blueprint('chatbot', __name__)

# Define the list of moods
MOOD_LIST = ['Angry', 'Content', 'Happy', 'Delighted', 'Calm', 'Sleepy', 'Sad', 'Depressed', 'Excited']

def map_sentiment_to_mood(sentiment):
    """
    Map the sentiment returned from LLM to one of the 9 predefined moods.
    """
    sentiment_lower = sentiment.lower()
    if 'angry' in sentiment_lower:
        return 'Angry'
    elif 'content' in sentiment_lower:
        return 'Content'
    elif 'happy' in sentiment_lower or 'delighted' in sentiment_lower:
        return 'Happy'  # Group 'happy' and 'delighted'
    elif 'calm' in sentiment_lower:
        return 'Calm'
    elif 'sleepy' in sentiment_lower:
        return 'Sleepy'
    elif 'sad' in sentiment_lower:
        return 'Sad'
    elif 'depressed' in sentiment_lower:
        return 'Depressed'
    elif 'excited' in sentiment_lower:
        return 'Excited'
    else:
        return 'Calm'  # Default fallback if no match

@chatbot_bp.route('/', methods=['POST'])
def chatbot():
    input_text = request.json.get('input')
    user_id = request.json.get('user_id')

    if not input_text or not user_id:
        return jsonify({"error": "Invalid request"}), 400

    # Perform sentiment analysis using LLM
    sentiment = invoke_llm(input_text)
    
    # Map the sentiment to one of the predefined moods
    mood = map_sentiment_to_mood(sentiment)
    
    # Update the user's mood in the database
    update_user_mood(user_id, mood)
    
    return jsonify({"mood": mood})
