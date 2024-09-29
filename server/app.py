from flask import Flask, jsonify
from flask_cors import CORS
from routes.playlist import playlists_bp
from routes.chatbot import chatbot_bp
from routes.recommendation import recommendation_bp
from utils.db import init_db  # Import init_db to initialize the database if needed

app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing for requests from the frontend

# Register blueprints (routes) from different modules
app.register_blueprint(playlists_bp, url_prefix='/playlists')
app.register_blueprint(chatbot_bp, url_prefix='/chatbot')
app.register_blueprint(recommendation_bp, url_prefix='/recommendation')

# Initialize the database
with app.app_context():
    init_db()  # Initialize database if tables are not created

# Add a route for the base URL '/'
@app.route('/')
def welcome():
    return jsonify({
        "message": "Welcome to Amano Backend API",
        "description": "This is the backend system for the Amano project, handling playlist recommendations, mood updates, and more."
    })

if __name__ == '__main__':
    # Run the Flask app with multithreading enabled to support background training
    app.run(debug=True, threaded=True)
