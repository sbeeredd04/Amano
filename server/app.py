from flask import Flask, jsonify
from flask_cors import CORS
from routes.playlist import playlists_bp
from routes.chatbot import chatbot_bp
from routes.recommendation import recommendation_bp
from utils.db import init_db  # Import init_db to initialize the database if needed
from routes.signup_login import signup_login_bp
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def create_app():
    app = Flask(__name__)

    # Configure CORS globally for the app
    CORS(app, 
         origins=["http://localhost:3000"],
         methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
         allow_headers=["Content-Type", "Authorization"],
         supports_credentials=True)
    
    logger.debug("CORS configuration applied")

    # Register blueprints (routes) from different modules
    app.register_blueprint(playlists_bp, url_prefix='/playlists')
    app.register_blueprint(chatbot_bp, url_prefix='/chatbot')
    app.register_blueprint(recommendation_bp, url_prefix='/recommendation')
    app.register_blueprint(signup_login_bp, url_prefix='/auth')

    # Initialize the database
    with app.app_context():
        init_db()  # Initialize database if tables are not created

    @app.before_request
    def before_request():
        """Initialize database session for each request"""
        init_db()

    @app.after_request
    def after_request(response):
        logger.debug(f"Response headers: {dict(response.headers)}")
        return response

    # Add a route for the base URL '/'
    @app.route('/')
    def welcome():
        return jsonify({
            "message": "Welcome to Amano Backend API",
            "description": "This is the backend system for the Amano project, handling playlist recommendations, mood updates, and more."
        })

    return app

if __name__ == "__main__":
    app = create_app()
    app.run(debug=True, threaded=True)
