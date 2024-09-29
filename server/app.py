# server/app.py
from flask import Flask
from flask_cors import CORS
from routes.playlists import playlists_bp
from routes.chatbot import chatbot_bp
from routes.recommendation import recommendation_bp

app = Flask(__name__)
CORS(app)

# Register routes from different files
app.register_blueprint(playlists_bp, url_prefix='/playlists')
app.register_blueprint(chatbot_bp, url_prefix='/chatbot')
app.register_blueprint(recommendation_bp, url_prefix='/recommendation')

if __name__ == '__main__':
    app.run(debug=True)
