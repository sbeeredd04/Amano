from flask import Blueprint, request, jsonify
from models.song_model import User
from utils.db import get_session
import logging

logging.basicConfig(level=logging.DEBUG)

signup_login_bp = Blueprint('signup_login', __name__)

@signup_login_bp.route('/login', methods=['POST'])
def login():
    """
    Log in an existing user.
    Expects JSON payload with 'email'.
    """
    data = request.json
    email = data.get('email')

    if not email:
        return jsonify({"error": "Missing required field: 'email'"}), 400

    session = get_session()
    try:
        # Check if the user exists
        existing_user = session.query(User).filter_by(email=email).first()
        if existing_user:
            logging.info(f"User with email {email} logged in successfully.")
            return jsonify({"message": "Login successful", "user": existing_user.serialize()}), 200

        # If the user does not exist, return an error
        logging.warning(f"Login failed: User with email {email} does not exist.")
        return jsonify({"error": "User does not exist. Please sign up first."}), 404

    except Exception as e:
        logging.error(f"Error logging in user: {e}")
        return jsonify({"error": "Internal server error"}), 500

    finally:
        session.close()


@signup_login_bp.route('/signup', methods=['POST'])
def signup():
    """
    Sign up a new user.
    Expects JSON payload with 'email', 'name', and optional 'image'.
    """
    data = request.json
    email = data.get('email')
    name = data.get('name')
    image = data.get('image', None)

    if not email or not name:
        return jsonify({"error": "Missing required fields: 'email' or 'name'"}), 400

    session = get_session()
    try:
        # Check if the user already exists
        existing_user = session.query(User).filter_by(email=email).first()
        if existing_user:
            logging.info(f"User with email {email} already exists.")
            return jsonify({"error": "User already exists. Please log in."}), 409

        # Create a new user
        new_user = User(email=email, name=name, image=image)
        session.add(new_user)
        session.commit()
        logging.info(f"User with email {email} signed up successfully.")
        return jsonify({"message": "Signup successful", "user": new_user.serialize()}), 201

    except Exception as e:
        session.rollback()
        logging.error(f"Error signupg up user: {e}")
        return jsonify({"error": "Internal server error"}), 500

    finally:
        session.close()
