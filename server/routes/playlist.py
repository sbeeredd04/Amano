# server/routes/playlists.py
from flask import Blueprint, request, jsonify
from services.spotify_service import get_playlist_tracks

playlists_bp = Blueprint('playlists', __name__)

@playlists_bp.route('/', methods=['POST'])
def handle_playlists():
    data = request.json
    playlist_ids = data.get('playlistIds')
    token = data.get('token')
    
    all_tracks = []
    for playlist_id in playlist_ids:
        tracks = get_playlist_tracks(playlist_id, token)
        all_tracks.extend(tracks)
    
    # Process tracks or save to DB
    return jsonify({"tracks": all_tracks, "track_count": len(all_tracks)})
