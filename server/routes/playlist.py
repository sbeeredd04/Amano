from flask import Blueprint, request, jsonify
from sqlalchemy.exc import SQLAlchemyError
from models.song_model import Song, Playlist, PlaylistSong
from utils.db import get_session
from flask_cors import CORS

playlists_bp = Blueprint('playlists', __name__)

# Configure CORS for the blueprint
CORS(playlists_bp, 
     origins=["http://localhost:3000"],
     methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
     allow_headers=["Content-Type", "Authorization"],
     supports_credentials=True)

@playlists_bp.route('/get', methods=['GET', 'POST'])
def get_playlists():
    try:
        if request.method == 'POST':
            data = request.json
        else:
            data = request.args

        user_id = data.get('user_id')

        if not user_id:
            return jsonify({"error": "Missing user_id"}), 400

        session = get_session()
        try:
            playlists = session.query(Playlist).filter_by(user_id=user_id).all()
            if not playlists:
                return jsonify({"message": "No playlists found", "playlists": []}), 200

            playlists_with_songs = []
            for playlist in playlists:
                songs = session.query(Song)\
                    .join(PlaylistSong)\
                    .filter(PlaylistSong.playlist_id == playlist.playlist_id)\
                    .all()
                
                playlist_data = playlist.serialize()
                playlist_data['songs'] = [{
                    'song_id': song.song_id,
                    'track_name': song.track_name,
                    'artist_name': song.artists,
                    'track_genre': song.track_genre
                } for song in songs]
                
                playlists_with_songs.append(playlist_data)

            return jsonify({"playlists": playlists_with_songs}), 200

        except SQLAlchemyError as e:
            return jsonify({"error": "Database error"}), 500
        finally:
            session.close()

    except Exception as e:
        return jsonify({"error": "Internal server error"}), 500

@playlists_bp.route('/add', methods=['POST'])
def add_playlist():
    data = request.json
    user_id = data.get('user_id')
    name = data.get('name')
    description = data.get('description', '')
    song_ids = data.get('song_ids', [])

    session = get_session()
    try:
        songs_info = session.query(Song).filter(Song.song_id.in_(song_ids)).all()
        new_playlist = Playlist(user_id=user_id, name=name, description=description)
        session.add(new_playlist)
        session.commit()

        for song_id in song_ids:
            playlist_song = PlaylistSong(playlist_id=new_playlist.playlist_id, song_id=song_id)
            session.add(playlist_song)

        session.commit()
        
        return jsonify({
            "message": "Playlist added successfully", 
            "playlist": new_playlist.serialize()
        }), 201

    except Exception as e:
        session.rollback()
        return jsonify({"error": str(e)}), 500
    finally:
        session.close()

@playlists_bp.route('/songs', methods=['GET'])
def get_songs():
    session = get_session()
    try:
        genre = request.args.get('genre', '').strip()
        search = request.args.get('search', '').strip()
        search_type = request.args.get('type', 'track')
        
        try:
            limit = int(request.args.get('limit', 20))
            offset = int(request.args.get('offset', 0))
        except (ValueError, TypeError):
            limit = 20
            offset = 0

        query = session.query(Song)

        if genre and genre.lower() != 'all':
            query = query.filter(Song.track_genre.ilike(f"%{genre}%"))

        if search:
            if search_type == 'artist':
                query = query.filter(Song.artists.ilike(f"%{search}%"))
            else:
                query = query.filter(Song.track_name.ilike(f"%{search}%"))
            
        total_count = query.count()
        songs = query.order_by(Song.popularity.desc()).offset(offset).limit(limit).all()

        serialized_songs = [{
            'song_id': song.song_id,
            'track_name': song.track_name,
            'artist_name': song.artists,
            'album_name': song.album_name,
            'track_genre': song.track_genre,
            'popularity': song.popularity
        } for song in songs]

        return jsonify({
            "songs": serialized_songs,
            "total_count": total_count
        }), 200
        
    except Exception as e:
        return jsonify({
            "error": str(e),
            "songs": [],
            "total_count": 0
        }), 500
    finally:
        session.close()

@playlists_bp.route('/user_songs', methods=['POST'])
def get_user_songs():
    data = request.json
    user_id = data.get('user_id')

    if not user_id:
        return jsonify({"error": "Missing user_id"}), 400

    session = get_session()
    try:
        user_playlists = session.query(Playlist).filter_by(user_id=user_id).all()

        if not user_playlists:
            return jsonify({"user_songs": [], "source": "global"}), 200

        user_song_ids = []
        for playlist in user_playlists:
            playlist_songs = session.query(PlaylistSong).filter_by(playlist_id=playlist.playlist_id).all()
            user_song_ids.extend([song.song_id for song in playlist_songs])

        if not user_song_ids:
            return jsonify({"user_songs": [], "source": "global"}), 200

        songs = session.query(Song).filter(Song.song_id.in_(user_song_ids)).all()
        songs_serialized = [
            {
                'song_id': song.song_id,
                'track_id': song.track_id,
                'track_name': song.track_name,
                'artist_name': song.artists,
                'album_name': song.album_name,
                'track_genre': song.track_genre,
                'popularity': song.popularity
            }
            for song in songs
        ]
        
        return jsonify({"user_songs": songs_serialized, "source": "playlists"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        session.close()

@playlists_bp.route('/delete', methods=['POST'])
def delete_playlist():
    data = request.json
    user_id = data.get('user_id')
    playlist_id = data.get('playlist_id')

    if not user_id or not playlist_id:
        return jsonify({"error": "Missing required fields"}), 400

    session = get_session()
    try:
        playlist = session.query(Playlist).filter_by(
            playlist_id=playlist_id, 
            user_id=user_id
        ).first()

        if not playlist:
            return jsonify({"error": "Playlist not found or unauthorized"}), 404

        session.query(PlaylistSong).filter_by(playlist_id=playlist_id).delete()
        session.delete(playlist)
        session.commit()

        return jsonify({"message": "Playlist deleted successfully"}), 200

    except Exception as e:
        session.rollback()
        return jsonify({"error": str(e)}), 500
    finally:
        session.close()

@playlists_bp.route('/edit', methods=['POST'])
def edit_playlist():
    data = request.json
    user_id = data.get('user_id')
    playlist_id = data.get('playlist_id')
    name = data.get('name')
    song_ids = data.get('song_ids', [])

    if not all([user_id, playlist_id, name]):
        return jsonify({"error": "Missing required fields"}), 400

    session = get_session()
    try:
        playlist = session.query(Playlist).filter_by(
            playlist_id=playlist_id, 
            user_id=user_id
        ).first()

        if not playlist:
            return jsonify({"error": "Playlist not found or unauthorized"}), 404

        playlist.name = name
        session.query(PlaylistSong).filter_by(playlist_id=playlist_id).delete()
        
        for song_id in song_ids:
            playlist_song = PlaylistSong(playlist_id=playlist_id, song_id=song_id)
            session.add(playlist_song)

        session.commit()

        updated_songs = session.query(Song)\
            .join(PlaylistSong)\
            .filter(PlaylistSong.playlist_id == playlist_id)\
            .all()

        response_data = playlist.serialize()
        response_data['songs'] = [{
            'song_id': song.song_id,
            'track_name': song.track_name,
            'artist_name': song.artists,
            'track_genre': song.track_genre
        } for song in updated_songs]

        return jsonify({
            "message": "Playlist updated successfully",
            "playlist": response_data
        }), 200

    except Exception as e:
        session.rollback()
        return jsonify({"error": str(e)}), 500
    finally:
        session.close()

@playlists_bp.route('/genres', methods=['GET'])
def get_genres():
    session = get_session()
    try:
        genres = ['All'] + [g[0] for g in session.query(Song.track_genre.distinct()).all()]
        return jsonify({"genres": genres}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@playlists_bp.route('/artists', methods=['GET'])
def get_artists():
    session = get_session()
    try:
        artists = [a[0] for a in session.query(Song.artists.distinct()).all()]
        return jsonify({"artists": artists}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@playlists_bp.route('/remove_song', methods=['POST'])
def remove_song():
    data = request.json
    user_id = data.get('user_id')
    playlist_id = data.get('playlist_id')
    song_id = data.get('song_id')

    if not all([user_id, playlist_id, song_id]):
        return jsonify({"error": "Missing required fields"}), 400

    session = get_session()
    try:
        playlist = session.query(Playlist).filter_by(
            playlist_id=playlist_id, 
            user_id=user_id
        ).first()

        if not playlist:
            return jsonify({"error": "Playlist not found or unauthorized"}), 404

        session.query(PlaylistSong).filter_by(
            playlist_id=playlist_id,
            song_id=song_id
        ).delete()

        session.commit()

        updated_songs = session.query(Song)\
            .join(PlaylistSong)\
            .filter(PlaylistSong.playlist_id == playlist_id)\
            .all()

        response_data = playlist.serialize()
        response_data['songs'] = [{
            'song_id': song.song_id,
            'track_name': song.track_name,
            'artist_name': song.artists,
            'track_genre': song.track_genre
        } for song in updated_songs]

        return jsonify({
            "message": "Song removed successfully",
            "playlist": response_data
        }), 200

    except Exception as e:
        session.rollback()
        return jsonify({"error": str(e)}), 500
    finally:
        session.close()

def get_all_user_playlist_songs(user_id):
    """Get all unique songs from all playlists of a user."""
    session = get_session()
    try:
        playlists = session.query(Playlist).filter_by(user_id=user_id).all()
        all_songs = set()
        for playlist in playlists:
            songs = session.query(Song)\
                .join(PlaylistSong)\
                .filter(PlaylistSong.playlist_id == playlist.playlist_id)\
                .all()
            all_songs.update(song.song_id for song in songs)
        
        return list(all_songs)
        
    except Exception as e:
        return []
    finally:
        session.close()
