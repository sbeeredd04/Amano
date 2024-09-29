import requests
def get_playlist_tracks(playlist_id, token):
    """
    Fetch songs from a Spotify playlist using the provided token.
    """
    url = f"https://api.spotify.com/v1/playlists/{playlist_id}/tracks"
    headers = {
        "Authorization": f"Bearer {token}"
    }

    tracks = []
    
    while url:
        response = requests.get(url, headers=headers)
        
        if response.status_code == 403:
            raise Exception("Spotify authorization failed. Check if token has required scopes.")
        elif response.status_code != 200:
            raise Exception(f"Error fetching tracks from playlist {playlist_id}: {response.status_code}")
        
        data = response.json()
        tracks.extend([{
            "name": item['track']['name'],
            "artist": item['track']['artists'][0]['name'],
            "album": item['track']['album']['name'],
            "id": item['track']['id']
        } for item in data['items'] if item['track']])

        url = data.get('next')  # Handle pagination
    
    return tracks
