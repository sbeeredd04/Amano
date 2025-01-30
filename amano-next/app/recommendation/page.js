"use client";

import { useState, useEffect } from "react";

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'https://70bnmmdc-5000.usw3.devtunnels.ms';

const SongCard = ({ song, isSelected, onToggle }) => {
  return (
    <div 
      className={`p-4 rounded-lg ${
        isSelected 
          ? 'bg-blue-600 hover:bg-blue-500' 
          : 'bg-gray-800 hover:bg-gray-700'
      } cursor-pointer transition-colors`}
      onClick={onToggle}
    >
      <div className="flex flex-col h-full">
        <h3 className="text-lg font-semibold mb-2 truncate" title={song.track_name}>
          {song.track_name}
        </h3>
        <p className="text-gray-300 mb-1 truncate" title={song.artists}>
          {song.artists}
        </p>
        <p className="text-gray-400 text-sm mb-2 truncate" title={song.album_name}>
          {song.album_name}
        </p>
        <div className="flex justify-between items-center mt-auto">
          <span className="px-2 py-1 bg-gray-700 rounded text-xs">
            {song.track_genre}
          </span>
          <span className="text-xs text-gray-400">
            Popularity: {song.popularity}
          </span>
        </div>
      </div>
    </div>
  );
};

const RecommendationSourceToggle = ({ useUserSongs, setUseUserSongs }) => (
  <div className="flex items-center space-x-2 mb-4">
    <label className="text-sm font-medium">Recommendation Source:</label>
    <select
      value={useUserSongs.toString()}
      onChange={(e) => setUseUserSongs(e.target.value === 'true')}
      className="bg-gray-800 text-white rounded px-3 py-1"
    >
      <option value="true">User History</option>
      <option value="false">Default Songs</option>
    </select>
  </div>
);

export default function RecommendationPage() {
  const [playlists, setPlaylists] = useState([]);
  const [recommendations, setRecommendations] = useState([]);
  const [mood, setMood] = useState("");
  const [message, setMessage] = useState("");
  const [showSongsSection, setShowSongsSection] = useState(false); // Toggle songs section visibility
  const [playlistName, setPlaylistName] = useState("");
  const [searchQuery, setSearchQuery] = useState("");
  const [genreFilter, setGenreFilter] = useState("");
  const [selectedSongs, setSelectedSongs] = useState([]);
  const [songs, setSongs] = useState([]);
  const [limit, setLimit] = useState(20);
  const [offset, setOffset] = useState(0);
  const [totalSongs, setTotalSongs] = useState(0);
  const [loading, setLoading] = useState(false);
  const [currentPage, setCurrentPage] = useState(1);
  const [itemsPerPage] = useState(12); // Number of songs per page
  const [editingPlaylist, setEditingPlaylist] = useState(null);
  const [searchType, setSearchType] = useState('track'); // 'track' or 'artist'
  const [genres, setGenres] = useState([]);
  const [artists, setArtists] = useState([]);
  const [currentMood, setCurrentMood] = useState("Calm");
  const moods = ["Angry", "Content", "Happy", "Delighted", "Calm", "Sleepy", "Sad", "Depressed", "Excited"];
  const [useUserSongs, setUseUserSongs] = useState(true);
  const [feedbackStatus, setFeedbackStatus] = useState({});
  const [userId, setUserId] = useState(1); // Add default userId state or get from auth

  useEffect(() => {
    const userId = sessionStorage.getItem("user_id");
    if (!userId) {
      console.error("User ID not found in session. Redirecting to login.");
      window.location.href = "/";
      return;
    }

    const fetchPlaylists = async () => {
      try {
        const userId = sessionStorage.getItem("user_id");
        console.debug("Fetching playlists for user:", userId);
        
        if (!userId) {
          console.warn("No user ID found in session");
          return;
        }

        const response = await fetch(`${API_URL}/playlists/get`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ user_id: userId })
        });
        
        console.debug("Playlists response status:", response.status);
        
        if (!response.ok) {
          const errorData = await response.json();
          throw new Error(`Server error: ${errorData.error || response.statusText}`);
        }

        const data = await response.json();
        console.debug("Playlists fetched:", data);
        setPlaylists(data.playlists || []);
      } catch (error) {
        console.error("Error fetching playlists:", error.message);
        setMessage("Failed to fetch playlists. Please try again.");
      }
    };

    const fetchMood = async () => {
      try {
        console.log("Fetching mood for user:", userId);
        const response = await fetch(`${API_URL}/recommendation/mood?user_id=${userId}`, {
          method: 'GET',
          credentials: 'include',
          headers: {
            'Content-Type': 'application/json',
          }
        });

        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        setMood(data.mood);
      } catch (error) {
        console.error("Error fetching current mood:", error);
      }
    };

    const fetchInitialData = async () => {
      try {
        console.log("Fetching initial data...");
        console.debug(`Parameters: user_id=${userId}, use_user_songs=${useUserSongs}`);
        
        const response = await fetch(
          `${API_URL}/recommendation/initial?user_id=${userId}&use_user_songs=${useUserSongs}`,
          {
            method: 'GET',
            credentials: 'include',
            headers: {
              'Content-Type': 'application/json',
            }
          }
        );

        if (!response.ok) {
          const errorData = await response.json();
          throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        console.debug("Received recommendations:", data);
        setRecommendations(data.recommendations || []);
      } catch (error) {
        console.error("Error fetching initial data:", error);
        setMessage(`Error: ${error.message}`);
      }
    };

    fetchPlaylists();
    fetchMood();
    fetchInitialData();
  }, [userId, useUserSongs]); // Add userId and useUserSongs as dependencies

  const handleGenerateRecommendations = async () => {
    const userId = sessionStorage.getItem("user_id");
    console.log("=== Starting Recommendation Generation ===");
    console.log(`Current State - Mood: ${currentMood}, UseUserSongs: ${useUserSongs}`);
    
    if (!userId) {
      console.error("User ID not found in session.");
      setMessage("User not logged in. Please log in again.");
      return;
    }

    try {
      console.log("Making POST request to /recommendation/refresh with params:", {
        userId,
        mood: currentMood,
        useUserSongs
      });

      const response = await fetch(
        `${API_URL}/recommendation/refresh`,
        {
          method: "POST",
          headers: { 
            "Content-Type": "application/json",
            "Accept": "application/json"
          },
          credentials: 'include',
          body: JSON.stringify({ 
            user_id: userId, 
            mood: currentMood,
            use_user_songs: useUserSongs 
          }),
        }
      );

      console.log("Response status:", response.status);

      if (!response.ok) {
        const errorData = await response.json();
        console.error("Server returned error:", errorData);
        throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      console.log("Received recommendations:", {
        count: data.recommendations?.length || 0,
        source: data.source,
        firstFew: data.recommendations?.slice(0, 3)
      });
      
      setRecommendations(data.recommendations || []);
      setMessage("Recommendations updated successfully!");
    } catch (error) {
      console.error("Error in recommendation generation:", error);
      setMessage(`Error: ${error.message}`);
    }
  };

  const fetchGenres = async () => {
    try {
      const response = await fetch(`${API_URL}/playlists/genres`);
      const data = await response.json();
      setGenres(data.genres);
    } catch (error) {
      console.error("Error fetching genres:", error);
    }
  };

  const fetchArtists = async () => {
    try {
      const response = await fetch(`${API_URL}/playlists/artists`);
      const data = await response.json();
      setArtists(data.artists);
    } catch (error) {
      console.error("Error fetching artists:", error);
    }
  };

  const fetchSongs = async (page = 1) => {
    setLoading(true);
    setMessage("");
    
    try {
      const offset = (page - 1) * itemsPerPage;
      const queryParams = new URLSearchParams({
        limit: itemsPerPage.toString(),
        offset: offset.toString(),
        genre: genreFilter || '',
        search: searchQuery || '',
        search_type: searchType
      });

      console.debug("Fetching songs with params:", {
        page,
        offset,
        genre: genreFilter,
        search: searchQuery,
        searchType
      });

      const response = await fetch(`${API_URL}/playlists/songs?${queryParams}`);
      
      if (!response.ok) {
        throw new Error(`Server error: ${response.statusText}`);
      }
      
      const data = await response.json();
      console.debug("Songs fetched:", data);
      
      setSongs(data.songs);
      setTotalSongs(data.total_count);
      setCurrentPage(page);
    } catch (error) {
      console.error("Error fetching songs:", error);
      setMessage(`Error: ${error.message}`);
      setSongs([]);
      setTotalSongs(0);
    } finally {
      setLoading(false);
    }
  };

  const toggleSongSelection = (songId) => {
    setSelectedSongs((prevSelected) =>
      prevSelected.includes(songId)
        ? prevSelected.filter((id) => id !== songId)
        : [...prevSelected, songId]
    );
  };

  const handleAddPlaylist = async () => {
    const userId = sessionStorage.getItem("user_id");
    if (!userId) {
      setMessage("User not logged in. Please log in again.");
      return;
    }

    if (!playlistName || selectedSongs.length === 0) {
      alert("Please provide a playlist name and select at least one song.");
      return;
    }

    try {
      const response = await fetch(
        `${API_URL}/playlists/add`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            user_id: userId,
            name: playlistName,
            song_ids: selectedSongs,
          }),
        }
      );

      const data = await response.json();
      if (response.ok) {
        setMessage("Playlist added successfully.");
        setPlaylists((prevPlaylists) => [...prevPlaylists, data.playlist]);
        console.debug("Playlist added successfully:", data.playlist);
        setShowSongsSection(false);
        setPlaylistName("");
        setSelectedSongs([]);
      } else {
        setMessage(data.error);
      }
    } catch (error) {
      console.error("Error adding playlist:", error);
    }
  };

  const handleDeletePlaylist = async (playlistId) => {
    const userId = sessionStorage.getItem("user_id");
    if (!userId) {
      setMessage("User not logged in. Please log in again.");
      return;
    }

    if (window.confirm("Are you sure you want to delete this playlist?")) {
      try {
        const response = await fetch(
          `${API_URL}/playlists/delete`,
          {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              user_id: userId,
              playlist_id: playlistId,
            }),
          }
        );

        if (response.ok) {
          setPlaylists(playlists.filter(p => p.playlist_id !== playlistId));
          setMessage("Playlist deleted successfully");
        }
      } catch (error) {
        console.error("Error deleting playlist:", error);
        setMessage("Error deleting playlist");
      }
    }
  };

  const handleEditPlaylist = async (playlist) => {
    setEditingPlaylist(playlist);
    setPlaylistName(playlist.name);
    setSelectedSongs([]); // You might want to fetch current songs here
    setShowSongsSection(true);
  };

  const handleSaveEdit = async () => {
    const userId = sessionStorage.getItem("user_id");
    if (!userId) {
      setMessage("User not logged in. Please log in again.");
      return;
    }

    try {
      const response = await fetch(
        `${API_URL}/playlists/edit`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            user_id: userId,
            playlist_id: editingPlaylist.playlist_id,
            name: playlistName,
            song_ids: selectedSongs,
          }),
        }
      );

      const data = await response.json();
      if (response.ok) {
        setPlaylists(playlists.map(p => 
          p.playlist_id === editingPlaylist.playlist_id ? data.playlist : p
        ));
        setMessage("Playlist updated successfully");
        setShowSongsSection(false);
        setEditingPlaylist(null);
      }
    } catch (error) {
      console.error("Error updating playlist:", error);
      setMessage("Error updating playlist");
    }
  };

  const renderPagination = () => {
    const totalPages = Math.ceil(totalSongs / itemsPerPage);
    const pages = [];
    
    // Show max 5 pages with current page in the middle when possible
    let startPage = Math.max(1, currentPage - 2);
    let endPage = Math.min(totalPages, startPage + 4);
    
    if (endPage - startPage < 4) {
      startPage = Math.max(1, endPage - 4);
    }

    for (let i = startPage; i <= endPage; i++) {
      pages.push(
        <button
          key={i}
          onClick={() => {
            setCurrentPage(i);
            fetchSongs(i);
          }}
          className={`px-3 py-1 rounded ${
            currentPage === i
              ? "bg-blue-500 text-white"
              : "bg-gray-200 text-gray-700 hover:bg-gray-300"
          }`}
        >
          {i}
        </button>
      );
    }

    return (
      <div className="flex space-x-2">
        {currentPage > 1 && (
          <button
            onClick={() => {
              setCurrentPage(currentPage - 1);
              fetchSongs(currentPage - 1);
            }}
            className="px-3 py-1 rounded bg-gray-200 text-gray-700 hover:bg-gray-300"
          >
            Previous
          </button>
        )}
        {pages}
        {currentPage < totalPages && (
          <button
            onClick={() => {
              setCurrentPage(currentPage + 1);
              fetchSongs(currentPage + 1);
            }}
            className="px-3 py-1 rounded bg-gray-200 text-gray-700 hover:bg-gray-300"
          >
            Next
          </button>
        )}
      </div>
    );
  };

  const handleMoodChange = async (newMood) => {
    const userId = sessionStorage.getItem("user_id");
    try {
      const response = await fetch(`${API_URL}/recommendation/mood`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          user_id: userId,
          mood: newMood
        })
      });

      if (response.ok) {
        setCurrentMood(newMood);
        handleGenerateRecommendations();
      } else {
        console.error("Failed to update mood");
      }
    } catch (error) {
      console.error("Error updating mood:", error);
    }
  };

  useEffect(() => {
    const fetchCurrentMood = async () => {
      try {
        const userId = sessionStorage.getItem("user_id");
        const response = await fetch(`${API_URL}/recommendation/mood?user_id=${userId}`, {
          credentials: 'include',
          headers: {
            'Content-Type': 'application/json'
          }
        });
        if (response.ok) {
          const data = await response.json();
          setCurrentMood(data.mood);
        }
      } catch (error) {
        console.error("Error fetching current mood:", error);
      }
    };

    fetchCurrentMood();
  }, []);

  useEffect(() => {
    fetchGenres();
    fetchArtists();
  }, []);

  useEffect(() => {
    fetchSongs(1);
  }, []);

  const GenreFilter = () => (
    <div className="flex items-center space-x-2">
      <label htmlFor="genre-filter" className="text-sm font-medium">
        Genre:
      </label>
      <select
        id="genre-filter"
        value={genreFilter}
        onChange={(e) => {
          setGenreFilter(e.target.value);
          fetchSongs(1); // Reset to first page when changing genre
        }}
        className="bg-gray-800 text-white rounded px-3 py-1"
      >
        {genres.map((genre) => (
          <option key={genre} value={genre === 'All' ? '' : genre}>
            {genre}
          </option>
        ))}
      </select>
    </div>
  );

  return (
    <div className="h-screen flex flex-col bg-black text-white">
      {/* Floating Navbar */}
      <nav className="fixed top-0 left-0 w-full bg-gradient-to-r from-green-700 via-green-600 to-green-500 text-white shadow-md z-10">
        <div className="flex justify-between items-center px-6 py-3">
          <h1 className="text-2xl font-bold">Amano</h1>
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-2">
              <span>Current Mood:</span>
              <select
                value={currentMood}
                onChange={(e) => handleMoodChange(e.target.value)}
                className="bg-black bg-opacity-40 rounded px-2 py-1 text-white"
              >
                {moods.map((mood) => (
                  <option key={mood} value={mood}>
                    {mood}
                  </option>
                ))}
              </select>
            </div>
            <button
              onClick={() => {
                setShowSongsSection(true);
                fetchSongs();
              }}
              className="px-4 py-2 bg-black bg-opacity-40 rounded hover:bg-opacity-60"
            >
              Add Playlist
            </button>
            <button
              onClick={handleGenerateRecommendations}
              className="px-4 py-2 bg-blue-500 rounded hover:bg-blue-400"
            >
              Get Recommendations
            </button>
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <div className="flex-1 mt-16 overflow-y-scroll p-6 space-y-6">
        {/* Playlists Section */}
        <div className="bg-gradient-to-b from-gray-900 to-black p-6 rounded-lg shadow-lg">
          <h2 className="text-xl font-semibold">Your Playlists</h2>
          <ul className="space-y-4">
            {playlists.map((playlist) => (
              <li
                key={playlist.playlist_id}
                className="flex justify-between items-center bg-gray-800 p-4 rounded-lg"
              >
                <span>{playlist.name}</span>
                <div className="space-x-2">
                  <button
                    onClick={() => handleEditPlaylist(playlist)}
                    className="px-3 py-1 bg-blue-500 rounded hover:bg-blue-400"
                  >
                    Edit
                  </button>
                  <button
                    onClick={() => handleDeletePlaylist(playlist.playlist_id)}
                    className="px-3 py-1 bg-red-500 rounded hover:bg-red-400"
                  >
                    Delete
                  </button>
                </div>
              </li>
            ))}
          </ul>
        </div>

        {/* Songs Section */}
        {showSongsSection && (
          <div className="bg-gradient-to-b from-gray-900 to-black p-6 rounded-lg shadow-lg">
            <h2 className="text-xl font-semibold mb-4">
              {editingPlaylist ? 'Edit Playlist' : 'Add Songs to Playlist'}
            </h2>
            <input
              type="text"
              placeholder="Playlist Name"
              value={playlistName}
              onChange={(e) => setPlaylistName(e.target.value)}
              className="w-full mb-4 px-4 py-2 rounded-lg text-black"
            />
            <div className="flex flex-wrap gap-4 items-center mb-4">
              <GenreFilter />
              <input
                type="text"
                placeholder="Search songs..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="flex-grow px-4 py-2 rounded-lg text-black"
              />
              <select
                value={searchType}
                onChange={(e) => setSearchType(e.target.value)}
                className="px-4 py-2 rounded-lg text-black"
              >
                <option value="track">Search by Track</option>
                <option value="artist">Search by Artist</option>
              </select>
              <button
                onClick={() => {
                  setCurrentPage(1);
                  fetchSongs(1);
                }}
                className="px-4 py-2 bg-blue-500 rounded hover:bg-blue-400"
              >
                Search
              </button>
            </div>

            {/* Songs Grid */}
            <div className="grid grid-cols-2 md:grid-cols-3 gap-4 mb-4">
              {loading ? (
                <div className="col-span-full text-center py-4">Loading...</div>
              ) : songs.length > 0 ? (
                songs.map((song) => (
                  <div
                    key={song.song_id}
                    className={`p-4 rounded-lg cursor-pointer ${
                      selectedSongs.includes(song.song_id)
                        ? "bg-green-500"
                        : "bg-gray-700 hover:bg-gray-600"
                    }`}
                    onClick={() => toggleSongSelection(song.song_id)}
                  >
                    <h3 className="font-bold">{song.track_name}</h3>
                    <p>{song.artist_name}</p>
                    <p className="text-sm text-gray-300">{song.track_genre}</p>
                  </div>
                ))
              ) : (
                <div className="col-span-full text-center py-4">No songs found</div>
              )}
            </div>

            {/* Pagination */}
            <div className="flex justify-center space-x-2 mb-4">
              {renderPagination()}
            </div>

            {/* Action Buttons */}
            <div className="flex justify-end space-x-4">
              <button
                onClick={() => {
                  setShowSongsSection(false);
                  setEditingPlaylist(null);
                }}
                className="px-4 py-2 bg-red-500 rounded-lg hover:bg-red-400"
              >
                Cancel
              </button>
              <button
                onClick={editingPlaylist ? handleSaveEdit : handleAddPlaylist}
                className="px-4 py-2 bg-green-500 rounded-lg hover:bg-green-400"
              >
                {editingPlaylist ? 'Save Changes' : 'Save Playlist'}
              </button>
            </div>
          </div>
        )}

        {/* Recommendations Section */}
        <div className="bg-gradient-to-b from-gray-900 to-black p-6 rounded-lg shadow-lg">
          <h2 className="text-xl font-semibold mb-4">Recommendations</h2>
          <RecommendationSourceToggle 
            useUserSongs={useUserSongs} 
            setUseUserSongs={setUseUserSongs}
          />
          <ul className="space-y-4">
            {recommendations.map((song) => (
              <li
                key={song.song_id}
                className="flex justify-between items-center bg-gray-800 p-4 rounded-lg"
              >
                <span>
                  {song.track_name} - {song.artist_name}
                </span>
              </li>
            ))}
          </ul>
        </div>
      </div>
    </div>
  );
}
