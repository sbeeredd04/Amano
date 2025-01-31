"use client";

import { useState, useEffect } from "react";
import { Menu, MenuItem, NavSection } from "../components/ui/navbar-menu";
import { Vortex } from "../components/ui/vortex";
import { ExpandablePlaylist } from "../components/ui/expandable-playlist";
import { useRouter } from "next/navigation";

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'https://70bnmmdc-5000.usw3.devtunnels.ms';

const SongCard = ({ song, onLike, onDislike, feedbackStatus, onAddToPlaylist }) => (
  <div className="bg-gray-800 p-4 rounded-lg shadow-lg relative">
    {/* Add to Playlist Button */}
    <button
      onClick={() => onAddToPlaylist(song)}
      className="absolute top-2 right-2 bg-green-500 hover:bg-green-600 text-white rounded-full w-8 h-8 flex items-center justify-center transition-colors"
      title="Add to Playlist"
    >
      <span className="text-xl">+</span>
    </button>

    <h3 className="text-lg font-semibold mb-2">{song.track_name}</h3>
    <p className="text-gray-400 mb-2">{song.artist_name}</p>
    <p className="text-gray-500 mb-4">{song.album_name}</p>
    
    {/* Existing feedback buttons */}
    <div className="flex justify-between mt-4">
      <button
        onClick={onLike}
        className={`px-4 py-2 rounded ${
          feedbackStatus[song.song_id] === 'liked'
            ? 'bg-green-600'
            : 'bg-gray-600 hover:bg-green-500'
        }`}
      >
        👍
      </button>
      <button
        onClick={onDislike}
        className={`px-4 py-2 rounded ${
          feedbackStatus[song.song_id] === 'disliked'
            ? 'bg-red-600'
            : 'bg-gray-600 hover:bg-red-500'
        }`}
      >
        👎
      </button>
    </div>
  </div>
);

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

const PlaylistModal = ({ isOpen, onClose, onSubmit, playlists, onCreateNew }) => {
  const [newPlaylistName, setNewPlaylistName] = useState('');
  const [selectedPlaylist, setSelectedPlaylist] = useState('');

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-gray-800 p-6 rounded-lg w-96">
        <h2 className="text-xl font-bold mb-4">Add to Playlist</h2>
        
        {playlists.length > 0 && (
          <>
            <select
              value={selectedPlaylist}
              onChange={(e) => setSelectedPlaylist(e.target.value)}
              className="w-full mb-4 p-2 bg-gray-700 rounded"
            >
              <option value="">Select a playlist</option>
              {playlists.map((playlist) => (
                <option key={playlist.playlist_id} value={playlist.playlist_id}>
                  {playlist.name}
                </option>
              ))}
            </select>
            <div className="mb-4">
              <div className="border-t border-gray-600 my-4"></div>
              <p className="text-gray-400">Or create a new playlist</p>
            </div>
          </>
        )}

        <input
          type="text"
          placeholder="New playlist name"
          value={newPlaylistName}
          onChange={(e) => setNewPlaylistName(e.target.value)}
          className="w-full mb-4 p-2 bg-gray-700 rounded"
        />

        <div className="flex justify-end space-x-2">
          <button
            onClick={onClose}
            className="px-4 py-2 bg-gray-600 rounded hover:bg-gray-500"
          >
            Cancel
          </button>
          <button
            onClick={() => {
              if (selectedPlaylist) {
                onSubmit(selectedPlaylist);
              } else if (newPlaylistName) {
                onCreateNew(newPlaylistName);
              }
            }}
            className="px-4 py-2 bg-green-600 rounded hover:bg-green-500"
            disabled={!selectedPlaylist && !newPlaylistName}
          >
            Add
          </button>
        </div>
      </div>
    </div>
  );
};

export default function RecommendationPage() {
  const router = useRouter();
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
  const [totalPages, setTotalPages] = useState(1);
  const songsPerPage = 12;
  const [editingPlaylist, setEditingPlaylist] = useState(null);
  const [searchType, setSearchType] = useState('track'); // 'track' or 'artist'
  const [genres, setGenres] = useState([]);
  const [artists, setArtists] = useState([]);
  const [currentMood, setCurrentMood] = useState("Calm");
  const moods = ["Angry", "Content", "Happy", "Delighted", "Calm", "Sleepy", "Sad", "Depressed", "Excited"];
  const [useUserSongs, setUseUserSongs] = useState(true);
  const [pendingFeedback, setPendingFeedback] = useState([]);
  const [feedbackStatus, setFeedbackStatus] = useState({});
  const [userId, setUserId] = useState(1); // Add default userId state or get from auth
  const [isPlaylistModalOpen, setIsPlaylistModalOpen] = useState(false);
  const [selectedSong, setSelectedSong] = useState(null);
  const [activeSection, setActiveSection] = useState(null);
  const [activeItem, setActiveItem] = useState(null);
  const [userName, setUserName] = useState("");

  useEffect(() => {
    const userId = sessionStorage.getItem("user_id");
    const name = sessionStorage.getItem("user_name");
    
    if (!userId) {
      console.error("User ID not found in session. Redirecting to login.");
      router.push("/");
      return;
    }

    setUserName(name || "User");

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
  }, [router]);

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

  const fetchSongs = async (page = currentPage) => {
    try {
      const offset = (page - 1) * songsPerPage;
      const response = await fetch(
        `${API_URL}/playlists/songs?limit=${songsPerPage}&offset=${offset}${
          genreFilter && genreFilter !== 'All' ? `&genre=${genreFilter}` : ''
        }${searchQuery ? `&search=${searchQuery}&search_type=${searchType}` : ''}`,
        {
          credentials: 'include',
        }
      );
      
      if (!response.ok) {
        throw new Error('Failed to fetch songs');
      }

      const data = await response.json();
      setSongs(data.songs);
      setTotalPages(Math.ceil(data.total_count / songsPerPage));
    } catch (error) {
      console.error('Error fetching songs:', error);
      setMessage('Failed to fetch songs');
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
    if (!playlistName.trim()) {
      setMessage("Please enter a playlist name");
      return;
    }

    try {
      const userId = sessionStorage.getItem("user_id");
      const endpoint = editingPlaylist ? '/playlists/edit' : '/playlists/add';
      const response = await fetch(`${API_URL}${endpoint}`, {
        method: 'POST',
        credentials: 'include',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          user_id: userId,
          name: playlistName,
          song_ids: selectedSongs,
          ...(editingPlaylist && { playlist_id: editingPlaylist.playlist_id })
        }),
      });

      if (response.ok) {
        setMessage(editingPlaylist ? "Playlist updated successfully!" : "Playlist created successfully!");
        // Refresh playlists
        fetchPlaylists();
        // Reset form
        setPlaylistName("");
        setSelectedSongs([]);
        setEditingPlaylist(null);
      } else {
        const data = await response.json();
        setMessage(data.error || "Failed to save playlist");
      }
    } catch (error) {
      console.error("Error saving playlist:", error);
      setMessage("Error saving playlist");
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

  const handleEditPlaylist = (playlist) => {
    setEditingPlaylist(playlist);
    // Pre-select the songs that are in the playlist
    setSelectedSongs(playlist.songs.map(song => song.song_id));
    setPlaylistName(playlist.name);
    // Scroll to songs section
    document.getElementById('songs').scrollIntoView({ behavior: 'smooth' });
  };

  const handleCancelEdit = () => {
    setEditingPlaylist(null);
    setPlaylistName("");
    setSelectedSongs([]);
  };

  const Pagination = () => {
    const pageNumbers = [];
    let startPage = Math.max(1, currentPage - 2);
    let endPage = Math.min(totalPages, currentPage + 2);

    // Always show first page
    if (startPage > 1) {
      pageNumbers.push(1);
      if (startPage > 2) pageNumbers.push('...');
    }

    // Add pages around current page
    for (let i = startPage; i <= endPage; i++) {
      pageNumbers.push(i);
    }

    // Always show last page
    if (endPage < totalPages) {
      if (endPage < totalPages - 1) pageNumbers.push('...');
      pageNumbers.push(totalPages);
    }

    return (
      <div className="flex justify-center gap-2 mt-4">
        {pageNumbers.map((number, index) => (
          number === '...' ? (
            <span key={`ellipsis-${index}`} className="px-3 py-2">...</span>
          ) : (
            <button
              key={number}
              onClick={() => {
                setCurrentPage(number);
                fetchSongs(number);
              }}
              className={`px-3 py-2 rounded ${
                currentPage === number
                  ? 'bg-white text-black'
                  : 'border border-white/[0.2] hover:bg-white/[0.1]'
              }`}
            >
              {number}
            </button>
          )
        ))}
      </div>
    );
  };

  const handleMoodChange = async (newMood) => {
    console.log(`=== Changing Mood to ${newMood} ===`);
    const userId = sessionStorage.getItem("user_id");
    
    if (!userId) {
      console.error("No user ID found in session");
      setMessage("Please log in to update mood");
      return;
    }

    try {
      console.log("Sending mood update request...");
      const response = await fetch(`${API_URL}/recommendation/mood`, {
        method: 'POST',
        credentials: 'include',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          user_id: userId,
          mood: newMood
        })
      });

      const data = await response.json();
      
      if (response.ok) {
        console.log("Mood updated successfully:", data);
        setCurrentMood(newMood);
        setMessage("Mood updated successfully!");
        // Fetch new recommendations with updated mood
        await handleGenerateRecommendations();
      } else {
        console.error("Failed to update mood:", data.error);
        setMessage(`Failed to update mood: ${data.error}`);
      }
    } catch (error) {
      console.error("Error updating mood:", error);
      setMessage("Error updating mood. Please try again.");
    }
  };

  useEffect(() => {
    const fetchCurrentMood = async () => {
      console.log("Fetching mood for user...");
      try {
        const userId = sessionStorage.getItem("user_id");
        const response = await fetch(
          `${API_URL}/recommendation/mood?user_id=${userId}`,
          {
            credentials: 'include',
            headers: {
              'Content-Type': 'application/json'
            }
          }
        );
        if (response.ok) {
          const data = await response.json();
          console.log("Current mood fetched:", data.mood);
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

  // Handle feedback submission
  const handleFeedback = async (songId, isLiked) => {
    console.log(`Processing feedback for song ${songId}: ${isLiked ? 'liked' : 'disliked'}`);
    try {
      setMessage("");
      const userId = sessionStorage.getItem("user_id");
      
      // Update feedback status
      setFeedbackStatus(prev => ({
        ...prev,
        [songId]: isLiked ? 'liked' : 'disliked'
      }));

      // Add to pending feedback
      setPendingFeedback(prev => [...prev, {
        user_id: userId,
        song_id: songId,
        is_liked: isLiked,
        mood: currentMood
      }]);

      console.log("Feedback added to pending list");
    } catch (error) {
      console.error("Error handling feedback:", error);
      setMessage("Failed to record feedback");
    }
  };

  // Submit pending feedback and refresh recommendations
  const submitFeedbackAndRefresh = async () => {
    console.log("Submitting feedback and refreshing recommendations...");
    try {
      // Submit all pending feedback
      for (const feedback of pendingFeedback) {
        const response = await fetch(`${API_URL}/recommendation/feedback`, {
          method: 'POST',
          credentials: 'include',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(feedback)
        });

        if (!response.ok) {
          throw new Error(`Failed to submit feedback: ${response.statusText}`);
        }
        console.log(`Feedback submitted for song ${feedback.song_id}`);
      }

      // Clear pending feedback
      setPendingFeedback([]);
      
      // Fetch new recommendations
      await fetchRecommendations();
      console.log("Recommendations refreshed after feedback");

    } catch (error) {
      console.error("Error in submitFeedbackAndRefresh:", error);
      setMessage("Failed to update recommendations");
    }
  };

  // Fetch recommendations
  const fetchRecommendations = async () => {
    console.log("Fetching recommendations...");
    try {
      const userId = sessionStorage.getItem("user_id");
      const response = await fetch(
        `${API_URL}/recommendation/initial?user_id=${userId}&use_user_songs=${useUserSongs}`,
        {
          credentials: 'include',
          headers: {
            'Content-Type': 'application/json'
          }
        }
      );
      
      if (response.ok) {
        const data = await response.json();
        console.log("Recommendations received:", data.recommendations.length);
        setRecommendations(data.recommendations);
        setFeedbackStatus({}); // Reset feedback status for new recommendations
      }
    } catch (error) {
      console.error("Error fetching recommendations:", error);
      setMessage("Failed to load recommendations");
    }
  };

  // Render recommendations section
  const renderRecommendations = () => (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
      {recommendations.map((song) => (
        <SongCard
          key={song.song_id}
          song={song}
          onLike={() => handleFeedback(song.song_id, true)}
          onDislike={() => handleFeedback(song.song_id, false)}
          feedbackStatus={feedbackStatus}
          onAddToPlaylist={(song) => {
            setSelectedSong(song);
            setIsPlaylistModalOpen(true);
          }}
        />
      ))}
    </div>
  );

  const handleAddToPlaylist = (song) => {
    setSelectedSong(song);
    setIsPlaylistModalOpen(true);
  };

  const handleAddToExistingPlaylist = async (playlistId) => {
    try {
      const userId = sessionStorage.getItem("user_id");
      const response = await fetch(`${API_URL}/playlists/edit`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          user_id: userId,
          playlist_id: playlistId,
          song_ids: [selectedSong.song_id]
        })
      });

      if (response.ok) {
        setMessage("Song added to playlist successfully!");
      } else {
        setMessage("Failed to add song to playlist");
      }
    } catch (error) {
      console.error("Error adding to playlist:", error);
      setMessage("Error adding song to playlist");
    } finally {
      setIsPlaylistModalOpen(false);
      setSelectedSong(null);
    }
  };

  const handleCreateNewPlaylist = async (playlistName) => {
    try {
      const userId = sessionStorage.getItem("user_id");
      const response = await fetch(`${API_URL}/playlists/add`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          user_id: userId,
          name: playlistName,
          song_ids: [selectedSong.song_id]
        })
      });

      if (response.ok) {
        const data = await response.json();
        setPlaylists([...playlists, data.playlist]);
        setMessage("New playlist created and song added successfully!");
      } else {
        setMessage("Failed to create playlist");
      }
    } catch (error) {
      console.error("Error creating playlist:", error);
      setMessage("Error creating playlist");
    } finally {
      setIsPlaylistModalOpen(false);
      setSelectedSong(null);
    }
  };

  // Add this function to handle song removal
  const handleRemoveSong = async (playlistId, songId) => {
    try {
      const userId = sessionStorage.getItem("user_id");
      const response = await fetch(`${API_URL}/playlists/remove_song`, {
        method: 'POST',
        credentials: 'include',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          user_id: userId,
          playlist_id: playlistId,
          song_id: songId
        }),
      });

      if (response.ok) {
        // Update the playlists state to reflect the change
        setPlaylists(playlists.map(playlist => {
          if (playlist.playlist_id === playlistId) {
            return {
              ...playlist,
              songs: playlist.songs.filter(song => song.song_id !== songId)
            };
          }
          return playlist;
        }));
        setMessage("Song removed from playlist");
      } else {
        const data = await response.json();
        setMessage(data.error || "Failed to remove song");
      }
    } catch (error) {
      console.error("Error removing song:", error);
      setMessage("Error removing song from playlist");
    }
  };

  const handleLogout = () => {
    // Clear session storage
    sessionStorage.clear();
    // Redirect to home page
    router.push("/");
  };

  return (
    <div className="relative min-h-screen font-ubuntu-mono">
      <Vortex
        particleCount={800}           // More particles for denser effect
        rangeY={250}                   // Larger vertical movement
        baseSpeed={0.01}                // Faster base movement
        rangeSpeed={0.5}               // More speed variation
        baseRadius={1.25}               // Larger particles
        rangeRadius={3}                // More size variation
        baseHue={200}                  // Different base color (180 = cyan)
        backgroundColor="rgba(0, 0, 0, 0.9)"  // Slightly more opaque background
        containerClassName="fixed inset-0 w-full h-full"  // Fill entire viewport
      />

      <div className="relative z-10">
        <div className="relative z-50">
          <Menu setActive={setActiveItem}>
            <div className="flex items-center space-x-6">
              <MenuItem 
                setActive={setActiveItem}
                active={activeItem}
                item="Home"
              >
                <NavSection
                  title="Home"
                  description="Return to the main dashboard"
                  href="#home"
                />
              </MenuItem>

              <MenuItem
                setActive={setActiveItem}
                active={activeItem}
                item="Songs"
              >
                <NavSection
                  title="Songs"
                  description="Browse and manage your music collection"
                  href="#songs"
                />
              </MenuItem>

              <MenuItem
                setActive={setActiveItem}
                active={activeItem}
                item="Playlists"
              >
                <NavSection
                  title="Playlists"
                  description="Manage your custom playlists"
                  href="#playlists"
                />
              </MenuItem>

              <MenuItem
                setActive={setActiveItem}
                active={activeItem}
                item="Recommendations"
              >
                <NavSection
                  title="Recommendations"
                  description="Get personalized music suggestions"
                  href="#recommendations"
                />
              </MenuItem>
            </div>

            <div className="flex-grow" />

            <div className="flex items-center space-x-6">
              <MenuItem
                setActive={setActiveItem}
                active={activeItem}
                item={`Hi, ${userName}`}
              >
                <NavSection
                  title="Profile"
                  description="View your profile settings"
                  href="#profile"
                />
              </MenuItem>

              <MenuItem
                setActive={setActiveItem}
                active={activeItem}
                item="Logout"
                className="border border-red-500 bg-red-500/40 hover:bg-red-500/60 rounded-full px-4 py-1 transition-colors"
              >
                <div 
                  onClick={handleLogout}
                  className="cursor-pointer px-4 py-2"
                >
                  <h4 className="text-red-500 font-bold mb-1">Logout</h4>
                  <p className="text-red-300 text-sm">Sign out of your account</p>
                </div>
              </MenuItem>
            </div>
          </Menu>
        </div>

        <div className="pt-16">
          <section id="home" className="min-h-screen p-6 flex items-center justify-center bg-transparent">
            <div className="text-center">
              <h1 className="text-9xl font-bold tracking-wider mb-4">
                AMANO
              </h1>
              <p className="text-2xl text-gray-400">Your Personal Music Companion</p>
            </div>
          </section>

          <section id="songs" className="min-h-screen p-6 bg-transparent">
            <h2 className="text-4xl font-bold text-center mb-12">
              {editingPlaylist ? `Edit Playlist: ${editingPlaylist.name}` : "Discover Songs"}
            </h2>
            <div className="max-w-6xl mx-auto">
              <div className="mb-8">
                <input
                  type="text"
                  value={playlistName}
                  onChange={(e) => setPlaylistName(e.target.value)}
                  placeholder="Enter playlist name"
                  className="bg-black/40 backdrop-blur-sm border border-white/10 rounded-lg px-4 py-2 w-full mb-4"
                />
                {editingPlaylist && (
                  <div className="flex justify-between">
                    <button
                      onClick={handleAddPlaylist}
                      className="px-4 py-2 bg-green-500 hover:bg-green-600 text-white rounded-lg transition-colors"
                    >
                      Update Playlist
                    </button>
                    <button
                      onClick={handleCancelEdit}
                      className="px-4 py-2 bg-gray-500 hover:bg-gray-600 text-white rounded-lg transition-colors"
                    >
                      Cancel Edit
                    </button>
                  </div>
                )}
              </div>

              <div className="flex flex-wrap gap-4 items-center mb-4">
                <GenreFilter />
                <input
                  type="text"
                  placeholder="Search songs..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="flex-grow px-4 py-2 bg-transparent border border-white/[0.2] rounded-lg"
                />
                <select
                  value={searchType}
                  onChange={(e) => setSearchType(e.target.value)}
                  className="px-4 py-2 bg-transparent border border-white/[0.2] rounded-lg"
                >
                  <option value="track">Search by Track</option>
                  <option value="artist">Search by Artist</option>
                </select>
                <button 
                  onClick={() => { 
                    setCurrentPage(1); 
                    fetchSongs(1); 
                  }}
                  className="px-4 py-2 bg-white text-black rounded hover:bg-gray-200 transition-colors"
                >
                  Search
                </button>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {songs.map((song) => (
                  <div
                    key={song.song_id}
                    className={`bg-black/40 backdrop-blur-sm border ${
                      selectedSongs.includes(song.song_id)
                        ? 'border-green-500'
                        : 'border-white/10'
                    } rounded-lg p-4 cursor-pointer transition-all`}
                    onClick={() => toggleSongSelection(song.song_id)}
                  >
                    <h3 className="font-semibold text-white">{song.track_name}</h3>
                    <p className="text-sm text-gray-300">{song.artist_name}</p>
                    <p className="text-xs text-gray-400">{song.track_genre}</p>
                  </div>
                ))}
              </div>

              <Pagination />

              <div className="flex justify-between mt-4">
                <button
                  onClick={() => setSelectedSongs([])}
                  className="px-4 py-2 border border-red-500 text-red-500 rounded hover:bg-red-500 hover:text-black"
                >
                  Clear Selection
                </button>
                {!editingPlaylist && (
                  <button
                    onClick={handleAddPlaylist}
                    className="px-4 py-2 border border-green-500 text-green-500 rounded hover:bg-green-500 hover:text-black"
                  >
                    Add Playlist
                  </button>
                )}
              </div>
            </div>
          </section>

          <section id="playlists" className="min-h-screen p-6 bg-transparent">
            <h2 className="text-4xl font-bold text-center mb-12">Your Playlists</h2>
            <div className="max-w-6xl mx-auto">
              <ExpandablePlaylist
                playlists={playlists}
                onEdit={handleEditPlaylist}
                onDelete={handleDeletePlaylist}
                onRemoveSong={handleRemoveSong}
              />
            </div>
          </section>

          <section id="recommendations" className="min-h-screen p-6 bg-transparent">
            <h2 className="text-4xl font-bold text-center mb-12">Your Recommendations</h2>
            
            <div className="max-w-6xl mx-auto mb-8 flex justify-center gap-4">
              <div className="flex items-center space-x-4">
                <label className="text-sm font-medium">Current Mood:</label>
                <select
                  value={currentMood}
                  onChange={(e) => setCurrentMood(e.target.value)}
                  className="bg-gray-800 text-white rounded px-3 py-1"
                >
                  {moods.map((mood) => (
                    <option key={mood} value={mood}>
                      {mood}
                    </option>
                  ))}
                </select>
              </div>

              <RecommendationSourceToggle 
                useUserSongs={useUserSongs} 
                setUseUserSongs={setUseUserSongs} 
              />
            </div>

            <div className="flex justify-center mb-8">
              <button
                onClick={handleGenerateRecommendations}
                className="bg-green-600 hover:bg-green-700 text-white px-6 py-2 rounded-full transition-colors"
              >
                Generate Recommendations
              </button>
            </div>

            <div className="max-w-6xl mx-auto grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {recommendations.map((song) => (
                <SongCard
                  key={song.song_id}
                  song={song}
                  onLike={() => handleFeedback(song.song_id, true)}
                  onDislike={() => handleFeedback(song.song_id, false)}
                  feedbackStatus={feedbackStatus}
                  onAddToPlaylist={(song) => {
                    setSelectedSong(song);
                    setIsPlaylistModalOpen(true);
                  }}
                />
              ))}
            </div>
          </section>
        </div>
      </div>

      <PlaylistModal
        isOpen={isPlaylistModalOpen}
        onClose={() => {
          setIsPlaylistModalOpen(false);
          setSelectedSong(null);
        }}
        onSubmit={handleAddToExistingPlaylist}
        onCreateNew={handleCreateNewPlaylist}
        playlists={playlists}
      />
    </div>
  );
}
