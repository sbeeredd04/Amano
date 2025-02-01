"use client";

import { useState, useEffect } from "react";
import { Menu, MenuItem, NavSection } from "../components/ui/navbar-menu";
import { Vortex } from "../components/ui/vortex";
import { ExpandablePlaylist } from "../components/ui/expandable-playlist";
import { useRouter } from "next/navigation";
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faHeart as fasHeart } from '@fortawesome/free-solid-svg-icons';
import { faHeart as farHeart } from '@fortawesome/free-regular-svg-icons';
import { faThumbsDown as fasThumbsDown } from '@fortawesome/free-solid-svg-icons';
import { faThumbsDown as farThumbsDown } from '@fortawesome/free-regular-svg-icons';
import { faRotate } from '@fortawesome/free-solid-svg-icons';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'https://70bnmmdc-5000.usw3.devtunnels.ms/';

const SongCard = ({ song, onLike, onDislike, feedbackStatus, onAddToPlaylist, isUserSong }) => (
  <div className={`bg-black/60 backdrop-blur-sm p-4 rounded-lg shadow-lg relative 
    ${isUserSong ? 'border-2 border-green-500' : 'border border-white/10'}`}>
    {isUserSong && (
      <div className="absolute top-2 left-2 bg-green-500/20 text-green-500 text-xs px-2 py-1 rounded-full">
        Your Playlist
      </div>
    )}
    
    <button
      onClick={() => onAddToPlaylist(song)}
      className="absolute top-2 right-2 bg-green-400/20 hover:bg-green-400/40 text-green-400 rounded-full w-8 h-8 flex items-center justify-center transition-colors"
      title="Add to Playlist"
    >
      <span className="text-xl">+</span>
    </button>

    <h3 className="text-lg font-semibold mb-2 mt-8">{song.track_name}</h3>
    <p className="text-gray-400 mb-2">{song.artist_name}</p>
    <p className="text-gray-500 mb-4">{song.album_name}</p>
    
    <div className="flex justify-between mt-4">
      <button
        onClick={onLike}
        className={`p-2 rounded-full transition-all ${
          feedbackStatus[song.song_id] === 'liked'
            ? 'text-red-500 scale-110'
            : 'text-gray-400 hover:text-red-500 hover:scale-105'
        }`}
      >
        <FontAwesomeIcon 
          icon={feedbackStatus[song.song_id] === 'liked' ? fasHeart : farHeart}
          className="w-6 h-6"
        />
      </button>
      <button
        onClick={onDislike}
        className={`p-2 rounded-full transition-all ${
          feedbackStatus[song.song_id] === 'disliked'
            ? 'text-blue-500 scale-110'
            : 'text-gray-400 hover:text-blue-500 hover:scale-105'
        }`}
      >
        <FontAwesomeIcon 
          icon={feedbackStatus[song.song_id] === 'disliked' ? fasThumbsDown : farThumbsDown}
          className="w-6 h-6"
        />
      </button>
    </div>
  </div>
);

const RecommendationSourceToggle = ({ useUserSongs, setUseUserSongs }) => (
  <div className="flex items-center space-x-2">
    <label className="text-sm font-medium">Recommendation Source:</label>
    <select
      value={useUserSongs.toString()}
      onChange={(e) => setUseUserSongs(e.target.value === 'true')}
      className="bg-black/40 text-white rounded px-3 py-1 border border-white/10 min-w-[140px]"
    >
      <option value="true">My Playlists</option>
      <option value="false">Default Songs</option>
    </select>
  </div>
);

const PlaylistModal = ({ isOpen, onClose, onSubmit, playlists, onCreateNew }) => {
  const [newPlaylistName, setNewPlaylistName] = useState('');
  const [selectedPlaylist, setSelectedPlaylist] = useState('');

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-50">
      <div className="bg-black/60 border border-white/10 p-6 rounded-lg w-96 backdrop-blur-sm">
        <h2 className="text-xl font-bold mb-6">Add to Playlist</h2>
        
        {playlists.length > 0 && (
          <>
            <select
              value={selectedPlaylist}
              onChange={(e) => setSelectedPlaylist(e.target.value)}
              className="w-full mb-4 p-2 bg-black/40 border border-white/10 rounded text-white"
            >
              <option value="">Select a playlist</option>
              {playlists.map((playlist) => (
                <option key={playlist.playlist_id} value={playlist.playlist_id}>
                  {playlist.name}
                </option>
              ))}
            </select>
            <div className="mb-4">
              <div className="border-t border-white/10 my-4"></div>
              <p className="text-gray-400">Or create a new playlist</p>
            </div>
          </>
        )}

        <input
          type="text"
          placeholder="New playlist name"
          value={newPlaylistName}
          onChange={(e) => setNewPlaylistName(e.target.value)}
          className="w-full mb-6 p-2 bg-black/40 border border-white/10 rounded text-white"
        />

        <div className="flex justify-end space-x-3">
          <button
            onClick={onClose}
            className="px-4 py-2 bg-white/10 hover:bg-white/20 rounded transition-colors"
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
            className="px-4 py-2 bg-green-400/20 hover:bg-green-400/40 text-green-400 rounded transition-colors"
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
  const [allSongs, setAllSongs] = useState([]); // Store all songs
  const [displayedSongs, setDisplayedSongs] = useState([]);
  const [totalSongs, setTotalSongs] = useState(0);
  const [loading, setLoading] = useState(false);
  const [currentPage, setCurrentPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const songsPerPage = 20;
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
  const [songs, setSongs] = useState([]);

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
        const response = await fetch(`${API_URL}/playlists/get`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            user_id: userId
          })
        });

        if (!response.ok) {
          throw new Error('Failed to fetch playlists');
        }

        const data = await response.json();
        setPlaylists(data.playlists || []);
      } catch (error) {
        console.error('Error fetching playlists:', error);
        setMessage('Failed to fetch playlists');
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

    const fetchData = async () => {
      try {
        // Only fetch playlists and mood
        await Promise.all([
          fetchPlaylists(),
          fetchMood()
        ]);
        
        // Log initial state
        console.debug("Initial data loaded");
        console.debug(`Current mood: ${currentMood}`);
        console.debug(`Use user songs: ${useUserSongs}`);
        
      } catch (error) {
        console.error("Error fetching initial data:", error);
        setMessage(`Error: ${error.message}`);
      }
    };

    fetchData();
  }, [router]);

  // Add a separate useEffect to trigger recommendations when needed
  useEffect(() => {
    // Only generate recommendations if we have a mood and userId
    if (currentMood && userId) {
      handleInitialRecommendations();
    }
  }, [currentMood, userId]);

  const handleInitialRecommendations = async () => {
    try {
      setMessage("Getting recommendations...");
      
      const response = await fetch(`${API_URL}/recommendation/recs`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          user_id: userId,
          mood: currentMood,
        }),
      });

      const data = await response.json();
      if (response.ok) {
        setRecommendations({
          new_songs: data.recommendations?.recommendation_pool || [],
          user_songs: data.recommendations?.user_songs || [],
          has_dqn_model: data.recommendations?.has_dqn_model || false,
          source: data.recommendations?.source || 'clustering'
        });
        setMessage("");
      } else {
        throw new Error(data.error || 'Failed to get recommendations');
      }
    } catch (error) {
      console.error("Error getting recommendations:", error);
      setMessage(`Error: ${error.message}`);
    }
  };

  const handleRefreshRecommendations = async () => {
    try {
      setMessage("Refreshing recommendations...");
      
      const response = await fetch(`${API_URL}/recommendation/refresh`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          user_id: userId,
          mood: currentMood,
          use_user_songs: useUserSongs,
          previous_recommendations: recommendations.new_songs || [],
          refresh_type: 'smart'
        }),
      });

      const data = await response.json();
      if (response.ok) {
        setRecommendations({
          new_songs: data.recommendations.new_songs,
          user_songs: data.recommendations.user_songs
        });
        setMessage(`Refreshed recommendations using ${data.source}`);
      } else {
        setMessage(`Error: ${data.error}`);
      }
    } catch (error) {
      setMessage("Error refreshing recommendations");
      console.error(error);
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
      setLoading(true);
      const offset = (page - 1) * songsPerPage;
      const response = await fetch(
        `${API_URL}/playlists/songs?limit=${songsPerPage}&offset=${offset}${
          genreFilter && genreFilter !== 'All' ? `&genre=${genreFilter}` : ''
        }${searchQuery ? `&search=${searchQuery}&type=${searchType}` : ''}`
      );

      if (!response.ok) {
        throw new Error('Failed to fetch songs');
      }

      const data = await response.json();
      setSongs(data.songs);
      setTotalPages(Math.ceil(data.total_count / songsPerPage));
      setCurrentPage(page);
      setLoading(false);
    } catch (error) {
      console.error('Error fetching songs:', error);
      setMessage('Failed to fetch songs');
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
    if (!playlistName.trim() || selectedSongs.length === 0) {
      setMessage("Please enter a playlist name and select songs");
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
        
        // Reset form
        setPlaylistName("");
        setSelectedSongs([]);
        setEditingPlaylist(null);

        // Refresh the page and then fetch playlists
        window.location.reload();
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
      <div className="flex justify-center gap-2 mt-8">
        <button
          onClick={() => fetchSongs(currentPage - 1)}
          disabled={currentPage === 1 || loading}
          className="px-4 py-2 bg-black/40 border border-white/10 rounded disabled:opacity-50"
        >
          Previous
        </button>
        
        {pageNumbers.map((number, index) => (
          number === '...' ? (
            <span key={`ellipsis-${index}`} className="px-4 py-2">...</span>
          ) : (
            <button
              key={number}
              onClick={() => fetchSongs(number)}
              disabled={loading}
              className={`px-4 py-2 rounded ${
                currentPage === number
                  ? 'bg-white text-black'
                  : 'bg-black/40 border border-white/10 hover:bg-white/10'
              }`}
            >
              {number}
            </button>
          )
        ))}
        
        <button
          onClick={() => fetchSongs(currentPage + 1)}
          disabled={currentPage === totalPages || loading}
          className="px-4 py-2 bg-black/40 border border-white/10 rounded disabled:opacity-50"
        >
          Next
        </button>
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
        await handleInitialRecommendations();
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
    fetchSongs();
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
          fetchSongs(); // Reset to first page when changing genre
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
  const handleFeedback = async (songId, isLike) => {
    try {
      // Toggle feedback if already exists
      if (feedbackStatus[songId] === (isLike ? 'liked' : 'disliked')) {
        const newFeedbackStatus = { ...feedbackStatus };
        delete newFeedbackStatus[songId];
        setFeedbackStatus(newFeedbackStatus);
        return;
      }

      // Set new feedback
      setFeedbackStatus({
        ...feedbackStatus,
        [songId]: isLike ? 'liked' : 'disliked'
      });

      const response = await fetch(`${API_URL}/recommendation/feedback`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          user_id: userId,
          song_id: songId,
          is_liked: isLike,
          mood: currentMood
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to record feedback');
      }

      const data = await response.json();
      if (data.training_triggered) {
        setMessage("Feedback recorded and model updated!");
      } else {
        setMessage(`Feedback recorded! ${5 - data.feedback_count} more needed for model training.`);
      }
    } catch (error) {
      console.error('Error recording feedback:', error);
      setMessage('Failed to record feedback');
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
          isUserSong={false}
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

  // Add this function at the top of your component
  const scrollToSection = (sectionId) => {
    const section = document.getElementById(sectionId);
    if (section) {
      section.scrollIntoView({ 
        behavior: 'smooth',
        block: 'start'
      });
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

  const handleLogout = async () => {
    try {
      // Clear all session data
      sessionStorage.clear();
      
      // Show a brief success message
      setMessage("Logging out...");
      
      // Small delay to show the message
      await new Promise(resolve => setTimeout(resolve, 500));
      
      // Redirect to home page
      router.push("/");
      
    } catch (error) {
      console.error("Error during logout:", error);
      setMessage("Error logging out. Please try again.");
    }
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
                <div 
                  onClick={() => scrollToSection('home')}
                  className="cursor-pointer hover:bg-white/10 px-4 py-2 rounded-lg transition-colors"
                >
                  <h4 className="text-xl font-bold mb-1 text-white">Home</h4>
                  <p className="text-neutral-300 text-sm">Return to the main dashboard</p>
                </div>
              </MenuItem>

              <MenuItem
                setActive={setActiveItem}
                active={activeItem}
                item="Songs"
              >
                <div 
                  onClick={() => scrollToSection('songs')}
                  className="cursor-pointer hover:bg-white/10 px-4 py-2 rounded-lg transition-colors"
                >
                  <h4 className="text-xl font-bold mb-1 text-white">Songs</h4>
                  <p className="text-neutral-300 text-sm">Browse and manage your music</p>
                </div>
              </MenuItem>

              <MenuItem
                setActive={setActiveItem}
                active={activeItem}
                item="Playlists"
              >
                <div 
                  onClick={() => scrollToSection('playlists')}
                  className="cursor-pointer hover:bg-white/10 px-4 py-2 rounded-lg transition-colors"
                >
                  <h4 className="text-xl font-bold mb-1 text-white">Playlists</h4>
                  <p className="text-neutral-300 text-sm">Manage your custom playlists</p>
                </div>
              </MenuItem>

              <MenuItem
                setActive={setActiveItem}
                active={activeItem}
                item="Recommendations"
              >
                <div 
                  onClick={() => scrollToSection('recommendations')}
                  className="cursor-pointer hover:bg-white/10 px-4 py-2 rounded-lg transition-colors"
                >
                  <h4 className="text-xl font-bold mb-1 text-white">Recommendations</h4>
                  <p className="text-neutral-300 text-sm">Get personalized suggestions</p>
                </div>
              </MenuItem>
            </div>

            <div className="flex-grow" />

            <div className="flex items-center space-x-6">
              <MenuItem
                setActive={setActiveItem}
                active={activeItem}
                item={`Hi, ${userName}`}
              >
                <div 
                  onClick={() => scrollToSection('profile')}
                  className="cursor-pointer hover:bg-white/10 px-4 py-2 rounded-lg transition-colors"
                >
                  <h4 className="text-xl font-bold mb-1 text-white">Profile</h4>
                  <p className="text-neutral-300 text-sm">View your profile settings</p>
                </div>
              </MenuItem>

              <MenuItem
                setActive={setActiveItem}
                active={activeItem}
                item="Logout"
                className="border border-red-500 bg-red-500/40 hover:bg-red-500/60 rounded-full px-4 py-1 transition-colors"
              >
                <div 
                  onClick={handleLogout}
                  className="cursor-pointer px-4 py-2 hover:bg-red-500/20 rounded-lg transition-colors"
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

          <section id="songs" className="min-h-screen p-6 pt-32 bg-transparent">
            <h2 className="text-4xl font-bold text-center mb-12">
              {editingPlaylist ? `Edit Playlist: ${editingPlaylist.name}` : "Discover Songs"}
            </h2>
            <div className="max-w-[90rem] mx-auto">
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
                    fetchSongs(); 
                  }}
                  className="px-4 py-2 bg-white text-black rounded hover:bg-gray-200 transition-colors"
                >
                  Search
                </button>
              </div>

              <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-4">
                {loading ? (
                  <div className="col-span-full text-center py-8">Loading songs...</div>
                ) : (
                  songs.map((song) => (
                    <div
                      key={song.song_id}
                      className={`bg-black/40 backdrop-blur-sm border ${
                        selectedSongs.includes(song.song_id)
                          ? 'border-green-500'
                          : 'border-white/10'
                      } rounded-lg p-4 cursor-pointer transition-all hover:border-white/30 flex flex-col`}
                      onClick={() => toggleSongSelection(song.song_id)}
                    >
                      {/* Track Name */}
                      <h3 className="font-semibold text-white text-lg truncate mb-1">
                        {song.track_name}
                      </h3>
                      
                      {/* Artist Name */}
                      <div className="flex items-center text-gray-300 mb-2">
                        <span className="text-sm truncate">
                          {song.artist_name}
                        </span>
                      </div>
                      
                      {/* Album Name */}
                      <div className="text-gray-400 text-sm mb-2 truncate">
                        {song.album_name}
                      </div>
                      
                      {/* Genre */}
                      <div className="mt-auto">
                        <span className="text-xs text-gray-400 px-2 py-1 rounded-full bg-white/5">
                          {song.track_genre}
                        </span>
                      </div>
                    </div>
                  ))
                )}
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

          <section id="playlists" className="min-h-screen p-6 pt-32 bg-transparent">
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

          <section id="recommendations" className="min-h-screen p-6 pt-32 bg-transparent">
            <h2 className="text-4xl font-bold text-center mb-12">Your Recommendations</h2>
            
            <div className="max-w-7xl mx-auto mb-8">
              <div className="bg-black/60 backdrop-blur-sm p-4 rounded-lg flex items-center justify-between gap-4 flex-wrap border border-white/10">
                <div className="flex items-center gap-6">
                  <div className="flex items-center space-x-2">
                    <label className="text-sm font-medium">Current Mood:</label>
                    <select
                      value={currentMood}
                      onChange={(e) => setCurrentMood(e.target.value)}
                      className="bg-black/40 text-white rounded px-3 py-1 border border-white/10 min-w-[140px]"
                    >
                      {moods.map((mood) => (
                        <option key={mood} value={mood}>{mood}</option>
                      ))}
                    </select>
                  </div>

                  <RecommendationSourceToggle 
                    useUserSongs={useUserSongs} 
                    setUseUserSongs={setUseUserSongs} 
                  />
                </div>

                <button
                  onClick={handleInitialRecommendations}
                  className="bg-green-400/20 hover:bg-green-400/40 text-green-400 px-6 py-1 rounded-lg transition-colors border border-green-400/20"
                >
                  Generate Recommendations
                </button>
              </div>
            </div>

            <div className="max-w-6xl mx-auto">
              {/* New Songs Section */}
              {recommendations.new_songs?.length > 0 && (
                <div className="mb-12">
                  <div className="flex items-center gap-4 mb-6">
                    <h3 className="text-2xl font-semibold">Recommended Songs</h3>
                    <button
                      onClick={handleRefreshRecommendations}
                      className="text-gray-400 hover:text-white transition-colors"
                      title="Refresh Recommendations"
                    >
                      <FontAwesomeIcon 
                        icon={faRotate}
                        className="w-5 h-5"
                      />
                    </button>
                  </div>
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                    {recommendations.new_songs.map((song) => (
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
                        isUserSong={false}
                      />
                    ))}
                  </div>
                </div>
              )}

              {/* User Songs Section */}
              {recommendations.user_songs?.length > 0 && (
                <div>
                  <h3 className="text-2xl font-semibold mb-6">From Your Playlists</h3>
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                    {recommendations.user_songs.map((song) => (
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
                        isUserSong={true}
                      />
                    ))}
                  </div>
                </div>
              )}
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

      {/* Show message if exists */}
      {message && (
        <div className="fixed top-20 left-1/2 transform -translate-x-1/2 bg-black/80 text-white px-6 py-3 rounded-full z-50">
          {message}
        </div>
      )}
    </div>
  );
}
