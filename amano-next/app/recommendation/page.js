"use client";

import { useState, useEffect } from "react";

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

  useEffect(() => {
    const userId = sessionStorage.getItem("user_id");
    if (!userId) {
      console.error("User ID not found in session. Redirecting to login.");
      window.location.href = "/";
      return;
    }

    const fetchPlaylists = async () => {
      try {
        const response = await fetch(
          "https://upgraded-funicular-jvpw9xrp6v72p444-5000.app.github.dev/playlists/",
          {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ user_id: userId }),
          }
        );
        const data = await response.json();
        console.debug("Playlists fetched:", data);
        setPlaylists(data.playlists || []);
      } catch (error) {
        console.error("Error fetching playlists:", error);
      }
    };

    const fetchMood = async () => {
      try {
        const response = await fetch(
          `https://upgraded-funicular-jvpw9xrp6v72p444-5000.app.github.dev/recommendation/mood?user_id=${userId}`
        );
        const data = await response.json();
        if (response.ok) {
          setMood(data.mood || "Calm");
          console.debug("Mood fetched successfully:", data.mood);
        }
      } catch (error) {
        console.error("Error fetching current mood:", error);
      }
    };

    fetchPlaylists();
    fetchMood();
  }, []);

  const handleGenerateRecommendations = async () => {
    const userId = sessionStorage.getItem("user_id");
    if (!userId) {
      console.error("User ID not found in session.");
      setMessage("User not logged in. Please log in again.");
      return;
    }

    try {
      const response = await fetch(
        "https://upgraded-funicular-jvpw9xrp6v72p444-5000.app.github.dev/recommendation/refresh",
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ user_id: userId, mood, feedback: [] }),
        }
      );
      const data = await response.json();
      if (response.ok) {
        setRecommendations(data.new_recommendations || []);
        console.debug("Recommendations fetched successfully:", data.new_recommendations);
      } else {
        console.error("Error generating recommendations:", data.error);
      }
    } catch (error) {
      console.error("Error generating recommendations:", error);
    }
  };

  const fetchSongs = async () => {
    setLoading(true);
    try {
      const response = await fetch(
        `https://upgraded-funicular-jvpw9xrp6v72p444-5000.app.github.dev/playlists/songs?limit=${limit}&offset=${offset}&genre=${genreFilter}&search=${searchQuery}`
      );
      const data = await response.json();
      if (data.songs) {
        setSongs((prev) => [...prev, ...data.songs]);
        setTotalSongs(data.total_songs);
        console.debug(`Fetched ${data.songs.length} songs, total: ${data.total_songs}`);
      }
    } catch (error) {
      console.error("Error fetching songs:", error);
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
        "https://upgraded-funicular-jvpw9xrp6v72p444-5000.app.github.dev/playlists/add",
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

  return (
    <div className="h-screen flex flex-col bg-black text-white">
      {/* Floating Navbar */}
      <nav className="fixed top-0 left-0 w-full bg-gradient-to-r from-green-700 via-green-600 to-green-500 text-white shadow-md z-10">
        <div className="flex justify-between items-center px-6 py-3">
          <h1 className="text-2xl font-bold">Amano</h1>
          <div className="flex space-x-4">
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
              </li>
            ))}
          </ul>
        </div>

        {/* Songs Section */}
        {showSongsSection && (
          <div className="bg-gradient-to-b from-gray-900 to-black p-6 rounded-lg shadow-lg">
            <h2 className="text-xl font-semibold mb-4">Add Songs to Playlist</h2>
            <input
              type="text"
              placeholder="Playlist Name"
              value={playlistName}
              onChange={(e) => setPlaylistName(e.target.value)}
              className="w-full mb-4 px-4 py-2 rounded-lg text-black"
            />
            <div className="flex items-center mb-4">
              <input
                type="text"
                placeholder="Search songs..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="w-full px-4 py-2 rounded-lg text-black"
              />
              <select
                value={genreFilter}
                onChange={(e) => setGenreFilter(e.target.value)}
                className="ml-4 px-4 py-2 rounded-lg text-black"
              >
                <option value="">All Genres</option>
                <option value="pop">Pop</option>
                <option value="rock">Rock</option>
                <option value="jazz">Jazz</option>
              </select>
              <button
                onClick={() => {
                  setOffset(0);
                  setSongs([]);
                  fetchSongs();
                }}
                className="ml-4 px-4 py-2 bg-blue-500 rounded hover:bg-blue-400"
              >
                Search
              </button>
            </div>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-4 overflow-y-scroll flex-grow">
              {songs.map((song) => (
                <div
                  key={song.song_id}
                  className={`p-4 rounded-lg cursor-pointer bg-gray-700 hover:bg-gray-600 ${
                    selectedSongs.includes(song.song_id) ? "bg-green-500" : ""
                  }`}
                  onClick={() => toggleSongSelection(song.song_id)}
                >
                  <h3 className="font-bold">{song.track_name}</h3>
                  <p>{song.artist_name}</p>
                  <p className="text-sm text-gray-300">{song.track_genre}</p>
                </div>
              ))}
            </div>
            {songs.length < totalSongs && !loading && (
              <button
                onClick={() => {
                  setOffset((prevOffset) => prevOffset + limit);
                  fetchSongs();
                }}
                className="mt-4 px-4 py-2 bg-blue-500 rounded-lg                 hover:bg-blue-400"
              >
                Load More
              </button>
            )}
            {loading && <p className="text-center">Loading...</p>}
            <div className="flex justify-end space-x-4 mt-4">
              <button
                onClick={() => setShowSongsSection(false)}
                className="px-4 py-2 bg-red-500 rounded-lg hover:bg-red-400"
              >
                Cancel
              </button>
              <button
                onClick={handleAddPlaylist}
                className="px-4 py-2 bg-green-500 rounded-lg hover:bg-green-400"
              >
                Save Playlist
              </button>
            </div>
          </div>
        )}

        {/* Recommendations Section */}
        <div className="bg-gradient-to-b from-gray-900 to-black p-6 rounded-lg shadow-lg">
          <h2 className="text-xl font-semibold mb-4">Recommendations</h2>
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
