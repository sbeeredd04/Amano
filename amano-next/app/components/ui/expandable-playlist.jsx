"use client";
import React, { useEffect, useId, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { useOutsideClick } from "../../hooks/use-outside-click";

export const ExpandablePlaylist = ({ playlists, onEdit, onDelete, onRemoveSong }) => {
  const [active, setActive] = useState(null);
  const [slidingSongId, setSlidingSongId] = useState(null);
  const ref = useRef(null);
  const id = useId();

  useEffect(() => {
    function onKeyDown(event) {
      if (event.key === "Escape") {
        setActive(null);
      }
    }

    if (active) {
      document.body.style.overflow = "hidden";
    } else {
      document.body.style.overflow = "auto";
    }

    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [active]);

  useOutsideClick(ref, () => setActive(null));

  const handleRemoveSong = async (songId, playlistId) => {
    await onRemoveSong(playlistId, songId);
    setSlidingSongId(null);
  };

  return (
    <>
      <AnimatePresence>
        {active && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black/50 backdrop-blur-sm h-full w-full z-40"
          />
        )}
      </AnimatePresence>

      <AnimatePresence>
        {active ? (
          <div className="fixed inset-0 grid place-items-center z-50">
            <motion.button
              key={`button-${active.playlist_id}-${id}`}
              layout
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="flex absolute top-4 right-4 items-center justify-center bg-white/10 backdrop-blur-sm rounded-full h-8 w-8"
              onClick={() => setActive(null)}
            >
              <CloseIcon />
            </motion.button>

            <motion.div
              layoutId={`card-${active.playlist_id}-${id}`}
              ref={ref}
              className="w-full max-w-[600px] h-[80vh] flex flex-col bg-black/80 backdrop-blur-md rounded-3xl overflow-hidden border border-white/10"
            >
              {/* Header */}
              <motion.div
                layoutId={`header-${active.playlist_id}-${id}`}
                className="p-6 border-b border-white/10"
              >
                <motion.h3
                  layoutId={`title-${active.playlist_id}-${id}`}
                  className="text-2xl font-bold text-white"
                >
                  {active.name}
                </motion.h3>
                <p className="text-sm text-gray-400 mt-1">
                  {active.songs?.length || 0} songs
                </p>
              </motion.div>

              {/* Songs List */}
              <div className="flex-1 overflow-auto p-6">
                {active.songs && active.songs.length > 0 ? (
                  <div className="grid gap-4">
                    {active.songs.map((song, index) => (
                      <motion.div
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ 
                          opacity: 1, 
                          y: 0,
                          x: slidingSongId === song.song_id ? -80 : 0,
                          transition: { 
                            opacity: { delay: index * 0.05 },
                            x: { type: "spring", stiffness: 300, damping: 30 }
                          }
                        }}
                        key={song.song_id}
                        className="relative"
                      >
                        <div
                          className="bg-white/5 backdrop-blur-sm p-4 rounded-lg border border-white/10 hover:bg-white/10 transition-colors cursor-pointer"
                          onClick={() => setSlidingSongId(
                            slidingSongId === song.song_id ? null : song.song_id
                          )}
                        >
                          <div className="flex items-center gap-4">
                            <div className="w-8 h-8 flex items-center justify-center rounded-full bg-white/10 text-sm text-gray-400">
                              {index + 1}
                            </div>
                            <div className="flex-1">
                              <h4 className="font-semibold text-white">{song.track_name}</h4>
                              <p className="text-sm text-gray-300">{song.artist_name}</p>
                              <p className="text-xs text-gray-400 mt-1">{song.track_genre}</p>
                            </div>
                          </div>
                        </div>

                        {/* Remove Button */}
                        <motion.button
                          initial={{ opacity: 0 }}
                          animate={{ 
                            opacity: slidingSongId === song.song_id ? 1 : 0,
                            transition: { duration: 0.2 }
                          }}
                          onClick={() => handleRemoveSong(song.song_id, active.playlist_id)}
                          className="absolute right-[-70px] top-1/2 -translate-y-1/2 w-12 h-12 bg-red-500 hover:bg-red-600 rounded-full flex items-center justify-center text-white transition-colors"
                        >
                          <svg
                            xmlns="http://www.w3.org/2000/svg"
                            width="24"
                            height="24"
                            viewBox="0 0 24 24"
                            fill="none"
                            stroke="currentColor"
                            strokeWidth="2"
                            strokeLinecap="round"
                            strokeLinejoin="round"
                          >
                            <path d="M18 6L6 18" />
                            <path d="M6 6l12 12" />
                          </svg>
                        </motion.button>
                      </motion.div>
                    ))}
                  </div>
                ) : (
                  <div className="flex items-center justify-center h-full text-gray-400">
                    No songs in this playlist
                  </div>
                )}
              </div>

              {/* Actions */}
              <div className="p-6 border-t border-white/10 flex justify-between">
                <button
                  onClick={() => {
                    setSlidingSongId(null);
                    onEdit(active);
                  }}
                  className="px-4 py-2 bg-blue-500 hover:bg-blue-600 text-white rounded-lg transition-colors"
                >
                  Edit Playlist
                </button>
                <button
                  onClick={() => onDelete(active.playlist_id)}
                  className="px-4 py-2 bg-red-500 hover:bg-red-600 text-white rounded-lg transition-colors"
                >
                  Delete Playlist
                </button>
              </div>
            </motion.div>
          </div>
        ) : null}
      </AnimatePresence>

      {/* Playlist Cards List */}
      <div className="grid gap-4">
        {playlists.map((playlist) => (
          <motion.div
            layoutId={`card-${playlist.playlist_id}-${id}`}
            key={playlist.playlist_id}
            onClick={() => setActive(playlist)}
            className="bg-black/40 backdrop-blur-sm border border-white/10 rounded-xl p-6 cursor-pointer hover:bg-black/60 transition-colors"
          >
            <motion.div
              layoutId={`header-${playlist.playlist_id}-${id}`}
              className="flex justify-between items-center"
            >
              <motion.h3
                layoutId={`title-${playlist.playlist_id}-${id}`}
                className="text-xl font-semibold text-white"
              >
                {playlist.name}
              </motion.h3>
              <span className="text-sm text-gray-400">
                {playlist.songs?.length || 0} songs
              </span>
            </motion.div>
          </motion.div>
        ))}
      </div>
    </>
  );
};

const CloseIcon = () => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    width="24"
    height="24"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
    className="h-4 w-4 text-white"
  >
    <path d="M18 6L6 18" />
    <path d="M6 6l12 12" />
  </svg>
); 