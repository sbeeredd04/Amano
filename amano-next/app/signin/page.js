"use client";

import { useState } from "react";
import { Menu, MenuItem, NavSection } from "../components/ui/navbar-menu";
import { Vortex } from "../components/ui/vortex";

export default function SignInPage() {
  const [activeItem, setActiveItem] = useState(null);
  const [isSignup, setIsSignup] = useState(false); // Toggle between Signup and Login
  const [email, setEmail] = useState("");
  const [name, setName] = useState("");
  const [image, setImage] = useState("");
  const [message, setMessage] = useState("");

  const handleSubmit = async () => {
    try {
      const endpoint = isSignup
        ? "https://70bnmmdc-5000.usw3.devtunnels.ms/auth/signup"
        : "https://70bnmmdc-5000.usw3.devtunnels.ms/auth/login";

      const payload = isSignup ? { email, name, image } : { email };
      const response = await fetch(endpoint, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
      });

      const data = await response.json();
      if (response.ok) {
        if (isSignup) {
          setMessage("Signup successful! Redirecting to login...");
          setTimeout(() => setIsSignup(false), 100);
        } else {
          // Save user_id and name in session storage
          sessionStorage.setItem("user_id", data.user.user_id);
          sessionStorage.setItem("user_name", data.user.name || email.split('@')[0]);
          setMessage("Login successful! Redirecting to recommendations...");
          setTimeout(() => {
            window.location.href = "/recommendation";
          }, 100);
        }
      } else {
        setMessage(`Error: ${data.error}`);
      }
    } catch (error) {
      setMessage("Error connecting to the server. Check the console for details.");
    }
  };

  return (
    <div className="relative min-h-screen font-ubuntu-mono">
      {/* Vortex Background */}
      <Vortex
        particleCount={800}
        rangeY={250}
        baseSpeed={0.01}
        rangeSpeed={0.5}
        baseRadius={1.25}
        rangeRadius={3}
        baseHue={200}
        backgroundColor="rgba(0, 0, 0, 0.9)"
        containerClassName="fixed inset-0 w-full h-full"
      />

      {/* Navigation */}
      <div className="relative z-50">
        <Menu setActive={setActiveItem}>
          <MenuItem 
            setActive={setActiveItem}
            active={activeItem}
            item="Home"
          >
            <NavSection
              title="Home"
              description="Return to the main dashboard"
              href="/"
            />
          </MenuItem>
        </Menu>
      </div>

      {/* Main Content */}
      <div className="relative z-10 h-screen flex flex-col items-center justify-center">
        <div className="flex mb-6">
          <button
            className={`px-6 py-3 ${
              !isSignup ? "bg-green-500 text-black" : "bg-gray-700 text-gray-300"
            } font-medium rounded-l-lg transition duration-300`}
            onClick={() => setIsSignup(false)}
          >
            Login
          </button>
          <button
            className={`px-6 py-3 ${
              isSignup ? "bg-green-500 text-black" : "bg-gray-700 text-gray-300"
            } font-medium rounded-r-lg transition duration-300`}
            onClick={() => setIsSignup(true)}
          >
            Sign Up
          </button>
        </div>
        <h1 className="text-4xl font-bold mb-6 text-white">{isSignup ? "Sign Up" : "Login"}</h1>
        <div className="w-80">
          <input
            type="email"
            placeholder="Email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            className="w-full mb-4 px-4 py-2 rounded-lg text-black"
          />
          {isSignup && (
            <>
              <input
                type="text"
                placeholder="Name"
                value={name}
                onChange={(e) => setName(e.target.value)}
                className="w-full mb-4 px-4 py-2 rounded-lg text-black"
              />
              <input
                type="text"
                placeholder="Image URL (optional)"
                value={image}
                onChange={(e) => setImage(e.target.value)}
                className="w-full mb-4 px-4 py-2 rounded-lg text-black"
              />
            </>
          )}
        </div>
        <button
          onClick={handleSubmit}
          className="px-6 py-3 bg-green-500 text-black rounded-lg hover:bg-green-400 font-medium transition duration-300"
        >
          {isSignup ? "Sign Up" : "Login"}
        </button>
        {message && <p className="text-white mt-4">{message}</p>}
      </div>
    </div>
  );
}
