"use client";

import { useState } from "react";

export default function SignInPage() {
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
          setTimeout(() => setIsSignup(false), 100); // Switch to login after 2 seconds
        } else {
          // Save user_id in session storage
          sessionStorage.setItem("user_id", data.user.user_id);
          setMessage("Login successful! Redirecting to recommendations...");
          setTimeout(() => {
            window.location.href = "/recommendation";
          }, 100); // Redirect after 2 seconds
        }
      } else {
        setMessage(`Error: ${data.error}`);
      }
    } catch (error) {
      setMessage("Error connecting to the server. Check the console for details.");
    }
  };

  return (
    <div className="h-screen flex flex-col items-center justify-center bg-gradient-to-b from-black via-gray-900 to-green-900 text-white">
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
      <h1 className="text-4xl font-bold mb-6">{isSignup ? "Sign Up" : "Login"}</h1>
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
  );
}
