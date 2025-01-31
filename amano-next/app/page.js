"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import { Menu, MenuItem, NavSection } from "./components/ui/navbar-menu";
import { Vortex } from "./components/ui/vortex";

export default function HomePage() {
  const router = useRouter();
  const [activeItem, setActiveItem] = useState(null);
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [userName, setUserName] = useState("");

  useEffect(() => {
    // Check if user is logged in
    const userId = sessionStorage.getItem("user_id");
    const name = sessionStorage.getItem("user_name");
    
    if (userId) {
      setIsLoggedIn(true);
      setUserName(name || "User");
      // Redirect to recommendation page if already logged in
      router.push("/recommendation");
    }
  }, [router]);

  const goToSignIn = () => {
    router.push("/signin");
  };

  const handleLogout = () => {
    // Clear session storage
    sessionStorage.clear();
    setIsLoggedIn(false);
    setUserName("");
    // Redirect to home page
    router.push("/");
  };

  // If user is logged in, don't show the landing page
  if (isLoggedIn) {
    return null; // This prevents flash of content before redirect
  }

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
              href="#"
            />
          </MenuItem>

          {isLoggedIn ? (
            <>
              <MenuItem
                setActive={setActiveItem}
                active={activeItem}
                item={`Hi, ${userName}`}
              >
                <NavSection
                  title="Profile"
                  description="View your profile"
                  href="/recommendation"
                />
              </MenuItem>
              <MenuItem
                setActive={setActiveItem}
                active={activeItem}
                item="Logout"
              >
                <div 
                  onClick={handleLogout}
                  className="cursor-pointer px-4 py-2"
                >
                  <h4 className="text-xl font-bold mb-1 text-white">Logout</h4>
                  <p className="text-neutral-300 text-sm">Sign out of your account</p>
                </div>
              </MenuItem>
            </>
          ) : (
            <MenuItem
              setActive={setActiveItem}
              active={activeItem}
              item="Login"
            >
              <NavSection
                title="Login"
                description="Sign in to your account"
                href="/signin"
              />
            </MenuItem>
          )}
        </Menu>
      </div>

      {/* Main Content */}
      <div className="relative z-10 h-screen flex flex-col items-center justify-center">
        <h1 className="text-9xl font-bold tracking-wider mb-4 text-white">
          AMANO
        </h1>
        <p className="text-2xl text-gray-300 mb-8">
          Discover music tailored to your mood and preferences.
        </p>
        <button
          onClick={goToSignIn}
          className="px-6 py-3 bg-green-500 text-black rounded-lg font-medium hover:bg-green-400 transition duration-300"
        >
          Get Started
        </button>
      </div>
    </div>
  );
}
