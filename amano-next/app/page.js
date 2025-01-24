"use client";

import { useRouter } from "next/navigation";

export default function HomePage() {
  const router = useRouter();

  const goToSignIn = () => {
    router.push("/signin");
  };

  return (
    <div className="h-screen flex flex-col items-center justify-center bg-gradient-to-b from-black via-gray-900 to-green-900 text-white">
      <h1 className="text-5xl font-extrabold mb-4">Welcome to Amano</h1>
      <p className="text-lg text-gray-300 mb-8">
        Discover music tailored to your mood and preferences.
      </p>
      <button
        onClick={goToSignIn}
        className="px-6 py-3 bg-green-500 text-black rounded-lg font-medium hover:bg-green-400 transition duration-300"
      >
        Sign In
      </button>
    </div>
  );
}
