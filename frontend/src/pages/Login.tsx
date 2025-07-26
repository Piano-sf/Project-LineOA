import { useState } from "react";
import { loginAdmin } from "../api";

interface LoginProps {
  setJwt: (token: string) => void;
}

export default function Login({ setJwt }: LoginProps) {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);

  const handleLogin = async () => {
    setLoading(true);
    try {
      const data = await loginAdmin(username, password);
      localStorage.setItem("jwt", data.access_token);
      setJwt(data.access_token);
    } catch (err) {
      alert("❌ Login failed! กรุณาตรวจสอบ Username/Password");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex items-center justify-center h-screen bg-gray-100">
      <div className="bg-white p-8 rounded-2xl shadow-lg w-96">
        <h2 className="text-2xl font-bold text-center mb-6 text-gray-700">🔑 Admin Login</h2>

        <input
          className="w-full px-4 py-2 border rounded-lg mb-4 focus:ring-2 focus:ring-blue-400 outline-none"
          placeholder="Username"
          value={username}
          onChange={(e) => setUsername(e.target.value)}
        />

        <input
          type="password"
          className="w-full px-4 py-2 border rounded-lg mb-4 focus:ring-2 focus:ring-blue-400 outline-none"
          placeholder="Password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
        />

        <button
          className={`w-full bg-blue-500 text-white py-2 rounded-lg transition ${
            loading ? "opacity-50 cursor-not-allowed" : "hover:bg-blue-600"
          }`}
          onClick={handleLogin}
          disabled={loading}
        >
          {loading ? "⏳ Logging in..." : "Login"}
        </button>
      </div>
    </div>
  );
}
