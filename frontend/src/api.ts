import axios from "axios";

const API_URL = "http://localhost:5000/api"; // ← เปลี่ยนเป็น URL จริงตอน deploy

export const loginAdmin = async (username: string, password: string) => {
  const res = await axios.post(`${API_URL}/admin/login`, { username, password });
  return res.data as { access_token: string };
};

export const getToken = async (jwt: string) => {
  const res = await axios.get(`${API_URL}/admin/token`, {
    headers: { Authorization: `Bearer ${jwt}` },
  });
  return res.data as { token: string };
};

export const updateToken = async (jwt: string, newToken: string) => {
  const res = await axios.post(
    `${API_URL}/admin/token`,
    { token: newToken },
    {
      headers: { Authorization: `Bearer ${jwt}` },
    }
  );
  return res.data as { msg: string; token: string };
};
