import React, { useEffect, useState } from 'react';
import axios from 'axios';

interface TokenConfig {
  _id?: string;
  serviceName: string;
  tokenType: string;
  tokenValue: string;
  note?: string;
  expiresAt?: string;
  createdAt?: string;
}

const Dashboard = () => {
  const [tokens, setTokens] = useState<TokenConfig[]>([]);
  const [form, setForm] = useState<TokenConfig>({
    serviceName: '',
    tokenType: '',
    tokenValue: '',
    note: '',
    expiresAt: '',
  });

  const fetchTokens = async () => {
    const res = await axios.get('/api/token-config');
    setTokens(res.data);
  };

  useEffect(() => {
    fetchTokens();
  }, []);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
    setForm({ ...form, [e.target.name]: e.target.value });
  };

  const handleSubmit = async () => {
    await axios.post('/api/token-config', form);
    fetchTokens();
    setForm({ serviceName: '', tokenType: '', tokenValue: '', note: '', expiresAt: '' });
  };

  return (
    <div className="p-6 max-w-4xl mx-auto">
      <h1 className="text-2xl font-bold mb-4">Token Configuration</h1>

      <div className="bg-white shadow p-4 rounded-lg mb-6">
        <input
          type="text"
          name="serviceName"
          placeholder="Service Name"
          className="input mb-2"
          value={form.serviceName}
          onChange={handleChange}
        />
        <input
          type="text"
          name="tokenType"
          placeholder="Token Type"
          className="input mb-2"
          value={form.tokenType}
          onChange={handleChange}
        />
        <input
          type="text"
          name="tokenValue"
          placeholder="Token Value"
          className="input mb-2"
          value={form.tokenValue}
          onChange={handleChange}
        />
        <textarea
          name="note"
          placeholder="Note"
          className="input mb-2"
          value={form.note}
          onChange={handleChange}
        />
        <input
          type="date"
          name="expiresAt"
          className="input mb-2"
          value={form.expiresAt?.slice(0, 10) || ''}
          onChange={handleChange}
        />
        <button className="bg-blue-600 text-white px-4 py-2 rounded" onClick={handleSubmit}>
          Save Token
        </button>
      </div>

      <table className="table-auto w-full text-left border">
        <thead>
          <tr className="bg-gray-100">
            <th className="p-2">Service</th>
            <th className="p-2">Type</th>
            <th className="p-2">Token</th>
            <th className="p-2">Expires</th>
            <th className="p-2">Updated</th>
          </tr>
        </thead>
        <tbody>
          {tokens.map((token) => (
            <tr key={token._id}>
              <td className="p-2">{token.serviceName}</td>
              <td className="p-2">{token.tokenType}</td>
              <td className="p-2 truncate max-w-[200px]">{token.tokenValue}</td>
              <td className="p-2">{token.expiresAt?.slice(0, 10)}</td>
              <td className="p-2">{token.createdAt?.slice(0, 10)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

export default Dashboard;
