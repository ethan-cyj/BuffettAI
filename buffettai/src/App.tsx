import { useState } from 'react';
import './App.css';

function App() {
  const [messages, setMessages] = useState<{ sender: string; text: string }[]>([]);
  const [input, setInput] = useState('');

  const handleSend = async () => {
    if (!input.trim()) return;

    // Add user message to the chat
    const userMessage = { sender: 'user', text: input };
    setMessages((prev) => [...prev, userMessage]);

    // Clear input field
    setInput('');

    // Simulate bot response
    const botResponse = await getBotResponse(input);
    setMessages((prev) => [...prev, { sender: 'bot', text: botResponse }]);
  };

  const getBotResponse = async (message: string) => {
    // Placeholder for backend API call
    return `Warren Buffett: "Your message was '${message}'. Here's some wisdom: Invest in yourself!"`;
  };

  return (
    <div className="chat-container">
      <header className="chat-header">
        <h1>B u f f e t t . A I</h1>
      </header>
      <div className="chat-messages">
        {messages.map((msg, index) => (
          <div
            key={index}
            className={`message ${msg.sender === 'user' ? 'user-message' : 'bot-message'}`}
          >
            {msg.text}
          </div>
        ))}
      </div>
      <div className="chat-input">
        <input
          type="text"
          placeholder="Ask Warren Buffett something..."
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && handleSend()}
        />
        <button onClick={handleSend}>Send</button>
      </div>
    </div>
  );
}

export default App;