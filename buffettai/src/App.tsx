import { useState } from 'react';
import './App.css';
import logo_img from "./assets/logo.png";

function App() {
  // State to track active tab and messages for each tab
  const [activeTab, setActiveTab] = useState('Conversation 1');
  const [messages, setMessages] = useState<{ [key: string]: { sender: string; text: string }[] }>({
    'Conversation 1': [{ sender: 'bot', text: 'Welcome! How can I help you today?' }],
  });
  const [input, setInput] = useState('');
  const [showInstructions, setShowInstructions] = useState(false);

  const generateTabName = () => {
    let tabNumber = 1; // Start with 1
    let newTab = `Conversation ${tabNumber}`; // Initial tab name
  
    // Check if the tab already exists, and increment the number if it does
    while (Object.keys(messages).includes(newTab)) {
      tabNumber += 1;
      newTab = `Conversation ${tabNumber}`; // Update the tab name with the incremented number
    }
  
    return newTab;
  };

  // Function to create a new tab
  const handleNewTab = () => {
    const newTab = generateTabName();
    setMessages((prevMessages) => ({
      ...prevMessages,
      [newTab]: [{ sender: 'bot', text: 'Welcome to a new conversation!' }],
    }));
    setActiveTab(newTab);  // Set new tab as active
  };

  // Function to handle tab switch with explicit typing for 'tab'
  const switchTab = (tab: string) => {
    setActiveTab(tab);
  };

  // Function to handle closing a tab
  const closeTab = (tab: string) => {
    if (Object.keys(messages).length === 1) return; // Prevent closing the last tab

    const newMessages = { ...messages };
    delete newMessages[tab];
    setMessages(newMessages);

    // Set active tab to another one
    const newActiveTab = Object.keys(newMessages)[0];
    setActiveTab(newActiveTab);
  };

  // Function to handle sending messages
  const handleSend = async () => {
    if (!input.trim()) return;

    // Add user message to the current active tab
    const userMessage = { sender: 'user', text: input };
    setMessages((prev) => ({
      ...prev,
      [activeTab]: [...prev[activeTab], userMessage],
    }));

    // Clear input field
    setInput('');

    // Simulate bot response
    const botResponse = await getBotResponse(input);
    setMessages((prev) => ({
      ...prev,
      [activeTab]: [...prev[activeTab], { sender: 'bot', text: botResponse }],
    }));
  };

  // Simulate the bot's response (you can replace this with a real API call)
  const getBotResponse = async (message:string) => {
    return `Your question is "${message}". Here's some wisdom: "Invest in yourself!"`;
  };

  // Function to handle instructions modal
  const handleInstructions = () => {
    setShowInstructions(true);
  };

  return (
    <div className="chat-container">
      <header className="chat-header">
        <img src={logo_img} style={{ width: "auto", height: "auto", marginTop: "-175px", marginBottom: "-335px"}}/>
        {/* <h1 style={{ marginBottom: "-35px"}}>
          <span style={{ fontSize: '100px' }}>B</span> u f f e t t .
          <span style={{ fontSize: '100px' }}> A I</span>
        </h1> */}
        <h1 style={{ marginBottom: "50px"}}>&nbsp;</h1>
        <h2> Buffett’s Secrets, Now in the Cloud.</h2>
      </header>
      {/* Instructions */}
      <button className="instructions-btn" onClick={handleInstructions}>
        Instructions
      </button>
      {/* Tab Buttons with Close Button */}
      <div className="tab-buttons">
        {Object.keys(messages).map((tab) => (
          <div key={tab} style={{ display: 'flex', alignItems: 'center' }}>
            <button
              className={activeTab === tab ? 'active' : ''}
              onClick={() => switchTab(tab)}
            >
              {tab}
            </button>
            {/* closing tab button */}
            <button
              onClick={() => closeTab(tab)}
              className="close-tab-btn"  // Use the CSS class here
            >
              X
            </button>
          </div>
        ))}
        <button onClick={handleNewTab}>+ New Tab</button>
      </div>

      {/* Chat Messages */}
      <div className="chat-messages">
        {messages[activeTab].map((msg, index) => (
          <div
            key={index}
            className={`message ${msg.sender === 'user' ? 'user-message' : 'bot-message'}`}
          >
            {msg.text}
          </div>
        ))}
      </div>

      {/* Chat Input */}
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

      {/* Instructions Modal (conditionally render) */}
      {showInstructions && (
        <div className="instructions-modal">
          <div className="modal-content">
            <h3>Model Instructions</h3>
            <p>
              This AI chatbot simulates Warren Buffett’s investment strategies,
              offering insights based on his famous investment philosophies.
            </p>
            <button onClick={() => setShowInstructions(false)}>Close</button>
          </div>
        </div>
      )}
    </div>
  );
}

// function App() {
//   const [messages, setMessages] = useState<{ sender: string; text: string }[]>([]);
//   const [input, setInput] = useState('');
//   const [showInstructions, setShowInstructions] = useState(false); // Add this line

//   const handleInstructions = () => {  // Add this function to handle the modal
//     setShowInstructions(true);
//   };

//   const handleSend = async () => {
//     if (!input.trim()) return;

//     // Add user message to the chat
//     const userMessage = { sender: 'user', text: input };
//     setMessages((prev) => [...prev, userMessage]);

//     // Clear input field
//     setInput('');

//     // Simulate bot response
//     const botResponse = await getBotResponse(input);
//     setMessages((prev) => [...prev, { sender: 'bot', text: botResponse }]);
//   };
//     const getBotResponse = async (message: string) => {
//     // Placeholder for backend API call
//     return `Alright, give me a second - Here's some wisdom: Invest in yourself!"`;
//   };

//   return (
//     <div className="chat-container">
//       <header className="chat-header">
//         {/* <h1>B u f f e t t . A I</h1> */}
//         <h1 style={{ marginBottom: "-35px"}}>
//           <span style={{ fontSize: "100px" }}>B</span> u f f e t t . 
//           <span style={{ fontSize: "100px" }}> A I</span>
//         </h1>
//         <h2 style={{ marginTop: "-10px" }}>
//           <p className="chat-subheader">Buffett’s Secrets, Now in the Cloud.</p>
//         </h2>
//       </header>
//       {/* Instructions */}
//       <button className="instructions-btn" onClick={handleInstructions}>
//         Instructions
//       </button>
//       <div className="chat-messages">
//         {messages.map((msg, index) => (
//           <div
//             key={index}
//             className={`message ${msg.sender === 'user' ? 'user-message' : 'bot-message'}`}
//           >
//             {msg.text}
//           </div>
//         ))}
//       </div>
//       <div className="chat-input">
//         <input
//           type="text"
//           placeholder="Ask Warren Buffett something..."
//           value={input}
//           onChange={(e) => setInput(e.target.value)}
//           onKeyDown={(e) => e.key === 'Enter' && handleSend()}
//         />
//         <button onClick={handleSend}>Send</button>
//       </div>
//     {/* Instructions Modal (conditionally render) */}
//     {showInstructions && (
//       <div className="instructions-modal">
//         <div className="modal-content">
//           <h3>Model Instructions</h3>
//           <p>This AI chatbot simulates Warren Buffett’s investment strategies, offering insights based on his famous investment philosophies.</p>
//           <button onClick={() => setShowInstructions(false)}>Close</button>
//         </div>
//       </div>
//     )}
//     </div>
//   );
// }

export default App;