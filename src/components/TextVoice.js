import React, { useState, useEffect } from 'react';
import './TextBox.css';

const TextVoice = ({ suggestions, setSelectionIndex, selectionIndex, text, onInput }) => {
  const [inputValue, setInputValue] = useState(text || '');
  const [filteredSuggestions, setFilteredSuggestions] = useState([]);
  const [selectedSuggestion, setSelectedSuggestion] = useState(null);

  useEffect(() => {
    const handleKeyDown = (e) => {
      if (filteredSuggestions.length === 0) return;
      if (e.keyCode === 13 && selectionIndex >= 0) {
        selectSuggestion(filteredSuggestions[selectionIndex]);
        e.preventDefault();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, [filteredSuggestions, selectionIndex]);

  const sendText = (value) => {
    console.log(value);
    fetch('http://localhost:8000/send-message', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: value }),
    })
      .then((response) => response.json())
      .then((data) => {
        console.log(data);
        const suggestionsList = data.map(item => ({
          id: item.id, // Extract the 'id' from each item
          name: item.text, // Extract the 'name' from each item
        }));
        setFilteredSuggestions(suggestionsList);
        console.log(suggestionsList, "Suggestions updated");
      })
      .catch((error) => console.error('Error sending message:', error));
  };

  const selectSuggestion = (suggestion) => {
    setInputValue(suggestion);
    setSelectedSuggestion(suggestion);
    if (onInput) {
      onInput(suggestion);
    }
  };

  // Update input value when text prop or voice result changes
  useEffect(() => {
    if (text) {
      setInputValue(text);
    }
  }, [text]);

  // Trigger the sendText function whenever inputValue changes
  useEffect(() => {
    if (inputValue.trim()) {
      sendText(inputValue);
    }
  }, [inputValue]);

  return (
    <div className="neon-container">
      <div className="neon-input-wrapper">
        <input
          type="text"
          value={inputValue}
          onChange={(e) => {
            setInputValue(e.target.value);
          }}
          className="neon-input"
          placeholder="Type a prompt..."
        />
        <div className="neon-glow"></div>
        <div className="neon-glow-wide"></div>
      </div>

      {filteredSuggestions.length > 0 && (
        <div className="suggestion-buttons">
          {filteredSuggestions.slice(0, 6).map((suggestion, index) => (
            <button
              key={suggestion.id}
              className={`suggestion-button ${selectionIndex === index ? 'active' : ''}`}
              onClick={() => {
                selectSuggestion(suggestion.name);
                setSelectionIndex(suggestion.id);
              }}
            >
              {suggestion.name}
            </button>
          ))}
        </div>
      )}
    </div>
  );
};

export default TextVoice;
