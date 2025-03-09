import React, { useState, useEffect } from 'react';
import './TextBox.css';

const TextBox = ({ suggestions, setSelectionIndex, selectionIndex, text, onInput }) => {
  const [inputValue, setInputValue] = useState(text || ''); // Initialize with the text prop
  const [showDropdown, setShowDropdown] = useState(false);
  const [filteredSuggestions, setFilteredSuggestions] = useState([]);
  const [selectedSuggestion, setSelectedSuggestion] = useState(null);
  const [hoverIndex, setHoverIndex] = useState(-1);

  useEffect(() => {
    const handleKeyDown = (e) => {
      if (!showDropdown || filteredSuggestions.length === 0) return;
      if (e.keyCode === 13 && selectionIndex >= 0) {
        selectSuggestion(filteredSuggestions[selectionIndex]);
        e.preventDefault();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, [showDropdown, filteredSuggestions, selectionIndex]);

  const sendText = () => {
    console.log(inputValue);
    fetch('http://localhost:8000/send-message', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: inputValue }),
    })
      .then((response) => response.json())
      .then((data) => {
        setFilteredSuggestions(data);
      })
      .catch((error) => console.error('Error sending message:', error));
  };

  const selectSuggestion = (suggestion) => {
    setInputValue(suggestion);
    setSelectedSuggestion(suggestion);
    setShowDropdown(false);
  };

  const onBlur = () => {
    setTimeout(() => {
      setShowDropdown(false);
    }, 200);
  };

  const handleSelectButtonClick = () => {
    if (hoverIndex >= 0) {
      const selected = filteredSuggestions[hoverIndex];
      selectSuggestion(selected);
      setSelectionIndex(hoverIndex);
    }
  };

  const handleMouseEnter = (index) => {
    setHoverIndex(index);
  };

  // Update input value when text prop or voice result changes
  useEffect(() => {
    if (text) {
      setInputValue(text); // Set input value to the text prop from VoiceButton
    }
  }, [text]);

  return (
    <div className="neon-container">
      <div className="neon-input-wrapper">
        <input
          type="text"
          value={inputValue}
          onChange={(e) => {
            setInputValue(e.target.value);
            sendText(e.target.value);
          }}
          onFocus={() => setShowDropdown(true)}
          onBlur={onBlur}
          className="neon-input"
          placeholder="Type a prompt..."
        />
        <div className="neon-glow"></div>
        <div className="neon-glow-wide"></div>
      </div>

      {showDropdown && filteredSuggestions.length > 0 && (
        <div className="neon-dropdown">
          {filteredSuggestions.map((suggestion, index) => (
            <div
              key={index}
              onMouseDown={() => selectSuggestion(suggestion)}
              onMouseEnter={() => handleMouseEnter(index)}
              className={`neon-suggestion ${hoverIndex === index ? 'active' : ''}`}
            >
              {suggestion}
            </div>
          ))}
        </div>
      )}

      <div>
        <button
          className="select-button"
          onClick={handleSelectButtonClick}
          disabled={hoverIndex < 0}
        >
          Select
        </button>
      </div>
    </div>
  );
};

export default TextBox;
