import React, { useState, useEffect } from 'react';
import TextInput from './TextInput';
import './TextBox.css';

const TextVoice = ({ suggestions, setSelectionIndex, selectionIndex, text, onInput ,page,columnInfo}) => {
  const [inputValue, setInputValue] = useState(text || '');
  const [filteredSuggestions, setFilteredSuggestions] = useState([]);

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
        const suggestionsList = data.map((item) => ({
          id: item.id, // Extract the 'id' from each item
          name: item.text, // Extract the 'name' from each item
        }));
        setFilteredSuggestions(suggestionsList);
        console.log(suggestionsList, 'Suggestions updated');
      })
      .catch((error) => console.error('Error sending message:', error));
  };
  const sendPredict = (result) => {
    console.log(columnInfo)
    console.log(JSON.stringify(columnInfo,undefined,2))
    fetch('http://localhost:8000/filter_sentence', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ keys: Object.keys(columnInfo), message:"I want the species to be adelie, the culmen length to be between 20 and 21, and flipper length to be more than 190"  }),
    })
      .then((response) => response.json())
      .then((data) => {
console.log(data)
      })
      .catch((error) => console.error('Error sending message:', error));




    // fetch('http://localhost:8000/predict-message', {
    //   method: 'POST',
    //   headers: { 'Content-Type': 'application/json' },
    //   body: JSON.stringify({ message: result }),
    // })
    //   .then((response) => response.json())
    //   .then((data) => {

    //   })
    //   .catch((error) => console.error('Error sending message:', error));
  };
  const selectSuggestion = (suggestion) => {
    setInputValue(suggestion);
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
      console.log(page)
      if(page==="/graph"){
        sendPredict(inputValue);
      }
    }
  }, [inputValue]);

  return (
    <div className="neon-container">
      <TextInput
        value={inputValue}
        onChange={(e) => {
          setInputValue(e.target.value);
        }}
        placeholder="Type a prompt..."
      />

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
