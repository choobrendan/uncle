import React from 'react';
import './TextBox.css';

const TextInput = ({ value, onChange, onFocus, onBlur, placeholder }) => {
  return (
    <div className="neon-input-wrapper">
      <input
        type="text"
        value={value}
        onChange={onChange}
        onFocus={onFocus}
        onBlur={onBlur}
        className="neon-input"
        placeholder={placeholder}
      />
      <div className="neon-glow"></div>
      <div className="neon-glow-wide"></div>
    </div>
  );
};

export default TextInput;
