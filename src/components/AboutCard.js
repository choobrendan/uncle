import React from 'react';
import './AboutCard.css'; // Import the external CSS for styling

const AboutCard = ({ item, textSizeModifier }) => {
  return (
    <div className="card"style={{ minWidth: `${320 * textSizeModifier}px`}}>
      <div className="card-grid"></div>
      <div className="geometric-shape circle"></div>
      <div className="geometric-shape triangle"></div>
      <div className="card-image"></div>
      <div className="neon-line"></div>
      <div className="card-content">
        <h2 className="card-title" style={{ fontSize: `${24 * textSizeModifier}px` }}>
          {item.name}
        </h2>
        <p className="card-text" style={{ fontSize: `${16 * textSizeModifier}px` }}>
          {item.description}
        </p>
      </div>
      <div className="scanline"></div>
    </div>
  );
};

export default AboutCard;
