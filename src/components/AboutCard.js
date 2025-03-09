import React from 'react';
import './AboutCard.css'; // Import the external CSS for styling

const AboutCard = ({ item }) => {
  return (
    <div className="card">
      <div className="card-grid"></div>
      <div className="geometric-shape circle"></div>
      <div className="geometric-shape triangle"></div>
      <div className="card-image">

      </div>
      <div className="neon-line"></div>
      <div className="card-content">
        <h2 className="card-title">{item.name}</h2>
        <p className="card-text">{item.description}</p>
      </div>
      <div className="scanline"></div>
    </div>
  );
};

export default AboutCard;
