import React from 'react';
import './AboutCard.css'; // Import the external CSS for styling

const AboutCard = ({ item, textSizeModifier, simplify }) => {
  // Simplified card styles
  const simplifiedCardStyles = {
    card: {
      minWidth: '120px',
      width:"20%",
      padding: '20px',
      backgroundColor: '#f5f5f5',
      borderRadius: '8px',
      boxShadow: '0 2px 5px rgba(0, 0, 0, 0.1)',
      margin: '0',
      position: 'relative'
    },
    cardContent: {
      padding: '0',
      backgroundColor: 'transparent'
    },
    cardTitleContainer:{
      height: "100px",
    textAlign: "center",
    display: "flex",
    justifyContent: "center",
    alignItems: "center",
    },
    cardTitle: {
      fontSize: `${22 * textSizeModifier}px`,
      fontWeight: 'bold',
      color: '#333',
      marginBottom: '10px',
      textTransform: 'none',
      letterSpacing: 'normal',
      textShadow: 'none',
      fontFamily: 'Arial, sans-serif'
    },
    cardText: {
      fontSize: `${16 * textSizeModifier}px`,
      color: '#555',
      lineHeight: '1.5',
      margin: '0',
      textShadow: 'none',
      fontFamily: 'Arial, sans-serif'
    }
  };

  // Original card style adjustments
  const originalCardStyles = {
    minWidth: `${320 * textSizeModifier}px`
  };

  if (simplify) {
    return (
      <div style={simplifiedCardStyles.card}>
        <div style={simplifiedCardStyles.cardContent}>
        <div style={simplifiedCardStyles.cardTitleContainer}>
          <h2 style={simplifiedCardStyles.cardTitle}>
            {item.name}
          </h2>
          </div>
          <p style={simplifiedCardStyles.cardText}>
            {item.description}
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="card" style={originalCardStyles}>
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