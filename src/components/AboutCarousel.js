import React from 'react';
import AboutCard from '../components/AboutCard';
import "./AboutCarousel.css"
const AboutCarousel = () => {
  const items = [
    { name: 'Laptop', description: 'A portable computer for work and entertainment.' },
    { name: 'Smartphone', description: 'A handheld device that combines mobile phone features with computing functionality.' },
    { name: 'Headphones', description: 'A device for listening to audio, usually worn over or in the ears.' },
    { name: 'Watch', description: 'A wristwatch used for telling time, often with additional features like fitness tracking.' },
    { name: 'Camera', description: 'A device used to capture photos or videos.' },
  ];

  return (
    <div className="aboutCard" style={styles.aboutCard}>
      {items.map((item, index) => (
        <AboutCard key={index} item={item} />
      ))}
    </div>
  );
};

const styles = {
  aboutCard: {
    display: 'flex',
    flexDirection: 'row',
    overflow: 'auto',
    padding: '10px',
    marginLeft: '35vw',
    paddingBottom: '15px',
  },
};

export default AboutCarousel;
