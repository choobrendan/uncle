import React, { useState } from 'react';
import AboutCarousel from '../components/AboutCarousel';
import Background from '../components/Background';

const AboutPage = () => {
  const [showRender, setShowRender] = useState(true);

  return (
    <div class="aboutBody" style={{ height: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        
      <AboutCarousel />
      <Background showRender={false} />
    </div>
  );
};

export default AboutPage;
