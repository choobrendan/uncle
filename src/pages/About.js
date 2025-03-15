import React, { useState } from 'react';
import AboutCarousel from '../components/AboutCarousel';
import Background from '../components/Background';
import VoiceButton from '../components/VoiceButton';
const AboutPage = ({setSelectionIndex, selectionIndex,textSizeModifier,brightnessIndex, setBrightnessIndex}) => {
  const [showRender, setShowRender] = useState(true);

  return (
    <div class="aboutBody" style={{ height: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center', filter:`brightness(${1*brightnessIndex}` }}>
        
      <AboutCarousel textSizeModifier={textSizeModifier}/>
      <VoiceButton setSelectionIndex={setSelectionIndex} selectionIndex={selectionIndex} />
      

    </div>
  );
};

export default AboutPage;
