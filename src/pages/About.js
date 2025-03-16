import React, { useState } from 'react';
import AboutCarousel from '../components/AboutCarousel';
import Background from '../components/Background';
import VoiceButton from '../components/VoiceButton';
import { useOutletContext } from 'react-router-dom';

function About() {
  const {
    selectionIndex,
    setSelectionIndex,
    textSizeModifier,
    brightnessIndex,
    setBrightnessIndex
  } = useOutletContext();


  const [showRender, setShowRender] = useState(true);

  return (
    <div class="aboutBody" style={{ height: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center', filter:`brightness(${1*brightnessIndex}` }}>
        
      <AboutCarousel textSizeModifier={textSizeModifier}/>
      <VoiceButton setSelectionIndex={setSelectionIndex} selectionIndex={selectionIndex} />
      

    </div>
  );
};

export default About;
