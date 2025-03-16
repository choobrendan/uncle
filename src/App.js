import React, { useState, useEffect } from 'react';
import { Outlet, useNavigate } from 'react-router-dom';
import Header from './components/Header';
import Background from './components/Background';

function App() {
  const [selectionIndex, setSelectionIndex] = useState(-1);
  const [textSizeModifier, setTextSizeModifier] = useState(1);
  const navigate = useNavigate();
  const [brightnessIndex, setBrightnessIndex] = useState(1);

  useEffect(() => {
    switch (selectionIndex) {
      case 1:
        setTextSizeModifier(prev => prev * 1.25);
        break;
      case 2:
        setTextSizeModifier(prev => prev / 1.25);
        break;
      case 5:
        setBrightnessIndex(prev => prev * 1.1);
        break;
      case 6:
        setBrightnessIndex(prev => prev / 1.1);
        break;
      case 7:
        navigate('/');
        break;
      case 8:
        navigate('/about');
        break;
      default:
        break;
    }
    setSelectionIndex(-1);
  }, [selectionIndex, navigate]);

  return (
    <React.StrictMode>
      <Header
        setBrightnessIndex={setBrightnessIndex}
        brightnessIndex={brightnessIndex}
      />
      <Background
        showRender={false}
        setBrightnessIndex={setBrightnessIndex}
        brightnessIndex={brightnessIndex}
      />
      <div class="body" >
      <Outlet context={{
        selectionIndex,
        setSelectionIndex,
        textSizeModifier,
        brightnessIndex,
        setBrightnessIndex
      }} />
      </div>
    </React.StrictMode>
  );
}

export default App;