import React, { useState, useEffect } from 'react';
import { Outlet, useNavigate, useLocation  } from 'react-router-dom';
import Header from './components/Header';
import Background from './components/Background';

function App() {
  const location = useLocation();
  const [selectionIndex, setSelectionIndex] = useState(-1);
  const [textSizeModifier, setTextSizeModifier] = useState(1);
  const navigate = useNavigate();
  const [brightnessIndex, setBrightnessIndex] = useState(1);
  const [simplify, setSimplify]=useState(true)
  const [isUserLoggedIn, setIsUserLoggedIn]= useState("");
  useEffect(() => {
    switch (selectionIndex) {
      case 1: 
        setTextSizeModifier(prev => prev * 1.25);
        break;
      case 2:
        setTextSizeModifier(prev => prev / 1.25);
        break;
      case 3:
        setSimplify(!simplify)
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
      {location.pathname!=="/onboarding" && (<Header
        setBrightnessIndex={setBrightnessIndex}
        brightnessIndex={brightnessIndex}
        simplify={simplify}     isUserLoggedIn={    isUserLoggedIn}
      />)}
       {location.pathname!=="/onboarding" && ( <Background
        showRender={false}
        setBrightnessIndex={setBrightnessIndex}
        brightnessIndex={brightnessIndex}
        simplify={simplify}
      />)}
      
      <div class="body" >
      <Outlet context={{
        selectionIndex,
        setSelectionIndex,
        textSizeModifier,
        brightnessIndex,
        setBrightnessIndex,
        simplify,
        setIsUserLoggedIn,
        isUserLoggedIn,
      }} />
      </div>
    </React.StrictMode>
  );
}

export default App;