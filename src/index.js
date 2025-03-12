import React, { useState, useEffect } from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import reportWebVitals from './reportWebVitals';
import { BrowserRouter as Router, Routes, Route, useNavigate } from 'react-router-dom';
import Home from './pages/Home';
import About from './pages/About';
import Header from './components/Header';

function App() {
  const [selectionIndex, setSelectionIndex] = useState(-1);
  const [textSizeModifier, setTextSizeModifier] = useState(1);
  const navigate = useNavigate();
  const [brightnessIndex, setBrightnessIndex]= useState(1);
  useEffect(() => {
    switch (selectionIndex) {
      case 1:
        setTextSizeModifier(prev => prev * 1.25);
        break;
      case 2:
        setTextSizeModifier(prev => prev / 1.25);
        
        break;
      case 5:
        setBrightnessIndex(prev => prev*1.1)
        break;
        case 6:
          setBrightnessIndex(prev => prev/1.1)
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

      <Header setBrightnessIndex={setBrightnessIndex}
                    brightnessIndex={brightnessIndex}/>
      <Routes>
        <Route 
          path="/" 
          element={<Home 
                    selectionIndex={selectionIndex}
                    setSelectionIndex={setSelectionIndex}
                    textSizeModifier={textSizeModifier}
                    setBrightnessIndex={setBrightnessIndex}
                    brightnessIndex={brightnessIndex}
                  />} 
        />
        <Route 
          path="/about" 
          element={<About 
                    selectionIndex={selectionIndex}
                    setSelectionIndex={setSelectionIndex}
                    textSizeModifier={textSizeModifier}
                    setBrightnessIndex={setBrightnessIndex}
                    brightnessIndex={brightnessIndex}
                  />} 
        />
      </Routes>
    </React.StrictMode>
  );
}

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <Router>
    <App />
  </Router>
);
reportWebVitals();