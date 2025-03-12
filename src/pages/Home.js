import React, { useState, useEffect, useRef } from 'react';
import "./Home.css"
// Components
import Background from '../components/Background';
import VoiceButton from '../components/VoiceButton';
import TextBox from '../components/TextBox';
import Header from '../components/Header';
const Home = ({selectionIndex,setSelectionIndex}) => {
    
  // State variables
  const [mouseX, setMouseX] = useState('');
  const [mouseY, setMouseY] = useState('');
  const [gazeX, setGazeX] = useState(0);
  const [gazeY, setGazeY] = useState(0);
  const [displayedText, setDisplayedText] = useState('');
  const [cursorVisible, setCursorVisible] = useState(true);
  const [currentSentenceIndex, setCurrentSentenceIndex] = useState(0);
  const [currentCharIndex, setCurrentCharIndex] = useState(0);
  const [currentStyle, setCurrentStyle] = useState({
    color: '',
    fontSize: '',
    fontWeight: '',
    letterSpacing: '',
    fontFamily: ''
  });
  const [textSizeModifier, setTextSizeModifier] = useState(1);
  const [filteredOptions, setFilteredOptions] = useState([]);
  const [message, setMessage] = useState('');
  const [showRender, setShowRender] = useState(true);




  const typingInterval = useRef(null);
  const deletingInterval = useRef(null);

  const sentences = ["Customised?"];
  const command = [
    "Increase font size",
    "Decrease font size",
    "Increase container size",
    "Decrease container size",
    "Increase brightness",
    "Decrease brightness",
    "Navigate to home page",
    "Navigate to about page",
  ];



  // Text Style Computation
  const textStyle = {
    color: currentStyle.color,
    fontSize: currentStyle.fontSize,
    fontWeight: currentStyle.fontWeight,
    letterSpacing: currentStyle.letterSpacing,
    fontFamily: currentStyle.fontFamily
  };

  useEffect(() => {
    // Start typing effect on mount
    typingInterval.current = setInterval(type, 100);

    return () => {
      // Cleanup intervals on unmount
      if (typingInterval.current) clearInterval(typingInterval.current);
      if (deletingInterval.current) clearInterval(deletingInterval.current);
    };
  }, []);

  const handleMouseMove = (event) => {
    setMouseX(event.clientX.toString().padStart(4, '0'));
    setMouseY(event.clientY.toString().padStart(4, '0'));
  };

  // Random Style Generator Functions
  const getRandomColor = () => {
    const r = Math.floor(Math.random() * 156) + 100;
    const g = Math.floor(Math.random() * 156) + 100;
    const b = Math.floor(Math.random() * 156) + 100;
    return `rgb(${r}, ${g}, ${b})`;
  };

  const getRandomFontSize = () => `${Math.floor(Math.random() * (60 - 32 + 1)) + 32}px`;

  const getRandomFontWeight = () => {
    const weights = [100, 200, 300, 400, 500, 600, 700, 800, 900];
    return weights[Math.floor(Math.random() * weights.length)];
  };

  const getRandomLetterSpacing = () => `${Math.floor(Math.random() * 11)}px`;

  const getRandomFontFamily = () => {
    const fonts = ['Arial', 'Georgia', 'Courier New', 'Comic Sans MS', 'Tahoma', 'Times New Roman'];
    return fonts[Math.floor(Math.random() * fonts.length)];
  };

  const changeTextStyle = () => {
    setCurrentStyle({
      color: getRandomColor(),
      fontSize: getRandomFontSize(),
      fontWeight: getRandomFontWeight(),
      letterSpacing: getRandomLetterSpacing(),
      fontFamily: getRandomFontFamily()
    });
  };

  const type = () => {
    const currentSentence = sentences[currentSentenceIndex];

    setDisplayedText(currentSentence.substring(0, currentCharIndex + 1));
    setCurrentCharIndex(prevIndex => prevIndex + 1);

    if (displayedText === currentSentence) {
      clearInterval(typingInterval.current);
      setCursorVisible(false);
      setTimeout(() => {
        deletingInterval.current = setInterval(deleteText, 50);
      }, 1000);
    }
  };

  const deleteText = () => {
    const currentSentence = sentences[currentSentenceIndex];
    setDisplayedText(currentSentence.substring(0, currentCharIndex - 1));
    setCurrentCharIndex(prevIndex => prevIndex - 1);

    if (displayedText === '') {
      clearInterval(deletingInterval.current);
      changeTextStyle();

      if (currentSentenceIndex === sentences.length - 1) {
        setCurrentSentenceIndex(0);
      } else {
        setCurrentSentenceIndex(prevIndex => prevIndex + 1);
      }

      setCurrentCharIndex(0);

      setTimeout(() => {
        setCursorVisible(true);
        typingInterval.current = setInterval(type, 100);
      }, 200);
    }
  };


  useEffect(() => {
    console.log(selectionIndex);
  
    if (selectionIndex === 1) {
      setTextSizeModifier(prevModifier => prevModifier * 1.25);
    } else if (selectionIndex === 2) {
      setTextSizeModifier(prevModifier => prevModifier / 1.25);
    }
  
    // Set selectionIndex to -1 once the textSizeModifier is updated
    setSelectionIndex(-1);
  
    console.log(textSizeModifier);
  }, [selectionIndex]);
  
  

  const sendText = () => {
    fetch('http://localhost:8000/send-message', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ message })
    })
      .then(response => {
        if (!response.ok) {
          throw new Error('Network response was not ok');
        }
        return response.json();
      })
      .then(data => {
        setFilteredOptions(data);
      })
      .catch(error => {
        console.error('Error sending message:', error);
      });
  };

  return (
    <div  style={{ alignItems: 'center', justifyContent: 'center' }}onMouseMove={handleMouseMove}>


      <div className="content">
        <div className="content-1">
          <div className="how-would-you">
            <p style={{ fontFamily: 'Oxanium', fontWeight: 200, fontSize: `${36 * textSizeModifier}px` }}>
              How would you like your website
            </p>
          </div>
          <div style={{ width: '360px' }}></div>
          <div className="customised">
            <div id="banner-text" style={textStyle}>{displayedText}</div>
            <div id="cursor" style={{ visibility: cursorVisible ? 'visible' : 'hidden' }}></div>
          </div>
        </div>

        <div className="content-desc">
          <div className="backdrop-filter">
            <div className="flip-card-inner">
              <div className="flip-card-front">
                <div>
                  <p>Front</p>
                </div>
              </div>
              <div className="flip-card-back">
                <div>
                  <h1>John Doe</h1>
                  <p>Architect & Engineer</p>
                  <p>We love that guy</p>
                </div>
              </div>
            </div>
          </div>
        </div>

        <div className="content-1">
          <div className="how-would-you">
            <p style={{ fontFamily: 'Oxanium', fontWeight: 200, fontSize:`${36 * textSizeModifier}px` }}>
              How would you want to modify this website?
            </p>
          </div>
          <div style={{ width: '360px' }}></div>
          <div className="customised">
            <p style={{ fontFamily: 'Oxanium', fontWeight: 200, fontSize: `${36 * textSizeModifier}px`}}>
              Try it out now!
            </p>
            <TextBox selectionIndex={selectionIndex} setSelectionIndex={setSelectionIndex}/>
          </div>
        </div>
      </div>
      <VoiceButton  setSelectionIndex={setSelectionIndex} selectionIndex={selectionIndex}/>
      <Background showRender={false} />
    </div>
  );
};

export default Home;
