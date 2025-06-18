import React, { useState, useEffect, useRef } from "react";
import "./Home.css";
// Components
import Background from "../components/Background";
import VoiceButton from "../components/VoiceButton";
import TextBox from "../components/TextBox";
import Header from "../components/Header";
import { useOutletContext } from "react-router-dom";

function Home() {
  const {
    selectionIndex,
    setSelectionIndex,
    textSizeModifier,
    brightnessIndex,
    setBrightnessIndex,
    simplify,
    isUserLoggedIn,
    setIsUserLoggedIn,
    font,
  } = useOutletContext();

  // State variables
  const [mouseX, setMouseX] = useState("");
  const [mouseY, setMouseY] = useState("");
  const [gazeX, setGazeX] = useState(0);
  const [gazeY, setGazeY] = useState(0);
  const [displayedText, setDisplayedText] = useState("");
  const [cursorVisible, setCursorVisible] = useState(true);
  const [currentSentenceIndex, setCurrentSentenceIndex] = useState(0);
  const [currentCharIndex, setCurrentCharIndex] = useState(0);
  const [currentStyle, setCurrentStyle] = useState({
    color: "",
    fontSize: "",
    fontWeight: "",
    letterSpacing: "",
    fontFamily: "",
  });
  const [filteredOptions, setFilteredOptions] = useState([]);
  const [message, setMessage] = useState("");
  const [showRender, setShowRender] = useState(true);
  const [aboutRender, setaboutRender] = useState(true);

  const typingInterval = useRef(null);
  const deletingInterval = useRef(null);

  const sentences = ["Customised?"];
  const command = [
    "Increase font size",
    "Decrease font size",
    "simplify webpage",
    "Change font",
    "Increase brightness",
    "Decrease brightness",
    "Navigate to home page",
    "Navigate to about page",
  ];

  // Simplified styles
  const simplifiedStyles = {
    container: {
      display: "flex",
      flexDirection: "column",
      alignItems: "center",
      justifyContent: "center",
      minHeight: "100vh",
      backgroundColor: "#fff",
      color: "#333",
      fontFamily: "Arial, sans-serif",
      paddingTop: "100px",
      textAlign: "center",
      overflow: "auto",
    },
    content: {
      minWidth: "800px",
      maxWidth: "80%",
      width: "100%",
    },
    heading: {
      fontSize: `${28 * textSizeModifier}px`,
      fontWeight: "bold",
      marginBottom: "20px",
      color: "#333",
    },
    subheading: {
      fontSize: `${20 * textSizeModifier}px`,
      marginBottom: "10px",
      color: "#333",
    },
    section: {
      marginBottom: "30px",
      padding: "20px",
      backgroundColor: "#f5f5f5",
      borderRadius: "8px",
      boxShadow: "0 2px 4px rgba(0, 0, 0, 0.1)",
    },
    textBox: {
      marginTop: "20px",
    },
    cursor: {
      display: "inline-block",
      verticalAlign: "middle",
      width: "2px",
      height: "20px",
      backgroundColor: "#333",
      animation: "blink 0.75s step-end infinite",
    },
  };

  // Text Style Computation
  const textStyle = {
    color: currentStyle.color,
    fontSize: currentStyle.fontSize,
    fontWeight: currentStyle.fontWeight,
    letterSpacing: currentStyle.letterSpacing,
    fontFamily: currentStyle.fontFamily,
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
    setMouseX(event.clientX.toString().padStart(4, "0"));
    setMouseY(event.clientY.toString().padStart(4, "0"));
  };

  // Random Style Generator Functions
  const getRandomColor = () => {
    const r = Math.floor(Math.random() * 156) + 100;
    const g = Math.floor(Math.random() * 156) + 100;
    const b = Math.floor(Math.random() * 156) + 100;
    return `rgb(${r}, ${g}, ${b})`;
  };

  const getRandomFontSize = () =>
    `${Math.floor(Math.random() * (60 - 32 + 1)) + 32}px`;

  const getRandomFontWeight = () => {
    const weights = [100, 200, 300, 400, 500, 600, 700, 800, 900];
    return weights[Math.floor(Math.random() * weights.length)];
  };

  const getRandomLetterSpacing = () => `${Math.floor(Math.random() * 11)}px`;

  const getRandomFontFamily = () => {
    const fonts = [
      "Arial",
      "Georgia",
      "Courier New",
      "Comic Sans MS",
      "Tahoma",
      "Times New Roman",
    ];
    return fonts[Math.floor(Math.random() * fonts.length)];
  };

  const changeTextStyle = () => {
    setCurrentStyle({
      color: getRandomColor(),
      fontSize: getRandomFontSize(),
      fontWeight: getRandomFontWeight(),
      letterSpacing: getRandomLetterSpacing(),
      fontFamily: getRandomFontFamily(),
    });
  };

  const type = () => {
    const currentSentence = sentences[currentSentenceIndex];

    setDisplayedText(currentSentence.substring(0, currentCharIndex + 1));
    setCurrentCharIndex((prevIndex) => prevIndex + 1);

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
    setCurrentCharIndex((prevIndex) => prevIndex - 1);

    if (displayedText === "") {
      clearInterval(deletingInterval.current);
      changeTextStyle();

      if (currentSentenceIndex === sentences.length - 1) {
        setCurrentSentenceIndex(0);
      } else {
        setCurrentSentenceIndex((prevIndex) => prevIndex + 1);
      }

      setCurrentCharIndex(0);

      setTimeout(() => {
        setCursorVisible(true);
        typingInterval.current = setInterval(type, 100);
      }, 200);
    }
  };

  const sendText = () => {
    fetch("http://localhost:8000/send-message", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ message }),
    })
      .then((response) => {
        if (!response.ok) {
          throw new Error("Network response was not ok");
        }
        return response.json();
      })
      .then((data) => {
        setFilteredOptions(data);
      })
      .catch((error) => {
        console.error("Error sending message:", error);
      });
  };

  // Render simplified version

  // Render original version
  return (
    <div
      style={{
        alignItems: "center",
        justifyContent: "center",
        fontFamily: font,
      }}
      onMouseMove={handleMouseMove}
    >
      {simplify && (
        <div style={simplifiedStyles.container} onMouseMove={handleMouseMove}>
          <div style={simplifiedStyles.content}>
            <div style={simplifiedStyles.section}>
              <h1 style={simplifiedStyles.heading}>
                How would you like your website
              </h1>
              <div>
                <h1>Customised?</h1>
              </div>
            </div>

            <div style={simplifiedStyles.section}>
              <h2 style={simplifiedStyles.heading}>Website Customisation</h2>
              <p style={simplifiedStyles.subheading}>
                Our website offers easy automatic and manual customisation
                options to make your browsing experience better. You can adjust
                font size, brightness, and more to suit your needs.
              </p>
            </div>

            <div style={simplifiedStyles.section}>
              <h2 style={simplifiedStyles.heading}>
                How would you want to modify this website?
              </h2>
              <p style={simplifiedStyles.subheading}>Try it out now!</p>
              <div style={simplifiedStyles.textBox}>
                <TextBox
                  selectionIndex={selectionIndex}
                  setSelectionIndex={setSelectionIndex}
                  isApiCallDisabled={false}
                />
              </div>
            </div>
          </div>
          <VoiceButton
            setSelectionIndex={setSelectionIndex}
            simplify={simplify}
            selectionIndex={selectionIndex}
            font={font}
          />
        </div>
      )}
      {!simplify && (
        <div className="content">
          <div className="content-1">
            <div className="how-would-you">
              <p
                style={{
                  fontFamily: font,
                  fontWeight: 200,
                  fontSize: `${36 * textSizeModifier}px`,
                }}
              >
                How would you like your website
              </p>
            </div>
            <div style={{ width: "360px" }}></div>
            <div className="customised">
              <div id="banner-text" style={textStyle}>
                {displayedText}
              </div>
              <div
                id="cursor"
                style={{ visibility: cursorVisible ? "visible" : "hidden" }}
              ></div>
            </div>
          </div>

          <div className="content-desc">
            <div className="backdrop-filter">
              <div className="flip-card-inner">
                <div className="flip-card-front">
                  <div style={{ margin: "20px" }}>
                    <p style={{ fontSize: "36px" }}>
                      Struggling to use a Website?
                    </p>
                    <img
                      style={{
                        maxWidth: "80%",
                        minWidth: "300px",
                        padding: "20px",
                        paddingLeft: "40px",
                        paddingRight: "40px",
                      }}
                      src="output-onlinepngtools.png"
                    ></img>
                    <p style={{ fontSize: "20px", margin: "0px" }}>
                      Aging related issues may cause the elderly to be less
                      comfortable in using websites. How can we solve it?
                    </p>
                  </div>
                </div>
                <div className="flip-card-back">
                  <div
                    style={{
                      display: "flex",
                      flexDirection: "column",
                      gap: "20px",
                      margin: "40px",
                      height: "calc(100% - 80px)", // Subtract total margin (40px top + 40px bottom)
                      overflow: "hidden",
                      boxSizing: "border-box",
                    }}
                  >
                    {/* First section */}
                    <div
                      style={{
                        display: "flex",
                        flexDirection:
                          window.innerWidth > 768 ? "row" : "column",
                        alignItems: "center",
                        gap: "20px",
                        minHeight: "0", // Allows flex item to shrink
                      }}
                    >
                      <img
                        style={{
                          width: "100%",
                          maxWidth: window.innerWidth > 768 ? "50%" : "200px",
                          minWidth: "150px",
                          height: "auto",
                          objectFit: "contain",
                          flexShrink: 1,
                        }}
                        src="output-onlinepngtools (1).png"
                        alt="AI Detection Algorithm"
                      />
                      <p
                        style={{
                          fontSize: "clamp(14px, 2.5vw, 20px)",
                          margin: "0",
                          textAlign: "justify",
                          lineHeight: "1.4",
                          flex: 1,
                          minWidth: "0", // Allows text to wrap properly
                        }}
                      >
                        Using a self made AI algorithm, we can automatically
                        detect user's issues based on their input to the
                        application. We would then modify the website to allow
                        automatic UI adjustments.
                      </p>
                    </div>

                    {/* Second section */}
                    <div
                      style={{
                        display: "flex",
                        flexDirection:
                          window.innerWidth > 768 ? "row-reverse" : "column",
                        alignItems: "center",
                        gap: "20px",
                        minHeight: "0", // Allows flex item to shrink
                      }}
                    >
                      <img
                        style={{
                          width: "100%",
                          maxWidth: window.innerWidth > 768 ? "50%" : "200px",
                          minWidth: "150px",
                          height: "auto",
                          objectFit: "contain",
                          flexShrink: 1,
                        }}
                        src="output-onlinepngtools (2).png"
                        alt="Manual Adjustment Interface"
                      />
                      <p
                        style={{
                          fontSize: "clamp(14px, 2.5vw, 20px)",
                          margin: "0",
                          textAlign: "justify",
                          lineHeight: "1.4",
                          flex: 1,
                          minWidth: "0", // Allows text to wrap properly
                        }}
                      >
                        Or, if you want to adjust things manually, you can type
                        in instructions or ways to work on modifying the website
                        based on your specific needs.
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <div className="content-1">
            <div className="how-would-you">
              <p
                style={{
                  fontFamily: font,
                  fontWeight: 200,
                  fontSize: `${36 * textSizeModifier}px`,
                }}
              >
                How would you want to modify this website?
              </p>
            </div>
            <div style={{ width: "360px" }}></div>
            <div className="customised">
              <p
                style={{
                  fontFamily: font,
                  fontWeight: 200,
                  fontSize: `${36 * textSizeModifier}px`,
                }}
              >
                Try it out now!
              </p>
              <TextBox
                selectionIndex={selectionIndex}
                setSelectionIndex={setSelectionIndex}
                simplify={simplify}
                isApiCallDisabled={false}
              />
            </div>
          </div>
        </div>
      )}

      <VoiceButton
        simplify={simplify}
        setSelectionIndex={setSelectionIndex}
        selectionIndex={selectionIndex}
        font={font}
      />
    </div>
  );
}

export default Home;
