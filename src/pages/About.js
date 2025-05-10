import React, { useState } from "react";
import AboutCarousel from "../components/AboutCarousel";
import Background from "../components/Background";
import VoiceButton from "../components/VoiceButton";
import { useOutletContext } from "react-router-dom";

function About() {
  const {
    selectionIndex,
    setSelectionIndex,
    textSizeModifier,
    brightnessIndex,
    setBrightnessIndex,
    simplify,
  } = useOutletContext();

  const [showRender, setShowRender] = useState(true);
  const [showAbout, setShowAbout] = useState(true);
  // Simplified styles for About page
  const simplifiedStyles = {
    container: {
      height: "100vh",
      display: "flex",
      flexDirection: "column",
      alignItems: "center",
      justifyContent: "center",
      backgroundColor: "#fff",
      padding: "20px",
      overflow: "auto",
    },
    title: {
      fontSize: `${28 * textSizeModifier}px`,
      color: "#333",
      marginBottom: "20px",
      textAlign: "center",
      fontFamily: "Arial, sans-serif",
      fontWeight: "bold",
    },
  };

  if (simplify) {
    return (
      <div style={simplifiedStyles.container}>
        <h1 style={simplifiedStyles.title}>Features</h1>
        <AboutCarousel
          textSizeModifier={textSizeModifier}
          simplify={simplify}
        />
        <VoiceButton
          setSelectionIndex={setSelectionIndex}
          selectionIndex={selectionIndex}
        />
      </div>
    );
  }

  return (
    <div
      className="aboutBody"
      style={{
        height: "100vh",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        filter: `brightness(${1 * brightnessIndex}`,
      }}
    >
      <AboutCarousel textSizeModifier={textSizeModifier} simplify={simplify} />
      <VoiceButton
        setSelectionIndex={setSelectionIndex}
        selectionIndex={selectionIndex}
      />
    </div>
  );
}

export default About;
