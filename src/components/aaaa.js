import React, { useState, useEffect } from "react";
import { useSpeechRecognition } from "../composables/useSpeechRecognition"; // Assuming custom hook
import "./VoiceButton.css"; // Add styles for your
import TextBox from "./TextBox";
//component

const VoiceButton = ({ toggleMainTextDiv }) => {
  const [isExpanded, setIsExpanded] = useState(false);
  const [buttonHeight, setButtonHeight] = useState("75px");
  const [buttonWidth, setButtonWidth] = useState("75px");
  const [borderRadius, setBorderRadius] = useState(75);
  const [buttonColour, setButtonColour] = useState("grey");
  const [result, setResult] = useState("");
  const [error, setError] = useState("");
  const [timer, setTimer] = useState(null);

  // Using the custom hook for speech recognition
  const {
    isListening,
    isSupported,
    stop,
    result: speechResult,
    start,
    error: speechError,
  } = useSpeechRecognition({
    lang: "en-US",
    continuous: true,
    interimResults: true,
  });

  useEffect(() => {
    if (speechResult) {
      setResult(speechResult);
    }
    if (speechError) {
      setError(speechError);
    }
  }, [speechResult, speechError]);

  const toggleSize = () => {
    if (isExpanded) {
      setButtonHeight("75px");
      setButtonWidth("75px");
      setBorderRadius(75);
    } else {
      setButtonHeight("50%");
      setButtonWidth("35%");
      setBorderRadius(25);
    }
    setIsExpanded(!isExpanded);
  };

  const voiceTimer = () => {
    if (timer) {
      clearTimeout(timer);
    }

    const newTimer = setTimeout(() => {
      sendVoice();
    }, 2000);
    setTimer(newTimer);
  };

  const sendVoice = () => {
    console.log("Function executed after timeout");
    fetch("http://localhost:8000/send-voice", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ message: result }),
    })
      .then((response) => {
        if (!response.ok) {
          throw new Error("Network response was not ok");
        }
        return response.json();
      })
      .then((data) => {
        console.log(data);
        // Emitting voice data to parent component
        if (toggleMainTextDiv) {
          toggleMainTextDiv(data);
        }
      })
      .catch((error) => {
        console.error("Error sending message:", error);
      });
  };

  const startTrigger = () => {
    start();
    setButtonColour("red");
  };

  const stopTrigger = () => {
    setButtonColour("grey");
    stop();
  };

  return (
    <section
      className="voice-button"
      style={{
        height: buttonHeight,
        width: buttonWidth,
        borderRadius: `${borderRadius}px`,
      }}
    >
      {isExpanded && (
        <div className="voice-instructions">
          <p>Please speak!!!!</p>
        </div>
      )}

      {isExpanded && (
        <div className="close-voice">
          <button className="close-voice-button" onClick={toggleSize}>
            Close
          </button>
        </div>
      )}

      <div className="button-base">
        <button
          style={{
            backgroundColor: buttonColour,
          }}
          className="voice-trigger"
          onClick={() =>
            !isExpanded
              ? toggleSize()
              : isListening
              ? stopTrigger()
              : startTrigger()
          }
        >
          <img
            className="mic"
            src="https://www.iconpacks.net/icons/1/free-microphone-icon-342-thumb.png"
            alt="mic-icon"
          />
        </button>
      </div>

      {isExpanded && (
        <div>
          {error ? (
            <p>{error}</p>
          ) : (
            <textarea
              value={result}
              className="text-transcript"
              cols="30"
              rows="10"
              onInput={voiceTimer}
            />
          )}

          <TextBox text={result} onInput={voiceTimer}></TextBox>
        </div>
      )}
    </section>
  );
};

export default VoiceButton;
