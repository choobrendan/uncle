import { useState, useEffect, useRef } from 'react';

export function useSpeechRecognition({ lang, continuous, interimResults }) {
  const [isListening, setIsListening] = useState(false);
  const [isFinal, setIsFinal] = useState(false);
  const [result, setResult] = useState('');
  const [error, setError] = useState(undefined);

  const recognitionRef = useRef(null);

  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

  // Check if the browser supports SpeechRecognition API
  const isSupported = Boolean(SpeechRecognition);

  // Initialize speech recognition if supported
  useEffect(() => {
    if (isSupported) {
      const recognition = new SpeechRecognition();
      recognition.continuous = continuous;
      recognition.interimResults = interimResults;
      recognition.lang = lang;

      recognition.onstart = () => {
        setIsFinal(false);
      };

      recognition.onresult = (event) => {
        const transcript = Array.from(event.results)
          .map((result) => {
            setIsFinal(result.isFinal);
            return result[0];
          })
          .map((result) => result.transcript)
          .join('');

        setResult(transcript);
        setError(undefined);
      };

      recognition.onerror = (event) => {
        setError('Speech recognition error detected: ' + event.error);
      };

      recognition.onend = () => {
        setIsListening(false);
      };

      recognitionRef.current = recognition;

      // Cleanup recognition on component unmount
      return () => {
        if (recognitionRef.current) {
          recognitionRef.current.stop();
        }
      };
    }
  }, [isSupported, continuous, interimResults, lang]);

  // Start recognition
  const start = () => {
    if (recognitionRef.current) {
      setIsListening(true);
      recognitionRef.current.start();
    }
  };

  // Stop recognition
  const stop = () => {
    if (recognitionRef.current) {
      setIsListening(false);
      recognitionRef.current.stop();
    }
  };

  // Toggle listening state
  const toggle = (value = isListening) => {
    setIsListening(value);
  };

  return {
    isSupported,
    isListening,
    isFinal,
    result,
    error,
    toggle,
    start,
    stop,
  };
}
