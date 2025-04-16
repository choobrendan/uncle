import React, { useState, useEffect } from 'react';
import { Outlet, useNavigate, useLocation } from 'react-router-dom';
import { createClient } from "@supabase/supabase-js";
import Header from './components/Header';
import Background from './components/Background';

const supabaseUrl = "https://hgatxkpmrskbdqigenav.supabase.co";
const supabaseKey = process.env.REACT_APP_SUPABASE_KEY;
const supabase = createClient(supabaseUrl, supabaseKey);

function App() {
  const location = useLocation();
  const navigate = useNavigate();

  // Initialize isUserLoggedIn from local storage.
  // If it's falsy, we'll work in "guest" mode.
  const [isUserLoggedIn, setIsUserLoggedIn] = useState(() => {
    return localStorage.getItem("isUserLoggedIn") || "";
  });

  const [selectionIndex, setSelectionIndex] = useState(-1);
  const [textSizeModifier, setTextSizeModifier] = useState(1);
  const [brightnessIndex, setBrightnessIndex] = useState(1);
  const [simplify, setSimplify] = useState(false);
  const [userId, setUserId] = useState(null); // for future updates
  
  // If user is not logged in, you could optionally load settings from local storage if they exist.
  useEffect(() => {
    if (!isUserLoggedIn) {
      const storedTextSize = localStorage.getItem("textSizeModifier");
      const storedBrightness = localStorage.getItem("brightnessIndex");
      const storedSimplify = localStorage.getItem("simplify");
  
      if (storedTextSize !== null) setTextSizeModifier(parseFloat(storedTextSize));
      if (storedBrightness !== null) setBrightnessIndex(parseFloat(storedBrightness));
      if (storedSimplify !== null) setSimplify(storedSimplify === "true");
    }
  }, [isUserLoggedIn]);
  

  // Fetch user settings from the DB only if user is logged in
  const fetchUserSettings = async () => {
    if (!isUserLoggedIn) {
      // If not logged in, ensure default values.
      setTextSizeModifier(1);
      setBrightnessIndex(1);
      setSimplify(false);
      return;
    }
    
    try {
      const { data, error } = await supabase
        .from("userData")
        .select("*")
        .limit(1)
        .single();

      if (error) {
        console.error("Error fetching user settings:", error);
        return;
      }

      // Set state from DB if available; otherwise, use defaults.
      setTextSizeModifier(data.textSizeModifier || 1);
      setBrightnessIndex(data.brightnessIndex || 1);
      setSimplify(data.simplify || false);
      setUserId(data.id);
    } catch (err) {
      console.error("Unexpected error:", err);
    }
  };

  // Update database with changed values (if user is logged in).
  const updateUserSettings = async (updates) => {
    if (userId === null) return;

    try {
      const { error } = await supabase
        .from("userData")
        .update(updates)
        .eq("id", userId);

      if (error) {
        console.error("Failed to update user settings:", error);
      }
    } catch (err) {
      console.error("Unexpected error updating user settings:", err);
    }
  };

  // Save or clear local storage settings based on login status.
  useEffect(() => {
    localStorage.setItem("isUserLoggedIn", isUserLoggedIn);
    fetchUserSettings();

    if (isUserLoggedIn) {
      // Remove guest settings when user logs in.
      localStorage.removeItem("textSizeModifier");
      localStorage.removeItem("brightnessIndex");
      localStorage.removeItem("simplify");
    }
  }, [isUserLoggedIn]);

  // Save modifications to local storage when not logged in.
  useEffect(() => {
    if (!isUserLoggedIn) {
      localStorage.setItem("textSizeModifier", textSizeModifier.toString());
      localStorage.setItem("brightnessIndex", brightnessIndex.toString());
      localStorage.setItem("simplify", simplify.toString());
    }
    console.log(localStorage)
  }, [textSizeModifier, brightnessIndex, simplify, isUserLoggedIn]);

  // If user is logged in, sync state to the DB when each value changes.
  useEffect(() => {
    if (isUserLoggedIn) updateUserSettings({ textSizeModifier });
  }, [textSizeModifier, isUserLoggedIn]);

  useEffect(() => {
    if (isUserLoggedIn) updateUserSettings({ brightnessIndex });
  }, [brightnessIndex, isUserLoggedIn]);

  useEffect(() => {
    if (isUserLoggedIn) updateUserSettings({ simplify });
  }, [simplify, isUserLoggedIn]);

  // Handle selection changes
  useEffect(() => {
    switch (selectionIndex) {
      case 1:
        setTextSizeModifier(prev => prev * 1.25);
        break;
      case 2:
        setTextSizeModifier(prev => prev / 1.25);
        break;
      case 3:
        setSimplify(prev => !prev);
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
      {location.pathname !== "/onboarding" && (
        <Header
          setBrightnessIndex={setBrightnessIndex}
          brightnessIndex={brightnessIndex}
          simplify={simplify}
          setSimplify={setSimplify}
          isUserLoggedIn={isUserLoggedIn}
          setIsUserLoggedIn={setIsUserLoggedIn}
        />
      )}
      {location.pathname !== "/onboarding" && location.pathname === "/" && (
        <Background
          showRender={true}
          setBrightnessIndex={setBrightnessIndex}
          brightnessIndex={brightnessIndex}
          simplify={simplify}
        />
      )}
      {location.pathname !== "/onboarding" && location.pathname !== "/" && (
        <Background
          showRender={false}
          setBrightnessIndex={setBrightnessIndex}
          brightnessIndex={brightnessIndex}
          simplify={simplify}
        />
      )}

      <div className="body">
        <Outlet context={{
          selectionIndex,
          setSelectionIndex,
          textSizeModifier,
          brightnessIndex,
          setBrightnessIndex,
          simplify,
          setSimplify,
          setIsUserLoggedIn,
          isUserLoggedIn,
        }} />
      </div>
    </React.StrictMode>
  );
}

export default App;
