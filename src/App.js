import React, { useState, useEffect, useRef } from "react";
import { Outlet, useNavigate, useLocation } from "react-router-dom";
import { createClient } from "@supabase/supabase-js";
import Header from "./components/Header";
import Background from "./components/Background";
import localforage from "localforage";
const supabaseUrl = "https://hgatxkpmrskbdqigenav.supabase.co";
const supabaseKey = process.env.REACT_APP_SUPABASE_KEY;
const supabase = createClient(supabaseUrl, supabaseKey);

function App() {
  const [latestData, setLatestData] = useState([]);

  const dataRef = useRef({
    timeseries: [],
    question1: "",
    question2: "",
    question3: "",
    question4: "",
    question5: "",
    question6: "",
    question7: "",
    question8: "",
  });
  const location = useLocation();
  const navigate = useNavigate();
  const [gazeX, setGazeX] = useState(0);
  const [gazeY, setGazeY] = useState(0);
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
  const lastGaze = useRef(null);
  const lastTime = useRef(null);
  const mousePosition = useRef({ x: 0, y: 0 });
  const isMouseDown = useRef(false);
  const lastHoverElement = useRef(null);
  const [nextGame, setNextGame] = useState(0);
  const lastScrollPosition = useRef(0);
  const scrollDirection = useRef("none");
  const [responses, setResponses] = useState({});

  // -- Updated WebGazer setup and gaze listener --------------------------------

  useEffect(() => {
    const eyeListener = (data, clock) => {
      if (!lastTime.current) {
        lastTime.current = clock;
      }

      const duration = clock - lastTime.current;
      const entry = {
        time: new Date().toISOString(),
        positionX:
          lastGaze.current && typeof lastGaze.current.x === "number"
            ? Math.floor(lastGaze.current.x)
            : -1,

        positionY:
          lastGaze.current && typeof lastGaze.current.y === "number"
            ? Math.floor(lastGaze.current.y)
            : -1,

        eyeX: data?.x || null,
        eyeY: data?.y || null,
        hoverType: lastHoverElement.current
          ? getHoverType(lastHoverElement.current)
          : "none",
        isMouseDown: isMouseDown.current,
        scrollDirection: scrollDirection.current,
      };

      if (dataRef.current.timeseries.length > 100) {
        dataRef.current.timeseries.shift();
      }
      dataRef.current.timeseries.push(entry);
      console.log(dataRef.current.timeseries);
      lastGaze.current = data;
      lastTime.current = clock;
    };

    const initializeWebGazer = async () => {
      if (!window.saveDataAcrossSessions) {
        await localforage.setItem("webgazerGlobalData", null);
        await localforage.setItem("webgazerGlobalSettings", null);
      }
      const webgazerInstance = await window.webgazer
        .setRegression("ridge")
        .setTracker("TFFacemesh")
        .begin();

      webgazerInstance
        .showVideoPreview(true)
        .showPredictionPoints(true)
        .applyKalmanFilter(true);

      window.webgazer.setGazeListener(eyeListener);
    };

    initializeWebGazer();

    return () => {
      if (window.webgazer) {
        try {
          window.webgazer.end();
        } catch (err) {
          console.warn("Error ending WebGazer:", err);
        }
      }
    };
  }, []);

  // ------------------------------------------------------------------------------

  useEffect(() => {
    // Mouse and scroll tracking (unchanged)...
    const handleMouseMove = (e) => {
      mousePosition.current = { x: e.clientX, y: e.clientY };
      lastHoverElement.current = document.elementFromPoint(
        e.clientX,
        e.clientY
      );
    };

    const handleMouseDown = () => {
      isMouseDown.current = true;
      lastHoverElement.current = document.elementFromPoint(
        mousePosition.current.x,
        mousePosition.current.y
      );
    };

    const handleMouseUp = () => {
      isMouseDown.current = false;
      lastHoverElement.current = document.elementFromPoint(
        mousePosition.current.x,
        mousePosition.current.y
      );
    };

    document.addEventListener("mousemove", handleMouseMove);
    document.addEventListener("mousedown", handleMouseDown);
    document.addEventListener("mouseup", handleMouseUp);

    const getScrollDirection = () => {
      const current = window.scrollY;
      if (current > lastScrollPosition.current)
        scrollDirection.current = "down";
      else if (current < lastScrollPosition.current)
        scrollDirection.current = "up";
      else scrollDirection.current = "none";
      lastScrollPosition.current = current;
    };

    const interval = setInterval(() => {
      getScrollDirection();
      // mouse-based entry (optionally keep or remove)...
    }, 50);

    return () => {
      clearInterval(interval);
      document.removeEventListener("mousemove", handleMouseMove);
      document.removeEventListener("mousedown", handleMouseDown);
      document.removeEventListener("mouseup", handleMouseUp);
      window.removeEventListener("scroll", getScrollDirection);
    };
  }, []);

  const getHoverType = (element) => {
    if (!element) return "none";
    const role = element.getAttribute?.("role") || "";
    const classList = element.classList?.toString().toLowerCase() || "";
    const tagName = element.tagName?.toLowerCase() || "";
    const inputType = element.type?.toLowerCase() || "";
    const parent = element.parentElement;

    if (
      role === "button" ||
      classList.includes("btn") ||
      classList.includes("button")
    )
      return "button";
    if (role === "link" || tagName === "a") return "link";
    if (classList.includes("icon")) return "icon";
    if (tagName === "input") {
      if (inputType === "search" || classList.includes("search"))
        return "search-bar";
      if (inputType === "checkbox") return "checkbox";
      if (inputType === "radio") return "radio";
      return "text-input";
    }
    if (tagName === "textarea") return "text-area";
    if (tagName === "select") return "dropdown";
    if (role === "navigation" || classList.includes("nav"))
      return "navigation-container";
    if (classList.includes("navbar-item") || classList.includes("nav-item"))
      return "navbar-item";
    if (classList.includes("breadcrumb")) return "breadcrumb";
    if (classList.includes("pagination")) return "pagination";
    if (["h1", "h2", "h3", "h4", "h5", "h6"].includes(tagName)) return "text";
    if (tagName === "p") return "text";
    if (classList.includes("caption")) return "text";
    if (tagName === "button") return "button";
    if (tagName === "img") return classList.includes("icon") ? "icon" : "image";
    if (tagName === "video") return "video";
    if (tagName === "audio") return "audio";
    if (["header", "footer", "aside", "main", "section"].includes(tagName))
      return "layout-container";
    if (classList.includes("card")) return "card-container";
    if (classList.includes("modal") || classList.includes("dialog"))
      return "modal";
    if (classList.includes("tooltip")) return "tooltip";
    if (tagName === "li") return "list-item";
    if (tagName === "tr") return "table-row";
    if (tagName === "td") return "table-cell";
    if (classList.includes("spinner")) return "loader";
    if (classList.includes("progress")) return "progress-indicator";
    if (classList.includes("badge")) return "status-badge";
    if (parent) {
      const parentType = getHoverType(parent);
      if (parentType === "button") return "button-text";
      if (parentType === "link") return "link-text";
      if (parentType === "card") return "card-text";
    }
    if (classList.includes("text")) return "static-text";
    if (element.isContentEditable) return "editable-content";
    return "container";
  };
  // If user is not logged in, you could optionally load settings from local storage if they exist.
  useEffect(() => {
    if (!isUserLoggedIn) {
      const storedTextSize = localStorage.getItem("textSizeModifier");
      const storedBrightness = localStorage.getItem("brightnessIndex");
      const storedSimplify = localStorage.getItem("simplify");

      if (storedTextSize !== null)
        setTextSizeModifier(parseFloat(storedTextSize));
      if (storedBrightness !== null)
        setBrightnessIndex(parseFloat(storedBrightness));
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
    console.log(localStorage);
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
        setTextSizeModifier((prev) => prev * 1.25);
        break;
      case 2:
        setTextSizeModifier((prev) => prev / 1.25);
        break;
      case 3:
        setSimplify((prev) => !prev);
        break;
      case 5:
        setBrightnessIndex((prev) => prev * 1.1);
        break;
      case 6:
        setBrightnessIndex((prev) => prev / 1.1);
        break;
      case 7:
        navigate("/");
        break;
      case 8:
        navigate("/about");
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
          aboutRender={false}
          setBrightnessIndex={setBrightnessIndex}
          brightnessIndex={brightnessIndex}
          simplify={simplify}
          gazeX={gazeX}
          gazeY={gazeY}
        />
      )}
      {location.pathname === "/about" && (
        <Background
          showRender={false}
          aboutRender={true}
          setBrightnessIndex={setBrightnessIndex}
          brightnessIndex={brightnessIndex}
          simplify={simplify}
          gazeX={gazeX}
          gazeY={gazeY}
        />
      )}
      {location.pathname !== "/onboarding" &&
        location.pathname !== "/" &&
        location.pathname !== "about" && (
          <Background
            showRender={false}
            aboutRender={false}
            setBrightnessIndex={setBrightnessIndex}
            brightnessIndex={brightnessIndex}
            simplify={simplify}
            gazeX={gazeX}
            gazeY={gazeY}
          />
        )}

      <div className="body">
        <Outlet
          context={{
            selectionIndex,
            setSelectionIndex,
            textSizeModifier,
            brightnessIndex,
            setBrightnessIndex,
            simplify,
            setSimplify,
            setIsUserLoggedIn,
            isUserLoggedIn,
          }}
        />
      </div>
    </React.StrictMode>
  );
}

export default App;
