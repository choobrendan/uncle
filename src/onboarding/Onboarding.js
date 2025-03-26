import React, { useState, useEffect, useRef } from "react";
import Basketball from "./Basketball";
import "./Onboarding.css";
import CommunityCenter from "./Community";
import Survey from "./Survey";
import { createClient } from "@supabase/supabase-js";
const supabaseUrl = "https://hgatxkpmrskbdqigenav.supabase.co";
const supabaseKey = process.env.REACT_APP_SUPABASE_KEY;
const supabase = createClient(supabaseUrl, supabaseKey);
const Onboarding = () => {
  const [latestData, setLatestData] = useState({
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

  const mousePosition = useRef({ x: 0, y: 0 });
  const isMouseDown = useRef(false);
  const lastHoverElement = useRef(null);
  const [nextGame, setNextGame] = useState(0

  );
  const eyeMovement = useRef({ x: null, y: null });
  const lastScrollPosition = useRef(0);
  const scrollDirection = useRef("none");

  const [responses, setResponses] = useState({});


  // useEffect(() => {
  //   const initializeWebGazer = async () => {
  //     console.log("started");
  
  //     const webgazer = await window.webgazer
  //       .setGazeListener((data) => {
  //         if (data) {
  //           eyeMovement.current = {
  //             x: data.x,
  //             y: data.y,
  //           };
  //         }
  //       })
  //       .begin();
  
  //     webgazer.showPredictionPoints(false);
  
  //     // Use requestIdleCallback to run gaze tracking in idle time
  //     const trackGaze = () => {
  //       webgazer.getCurrentPrediction();
  //     };
  
  //     if ('requestIdleCallback' in window) {
  //       requestIdleCallback(trackGaze);
  //     } else {
  //       setInterval(trackGaze, 50); // fallback for unsupported browsers
  //     }
  //   };
  
  //   initializeWebGazer();
  
  //   return () => {

  //   };
  // }, []);

  const lastScrollTime = useRef(Date.now());
  
useEffect(() => {
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

    // Add scroll direction tracking
    const getScrollDirection = () => {
      const currentScrollPosition = window.scrollY;
      if (currentScrollPosition > lastScrollPosition.current) {
        scrollDirection.current = "down";
      } else if (currentScrollPosition < lastScrollPosition.current) {
        scrollDirection.current = "up";
      }
      lastScrollPosition.current = currentScrollPosition;
      lastScrollTime.current = Date.now(); // Update last scroll time
    };

    // Add event listeners
    document.addEventListener("mousemove", handleMouseMove);
    document.addEventListener("mousedown", handleMouseDown);
    document.addEventListener("mouseup", handleMouseUp);
    window.addEventListener("scroll", getScrollDirection);

    const interval = setInterval(() => {
      // Check if scrolling has stopped (100ms threshold)
      const timeSinceLastScroll = Date.now() - lastScrollTime.current;
      if (timeSinceLastScroll > 100) {
        scrollDirection.current = "none";
      }

      const timestamp = new Date().toISOString();
      const currentElement = document.elementFromPoint(
        mousePosition.current.x,
        mousePosition.current.y
      );

      const newEntry = {
        time: timestamp,
        positionX: mousePosition.current.x,
        positionY: mousePosition.current.y,
        hoverType: getHoverType(currentElement),
        isMouseDown: isMouseDown.current,
        eyeX: eyeMovement.current.x,
        eyeY: eyeMovement.current.y,
        scrollDirection: scrollDirection.current,
        currentState: nextGame
      };

      dataRef.current.timeseries = [...dataRef.current.timeseries, newEntry];
      setLatestData({ ...dataRef.current });
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

  const handleDownload = () => {
    const jsonString = JSON.stringify(dataRef.current, null, 2);
    const blob = new Blob([jsonString], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `interaction-data-${new Date().toISOString()}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const overwriteTable = async () => {
    try {
      for (let key in responses) {
        if (dataRef.current.hasOwnProperty(key)) {
          console.log(dataRef.current);
          dataRef.current[key] = responses[key];
        }
      }
      console.log(dataRef.current);
      const { timeseries, ...testDataFields } = dataRef.current;

      const { data: dataDataFetch, error: errorDataFetch } = await supabase
        .from("testData")
        .select(); // Using supabase instance directly, no need for 'await supabase()'

      console.log(dataDataFetch, "dataData");
      if (errorDataFetch) {
        console.error("Error fetching testData:", errorDataFetch);
        return { data: null, error: errorDataFetch };
      }

      let testDataId = dataDataFetch?.slice(-1)[0]?.id + 1 ?? 0;
      if (!testDataId) {
        testDataId = 0;
      }

      // Insert or update the testData table
      const { data: dataData, error: errorData } = await supabase
        .from("testData")
        .upsert({ ...testDataFields, id: testDataId });

      if (errorData) {
        console.error("Error inserting testData:", errorData);
        return { data: null, error: errorData };
      }

      console.log("Inserted testDataId:", testDataId);

      // Handle timeseries data
      const { data: dataTime, error: errorTime } = await supabase
        .from("testTime")
        .upsert(
          timeseries.map((item) => ({
            ...item,
            testDataId: testDataId,
          }))
        );

      if (errorTime) {
        console.error("Error inserting testTime:", errorTime);
        return { data: null, error: errorTime };
      }

      return { dataData, errorData, dataTime, errorTime };
    } catch (error) {
      console.error("Unexpected error:", error);
      return { data: null, error };
    }
  };

  return (
    <div className="app-container">

      {nextGame === 0 && <Basketball setNextGame={setNextGame} />}
      {nextGame === 1 && <CommunityCenter setNextGame={setNextGame} />}
      {nextGame === 2 && (
        <Survey
          responses={responses}
          setResponses={setResponses}
          setNextGame={setNextGame}
        />
      )}
    </div>
  );
};

export default Onboarding;
