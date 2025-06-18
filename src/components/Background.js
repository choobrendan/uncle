import React, { useState, useEffect } from "react";
import FaceRender from "../components/FaceRender";
import HandRender from "../components/HandRender";
import "./Background.css";
const Background = ({
  showRender,
  brightnessIndex,
  simplify,
  gazeX,
  gazeY,
  aboutRender,
  font,
}) => {
  const [mouseX, setMouseX] = useState("");
  const [mouseY, setMouseY] = useState("");
  const handleMouseMove = (event) => {
    setMouseX(event.clientX.toString().padStart(4, "0"));
    setMouseY(event.clientY.toString().padStart(4, "0"));
  };

  useEffect(() => {
    // Add mouse move listener
    window.addEventListener("mousemove", handleMouseMove);

    // Cleanup the event listener when the component is unmounted
    return () => {
      window.removeEventListener("mousemove", handleMouseMove);
    };
  }, []);

  const bgStyles = {
    height: "100vh",
    position: "fixed",
    top: 0,
    width: "100vw",
    zIndex: -999,
    backgroundColor: !simplify ? "#000000" : "#231f01",
    filter: `brightness(${brightnessIndex})`,
    fontFamily: font,
  };

  return (
    <div style={bgStyles}>
      {!simplify && (
        <div>
          {" "}
          <div className="coordinates">
            <div className="mouse-coordinates">
              <div className="mouse-box">
                <p>Mouse X: {mouseX}</p>
              </div>
              <div className="mouse-box">
                <p>Mouse Y: {mouseY}</p>
              </div>
            </div>
            <div className="head-coordinates">
              {/* You can add additional gaze coordinates if required */}
            </div>
          </div>{" "}
          <div className="body">
            <div className="overlay o-top">
              <div className="grid top">
                <div className="grid-fade"></div>
                <div className="grid-lines"></div>
              </div>
            </div>

            <div className="o-mid"></div>

            <div className="overlay o-bottom">
              <div className="grid">
                <div className="grid-fade"></div>
                <div className="grid-lines"></div>
              </div>
            </div>
          </div>
          <div
            style={{
              position: "absolute",
              top: 0,
              height: "100vh",
              width: "100vw",
              display: "flex",
              flexDirection: "row",
              alignItems: "center",
              justifyContent: "center",
              zIndex: 9,
            }}
          >
            {/* {showRender && <HandRender />}
            {showRender && <FaceRender X={gazeX} Y={gazeY} />}
            {aboutRender === "about" && <FaceRender X={gazeX} Y={gazeY} />} */}
          </div>
        </div>
      )}
    </div>
  );
};

export default Background;
