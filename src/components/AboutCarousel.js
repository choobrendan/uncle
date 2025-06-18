import React from "react";
import AboutCard from "../components/AboutCard";
import "./AboutCarousel.css";

const AboutCarousel = ({ textSizeModifier, simplify }) => {
  const items = [
    {
      name: "Pre-Use Issue Detection",
      description:
        "A series of tests designed to identify potential user issues such as sight, language, and motor skills before interacting with the website.",
      index: 0,
    },
    {
      name: "In-Use Issue Detection",
      description:
        "Monitors real-time data like mouse and eye movements to detect and classify any issues users may face during their interaction with the website.",
      index: 1,
    },
    {
      name: "Vague Prompt Search Function",
      description:
        "Matches user input to existing commands or prompts using natural language processing to find the closest match.",
      index: 2,
    },
    {
      name: "Website Customisation Function",
      description:
        "Adjusts the website's layout, font size, and components based on the user's needs and preferences to ensure accessibility and convenience.",
      index: 3,
    },
    {
      name: "App Guidance Function",
      description:
        " Provides assistance by remembering past actions, predicting user behavior, and explaining actions to help users navigate and interact with the app more effectively.",
      index: 4,
    },
  ];

  // Simplified styles for carousel
  const simplifiedStyles = {
    carouselContainer: {
      display: "flex",
      flexDirection: "row",
      width: "100%",
      maxWidth: "100%",
      gap: "20px",
      padding: "100px",
    },
  };

  // Original styles
  const originalStyles = {
    aboutCard: {
      display: "flex",
      flexDirection: "row",
      overflow: "auto",
      padding: "10px",
      marginLeft: "15vw",
      paddingBottom: "15px",
    },
  };

  return (
    <div
      className={simplify ? "" : "aboutCard"}
      style={
        simplify ? simplifiedStyles.carouselContainer : originalStyles.aboutCard
      }
    >
      {items.map((item, index) => (
        <AboutCard
          key={index}
          item={item}
          textSizeModifier={textSizeModifier}
          simplify={simplify}
        />
      ))}
    </div>
  );
};

export default AboutCarousel;
