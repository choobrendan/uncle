import React, { useState, useEffect } from "react";
import { useOutletContext, useNavigate } from "react-router-dom";
import { createClient } from "@supabase/supabase-js";
import VoiceButton from "../components/VoiceButton";
import TextInput from "../components/TextInput";
import "./SignIn.css";

const supabaseUrl = "https://hgatxkpmrskbdqigenav.supabase.co";
const supabaseKey = process.env.REACT_APP_SUPABASE_KEY;
const supabase = createClient(supabaseUrl, supabaseKey);

function SignIn() {
  const {
    selectionIndex,
    setSelectionIndex,
    textSizeModifier,
    brightnessIndex,
    setBrightnessIndex,
    simplify,
    setSimplify,
    setIsUserLoggedIn,
  } = useOutletContext();


  const navigate = useNavigate();
  const [email, setEmail] = useState("");
  const [errorMessage, setErrorMessage] = useState("");
  const [isLoading, setIsLoading] = useState(true);

  const handleSignIn = async (e) => {
    e.preventDefault();
    setErrorMessage("");

    try {
      const { data: existingUser, error: fetchError } = await supabase
        .from("userData")
        .select()
        .eq("email", email);

      if (fetchError) {
        console.error("Error fetching user data:", fetchError);
        setErrorMessage(
          "An error occurred while checking the email. Please try again."
        );
        return;
      }

      if (existingUser.length === 0) {
        setErrorMessage("Email not found. Please SIGN UP.");
        return;
      }

      console.log("User found, proceed with sign-in logic");
      setIsUserLoggedIn(email);
      navigate("/");
    } catch (error) {
      setErrorMessage("Sign-in failed. Please try again.");
      console.error("Sign-in error:", error);
    }
  };

  // Simplified styles
  const simplifiedStyles = {
    container: {
      display: "flex",
      justifyContent: "center",
      alignItems: "center",
      height: "100vh",
      backgroundColor: "#fff",
      fontFamily: "Arial, sans-serif",
    },
    card: {
      width: "400px",
      padding: "30px",
      backgroundColor: "#f8f8f8",
      borderRadius: "8px",
      boxShadow: "0 2px 10px rgba(0, 0, 0, 0.1)",
      textAlign: "center",
    },
    title: {
      fontSize: `${28 * textSizeModifier}px`,
      fontWeight: "bold",
      color: "#333",
      marginBottom: "15px",
    },
    description: {
      fontSize: `${16 * textSizeModifier}px`,
      color: "#555",
      marginBottom: "30px",
    },
    formGroup: {
      marginBottom: "20px",
      textAlign: "left",
    },
    label: {
      display: "block",
      fontSize: `${16 * textSizeModifier}px`,
      fontWeight: "bold",
      color: "#333",
      marginBottom: "8px",
    },
    input: {
      width: "100%",
      padding: "12px",
      fontSize: `${16 * textSizeModifier}px`,
      border: "1px solid #ddd",
      borderRadius: "4px",
      backgroundColor: "#fff",
    },
    errorMessage: {
      color: "#d32f2f",
      fontSize: `${14 * textSizeModifier}px`,
      marginBottom: "15px",
      textAlign: "left",
      fontWeight: "bold",
    },
    button: {
      backgroundColor: "#2196f3",
      color: "white",
      border: "none",
      borderRadius: "4px",
      padding: "12px 24px",
      fontSize: `${16 * textSizeModifier}px`,
      fontWeight: "bold",
      cursor: "pointer",
      width: "100%",
      marginBottom: "20px",
    },
    linkText: {
      fontSize: `${14 * textSizeModifier}px`,
      color: "#555",
    },
    link: {
      color: "#2196f3",
      cursor: "pointer",
      fontWeight: "bold",
      textDecoration: "none",
    },
  };

  if (simplify) {
    return (
      <div style={simplifiedStyles.container}>
        <div style={simplifiedStyles.card}>
          <h2 style={simplifiedStyles.title}>Log In to Your Account</h2>
          <p style={simplifiedStyles.description}>
            Ensure the changes made to your website are remembered across
            different devices!
          </p>

          <form onSubmit={handleSignIn}>
            <div style={simplifiedStyles.formGroup}>
              <label htmlFor="email" style={simplifiedStyles.label}>
                Email
              </label>
              <input
                id="email"
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="Enter your email..."
                style={simplifiedStyles.input}
                required
              />
            </div>

            {errorMessage && (
              <div style={simplifiedStyles.errorMessage}>{errorMessage}</div>
            )}

            <button type="submit" style={simplifiedStyles.button}>
              Sign In
            </button>
          </form>

          <p style={simplifiedStyles.linkText}>
            Haven't got an account?{" "}
            <span
              style={simplifiedStyles.link}
              onClick={() => navigate("/signup")}
            >
              SIGN UP
            </span>{" "}
            Now!
          </p>
        </div>

        <VoiceButton
          setSelectionIndex={setSelectionIndex}
          selectionIndex={selectionIndex}
        />
      </div>
    );
  }

  // Original version
  return (
    <div
      className="signInPage"
      style={{ filter: `brightness(${1 * brightnessIndex}` }}
    >
      <div className="signInCard">
        <h2 style={{ fontSize: `${36 * textSizeModifier}px` }}>
          Log In to Your Account
        </h2>

        <p style={{ fontSize: `${18 * textSizeModifier}px` }}>
          Ensure the changes made to your website are remembered across
          different devices!
        </p>

        <form onSubmit={handleSignIn}>
          <div style={{ marginBottom: "20px", textAlign: "left" }}>
            <label
              htmlFor="email"
              style={{
                display: "block",
                fontSize: `${18 * textSizeModifier}px`,
                color: "#c3ff9e",
                textShadow: "0 0 5px #c3ff9e80",
                marginBottom: "8px",
              }}
            >
              Email
            </label>
            <TextInput
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="Enter your email..."
              style={{ fontSize: `${16 * textSizeModifier}px` }}
            />
          </div>

          {errorMessage && (
            <div
              style={{
                color: "#ff6b6b",
                fontSize: `${16 * textSizeModifier}px`,
                marginBottom: "15px",
                textShadow: "0 0 5px rgba(255, 107, 107, 0.5)",
              }}
            >
              {errorMessage}
            </div>
          )}

          <button
            type="submit"
            className="scifi-button"
            style={{
              fontSize: `${18 * textSizeModifier}px`,
              width: "100%",
              marginTop: "10px",
              cursor: "pointer",
            }}
          >
            Sign In
          </button>
        </form>

        <p
          style={{
            marginTop: "30px",
            fontSize: `${16 * textSizeModifier}px`,
            color: "#c3ff9e",
            textShadow: "0 0 5px #c3ff9e80",
          }}
        >
          {" "}
          Haven't got an account?{" "}
          <button
            className="link-button"
            onClick={() => navigate("/signup")}
            style={{
              fontSize: `${16 * textSizeModifier}px`,
              cursor: "pointer",
              fontWeight: "bold",
            }}
          >
            SIGN UP
          </button>{" "}
          Now!
        </p>
      </div>

      <VoiceButton
        setSelectionIndex={setSelectionIndex}
        selectionIndex={selectionIndex}
      />
    </div>
  );
}

export default SignIn;
