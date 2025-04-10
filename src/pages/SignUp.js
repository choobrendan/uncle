import React, { useState } from 'react';
import { useOutletContext, useNavigate } from 'react-router-dom';
import { createClient } from '@supabase/supabase-js';
import VoiceButton from '../components/VoiceButton';
import TextInput from '../components/TextInput';
import "./SignIn.css";

const supabaseUrl = "https://hgatxkpmrskbdqigenav.supabase.co";
const supabaseKey = process.env.REACT_APP_SUPABASE_KEY;
const supabase = createClient(supabaseUrl, supabaseKey);

function SignUp() {
  const {
    selectionIndex,
    setSelectionIndex,
    textSizeModifier,
    brightnessIndex,
    setBrightnessIndex,
    simplify,
    setIsUserLoggedIn,
    isUserLoggedIn,
  } = useOutletContext();
  
  const navigate = useNavigate();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [errorMessage, setErrorMessage] = useState('');
  const [successMessage, setSuccessMessage] = useState('');

  const handleSignUp = async (e) => {
    e.preventDefault();
    setErrorMessage('');

    try {

      const { data: existingUser, error: fetchError } = await supabase
        .from("userData")
        .select()
        .eq('email', email); 

      if (fetchError) {
        console.error("Error fetching user data:", fetchError);
        return { data: null, error: fetchError };
      }

      if (existingUser.length > 0) {
        setErrorMessage('Email is already in use. Please LOG IN.');
        return;
      }

      const { data: userDataFetch, error: errorDataFetch } = await supabase
        .from("userData")
        .select();

      console.log(userDataFetch, "userData");
      if (errorDataFetch) {
        console.error("Error fetching user data:", errorDataFetch);
        return { data: null, error: errorDataFetch };
      }

      let userDataId = userDataFetch?.slice(-1)[0]?.id + 1 ?? 0;
      if (!userDataId) {
        userDataId = 0;
      }

      const { data: emailData, error: errorData } = await supabase
        .from("userData")
        .upsert({ email: email, id: userDataId });

      if (errorData) {
        console.error("Error inserting user data:", errorData);
        return { data: null, error: errorData };
      }

      console.log("Inserted user with ID:", userDataId);
      setIsUserLoggedIn(email)
      navigate('/onboarding');
    } catch (error) {
      setErrorMessage(error.message.includes('Invalid login credentials') 
        ? 'Invalid email' 
        : 'Sign-up failed. Please try again.');
    }
  };

  // Simplified styles
  const simplifiedStyles = {
    container: {
      display: 'flex',
      justifyContent: 'center',
      alignItems: 'center',
      height: '100vh',
      backgroundColor: '#fff',
      fontFamily: 'Arial, sans-serif'
    },
    card: {
      width: '400px',
      padding: '30px',
      backgroundColor: '#f8f8f8',
      borderRadius: '8px',
      boxShadow: '0 2px 10px rgba(0, 0, 0, 0.1)',
      textAlign: 'center'
    },
    title: {
      fontSize: `${28 * textSizeModifier}px`,
      fontWeight: 'bold',
      color: '#333',
      marginBottom: '15px'
    },
    description: {
      fontSize: `${16 * textSizeModifier}px`,
      color: '#555',
      marginBottom: '30px'
    },
    formGroup: {
      marginBottom: '20px',
      textAlign: 'left'
    },
    label: {
      display: 'block',
      fontSize: `${16 * textSizeModifier}px`,
      fontWeight: 'bold',
      color: '#333',
      marginBottom: '8px'
    },
    input: {
      width: '100%',
      padding: '12px',
      fontSize: `${16 * textSizeModifier}px`,
      border: '1px solid #ddd',
      borderRadius: '4px',
      backgroundColor: '#fff'
    },
    errorMessage: {
      color: '#d32f2f',
      fontSize: `${14 * textSizeModifier}px`,
      marginBottom: '15px',
      textAlign: 'left',
      fontWeight: 'bold'
    },
    successMessage: {
      color: '#4caf50',
      fontSize: `${14 * textSizeModifier}px`,
      marginBottom: '15px',
      textAlign: 'left',
      fontWeight: 'bold'
    },
    button: {
      backgroundColor: '#2196f3',
      color: 'white',
      border: 'none',
      borderRadius: '4px',
      padding: '12px 24px',
      fontSize: `${16 * textSizeModifier}px`,
      fontWeight: 'bold',
      cursor: 'pointer',
      width: '100%',
      marginBottom: '20px'
    },
    linkText: {
      fontSize: `${14 * textSizeModifier}px`,
      color: '#555'
    },
    link: {
      color: '#2196f3', 
      cursor: 'pointer',
      fontWeight: 'bold',
      textDecoration: 'none'
    }
  };

  // Render simplified version
  if (simplify) {
    return (
      <div style={simplifiedStyles.container}>
        <div style={simplifiedStyles.card}>
          <h2 style={simplifiedStyles.title}>Create New Account</h2>
          <p style={simplifiedStyles.description}>
            Join our community to save your preferences and settings!
          </p>
          
          <form onSubmit={handleSignUp}>
            <div style={simplifiedStyles.formGroup}>
              <label htmlFor="email" style={simplifiedStyles.label}>Email</label>
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
              <div style={simplifiedStyles.errorMessage}>
                {errorMessage} {errorMessage.includes('LOG IN') && 
                  <span 
                    style={simplifiedStyles.link}
                    onClick={() => navigate('/signin')}
                  >
                    LOG IN
                  </span>
                }
              </div>
            )}

            {successMessage && (
              <div style={simplifiedStyles.successMessage}>
                {successMessage}
              </div>
            )}

            <button type="submit" style={simplifiedStyles.button}>
              Sign Up
            </button>
          </form>

          <p style={simplifiedStyles.linkText}>
            Already have an account?{" "}
            <span 
              style={simplifiedStyles.link}
              onClick={() => navigate('/signin')}
            >
              LOG IN
            </span>
          </p>
        </div>
        
        <VoiceButton setSelectionIndex={setSelectionIndex} selectionIndex={selectionIndex} />
      </div>
    );
  }

  // Render original version
  return (
    <div className="signInPage" style={{ filter: `brightness(${1 * brightnessIndex})` }}>
      <div className="signInCard">
        <h2 style={{ fontSize: `${36 * textSizeModifier}px`, fontWeight: "300" }}>Create New Account</h2>
        <p style={{ fontSize: `${20 * textSizeModifier}px`, fontWeight: "300" }}>Join our community to save your preferences and settings!</p>

        <form style={{ 
          width: "100%", 
          justifyContent: "center"
        }} onSubmit={handleSignUp}>
          <div style={{ display: "flex", width: "100%", justifyContent: "center", textAlign: "left", padding: "10px" }}>
            <h2 style={{ width: "20%", fontSize: `${20 * textSizeModifier}px` }}>Email</h2>
            <TextInput
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="Enter your email..."
              style={{ fontSize: `${16 * textSizeModifier}px` }}
            />
          </div>

          {errorMessage && (
            <div style={{ 
              color: "#ff6b6b", 
              fontSize: `${16 * textSizeModifier}px`, 
              marginBottom: '15px',
              textShadow: '0 0 5px rgba(255, 107, 107, 0.5)'
            }}>
              {errorMessage.includes('LOG IN') ? (
                <>
                  Email is already in use. Please{" "}
                  <button 
                    className="link-button" 
                    onClick={() => navigate('/signin')}
                    style={{ 
                      fontSize: `${16 * textSizeModifier}px`,
                      cursor: 'pointer',
                      fontWeight: 'bold' 
                    }}
                  >
                    LOG IN
                  </button>
                </>
              ) : errorMessage}
            </div>
          )}
          
          {successMessage && (
            <p style={{ 
              color: '#7dff8e', 
              fontSize: `${16 * textSizeModifier}px`, 
              textShadow: '0 0 5px rgba(125, 255, 142, 0.5)'
            }}>
              {successMessage}
            </p>
          )}

          <button 
            type="submit" 
            className="scifi-button"
            style={{ 
              fontSize: `${18 * textSizeModifier}px`,
              width: '100%',
              marginTop: '10px',
              cursor: 'pointer'
            }}
          >
            Sign Up
          </button>
        </form>

        <div>
          <p style={{ 
            marginTop: '30px', 
            fontSize: `${16 * textSizeModifier}px`,
            color: '#c3ff9e',
            textShadow: '0 0 5px #c3ff9e80'
          }}>
            Already have an account?{" "}
            <button 
              className="link-button" 
              onClick={() => navigate('/signin')}
              style={{ 
                fontSize: `${16 * textSizeModifier}px`,
                cursor: 'pointer',
                fontWeight: 'bold' 
              }}
            >
              LOG IN
            </button>
          </p>
        </div>
      </div>
      <VoiceButton setSelectionIndex={setSelectionIndex} selectionIndex={selectionIndex} />
    </div>
  );
}

export default SignUp;