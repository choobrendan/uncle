import React, { useState, useEffect } from 'react'; // Added useEffect
import { useOutletContext, useNavigate } from 'react-router-dom';
import { createClient } from '@supabase/supabase-js';
import VoiceButton from '../components/VoiceButton';
import TextInput from '../components/TextInput';
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
    setBrightnessIndex
  } = useOutletContext();

  const navigate = useNavigate();
  const [email, setEmail] = useState('');
  const [errorMessage, setErrorMessage] = useState('');
  const [isLoading, setIsLoading] = useState(true); // Added loading state


  const handleSignIn = async (e) => {
    e.preventDefault();
    setErrorMessage('');

    try {
      // Check if the email exists in the database
      const { data: existingUser, error: fetchError } = await supabase
        .from("userData")
        .select()
        .eq('email', email); // Check if email exists

      if (fetchError) {
        console.error("Error fetching user data:", fetchError);
        setErrorMessage('An error occurred while checking the email. Please try again.');
        return;
      }

      // If no user found with this email, show an error and navigate to sign-up page
      if (existingUser.length === 0) {
        setErrorMessage('Email not found. Please <a href="/signup" class="link-button">SIGN UP</a>.');
        return;
      }

      // If the email exists, proceed with the sign-in logic (e.g., set the user session)
      // You can handle this based on your sign-in requirements (e.g., password check)

      console.log("User found, proceed with sign-in logic");

      // Add sign-in logic here (e.g., validating password, redirecting to dashboard, etc.)
      // This might involve another API call or client-side logic.

    } catch (error) {
      setErrorMessage('Sign-in failed. Please try again.');
      console.error('Sign-in error:', error);
    }
  };
  return (
    <div className="signInPage">
      <div className="signInCard">
        <h2 style={{ fontSize: "36px", fontWeight: "300" }}>Log In to Your Account</h2>
        <p style={{ fontSize: "20px", fontWeight: "300" }}>
          Ensure the changes made to your website are remembered across different devices!
        </p>
        <form
          style={{
            width: "100%",
            justifyContent: "spaceBetween",
            justifyContent: "center",
          }}
          onSubmit={handleSignIn}
        >
          <div
            style={{
              display: "flex",
              width: "100%",
              justifyContent: "center",
              textAlign: "left",
              padding: "10px",
            }}
          >
            <h2 style={{ width: "20%" }}>Email</h2>
            <TextInput
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="Enter your email..."
            />
          </div>

          {errorMessage && (
        <div  style={{ color: "red" }} className="error-message" dangerouslySetInnerHTML={{ __html: errorMessage }} />
      )}


          <button
            type="submit"
            className="auth-button"
            style={{ marginTop: "1rem" }}
          >
            Sign In
          </button>
        </form>
        <div>
          <p>
            {" "}
            Haven't got an account?{" "}
            <a href="/signup" class="link-button">
              <router-link to="/signup">SIGN UP</router-link>
            </a>{" "}
            Now!
          </p>
        </div>
      </div>
      <VoiceButton
        setSelectionIndex={setSelectionIndex}
        selectionIndex={selectionIndex}
      />
    </div>
  );
}

export default SignIn;
