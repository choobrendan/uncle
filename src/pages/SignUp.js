import React, { useState } from 'react';
import { useOutletContext, useNavigate } from 'react-router-dom';
import { createClient } from '@supabase/supabase-js'
import VoiceButton from '../components/VoiceButton';
import TextInput from '../components/TextInput';
import "./SignIn.css"

const supabaseUrl = "https://hgatxkpmrskbdqigenav.supabase.co";
const supabaseKey = process.env.REACT_APP_SUPABASE_KEY;
const supabase = createClient(supabaseUrl, supabaseKey);

function SignUp() {
  const {
    selectionIndex,
    setSelectionIndex,
    textSizeModifier,
    brightnessIndex,
    setBrightnessIndex
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
      // Check if the email already exists in the database
      const { data: existingUser, error: fetchError } = await supabase
        .from("userData")
        .select()
        .eq('email', email); // Check if email already exists

      if (fetchError) {
        console.error("Error fetching user data:", fetchError);
        return { data: null, error: fetchError };
      }

      // If user with this email exists, handle it accordingly
      if (existingUser.length > 0) {
        setErrorMessage('Email is already in use. Please <a href="/signin" class="link-button">LOG IN</a>.');
        return;
      }

      // If email does not exist, insert new user
      const { data: userDataFetch, error: errorDataFetch } = await supabase
        .from("userData")
        .select(); // Fetching user data to determine new ID

      console.log(userDataFetch, "userData");
      if (errorDataFetch) {
        console.error("Error fetching user data:", errorDataFetch);
        return { data: null, error: errorDataFetch };
      }

      let userDataId = userDataFetch?.slice(-1)[0]?.id + 1 ?? 0;
      if (!userDataId) {
        userDataId = 0;
      }

      // Insert the new user with the email
      const { data: emailData, error: errorData } = await supabase
        .from("userData")
        .upsert({ email: email, id: userDataId });

      if (errorData) {
        console.error("Error inserting user data:", errorData);
        return { data: null, error: errorData };
      }

      console.log("Inserted user with ID:", userDataId);
      navigate('/onboarding');
    } catch (error) {
      setErrorMessage(error.message.includes('Invalid login credentials') 
        ? 'Invalid email' 
        : 'Sign-in failed. Please try again.');
    }
  };

  return (
    <div className="signInPage">
      <div className="signInCard">
        <h2 style={{ fontSize: "36px", fontWeight: "300" }}>Create New Account</h2>
        <p style={{ fontSize: "20px", fontWeight: "300" }}>Join our community to save your preferences and settings!</p>

        <form style={{ 
          width: "100%", 
          justifyContent: "spaceBetween",
          justifyContent: "center"
        }} onSubmit={handleSignUp}>
          <div style={{ display: "flex", width: "100%", justifyContent: "center", textAlign: "left", padding: "10px" }}>
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
          {successMessage && <p style={{ color: 'green' }}>{successMessage}</p>}

          <button type="submit">Sign Up</button>
        </form>

        <div>
          <p>Already have an account? <a href="/signin" class="link-button"><router-link to="/signin">LOG IN</router-link></a></p>
        </div>
      </div>
      <VoiceButton setSelectionIndex={setSelectionIndex} selectionIndex={selectionIndex} />
    </div>
  );
}

export default SignUp;