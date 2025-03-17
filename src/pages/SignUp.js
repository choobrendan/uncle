import React, { useState } from 'react';
import { useOutletContext } from 'react-router-dom';
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

  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [errorMessage, setErrorMessage] = useState('');
  const [successMessage, setSuccessMessage] = useState('');

  const handleSignUp = async (e) => {
    e.preventDefault();
    setErrorMessage('');
    setSuccessMessage('');

    try {
      const { data, error } = await supabase.auth.signUp({
        email,
        password,
      });

      if (error) throw error;

      setSuccessMessage('Success!');
      setEmail('');
      setPassword('');

    } catch (error) {
      setErrorMessage(error.message);
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

          <div style={{ display: "flex", width: "100%", textAlign: "left", justifyContent: "center", padding: "10px" }}>
            <h2 style={{ width: "20%" }}>Password</h2>
            <TextInput
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="Create a password"
            />
          </div>

          {errorMessage && <p style={{ color: 'red' }}>{errorMessage}</p>}
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