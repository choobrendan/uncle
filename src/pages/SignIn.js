import React, { useState } from 'react';
import { useOutletContext } from 'react-router-dom';
import { createClient } from '@supabase/supabase-js'
import VoiceButton from '../components/VoiceButton';
import "./SignIn.css"
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

  // State for user input
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [errorMessage, setErrorMessage] = useState('');
  const [showRender, setShowRender] = useState(true);

  const handleSignIn = async (e) => {
    e.preventDefault(); // Prevent page refresh on form submission
    setErrorMessage(''); // Reset error message

    try {
      // Attempt to sign in the user via Supabase
      const { data, error } = await supabase.auth.signInWithPassword({
        email,
        password,
      });

      if (error) {
        throw error;
      }

      // If successful, handle the signed-in user data
      console.log('Signed in:', data.user);

      // Redirect or update UI after successful login (you can use react-router here)
      // For example:
      // history.push('/dashboard');  // if you have a dashboard route

    } catch (error) {
      setErrorMessage(error.message); // Display error message if sign-in fails
    }
  };

  return (
    <div className="signInPage">
      <div className="signInCard">
      <h2>Sign In</h2>

      <form onSubmit={handleSignIn}>
        <div>
          <label>Email</label>
          <input
            type="email"
            placeholder="Enter your email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            required
          />
        </div>

        <div>
          <label>Password</label>
          <input
            type="password"
            placeholder="Enter your password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
          />
        </div>

        {errorMessage && <p style={{ color: 'red' }}>{errorMessage}</p>}

        <button type="submit">Sign In</button>
      </form>

      <VoiceButton setSelectionIndex={setSelectionIndex} selectionIndex={selectionIndex} />
      </div>
    </div>
  );
}

export default SignIn;
