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
  const [password, setPassword] = useState('');
  const [errorMessage, setErrorMessage] = useState('');
  const [isLoading, setIsLoading] = useState(true); // Added loading state

  // Check if user is already authenticated
  useEffect(() => {
    const checkAuth = async () => {
      try {
        const { data: { user } } = await supabase.auth.getUser();
        
        if (user) {
          navigate('/'); // Redirect if user is already logged in
        }
      } catch (error) {
        console.error('Auth check error:', error);
      } finally {
        setIsLoading(false);
      }
    };

    checkAuth();
  }, [navigate]);

  const handleSignIn = async (e) => {
    e.preventDefault();
    setErrorMessage('');

    try {
      const { data, error } = await supabase.auth.signInWithPassword({
        email,
        password,
      });

      if (error) throw error;

      console.log('Signed in:', data.user);
      setPassword('');
      navigate('/');
      
    } catch (error) {
      setErrorMessage(error.message.includes('Invalid login credentials') 
        ? 'Invalid email or password' 
        : 'Sign-in failed. Please try again.');
    }
  };

  if (isLoading) {
    return <div className="signInPage">Loading...</div>; // Loading state
  }
const check = async () => {
  const { data: { users }, error } = await supabase.auth.admin.listUsers()
  console.log(users)
}
check();

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

          <div
            style={{
              display: "flex",
              width: "100%",
              textAlign: "left",
              justifyContent: "center",
              padding: "10px",
            }}
          >
            <h2 style={{ width: "20%" }}>Password</h2>

            <TextInput
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="Enter your password"
              required
            />
          </div>

          {errorMessage && <p style={{ color: "red" }}>{errorMessage}</p>}

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
