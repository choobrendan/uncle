import React from 'react';
import { createClient } from '@supabase/supabase-js';

const supabaseUrl = "https://hgatxkpmrskbdqigenav.supabase.co";
const supabaseKey = process.env.REACT_APP_SUPABASE_KEY;
const supabase = createClient(supabaseUrl, supabaseKey);

const Header = ({ item, brightnessIndex, setBrightnessIndex, simplify, isUserLoggedIn, setIsUserLoggedIn }) => {

  async function handleSignOut() {
    setIsUserLoggedIn("");
    localStorage.removeItem("isUserLoggedIn");
  }

  const simplifiedStyles = {
    header: {
      width: '100%',
      position: 'fixed',
      top: 0,
      left: 0,
      zIndex: 999,
      backgroundColor: '#fff',
      boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
      fontFamily: 'Arial, sans-serif',
      padding: '10px 0'
    },
    centered: {
      display: 'flex',
      justifyContent: 'center',
      alignItems: 'center',
      padding: '10px 20px',
      position: 'relative'
    },
    navLink: {
      display: 'inline-block',
      padding: '10px 15px',
      margin: '0 10px',
      color: '#333',
      textDecoration: 'none',
      fontWeight: 'bold',
      fontSize: '16px',
      borderRadius: '4px',
      transition: 'background-color 0.3s ease'
    },
    navLinkHover: {
      backgroundColor: '#f5f5f5'
    },
    signInButton: {
      display: 'flex',
      position: 'absolute',
      right: '40px',
      padding: '10px 15px',
      backgroundColor: '#007bff',
      color: '#fff',
      borderRadius: '4px',
      textDecoration: 'none',
      fontWeight: 'bold',
      cursor: 'pointer'
    }
  };

  // Simplified header
  if (simplify) {
    return (
      <div style={simplifiedStyles.header}>
        <div style={simplifiedStyles.centered}>
          <a href="/about" style={simplifiedStyles.navLink}>About</a>
          <a href="/navigation" style={simplifiedStyles.navLink}>Navigation</a>
          <a href="/graph" style={simplifiedStyles.navLink}>Graph</a>

          {isUserLoggedIn === "" ? (
            <a href="/signin" style={simplifiedStyles.signInButton}>Sign In</a>
          ) : (
            <a onClick={handleSignOut} style={simplifiedStyles.signInButton}>Sign Out</a>
          )}
        </div>
      </div>
    );
  }

  // Sci-fi header
  return (
    <div className="header" style={{ filter: `brightness(${1 * brightnessIndex}` }}>
      <div className="centered">
        <a className="scifi-button" href="/about">About</a>
        <a className="scifi-button" href="/navigation">Navigation</a>
        <a className="scifi-button" href="/graph">Graph</a>

        {isUserLoggedIn === "" ? (
          <a className="scifi-button" href="/signin" style={{ position: "absolute", right: "40px" }}>Sign In</a>
        ) : (
          <a className="scifi-button" onClick={handleSignOut} style={{ position: "absolute", right: "40px", cursor: 'pointer' }}>Sign Out</a>
        )}
      </div>
    </div>
  );
};

export default Header;
