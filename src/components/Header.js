import React from 'react';

import { createClient } from '@supabase/supabase-js';

const supabaseUrl = "https://hgatxkpmrskbdqigenav.supabase.co";
const supabaseKey = process.env.REACT_APP_SUPABASE_KEY;
const supabase = createClient(supabaseUrl, supabaseKey);

const Header = ({ item, brightnessIndex, setBrightnessIndex, simplify,isUserLoggedIn }) => {
  // Simplified styles for the header and navigation
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
      fontWeight: 'bold'
    }
  };
console.log(isUserLoggedIn)
  // Apply simplified styles when simplify is true
  if (simplify) {
    return (
      <div style={simplifiedStyles.header}>
        <div style={simplifiedStyles.centered}>
          <a 
            href="/about" 
            style={simplifiedStyles.navLink}
            onMouseOver={(e) => {
              e.target.style.backgroundColor = '#f5f5f5';
            }}
            onMouseOut={(e) => {
              e.target.style.backgroundColor = 'transparent';
            }}
          >
            About
          </a>
          <a 
            href="/navigation" 
            style={simplifiedStyles.navLink}
            onMouseOver={(e) => {
              e.target.style.backgroundColor = '#f5f5f5';
            }}
            onMouseOut={(e) => {
              e.target.style.backgroundColor = 'transparent';
            }}
          >
            Navigation
          </a>
          <a 
            href="/graph" 
            style={simplifiedStyles.navLink}
            onMouseOver={(e) => {
              e.target.style.backgroundColor = '#f5f5f5';
            }}
            onMouseOut={(e) => {
              e.target.style.backgroundColor = 'transparent';
            }}
          >
            Graph
          </a>
          {isUserLoggedIn === "" &&
          (<a 
            href="/signin" 
            style={simplifiedStyles.signInButton}
          >
            Sign In
          </a>) }
          {
          (isUserLoggedIn !=="" &&
            <p>{isUserLoggedIn}</p>
          )}

        </div>
      </div>
    );
  }

  // Original header with sci-fi styling
  return (
    <div className="header" style={{ filter: `brightness(${1 * brightnessIndex}` }}>
      <div className="centered">
        <a className="scifi-button" href="/about"><router-link to="/about">About</router-link></a>
        <a className="scifi-button" href="/navigation"><router-link to="/navigation">Navigation</router-link></a>
        <a className="scifi-button" href="/graph"><router-link to="/graph">Graph</router-link></a>
        <a className="scifi-button" href="/signin" style={{ display: "flex", position: "absolute", right: "40px" }}>Sign In</a>
      </div>
      <div>
      </div>
    </div>
  );
};

export default Header;