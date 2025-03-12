
import React from 'react';

const Header = ({ item ,brightnessIndex, setBrightnessIndex}) => {
    return (
<div className="header" style={{filter:`brightness(${1*brightnessIndex}`}}>
<div className="centered">
  <a className="scifi-button" href="/about"><router-link to="/about">About</router-link></a>
  <a className="scifi-button" href="#0">Cancel</a>
</div>
</div>
    )
};

export default Header