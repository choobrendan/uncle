
import React from 'react';

const Header = ({ item ,brightnessIndex, setBrightnessIndex}) => {
    return (
<div className="header" style={{filter:`brightness(${1*brightnessIndex}`}}>
<div className="centered">
  <a className="scifi-button" href="/about"><router-link to="/about">About</router-link></a>
  <a className="scifi-button" href="/navigation"><router-link to="/navigation">Navigation</router-link></a>
  <a className="scifi-button" href="/signin" style={{display:"flex",position:"absolute", right:"40px"}}>Sign In</a>
</div>
<div>

</div>
</div>
    )
};

export default Header