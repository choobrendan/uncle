import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import App from './App';
import reportWebVitals from './reportWebVitals';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Home from './pages/Home'; // Assuming Home.js exists in the same directory
import About from './pages/About';
import Header from './components/Header';
import VoiceButton from './components/VoiceButton';
const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
          <VoiceButton  />
          <Header></Header>
    <Router>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/about" element={<About />} />
      </Routes>
    </Router>
  </React.StrictMode>
);

// If you want to start measuring performance in your app, pass a function
// to log results (for example: reportWebVitals(console.log))
// or send to an analytics endpoint. Learn more: https://bit.ly/CRA-vitals
reportWebVitals();
