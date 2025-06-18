import React, { useState, useRef, useEffect } from "react";
import "./Onboarding.css"; // Import the CSS file

const Basketball = ({ setNextGame }) => {
  const [level, setLevel] = useState(1);
  const [ballPosition, setBallPosition] = useState({ x: 100, y: 100 });
  const [hoopPosition, setHoopPosition] = useState({ x: 700, y: 240 });
  const [isDragging, setIsDragging] = useState(false);
  const [velocity, setVelocity] = useState({ x: 0, y: 0 });
  const [isDropped, setIsDropped] = useState(false);
  const [message, setMessage] = useState("");
  const [scored, setScored] = useState(false);
  const [netOffset, setNetOffset] = useState(0);
  const requestRef = useRef();
  const previousTimeRef = useRef();
  const dragOffset = useRef({ x: 0, y: 0 });

  // Constants for physics
  const gravity = 0.6;
  const friction = 0.99;
  const bounceFactor = 0.8;

  // Court dimensions
  const courtWidth = 800;
  const courtHeight = 500;
  const floorHeight = 450;

  // Ball properties
  const ballRadius = 20;

  // Level configurations
  const levelConfigs = [
    {
      startBall: {
        x: Math.random() * (courtWidth - 2 * ballRadius) + ballRadius, // Random x, keeping ball inside court
        y: Math.random() * (floorHeight - 2 * ballRadius) + ballRadius, // Random y, keeping ball above the floor
      },
      hoopPos: {
        x: Math.random() * (courtWidth - 2 * ballRadius) + ballRadius, // Random hoop x within court width
        y: Math.random() * (floorHeight - 100) + 50, // Random hoop y within court, above the floor
      },
      netMovement: false,
    },
    {
      startBall: {
        x: Math.random() * (courtWidth - 2 * ballRadius) + ballRadius,
        y: Math.random() * (floorHeight - 2 * ballRadius) + ballRadius,
      },
      hoopPos: {
        x: Math.random() * (courtWidth - 2 * ballRadius) + ballRadius,
        y: Math.random() * (floorHeight - 100) + 50,
      },
      netMovement: false,
    },
    // {
    //   startBall: {
    //     x: Math.random() * (courtWidth - 2 * ballRadius) + ballRadius,
    //     y: Math.random() * (floorHeight - 2 * ballRadius) + ballRadius,
    //   },
    //   hoopPos: {
    //     x: Math.random() * (courtWidth - 2 * ballRadius) + ballRadius,
    //     y: Math.random() * (floorHeight - 100) + 50,
    //   },
    //   netMovement: true,
    //   netMoveSpeed: 0.5,
    //   moveDistance: 50,
    // },
    // {
    //   startBall: {
    //     x: Math.random() * (courtWidth - 2 * ballRadius) + ballRadius,
    //     y: Math.random() * (floorHeight - 2 * ballRadius) + ballRadius,
    //   },
    //   hoopPos: {
    //     x: Math.random() * (courtWidth - 2 * ballRadius) + ballRadius,
    //     y: Math.random() * (floorHeight - 100) + 50,
    //   },
    //   netMovement: true,
    //   netMoveSpeed: 0.3,
    //   moveDistance: 120,
    // },
    // {
    //   startBall: {
    //     x: Math.random() * (courtWidth - 2 * ballRadius) + ballRadius,
    //     y: Math.random() * (floorHeight - 2 * ballRadius) + ballRadius,
    //   },
    //   hoopPos: {
    //     x: Math.random() * (courtWidth - 2 * ballRadius) + ballRadius,
    //     y: Math.random() * (floorHeight - 100) + 50,
    //   },
    //   netMovement: true,
    //   netMoveSpeed: 0.8,
    //   moveDistance: 140,
    // },
  ];
  useEffect(() => {
    const config = levelConfigs[level - 1];
    setBallPosition(config.startBall);
    setHoopPosition(config.hoopPos);
    setIsDropped(false);
    setScored(false);
    setMessage("");
  }, [level]);

  const checkScore = (ballX, ballY) => {
    // Calculate adjusted hoop position with net offset
    const hoopCenterX = hoopPosition.x + netOffset;
    const hoopInnerLeft = hoopCenterX - hoopWidth / 2 + rimWidth;
    const hoopInnerRight = hoopCenterX + hoopWidth / 2 - rimWidth;
    const scoringYStart = hoopPosition.y + rimWidth;
    const scoringYEnd = hoopPosition.y + rimWidth + hoopHeight;

    // Check if ball passes through the hoop rim area while moving downward
    if (
      !scored &&
      ballX > hoopInnerLeft &&
      ballX < hoopInnerRight &&
      ballY >= scoringYStart &&
      ballY <= scoringYEnd &&
      velocity.y > 0
    ) {
      setScored(true);
      setMessage(`Level ${level} Complete!`);
    }
  };
  const handleProceed = () => {
    if (scored) {
      nextLevel(); // Advance to the next level
    }
    if (scored && level === 2) {
      setNextGame(1); // Proceed to the next game
    }
  };

  const nextLevel = () => {
    if (level < levelConfigs.length) {
      setLevel((prev) => prev + 1);
    } else {
      setMessage("Congratulations! You've completed all levels!");
    }
  };

  const resetGame = () => {
    if (scored) {
      nextLevel();
    } else {
      const config = levelConfigs[level - 1];
      setBallPosition(config.startBall);
      setVelocity({ x: 0, y: 0 });
      setIsDropped(false);
      setScored(false);
      setMessage("");
    }
  };

  const handleMouseDown = (e) => {
    if (isDropped) {
      resetGame();
      return;
    }

    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    // Only allow dragging if clicking on the ball
    const dx = x - ballPosition.x;
    const dy = y - ballPosition.y;
    const distance = Math.sqrt(dx * dx + dy * dy);

    if (distance <= ballRadius) {
      setIsDragging(true);
      dragOffset.current = { x: dx, y: dy };
    }
  };

  const handleMouseMove = (e) => {
    if (isDragging) {
      const rect = e.currentTarget.getBoundingClientRect();
      const mouseX = e.clientX - rect.left;
      const mouseY = e.clientY - rect.top;

      // Calculate new position considering drag offset
      let newX = mouseX - dragOffset.current.x;
      let newY = mouseY - dragOffset.current.y;

      // Constrain to court boundaries
      newX = Math.min(Math.max(newX, ballRadius), courtWidth - ballRadius);
      newY = Math.min(Math.max(newY, ballRadius), floorHeight - ballRadius);

      setBallPosition({ x: newX, y: newY });
    }
  };

  const handleMouseUp = () => {
    if (isDragging) {
      setIsDragging(false);
      setIsDropped(true);
    }
  };

  // Hoop properties
  const hoopWidth = 60;
  const hoopHeight = 40;
  const rimWidth = 10;

  // Animation loop for physics and net movement
  const animatePhysics = (time) => {
    if (previousTimeRef.current === undefined) {
      previousTimeRef.current = time;
    }

    const deltaTime = time - previousTimeRef.current;
    previousTimeRef.current = time;

    // Net movement for levels with moving nets
    const currentConfig = levelConfigs[level - 1];

    if (currentConfig.netMovement) {
      const newNetOffset =
        Math.sin(time * 0.005 * currentConfig.netMoveSpeed) * 200;
      setNetOffset(newNetOffset);
    }

    if (isDropped && !isDragging) {
      let newVelocityX = velocity.x * friction;
      let newVelocityY = velocity.y + gravity;

      let newX = ballPosition.x + newVelocityX;
      let newY = ballPosition.y + newVelocityY;

      // Handle collision with floor
      if (newY > floorHeight - ballRadius) {
        newY = floorHeight - ballRadius;
        newVelocityY = -newVelocityY * bounceFactor;
      }

      // Handle collision with walls
      if (newX < ballRadius) {
        newX = ballRadius;
        newVelocityX = -newVelocityX * bounceFactor;
      } else if (newX > courtWidth - ballRadius) {
        newX = courtWidth - ballRadius;
        newVelocityX = -newVelocityX * bounceFactor;
      }

      // Check for scoring
      checkScore(newX, newY);

      setVelocity({ x: newVelocityX, y: newVelocityY });
      setBallPosition({ x: newX, y: newY });
    }

    requestRef.current = requestAnimationFrame(animatePhysics);
  };

  useEffect(() => {
    if (level === 6) {
      setNextGame(1);
    }
    requestRef.current = requestAnimationFrame(animatePhysics);
    return () => cancelAnimationFrame(requestRef.current);
  }, [isDropped, isDragging, velocity, ballPosition, level]);
  return (
    <div className="container">
      <h1 className="title">Level {level}</h1>

      <div
        className="court"
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
      >
        <div className="floor" />
        <div>
          <div
            className="hoop"
            style={{
              left: hoopPosition.x - hoopWidth / 2 + netOffset,
              top: hoopPosition.y,
              zIndex: 100,
            }}
          >
            <div className="hoop-rim" />
          </div>
          {/* <div 
className="hitbox"
style={{ 
  left: (hoopPosition.x + netOffset) - hoopWidth/2 + rimWidth,
  width: hoopWidth - 2 * rimWidth,
  top: hoopPosition.y + rimWidth,
  height: hoopHeight
}}
/> */}
          <div
            className="net"
            style={{
              left: hoopPosition.x - hoopWidth / 2 + netOffset + 10,
              top: hoopPosition.y + rimWidth,
              zIndex: 100,
            }}
          />
        </div>
        <div
          className="ball"
          style={{
            left: ballPosition.x - ballRadius,
            top: ballPosition.y - ballRadius,
          }}
        >
          <div className="ball-lines" />
          <div className="ball-horizontal" />
        </div>
      </div>
      {message && <div className="message">{message}</div>}
      <div className="button-container">
        {isDropped ? (
          <button onClick={handleProceed} className="button">
            {scored ? `Proceed` : `Reset`}
          </button>
        ) : (
          <p class="onboarding">Drag the ball and release it to shoot!</p>
        )}
      </div>
    </div>
  );
};

export default Basketball;
