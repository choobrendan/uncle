import React, { useState, useRef, useEffect } from "react";
import VoiceButton from "../components/VoiceButton";
import { useOutletContext } from "react-router-dom";
// Main App Component

function Navigation() {
  const { selectionIndex, setSelectionIndex, simplify, font } =
    useOutletContext();
  const [currentPage, setCurrentPage] = useState("home");
  const [highlightedElement, setHighlightedElement] = useState(null);
  const [userInput, setUserInput] = useState("");
  const [posts, setPosts] = useState(mockPosts);
  const [user, setUser] = useState(mockUser);
  const [showCreatePostModal, setShowCreatePostModal] = useState(false);
  const [data, setData] = useState([]);
  // Refs for elements that need to be highlighted
  const homeButtonRef = useRef(null);
  const searchButtonRef = useRef(null);
  const createPostButtonRef = useRef(null);
  const activityButtonRef = useRef(null);
  const profileButtonRef = useRef(null);
  const storyRef = useRef(null);
  const likeButtonRef = useRef(null);
  const commentButtonRef = useRef(null);
  const shareButtonRef = useRef(null);
  const saveButtonRef = useRef(null);
  const followButtonRef = useRef(null);
  const messageButtonRef = useRef(null);
  const settingsButtonRef = useRef(null);

  // Function to process user commands
  const processCommand = (command) => {
    console.log(command);

    fetch("http://localhost:8000/send-message-navigation", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: command }),
    })
      .then((response) => response.json())
      .then((data) => {
        console.log(data[0].id);
        setData(data);

        // Reset highlight
        setHighlightedElement(null);

        // Navigation based on API response
        if (data[0].id === 1) {
          setCurrentPage("home");
          setHighlightedElement("home");
          scrollToRef(homeButtonRef);
        } else if (data[0].id === 2) {
          setCurrentPage("search");
          setHighlightedElement("search");
          scrollToRef(searchButtonRef);
        } else if (data[0].id === 3) {
          setCurrentPage("profile");
          setHighlightedElement("profile");
          scrollToRef(profileButtonRef);
        } else if (data[0].id === 4) {
          setCurrentPage("activity");
          setHighlightedElement("activity");
          scrollToRef(activityButtonRef);
        }

        // Action commands
        else if (data[0].id === 5) {
          setHighlightedElement("create");
          scrollToRef(createPostButtonRef);
          setTimeout(() => setShowCreatePostModal(true), 1000);
        } else if (data[0].id === 6) {
          setHighlightedElement("story");
          scrollToRef(storyRef);
        } else if (data[0].id === 7) {
          setHighlightedElement("like");
          scrollToRef(likeButtonRef);
        } else if (data[0].id === 8) {
          setHighlightedElement("comment");
          scrollToRef(commentButtonRef);
        } else if (data[0].id === 9) {
          setHighlightedElement("share");
          scrollToRef(shareButtonRef);
        } else if (data[0].id === 10) {
          console.log("okok");
          setHighlightedElement("save");
          scrollToRef(saveButtonRef);
        } else if (data[0].id === 11) {
          setHighlightedElement("follow");
          scrollToRef(followButtonRef);
        } else if (data[0].id === 12) {
          setHighlightedElement("message");
          scrollToRef(messageButtonRef);
        } else if (data[0].id === 13) {
          setHighlightedElement("settings");
          scrollToRef(settingsButtonRef);
        } else {
          // No recognized command
          console.log("Command not recognized:", command);
        }
      })
      .catch((error) => console.error("Error sending message:", error));

    // This line will be executed immediately, before the fetch is complete
    console.log(command);
  };

  // Helper function to scroll to a ref
  const scrollToRef = (ref) => {
    if (ref && ref.current) {
      ref.current.scrollIntoView({ behavior: "smooth", block: "center" });
    }
  };

  // Handle user input submission
  const handleInputSubmit = (e) => {
    e.preventDefault();
    if (userInput.trim()) {
      processCommand(userInput);
      setUserInput("");
    }
  };

  // Render the current page
  const renderPage = () => {
    switch (currentPage) {
      case "home":
        return (
          <HomePage
            posts={posts}
            highlightedElement={highlightedElement}
            refs={{
              storyRef,
              likeButtonRef,
              commentButtonRef,
              shareButtonRef,
              saveButtonRef,
            }}
          />
        );
      case "search":
        return <SearchPage />;
      case "profile":
        return (
          <ProfilePage
            user={user}
            highlightedElement={highlightedElement}
            refs={{
              followButtonRef,
              messageButtonRef,
              settingsButtonRef,
            }}
          />
        );
      case "activity":
        return <ActivityPage />;
      default:
        return <HomePage posts={posts} />;
    }
  };

  return (
    <div
      className="app-body"
      style={{
        ...styles.appBody,
        backgroundColor: simplify ? "#ffffff" : "inherit",
      }}
    >
      <div className="app-container" style={styles.appContainer}>
        {/* Main Content Area */}
        <div className="content-area" style={styles.contentArea}>
          {renderPage()}
        </div>

        {/* Create Post Modal */}
        {showCreatePostModal && (
          <CreatePostModal onClose={() => setShowCreatePostModal(false)} />
        )}

        {/* Navigation Bar */}
        <div className="navigation-bar" style={styles.navigationBar}>
          <div
            ref={homeButtonRef}
            className={`nav-item ${currentPage === "home" ? "active" : ""} ${
              highlightedElement === "home" ? "highlighted" : ""
            }`}
            style={{
              ...styles.navItem,
              ...(currentPage === "home" ? styles.activeNavItem : {}),
              ...(highlightedElement === "home"
                ? styles.highlightedElement
                : {}),
            }}
            onClick={() => setCurrentPage("home")}
          >
            <svg height="24" viewBox="0 0 24 24" width="24">
              <path
                d="M9.005 16.545a2.997 2.997 0 0 1 2.997-2.997A2.997 2.997 0 0 1 15 16.545V22h7V11.543L12 2 2 11.543V22h7.005Z"
                fill="none"
                stroke="currentColor"
                strokeLinejoin="round"
                strokeWidth="2"
              />
            </svg>
          </div>

          <div
            ref={searchButtonRef}
            className={`nav-item ${currentPage === "search" ? "active" : ""} ${
              highlightedElement === "search" ? "highlighted" : ""
            }`}
            style={{
              ...styles.navItem,
              ...(currentPage === "search" ? styles.activeNavItem : {}),
              ...(highlightedElement === "search"
                ? styles.highlightedElement
                : {}),
            }}
            onClick={() => setCurrentPage("search")}
          >
            <svg height="24" viewBox="0 0 24 24" width="24">
              <path
                d="M19 10.5A8.5 8.5 0 1 1 10.5 2a8.5 8.5 0 0 1 8.5 8.5Z"
                fill="none"
                stroke="currentColor"
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth="2"
              />
              <line
                fill="none"
                stroke="currentColor"
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth="2"
                x1="16.511"
                x2="22"
                y1="16.511"
                y2="22"
              />
            </svg>
          </div>

          <div
            ref={createPostButtonRef}
            className={`nav-item ${
              highlightedElement === "create" ? "highlighted" : ""
            }`}
            style={{
              ...styles.navItem,
              ...(highlightedElement === "create"
                ? styles.highlightedElement
                : {}),
            }}
            onClick={() => setShowCreatePostModal(true)}
          >
            <svg height="24" viewBox="0 0 24 24" width="24">
              <path
                d="M2 12v3.45c0 2.849.698 4.005 1.606 4.944.94.909 2.098 1.608 4.946 1.608h6.896c2.848 0 4.006-.7 4.946-1.608C21.302 19.455 22 18.3 22 15.45V8.552c0-2.849-.698-4.006-1.606-4.945C19.454 2.7 18.296 2 15.448 2H8.552c-2.848 0-4.006.699-4.946 1.607C2.698 4.547 2 5.703 2 8.552Z"
                fill="none"
                stroke="currentColor"
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth="2"
              />
              <line
                fill="none"
                stroke="currentColor"
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth="2"
                x1="12"
                x2="12"
                y1="8"
                y2="16"
              />
              <line
                fill="none"
                stroke="currentColor"
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth="2"
                x1="8"
                x2="16"
                y1="12"
                y2="12"
              />
            </svg>
          </div>

          <div
            ref={activityButtonRef}
            className={`nav-item ${
              currentPage === "activity" ? "active" : ""
            } ${highlightedElement === "activity" ? "highlighted" : ""}`}
            style={{
              ...styles.navItem,
              ...(currentPage === "activity" ? styles.activeNavItem : {}),
              ...(highlightedElement === "activity"
                ? styles.highlightedElement
                : {}),
            }}
            onClick={() => setCurrentPage("activity")}
          >
            <svg height="24" viewBox="0 0 24 24" width="24">
              <path
                d="M16.792 3.904A4.989 4.989 0 0 1 21.5 9.122c0 3.072-2.652 4.959-5.197 7.222-2.512 2.243-3.865 3.469-4.303 3.752-.477-.309-2.143-1.823-4.303-3.752C5.141 14.072 2.5 12.167 2.5 9.122a4.989 4.989 0 0 1 4.708-5.218 4.21 4.21 0 0 1 3.675 1.941c.84 1.175.98 1.763 1.12 1.763s.278-.588 1.11-1.766a4.17 4.17 0 0 1 3.679-1.938m0-2a6.04 6.04 0 0 0-4.797 2.127 6.052 6.052 0 0 0-4.787-2.127A6.985 6.985 0 0 0 .5 9.122c0 3.61 2.55 5.827 5.015 7.97.283.246.569.494.853.747l1.027.918a44.998 44.998 0 0 0 3.518 3.018 2 2 0 0 0 2.174 0 45.263 45.263 0 0 0 3.626-3.115l.922-.824c.293-.26.59-.519.885-.774 2.334-2.025 4.98-4.32 4.98-7.94a6.985 6.985 0 0 0-6.708-7.218Z"
                fill="currentColor"
              />
            </svg>
          </div>

          <div
            ref={profileButtonRef}
            className={`nav-item ${currentPage === "profile" ? "active" : ""} ${
              highlightedElement === "profile" ? "highlighted" : ""
            }`}
            style={{
              ...styles.navItem,
              ...(currentPage === "profile" ? styles.activeNavItem : {}),
              ...(highlightedElement === "profile"
                ? styles.highlightedElement
                : {}),
            }}
            onClick={() => setCurrentPage("profile")}
          >
            <img
              src="https://picsum.photos/id/100/50/50"
              alt="Profile"
              style={styles.profileImage}
            />
          </div>
        </div>

        {/* Command Input Bar */}
        <div className="command-input" style={styles.commandInputContainer}>
          <form onSubmit={handleInputSubmit} style={styles.commandForm}>
            <input
              type="text"
              value={userInput}
              onChange={(e) => setUserInput(e.target.value)}
              placeholder="Type a command (e.g., 'How can I create a post?')"
              style={styles.commandInput}
            />
            <button
              type="submit"
              style={{ fontFamily: font, fontWeight: "bold" }}
            >
              PROCESS
            </button>
          </form>
        </div>
      </div>
      <VoiceButton
        setSelectionIndex={setSelectionIndex}
        selectionIndex={selectionIndex}
      />
    </div>
  );
}

// Home Page Component
const HomePage = ({ posts, highlightedElement, refs }) => {
  return (
    <div className="home-page" style={styles.homePage}>
      {/* Stories Section */}
      <div className="stories-section" style={styles.storiesSection}>
        {mockUsers.map((user, index) => (
          <div
            key={index}
            ref={index === 0 ? refs?.storyRef : null}
            className={`story ${
              highlightedElement === "story" && index === 0 ? "highlighted" : ""
            }`}
            style={{
              ...styles.story,
              ...(highlightedElement === "story" && index === 0
                ? styles.highlightedElement
                : {}),
            }}
          >
            <div style={styles.storyRing}>
              <img
                src={user.profilePic}
                alt={user.username}
                style={styles.storyImage}
              />
            </div>
            <span style={styles.storyUsername}>{user.username}</span>
          </div>
        ))}
      </div>

      {/* Posts */}
      <div className="posts" style={styles.posts}>
        {posts.map((post, index) => (
          <div key={index} className="post" style={styles.post}>
            {/* Post Header */}
            <div className="post-header" style={styles.postHeader}>
              <div style={styles.postHeaderLeft}>
                <img
                  src={post.user.profilePic}
                  alt={post.user.username}
                  style={styles.postProfileImage}
                />
                <span style={styles.postUsername}>{post.user.username}</span>
              </div>
              <div style={styles.postHeaderRight}>
                <svg height="24" viewBox="0 0 24 24" width="24">
                  <circle cx="12" cy="12" r="1.5" fill="currentColor" />
                  <circle cx="6" cy="12" r="1.5" fill="currentColor" />
                  <circle cx="18" cy="12" r="1.5" fill="currentColor" />
                </svg>
              </div>
            </div>

            {/* Post Image */}
            <div className="post-image" style={styles.postImage}>
              <img
                src={post.imageUrl}
                alt="Post"
                style={{ width: "100%", height: "100%", objectFit: "cover" }}
              />
            </div>

            {/* Post Actions */}
            <div className="post-actions" style={styles.postActions}>
              <div style={styles.postActionsLeft}>
                <div
                  ref={index === 0 ? refs?.likeButtonRef : null}
                  className={`like-button ${
                    highlightedElement === "like" && index === 0
                      ? "highlighted"
                      : ""
                  }`}
                  style={{
                    ...styles.actionIcon,
                    ...(highlightedElement === "like" && index === 0
                      ? styles.highlightedElement
                      : {}),
                  }}
                >
                  <svg height="24" viewBox="0 0 24 24" width="24">
                    <path
                      d="M16.792 3.904A4.989 4.989 0 0 1 21.5 9.122c0 3.072-2.652 4.959-5.197 7.222-2.512 2.243-3.865 3.469-4.303 3.752-.477-.309-2.143-1.823-4.303-3.752C5.141 14.072 2.5 12.167 2.5 9.122a4.989 4.989 0 0 1 4.708-5.218 4.21 4.21 0 0 1 3.675 1.941c.84 1.175.98 1.763 1.12 1.763s.278-.588 1.11-1.766a4.17 4.17 0 0 1 3.679-1.938m0-2a6.04 6.04 0 0 0-4.797 2.127 6.052 6.052 0 0 0-4.787-2.127A6.985 6.985 0 0 0 .5 9.122c0 3.61 2.55 5.827 5.015 7.97.283.246.569.494.853.747l1.027.918a44.998 44.998 0 0 0 3.518 3.018 2 2 0 0 0 2.174 0 45.263 45.263 0 0 0 3.626-3.115l.922-.824c.293-.26.59-.519.885-.774 2.334-2.025 4.98-4.32 4.98-7.94a6.985 6.985 0 0 0-6.708-7.218Z"
                      fill="currentColor"
                    />
                  </svg>
                </div>
                <div
                  ref={index === 0 ? refs?.commentButtonRef : null}
                  className={`comment-button ${
                    highlightedElement === "comment" && index === 0
                      ? "highlighted"
                      : ""
                  }`}
                  style={{
                    ...styles.actionIcon,
                    ...(highlightedElement === "comment" && index === 0
                      ? styles.highlightedElement
                      : {}),
                  }}
                >
                  <svg height="24" viewBox="0 0 24 24" width="24">
                    <path
                      d="M20.656 17.008a9.993 9.993 0 1 0-3.59 3.615L22 22Z"
                      fill="none"
                      stroke="currentColor"
                      strokeLinejoin="round"
                      strokeWidth="2"
                    />
                  </svg>
                </div>
                <div
                  ref={index === 0 ? refs?.shareButtonRef : null}
                  className={`share-button ${
                    highlightedElement === "share" && index === 0
                      ? "highlighted"
                      : ""
                  }`}
                  style={{
                    ...styles.actionIcon,
                    ...(highlightedElement === "share" && index === 0
                      ? styles.highlightedElement
                      : {}),
                  }}
                >
                  <svg height="24" viewBox="0 0 24 24" width="24">
                    <line
                      fill="none"
                      stroke="currentColor"
                      strokeLinejoin="round"
                      strokeWidth="2"
                      x1="22"
                      x2="9.218"
                      y1="3"
                      y2="10.083"
                    />
                    <polygon
                      fill="none"
                      points="11.698 20.334 22 3.001 2 3.001 9.218 10.084 11.698 20.334"
                      stroke="currentColor"
                      strokeLinejoin="round"
                      strokeWidth="2"
                    />
                  </svg>
                </div>
              </div>
              <div
                ref={index === 0 ? refs?.saveButtonRef : null}
                className={`save-button ${
                  highlightedElement === "save" && index === 0
                    ? "highlighted"
                    : ""
                }`}
                style={{
                  ...styles.actionIcon,
                  ...(highlightedElement === "save" && index === 0
                    ? styles.highlightedElement
                    : {}),
                }}
              >
                <svg height="24" viewBox="0 0 24 24" width="24">
                  <polygon
                    fill="none"
                    points="20 21 12 13.44 4 21 4 3 20 3 20 21"
                    stroke="currentColor"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth="2"
                  />
                </svg>
              </div>
            </div>

            {/* Likes Count */}
            <div className="post-likes" style={styles.postLikes}>
              {post.likes.toLocaleString()} likes
            </div>

            {/* Caption */}
            <div className="post-caption" style={styles.postCaption}>
              <span style={styles.postUsername}>{post.user.username}</span>{" "}
              {post.caption}
            </div>

            {/* View Comments */}
            <div className="view-comments" style={styles.viewComments}>
              View all {post.commentCount} comments
            </div>

            {/* Post Time */}
            <div className="post-time" style={styles.postTime}>
              {post.timeAgo}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

// Search Page Component
const SearchPage = () => {
  return (
    <div className="search-page" style={styles.searchPage}>
      <div style={styles.searchHeader}>
        <input type="text" placeholder="Search" style={styles.searchInput} />
      </div>
      <div style={styles.searchGrid}>
        {Array(15)
          .fill(0)
          .map((_, index) => {
            // Generate a random ID using Math.random
            const randomId = Math.floor(Math.random() * 1000); // Random ID between 0 and 999
            return (
              <div key={randomId} style={styles.searchGridItem}>
                <img
                  src={`https://picsum.photos/id/${randomId}/300/300`}
                  alt="Post"
                  style={styles.profilePost}
                />
              </div>
            );
          })}
      </div>
    </div>
  );
};

// Profile Page Component
const ProfilePage = ({ user, highlightedElement, refs }) => {
  return (
    <div className="profile-page" style={styles.profilePage}>
      {/* Profile Header */}
      <div className="profile-header" style={styles.profileHeader}>
        <div style={styles.profileHeaderTop}>
          <div style={styles.profileUsername}>
            {user.username}
            <div
              ref={refs?.settingsButtonRef}
              className={`settings-button ${
                highlightedElement === "settings" ? "highlighted" : ""
              }`}
              style={{
                ...styles.settingsButton,
                ...(highlightedElement === "settings"
                  ? styles.highlightedElement
                  : {}),
              }}
            >
              <svg height="24" viewBox="0 0 24 24" width="24">
                <circle cx="12" cy="12" r="1.5" fill="currentColor" />
                <circle cx="6" cy="12" r="1.5" fill="currentColor" />
                <circle cx="18" cy="12" r="1.5" fill="currentColor" />
              </svg>
            </div>
          </div>
        </div>

        <div style={styles.profileInfo}>
          <div style={styles.profilePicSection}>
            <img
              src={user.profilePic}
              alt="Profile"
              style={styles.profilePageImage}
            />
          </div>

          <div style={styles.profileStats}>
            <div style={styles.stat}>
              <span style={styles.statNumber}>{user.posts}</span>
              <span style={styles.statLabel}>Posts</span>
            </div>
            <div style={styles.stat}>
              <span style={styles.statNumber}>{user.followers}</span>
              <span style={styles.statLabel}>Followers</span>
            </div>
            <div style={styles.stat}>
              <span style={styles.statNumber}>{user.following}</span>
              <span style={styles.statLabel}>Following</span>
            </div>
          </div>
        </div>

        <div style={styles.profileBio}>
          <div style={styles.fullName}>{user.fullName}</div>
          <div style={styles.bio}>{user.bio}</div>
        </div>

        <div style={styles.profileActions}>
          <button
            ref={refs?.followButtonRef}
            className={`follow-button ${
              highlightedElement === "follow" ? "highlighted" : ""
            }`}
            style={{
              ...styles.profileActionButton,
              ...(highlightedElement === "follow"
                ? styles.highlightedElement
                : {}),
            }}
          >
            Follow
          </button>
          <button
            ref={refs?.messageButtonRef}
            className={`message-button ${
              highlightedElement === "message" ? "highlighted" : ""
            }`}
            style={{
              ...styles.profileActionButton,
              ...(highlightedElement === "message"
                ? styles.highlightedElement
                : {}),
            }}
          >
            Message
          </button>
        </div>
      </div>

      {/* Profile Posts */}
      <div className="profile-posts" style={styles.profilePosts}>
        <div style={styles.profileTabsContainer}>
          <div style={styles.profileTabActive}>
            <svg height="24" viewBox="0 0 24 24" width="24">
              <rect
                fill="none"
                height="18"
                stroke="currentColor"
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth="2"
                width="18"
                x="3"
                y="3"
              />
              <line
                fill="none"
                stroke="currentColor"
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth="2"
                x1="9.015"
                x2="9.015"
                y1="3"
                y2="21"
              />
              <line
                fill="none"
                stroke="currentColor"
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth="2"
                x1="14.985"
                x2="14.985"
                y1="3"
                y2="21"
              />
              <line
                fill="none"
                stroke="currentColor"
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth="2"
                x1="21"
                x2="3"
                y1="9.015"
                y2="9.015"
              />
              <line
                fill="none"
                stroke="currentColor"
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth="2"
                x1="21"
                x2="3"
                y1="14.985"
                y2="14.985"
              />
            </svg>
          </div>
          <div style={styles.profileTab}>
            <svg height="24" viewBox="0 0 24 24" width="24">
              <polygon
                fill="none"
                points="20 21 12 13.44 4 21 4 3 20 3 20 21"
                stroke="currentColor"
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth="2"
              />
            </svg>
          </div>
          <div style={styles.profileTab}>
            <svg height="24" viewBox="0 0 24 24" width="24">
              <path
                d="M18 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V4a2 2 0 0 0-2-2Z"
                fill="none"
                stroke="currentColor"
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth="2"
              />
              <path
                d="m9 13 2 2 4-4"
                fill="none"
                stroke="currentColor"
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth="2"
              />
            </svg>
          </div>
        </div>

        <div style={styles.profileGrid}>
          {Array(9)
            .fill(0)
            .map((_, index) => {
              // Generate a random ID using Math.random
              const randomId = Math.floor(Math.random() * 1000); // Random ID between 0 and 999
              return (
                <div key={randomId} style={styles.profileGridItem}>
                  <img
                    src={`https://picsum.photos/id/${randomId}/300/300`}
                    alt="Post"
                    style={styles.profilePost}
                  />
                </div>
              );
            })}
        </div>
      </div>
    </div>
  );
};

// Activity Page Component
const ActivityPage = () => {
  return (
    <div className="activity-page" style={styles.activityPage}>
      <h2 style={styles.activityHeader}>Activity</h2>

      <div style={styles.activitySection}>
        <h3 style={styles.activitySectionTitle}>Today</h3>
        {Array(5)
          .fill(0)
          .map((_, index) => (
            <div key={index} style={styles.activityItem}>
              <img
                src={`https://picsum.photos/id/${150 + index}/300/300`}
                alt="User"
                style={styles.activityUserImage}
              />
              <div style={styles.activityContent}>
                <span style={styles.activityUsername}>user{index + 1}</span>
                <span style={styles.activityText}>
                  {index % 3 === 0
                    ? "liked your photo."
                    : index % 3 === 1
                    ? "started following you."
                    : 'commented: "Amazing post!"'}
                </span>
              </div>
              <div style={styles.activityTime}>2h</div>
            </div>
          ))}
      </div>

      <div style={styles.activitySection}>
        <h3 style={styles.activitySectionTitle}>This Week</h3>
        {Array(5)
          .fill(0)
          .map((_, index) => (
            <div key={index} style={styles.activityItem}>
              <img
                src={`https://picsum.photos/id/${160 + index}/300/300`}
                alt="User"
                style={styles.activityUserImage}
              />
              <div style={styles.activityContent}>
                <span style={styles.activityUsername}>user{index + 6}</span>
                <span style={styles.activityText}>
                  {index % 3 === 0
                    ? "liked your photo."
                    : index % 3 === 1
                    ? "started following you."
                    : 'commented: "Great content!"'}
                </span>
              </div>
              <div style={styles.activityTime}>{index + 1}d</div>
            </div>
          ))}
      </div>
    </div>
  );
};

// Styles object for the SocialMediaApp component
const styles = {
  appBody: {
    height: "100vh",
    display: "flex",
    flexWrap: "wrap",
    alignContent: "center",
    boxShadow: "0px 4px 20px rgba(0, 0, 0, 0.24)",
  },
  appContainer: {
    fontFamily: "Helvetica, Arial, sans-serif",
    maxWidth: "1414px",
    height: "80vh",
    paddingLeft: "20px",
    paddingRight: "20px",
    margin: "0 auto",
    display: "flex",
    flexDirection: "column",
    position: "relative",
    border: "1px solid #dbdbdb",
    backgroundColor: "#fafafa",
    overflowY: "hidden",
  },

  contentArea: {
    flex: 1,
    overflowY: "auto",
    paddingBottom: "50px",
    width: "600px",
  },

  // Navigation Bar
  navigationBar: {
    position: "absolute",
    bottom: "-2px",
    right: "2px",
    width: "100%",
    maxWidth: "1414px", // Match container
    height: "50px",
    backgroundColor: "white",
    display: "flex",
    justifyContent: "space-around",
    alignItems: "center",
    borderTop: "1px solid #dbdbdb",
    zIndex: 10,
  },

  navItem: {
    display: "flex",
    justifyContent: "center",
    alignItems: "center",
    height: "50px",
    width: "50px",
    cursor: "pointer",
    transition: "all 0.2s ease",
  },

  activeNavItem: {
    fontWeight: "bold",
  },

  profileImage: {
    width: "28px",
    height: "28px",
    borderRadius: "50%",
    objectFit: "cover",
  },

  // Command Input Container
  commandInputContainer: {
    position: "absolute",
    top: "2px",
    right: "2px",
    width: "100%",
    maxWidth: "1414px",
    padding: "10px",
    backgroundColor: "#f0f0f0",
    zIndex: 100,
    boxShadow: "0 2px 4px rgba(0,0,0,0.1)",
  },

  commandForm: {
    display: "flex",
    width: "100%",
  },

  commandInput: {
    flex: 1,
    padding: "8px 12px",
    border: "1px solid #dbdbdb",
    borderRadius: "4px",
    fontSize: "14px",
  },

  commandButton: {
    marginLeft: "10px",
    padding: "8px 16px",
    backgroundColor: "#0095f6",
    color: "white",
    border: "none",
    borderRadius: "4px",
    cursor: "pointer",
    fontWeight: "bold",
  },

  // Highlighted Element
  highlightedElement: {
    boxShadow: "0 0 0 2pxrgb(246, 0, 0)",
    animation: "pulse 1.5s infinite",
    backgroundColor: "rgba(246, 0, 0, 0.7)",
  },

  // Home Page
  homePage: {
    paddingTop: "60px",
    paddingLeft: "60px",
    paddingRight: "60px",
    width: "600px", // Space for command input
    backgroundColor: "white",
  },

  // Stories Section
  storiesSection: {
    display: "flex",
    overflowX: "auto",
    padding: "10px",
    borderBottom: "1px solid #dbdbdb",
  },

  story: {
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    marginRight: "15px",
    cursor: "pointer",
  },

  storyRing: {
    width: "66px",
    height: "66px",
    borderRadius: "50%",
    background:
      "linear-gradient(45deg, #feda75, #fa7e1e, #d62976, #962fbf, #4f5bd5)",
    display: "flex",
    justifyContent: "center",
    alignItems: "center",
  },

  storyImage: {
    width: "60px",
    height: "60px",
    borderRadius: "50%",
    border: "2px solid white",
    objectFit: "cover",
  },

  storyUsername: {
    fontSize: "12px",
    marginTop: "5px",
    maxWidth: "64px",
    overflow: "hidden",
    textOverflow: "ellipsis",
    whiteSpace: "nowrap",
  },

  // Posts
  posts: {
    marginTop: "10px",
  },

  post: {
    marginBottom: "15px",
    borderBottom: "1px solid #dbdbdb",
    backgroundColor: "white",
  },

  postHeader: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    padding: "10px",
  },

  postHeaderLeft: {
    display: "flex",
    alignItems: "center",
  },

  postHeaderRight: {
    cursor: "pointer",
  },

  postProfileImage: {
    width: "32px",
    height: "32px",
    borderRadius: "50%",
    marginRight: "10px",
    objectFit: "cover",
  },

  postUsername: {
    fontWeight: "bold",
    fontSize: "14px",
  },

  postImage: {
    width: "100%",
    height: "500px", // Square for Instagram-like feel
    backgroundColor: "#fafafa",
  },

  postActions: {
    display: "flex",
    justifyContent: "space-between",
    padding: "10px",
  },

  postActionsLeft: {
    display: "flex",
  },

  actionIcon: {
    marginRight: "15px",
    cursor: "pointer",
  },

  postLikes: {
    padding: "0 10px",
    fontWeight: "bold",
    fontSize: "14px",
  },

  postCaption: {
    padding: "5px 10px",
    fontSize: "14px",
    lineHeight: "18px",
  },

  viewComments: {
    padding: "5px 10px",
    color: "#8e8e8e",
    fontSize: "14px",
    cursor: "pointer",
  },

  postTime: {
    padding: "5px 10px",
    color: "#8e8e8e",
    fontSize: "12px",
    marginBottom: "5px",
  },

  // Search Page
  searchPage: {
    paddingTop: "60px",
    width: "600px", // Space for command input
  },

  searchHeader: {
    padding: "10px",
    backgroundColor: "white",
    borderBottom: "1px solid #dbdbdb",
  },

  searchInput: {
    width: "100%",
    padding: "8px 12px",
    backgroundColor: "#efefef",
    border: "none",
    borderRadius: "8px",
    fontSize: "14px",
  },

  searchGrid: {
    display: "grid",
    gridTemplateColumns: "repeat(3, 1fr)",
    gap: "2px",
    marginTop: "2px",
  },

  searchGridItem: {
    aspectRatio: "1/1",
    backgroundColor: "#fafafa",
  },

  searchImage: {
    width: "100%",
    height: "100%",
    objectFit: "cover",
  },

  // Profile Page
  profilePage: {
    paddingTop: "60px", // Space for command input
    backgroundColor: "white",
  },

  profileHeader: {
    padding: "15px",
  },

  profileHeaderTop: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: "10px",
  },

  profileUsername: {
    fontSize: "20px",
    fontWeight: "bold",
    display: "flex",
    alignItems: "center",
  },

  settingsButton: {
    marginLeft: "10px",
    cursor: "pointer",
  },

  profileInfo: {
    display: "flex",
    alignItems: "center",
    marginBottom: "15px",
  },

  profilePicSection: {
    marginRight: "20px",
  },

  profilePageImage: {
    width: "80px",
    height: "80px",
    borderRadius: "50%",
    objectFit: "cover",
  },

  profileStats: {
    display: "flex",
    justifyContent: "space-around",
    flex: 1,
  },

  stat: {
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
  },

  statNumber: {
    fontWeight: "bold",
    fontSize: "18px",
  },

  statLabel: {
    fontSize: "14px",
    color: "#8e8e8e",
  },

  profileBio: {
    marginBottom: "15px",
  },

  fullName: {
    fontWeight: "bold",
    fontSize: "16px",
  },

  bio: {
    fontSize: "14px",
    lineHeight: "18px",
  },

  profileActions: {
    display: "flex",
    justifyContent: "space-between",
  },

  profileActionButton: {
    flex: 1,
    padding: "7px 0",
    margin: "0 5px",
    backgroundColor: "#0095f6",
    color: "white",
    border: "none",
    borderRadius: "4px",
    fontWeight: "bold",
    cursor: "pointer",
  },

  profileTabsContainer: {
    display: "flex",
    justifyContent: "space-around",
    borderTop: "1px solid #dbdbdb",
    marginTop: "10px",
  },

  profileTab: {
    flex: 1,
    display: "flex",
    justifyContent: "center",
    alignItems: "center",
    padding: "10px 0",
    color: "#8e8e8e",
    cursor: "pointer",
  },

  profileTabActive: {
    flex: 1,
    display: "flex",
    justifyContent: "center",
    alignItems: "center",
    padding: "10px 0",
    borderTop: "1px solid #262626",
    marginTop: "-1px",
    color: "#262626",
    cursor: "pointer",
  },

  profileGrid: {
    display: "grid",
    gridTemplateColumns: "repeat(3, 1fr)",
    gap: "2px",
  },

  profileGridItem: {
    aspectRatio: "1/1",
    backgroundColor: "#fafafa",
  },

  profilePost: {
    width: "100%",
    height: "100%",
    objectFit: "cover",
  },

  profilePosts: {
    marginBottom: "50px",
  },

  // Activity Page
  activityPage: {
    paddingTop: "60px",
    width: "600px",
    backgroundColor: "white",
  },

  activityHeader: {
    fontSize: "18px",
    fontWeight: "bold",
    padding: "15px",
    borderBottom: "1px solid #dbdbdb",
    color: "black",
  },

  activitySection: {
    marginBottom: "20px",
  },

  activitySectionTitle: {
    fontSize: "16px",
    fontWeight: "bold",
    padding: "10px 15px",
    color: "black",
  },

  activityItem: {
    display: "flex",
    alignItems: "center",
    padding: "10px 15px",
  },

  activityUserImage: {
    width: "44px",
    height: "44px",
    borderRadius: "50%",
    marginRight: "10px",
    objectFit: "cover",
  },

  activityContent: {
    flex: 1,
    fontSize: "14px",
  },

  activityUsername: {
    fontWeight: "bold",
    marginRight: "5px",
  },

  activityText: {
    color: "#262626",
  },

  activityTime: {
    fontSize: "12px",
    color: "#8e8e8e",
    marginLeft: "10px",
  },

  // Create Post Modal
  modalOverlay: {
    position: "fixed",
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: "rgba(0, 0, 0, 0.5)",
    display: "flex",
    justifyContent: "center",
    alignItems: "center",
    zIndex: 1000,
  },

  modalContent: {
    width: "90%",
    maxWidth: "400px",
    backgroundColor: "white",
    borderRadius: "12px",
    overflow: "hidden",
  },

  modalHeader: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    padding: "10px 15px",
    borderBottom: "1px solid #dbdbdb",
  },

  modalTitle: {
    fontSize: "16px",
    fontWeight: "bold",
    color: "black",
  },

  closeButton: {
    background: "none",
    border: "none",
    cursor: "pointer",
    fontSize: "18px",
  },

  uploadArea: {
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    justifyContent: "center",
    padding: "40px 20px",
    backgroundColor: "#fafafa",
    borderBottom: "1px solid #dbdbdb",
  },

  uploadIcon: {
    fontSize: "40px",
    marginBottom: "15px",
    color: "#8e8e8e",
  },

  uploadText: {
    fontSize: "16px",
    fontWeight: "bold",
    marginBottom: "10px",
    color: "black",
  },

  uploadDescription: {
    fontSize: "14px",
    color: "#8e8e8e",
    textAlign: "center",
    marginBottom: "20px",
  },

  uploadButton: {
    padding: "7px 16px",
    backgroundColor: "#0095f6",
    color: "white",
    border: "none",
    borderRadius: "4px",
    fontWeight: "bold",
    cursor: "pointer",
  },

  captionArea: {
    padding: "15px",
  },

  captionInput: {
    width: "100%",
    padding: "10px",
    border: "1px solid #dbdbdb",
    borderRadius: "4px",
    fontSize: "14px",
    minHeight: "80px",
    resize: "none",
  },

  shareButton: {
    width: "100%",
    padding: "10px",
    backgroundColor: "#0095f6",
    color: "white",
    border: "none",
    borderRadius: "4px",
    fontWeight: "bold",
    cursor: "pointer",
    marginTop: "15px",
  },
};

const CreatePostModal = ({ onClose }) => {
  const [caption, setCaption] = useState("");
  const [selectedImage, setSelectedImage] = useState(null);

  const handleImageChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setSelectedImage(URL.createObjectURL(file));
    }
  };

  const handleSubmit = () => {
    // Here you would normally upload the image and create a post
    console.log("Creating post with caption:", caption);
    onClose();
  };

  return (
    <div className="modal-overlay" style={styles.modalOverlay}>
      <div className="modal-content" style={styles.modalContent}>
        <div className="modal-header" style={styles.modalHeader}>
          <h3 style={styles.modalTitle}>Create New Post</h3>
          <button onClick={onClose} style={styles.closeButton}>
            ×
          </button>
        </div>

        {!selectedImage ? (
          <div className="upload-area" style={styles.uploadArea}>
            <div style={styles.uploadIcon}>
              <svg height="48" viewBox="0 0 48 48" width="48">
                <path
                  d="M38.5 33.5v-19c0-2.8-2.2-5-5-5h-19c-2.8 0-5 2.2-5 5v19c0 2.8 2.2 5 5 5h19c2.7 0 5-2.2 5-5zm-23-19c0-1.1.9-2 2-2h7.5v7.5h7.5V12.5h2c1.1 0 2 .9 2 2v7.5h-7.5v7.5h7.5v4c0 1.1-.9 2-2 2h-18c-1.1 0-2-.9-2-2v-19zm7.5 15v-7.5h-7.5v7.5h7.5z"
                  fill="currentColor"
                />
              </svg>
            </div>
            <h2 style={styles.uploadText}>Drag photos and videos here</h2>
            <p style={styles.uploadDescription}>
              Share your moments with friends and followers
            </p>
            <label htmlFor="file-upload" style={styles.uploadButton}>
              Select from computer
            </label>
            <input
              id="file-upload"
              type="file"
              accept="image/*"
              onChange={handleImageChange}
              style={{ display: "none" }}
            />
          </div>
        ) : (
          <>
            <div
              style={{
                ...styles.postImage,
                backgroundImage: `url(${selectedImage})`,
                backgroundSize: "contain",
                backgroundPosition: "center",
                backgroundRepeat: "no-repeat",
                backgroundColor: "#000",
              }}
            ></div>
            <div style={styles.captionArea}>
              <textarea
                placeholder="Write a caption..."
                value={caption}
                onChange={(e) => setCaption(e.target.value)}
                style={styles.captionInput}
              />
              <button onClick={handleSubmit} style={styles.shareButton}>
                Share
              </button>
            </div>
          </>
        )}
      </div>
    </div>
  );
};

// Mock Data
const mockUsers = [
  {
    id: 1,
    username: "your_story",
    profilePic: "https://picsum.photos/id/237/300/300",
  },
  {
    id: 2,
    username: "user1",
    profilePic: "https://picsum.photos/id/64/300/300",
  },
  {
    id: 3,
    username: "travel_lover",
    profilePic: "https://picsum.photos/id/65/300/300",
  },
  {
    id: 4,
    username: "foodie",
    profilePic: "https://picsum.photos/id/66/300/300",
  },
  {
    id: 5,
    username: "photographer",
    profilePic: "https://picsum.photos/id/67/300/300",
  },
  {
    id: 6,
    username: "artist",
    profilePic: "https://picsum.photos/id/68/300/300",
  },
  {
    id: 7,
    username: "fitness_guru",
    profilePic: "https://picsum.photos/id/69/300/300",
  },
];

const mockPosts = [
  {
    id: 1,
    user: {
      username: "travel_lover",
      profilePic: "https://picsum.photos/id/65/300/300",
    },
    imageUrl: "https://picsum.photos/id/10/500/500",
    caption: "Exploring the beautiful mountains! #nature #adventure",
    likes: 1248,
    commentCount: 42,
    timeAgo: "2 HOURS AGO",
  },
  {
    id: 2,
    user: {
      username: "foodie",
      profilePic: "https://picsum.photos/id/66/300/300",
    },
    imageUrl: "https://picsum.photos/id/20/500/500",
    caption: "This pasta was amazing! 🍝 #foodie #yummy",
    likes: 856,
    commentCount: 23,
    timeAgo: "4 HOURS AGO",
  },
  {
    id: 3,
    user: {
      username: "photographer",
      profilePic: "https://picsum.photos/id/67/300/300",
    },
    imageUrl: "https://picsum.photos/id/30/500/500",
    caption: "Perfect sunset shot today! #photography #sunset",
    likes: 2103,
    commentCount: 87,
    timeAgo: "5 HOURS AGO",
  },
  {
    id: 4,
    user: {
      username: "artist",
      profilePic: "https://picsum.photos/id/68/300/300?random=1",
    },
    imageUrl: "https://picsum.photos/id/40/500/500",
    caption: "My latest artwork. What do you think? #art #creative",
    likes: 1567,
    commentCount: 103,
    timeAgo: "6 HOURS AGO",
  },
  {
    id: 5,
    user: {
      username: "fitness_guru",
      profilePic: "https://picsum.photos/id/69/300/300",
    },
    imageUrl: "https://picsum.photos/id/50/500/500",
    caption: "Morning workout done! 💪 #fitness #healthy",
    likes: 948,
    commentCount: 36,
    timeAgo: "12 HOURS AGO",
  },
];

const mockUser = {
  username: "instagram_user",
  fullName: "Instagram User",
  profilePic: "https://picsum.photos/id/237/300/300",
  posts: 15,
  followers: 458,
  following: 285,
  bio: "Welcome to my profile! 📱 Photography enthusiast | Food lover | Travel addict",
};

export default Navigation;
