// Timer Variables
let timerInterval;
let elapsedTime = 0; // Time in seconds
let feedStarted = false; // Flag to indicate when the live feed has started

// Start Live Feed
function startLiveFeed() {
    const startButton = document.getElementById("start-button");
    const liveFeedContainer = document.getElementById("live-feed-container");
    const liveFeed = document.getElementById("live-feed");

    // Hide start button and show loading animation
    startButton.style.display = "none";
    showLoading();

    // Set the live feed source
    liveFeed.src = "/video_feed";

    // Wait until the first frame loads, then show the live feed container
    liveFeed.onload = () => {
        hideLoading();
        liveFeedContainer.style.display = "block";
        if (!feedStarted) {
            feedStarted = true;
            startTimer();
        }
    };
}

// Show Loading Animation
function showLoading() {
    const loading = document.createElement("div");
    loading.id = "loading";
    loading.innerHTML = `
        <img src="/static/images/loading.gif" alt="Loading..." style="width: 80px; height: 80px;" />
        <p>Loading camera...</p>
    `;
    loading.style.cssText = `
        position: fixed;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        text-align: center;
        font-size: 1.2em;
        color: #4a5568;
    `;
    document.body.appendChild(loading);
}

// Hide Loading Animation
function hideLoading() {
    const loading = document.getElementById("loading");
    if (loading) loading.remove();
}

// Timer Logic
function startTimer() {
    const timerElement = document.getElementById("timer");

    timerInterval = setInterval(() => {
        elapsedTime++;
        const minutes = Math.floor(elapsedTime / 60);
        const seconds = elapsedTime % 60;
        timerElement.textContent = `Timer: ${formatTime(minutes)}:${formatTime(seconds)}`;
    }, 1000);
}

function formatTime(time) {
    return time < 10 ? `0${time}` : time;
}

// End Live Feed
function endLiveFeed() {
    const liveFeedContainer = document.getElementById("live-feed-container");
    const afterFeedContainer = document.getElementById("after-feed-container");

    // Stop the timer
    clearInterval(timerInterval);
    elapsedTime = 0;
    feedStarted = false;

    // Hide live feed and show after-feed options
    liveFeedContainer.style.display = "none";
    afterFeedContainer.style.display = "block";

    // Inform the server to end the session and turn off the camera
    fetch('/end_session')
        .then(response => response.json())
        .then(data => console.log(data.message))
        .catch(err => console.error('Error ending session:', err));
}

// Go to Dashboard
function goToDashboard() {
    window.location.href = '/dashboard';
}

// Go to Session History
function viewSessionHistory() {
    window.location.href = '/session_history';
}

// Go to Home Page
function goToHome() {
    window.location.href = '/';
}

// Session History View (Interactive Cards)
function viewSession(sessionId) {
    window.location.href = `/dashboard?session_id=${sessionId}`;
}