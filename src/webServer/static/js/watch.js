let movies;
async function load_movies() {
    await fetch_movies();
    movies = JSON.parse(localStorage.getItem("movies"));
}

async function setup_page() {
    await load_movies();
    let pfp = document.getElementById("pfp");
    let dorpdown_options = document.getElementById("dropdown-options");
    let settings = document.getElementById("settings");
    let switch_profiles = document.getElementById("switch-profiles");

    pfp.addEventListener("click", () => {
        dorpdown_options.classList.toggle("hidden");
    });

    switch_profiles.addEventListener("click", () => {
        choose_profile_popup.classList.remove("hidden");
    });

    settings.addEventListener("click", () => {
        location.assign(`/settings/${userId}`);
    });

    let pathname = window.location.pathname;
    let segments = pathname.split('/');
    let userId = Number(segments[segments.length - 2]);
    let movie_name = segments[segments.length - 1];
    movie_name = movie_name.replaceAll("%20", " ");
    movie_name = decodeURIComponent(movie_name);
    let color;
    let profile_exists = false;
    let movie_exists = false;

    let movie_index = -1;

    for (let i = 0; i < profiles.length; i++) {
        if (profiles[i].id == userId) {
            profile_exists = true;
            color = profiles[i].colour;
            pfp.style.backgroundColor = color;
        }
    }

    for (let i = 0; i < movies.length; i++) {
        if (movies[i][0] == movie_name) {
            movie_exists = true;
            movie_index = i;
        }
    }

    if (!profile_exists || !movie_exists) {
        location.assign("/watch");
    }

    let movie_title = movies[movie_index][0];
    let title = document.getElementById("movie-title");
    title.innerText = movie_title;
}
setup_page();

let video = document.getElementById("video");

// Dynamically build the URL based on browser address
let hostname = window.location.hostname;  // example: 192.168.2.27
let port = window.location.port || 8000;  // fallback if no port visible
let src = "http://142.112.252.221:50000/stream.m3u8";

console.log("Using dynamic video source:", src);

if (Hls.isSupported()) {
    const hls = new Hls(
        {
            maxBufferLength: 100,
            liveMaxLatencyDuration: 10,
            backBufferLength: 30,
            liveSyncDuration: 3
        }
    );
    hls.loadSource(src);
    hls.attachMedia(video);
} else if (video.canPlayType("application/vnd.apple.mpegurl")) {
    video.src = src;  // Safari native support
}

document.addEventListener("fullscreenchange", handleFullscreenChange);

function handleFullscreenChange() {
    if (document.fullscreenElement === video) {
        video.classList.remove("rounded-corner");
    } else if (document.fullscreenElement === null) {
        video.classList.add("rounded-corner");
    }
}