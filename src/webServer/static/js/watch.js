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

    let history = JSON.parse(localStorage.getItem("watch_history"));
    let dateObj = new Date();
    let month   = dateObj.getUTCMonth() + 1; // months from 1-12
    let day     = dateObj.getUTCDate();
    let year    = dateObj.getUTCFullYear();
    let newDate = `${year}/${month}/${day}`;
    let obj = {
        "name": movie_title,
        "opinion": "unknown",
        "last_watched": newDate
    };
    if (history) {
        for (let i = 0; i < history.length; i++) {
            if (history[i].name == movie_title) {
                obj.opinion = history[i].opinion;
                if (history[i].last_watched != obj.last_watched) {
                    history.push(obj);
                    break;
                }
            }
        }
    }
    else {
        history = [];
        history.push(obj);
    }
    localStorage.removeItem("watch_history");
    localStorage.setItem("watch_history", JSON.stringify(history));

    let like_button_solid = document.getElementById("watch-like-icon-select");
    let like_button_regular = document.getElementById("watch-like-icon-deselect");
    let dislike_button_solid = document.getElementById("watch-dislike-icon-select");
    let dislike_button_regular = document.getElementById("watch-dislike-icon-deselect");
    let like_button = document.getElementById("watch-button-like");
    let dislike_button = document.getElementById("watch-button-dislike");

    if (obj.opinion == "like") {
        like_button_solid.classList.remove("hidden");
        like_button_regular.classList.add("hidden");
        dislike_button_solid.classList.add("hidden");
        dislike_button_regular.classList.remove("hidden");
    }

    if (obj.opinion == "dislike") {
        like_button_solid.classList.add("hidden");
        like_button_regular.classList.remove("hidden");
        dislike_button_solid.classList.remove("hidden");
        dislike_button_regular.classList.add("hidden");
    }

    like_button.addEventListener("click", () => {
        if (!like_button_solid.classList.contains("hidden")) {
            like_button_solid.classList.add("hidden");
            like_button_regular.classList.remove("hidden");
            dislike_button_solid.classList.add("hidden");
            dislike_button_regular.classList.remove("hidden");
            for (let i = 0; i < history.length; i++) {
                if (history[i].name == movie_title) {
                    history[i].opinion = "unknown";
                }
            }
        }
        else {
            like_button_solid.classList.remove("hidden");
            like_button_regular.classList.add("hidden");
            dislike_button_solid.classList.add("hidden");
            dislike_button_regular.classList.remove("hidden");
            for (let i = 0; i < history.length; i++) {
                if (history[i].name == movie_title) {
                    history[i].opinion = "like";
                }
            }
        }
        localStorage.removeItem("watch_history");
        localStorage.setItem("watch_history", JSON.stringify(history));
    });

    dislike_button.addEventListener("click", () => {
        if (!dislike_button_solid.classList.contains("hidden")) {
            like_button_solid.classList.add("hidden");
            like_button_regular.classList.remove("hidden");
            dislike_button_solid.classList.add("hidden");
            dislike_button_regular.classList.remove("hidden");
            for (let i = 0; i < history.length; i++) {
                if (history[i].name == movie_title) {
                    history[i].opinion = "unknown";
                }
            }
        }
        else {
            like_button_solid.classList.add("hidden");
            like_button_regular.classList.remove("hidden");
            dislike_button_solid.classList.remove("hidden");
            dislike_button_regular.classList.add("hidden");
            for (let i = 0; i < history.length; i++) {
                if (history[i].name == movie_title) {
                    history[i].opinion = "dislike";
                }
            }
        }
        localStorage.removeItem("watch_history");
        localStorage.setItem("watch_history", JSON.stringify(history));
    });

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