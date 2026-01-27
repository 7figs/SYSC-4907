let movies;
let movie_name;
let userId;
let user_index;
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
    let choose_profile_container = document.getElementById("choose-profile-container");

    pfp.addEventListener("click", () => {
        dorpdown_options.classList.toggle("hidden");
    });

    switch_profiles.addEventListener("click", () => {
        choose_profile_container.classList.remove("hidden");
    });

    settings.addEventListener("click", () => {
        location.assign(`/settings/${userId}`);
    });

    let pathname = window.location.pathname;
    let segments = pathname.split('/');
    userId = Number(segments[segments.length - 2]);
    let movie_id = segments[segments.length - 1];
    movies.forEach(movie => {
        if (movie[0] == movie_id) {
            movie_name = movie[1];
        }
    });
    movie_name = movie_name.replaceAll("%20", " ");
    movie_name = decodeURIComponent(movie_name);
    let color;
    let profile_exists = false;
    let movie_exists = false;
    let profiles = JSON.parse(localStorage.getItem("users"));

    let movie_index = -1;

    for (let i = 0; i < profiles.length; i++) {
        if (profiles[i].id == userId) {
            user_index = i;
            profile_exists = true;
            color = profiles[i].colour;
            pfp.style.backgroundColor = color;
        }
    }

    for (let i = 0; i < movies.length; i++) {
        if (movies[i][1] == movie_name) {
            movie_exists = true;
            movie_index = i;
        }
    }

    if (!profile_exists || !movie_exists) {
        location.assign("/watch");
    }

    let see_more = document.getElementById("see_more_link");
    see_more.addEventListener("click", () => {
        location.assign(`/feed/${userId}`);
    });

    let movie_title = movies[movie_index][1];
    let title = document.getElementById("movie-title");
    title.innerText = movie_title;

    let history = profiles[user_index].watch_history;
    let dateObj = new Date();
    let month = dateObj.getMonth() + 1; // months from 1-12
    let day = dateObj.getDate();
    let year = dateObj.getFullYear();
    let newDate = `${year}/${month}/${day}`;
    let obj = {
        "id": movie_id,
        "name": movie_title,
        "opinion": "unknown",
        "guess": "dislike",
        "last_watched": newDate,
        "timestamp": Date.now()
    };
    let new_movie = true;
    if (history) {
        for (let i = 0; i < history.length; i++) {
            if (history[i].name == movie_title) {
                new_movie = false;
                obj.opinion = history[i].opinion;
                if (history[i].last_watched != obj.last_watched) {
                    history.unshift(obj);
                    break;
                }
            }
        }

        if (new_movie) {
            history.unshift(obj);
        }
        
        if (history.length == 0) {
            history.unshift(obj);
        }
    }
    else {
        history = [];
        history.unshift(obj);
    }
    localStorage.removeItem("users");
    localStorage.setItem("users", JSON.stringify(profiles));

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
        localStorage.removeItem("users");
        localStorage.setItem("users", JSON.stringify(profiles));
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
        localStorage.removeItem("users");
        localStorage.setItem("users", JSON.stringify(profiles));
    });

    let users = JSON.parse(localStorage.getItem("users"));

    let id_history = [];
    let days = [];
    let preferences = [];

    if (history.length > 2) {
        for (let i = 1; i < history.length; i++) {
            id_history.push(history[i].id);
            let current_time = Date.now();
            let time_dif = current_time - history[i].timestamp;
            time_dif = time_dif / 8640000;
            days.push(time_dif);
            if (history[i].opinion == "like") {
                preferences.push(1);
            }
            else if (history[i].opinion == "dislike") {
                preferences.push(-1);
            }
            else if (history[i].guess == "like") {
                preferences.push(1);
            }
            else if (history[i].guess == "dislike") {
                preferences.push(-1);
            }
        }

        let user_vector = await fetch(`/user-vector?h=${JSON.stringify(id_history)}&p=${JSON.stringify(preferences)}&d=${JSON.stringify(days)}`);
        user_vector = await user_vector.json();
        users[user_index].vector = user_vector;
        localStorage.removeItem("users");
        localStorage.setItem("users", JSON.stringify(users));
    }

    if (history.length >= 3 && history.length % 3 == 0) {
        let new_like = users[userId].initial_like;
        let new_dislike = users[userId].initial_dislike;
        for (let i = 0; i < history.length - 1; i++) {
            if (history[i].opinion == "like") {
                new_like.push(history[i].id);
            }
            if (history[i].opinion == "dislike") {
                new_dislike.push(history[i].id);
            }
            if (history[i].opinion == "unknown") {
                if (history[i].guess == "like") {
                    new_like.push(history[i].id);
                }
                if (history[i].guess == "dislike") {
                    new_dislike.push(history[i].id);
                }
            }
        }
        let tree = await fetch(`/tree?l=${JSON.stringify(new_like)}&d=${JSON.stringify(new_dislike)}`);
        tree = await tree.json();
        users[user_index].tree = tree;
        localStorage.removeItem("users");
        localStorage.setItem("users", JSON.stringify(users));
    }
}

async function play_video() {
    await setup_page();
    let video = document.getElementById("video");

    let movie_url = movie_name.replaceAll("%20", "");
    movie_url = movie_url.replaceAll(".","");
    movie_url = movie_url.replaceAll(":","");
    movie_url = decodeURIComponent(movie_url);
    movie_url = movie_url.toLowerCase();
    movie_url = movie_url.normalize("NFD").replace(/[\u0300-\u036f]/g, "").replace(/[^\p{L}\p{N}]+/gu, "");

    // Dynamically build the URL based on browser address
    let hostname = window.location.hostname;  // example: 192.168.2.27
    let port = window.location.port || 8000;  // fallback if no port visible
    let src = `http://144.217.34.146:8000/movies/${movie_url}/stream.m3u8`;
    video.src = src;

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

    let users = JSON.parse(localStorage.getItem("users"));
    let current_movie = users[user_index].watch_history[0];
    let hasTriggered = false;

    video.addEventListener('timeupdate', () => {
    let percentage = (video.currentTime / video.duration) * 100;

    if (percentage >= 75 && !hasTriggered && current_movie.opinion == "unknown") {
        current_movie.guess = "like";
        localStorage.removeItem("users");
        localStorage.setItem("users", JSON.stringify(users));
        hasTriggered = true;
    }
    });

    let tree = users[user_index].tree[0];
    let user_vector = users[user_index].vector;
    if (!user_vector) {
        user_vector = [];
    }
    let history = users[user_index].watch_history;
    let too_soon = [];
    for (let i = 0; i < history.length; i++) {
        let cur_time = Date.now();
        let time_dif = cur_time - history[i].timestamp;
        time_dif = time_dif / 8640000;
        if (time_dif < 3) {
            too_soon.push(Number(history[i].id));
        }
    }
    let recommendations = await fetch(`/recommend?t=${JSON.stringify(tree)}&v=${JSON.stringify(user_vector)}&s=${JSON.stringify(too_soon)}`);
    recommendations = await recommendations.json();
    let recommended_section = document.getElementById("recommended-movies");
    for (let i = 0; i < recommendations.length; i++) {
        let movie = document.createElement("div");
        movie.classList.add("movie");
        let image = document.createElement("img");
        let name = recommendations[i][1];
        let id = recommendations[i][0];
        let file_name = name.toLowerCase();
        file_name = file_name.replaceAll(" ","");
        file_name = file_name.replaceAll(".","");
        file_name = file_name.replaceAll(":","");
        image.setAttribute("src", `/static/images/portrait/${file_name}.jpg`);
        image.setAttribute("onerror", "this.onerror=null;this.src='../static/images/portrait/PLACEHOLDER.jpg'");
        let p = document.createElement("p");
        p.classList.add("movie-title");
        p.innerText = recommendations[i][1];
        p.setAttribute("title", recommendations[i][1]);
        movie.appendChild(image);
        movie.appendChild(p);
        movie.setAttribute("data-name", name);
        movie.addEventListener("click", () => {
            location.assign(`/preview/${userId}/${id}`);
        });
        recommended_section.appendChild(movie);
    }
}
play_video();