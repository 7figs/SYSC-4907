let movies;
async function load_movies() {
    await fetch_movies();
    movies = JSON.parse(localStorage.getItem("movies"));
}

async function preview_movie() {
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
    let userId = Number(segments[segments.length - 2]);
    let movie_name = segments[segments.length - 1];
    movie_name = movie_name.replaceAll("%20", " ");
    movie_name = decodeURIComponent(movie_name);
    let color;
    let profile_exists = false;
    let movie_exists = false;
    let profiles = JSON.parse(localStorage.getItem("users"));

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
        location.assign("/preview");
    }

    let description_section = document.getElementById("movie-description");
    let movie_title = movies[movie_index][0];
    let movie_overview = movies[movie_index][1];
    let title = document.createElement("h1");
    let overview = document.createElement("p");
    title.innerText = movie_title;
    overview.innerText = movie_overview;
    description_section.appendChild(title);
    let hr = document.createElement("hr");
    hr.style.marginBottom = "20px";
    description_section.appendChild(hr);
    description_section.appendChild(overview);
    let file_name = movie_title.toLowerCase();
    file_name = file_name.replaceAll(" ","");
    file_name = file_name.replaceAll(".","");
    file_name = file_name.replaceAll(":","");
    let poster = document.getElementById("movie-poster");
    poster.setAttribute("src", `/static/images/portrait/${file_name}.jpg`);
    poster.setAttribute("onerror", "this.onerror=null;this.src='/static/images/portrait/PLACEHOLDER.jpg'");

    let watch_button = document.getElementById("watch-movie-button");
    watch_button.addEventListener("click", () => {
        location.assign(`/watch/${userId}/${movie_title}`);
    });
}
preview_movie();