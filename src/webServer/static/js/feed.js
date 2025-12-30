let pathname = window.location.pathname;
let segments = pathname.split('/');
let userId = Number(segments[segments.length - 1]);
let color;
let profile_exists = false;
let pfp = document.getElementById("pfp");
let dorpdown_options = document.getElementById("dropdown-options");
let settings = document.getElementById("settings");
let switch_profiles = document.getElementById("switch-profiles");

async function load_movies() {
    localStorage.removeItem("movies");
    let movies = await fetch("/movies");
    movies = await movies.json();
    localStorage.setItem("movies", JSON.stringify(movies));
}


for (let i = 0; i < profiles.length; i++) {
    if (profiles[i].id == userId) {
        profile_exists = true;
        color = profiles[i].colour;
        pfp.style.backgroundColor = color;
    }
}

if (!profile_exists) {
    location.assign("/feed");
}

pfp.addEventListener("click", () => {
    dorpdown_options.classList.toggle("hidden");
});

switch_profiles.addEventListener("click", () => {
    choose_profile_popup.classList.remove("hidden");
});

settings.addEventListener("click", () => {
    location.assign(`/settings/${userId}`);
});

let results_panel = document.getElementById("results-panel");

async function populate_panel() {
    await load_movies();
    
    let movies_list = JSON.parse(localStorage.getItem("movies"));

    for (let i = 0; i < movies_list.length; i++) {
        let name = movies_list[i][0];
        let file_name = name.toLowerCase();
        file_name = file_name.replaceAll(" ","");
        file_name = file_name.replaceAll(".","");
        file_name = file_name.replaceAll(":","");
        let movie = document.createElement("div");
        movie.classList.add("movie-result");
        let image = document.createElement("img");
        image.setAttribute("src", `../static/images/portrait/${file_name}.jpg`);
        image.setAttribute("onerror", "this.onerror=null;this.src='../static/images/portrait/PLACEHOLDER.jpg'");
        let span = document.createElement("span");
        span.classList.add("movie-result-title");
        span.innerText = name;
        movie.appendChild(image);
        movie.appendChild(span);
        movie.setAttribute("data-name", name);
        results_panel.appendChild(movie);
        movie.addEventListener("click", () => {
            console.log("test");
            location.assign(`/preview/${userId}/${name}`);
        });
    }
}
populate_panel();

let search_bar = document.getElementById("search-bar");
search_bar.addEventListener("focus", () => {
    results_panel.classList.remove("hidden");
});

search_bar.addEventListener("blur", () => {
    setTimeout(function(){ results_panel.classList.add("hidden"); }, 100);
    
});

search_bar.addEventListener("input", () => {
    let text = search_bar.value;
    let movies = results_panel.children;
    for (let i = 0; i < movies.length; i++) {
        if (!(movies[i].getAttribute("data-name").toLowerCase().includes(text.toLowerCase()))) {
            movies[i].classList.add("hidden");
        }
        else {
            movies[i].classList.remove("hidden");
        }
    }
});