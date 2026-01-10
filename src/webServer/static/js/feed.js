let pathname = window.location.pathname;
let segments = pathname.split('/');
let userId = Number(segments[segments.length - 1]);
let color;
let profile_exists = false;
let pfp = document.getElementById("pfp");
let dorpdown_options = document.getElementById("dropdown-options");
let settings = document.getElementById("settings");
let switch_profiles = document.getElementById("switch-profiles");
let num_other_movies = 30;
let user_index;
let profiles = JSON.parse(localStorage.getItem("users"));

async function load_movies() {
    localStorage.removeItem("movies");
    let movies = await fetch("/movies");
    movies = await movies.json();
    localStorage.setItem("movies", JSON.stringify(movies));
}


for (let i = 0; i < profiles.length; i++) {
    if (profiles[i].id == userId) {
        user_index = i;
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
        if (!(movies[i].getAttribute("data-name").toLowerCase().normalize("NFD").replace(/[\u0300-\u036f]/g, "").replace(/[^\p{L}\p{N}]+/gu, " ").includes(text.toLowerCase().normalize("NFD").replace(/[\u0300-\u036f]/g, "").replace(/[^\p{L}\p{N}]+/gu, " ")))) {
            movies[i].classList.add("hidden");
        }
        else {
            movies[i].classList.remove("hidden");
        }
    }
});

async function populate_feed() {
    let tree = profiles[user_index].tree[0];
    let recommendations = await fetch(`/recommend?t=${JSON.stringify(tree)}`);
    recommendations = await recommendations.json();
    let recommended_section = document.getElementById("recommended-movies");
    let other_section = document.getElementById("other-movies");
    for (let i = 0; i < recommendations.length; i++) {
        let movie = document.createElement("div");
        movie.classList.add("movie");
        let image = document.createElement("img");
        let name = recommendations[i];
        let file_name = name.toLowerCase();
        file_name = file_name.replaceAll(" ","");
        file_name = file_name.replaceAll(".","");
        file_name = file_name.replaceAll(":","");
        image.setAttribute("src", `/static/images/portrait/${file_name}.jpg`);
        image.setAttribute("onerror", "this.onerror=null;this.src='../static/images/portrait/PLACEHOLDER.jpg'");
        let p = document.createElement("p");
        p.classList.add("movie-title");
        p.innerText = recommendations[i];
        p.setAttribute("title", recommendations[i]);
        movie.appendChild(image);
        movie.appendChild(p);
        movie.setAttribute("data-name", name);
        movie.addEventListener("click", () => {
            location.assign(`/preview/${userId}/${name}`);
        });
        recommended_section.appendChild(movie);
    }
    let movies = JSON.parse(localStorage.getItem("movies"));
    let other_movies = [];
    while (other_movies.length < num_other_movies) {
            add_movie = true;
            let choice = movies[Math.floor(Math.random() * movies.length)];
            for (let i = 0; i < other_movies.length; i++) {
                if (other_movies[i][0] == choice[0] && other_movies[i][1] == choice[1]) {
                    add_movie = false;
                    break;
                }
            }
            if (add_movie) {
                other_movies.push(choice);
                let movie = document.createElement("div");
                movie.classList.add("movie");
                let image = document.createElement("img");
                let name = choice[0];
                let file_name = name.toLowerCase();
                file_name = file_name.replaceAll(" ","");
                file_name = file_name.replaceAll(".","");
                file_name = file_name.replaceAll(":","");
                image.setAttribute("src", `/static/images/portrait/${file_name}.jpg`);
                image.setAttribute("onerror", "this.onerror=null;this.src='../static/images/portrait/PLACEHOLDER.jpg'");
                let p = document.createElement("p");
                p.classList.add("movie-title");
                p.innerText = choice[0];
                p.setAttribute("title", choice[0]);
                movie.appendChild(image);
                movie.appendChild(p);
                movie.setAttribute("data-name", name);
                movie.addEventListener("click", () => {
                    location.assign(`/preview/${userId}/${name}`);
                });
                other_section.appendChild(movie);
            }
        }
}
populate_feed();