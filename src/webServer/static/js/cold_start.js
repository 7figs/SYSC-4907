import messages from "../messages.json" with {type: "json"};

let panels = document.querySelectorAll(".panel");
let num_panels = panels.length;
let next_buttons = [];
let like_grid = document.querySelectorAll(".like")[0];
let dislike_grid = document.querySelectorAll(".dislike")[0];
let movies;
let load_new_randoms;
let random_like = [];
let random_dislike = [];
const num_options = 12;
let like_list = [];
let dislike_list = [];
let create_profile_btn = document.getElementById("create-profile");
let username = document.getElementById("name");
let toast = document.getElementById("toast");
let pfp = document.getElementById("pfp");
let color = document.getElementById("color");

color.addEventListener("input", () => {
    pfp.style.backgroundColor = color.value;
});

async function generate_random_movies() {
    await fetch_movies();
    movies = JSON.parse(localStorage.getItem("movies"));
    load_new_randoms = !((localStorage.getItem("random_like")) && (localStorage.getItem("random_dislike")));
    let add_movie = true;
    if (load_new_randoms) {
        while (random_like.length < num_options) {
            add_movie = true;
            let movie = movies[Math.floor(Math.random() * movies.length)];
            for (let i = 0; i < random_like.length; i++) {
                if (random_like[i][0] == movie[0]) {
                    add_movie = false;
                    break;
                }
            }
            if (add_movie) {
                random_like.push(movie);
            }
        }
        localStorage.removeItem("random_like");
        localStorage.setItem("random_like", JSON.stringify(random_like));
        random_like = JSON.parse(localStorage.getItem("random_like"));
        add_movie = true;
        while (random_dislike.length < num_options) {
            add_movie = true;
            let movie = movies[Math.floor(Math.random() * movies.length)];
            for (let i = 0; i < random_like.length; i++) {
                if (random_like[i][0] == movie[0]) {
                    add_movie = false;
                    break;
                }
            }
            for (let i = 0; i < random_dislike.length; i++) {
                if (random_dislike[i][0] == movie[0]) {
                    add_movie = false;
                    break;
                }
            }
            if (add_movie) {
                random_dislike.push(movie);
            }
        }
        localStorage.removeItem("random_dislike");
        localStorage.setItem("random_dislike", JSON.stringify(random_dislike));
    }
    else {
        random_like = JSON.parse(localStorage.getItem("random_like"));
        random_dislike = JSON.parse(localStorage.getItem("random_dislike"));
    }

    for (let i = 0; i < num_options; i++) {
        let like_card = create_card(random_like[i][1], random_like[i][0]);
        let dislike_card = create_card(random_dislike[i][1], random_dislike[i][0]);

        like_grid.appendChild(like_card);
        dislike_grid.appendChild(dislike_card);
    }

    like_grid = like_grid.children;
    dislike_grid = dislike_grid.children;

    for (let i = 0; i < num_panels; i++) {
        panels[i].id = i;
        panels[i].querySelectorAll(".submit-btn")[0].addEventListener("click", () => change_panel(i, 1));
        let temp = panels[i].querySelectorAll(".back-btn");
        if (temp.length > 0) {
            panels[i].querySelectorAll(".back-btn")[0].addEventListener("click", () => change_panel(i, 0));
        }
    }
}

async function add_eventlisteners() {
    await generate_random_movies();
    for (let i = 0; i < like_grid.length; i++) {
    like_grid[i].addEventListener("click", () => toggle_selection(like_grid[i]));
    }

    for (let i = 0; i < dislike_grid.length; i++) {
        dislike_grid[i].addEventListener("click", () => toggle_selection(dislike_grid[i]));
    }

    if ((localStorage.getItem("likes")) && (localStorage.getItem("dislikes"))) {
        like_list = JSON.parse(localStorage.getItem("likes"));
        dislike_list = JSON.parse(localStorage.getItem("dislikes"));

        for (let i = 0; i < like_grid.length; i++) {
            if (like_list.includes(like_grid[i].children[1].innerText)) {
                like_grid[i].children[0].classList.toggle("card-not-selected");
                like_grid[i].classList.toggle("card-selected");
            }
        }

        for (let i = 0; i < dislike_grid.length; i++) {
            if (dislike_list.includes(dislike_grid[i].children[1].innerText)) {
                dislike_grid[i].children[0].classList.toggle("card-not-selected");
                dislike_grid[i].classList.toggle("card-selected");
            }
        }
    }

    create_profile_btn.addEventListener("click", () => create_profile());
}
add_eventlisteners();

async function create_profile() {
    if (username.value != "") {
        let tree = await fetch(`/tree?l=${JSON.stringify(like_list)}&d=${JSON.stringify(dislike_list)}`);
        tree = await tree.json();
        let users = localStorage.getItem("users");
        if (users) {
            users = JSON.parse(users);
            let user = {
                "id": users.length,
                "name": username.value,
                "tree": tree[0],
                "figure": tree[1],
                "watch_history": [],
                "colour": color.value,
                "initial_like": like_list,
                "initial_dislike": dislike_list
            }
            users.push(user);
            localStorage.removeItem("users");
            localStorage.setItem("users", JSON.stringify(users));
        }
        else {
            let user = {
                "id": 0,
                "name": username.value,
                "tree": tree[0],
                "figure": tree[1],
                "watch_history": [],
                "colour": color.value,
                "initial_like": like_list,
                "initial_dislike": dislike_list
            }
            localStorage.setItem("users", JSON.stringify([user]));
        }
        localStorage.removeItem("dislikes");
        localStorage.removeItem("likes");
        localStorage.removeItem("random_dislike");
        localStorage.removeItem("random_like");
        let message = {
            "type": 1,
            "message": messages.cold_start_success
        }
        localStorage.setItem("show_toast", JSON.stringify(message));
        location.assign("/");
    }
    else {
        toast.innerHTML = messages.cold_start_fail;
        toast.classList.add("toast-error");
        toast.classList.add("toast-show");
        setTimeout(function(){ toast.classList.remove("toast-show") }, 2900);
    }
}

function change_panel(i, dir) {
    document.body.scrollTop = 0;
    document.documentElement.scrollTop = 0;
    if (dir) {
        if (i == num_panels - 1) {

        }
        else {
            panels[i].classList.add("hidden");
            panels[i + 1].classList.remove("hidden");
        }
    }
    else {
        if (i == 0) {

        }
        else {
            panels[i].classList.add("hidden");
            panels[i- 1].classList.remove("hidden");
        }
    }
}

function create_card(name, id) {
    let card = document.createElement("div");
    card.classList.add("card");
    card.setAttribute("id", id);

    let file_name = name.toLowerCase();
    file_name = file_name.replaceAll(" ","");
    file_name = file_name.replaceAll(".","");
    file_name = file_name.replaceAll(":","");

    let image = document.createElement("img");
    image.setAttribute("src", `../static/images/portrait/${file_name}.jpg`);
    image.setAttribute("onerror", "this.onerror=null;this.src='../static/images/portrait/PLACEHOLDER.jpg'");
    image.classList.add("card-not-selected");

    let title = document.createElement("p");
    title.innerText = name;
    title.setAttribute("title", name);

    card.appendChild(image);
    card.appendChild(title);

    return card;
}

function toggle_selection(card) {
    card.children[0].classList.toggle("card-not-selected");
    card.classList.toggle("card-selected");

    let temp_like = [];
    let temp_dislike = [];
    for (let i = 0; i < like_grid.length; i++) {
        if (like_grid[i].classList.contains("card-selected")) {
            temp_like.push(random_like[i][0]);
        }
    }
    for (let i = 0; i < dislike_grid.length; i++) {
        if (dislike_grid[i].classList.contains("card-selected")) {
            temp_dislike.push(random_dislike[i][0]);
        }
    }
    like_list = temp_like;
    dislike_list = temp_dislike;
    localStorage.removeItem("likes");
    localStorage.removeItem("dislikes");
    localStorage.setItem("likes", JSON.stringify(like_list));
    localStorage.setItem("dislikes", JSON.stringify(dislike_list));
}