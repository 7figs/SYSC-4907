let panels = document.querySelectorAll(".panel");
let num_panels = panels.length;
let next_buttons = [];
let like_grid = document.querySelectorAll(".like")[0].children;
let dislike_grid = document.querySelectorAll(".dislike")[0].children;
let movies = JSON.parse(localStorage.getItem("movies"));
let random_like = [];
let random_dislike = [];

for(let i = 0; i < 12; i++) {
    random_like.push(movies[Math.floor(Math.random() * movies.length)]);
}
for(let i = 0; i < 12; i++) {
    let index = Math.floor(Math.random() * movies.length);
    if (!(index in random_like)){
        random_dislike.push(movies[index]);
    }
    else {
        i--;
    }
}


for (let i = 0; i < num_panels; i++) {
    panels[i].id = i;
    panels[i].querySelectorAll(".submit-btn")[0].addEventListener("click", () => change_panel(i, 1));
    let temp = panels[i].querySelectorAll(".back-btn");
    if (temp.length > 0) {
        panels[i].querySelectorAll(".back-btn")[0].addEventListener("click", () => change_panel(i, 0));
    }

}

for (let i = 0; i < like_grid.length; i++) {
    like_grid[i].addEventListener("click", () => toggle_selection(like_grid[i]));
}

for (let i = 0; i < dislike_grid.length; i++) {
    dislike_grid[i].addEventListener("click", () => toggle_selection(dislike_grid[i]));
}

function change_panel(i, dir) {
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

function toggle_selection(card) {
    card.children[0].classList.toggle("card-not-selected");
    card.classList.toggle("card-selected");
}