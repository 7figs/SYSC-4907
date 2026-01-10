import messages from "../messages.json" with {type: "json"};

async function load_movies() {
    await fetch_movies();
}

let pathname = window.location.pathname;
let segments = pathname.split('/');
let userId = Number(segments[segments.length - 1]);
let color;
let profile_exists = false;
let pfp = document.getElementById("pfp");
let profiles = JSON.parse(localStorage.getItem("users"));
let name_field = document.getElementById("username");
let color_field = document.getElementById("profile-color");
let cancel_button = document.getElementById("cancel-change-button");
let save_button = document.getElementById("save-change-button");
let delete_profile_button = document.getElementById("delete-profile-button");
let confirmation_modal = document.getElementById("confirmation-modal");
let cancel_delete_button = document.getElementById("cancel-delete-button");
let confrim_delete_button = document.getElementById("confirm-delete-button");
let user_index;
let toast = document.getElementById("toast");
let download_button = document.getElementById("download-data-button");

let current_user = profiles[userId];

download_button.addEventListener("click", () => {
    download(JSON.stringify(current_user), `${current_user.name}.json`, "text/plain");
});

function download(content, fileName, contentType) {
    let a = document.createElement("a");
    let file = new Blob([content], { type: contentType });
    a.href = URL.createObjectURL(file);
    a.download = fileName;
    a.click();
}

for (let i = 0; i < profiles.length; i++) {
    if (profiles[i].id == userId) {
        profile_exists = true;
        color = profiles[i].colour;
        pfp.style.backgroundColor = color;
        color_field.value = color;
        name_field.value = profiles[i].name;
        user_index = i;
    }
}

if (!profile_exists) {
    location.assign("/settings");
}

color_field.addEventListener("input", () => {
    let color = color_field.value;
    pfp.style.backgroundColor = color;
});

cancel_button.addEventListener("click", () => {
    name_field.value = profiles[user_index].name;
    color_field.value = profiles[user_index].colour;
    let color = color_field.value;
    pfp.style.backgroundColor = color;
});

save_button.addEventListener("click", () => {
    if (name_field.value.length == 0) {
        toast.innerHTML = messages.settings_fail;
        toast.classList.remove("toast-success");
        toast.classList.add("toast-error");
        toast.classList.add("toast-show");
        setTimeout(function(){ toast.classList.remove("toast-show") }, 2900);
    }
    else {
        profiles[user_index].name = name_field.value;
        profiles[user_index].colour = color_field.value;
        localStorage.removeItem("users");
        localStorage.setItem("users",JSON.stringify(profiles));
        toast.innerHTML = messages.settings_profile_success;
        toast.classList.remove("toast-error");
        toast.classList.add("toast-success");
        toast.classList.add("toast-show");
        setTimeout(function(){ toast.classList.remove("toast-show") }, 2900);
    }
});

delete_profile_button.addEventListener("click", () => {
    confirmation_modal.classList.remove("hidden");
});

cancel_delete_button.addEventListener("click", () => {
    confirmation_modal.classList.add("hidden");
});

confrim_delete_button.addEventListener("click", () => {
    profiles.splice(user_index, 1);
    for (let i = 0; i < profiles.length; i++) {
        profiles[i].id = i;
    }
    localStorage.removeItem("users");
    localStorage.setItem("users", JSON.stringify(profiles));
    let message = {
        "type": 1,
        "message": messages.settings_delete_profile_success
    }
    localStorage.setItem("show_toast", JSON.stringify(message));
    location.assign("/");
});

async function load_watch_history() {
    await load_movies();
    let history = profiles[user_index].watch_history;
    let watch_history = document.getElementById("history");
    for (let i = 0; i < history.length; i++) {
        let log = document.createElement("div");
        log.classList.add("log");
        let first_half = document.createElement("div");
        first_half.classList.add("log-first-half");
        let image = document.createElement("img");
        let movie_title = history[i].name;
        let file_name = movie_title.toLowerCase();
        file_name = file_name.replaceAll(" ","");
        file_name = file_name.replaceAll(".","");
        file_name = file_name.replaceAll(":","");
        image.setAttribute("src", `/static/images/portrait/${file_name}.jpg`);
        image.setAttribute("onerror", "this.onerror=null;this.src='../static/images/portrait/PLACEHOLDER.jpg'");
        let title = document.createElement("p");
        title.setAttribute("id", "log-movie-title");
        title.innerText = movie_title;
        first_half.appendChild(image);
        first_half.appendChild(title);
        log.appendChild(first_half);

        let second_half = document.createElement("div");
        second_half.classList.add("log-second-half");
        let date = document.createElement("p");
        date.setAttribute("id", "log-watch-date");
        date.innerText = history[i].last_watched;
        let buttons = document.createElement("div");
        buttons.classList.add("log-button-group");

        let like_button = document.createElement("button");
        like_button.setAttribute("id", "log-like");
        like_button.setAttribute("data-movie", movie_title);
        let like_deselect = document.createElement("span");
        like_deselect.setAttribute("id", "log-like-icon-deselect");
        like_deselect.classList.add("fa-regular");
        like_deselect.classList.add("fa-thumbs-up");
        let like_select = document.createElement("span");
        like_select.setAttribute("id", "log-like-icon-select");
        like_select.classList.add("fa-solid");
        like_select.classList.add("fa-thumbs-up");
        like_button.appendChild(like_deselect);
        like_button.appendChild(like_select);

        let dislike_button = document.createElement("button");
        dislike_button.setAttribute("id", "log-dislike");
        dislike_button.setAttribute("data-movie", movie_title);
        let dislike_deselect = document.createElement("span");
        dislike_deselect.setAttribute("id", "log-dislike-icon-deselect");
        dislike_deselect.classList.add("fa-regular");
        dislike_deselect.classList.add("fa-thumbs-down");
        let dislike_select = document.createElement("span");
        dislike_select.setAttribute("id", "log-dislike-icon-select");
        dislike_select.classList.add("fa-solid");
        dislike_select.classList.add("fa-thumbs-down");
        dislike_button.appendChild(dislike_deselect);
        dislike_button.appendChild(dislike_select);

        let delete_button = document.createElement("button");
        delete_button.setAttribute("id", "log-delete");
        delete_button.setAttribute("data-id", i);
        let trash_can = document.createElement("span");
        trash_can.setAttribute("id", "delete-log");
        trash_can.classList.add("fa-solid");
        trash_can.classList.add("fa-trash-can");
        delete_button.appendChild(trash_can);

        if (history[i].opinion == "like") {
            like_deselect.classList.add("hidden");
            dislike_select.classList.add("hidden");
        }

        if (history[i].opinion == "dislike") {
            like_select.classList.add("hidden");
            dislike_deselect.classList.add("hidden");
        }

        if (history[i].opinion == "unknown") {
            like_select.classList.add("hidden");
            dislike_select.classList.add("hidden");
        }

        like_button.addEventListener("click", () => {
            if (!like_select.classList.contains("hidden")) {
                like_select.classList.add("hidden");
                like_deselect.classList.remove("hidden");
                dislike_select.classList.add("hidden");
                dislike_deselect.classList.remove("hidden");
                let name = like_button.getAttribute("data-movie");
                for (let i = 0; i < history.length; i++) {
                    if (history[i].name == name) {
                        history[i].opinion = "unknown";
                    }
                }
            }
            else {
                like_select.classList.remove("hidden");
                like_deselect.classList.add("hidden");
                dislike_select.classList.add("hidden");
                dislike_deselect.classList.remove("hidden");
                let name = like_button.getAttribute("data-movie");
                for (let i = 0; i < history.length; i++) {
                    if (history[i].name == name) {
                        history[i].opinion = "like";
                    }
                }
            }
            localStorage.removeItem("users");
            localStorage.setItem("users", JSON.stringify(profiles));
        });

        dislike_button.addEventListener("click", () => {
            if (!dislike_select.classList.contains("hidden")) {
                like_select.classList.add("hidden");
                like_deselect.classList.remove("hidden");
                dislike_select.classList.add("hidden");
                dislike_deselect.classList.remove("hidden");
                let name = dislike_button.getAttribute("data-movie");
                for (let i = 0; i < history.length; i++) {
                    if (history[i].name == name) {
                        history[i].opinion = "unknown";
                    }
                }
            }
            else {
                like_select.classList.add("hidden");
                like_deselect.classList.remove("hidden");
                dislike_select.classList.remove("hidden");
                dislike_deselect.classList.add("hidden");
                let name = dislike_button.getAttribute("data-movie");
                for (let i = 0; i < history.length; i++) {
                    if (history[i].name == name) {
                        history[i].opinion = "dislike";
                    }
                }
            }
            localStorage.removeItem("users");
            localStorage.setItem("users", JSON.stringify(profiles));
        });

        delete_button.addEventListener("click", () => {
            let index = delete_button.getAttribute("data-id");
            history.splice(index, 1);
            localStorage.removeItem("users");
            localStorage.setItem("users", JSON.stringify(profiles));
            delete_button.parentElement.parentElement.parentElement.remove();
        });

        buttons.appendChild(like_button);
        buttons.appendChild(dislike_button);
        buttons.appendChild(delete_button);

        second_half.appendChild(date);
        second_half.appendChild(buttons);

        log.appendChild(second_half);
        watch_history.appendChild(log);
    }
}
load_watch_history();