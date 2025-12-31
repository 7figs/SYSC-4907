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
let user_index;
let toast = document.getElementById("toast");

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

async function load_profile() {
    await load_movies();
    
}