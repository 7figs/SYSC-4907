let choose_profile_popup = document.getElementById("choose-profile-container");
let choose_profile_close = document.getElementById("choose-profile-close");
let add_profile_close = document.getElementById("add-profile-close");
let add_profile_popup = document.getElementById("add-profile-container");
let add_profile_button = document.getElementById("add-profile-button");
let create_new_profile = document.getElementById("create-new-profile");
let profiles_container = document.getElementById("profiles");

let profiles = JSON.parse(localStorage.getItem("users"));

if (profiles) {
    for (let i = 0; i < profiles.length; i++) {
        let profile = document.createElement("div");
        profile.classList.add("profile");
        profile.setAttribute("id", profiles[i].id);
        let pfp = document.createElement("div");
        pfp.classList.add("pfp");
        let picture = document.createElement("img");
        picture.src = "../static/images/svg/profile.svg";
        picture.alt = "profile";
        picture.style.backgroundColor = profiles[i].colour;
        picture.style.borderRadius = "50%";
        pfp.appendChild(picture);
        profile.appendChild(pfp);
        let name = document.createElement("div");
        name.classList.add("name");
        name.innerText = profiles[i].name;
        profile.appendChild(name);
        profile.addEventListener("click", () => {
            location.assign(`/feed/${profiles[i].id}`);
        });
        profiles_container.appendChild(profile);
        let line = document.createElement("hr");
        profiles_container.appendChild(line);
    }
}

choose_profile_close.addEventListener("click", () => {
    choose_profile_popup.classList.add("hidden");
});

add_profile_button.addEventListener("click", () => {
    choose_profile_popup.classList.add("hidden");
    add_profile_popup.classList.remove("hidden");
});

add_profile_close.addEventListener("click", () => {
    choose_profile_popup.classList.remove("hidden");
    add_profile_popup.classList.add("hidden");
});

create_new_profile.addEventListener("click", () => {
    location.assign("/start");
});