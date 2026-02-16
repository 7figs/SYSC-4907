import messages from "../messages.json" with {type: "json"};

let choose_profile_popup = document.getElementById("choose-profile-container");
let choose_profile_close = document.getElementById("choose-profile-close");
let add_profile_close = document.getElementById("add-profile-close");
let add_profile_popup = document.getElementById("add-profile-container");
let add_profile_button = document.getElementById("add-profile-button");
let create_new_profile = document.getElementById("create-new-profile");
let profiles_container = document.getElementById("profiles");
let uploaded_profile = document.getElementById("upload_profile");
let submit_button = document.getElementById("submit_upload");
let load_profiles_button = document.getElementById("load-profiles");
let load_profiles_popup = document.getElementById("username-password-container");
let load_profiles_close = document.getElementById("username-password-close");
let load_data_button = document.getElementById("load-data-button");
let password = document.getElementById("password");
let username = document.getElementById("username");

uploaded_profile.addEventListener("change", function() {
    submit_button.disabled = !this.value;
});

let profiles = JSON.parse(localStorage.getItem("users"));

if (profiles) {
    for (let i = 0; i < profiles.length; i++) {
        let profile = document.createElement("div");
        profile.classList.add("profile");
        profile.setAttribute("id", profiles[i].id);
        let pfp = document.createElement("div");
        pfp.classList.add("pfp");
        let picture = document.createElement("img");
        picture.src = "/static/images/svg/profile.svg";
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

load_data_button.addEventListener("click", async () => {
    let password_value = password.value;
    let username_value = username.value;
    let id = `${username_value}`;
    id = sha256(id);
    let data = await fetch(`/load?i=${id}`);
    data = await data.json();
    console.log(data)
    let blob = data[1];
    let salt = data[0];
    let key = deriveKey(password_value, salt);
    let user_data = decryptData(blob, key);
    localStorage.removeItem("users");
    localStorage.setItem("users", user_data);
    localStorage.removeItem("id");
    localStorage.setItem("id", id);
    localStorage.removeItem("key");
    localStorage.setItem("key", key);
    location.assign("/");
});

submit_button.addEventListener("click", () => {
    let file = uploaded_profile.files[0];
    let reader = new FileReader();
    reader.addEventListener("load", (e) => {
        let test = {};
        test = JSON.parse(e.target.result);
        if ("name" in test && "colour" in test && "tree" in test && "watch_history" in test) {
            if (profiles) {
                test.id = profiles.length;
            }
            else {
                profiles = [];
                test.id = 0;
            }
            profiles.push(test);
            localStorage.removeItem("users");
            localStorage.setItem("users", JSON.stringify(profiles));
            let message = {
                "type": 1,
                "message": messages.profile_upload_success
            }
            localStorage.setItem("show_toast", JSON.stringify(message));
            location.assign("/");
        }
        else {
            let message = {
                "type": 0,
                "message": messages.profile_upload_fail
            }
            localStorage.setItem("show_toast", JSON.stringify(message));
            window.location.reload();
        }
    });

    reader.readAsText(file);
});

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

load_profiles_button.addEventListener("click", () => {
    load_profiles_popup.classList.remove("hidden");
});

load_profiles_close.addEventListener("click", () => {
    load_profiles_popup.classList.add("hidden");
});