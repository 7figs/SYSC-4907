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