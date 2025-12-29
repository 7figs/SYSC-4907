let pathname = window.location.pathname;
let segments = pathname.split('/');
let userId = Number(segments[segments.length - 1]);
let color;
let profile_exists = false;
let pfp = document.getElementById("pfp");
let dorpdown_options = document.getElementById("dropdown-options");
let settings = document.getElementById("settings");
let switch_profiles = document.getElementById("switch-profiles");

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

