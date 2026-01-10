async function load_movies() {
    await fetch_movies();
}
load_movies();

let choose_profile_link = document.getElementById("choose-profile-link");
let choose_profile_container = document.getElementById("choose-profile-container");

let show_toast = localStorage.getItem("show_toast");
let toast = document.getElementById("toast");

if (show_toast) {
    let message = JSON.parse(show_toast);
    toast.innerHTML = message.message;
    if (message.type) {
        toast.classList.add("toast-success");
        toast.classList.remove("toast-error");
    }
    else {
        toast.classList.remove("toast-success");
        toast.classList.add("toast-error");
    }
    toast.classList.add("toast-show");
    setTimeout(function(){ toast.classList.remove("toast-show") }, 2900);
    localStorage.removeItem("show_toast");
}

choose_profile_link.addEventListener("click", () => {
    choose_profile_container.classList.remove("hidden");
});