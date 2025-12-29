let choose_profile_link = document.getElementById("choose-profile-link");

let success_message = localStorage.getItem("show_toast");
let toast = document.getElementById("toast");

if (success_message) {
    success_message = JSON.parse(success_message);
    toast.innerHTML = success_message.message;
    toast.classList.add("toast-success");
    toast.classList.add("toast-show");
    setTimeout(function(){ toast.classList.remove("toast-show") }, 2900);
    localStorage.removeItem("show_toast");
}

choose_profile_link.addEventListener("click", () => {
    choose_profile_popup.classList.remove("hidden");
});

async function load_movies() {
    await fetch_movies();
}