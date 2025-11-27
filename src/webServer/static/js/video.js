const video = document.getElementById("video");

// Dynamically build the URL based on browser address
const hostname = window.location.hostname;  // example: 192.168.2.27
const port = window.location.port || 8000;  // fallback if no port visible
const src = "http://172.17.52.25:8000/stream.m3u8";

console.log("Using dynamic video source:", src);

if (Hls.isSupported()) {
    const hls = new Hls();
    hls.loadSource(src);
    hls.attachMedia(video);
} else if (video.canPlayType("application/vnd.apple.mpegurl")) {
    video.src = src;  // Safari native support
}

document.addEventListener("fullscreenchange", handleFullscreenChange);

function handleFullscreenChange() {
    if (document.fullscreenElement === video) {
        video.classList.remove("rounded-corner");
    } else if (document.fullscreenElement === null) {
        video.classList.add("rounded-corner");
    }
}