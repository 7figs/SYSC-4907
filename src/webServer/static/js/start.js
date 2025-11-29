async function fetch_movies() {
    localStorage.removeItem("movies");
    let movies = await fetch("/movies");
    movies = await movies.json();
    localStorage.setItem("movies", JSON.stringify(movies));
}
fetch_movies();