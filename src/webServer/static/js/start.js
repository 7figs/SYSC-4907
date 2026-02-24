async function fetch_movies() {
    sessionStorage.removeItem("movies");
    let movies = await fetch("/movies");
    movies = await movies.json();
    sessionStorage.setItem("movies", JSON.stringify(movies));
}

function deriveKey(password, salt) {
    return CryptoJS.PBKDF2(password, salt, {
        keySize: 256 / 32,
        iterations: 100000
    });
}

function encryptData(data, key) {
    let encrypted = CryptoJS.AES.encrypt(
        JSON.stringify(data),
        key.toString()
    );

    return encrypted.toString();
}

function decryptData(ciphertext, key) {
    let decrypted = CryptoJS.AES.decrypt(ciphertext, key.toString());

    return JSON.parse(
        decrypted.toString(CryptoJS.enc.Utf8)
    );
}


