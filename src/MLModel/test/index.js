let title = document.getElementById("title");
let overview = document.getElementById("overview");
let like = document.getElementById("like_button");
let dislike = document.getElementById("dislike_button");
let img = document.getElementById("img");
let tracker = document.getElementById("tracker");

let data = [];
let movies;
let current_movie_index = 0;

/* LOAD CSV FILE WITH MOVIE TITLES AND OVERVIEWS */
async function loadCSV(url) {
  // Fetch the CSV file
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to fetch CSV: ${response.status}`);
  }

  const csvText = await response.text();

  // Parse with Papa Parse
  return new Promise((resolve, reject) => {
    Papa.parse(csvText, {
      header: true,          // first row → object keys
      skipEmptyLines: true,
      complete: results => resolve(results.data),
      error: err => reject(err)
    });
  });
}

async function gather_data() {
    let content = await loadCSV("top500_movies_features.csv")
    movies = content;

    tracker.innerText = `${current_movie_index + 1} / ${movies.length}`;
    let name = movies[current_movie_index].title;
    let description = movies[current_movie_index].overview;
    let file_name = name.toLowerCase();
    file_name = file_name.replaceAll(" ","");
    file_name = file_name.replaceAll(".","");
    file_name = file_name.replaceAll(":","");
    img.setAttribute("src", `images/${file_name}.jpg`);
    img.setAttribute("onerror", "this.onerror=null;this.src='images/PLACEHOLDER.jpg'");
    title.innerText = name;
    overview.innerText = description;

    like.addEventListener("click", () => {
        obj = {
            "movie_id": current_movie_index + 1,
            "opinion": 1
        }
        data.push(obj);
        current_movie_index += 1;
        if (current_movie_index == movies.length) {
            download(objectToCsv(data));
        }
        else {
            tracker.innerText = `${current_movie_index + 1} / ${movies.length}`;
            let name = movies[current_movie_index].title;
            let description = movies[current_movie_index].overview;
            let file_name = name.toLowerCase();
            file_name = file_name.replaceAll(" ","");
            file_name = file_name.replaceAll(".","");
            file_name = file_name.replaceAll(":","");
            img.setAttribute("src", `images/${file_name}.jpg`);
            img.setAttribute("onerror", "this.onerror=null;this.src='images/PLACEHOLDER.jpg'");
            title.innerText = name;
            overview.innerText = description;
        }
    });

    dislike.addEventListener("click", () => {
        obj = {
            "movie_id": current_movie_index + 1,
            "opinion": 0
        }
        data.push(obj);
        current_movie_index += 1;
        if (current_movie_index == movies.length) {
            download(objectToCsv(data));
        }
        else {
            tracker.innerText = `${current_movie_index + 1} / ${movies.length}`;
            let name = movies[current_movie_index].title;
            let description = movies[current_movie_index].overview;
            let file_name = name.toLowerCase();
            file_name = file_name.replaceAll(" ","");
            file_name = file_name.replaceAll(".","");
            file_name = file_name.replaceAll(":","");
            img.setAttribute("src", `images/${file_name}.jpg`);
            img.setAttribute("onerror", "this.onerror=null;this.src='images/PLACEHOLDER.jpg'");
            title.innerText = name;
            overview.innerText = description;
        }
    });
}
gather_data();



/* CONVERT COLLECTED DATA INTO CSV FILE */
const objectToCsv = function (data) {
    
        const csvRows = [];

        const headers = Object.keys(data[0]);
    
        csvRows.push(headers.join(','));
    
        // Loop to get value of each objects key
        for (const row of data) {
            const values = headers.map(header => {
                const val = row[header]
                return `"${val}"`;
            });
    
            // To add, separator between each value
            csvRows.push(values.join(','));
        }

        return csvRows.join('\n');
    };

// const csvData = objectToCsv(data);
const download = (data) => {
    // Create a Blob with the CSV data and type
    const blob = new Blob([data], { type: 'text/csv' });
    
    // Create a URL for the Blob
    const url = URL.createObjectURL(blob);
    
    // Create an anchor tag for downloading
    const a = document.createElement('a');
    
    // Set the URL and download attribute of the anchor tag
    a.href = url;
    a.download = 'download.csv';
    
    // Trigger the download by clicking the anchor tag
    a.click();
}

// download(csvData)

