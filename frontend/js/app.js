/* =========================================================
   app.js  –  JavaScript for both pages of the demo
   ---------------------------------------------------------
   index.html      -> upload audio, send to ML model, show result
   ornaments.html  -> play example ornaments, add temporary audio

   Each page only runs the part of the code whose HTML exists.
   ========================================================= */


/* =========================================================
   CONFIGURATION  –  CHANGE THE BACKEND URL HERE
   ---------------------------------------------------------
   This is the ONLY place where the backend address is written.
   If your FastAPI / Flask server runs somewhere else, change it.
   ========================================================= */
const API_URL = "http://localhost:8000/predict";

// Path of the configuration file shown by "View Model Configuration"
const MODEL_CONFIG_PATH = "assets/model_config.json";

// Audio formats we accept
const ALLOWED_EXTENSIONS = ["wav", "mp3", "m4a", "ogg"];


/* ---------------------------------------------------------
   Small helper functions used by both pages
   --------------------------------------------------------- */

// Returns true if the file name ends with an allowed audio extension
function isAllowedAudioFile(file) {
  const extension = file.name.split(".").pop().toLowerCase();
  return ALLOWED_EXTENSIONS.includes(extension);
}

// Shows a message in a status box. type = "info", "error" or "success"
function showStatus(element, message, type) {
  element.textContent = message;
  element.className = "status " + (type === "info" ? "" : type);
  element.hidden = false;
}

// Formats seconds as "12.5s"
function formatSeconds(seconds) {
  return Number(seconds).toFixed(1) + "s";
}


/* =========================================================
   PART 1 – MAIN PAGE (index.html): ML MODEL DEMO
   ========================================================= */

const fileInput = document.getElementById("audio-file-input");

if (fileInput) {
  const fileNameText = document.getElementById("selected-file-name");
  const uploadPlayer = document.getElementById("upload-audio-player");
  const analyzeButton = document.getElementById("analyze-button");
  const statusMessage = document.getElementById("status-message");
  const resultSection = document.getElementById("result-section");
  const onlyDetectedCheckbox = document.getElementById("only-detected-checkbox");

  let selectedFile = null;   // the audio file the user picked
  let lastResults = [];      // results from the model (kept for the filter checkbox)
  let segmentEndTime = null; // used to stop playback at the end of a segment

  /* ---------- 1. Audio upload ----------
     When a file is chosen we check its type, show its name,
     and load it into the <audio> player so it can be heard. */
  fileInput.addEventListener("change", function () {
    const file = fileInput.files[0];
    resultSection.hidden = true;
    statusMessage.hidden = true;

    if (!file) {
      selectedFile = null;
      fileNameText.textContent = "No file selected.";
      uploadPlayer.hidden = true;
      return;
    }

    if (!isAllowedAudioFile(file)) {
      selectedFile = null;
      fileNameText.textContent = "No file selected.";
      uploadPlayer.hidden = true;
      showStatus(statusMessage, "Invalid file. Please choose a WAV, MP3, M4A or OGG audio file.", "error");
      return;
    }

    selectedFile = file;
    fileNameText.textContent = "Selected: " + file.name;

    // URL.createObjectURL gives a temporary local link to the file,
    // so the browser can play it without uploading it anywhere.
    uploadPlayer.src = URL.createObjectURL(file);
    uploadPlayer.hidden = false;
  });

  // If the browser cannot decode the file, it is probably not real audio
  uploadPlayer.addEventListener("error", function () {
    if (selectedFile) {
      showStatus(statusMessage, "This file could not be played. It may be damaged or not a valid audio file.", "error");
    }
  });

  /* ---------- 2. Send the audio to the ML model ---------- */
  analyzeButton.addEventListener("click", async function () {
    if (!selectedFile) {
      showStatus(statusMessage, "Please choose an audio file first.", "error");
      return;
    }

    // FormData packs the file like an HTML form upload.
    // The backend reads it from the field named "audio".
    const formData = new FormData();
    formData.append("audio", selectedFile);

    // Loading state
    analyzeButton.disabled = true;
    resultSection.hidden = true;
    showStatus(statusMessage, "Analyzing audio... (pitch extraction with CREPE can take up to a minute)", "info");

    let response;
    try {
      // POST request to the backend: POST /predict
      response = await fetch(API_URL, { method: "POST", body: formData });
    } catch (networkError) {
      // fetch() throws only when the server cannot be reached at all
      showStatus(statusMessage,
        "Unable to connect to the ML model.\nPlease make sure the backend server is running.", "error");
      analyzeButton.disabled = false;
      return;
    }

    try {
      const data = await response.json();

      if (!response.ok) {
        // The server answered, but with an error (e.g. invalid audio)
        const serverMessage = data.detail || data.error || ("Server error " + response.status);
        showStatus(statusMessage, "The model could not process this audio.\n" + serverMessage, "error");
      } else {
        // ---------- 3. Prediction response ----------
        lastResults = normalizeResults(data);
        if (lastResults.length === 0) {
          showStatus(statusMessage, "The model returned no segments. The audio may be too short or silent.", "error");
        } else {
          statusMessage.hidden = true;
          displayModelInfo(data);
          displayResults(lastResults);
        }
      }
    } catch (parseError) {
      showStatus(statusMessage, "The server sent a response that is not valid JSON.", "error");
    }

    analyzeButton.disabled = false;
  });

  /* ---------- Convert the backend JSON into one simple list ----------
     Preferred backend format (one entry per 1-second segment):
     {
       "results": [
         { "start": 0.0, "end": 1.0,
           "ornaments": { "Meend": { "present": false } } },
         { "start": 0.5, "end": 1.5,
           "ornaments": { "Meend": { "present": true } } }
       ]
     }
     The simpler format  { "timestamp": "1s - 3s", "prediction": "Meend" }
     (or a list of those) is also accepted.

     Output: [ { label, start, end, ornaments: { Meend: {present} } }, ... ] */
  function normalizeResults(data) {
    let items;
    if (Array.isArray(data.results)) items = data.results;
    else if (Array.isArray(data)) items = data;
    else items = [data];

    return items.map(function (item) {
      const start = item.start !== undefined ? item.start : item.start_time;
      const end = item.end !== undefined ? item.end : item.end_time;

      let label = item.timestamp || "—";
      if (start !== undefined && end !== undefined) {
        label = formatSeconds(start) + " – " + formatSeconds(end);
      }

      let ornaments = item.ornaments;
      if (!ornaments && item.prediction) {
        ornaments = {};
        ornaments[item.prediction] = {
          present: item.present !== undefined ? item.present : true
        };
      }

      // "silent" = the segment is mostly silence, so the model did not judge it
      return { label: label, start: start, end: end, silent: item.silent === true, ornaments: ornaments || {} };
    });
  }

  /* ---------- Small line with audio length + standardisation used ---------- */
  function displayModelInfo(data) {
    const info = document.getElementById("result-info");
    const parts = [];
    if (data.duration !== undefined) parts.push("Audio length: " + formatSeconds(data.duration));
    if (data.standardisation) parts.push("Standardisation: " + data.standardisation);
    info.textContent = parts.join("  ·  ");
  }

  /* ---------- Show the summary + the segment table ---------- */
  function displayResults(results) {
    // Collect every ornament name that appears in the results
    const ornamentNames = [];
    results.forEach(function (row) {
      Object.keys(row.ornaments).forEach(function (name) {
        if (!ornamentNames.includes(name)) ornamentNames.push(name);
      });
    });

    displaySummary(results, ornamentNames);
    displayTable(results, ornamentNames);
    resultSection.hidden = false;
    resultSection.scrollIntoView({ behavior: "smooth" });
  }

  // Summary: "Meend – Detected in 4 of 66 segments – found at 14.5s – 17.0s ..."
  function displaySummary(results, ornamentNames) {
    const summaryBox = document.getElementById("result-summary");
    summaryBox.innerHTML = "";

    ornamentNames.forEach(function (name) {
      const detectedRows = results.filter(function (row) {
        return row.ornaments[name] && row.ornaments[name].present;
      });
      // Silent segments are not counted
      const analysedCount = results.filter(function (row) { return !row.silent; }).length;
      const silentCount = results.length - analysedCount;
      const silentNote = silentCount > 0 ? " (" + silentCount + " silent segments skipped)" : "";

      const item = document.createElement("div");
      item.className = "summary-item" + (detectedRows.length > 0 ? " found" : "");

      const title = document.createElement("div");
      title.className = "summary-title";
      title.textContent = name + (detectedRows.length > 0 ? " — Detected" : " — Not detected");
      item.appendChild(title);

      const details = document.createElement("div");
      if (detectedRows.length > 0) {
        details.textContent = "Present in " + detectedRows.length + " of " + analysedCount +
          " analysed segments" + silentNote + ". Found at: " + mergeTimeRanges(detectedRows);
      } else {
        details.textContent = "Not present in any of the " + analysedCount + " analysed segments" + silentNote + ".";
      }
      item.appendChild(details);

      summaryBox.appendChild(item);
    });
  }

  // Overlapping detected segments are joined into one time range,
  // e.g. 14.0–15.0 and 14.5–15.5 become "14.0s – 15.5s"
  function mergeTimeRanges(detectedRows) {
    if (detectedRows[0].start === undefined) {
      return detectedRows.map(function (row) { return row.label; }).join(", ");
    }

    const ranges = [];
    detectedRows.forEach(function (row) {
      const lastRange = ranges[ranges.length - 1];
      if (lastRange && row.start <= lastRange.end) {
        lastRange.end = Math.max(lastRange.end, row.end);
      } else {
        ranges.push({ start: row.start, end: row.end });
      }
    });

    return ranges.map(function (range) {
      return formatSeconds(range.start) + " – " + formatSeconds(range.end);
    }).join(",  ");
  }

  // Table: Time | Meend | Kanswar | Murki | Listen
  function displayTable(results, ornamentNames) {
    const tableHead = document.querySelector("#result-table thead");
    const tableBody = document.querySelector("#result-table tbody");

    tableHead.innerHTML = "";
    const headerRow = document.createElement("tr");
    ["Time"].concat(ornamentNames).concat(["Listen"]).forEach(function (heading) {
      const th = document.createElement("th");
      th.textContent = heading;
      headerRow.appendChild(th);
    });
    tableHead.appendChild(headerRow);

    tableBody.innerHTML = "";
    results.forEach(function (row) {
      const anyPresent = ornamentNames.some(function (name) {
        return row.ornaments[name] && row.ornaments[name].present;
      });

      // Filter: hide rows with no ornament if the checkbox is ticked
      if (onlyDetectedCheckbox.checked && !anyPresent) return;

      const tr = document.createElement("tr");
      if (anyPresent) tr.className = "row-detected";

      const timeCell = document.createElement("td");
      timeCell.textContent = row.label;
      tr.appendChild(timeCell);

      ornamentNames.forEach(function (name) {
        const cell = document.createElement("td");
        const value = row.ornaments[name];
        if (row.silent) {
          cell.textContent = "Silence (not analysed)";
          cell.className = "absent";
        } else if (!value) {
          cell.textContent = "—";
        } else {
          cell.textContent = value.present ? "✔ Present" : "✘ Not present";
          cell.className = value.present ? "present" : "absent";
        }
        tr.appendChild(cell);
      });

      // Button to listen to just this segment of the uploaded audio
      const listenCell = document.createElement("td");
      if (row.start !== undefined) {
        const playButton = document.createElement("button");
        playButton.className = "button button-secondary button-small";
        playButton.textContent = "▶";
        playButton.addEventListener("click", function () {
          playSegment(row.start, row.end);
        });
        listenCell.appendChild(playButton);
      }
      tr.appendChild(listenCell);

      tableBody.appendChild(tr);
    });
  }

  onlyDetectedCheckbox.addEventListener("change", function () {
    if (lastResults.length > 0) displayResults(lastResults);
  });

  /* ---------- Audio playback of one segment ----------
     Jump to the segment start, play, and pause at the segment end. */
  function playSegment(start, end) {
    uploadPlayer.currentTime = start;
    segmentEndTime = end;
    uploadPlayer.play();
  }

  uploadPlayer.addEventListener("timeupdate", function () {
    if (segmentEndTime !== null && uploadPlayer.currentTime >= segmentEndTime) {
      uploadPlayer.pause();
      segmentEndTime = null;
    }
  });

  // If the user drags the player manually, stop the automatic pause
  uploadPlayer.addEventListener("seeking", function () {
    if (segmentEndTime !== null && Math.abs(uploadPlayer.currentTime - segmentEndTime) > 1.5) {
      segmentEndTime = null;
    }
  });

  /* ---------- Model configuration viewer ----------
     Loads assets/model_config.json and shows it in a <pre> block. */
  const configButton = document.getElementById("view-config-button");
  const configDisplay = document.getElementById("config-display");

  configButton.addEventListener("click", async function () {
    // Second click hides it again
    if (!configDisplay.hidden) {
      configDisplay.hidden = true;
      configButton.textContent = "View Model Configuration";
      return;
    }

    try {
      const response = await fetch(MODEL_CONFIG_PATH);
      const config = await response.json();
      configDisplay.textContent = JSON.stringify(config, null, 4); // pretty print
    } catch (error) {
      // Browsers block fetch() on file:// pages, so explain how to fix it
      configDisplay.textContent =
        "Could not load " + MODEL_CONFIG_PATH + ".\n" +
        "Run the site with a local server (see README), or open the file directly.";
    }
    configDisplay.hidden = false;
    configButton.textContent = "Hide Model Configuration";
  });
}


/* =========================================================
   PART 2 – ORNAMENTS PAGE (ornaments.html)
   ========================================================= */

const playButtons = document.querySelectorAll(".play-button");

if (playButtons.length > 0) {

  /* ---------- Playing the permanent example files ----------
     Each button has data-audio="assets/audio/xxx.mp3".
     To change an example, replace the file in assets/audio/. */
  let currentAudio = null;
  let currentButton = null;

  playButtons.forEach(function (button) {
    const originalText = button.textContent;
    const message = button.parentElement.querySelector(".play-message");

    button.addEventListener("click", function () {
      // Clicking the playing button again stops it
      if (currentButton === button) {
        stopCurrentAudio();
        return;
      }
      stopCurrentAudio();

      const audioPath = button.getAttribute("data-audio");
      currentAudio = new Audio(audioPath);   // JavaScript Audio() object
      currentButton = button;
      button.textContent = "■ Stop";
      message.textContent = "";

      currentAudio.play().catch(function () {
        // Happens when the file does not exist yet
        message.textContent = "Audio not found. Please add the file: " + audioPath;
        stopCurrentAudio();
      });

      // Reset the button when the example finishes
      currentAudio.addEventListener("ended", stopCurrentAudio);
    });

    button.dataset.originalText = originalText;
  });

  function stopCurrentAudio() {
    if (currentAudio) currentAudio.pause();
    if (currentButton) currentButton.textContent = currentButton.dataset.originalText;
    currentAudio = null;
    currentButton = null;
  }

  /* ---------- Temporary demonstration audio ----------
     IMPORTANT: JavaScript running in a browser is NOT allowed to save
     files into the website's folders (for security reasons).
     So a recording added here is kept only in browser memory and is
     lost when the page is refreshed.

     To add a recording PERMANENTLY, copy it manually into:
         assets/audio/meend.mp3
         assets/audio/kanswar.mp3
         assets/audio/murki.mp3
     (replace the existing file, keeping the same name). */
  const demoFileInput = document.getElementById("demo-file-input");
  const ornamentSelect = document.getElementById("ornament-select");
  const previewPlayer = document.getElementById("demo-preview-player");
  const addAudioButton = document.getElementById("add-audio-button");
  const addAudioMessage = document.getElementById("add-audio-message");

  // Preview the chosen file immediately
  demoFileInput.addEventListener("change", function () {
    const file = demoFileInput.files[0];
    addAudioMessage.hidden = true;

    if (!file) {
      previewPlayer.hidden = true;
      return;
    }
    if (!isAllowedAudioFile(file)) {
      previewPlayer.hidden = true;
      showStatus(addAudioMessage, "Invalid file. Please choose a WAV, MP3, M4A or OGG audio file.", "error");
      return;
    }
    previewPlayer.src = URL.createObjectURL(file);
    previewPlayer.hidden = false;
  });

  // Add the file to the selected ornament card
  addAudioButton.addEventListener("click", function () {
    const file = demoFileInput.files[0];

    if (!file) {
      showStatus(addAudioMessage, "Please choose an audio file first.", "error");
      return;
    }
    if (!isAllowedAudioFile(file)) {
      showStatus(addAudioMessage, "Invalid file. Please choose a WAV, MP3, M4A or OGG audio file.", "error");
      return;
    }

    const ornament = ornamentSelect.value;  // "meend", "kanswar" or "murki"
    const ornamentLabel = ornamentSelect.options[ornamentSelect.selectedIndex].text;
    const list = document.getElementById("added-" + ornament);

    // Build: file name + its own audio player, inside the ornament card
    const item = document.createElement("div");
    item.className = "added-audio-item";

    const name = document.createElement("div");
    name.textContent = "Added: " + file.name + " (this session only)";
    item.appendChild(name);

    const player = document.createElement("audio");
    player.controls = true;
    player.src = URL.createObjectURL(file);  // temporary link, lives in memory
    item.appendChild(player);

    list.appendChild(item);

    showStatus(addAudioMessage, "Added \"" + file.name + "\" to " + ornamentLabel + ".", "success");

    // Reset the form for the next file
    demoFileInput.value = "";
    previewPlayer.hidden = true;
  });
}
