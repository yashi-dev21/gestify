// static/script.js (full)
const videoElement = document.getElementById("input_video");
const canvasElement = document.getElementById("output_canvas");
const canvasCtx = canvasElement.getContext("2d");
const predictionText = document.getElementById("prediction_text");
const signGif = document.getElementById("sign_gif");
const startSpeechBtn = document.getElementById("startSpeechBtn");
const speechResult = document.getElementById("speechResult");
const speakPredictionBtn = document.getElementById("speakPredictionBtn");

// GIF mapping fallback (words)
const WORD_TO_GIF = {
  "HELLO": "/static/animations/hello.gif",
  "YES": "/static/animations/yes.gif",
  "NO": "/static/animations/no.gif",
  "THANKS": "/static/animations/thanks.gif",
  "THANK YOU": "/static/animations/thanks.gif"
};

// when backend returns A-Z label we will try to find /static/animations/A.gif
function gifForLabel(label){
  if(!label) return null;
  const up = String(label).toUpperCase().trim();
  // direct A-Z gif path
  const azPath = `/static/animations/${up}.gif`;
  // word mapping fallback
  if(WORD_TO_GIF[up]) return WORD_TO_GIF[up];
  return azPath;
}

// speak text via browser (SpeechSynthesis)
function speakText(text){
  if(!text) return;
  try {
    const utter = new SpeechSynthesisUtterance(text);
    utter.rate = 1;
    window.speechSynthesis.cancel();
    window.speechSynthesis.speak(utter);
  } catch(e){
    console.warn("TTS error", e);
  }
}

// throttle request rate
let lastSentTime = 0;
const SEND_INTERVAL_MS = 400;

async function sendLandmarksToBackend(landmarks){
  const now = Date.now();
  if(now - lastSentTime < SEND_INTERVAL_MS) return;
  lastSentTime = now;

  try {
    const res = await fetch("/predict", {
      method: "POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify({landmarks})
    });
    // debug
    const data = await res.json();
    console.log("predict response:", data);
    if(data.prediction){
      const label = data.prediction;
      predictionText.textContent = label;
      // speak predicted label
      // (you can disable if not needed)
      // speakText(label);

      // show GIF (A-Z or word)
      const gif = gifForLabel(label);
      if(gif){
        signGif.src = gif;
        signGif.style.display = "block";
      } else {
        signGif.style.display = "none";
      }
    } else {
      // show error or clear
      predictionText.textContent = "-";
      signGif.style.display = "none";
      if(data.error) console.warn("Predict error:", data.error);
    }
  } catch(err){
    console.error("Error sending landmarks:", err);
  }
}

// Mediapipe hands setup
const hands = new Hands({
  locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
});
hands.setOptions({
  maxNumHands: 1,
  modelComplexity: 1,
  minDetectionConfidence: 0.6,
  minTrackingConfidence: 0.6
});
hands.onResults(onResults);

function onResults(results){
  canvasElement.width = videoElement.videoWidth;
  canvasElement.height = videoElement.videoHeight;
  canvasCtx.save();
  canvasCtx.clearRect(0,0,canvasElement.width, canvasElement.height);
  canvasCtx.drawImage(results.image, 0, 0, canvasElement.width, canvasElement.height);

  if(results.multiHandLandmarks && results.multiHandLandmarks.length > 0){
    const landmarks = results.multiHandLandmarks[0];
    drawConnectors(canvasCtx, landmarks, HAND_CONNECTIONS, {color:"#00FF00", lineWidth:2});
    drawLandmarks(canvasCtx, landmarks, {color:"#FF0000", lineWidth:1});

    // flatten to 63 floats
    const flat = [];
    landmarks.forEach(lm => flat.push(lm.x, lm.y, lm.z));
    if(flat.length === 63){
      console.log("Landmarks length OK; sending...");
      sendLandmarksToBackend(flat);
    }
  }
  canvasCtx.restore();
}

const camera = new Camera(videoElement, {
  onFrame: async () => await hands.send({image: videoElement}),
  width: 640,
  height: 480
});
camera.start().then(()=>console.log("Camera started")).catch(e=>console.error(e));

// ------- Speech → Sign using Web Speech API (browser) -------
let recognition = null;
if('webkitSpeechRecognition' in window || 'SpeechRecognition' in window){
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  recognition = new SR();
  recognition.lang = 'en-US';
  recognition.interimResults = false;
  recognition.maxAlternatives = 1;

  recognition.onresult = (evt) => {
    const text = evt.results[0][0].transcript;
    console.log("Speech recognized:", text);
    speechResult.textContent = text;
    // find GIF for the spoken phrase (search keys)
    const normalized = text.trim().toUpperCase();
    // try exact mapping first
    let gif = null;
    for(const key of Object.keys(WORD_TO_GIF)){
      if(normalized.includes(key)) { gif = WORD_TO_GIF[key]; break; }
    }
    if(!gif){
      // if the spoken text is a single letter A..Z, show that GIF
      const first = normalized.charAt(0);
      if(first >= 'A' && first <= 'Z') gif = `/static/animations/${first}.gif`;
    }
    if(gif){
      signGif.src = gif;
      signGif.style.display = "block";
    } else {
      signGif.style.display = "none";
      console.log("No GIF mapped for spoken text");
    }
  };

  recognition.onerror = (e) => {
    console.warn("Speech error", e);
    speechResult.textContent = "Could not understand speech";
  };
} else {
  startSpeechBtn.disabled = true;
  startSpeechBtn.textContent = "Speech Recognition Unsupported";
}

// Connect the UI button
startSpeechBtn.addEventListener("click", () => {
  if(!recognition) return;
  speechResult.textContent = "Listening...";
  recognition.start();
});

// Speak prediction button
speakPredictionBtn.addEventListener("click", () => {
  const txt = predictionText.textContent;
  if(txt && txt !== "-") speakText(txt);
});
