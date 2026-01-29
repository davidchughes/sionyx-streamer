// ---- ENDPOINTS ----
// Use window.location.origin since we're served from the same server
//const STREAM_URL = window.location.origin + "/stream_dots";
const STATUS_URL = window.location.origin + "/status";
const CMD_URL    = window.location.origin + "/cmd";

let img = document.getElementById("mjpeg");
const videoContainer = document.getElementById("video-bg");

// const fpsText  = document.getElementById("fps-text");
// const connText = document.getElementById("conn-status");

const btnRecord = document.getElementById("btn-record");
const btnPhoto  = document.getElementById("btn-photo");
const btnEIS    = document.getElementById("btn-eis");

const zoom      = document.getElementById("zoom");
const zoomDisp  = document.getElementById("zoom-disp");

const colorItems=document.querySelectorAll(".color-item");
const btnDarkMode = document.getElementById("btn-darkmode");
const btnStreamMode = document.getElementById("btn-streammode");

const btnReconnect = document.getElementById("btn-reconnect");

const LONG_PRESS_MS = 1200;

let reconnectPressTimer = null;
let reconnectLongPressFired = false;

const STREAM_ENHANCED = window.location.origin + "/stream_dots";
const STREAM_RAW      = window.location.origin + "/stream";

let currentStreamURL = STREAM_ENHANCED;

const noSleep = new NoSleep();
let noSleepEnabled = false;

let status = { zoom:100, recording:false, eis:false, connected:false };

let frames=0;
let zeroFpsCount=0;

// ---- RESTORE STREAM ----
img.src = currentStreamURL;

function restartStream() {
    console.log("[STREAM] Nuclear restart");

    // 1. Halt the watchdog timers for a few seconds
    lastFrameTime = performance.now() + 5000; 
    zeroFpsCount = -10;

    // 2. Kill the container's contents entirely
    videoContainer.innerHTML = ""; 

    // 3. Create a totally new element
    const newImg = new Image();
    newImg.id = "mjpeg";
    
    // 4. Force a Query String (NOT a fragment #)
    // Browsers ignore # for network requests, but ? forces a new socket
    const url = new URL(currentStreamURL);
    //url.searchParams.set('t', Date.now());
    
    newImg.src = url.toString();
    newImg.draggable = false;

    // 5. Use the 'onload' event to confirm recovery
    newImg.onload = () => {
        console.log("[STREAM] New stream connected successfully");
        img = newImg; // Re-assign global reference
    };

    videoContainer.appendChild(newImg);
}

function startReconnectPress(){
  if(!status.connected) return;

  reconnectLongPressFired = false;

  reconnectPressTimer = setTimeout(()=>{
    reconnectLongPressFired = true;

    console.log("[UI] Long-press: reset camera connection");

    sendCommand({
      action: "reset_camera_connection"
    });

    // Optional: visual feedback
    btnReconnect.textContent = "⟳⟳";
    setTimeout(()=>btnReconnect.textContent="⟳", 500);

  }, LONG_PRESS_MS);
}

function endReconnectPress(){
  clearTimeout(reconnectPressTimer);

  // Short press → local stream restart
  if(!reconnectLongPressFired && status.connected){
    console.log("[UI] Short press: restart stream");
    restartStream();
  }
}


// ---- FPS COUNT WITH AUTO-RESTART ----
setInterval(()=>{
 if(img.complete && img.naturalHeight!==0)
    frames++;
},33);

setInterval(()=>{
 const fps = frames;
 // fpsText.textContent = fps+" fps";
 
 // Auto-restart if 0 fps for 1 second
 if(fps === 0){
   zeroFpsCount++;
   if(zeroFpsCount >= 2){
     zeroFpsCount = -5
     restartStream();
   }
 }else{
   zeroFpsCount = 0;
 }
 
 frames=0;
},1000);



let lastFrameTime = performance.now();

setInterval(()=>{
  if(img.complete && img.naturalHeight !== 0){
    lastFrameTime = performance.now();
  }
}, 100);

setInterval(()=>{
  if(performance.now() - lastFrameTime > 300){
    console.warn("[STREAM] Frame timeout → restarting");
    lastFrameTime = performance.now();  // give it time to restart
    restartStream();
  }
}, 50);

// ---- Switch streams ----

btnStreamMode.addEventListener("click", ()=>{
  if(!status.connected) return;

  const switchingToRaw = (currentStreamURL === STREAM_ENHANCED);

  currentStreamURL = switchingToRaw
    ? STREAM_RAW
    : STREAM_ENHANCED;

  console.log(
    "[UI] Stream mode:",
    switchingToRaw ? "RAW" : "ENHANCED"
  );

  restartStream();
  syncStreamModeUI();
});

function syncStreamModeUI(){
  if(currentStreamURL === STREAM_RAW){
    btnStreamMode.textContent = "RAW";
    btnStreamMode.classList.add("active");
  }else{
    btnStreamMode.textContent = "FX";
    btnStreamMode.classList.remove("active");
  }

  // disable if disconnected
  btnStreamMode.classList.toggle("disabled", !status.connected);
}



// ---- SEND COMMAND ----
// function sendCommand(cmd){
//  fetch(CMD_URL,{
//   method:"POST",
//   body:JSON.stringify(cmd),
//   headers:{ "Content-Type":"application/json"}
//  }).catch(()=>{});
// }
async function sendCommand(cmd) {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 2000); // 2-second timeout

  try {
    const response = await fetch(CMD_URL, {
      method: "POST",
      body: JSON.stringify(cmd),
      headers: { "Content-Type": "application/json" },
      signal: controller.signal
    });
    clearTimeout(timeoutId);
    return await response.json();
  } catch (err) {
    console.warn("[CMD] Send failed:", err.name === 'AbortError' ? "Timeout" : err);
    // If commands are failing, it's a sign the network is unstable
  }
}
// ---- POLL STATUS ----
async function pollStatus(){
 try{
   const r=await fetch(STATUS_URL,{cache:"no-store"});
   status=await r.json();
 }catch(err){
   status.connected=false;
 }

 syncUIFromStatus();
}
setInterval(pollStatus,1000);
pollStatus();

// ---- REFLECT TOGGLES ----
function syncUIFromStatus(){

 // // connection
 // connText.textContent = status.connected ? "🟢" : "❌";

 // ZOOM SYNC
 const z=status.zoom || 100;
 if(parseInt(zoom.value)!==z){
     zoom.value=z;
 }
 zoomDisp.textContent = (z/100).toFixed(1)+"x";

 // recording indicator - RED outline when recording
 if(status.recording){
    btnRecord.classList.add("recording");
    btnRecord.style.borderColor = "#ff3b30";
 }else{
    btnRecord.classList.remove("recording");
    btnRecord.style.borderColor = "#444";
 }

 // eis indicator - BLUE outline when active
 if(status.eis){
    btnEIS.classList.add("active");
    btnEIS.style.borderColor = "#0a84ff";
 }else{
    btnEIS.classList.remove("active");
    btnEIS.style.borderColor = "#444";
 }

 // color selector
 const mode=status.night_mode || "NightColor";
 colorItems.forEach(c=>{
   if(c.dataset.mode===mode)
        c.classList.add("active");
   else c.classList.remove("active");
 });
 // dark / light enhancement mode
 if(status.dark_mode){
   btnDarkMode.classList.add("active");
   btnDarkMode.textContent = "🌙";
 }else{
   btnDarkMode.classList.remove("active");
   btnDarkMode.textContent = "☀️";
 }
 // reconnect button enabled only when live
 if(status.connected){
   btnReconnect.classList.remove("disabled");
 }else{
   btnReconnect.classList.add("disabled");
 }
 syncStreamModeUI();
}

// ---- USER INTERACTION ----
btnRecord.addEventListener("click",()=>{
 sendCommand({action:"record"});
 // Optimistic update
 status.recording = !status.recording;
 syncUIFromStatus();
});

btnEIS.addEventListener("click",()=>{
 sendCommand({action:"eis"});
 // Optimistic update
 status.eis = !status.eis;
 syncUIFromStatus();
});

btnPhoto.addEventListener("click",()=>{
 sendCommand({action:"photo"});
});

// ---- ZOOM SLIDER ----
let lastZoomSendTime = 0;
const ZOOM_COOLDOWN = 100;

zoom.addEventListener("input",()=>{
 const z = parseInt(zoom.value);
 zoomDisp.textContent = (z/100).toFixed(1)+"x";
 
 const now = performance.now();
 if(now - lastZoomSendTime > ZOOM_COOLDOWN){
   sendCommand({action:"zoom",level:z});
   lastZoomSendTime = now;
 }
});

zoom.addEventListener("change",()=>{
 const z = parseInt(zoom.value);
 sendCommand({action:"zoom",level:z});
});

// ---- COLOR SELECTOR ----
colorItems.forEach(c=>{
 c.addEventListener("click",()=>{
   sendCommand({action:"night_mode", mode:c.dataset.mode});
   // Optimistic update
   status.night_mode = c.dataset.mode;
   syncUIFromStatus();
 });
});

// ---- LIGHT/DARK MODE TOGGLE ----
btnDarkMode.addEventListener("click", ()=>{
  const newMode = !status.dark_mode;

  sendCommand({
    action: "dark_mode",
    value: newMode
  });

  // Optimistic update
  status.dark_mode = newMode;
  syncUIFromStatus();
});


// ---- FULLSCREEN TOGGLE ----
document.getElementById("btn-full")
        .addEventListener("click", async ()=>{
  try {
    if (!document.fullscreenElement) {
      await document.documentElement.requestFullscreen();
      noSleep.enable();
      noSleepEnabled = true;
      console.log("[NoSleep] enabled");
    } else {
      await document.exitFullscreen();
      noSleep.disable();
      noSleepEnabled = false;
      console.log("[NoSleep] disabled");
    }
  } catch (e) {
    console.warn("[Fullscreen/NoSleep] error", e);
  }
});

document.addEventListener("fullscreenchange", ()=>{
  if (!document.fullscreenElement && noSleepEnabled) {
    noSleep.disable();
    noSleepEnabled = false;
    console.log("[NoSleep] disabled (fullscreen exit)");
  }
});
// ---- Refresh Stream ----
btnReconnect.addEventListener("pointerdown", startReconnectPress);
btnReconnect.addEventListener("pointerup", endReconnectPress);
btnReconnect.addEventListener("pointerleave", endReconnectPress);
btnReconnect.addEventListener("pointercancel", endReconnectPress);

function startReconnectPress(){
  if(!status.connected) return;

  btnReconnect.classList.add("arming");

  reconnectLongPressFired = false;
  reconnectPressTimer = setTimeout(()=>{
    reconnectLongPressFired = true;
    btnReconnect.classList.remove("arming");

    sendCommand({ action:"reset_camera_connection" });
  }, LONG_PRESS_MS);
}

function endReconnectPress() {
  btnReconnect.classList.remove("arming");
  clearTimeout(reconnectPressTimer);

  if (!reconnectLongPressFired && status.connected) {
    console.log("[UI] Performing Full Interface Recovery...");
    
    // 1. Restart the stream (using the "Nuclear" method from before)
    restartStream();

    // 2. Immediately force a status poll to re-verify the control channel
    pollStatus(); 

    // 3. Optional: Clear any "stuck" UI states
    syncUIFromStatus();
  }
}

// ---- KEYBOARD SHORTCUTS ----
window.addEventListener("keydown", e => {
    if (e.key === " ") { 
        e.preventDefault(); 
        btnPhoto.click(); 
    }
    else if (e.key === "r") btnRecord.click();
    else if (e.key === "e") btnEIS.click();
    else if (e.key === "d") btnDarkMode.click();
    else if (e.key === "q") location.reload();
});

// ---- kill long touch > right click ----
// document.addEventListener("contextmenu", e => {
//   e.preventDefault();
// }, { passive: false });
document.addEventListener("contextmenu", e => {
  if (
    e.target.closest(".btn") ||
    e.target.closest(".color-item")
  ) {
    e.preventDefault();
  }
}, { passive: false });