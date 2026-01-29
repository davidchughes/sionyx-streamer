#!/usr/bin/env python3
"""
SiOnyx Aurora Windows Client
Connects to RPi5 server for control and viewing
"""

import socket
import json
import threading
import time
import cv2
import numpy as np
import os
import requests  # Better streaming than urllib

# Try to import TurboJPEG for faster decoding
try:
    from turbojpeg import TurboJPEG
    TURBOJPEG_AVAILABLE = True
except ImportError:
    TURBOJPEG_AVAILABLE = False
    print("[Warning] TurboJPEG not available, using cv2 (slower)")

# Configuration
SERVER_IP = "sionyx.local"  # Change to your RPi IP
CONTROL_PORT = 5000
STREAM_PORT = 8080

# Performance settings
SHOW_OVERLAY = True  # Toggle with 'O' key

# State
status = {}
status_lock = threading.Lock()
running = True
jpeg_decoder = None
last_zoom_time = 0
ZOOM_COOLDOWN = 0.1  # 100ms between zoom commands
local_zoom = 100  # Track zoom locally for immediate updates

def init_jpeg_decoder():
    """Initialize TurboJPEG decoder if available"""
    global jpeg_decoder
    
    if not TURBOJPEG_AVAILABLE:
        print("[Decoder] Using cv2.imdecode (standard)")
        return None
    
    try:
        turbojpeg_lib = None
        if os.name == 'nt':  # Windows
            turbojpeg_lib = r'C:\libjpeg-turbo-gcc64\bin\libturbojpeg.dll'
            if not os.path.exists(turbojpeg_lib):
                print(f"[Decoder] TurboJPEG DLL not found at: {turbojpeg_lib}")
                print("[Decoder] Using cv2.imdecode instead")
                return None
        
        jpeg_decoder = TurboJPEG(turbojpeg_lib)
        print("[Decoder] Using TurboJPEG (faster)")
        return jpeg_decoder
    
    except Exception as e:
        print(f"[Decoder] TurboJPEG init failed: {e}")
        print("[Decoder] Using cv2.imdecode instead")
        return None

def decode_jpeg(jpeg_data):
    """Decode JPEG using TurboJPEG or cv2"""
    if jpeg_decoder is not None:
        try:
            return jpeg_decoder.decode(jpeg_data)
        except:
            pass
    
    # Fallback to cv2
    np_arr = np.frombuffer(jpeg_data, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

def control_client_thread():
    """Maintain control connection and get status updates"""
    global status
    
    print(f"[Control] Connecting to {SERVER_IP}:{CONTROL_PORT}")
    
    while running:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            sock.connect((SERVER_IP, CONTROL_PORT))
            
            print("[Control] Connected")
            
            # Request status every second
            while running:
                try:
                    # Send status request
                    cmd = json.dumps({'action': 'status'}) + '\n'
                    sock.send(cmd.encode())
                    
                    # Receive response
                    data = sock.recv(4096)
                    if not data:
                        break
                    
                    response = json.loads(data.decode().strip())
                    with status_lock:
                        status = response
                    
                    # Sync local_zoom with server status
                    global local_zoom
                    if 'zoom' in response:
                        local_zoom = response['zoom']
                    
                    time.sleep(1)
                
                except socket.timeout:
                    continue
                except Exception as e:
                    print(f"[Control] Error: {e}")
                    break
            
            sock.close()
        
        except Exception as e:
            print(f"[Control] Connection error: {e}")
        
        print("[Control] Disconnected, retrying in 5s...")
        time.sleep(5)

def send_command(action, **kwargs):
    """Send command to server - fire and forget, no response wait"""
    def _send():
        try:
            # Create new socket for each command (fire-and-forget)
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(0.5)  # Short timeout
            sock.connect((SERVER_IP, CONTROL_PORT))
            
            cmd = {'action': action, **kwargs}
            sock.send((json.dumps(cmd) + '\n').encode())
            
            # Don't wait for response - close immediately
            sock.close()
            
            # Print command sent for feedback
            if action == 'zoom':
                level = kwargs.get('level', 100)
                print(f"[Cmd→] Zoom: {level/100:.1f}x")
        
        except Exception as e:
            pass  # Silently fail, don't spam errors
    
    # Send in background thread
    threading.Thread(target=_send, daemon=True).start()

def zoom_in():
    """Increase zoom"""
    global last_zoom_time, local_zoom
    
    # Rate limit zoom commands
    now = time.time()
    if now - last_zoom_time < ZOOM_COOLDOWN:
        return  # Too soon, ignore
    last_zoom_time = now
    
    # Update local zoom immediately
    local_zoom = min(300, local_zoom + 20)
    send_command('zoom', level=local_zoom)

def zoom_out():
    """Decrease zoom"""
    global last_zoom_time, local_zoom
    
    # Rate limit zoom commands
    now = time.time()
    if now - last_zoom_time < ZOOM_COOLDOWN:
        return  # Too soon, ignore
    last_zoom_time = now
    
    # Update local zoom immediately
    local_zoom = max(100, local_zoom - 20)
    send_command('zoom', level=local_zoom)

def toggle_eis():
    """Toggle EIS"""
    send_command('eis')

def toggle_recording():
    """Toggle recording"""
    send_command('record')

def capture_photo():
    """Capture photo"""
    send_command('photo')

def set_night_mode(mode):
    """Set night mode"""
    send_command('night_mode', mode=mode)

def mjpeg_viewer():
    """View MJPEG stream - optimized for minimum latency"""
    global running, SHOW_OVERLAY
    
    stream_url = f"http://{SERVER_IP}:{STREAM_PORT}/stream"
    
    print(f"[Viewer] Connecting to {stream_url}")

    cv2.namedWindow("SiOnyx Aurora - Remote", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("SiOnyx Aurora - Remote", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    
    retry_count = 0
    max_retries = 3
    
    while running and retry_count < max_retries:
        # Open stream with requests (better streaming than urllib)
        try:
            response = requests.get(stream_url, stream=True, timeout=10)
            response.raise_for_status()
            print("[Viewer] Connected to stream")
            retry_count = 0  # Reset on successful connection
        except Exception as e:
            retry_count += 1
            print(f"[Viewer] Connection failed (attempt {retry_count}/{max_retries}): {e}")
            if retry_count >= max_retries:
                print("[Viewer] Max retries reached. Check server status.")
                return
            time.sleep(2)
            continue
        
        # Read MJPEG stream
        bytes_data = bytes()
        frame_count = 0
        last_time = time.time()
        fps_count = 0
        last_frame_time = time.time()
        frames_received = 0
        frames_decoded = 0
        
        try:
            # Stream with minimal buffering
            for chunk in response.iter_content(chunk_size=4096):
                if not running:
                    break
                
                chunk_time = time.time()
                bytes_data += chunk
                
                # CRITICAL: Prevent buffer buildup - discard old frames
                # If buffer > 100KB, skip to most recent JPEG
                if len(bytes_data) > 100000:
                    # Find last JPEG start marker
                    last_jpeg = bytes_data.rfind(b'\xff\xd8')
                    if last_jpeg > 0:
                        print(f"[Viewer] Buffer overflow ({len(bytes_data)} bytes), skipping to latest frame")
                        bytes_data = bytes_data[last_jpeg:]
                
                # Find JPEG boundaries - process ONLY ONE frame per iteration
                # But if multiple frames buffered, skip to the latest
                while len(bytes_data) > 60000:  # If we have >60KB buffered (likely multiple frames)
                    first_jpeg = bytes_data.find(b'\xff\xd8')
                    if first_jpeg == -1:
                        break
                    # Find the end of first frame
                    first_end = bytes_data.find(b'\xff\xd9', first_jpeg)
                    if first_end == -1:
                        break
                    # Check if there's another frame after
                    second_jpeg = bytes_data.find(b'\xff\xd8', first_end)
                    if second_jpeg == -1:
                        break  # Only one frame, process it
                    # Skip first frame, keep the rest
                    bytes_data = bytes_data[first_end+2:]
                    frames_received += 1
                    #print(f"[SKIP] Skipping old frame, buffer has {len(bytes_data)} bytes")
                
                a = bytes_data.find(b'\xff\xd8')  # JPEG start
                b = bytes_data.find(b'\xff\xd9')  # JPEG end
                
                if a != -1 and b != -1:
                    jpg = bytes_data[a:b+2]
                    bytes_data = bytes_data[b+2:]
                    
                    frames_received += 1
                    frame_count += 1
                    fps_count += 1
                    
                    now = time.time()
                    frame_interval = now - last_frame_time
                    last_frame_time = now
                    
                    if frames_received <= 10 or frames_received % 50 == 0:
                        print(f"[RX] Frame #{frames_received} received, interval: {frame_interval*1000:.1f}ms, buffer: {len(bytes_data)} bytes")
                    
                    if frame_count == 1:
                        print("[Viewer] ✓ First frame received!")
                    
                    # Decode and display this frame
                    decode_start = time.time()
                    frame = decode_jpeg(jpg)
                    decode_time = (time.time() - decode_start) * 1000
                    
                    frames_decoded += 1
                    
                    if frames_decoded <= 10 or frames_decoded % 50 == 0:
                        print(f"[DECODE] Frame #{frames_decoded}, decode time: {decode_time:.1f}ms")
                    
                    if frame is not None:
                        h, w = frame.shape[:2]
                        
                        # Add overlay if enabled
                        if SHOW_OVERLAY:
                            with status_lock:
                                s = status.copy()
                            
                            # Connection status
                            conn_color = (0, 255, 0) if s.get('connected') else (0, 0, 255)
                            cv2.putText(frame, f"Camera: {'CONNECTED' if s.get('connected') else 'DISCONNECTED'}", 
                                       (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, conn_color, 2)
                            
                            y = 50
                            if s.get('recording'):
                                cv2.circle(frame, (20, y+5), 8, (0, 0, 255), -1)
                                cv2.putText(frame, "REC", (35, y+10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                                y += 30
                            
                            # EIS
                            eis_color = (0, 255, 0) if s.get('eis') else (128, 128, 128)
                            cv2.putText(frame, f"EIS: {'ON' if s.get('eis') else 'OFF'}", 
                                       (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, eis_color, 1)
                            y += 20
                            
                            # Zoom
                            zoom = s.get('zoom', 100)
                            cv2.putText(frame, f"Zoom: {zoom/100:.1f}x", 
                                       (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                            y += 20
                            
                            # Night mode
                            cv2.putText(frame, f"Mode: {s.get('night_mode', 'Unknown')}", 
                                       (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        
                        # Always show FPS (top right)
                        now = time.time()
                        if now - last_time >= 1.0:
                            fps = fps_count / (now - last_time)
                            print(f"[FPS] {fps:.1f}")
                            last_time = now
                            fps_count = 0
                        else:
                            fps = 0  # Don't show until we have a full second
                        
                        if fps > 0:
                            cv2.putText(frame, f"{fps:.1f} fps", (w-120, 25), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        
                        # Controls help (bottom)
                        if SHOW_OVERLAY:
                            cv2.putText(frame, "R:Rec SPACE:Photo E:EIS +/-:Zoom O:Overlay Q:Quit", 
                                       (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                        
                        # Display
                        display_start = time.time()
                        cv2.imshow('SiOnyx Aurora - Remote', frame)
                        display_time = (time.time() - display_start) * 1000
                        
                        if frames_decoded <= 10 or frames_decoded % 50 == 0:
                            print(f"[DISPLAY] Frame #{frames_decoded}, display time: {display_time:.1f}ms")
                        
                        # Handle keys
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q'):
                            running = False
                            break
                        elif key == ord('r'):
                            toggle_recording()
                        elif key == ord(' '):
                            capture_photo()
                        elif key == ord('e'):
                            toggle_eis()
                        elif key == ord('+') or key == ord('='):
                            zoom_in()
                        elif key == ord('-') or key == ord('_'):
                            zoom_out()
                        elif key == ord('1'):
                            set_night_mode('NightColor')
                        elif key == ord('2'):
                            set_night_mode('Green')
                        elif key == ord('3'):
                            set_night_mode('GrayScale')
                        elif key == ord('o'):
                            SHOW_OVERLAY = not SHOW_OVERLAY
                            print(f"[Overlay] {'ON' if SHOW_OVERLAY else 'OFF'}")
                    else:
                        # Another frame in buffer - skip decoding this old one
                        frames_skipped += 1
                        if frames_skipped <= 10 or frames_skipped % 50 == 0:
                            print(f"[SKIP] Skipped frame #{frames_received} (buffer has more)")
        
        except KeyboardInterrupt:
            running = False
        except Exception as e:
            print(f"[Viewer] Error: {e}")
            retry_count += 1
            if retry_count < max_retries:
                print(f"[Viewer] Reconnecting... (attempt {retry_count}/{max_retries})")
                time.sleep(2)
        finally:
            try:
                response.close()
            except:
                pass
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("="*60)
    print("SiOnyx Aurora Windows Client - Optimized")
    print("="*60)
    print(f"Server: {SERVER_IP}")
    print()
    
    # Initialize JPEG decoder
    init_jpeg_decoder()
    
    # Start control thread
    control_thread = threading.Thread(target=control_client_thread, daemon=True)
    control_thread.start()
    
    # Give control thread time to connect
    time.sleep(2)
    
    print("\nControls:")
    print("  SPACE = Capture photo")
    print("  R = Toggle recording")
    print("  E = Toggle EIS")
    print("  +/- = Zoom in/out")
    print("  O = Toggle overlay")
    print("  1/2/3 = Night mode (Color/Green/Gray)")
    print("  Q = Quit\n")
    
    # Start viewer
    mjpeg_viewer()
    
    print("\nShutting down...")