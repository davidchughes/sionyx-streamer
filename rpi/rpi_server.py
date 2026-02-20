#!/usr/bin/env python3
"""
SiOnyx Aurora RPi5 Server with STREAMLINED ADMD Dot Detection
- STREAMLINED PROCESSING: Chroma denoise → Shadow desat → Soft-knee black
- MULTI-THREADED R/G/B ADMD processing
- UINT16 rolling accumulator with running sum
- THREADED overlay blur operations
- Natural, film-like results

STREAMLINED PIPELINE:
1. Temporal Averaging (8 frames, uint16 sum)
2. Chroma Denoising (25×17 YUV blur - removes color splotches)
3. Shadow Desaturation (LAB space - neutralizes dark green)
4. Soft-Knee Black Level (gradient-preserving - no banding)
5. Multi-threaded ADMD (parallel R/G/B)

PERFORMANCE:
- Processing: ~200ms/frame (5 FPS)
- Natural appearance, no over-processing
"""

import socket
import requests
import threading
import time
import json
import struct
import subprocess
import cv2
import numpy as np
import queue
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn

import os
print("[HTTP] CWD:", os.getcwd())

# Configuration
CAMERA_IP = "192.168.0.1"
CAMERA_INTERFACE = "wlan0"
UDP_PORT = 34001
CONTROL_PORT = 5000
STREAM_PORT = 8080

# Dot detection config
CAM_NUMBER = 5
DOT_MULTICAST_IP = f"224.1.2.{CAM_NUMBER}"
DOT_MULTICAST_PORT = 5000
HOSTNAME = socket.gethostname()

# Global state - Camera
camera_connected = False
liveview_active = False
recording = False
eis_enabled = False
zoom_level = 100
night_mode = "NightColor"

# Global state - Dot detection
dot_detection_enabled = True
dot_min_radius = 0
dot_max_dots = 5
dot_min_z_score = 7.0
dot_min_distance = 10
detect_dark_spots = False
dot_edge_crop = 4
dark_mode = True  # Enhancement mode toggle
dark_mode_lock = threading.Lock()

# Frame storage
latest_frame = None
latest_timestamp = 0.0
latest_frame_id = 0
frame_lock = threading.Lock()
status_lock = threading.Lock()

# Dot detection storage
latest_dots_image = None
dots_image_lock = threading.Lock()

# Frame history for temporal filtering
HISTORY_LEN = 8

# Pipeline queues - 2-stage architecture
processed_queue = queue.Queue(maxsize=2)  # Stage1 output → Stage2 input

# Thread pool for parallel ADMD processing
executor = ThreadPoolExecutor(max_workers=3)

# Pending state changes
pending_zoom = None
pending_eis = None
pending_night_mode = None
frames_since_change = 0
FRAMES_TO_CONFIRM = 2

# UDP socket for broadcasting detection results
udp_broadcast_socket = None


# ============================================================================
# STREAMLINED PROCESSING CLASSES
# ============================================================================

class RollingAccumulator:
    """uint16 rolling accumulator for high-SNR live stacking."""

    def __init__(self, maxlen, shape):
        self.maxlen = maxlen
        self.buffer = [np.zeros(shape, dtype=np.uint8) for _ in range(maxlen)]
        self.running_sum = np.zeros(shape, dtype=np.uint16)
        self.index = 0
        self.filled = 0

    def update(self, frame):
        """Standard rolling sum: Add new, subtract oldest."""
        if self.filled == self.maxlen:
            self.running_sum -= self.buffer[self.index].astype(np.uint16)

        self.buffer[self.index] = frame
        self.running_sum += frame.astype(np.uint16)

        self.index = (self.index + 1) % self.maxlen
        if self.filled < self.maxlen:
            self.filled += 1

    def get_mean(self):
        """Returns the high-precision average as uint8."""
        if self.filled == 0:
            return None
        return (self.running_sum // self.filled).astype(np.uint8)

    def get_sum(self):
        return (self.running_sum).astype(np.uint16)

    def __len__(self):
        return self.filled


class StarFieldCleaner:
    def __init__(self, chroma_kernel=(25, 17)):
        self.kernel = chroma_kernel

    def process(self, frame_u8):
        lab = cv2.cvtColor(frame_u8, cv2.COLOR_BGR2LAB).astype(np.float32)
        l, a, b = cv2.split(lab)
        a_smooth = cv2.blur(a, self.kernel)
        b_smooth = cv2.blur(b, self.kernel)
        lab_clean = cv2.merge([l, a_smooth, b_smooth])
        return cv2.cvtColor(lab_clean.astype(np.uint8), cv2.COLOR_LAB2BGR)


class FastContrastProcessor:
    """Contrast processor with manual dark mode toggle"""

    def __init__(self, boost_factor=8, gamma=1.4, ema_alpha=0.1, dark_mode=True):
        self.boost = boost_factor
        self.gamma = gamma
        self.alpha = ema_alpha
        self.dark_mode = dark_mode
        self.stats_ema = None
        self._luts = None
        self._last_stats = None
        self.strength = 1.0 if dark_mode else 0.0

    def _estimate_stats(self, frame_u8):
        small = frame_u8[::16, ::16]
        b_avg = np.mean(small[:,:,0])
        g_avg = np.mean(small[:,:,1])
        r_avg = np.mean(small[:,:,2])
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        black_val = np.percentile(gray, 10)
        return np.array([b_avg, g_avg, r_avg, black_val])

    def _generate_luts(self, stats):
        b_avg, g_avg, r_avg, black_val = stats
        g_avg += 8
        self.strength = 1.0 if self.dark_mode else 0.0
        avg_all = (b_avg + g_avg + r_avg) / 3.0

        b_gain_full = 1.0 + ((avg_all / (b_avg + 1e-6)) - 1.0) * 0.5
        g_gain_full = 1.0 + ((avg_all / (g_avg + 1e-6)) - 1.0) * 0.5
        r_gain_full = 1.0 + ((avg_all / (r_avg + 1e-6)) - 1.0) * 0.5

        b_gain = 1.0 + (b_gain_full - 1.0) * self.strength
        g_gain = 1.0 + (g_gain_full - 1.0) * self.strength
        r_gain = 1.0 + (r_gain_full - 1.0) * self.strength
        b_gain, g_gain, r_gain = [np.clip(g, 0.8, 1.2) for g in [b_gain, g_gain, r_gain]]

        luts = []
        x = np.arange(2048, dtype=np.float32)
        active_floor = (black_val * self.boost) * self.strength
        active_gamma = 1.0 + (self.gamma - 1.0) * self.strength

        for gain in [b_gain, g_gain, r_gain]:
            x_val = x * gain
            norm_x = (x_val - active_floor) / (2047.0 - active_floor)
            safe_x = np.clip(norm_x, 0, 1)
            gamma_x = np.power(safe_x, active_gamma)

            shoulder_start = 0.9
            mask = gamma_x > shoulder_start
            gamma_x[mask] = shoulder_start + (1.0 - shoulder_start) * \
                            np.sin((gamma_x[mask] - shoulder_start) / (1.0 - shoulder_start) * (np.pi / 2))
            luts.append((gamma_x * 255.0).astype(np.uint8))

        return luts

    def process(self, frame_u8):
        current_stats = self._estimate_stats(frame_u8)

        if self.stats_ema is None:
            self.stats_ema = current_stats
        else:
            self.stats_ema = (self.stats_ema * (1.0 - self.alpha)) + (current_stats * self.alpha)

        stats_check = (self.stats_ema * 2).astype(int)
        if not np.array_equal(stats_check, self._last_stats) or self._luts is None:
            self._luts = self._generate_luts(self.stats_ema)
            self._last_stats = stats_check

        f16 = frame_u8.astype(np.uint16) * self.boost
        b, g, r = cv2.split(f16)
        return cv2.merge([self._luts[0][b], self._luts[1][g], self._luts[2][r]])


# ============================================================================
# OPTIMIZED UINT16 ADMD (NO FLOAT CONVERSIONS)
# ============================================================================

# ============================================================================
# ADMD
# ============================================================================

# def get_directional_kernels_uint16(k):
#     size = 3 + 2 * k
#     center = size // 2
#     kernels = []
#     #directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
#     directions = [
#         (0, 1), (1, 1), (1, 0), (1, -1), 
#         (0, -1), (-1, -1), (-1, 0), (-1, 1)
#     ]
#     for dx, dy in directions:
#         kern = np.zeros((size, size), np.int16)
#         for i in range(1, center + 1):
#             kern[center + (i * dy), center + (i * dx)] = 1
#         kernels.append(kern)
#     kernel_sum = int(np.sum(kernels[0]))
#     shift_amount = 1 if kernel_sum == 2 else 0
#     return kernels, shift_amount

# def ADMD_single_channel_uint16(image, _k=2):
#     img_half = image >> 1
#     kernels, shift = get_directional_kernels_uint16(_k+1)
#     diffs = []
#     for kern in kernels:
#         directional_sum = cv2.filter2D(img_half, cv2.CV_16U, kern, borderType=cv2.BORDER_REPLICATE)
#         if shift > 0:
#             directional_mean = directional_sum >> shift
#         else:
#             directional_mean = directional_sum
#         diff = cv2.absdiff(img_half, directional_mean)
#         diffs.append(diff)
#     admd_map = diffs[0]
#     for i in range(1, len(diffs)):
#         admd_map = cv2.min(admd_map, diffs[i])
#     low_contrast_dots = cv2.min(admd_map, 255).astype(np.uint16)
#     ADMD_image = ((low_contrast_dots * low_contrast_dots) >> 1)
#     return ADMD_image

# ============================================================================
# OPTIMIZED UINT16 ADMD - MODULE LEVEL CACHES
# ============================================================================

# Thread-safe caches for kernels and buffers
_PRECOMPUTED_KERNELS = {}
_KERNEL_LOCK = threading.Lock()
_ADMD_BUFFERS = {}
_BUFFER_LOCK = threading.Lock()
direction_executor = ThreadPoolExecutor(max_workers=8)  # For 8 directions within each channel

def get_or_create_kernels(k):
    """
    Thread-safe kernel cache.
    Precomputes directional kernels once and reuses them.
    """
    if k not in _PRECOMPUTED_KERNELS:
        with _KERNEL_LOCK:
            # Double-check pattern for thread safety
            if k not in _PRECOMPUTED_KERNELS:
                size = 3 + 2 * k
                center = size // 2
                kernels = []
                
                # 8 directional kernels (including diagonals)
                directions = [
                    (0, 1), (1, 1), (1, 0), (1, -1), 
                    (0, -1), (-1, -1), (-1, 0), (-1, 1)
                ]
                
                for dx, dy in directions:
                    kern = np.zeros((size, size), np.int16)
                    for i in range(1, center + 1):
                        kern[center + (i * dy), center + (i * dx)] = 1
                    kernels.append(kern)
                
                kernel_sum = int(np.sum(kernels[0]))
                shift_amount = 1 if kernel_sum == 2 else 0
                
                _PRECOMPUTED_KERNELS[k] = (kernels, shift_amount)
                print(f"[ADMD] Precomputed kernels for k={k}, size={size}x{size}, shift={shift_amount}")
    
    return _PRECOMPUTED_KERNELS[k]

def get_admd_buffers(shape):
    """
    Thread-safe buffer cache.
    Pre-allocates arrays to avoid repeated memory allocation.
    Returns 8 diff buffers (one per direction).
    """
    key = shape
    if key not in _ADMD_BUFFERS:
        with _BUFFER_LOCK:
            if key not in _ADMD_BUFFERS:
                # Pre-allocate 8 diff buffers
                _ADMD_BUFFERS[key] = [
                    np.empty(shape, dtype=np.uint16) for _ in range(8)
                ]
                print(f"[ADMD] Pre-allocated buffers for shape {shape}")
    
    return _ADMD_BUFFERS[key]

def ADMD_single_channel_uint16(image, _k=0):
    """
    Optimized ADMD with:
    - Precomputed kernels (no repeated kernel generation)
    - Vectorized minimum operation (faster than sequential cv2.min)
    - Fused operations (fewer intermediate arrays)
    - Pre-allocated buffers (reduced memory allocation overhead)
    Each of the 8 filter2D operations runs in parallel.
    
    Args:
        image: uint16 input image (single channel)
        _k: Kernel size parameter (0 = 3x3, 1 = 5x5, 2 = 7x7)
    
    Returns:
        uint16 ADMD response map
    """
    img_half = image >> 1
    kernels, shift = get_or_create_kernels(_k)
    
    def compute_single_direction(kern):
        """Single direction computation - runs in parallel"""
        directional_sum = cv2.filter2D(
            img_half, 
            cv2.CV_16U, 
            kern, 
            borderType=cv2.BORDER_REPLICATE
        )
        
        if shift > 0:
            return cv2.absdiff(img_half, directional_sum >> shift)
        else:
            return cv2.absdiff(img_half, directional_sum)
    
    # Submit all 8 directions to thread pool
    futures = [
        direction_executor.submit(compute_single_direction, kern) 
        for kern in kernels
    ]
    
    # Gather results
    diffs = [f.result() for f in futures]
    
    # Vectorized minimum
    admd_map = np.minimum.reduce(diffs)
    low_contrast_dots = np.minimum(admd_map, 255).astype(np.uint16)
    return (low_contrast_dots * low_contrast_dots) >> 1


# ============================================================================
# TOPHAT AND PEAK DETECTION
# ============================================================================

def initialize_detection_kernels():
    """Initialize TopHat detection kernels (precomputed for speed)"""
    kernels = []
    for ksize in range(20):
        TopHat_size = 3 + ksize * 2
        kernel = np.zeros((2 * TopHat_size + 1, 2 * TopHat_size + 1), np.uint8)
        kernel[0, 0] = 1
        kernel[2 * TopHat_size, 2 * TopHat_size] = 1
        kernel[TopHat_size, 0] = 1
        kernel[2 * TopHat_size, 0] = 1
        kernel[0, TopHat_size] = 1
        kernel[0, 2 * TopHat_size] = 1
        kernel[TopHat_size, 2 * TopHat_size] = 1
        kernel[2 * TopHat_size, TopHat_size] = 1
        kernels.append(kernel)
    return kernels

def TopHat_single_channel(image, _k=0):
    """Apply TopHat to a single channel"""
    _ksize = 3 + 2*_k
    _muImg = cv2.blur(image, (_ksize, _ksize), cv2.BORDER_ISOLATED)

    kernel = TopHat__KERNELS[_k]
    _temp3 = cv2.dilate(_muImg, kernel, borderValue=cv2.BORDER_ISOLATED)
    _temp_thresh3 = cv2.subtract(image, _temp3)

    low_contrast_dots = cv2.min(cv2.divide(_temp_thresh3, 2), 255)
    return (low_contrast_dots * low_contrast_dots)

TopHat__KERNELS = initialize_detection_kernels()

def find_top_peaks(image, num_peaks=5, min_distance=10):
    """Find top N peaks in image with minimum distance constraint"""
    peaks = []
    work_img = image.astype(np.float32)

    for _ in range(num_peaks):
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(work_img)

        if max_val <= 0:
            break

        x, y = max_loc
        peaks.append((x, y, max_val))

        cv2.circle(work_img, (x, y), min_distance, 0, -1)

    return peaks


# ============================================================================
# STAGE 1: PREPROCESSING WORKER
# ============================================================================

def stage1_worker():
    """Stage 1: Grab latest frame → Temporal → Chroma → Contrast → Queue"""
    global latest_frame, latest_frame_id, dark_mode

    print("[Stage1] Preprocessing worker started")

    frames_acc = None
    cleaner = None
    contrast_proc = None
    last_frame_id = -1

    while True:
        try:
            time.sleep(0.001)  # Yield to other threads

            # Grab latest frame
            with frame_lock:
                if latest_frame is None or latest_frame_id == last_frame_id:
                    continue

                jpeg_copy = latest_frame
                timestamp_copy = latest_timestamp
                frame_id_copy = latest_frame_id
                last_frame_id = frame_id_copy

            # Decode
            np_arr = np.frombuffer(jpeg_copy, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is None:
                continue

            # Initialize on first frame
            if frames_acc is None:
                h, w, c = frame.shape
                frames_acc = RollingAccumulator(HISTORY_LEN, (h, w, c))
                cleaner = StarFieldCleaner(chroma_kernel=(25, 17))

                with status_lock:
                    contrast_proc = FastContrastProcessor(boost_factor=8, gamma=1.4, ema_alpha=0.2, dark_mode=dark_mode)

                print(f"[Stage1] Initialized for {w}×{h}×{c}")

            # Update dark mode if changed
            with status_lock:
                if contrast_proc.dark_mode != dark_mode:
                    contrast_proc.dark_mode = dark_mode
                    contrast_proc._luts = None

            # PREPROCESSING PIPELINE
            frames_acc.update(frame)
            denoised = frames_acc.get_sum()
            if denoised is None:
                denoised = frame

            processed_clean = cleaner.process(frames_acc.get_mean())
            processed = contrast_proc.process(processed_clean)

            # Send to Stage 2 (drop if full)
            try:
                processed_queue.put_nowait((frame_id_copy, timestamp_copy, denoised, processed, processed_clean))
            except queue.Full:
                pass  # Drop frame if Stage 2 backed up

        except Exception as e:
            print(f"[Stage1] Error: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(0.1)


# ============================================================================
# STAGE 2: ADMD + OVERLAY WORKER
# ============================================================================

def detection_worker():
    """Stage 2: ADMD → Overlay → Broadcast dots + serve /stream_dots"""
    global latest_dots_image, udp_broadcast_socket, dark_mode

    print("[Stage2] ADMD worker started")

    # Create UDP socket
    udp_broadcast_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_broadcast_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    ttl = struct.pack('b', 1)
    udp_broadcast_socket.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, ttl)

    # Stage 2 state
    dots_acc_1 = None
    dots_acc_2 = None
    accum_dots_1 = None
    accum_dots_2 = None
    frame_count = 0

    while True:
        try:
            time.sleep(0.001)  # Yield

            # Get preprocessed frame from Stage 1
            try:
                frame_id, timestamp, denoised, processed, processed_clean = processed_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            # Initialize on first frame
            if dots_acc_1 is None:
                h, w = denoised.shape[:2]
                dots_acc_1 = RollingAccumulator(HISTORY_LEN, (h, w))
                dots_acc_2 = RollingAccumulator(HISTORY_LEN, (h, w))
                print(f"[Stage2] Initialized for {w}×{h}")

            # Get current dark mode
            with status_lock:
                current_dark_mode = dark_mode

            # Convert to uint16 for ADMD
            frame_16 = denoised.astype(np.uint16)
            b_uint16, g_uint16, r_uint16 = cv2.split(frame_16)

            # PARALLEL ADMD
            future_r = executor.submit(ADMD_single_channel_uint16, r_uint16, 0)
            future_g = executor.submit(ADMD_single_channel_uint16, g_uint16, 0)
            future_b = executor.submit(ADMD_single_channel_uint16, b_uint16, 0)

            r_admd = future_r.result()
            g_admd = future_g.result()
            b_admd = future_b.result()

            # Combine — dynamic shift scales with HISTORY_LEN
            min_clip = int(np.sqrt(HISTORY_LEN))
            b_div = b_admd >> min_clip
            g_div = g_admd >> min_clip
            r_div = r_admd >> min_clip

            mul = cv2.multiply(cv2.multiply(b_div, g_div), r_div)
            sum_rgb = cv2.max(r_admd, cv2.max(g_admd, b_admd)) >> round(min_clip/2)
            sum_combined = cv2.min(cv2.max(sum_rgb, mul), 255).astype(np.uint8)

            # Accumulate (odd/even buffers)
            if frame_count % 2 == 0:
                dots_acc_1.update(sum_combined)
                accum_dots_1 = dots_acc_1.get_mean()
            else:
                dots_acc_2.update(sum_combined)
                accum_dots_2 = dots_acc_2.get_mean()
            if accum_dots_1 is None:
                accum_dots_1 = sum_combined
            if accum_dots_2 is None:
                accum_dots_2 = sum_combined
            frame_count += 1

            # Temporal AND
            accum_sq = cv2.multiply(accum_dots_1.astype(np.uint16), accum_dots_2.astype(np.uint16))
            accum_smoothed = cv2.GaussianBlur(accum_sq, (13, 13), 0)

            # TopHat
            accum_combined = TopHat_single_channel(accum_smoothed >> 7, 0)  # sorry for the noise floor clip + signal boost hack but im lazy

            # Find peaks
            peaks = find_top_peaks(accum_combined, num_peaks=dot_max_dots, min_distance=dot_min_distance)
            circles = []
            for x, y, intensity in peaks:
                circles.append([int(x), int(y), 0, float(intensity)])

            # OVERLAY
            if current_dark_mode:
                # Kick off first blur in background
                future_accum_blur = executor.submit(cv2.GaussianBlur, accum_combined, (9, 9), 0)

                # Prepare RGB while waiting
                rgb_enhanced = cv2.merge([
                    b_admd.astype(np.uint8),
                    g_admd.astype(np.uint8),
                    r_admd.astype(np.uint8)
                ])
                rgb_enhanced = cv2.multiply(rgb_enhanced, 8)

                # Kick off second blur in background
                future_rgb_blur = executor.submit(cv2.GaussianBlur, rgb_enhanced, (9, 9), 0)

                # Collect both results
                accum_blurred = future_accum_blur.result()
                rgb_enhanced = future_rgb_blur.result() // 4

                mask_dark = (accum_combined > 2).astype(np.uint8) * 4 // 3  # clips noise, boosts faint stars (magnitude <7 ish)
                mask_bright = (accum_blurred > 32).astype(np.uint8)          # blurred glow of quite bright stars - magnitude < 5.2 ish
                mask = cv2.max(mask_dark, mask_bright)
                stars_only = np.multiply(rgb_enhanced, np.dstack((mask, mask, mask)))
                dots_viz = cv2.add(stars_only, processed)
            else:
                dots_viz = processed_clean

            # Store for /stream_dots
            with dots_image_lock:
                latest_dots_image = dots_viz

            # Broadcast
            if dot_detection_enabled and circles:
                message = {
                    'timestamp': timestamp,
                    'frame_count': frame_count,
                    'circles': [circles],
                    'count': len(circles),
                    'hostname': HOSTNAME,
                    'service': 'find-dots',
                    'detect_dark_spots': detect_dark_spots
                }
                json_data = json.dumps(message, separators=(',', ':')).encode('utf-8')
                udp_broadcast_socket.sendto(json_data, (DOT_MULTICAST_IP, DOT_MULTICAST_PORT))

        except queue.Empty:
            continue
        except Exception as e:
            print(f"[Stage2] Error: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(0.1)


# ============================================================================
# CAMERA MANAGEMENT
# ============================================================================

def validate_jpeg(jpeg_data):
    """JPEG validator - prevents frozen streams from malformed frames"""
    if not jpeg_data or len(jpeg_data) < 4:
        return None

    if jpeg_data[0:2] != b'\xff\xd8':
        return None

    if jpeg_data[-2:] != b'\xff\xd9':
        end_pos = jpeg_data.rfind(b'\xff\xd9')
        if end_pos > 0:
            return jpeg_data[:end_pos+2]
        else:
            return jpeg_data + b'\xff\xd9'

    return jpeg_data

def setup_camera_route():
    """Add host route to force camera traffic through wlan0"""
    try:
        subprocess.run(['ip', 'route', 'del', CAMERA_IP],
                      stderr=subprocess.DEVNULL, check=False)

        result = subprocess.run(['ip', 'route', 'add', CAMERA_IP, 'dev', CAMERA_INTERFACE],
                               capture_output=True, text=True)

        if result.returncode == 0:
            print(f"[Route] Added route: {CAMERA_IP} via {CAMERA_INTERFACE}")
            return True
        else:
            print(f"[Route] Warning: Could not add route: {result.stderr}")
            return False
    except Exception as e:
        print(f"[Route] Error setting up route: {e}")
        return False

def check_camera_reachable():
    """Check if camera is reachable"""
    try:
        resp = requests.get(f"http://{CAMERA_IP}/xaccma.cgi?cmd=CheckConnection", timeout=2)
        return resp.status_code == 200
    except:
        return False

def send_camera_command(cmd, arg=None):
    """Send command to camera"""
    headers = {
        'User-Agent': 'SiOnyx Aurora/1 CFNetwork/3860.100.1 Darwin/25.0.0',
        'Accept': '*/*',
    }

    url = f"http://{CAMERA_IP}/xaccma.cgi?cmd={cmd}"
    if arg:
        url += f"&arg={arg}"

    try:
        resp = requests.get(url, headers=headers, timeout=2)
        return resp.status_code == 200
    except:
        return False

def camera_session_thread():
    """Manage camera session and liveview"""
    global liveview_active, camera_connected, latest_frame, latest_timestamp, latest_frame_id

    print("[Camera] Session manager started")

    while True:
        try:
            reachable = check_camera_reachable()

            with status_lock:
                camera_connected = reachable

            if reachable and not liveview_active:
                print("[Camera] Camera reachable, starting session...")

                if send_camera_command("OpenSession", "34001"):
                    print("[Camera] ✓ Session opened")

                    send_camera_command("SetCameraMode", "StillImageMode")
                    time.sleep(0.1)

                    if send_camera_command("StartLiveview"):
                        print("[Camera] ✓ Liveview started")
                        liveview_active = True
                    else:
                        print("[Camera] ✗ Failed to start liveview")
                else:
                    print("[Camera] ✗ Failed to open session")

            elif not reachable and liveview_active:
                print("[Camera] Camera unreachable, stopping liveview")
                liveview_active = False
                with frame_lock:
                    latest_frame = None
                    latest_timestamp = 0.0
                    latest_frame_id = 0

            elif not reachable:
                with frame_lock:
                    if latest_frame is not None:
                        latest_frame = None
                        latest_timestamp = 0.0
                        latest_frame_id = 0

            time.sleep(5)
        except Exception as e:
            print(f"[Camera] Error: {e}")
            liveview_active = False
            with frame_lock:
                latest_frame = None
                latest_timestamp = 0.0
                latest_frame_id = 0
            time.sleep(5)

def keepalive_thread():
    """Send periodic keepalive to camera"""
    print("[Keepalive] Started")

    while True:
        try:
            if liveview_active:
                send_camera_command("CheckConnection")
            time.sleep(10)
        except Exception as e:
            print(f"[Keepalive] Error: {e}")
            time.sleep(10)

def udp_receiver_thread():
    """Receive UDP video stream from camera"""
    global latest_frame, latest_timestamp, latest_frame_id
    global frames_since_change, zoom_level, eis_enabled, night_mode
    global pending_zoom, pending_eis, pending_night_mode

    print("[UDP] Receiver started")

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1048576)
    sock.bind(('0.0.0.0', UDP_PORT))
    sock.settimeout(1.0)

    print(f"[UDP] Listening on 0.0.0.0:{UDP_PORT}")

    frame_count = 0
    packet_count = 0
    last_log_time = time.time()

    while True:
        try:
            receive_time = time.time()
            data, addr = sock.recvfrom(65535)
            packet_count += 1

            if packet_count <= 10 or (time.time() - last_log_time) > 10.0:
                print(f"[UDP] Packet #{packet_count}: {len(data)} bytes")
                last_log_time = time.time()

            # Find JPEG in packet
            jpeg_start = data.find(b'\xff\xd8')
            if jpeg_start != -1:
                jpeg_data = data[jpeg_start:]

                # Update latest frame
                with frame_lock:
                    latest_frame = jpeg_data
                    latest_timestamp = receive_time
                    latest_frame_id += 1
                    frame_count = latest_frame_id

                # Confirmation counter for pending changes
                if pending_zoom is not None or pending_eis is not None or pending_night_mode is not None:
                    frames_since_change += 1

                    if frames_since_change >= FRAMES_TO_CONFIRM:
                        with status_lock:
                            if pending_zoom is not None:
                                zoom_level = pending_zoom
                                pending_zoom = None
                                print(f"[Camera] Zoom confirmed: {zoom_level}%")

                            if pending_eis is not None:
                                eis_enabled = pending_eis
                                pending_eis = None
                                print(f"[Camera] EIS confirmed: {eis_enabled}")

                            if pending_night_mode is not None:
                                night_mode = pending_night_mode
                                pending_night_mode = None
                                print(f"[Camera] Night mode confirmed: {night_mode}")

                            frames_since_change = 0

        except socket.timeout:
            continue
        except Exception as e:
            print(f"[UDP] Error: {e}")


# ============================================================================
# CONTROL HANDLERS
# ============================================================================

class ControlHandler:
    """Handle control commands"""

    @staticmethod
    def get_status():
        """Get current status"""
        with status_lock:
            return {
                'connected': camera_connected,
                'liveview': liveview_active,
                'recording': recording,
                'eis': eis_enabled,
                'zoom': zoom_level,
                'night_mode': night_mode,
                'dark_mode': dark_mode,
                'dot_detection_enabled': dot_detection_enabled,
                'dot_min_radius': dot_min_radius,
                'dot_max_dots': dot_max_dots,
                'dot_min_z_score': dot_min_z_score,
                'dot_min_distance': dot_min_distance,
                'detect_dark_spots': detect_dark_spots,
                'dot_edge_crop': dot_edge_crop
            }

    @staticmethod
    def set_zoom(level):
        """Set zoom level"""
        global pending_zoom, frames_since_change
        level = max(100, min(300, level))
        if send_camera_command("SetDigitalZoomMode", str(level)):
            with status_lock:
                pending_zoom = level
                frames_since_change = 0
            return True
        return False

    @staticmethod
    def toggle_eis():
        """Toggle EIS"""
        global pending_eis, frames_since_change, eis_enabled
        new_state = not eis_enabled
        arg = "On" if new_state else "Off"
        if send_camera_command("SetEIS", arg):
            with status_lock:
                pending_eis = new_state
                frames_since_change = 0
            return True
        return False


def process_command(cmd):
    """Process incoming command dict"""
    global dark_mode, dot_detection_enabled, dot_min_radius, dot_max_dots
    global dot_min_z_score, dot_min_distance, detect_dark_spots, dot_edge_crop
    global recording, liveview_active

    action = cmd.get('action', '')

    if action == 'zoom':
        level = int(cmd.get('level', 100))
        success = ControlHandler.set_zoom(level)
        return {'success': success, 'zoom': level}

    elif action == 'eis':
        success = ControlHandler.toggle_eis()
        return {'success': success, 'eis': eis_enabled}

    elif action == 'photo':
        success = send_camera_command("TakePicture")
        return {'success': success}

    elif action == 'night_mode':
        mode = cmd.get('mode', 'NightColor')
        success = send_camera_command("SetNightMode", mode)
        if success:
            with status_lock:
                pending_night_mode = mode
                frames_since_change = 0
        return {'success': success, 'night_mode': mode}

    elif action == 'dark_mode':
        value = cmd.get('value', True)
        with status_lock:
            dark_mode = value
        return {'success': True, 'dark_mode': dark_mode}

    elif action == 'reset_camera_connection':
        global liveview_active, camera_connected
        liveview_active = False
        camera_connected = False
        with frame_lock:
            pass  # camera_session_thread will reinitialize
        return {'success': True}

    elif action == 'dot_detection_enabled':
        value = cmd.get('value', True)
        dot_detection_enabled = value
        return {'success': True, 'dot_detection_enabled': dot_detection_enabled}

    elif action == 'dot_min_radius':
        value = int(cmd.get('value', 0))
        value = max(0, min(15, value))
        dot_min_radius = value
        return {'success': True, 'dot_min_radius': dot_min_radius}

    elif action == 'dot_max_dots':
        value = int(cmd.get('value', 5))
        value = max(1, min(20, value))
        dot_max_dots = value
        return {'success': True, 'dot_max_dots': dot_max_dots}

    elif action == 'dot_min_z_score':
        value = float(cmd.get('value', 7.0))
        value = max(1.0, min(20.0, value))
        dot_min_z_score = value
        return {'success': True, 'dot_min_z_score': dot_min_z_score}

    elif action == 'dot_min_distance':
        value = int(cmd.get('value', 12))
        value = max(5, min(100, value))
        dot_min_distance = value
        return {'success': True, 'dot_min_distance': dot_min_distance}

    elif action == 'detect_dark_spots':
        value = cmd.get('value', False)
        detect_dark_spots = value
        return {'success': True, 'detect_dark_spots': detect_dark_spots}

    elif action == 'dot_edge_crop':
        value = int(cmd.get('value', 4))
        value = max(0, min(50, value))
        dot_edge_crop = value
        return {'success': True, 'dot_edge_crop': dot_edge_crop}

    else:
        return {'error': 'Unknown action'}

def control_server_thread():
    """TCP control server"""
    print(f"[Control] Server starting on port {CONTROL_PORT}")

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(('0.0.0.0', CONTROL_PORT))
    sock.listen(5)

    print(f"[Control] Listening on port {CONTROL_PORT}")

    while True:
        try:
            client, addr = sock.accept()
            threading.Thread(target=handle_control_client, args=(client, addr), daemon=True).start()
        except Exception as e:
            print(f"[Control] Error: {e}")

def handle_control_client(client, addr):
    """Handle control client connection"""
    print(f"[Control] Client connected: {addr}")

    try:
        while True:
            data = client.recv(4096)
            if not data:
                break

            try:
                cmd = json.loads(data.decode())
                response = process_command(cmd)
                client.send((json.dumps(response) + '\n').encode())
            except Exception as e:
                error_response = {'success': False, 'error': str(e)}
                client.send((json.dumps(error_response) + '\n').encode())

    except Exception as e:
        print(f"[Control] Client error: {e}")
    finally:
        client.close()
        print(f"[Control] Client disconnected: {addr}")


# ============================================================================
# HTTP STREAM SERVER
# ============================================================================

class MJPEGStreamHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    @staticmethod
    def get_status():
        """Get current status"""
        with status_lock:
            return {
                'connected': camera_connected,
                'liveview': liveview_active,
                'recording': recording,
                'zoom': zoom_level,
                'eis': eis_enabled,
                'night_mode': night_mode,
                'dark_mode': dark_mode,
                'dot_detection_enabled': dot_detection_enabled,
                'dot_min_radius': dot_min_radius,
                'dot_max_dots': dot_max_dots,
                'dot_min_z_score': dot_min_z_score,
                'dot_min_distance': dot_min_distance,
                'detect_dark_spots': detect_dark_spots,
                'dot_edge_crop': dot_edge_crop
            }

    def do_GET(self):
        if self.path == '/stream':
            self.stream_standard()

        elif self.path == '/stream_ts':
            self.stream_with_timestamps()

        elif self.path == '/stream_dots':
            self.stream_dots_visualization()

        elif self.path == '/status':
            self.send_status()

        elif self.path == '/' or self.path == '/index.html':
            with open('index.html', 'rb') as f:
                self.send_response(200)
                self.send_header('Content-Type', 'text/html; charset=utf-8')
                self.end_headers()
                self.wfile.write(f.read())

        elif self.path.endswith('.js'):
            try:
                filename = self.path.lstrip('/')
                with open(filename, 'rb') as f:
                    data = f.read()
                self.send_response(200)
                self.send_header('Content-Type', 'application/javascript')
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            except Exception as e:
                print("[HTTP] JS serve error:", e)
                self.send_error(404)

        elif self.path == '/demo':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            html = b"""<!DOCTYPE html>
<html><head><title>SiOnyx + Dots Demo</title></head>
<body style="margin:0;background:#000;color:#fff;font-family:monospace;">
<h2 style="margin:10px;">Camera + Dot Visualization</h2>
<div style="display:flex;gap:10px;padding:10px;">
<div><h3>Camera</h3><img src="/stream" style="width:640px;border:1px solid #444;"></div>
<div><h3>Dots</h3><img src="/stream_dots" style="width:640px;border:1px solid #444;"></div>
</div></body></html>"""
            self.wfile.write(html)

        else:
            self.send_error(404)

    def do_POST(self):
        if self.path == '/cmd':
            self.handle_command()
        else:
            self.send_error(404)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def stream_with_timestamps(self):
        """Stream with timestamp headers"""
        self.send_response(200)
        self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=--jpgboundary')
        self.send_header('Cache-Control', 'no-cache')
        self.end_headers()

        print(f"[Stream] Timestamped client connected: {self.client_address[0]}")

        last_sent_frame_id = 0

        while True:
            try:
                with frame_lock:
                    if latest_frame is None or latest_frame_id == last_sent_frame_id:
                        time.sleep(0.001)
                        continue

                    frame_data = latest_frame
                    frame_ts = latest_timestamp
                    frame_id = latest_frame_id
                    last_sent_frame_id = frame_id

                header = struct.pack('dI', frame_ts, frame_id)
                frame_with_header = header + frame_data

                self.wfile.write(b'--jpgboundary\r\n')
                self.send_header('Content-Type', 'application/octet-stream')
                self.send_header('Content-Length', str(len(frame_with_header)))
                self.end_headers()
                self.wfile.write(frame_with_header)
                self.wfile.write(b'\r\n')
                self.wfile.flush()

                time.sleep(0.033)

            except (BrokenPipeError, ConnectionResetError):
                break
            except Exception as e:
                print(f"[Stream] Error: {e}")
                break

        print(f"[Stream] Timestamped client disconnected: {self.client_address[0]}")

    def stream_standard(self):
        """Standard MJPEG stream - with MAX frame rate to prevent browser overflow"""
        self.send_response(200)
        self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=--frame')
        self.send_header('Cache-Control', 'no-cache')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()

        print(f"[Stream] Browser client connected: {self.client_address[0]}")

        last_sent_frame_id = 0
        invalid_frame_count = 0
        min_frame_interval = 0.030  # 30ms = 33fps max
        last_send_time = None
        frames_dropped = 0

        while True:
            try:
                current_time = time.time()

                with frame_lock:
                    if latest_frame is None or latest_frame_id == last_sent_frame_id:
                        time.sleep(0.001)
                        continue

                    jpg = latest_frame
                    frame_id = latest_frame_id
                    last_sent_frame_id = frame_id

                # Rate limit
                if last_send_time is not None:
                    time_since_last = current_time - last_send_time
                    if time_since_last < min_frame_interval:
                        frames_dropped += 1
                        continue

                # Validate JPEG
                jpg = validate_jpeg(jpg)
                if jpg is None:
                    invalid_frame_count += 1
                    continue

                # Send
                self.wfile.write(b'--frame\r\n')
                self.wfile.write(b'Content-Type: image/jpeg\r\n')
                self.wfile.write(f'Content-Length: {len(jpg)}\r\n\r\n'.encode())
                self.wfile.write(jpg)
                self.wfile.write(b'\r\n')
                self.wfile.flush()

                last_send_time = current_time

            except (BrokenPipeError, ConnectionResetError):
                break
            except Exception as e:
                print(f"[Stream] Error: {e}")
                break

        print(f"[Stream] Browser client disconnected: {self.client_address[0]}")

    def stream_dots_visualization(self):
        """Stream dots overlay visualization as color MJPEG"""
        self.send_response(200)
        self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=--frame')
        self.send_header('Cache-Control', 'no-cache')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()

        print(f"[Stream] Dots visualization client connected: {self.client_address[0]}")

        while True:
            try:
                with dots_image_lock:
                    if latest_dots_image is None:
                        time.sleep(0.001)
                        continue

                    dots_img = latest_dots_image.copy()

                _, jpg = cv2.imencode('.jpg', dots_img, [cv2.IMWRITE_JPEG_QUALITY, 85])
                jpg_bytes = jpg.tobytes()

                self.wfile.write(b'--frame\r\n')
                self.wfile.write(b'Content-Type: image/jpeg\r\n')
                self.wfile.write(f'Content-Length: {len(jpg_bytes)}\r\n\r\n'.encode())
                self.wfile.write(jpg_bytes)
                self.wfile.write(b'\r\n')
                self.wfile.flush()

                time.sleep(0.033)

            except (BrokenPipeError, ConnectionResetError):
                break
            except Exception as e:
                print(f"[Stream] Dots vis error: {e}")
                break

        print(f"[Stream] Dots visualization client disconnected: {self.client_address[0]}")

    def send_status(self):
        """Send status as JSON"""
        status = MJPEGStreamHandler.get_status()

        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(json.dumps(status).encode())

    def handle_command(self):
        """Handle command POST"""
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            cmd = json.loads(body.decode())
            response = process_command(cmd)

            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())
        except Exception as e:
            self.send_response(400)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({'error': str(e)}).encode())

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """HTTP server with TCP optimizations"""

    def server_bind(self):
        HTTPServer.server_bind(self)
        self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

def stream_server_thread():
    """HTTP MJPEG streaming server"""
    print(f"[Stream] Server starting on port {STREAM_PORT}")

    server = ThreadedHTTPServer(('0.0.0.0', STREAM_PORT), MJPEGStreamHandler)

    print(f"[Stream] Listening on port {STREAM_PORT}")
    server.serve_forever()


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("="*60)
    print("SiOnyx Aurora Server - 2-STAGE PIPELINE")
    print("="*60)
    print(f"Hostname: {HOSTNAME}")
    print(f"Camera number: {CAM_NUMBER}")
    print(f"Dot detection multicast: {DOT_MULTICAST_IP}:{DOT_MULTICAST_PORT}")
    print()
    print("2-STAGE PIPELINE:")
    print("  Stage 1 (Background): Temporal → Chroma → Contrast")
    print("  Stage 2 (Main): ADMD → Overlay → Broadcast")
    print()
    print("FEATURES:")
    print("  • Low latency (always works with freshest frame)")
    print("  • Dark/Light mode toggle via command")
    print("  • Parallel ADMD processing (R/G/B)")
    print("  • Natural film-like output")
    print()

    # Setup routing
    print("Setting up network routing...")
    setup_camera_route()
    print()

    # Diagnostics
    print("Running startup diagnostics...")
    if check_camera_reachable():
        print(f"[Diagnostic] ✓ Camera is reachable")
    else:
        print(f"[Diagnostic] ✗ Camera NOT reachable")
        print(f"[Diagnostic] Continuing anyway - will retry automatically...")
    print()

    # Start all threads
    threads = [
        threading.Thread(target=camera_session_thread, daemon=True, name="Camera"),
        threading.Thread(target=keepalive_thread, daemon=True, name="Keepalive"),
        threading.Thread(target=udp_receiver_thread, daemon=True, name="UDP"),
        threading.Thread(target=stage1_worker, daemon=True, name="Stage1"),
        threading.Thread(target=detection_worker, daemon=True, name="Stage2"),
        threading.Thread(target=control_server_thread, daemon=True, name="Control"),
        threading.Thread(target=stream_server_thread, daemon=True, name="Stream"),
    ]

    for t in threads:
        t.start()
        print(f"Started thread: {t.name}")

    print("\n✓ Server running with 2-stage pipeline")
    print(f"  Camera view: http://localhost:{STREAM_PORT}/")
    print(f"  Standard stream: http://localhost:{STREAM_PORT}/stream")
    print(f"  Enhanced stream: http://localhost:{STREAM_PORT}/stream_dots")
    print(f"  Status: http://localhost:{STREAM_PORT}/status")
    print(f"  Detection results: UDP multicast {DOT_MULTICAST_IP}:{DOT_MULTICAST_PORT}")
    print("\nPress Ctrl+C to stop.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")

        # Cleanup
        executor.shutdown(wait=True)
        try:
            subprocess.run(['ip', 'route', 'del', CAMERA_IP],
                          stderr=subprocess.DEVNULL, check=False)
            print("[Route] Cleaned up camera route")
        except:
            pass