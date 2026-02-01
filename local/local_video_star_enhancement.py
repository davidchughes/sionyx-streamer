import cv2
import numpy as np
import sys
import time
import argparse
import threading
import queue
from concurrent.futures import ThreadPoolExecutor

# Benchmarking class
class Timer:
    def __init__(self):
        self.times = {}
        self.counts = {}
    
    def start(self, name):
        self.times[name] = time.perf_counter()
    
    def end(self, name):
        if name in self.times:
            elapsed = (time.perf_counter() - self.times[name]) * 1000
            if name not in self.counts:
                self.counts[name] = []
            self.counts[name].append(elapsed)
            return elapsed
        return 0
    
    def print_stats(self):
        print("\n=== Timing Statistics (ms) ===")
        for name, times in self.counts.items():
            if times:
                avg = np.mean(times)
                print(f"{name:20s}: {avg:6.2f}ms avg")

timer = Timer()

# Configuration
HISTORY_LEN = 8
latest_dots_image = None

# Thread pool for parallel ADMD processing
executor = ThreadPoolExecutor(max_workers=3) # R/G/B

# Global dark mode flag (thread-safe with lock)
dark_mode_lock = threading.Lock()
dark_mode_flag = True


# ============================================================================
# ROLLING ACCUMULATOR
# ============================================================================

class RollingAccumulator:
    def __init__(self, maxlen, shape):
        self.maxlen = maxlen
        self.buffer = [np.zeros(shape, dtype=np.uint8) for _ in range(maxlen)]
        self.running_sum = np.zeros(shape, dtype=np.uint32)
        self.index = 0
        self.filled = 0
    
    def update(self, frame):
        if self.filled == self.maxlen:
            self.running_sum -= self.buffer[self.index].astype(np.uint32)
        
        self.buffer[self.index] = frame
        self.running_sum += frame.astype(np.uint32)
        
        self.index = (self.index + 1) % self.maxlen
        if self.filled < self.maxlen:
            self.filled += 1
    
    def get_mean(self):
        if self.filled == 0:
            return None
        return (self.running_sum // self.filled).astype(np.uint8)
    
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

# def initialize_admd_cache(k_value=0, image_shape=(360, 640)):
#     """
#     Pre-warm caches during worker startup.
#     Call this once at the beginning of detection_worker.
    
#     Args:
#         k_value: Kernel size parameter (0 = 3x3, 1 = 5x5, etc.)
#         image_shape: Expected image dimensions (height, width)
#     """
#     print(f"[ADMD] Initializing caches for k={k_value}, shape={image_shape}")
#     get_or_create_kernels(k_value)
#     get_admd_buffers(image_shape)
#     print("[ADMD] Cache initialization complete")

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
# TOPHAT
# ============================================================================

def initialize_detection_kernels():
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
    _ksize = 3 + 2*_k
    _muImg = cv2.blur(image, (_ksize, _ksize), cv2.BORDER_ISOLATED)
    kernel = TopHat__KERNELS[_k]
    _temp3 = cv2.dilate(_muImg, kernel, borderValue=cv2.BORDER_ISOLATED)
    _temp_thresh3 = cv2.subtract(image, _temp3)
    low_contrast_dots = cv2.min(cv2.divide(_temp_thresh3, 2), 255)
    return (low_contrast_dots * low_contrast_dots)

TopHat__KERNELS = initialize_detection_kernels()


# ============================================================================
# STAGE 1: PREPROCESSING WORKER
# ============================================================================

def preprocessing_worker(cap, output_queue, stop_event):
    """Stage 1: Capture → Temporal → Chroma → Contrast"""
    print("[Stage1] Preprocessing worker started")
    
    frames_acc = None
    cleaner = None
    black_proc = None
    frame_id = 0
    
    while not stop_event.is_set():
        try:
            # Capture
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.001)
                continue
            
            # Initialize on first frame
            if frames_acc is None:
                h, w, c = frame.shape
                frames_acc = RollingAccumulator(HISTORY_LEN, (h, w, c))
                cleaner = StarFieldCleaner(chroma_kernel=(25, 17))
                
                with dark_mode_lock:
                    black_proc = FastContrastProcessor(boost_factor=8, gamma=1.4, ema_alpha=0.2, dark_mode=dark_mode_flag)
                
                print(f"[Stage1] Initialized for {w}×{h}×{c}")
            
            # Update dark mode if changed
            with dark_mode_lock:
                if black_proc.dark_mode != dark_mode_flag:
                    black_proc.dark_mode = dark_mode_flag
                    black_proc._luts = None
            
            # PREPROCESSING PIPELINE
            frames_acc.update(frame)
            denoised = frames_acc.get_mean()
            if denoised is None:
                denoised = frame
            
            processed_clean = cleaner.process(denoised)
            processed = black_proc.process(processed_clean)
            
            # Send to Stage 2
            try:
                output_queue.put((frame_id, denoised, processed, processed_clean), timeout=0.01)
                frame_id += 1
            except queue.Full:
                pass  # Drop frame if Stage 2 is backed up
                
        except Exception as e:
            print(f"[Stage1] Error: {e}")
            import traceback
            traceback.print_exc()
    
    print("[Stage1] Worker stopped")


# ============================================================================
# MAIN PROCESSING (STAGE 2)
# ============================================================================

def process_video(video_path, save_path=None):
    """2-stage pipeline: Stage1 (background) → Stage2 (main)"""
    global latest_dots_image, dark_mode_flag
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video source: {video_path}")
        return
    
    # Get properties
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps > 120:
        fps = 30.0
    enlarge_display = True if original_width < 1000 else False
    
    print(f"Video source: {video_path}")
    print(f"Resolution: {original_width}×{original_height}")
    print(f"FPS: {fps:.2f}")
    print(f"\n2-Stage Pipeline:")
    print(f"  Stage 1 (Background): Capture → Temporal → Chroma → Contrast")
    print(f"  Stage 2 (Main): ADMD → Overlay → Display/Record")
    print(f"  Press 'D' to toggle Dark/Light mode")
    print()
    
    cv2.namedWindow('Output')
    
    # Stage 2 state
    dots_acc_1 = None
    dots_acc_2 = None
    accum_dots_1 = None
    accum_dots_2 = None
    video_writer = None
    recorded_frames = 0
    
    # Pipeline queue and thread
    preprocess_queue = queue.Queue(maxsize=3)
    stop_event = threading.Event()
    preprocess_thread = threading.Thread(
        target=preprocessing_worker, 
        args=(cap, preprocess_queue, stop_event),
        daemon=True
    )
    preprocess_thread.start()
    
    # FPS tracking
    fps_start_time = time.time()
    fps_frame_count = 0
    last_frame_id = -1
    
    print("[Stage2] Main processing started")
    
    while True:
        try:
            timer.start("stage2_total")
            
            # Get preprocessed frame from Stage 1
            timer.start("queue_wait")
            try:
                frame_id, denoised, processed, processed_clean = preprocess_queue.get(timeout=0.1)
            except queue.Empty:
                timer.end("queue_wait")
                continue
            timer.end("queue_wait")
            
            # Check for dropped frames
            if frame_id != last_frame_id + 1 and last_frame_id != -1:
                dropped = frame_id - last_frame_id - 1
                if dropped > 0:
                    print(f"\n[Stage2] Dropped {dropped} frames")
            last_frame_id = frame_id
            
            # Initialize Stage 2 state
            if dots_acc_1 is None:
                h, w, c = denoised.shape
                dots_acc_1 = RollingAccumulator(HISTORY_LEN, (h, w))
                dots_acc_2 = RollingAccumulator(HISTORY_LEN, (h, w))
                print(f"[Stage2] Initialized for {w}×{h}")
                
                if save_path:
                    fourcc = cv2.VideoWriter_fourcc(*'avc1')
                    video_writer = cv2.VideoWriter(save_path, fourcc, fps, (w, h))
                    if video_writer.isOpened():
                        print(f"[Stage2] Recording to {save_path}")
            
            # Get current dark mode
            with dark_mode_lock:
                current_dark_mode = dark_mode_flag
            
            # ADMD PROCESSING
            timer.start("convert_split")
            frame_16 = denoised.astype(np.uint16)
            b_uint16, g_uint16, r_uint16 = cv2.split(frame_16)
            timer.end("convert_split")
            
            timer.start("admd")
            future_r = executor.submit(ADMD_single_channel_uint16, r_uint16, 0)
            future_g = executor.submit(ADMD_single_channel_uint16, g_uint16, 0)
            future_b = executor.submit(ADMD_single_channel_uint16, b_uint16, 0)
            r_admd = future_r.result()
            g_admd = future_g.result()
            b_admd = future_b.result()
            timer.end("admd")
            
            timer.start("combine")
            b_div = b_admd >> 2
            g_div = g_admd >> 2
            r_div = r_admd >> 2
            mul = cv2.multiply(cv2.multiply(b_div, g_div), r_div)
            sum_rgb = cv2.max(r_admd, cv2.max(g_admd, b_admd)) >> 1
            sum_combined = cv2.min(cv2.max(sum_rgb, mul), 255).astype(np.uint8)
            timer.end("combine")
            
            timer.start("dots_accum")
            if frame_id % 2 == 0:
                dots_acc_1.update(sum_combined)
                accum_dots_1 = dots_acc_1.get_mean()
            else:
                dots_acc_2.update(sum_combined)
                accum_dots_2 = dots_acc_2.get_mean()
            if accum_dots_1 is None:
                accum_dots_1 = sum_combined
            if accum_dots_2 is None:
                accum_dots_2 = sum_combined
            timer.end("dots_accum")
            
            timer.start("accum_multiply")
            accum_sq = cv2.multiply(accum_dots_1.astype(np.uint16), accum_dots_2.astype(np.uint16))
            accum_smoothed = cv2.GaussianBlur(accum_sq, (13, 13), 0)
            timer.end("accum_multiply")
            
            timer.start("tophat")
            accum_combined = TopHat_single_channel(accum_smoothed >> 7, 0) << 5 # sorry for the noise floor clip + signal boost hack but im lazy
            timer.end("tophat")
            
            # OVERLAY
            timer.start("overlay")
            if current_dark_mode:
                # Start first blur in background
                future_accum_blur = executor.submit(cv2.GaussianBlur, accum_combined, (9, 9), 0)
                
                # Prepare RGB while waiting
                rgb_enhanced = cv2.merge([
                    b_admd.astype(np.uint8),
                    g_admd.astype(np.uint8),
                    r_admd.astype(np.uint8)
                ])
                rgb_enhanced = cv2.multiply(rgb_enhanced, 8)
                
                # Start second blur in background
                future_rgb_blur = executor.submit(cv2.GaussianBlur, rgb_enhanced, (9, 9), 0)
                
                # Wait for both blurs to complete
                accum_blurred = future_accum_blur.result()
                rgb_enhanced = future_rgb_blur.result()
                
                # Masks (these are fast)
                mask_dark = (accum_combined > 1).astype(np.uint8) * 255
                mask_bright = (accum_blurred > 256).astype(np.uint8) * 255
                mask = cv2.max(mask_dark, mask_bright)
                
                stars_only = cv2.bitwise_and(rgb_enhanced, rgb_enhanced, mask=mask)
                latest_dots_image = cv2.add(processed, stars_only)
            else:
                latest_dots_image = processed_clean
            timer.end("overlay")
            
            # RECORD
            if video_writer is not None:
                timer.start("recording")
                video_writer.write(latest_dots_image)
                recorded_frames += 1
                timer.end("recording")
            
            # DISPLAY
            timer.start("display")
            if enlarge_display:
                cv2.imshow('Output', cv2.resize(latest_dots_image, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC))
            else:
                cv2.imshow('Output', latest_dots_image)
            timer.end("display")
            
            fps_frame_count += 1
            
            # FPS display
            if fps_frame_count >= 30:
                elapsed = time.time() - fps_start_time
                current_fps = fps_frame_count / elapsed
                mode_str = "DARK" if current_dark_mode else "LIGHT"
                q_size = preprocess_queue.qsize()
                status = f"FPS: {current_fps:.1f} | Mode: {mode_str} | Q: {q_size}"
                if video_writer is not None:
                    status += f" | Rec: {recorded_frames}"
                status += " | "
                print(f"\n{status}", end="")
                timer.end("stage2_total")
                
                for name in ["queue_wait", "convert_split", "admd", "combine", "dots_accum", 
                            "accum_multiply", "tophat", "overlay", "recording", "display", "stage2_total"]:
                    if name in timer.times:
                        avg = np.mean(timer.counts[name][-30:]) if len(timer.counts[name]) >= 30 else np.mean(timer.counts[name])
                        print(f"{name}:{avg:.1f}ms ", end="")
                print()
                
                fps_start_time = time.time()
                fps_frame_count = 0
            else:
                timer.end("stage2_total")
            
            # KEYBOARD
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\nQuitting...")
                break
            elif key == ord('d') or key == ord('D'):
                with dark_mode_lock:
                    dark_mode_flag = not dark_mode_flag
                    print(f"\n[Mode] Switched to {'DARK' if dark_mode_flag else 'LIGHT'} mode")
            
        except Exception as e:
            print(f"\n[Stage2] Error: {e}")
            import traceback
            traceback.print_exc()
    
    # Cleanup
    stop_event.set()
    preprocess_thread.join(timeout=2)
    if video_writer is not None:
        video_writer.release()
        print(f"\n[Recording] Saved {recorded_frames} frames to {save_path}")
    cap.release()
    cv2.destroyAllWindows()
    executor.shutdown(wait=True)
    print(f"\nTotal frames processed: {last_frame_id + 1}")
    timer.print_stats()


def main():
    parser = argparse.ArgumentParser(
        description='ADMD with 2-stage pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python video_admd_pipelined.py input.mp4
  python video_admd_pipelined.py input.mp4 -s output.mp4

Controls:
  D - Toggle Dark/Light mode
  Q - Quit
        '''
    )
    
    parser.add_argument('input', nargs='?', default="http://sionyx:8080/stream")
    parser.add_argument('-s', '--save', dest='output')
    args = parser.parse_args()
    
    print("="*60)
    print("ADMD - 2-Stage Pipeline")
    print("="*60)
    print(f"Input: {args.input}")
    if args.output:
        print(f"Output: {args.output}")
    print()
    
    process_video(args.input, args.output)

if __name__ == "__main__":
    main()