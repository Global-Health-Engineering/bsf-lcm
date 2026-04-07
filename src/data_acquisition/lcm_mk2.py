import cv2
import numpy as np
import time
from picamera2 import Picamera2

# === Camera Setup ===
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (3280, 1300), "format": "RGB888"})
picam2.configure(config)
picam2.start()
time.sleep(1)  # Allow camera to warm up

# === Parameters ===
counter = 0
countingline = 0.2
lowerlimit = 0.7

multiplier = 1
scalar = 0

min_brightness = 10
min_Area = 80
delta = 60

memory_current = []
memory_past = []

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

# === FPS CAP ===
TARGET_FPS = 16
FRAME_PERIOD = 1.0 / TARGET_FPS
# =================

# === LONG PRESS SETTINGS (reset counter) ===
LONG_PRESS_SECONDS = 0.8  # adjust (e.g. 0.5 .. 1.5)
press_start_time = None
press_active = False
# ==========================

# === EXIT BUTTON SETTINGS (black screen) ===
EXIT_HOLD_SECONDS = 1.2  # long-press EXIT for this many seconds to quit
exit_press_start = None
exit_press_active = False
exit_triggered = False
exit_btn = None  # (x1, y1, x2, y2)
# =========================================

# === DISPLAY MODE ===
maintenance = True  # True: show Live + Binary | False: show fullscreen black count screen
# ====================

# FPS shown on the black screen (updates once per second)
display_fps = 0

# === NO-COUNT WARNING SETTINGS ===
NO_COUNT_TIMEOUT_S = 120        # 2 minutes
FLASH_PERIOD_S = 1.0            # full on/off cycle duration (seconds)
BORDER_THICKNESS = 20
# ================================

# Track last time the counter increased
last_count_time = time.monotonic()

def point_in_rect(x, y, rect):
    x1, y1, x2, y2 = rect
    return (x1 <= x <= x2) and (y1 <= y <= y2)

# === Touch callback ===
def on_touch(event, x, y, flags, param):
    global press_start_time, press_active
    global exit_press_start, exit_press_active, exit_triggered
    global exit_btn

    if event == cv2.EVENT_LBUTTONDOWN:
        # If we're in black screen mode and the press is on EXIT button -> start exit hold
        if (not maintenance) and (exit_btn is not None) and point_in_rect(x, y, exit_btn):
            exit_press_start = time.monotonic()
            exit_press_active = True
        else:
            # Otherwise -> normal long-press reset
            press_start_time = time.monotonic()
            press_active = True

    elif event == cv2.EVENT_LBUTTONUP:
        # End both kinds of presses
        press_active = False
        press_start_time = None
        exit_press_active = False
        exit_press_start = None

# === Get First Frame ===
prev_frame = picam2.capture_array()
height, width = prev_frame.shape[:2]
prev_frame = cv2.convertScaleAbs(prev_frame, alpha=multiplier, beta=scalar)
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)

print(height, width)

prev_time = time.time()
frame_count = 0

countingline = round(countingline * height)
lowerlimit = round(lowerlimit * height)
calibration_area = 0

# Create windows + attach touch callback (touchscreen tap/press is mouse events)
if maintenance:
    cv2.namedWindow("Live")
    cv2.setMouseCallback("Live", on_touch)
    cv2.namedWindow("Binary")
else:
    cv2.namedWindow("Count", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Count", on_touch)
    # Borderless fullscreen (where supported by your window manager)
    cv2.setWindowProperty("Count", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

try:
    while True:
        loop_start = time.perf_counter()  # === FPS CAP ===

        frame = picam2.capture_array()
        frame = cv2.convertScaleAbs(frame, alpha=multiplier, beta=scalar)
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        # Frame differencing
        diff = cv2.subtract(gray, prev_gray)
        _, binary = cv2.threshold(diff, min_brightness, 255, cv2.THRESH_BINARY)
        binary = cv2.erode(binary, kernel, iterations=1)
        binary = cv2.dilate(binary, None, iterations=3)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Only draw overlays on the frame when in maintenance mode
        if maintenance:
            cv2.drawContours(frame, contours, -1, (0, 255, 0), 3)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > calibration_area:
                calibration_area = area
                print(area)

            if area > min_Area:
                x, y, w, h = cv2.boundingRect(cnt)
                if maintenance:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cx = x + w // 2

                found = any(cx - delta <= val <= cx + delta for val in (memory_current + memory_past))
                if not found and (y + h) > countingline and y < lowerlimit:
                    counter += 1
                    last_count_time = time.monotonic()  # <-- mark last increment time
                    memory_current.append(cx)

        # === Long-press reset (checked in loop) ===
        if press_active and press_start_time is not None:
            held = time.monotonic() - press_start_time
            if held >= LONG_PRESS_SECONDS:
                counter = 0
                last_count_time = time.monotonic()
                print("Counter reset by long press")
                # Prevent repeated resets while still holding
                press_active = False
                press_start_time = None
        # =========================================

        # Optional: show reset hold progress
        hold_text = None
        if press_active and press_start_time is not None:
            held = time.monotonic() - press_start_time
            prog = min(held / LONG_PRESS_SECONDS, 1.0)
            hold_text = f"Hold to reset: {int(prog * 100)}%"

        # === Exit hold detection (only relevant in black screen) ===
        exit_hold_text = None
        if (not maintenance) and exit_press_active and exit_press_start is not None:
            held_exit = time.monotonic() - exit_press_start
            pct = int(min(held_exit / EXIT_HOLD_SECONDS, 1.0) * 100)
            exit_hold_text = f"Exiting: {pct}%"
            if held_exit >= EXIT_HOLD_SECONDS:
                exit_triggered = True
        # =========================================================

        if maintenance:
            # Draw horizontal counting line + UI
            cv2.line(frame, (0, countingline), (width, countingline), (255, 0, 0), 2)
            cv2.line(frame, (0, lowerlimit), (width, lowerlimit), (255, 0, 0), 2)
            cv2.line(frame, (0, 100), (2 * delta, 100), (255, 255, 0), 2)

            if hold_text is not None:
                cv2.putText(frame, hold_text, (30, 95),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            cv2.putText(frame, f"Count: {counter}", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

            # Show previews
            preview = cv2.resize(frame, (960, 540))
            cv2.imshow("Live", preview)
            cv2.imshow("Binary", binary)

        else:
            # Fullscreen black screen (720x1280) with centered green count + FPS + warning border + EXIT button
            canvas_h, canvas_w = 720, 1280
            black = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

            # Define EXIT button rectangle (top-left)
            btn_w, btn_h = 220, 90
            btn_x1, btn_y1 = 20, 20
            btn_x2, btn_y2 = btn_x1 + btn_w, btn_y1 + btn_h
            exit_btn = (btn_x1, btn_y1, btn_x2, btn_y2)

            # Draw EXIT button (red)
            cv2.rectangle(black, (btn_x1, btn_y1), (btn_x2, btn_y2), (0, 0, 255), -1)
            cv2.putText(black, "EXIT", (btn_x1 + 45, btn_y1 + 62),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 3)

            # === Inactivity flashing border (blue if count=0, red otherwise) ===
            inactive_for = time.monotonic() - last_count_time
            if inactive_for >= NO_COUNT_TIMEOUT_S:
                phase = (time.monotonic() % FLASH_PERIOD_S) < (FLASH_PERIOD_S / 2.0)
                if phase:
                    # Green if counter is zero, otherwise red
                    border_color = (0, 255, 0) if counter == 0 else (0, 0, 255)

                    cv2.rectangle(
                        black,
                        (0, 0),
                        (canvas_w - 1, canvas_h - 1),
                        border_color,
                        BORDER_THICKNESS
                    )
            # ================================================================


            # Centered count
            text = str(counter)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 4.0
            thickness = 6
            color = (0, 255, 0)  # Green (BGR)

            (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
            x = (canvas_w - text_w) // 2
            y = (canvas_h + text_h) // 2
            cv2.putText(black, text, (x, y), font, font_scale, color, thickness)

            # Optional: show reset hold progress below count
            if hold_text is not None:
                cv2.putText(black, hold_text, (x, y + 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

            # Show exit hold progress under the EXIT button
            if exit_hold_text is not None:
                cv2.putText(black, exit_hold_text, (btn_x1, btn_y2 + 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            # FPS at top-right
            fps_text = f"FPS: {display_fps}"
            fps_scale = 1.0
            fps_thickness = 2
            (fps_w, fps_h), _ = cv2.getTextSize(fps_text, font, fps_scale, fps_thickness)
            fps_x = canvas_w - fps_w - 20
            fps_y = fps_h + 20
            cv2.putText(black, fps_text, (fps_x, fps_y),
                        font, fps_scale, (0, 255, 0), fps_thickness)

            cv2.imshow("Count", black)

        prev_gray = gray.copy()
        memory_past = memory_current
        memory_current = []

        frame_count += 1
        current_time = time.time()

        if current_time - prev_time >= 1.0:
            display_fps = frame_count
            print("FPS:", display_fps)
            print("Noise:", calibration_area)
            frame_count = 0
            prev_time = current_time

        # Exit button long-press triggers quit
        if exit_triggered:
            break

        # Press q to quit (if a keyboard is attached)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # === FPS CAP ===
        elapsed = time.perf_counter() - loop_start
        sleep_time = FRAME_PERIOD - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)
        # =================

finally:
    picam2.stop()
    cv2.destroyAllWindows()
