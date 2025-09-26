import cv2
import numpy as np
import time
import json
import os
from collections import deque

# Try to import gpiozero for button support
try:
    from gpiozero import Button
    GPIO_AVAILABLE = True
except Exception:
    GPIO_AVAILABLE = False

from picamera2 import Picamera2

# === Camera Setup ===
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (3280, 1080), "format": "RGB888"})
picam2.configure(config)
picam2.start()
time.sleep(1)  # Allow camera to warm up

# === Helpers & Interactive Parameters ===
def clamp(v, lo, hi):
    return max(lo, min(hi, v))

# Panel text styling
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.55
THICKNESS = 1

# Parameter definitions for interactive edits
params = {
    "countingline":  {"value": 620, "step": 10,  "min": 0,    "max": 4000,  "type": int},
    "multiplier":    {"value": 1.5, "step": 0.1, "min": 0.1,  "max": 5.0,   "type": float},
    "scalar":        {"value": -30, "step": 5,   "min": -255, "max": 255,   "type": int},
    "min_brightness":{"value": 30,  "step": 5,   "min": 0,    "max": 255,   "type": int},
    "min_box_size":  {"value": 80,  "step": 10,  "min": 1,    "max": 50000, "type": int},
    "delta":         {"value": 25,  "step": 1,   "min": 0,    "max": 500,   "type": int},
}

PARAMS_PATH = "params.json"

def serialize_params():
    return {k: v["value"] for k, v in params.items()}

def apply_loaded_values(loaded):
    applied = []
    for name, meta in params.items():
        if name in loaded:
            v = loaded[name]
            try:
                if meta["type"] is int:
                    v = int(round(float(v)))
                else:
                    v = float(v)
            except Exception:
                continue
            v = clamp(v, meta["min"], meta["max"])
            meta["value"] = v
            applied.append(name)
    return applied

def save_params(path=PARAMS_PATH):
    try:
        with open(path, "w") as f:
            json.dump(serialize_params(), f, indent=2)
        return True, f"Saved to {path}"
    except Exception as e:
        return False, f"Save error: {e}"

def load_params(path=PARAMS_PATH):
    if not os.path.exists(path):
        return False, f"No file {path}"
    try:
        with open(path, "r") as f:
            data = json.load(f)
        applied = apply_loaded_values(data)
        if applied:
            return True, f"Loaded {len(applied)} params"
        else:
            return False, "Loaded file contained no matching params"
    except Exception as e:
        return False, f"Load error: {e}"

# Try auto-load at startup
_ok, _msg = load_params(PARAMS_PATH)
if _ok:
    print(_msg)

param_names = list(params.keys())
selected_idx = 0
edit_mode = False

# === State ===
counter = 0
memory_current = []
memory_past = []

# Small status toast
status_text = ""
status_until = 0.0
def set_status(msg, duration=1.2):
    global status_text, status_until
    status_text = msg
    status_until = time.time() + duration

# === Get First Frame ===
prev_frame = picam2.capture_array()
height, width = prev_frame.shape[:2]
prev_frame = cv2.convertScaleAbs(prev_frame, alpha=params["multiplier"]["value"], beta=params["scalar"]["value"])
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)

# === UI: Parameter panel ===
def draw_param_panel(img, x=20, y=80, line_h=28):
    pad_x, pad_y = 12, 12
    panel_w = 420
    panel_h = pad_y*2 + line_h*(len(param_names)+2)

    # Background overlay
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x+panel_w, y+panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.35, img, 0.65, 0, img)

    # Titles
    cv2.putText(img, "Params (up/down: select, right: edit/apply)", (x+10, y+24),
                FONT, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, "Edit: up/+  down/-   S:Save  L:Load", (x+10, y+24+line_h),
                FONT, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

    # Items
    base_y = y + 24 + line_h*2
    for i, name in enumerate(param_names):
        val = params[name]["value"]
        text = f"{name}: {val:.2f}" if isinstance(val, float) else f"{name}: {val}"
        ty = base_y + i*line_h

        if i == selected_idx:
            cv2.rectangle(img, (x+6, ty-18), (x+panel_w-6, ty+8), (50, 125, 255), -1)
            color = (0, 0, 0)
            if edit_mode:
                text += "  [EDITING]"
        else:
            color = (255, 255, 255)

        cv2.putText(img, text, (x+12, ty), FONT, FONT_SCALE, color, THICKNESS, cv2.LINE_AA)

def draw_status(img):
    if time.time() <= status_until and status_text:
        (tw, th), _ = cv2.getTextSize(status_text, FONT, 0.6, 2)
        x, y = 20, 60
        cv2.rectangle(img, (x-10, y-25), (x+tw+10, y+10), (0, 0, 0), -1)
        cv2.putText(img, status_text, (x, y), FONT, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

def handle_keypress(key):
    """
    Arrow keys:
      - UP/DOWN: navigate list, or edit value when in edit mode
      - RIGHT: toggle edit mode
    S: save params   L: load params
    OpenCV special key codes often are: left=81, up=82, right=83, down=84
    """
    global selected_idx, edit_mode

    if key == 82:  # UP
        if edit_mode:
            name = param_names[selected_idx]
            meta = params[name]
            newv = meta["value"] + meta["step"]
            if meta["type"] is int:
                newv = int(round(newv))
            meta["value"] = clamp(newv, meta["min"], meta["max"])
        else:
            selected_idx = (selected_idx - 1) % len(param_names)

    elif key == 84:  # DOWN
        if edit_mode:
            name = param_names[selected_idx]
            meta = params[name]
            newv = meta["value"] - meta["step"]
            if meta["type"] is int:
                newv = int(round(newv))
            meta["value"] = clamp(newv, meta["min"], meta["max"])
        else:
            selected_idx = (selected_idx + 1) % len(param_names)

    elif key == 83:  # RIGHT -> toggle edit mode
        edit_mode = not edit_mode

    elif key == ord('s') or key == ord('S'):
        ok, msg = save_params(PARAMS_PATH)
        print(msg)
        set_status("Saved" if ok else msg)

    elif key == ord('l') or key == ord('L'):
        ok, msg = load_params(PARAMS_PATH)
        print(msg)
        set_status("Loaded" if ok else msg)

# === Button support ===
# We'll push "virtual key codes" into a queue so the main loop can process them
event_queue = deque()

def enqueue(code):
    event_queue.append(code)

# Map six buttons to actions:
#  - UP, DOWN, EDIT (RIGHT), SAVE ('s'), LOAD ('l'), RESET ('r')
BUTTON_PINS = {
    "UP":    5,    # GPIO5
    "DOWN":  6,    # GPIO6
    "EDIT":  13,   # GPIO13
    "SAVE":  19,   # GPIO19
    "LOAD":  26,   # GPIO26
    "RESET": 21,   # GPIO21
}

buttons = {}
try:
    if GPIO_AVAILABLE:
        # Pull-up inputs; button to GND. Small debounce to prevent chatter.
        buttons["UP"]    = Button(BUTTON_PINS["UP"], pull_up=True, bounce_time=0.03)
        buttons["DOWN"]  = Button(BUTTON_PINS["DOWN"], pull_up=True, bounce_time=0.03)
        buttons["EDIT"]  = Button(BUTTON_PINS["EDIT"], pull_up=True, bounce_time=0.05)
        buttons["SAVE"]  = Button(BUTTON_PINS["SAVE"], pull_up=True, bounce_time=0.05)
        buttons["LOAD"]  = Button(BUTTON_PINS["LOAD"], pull_up=True, bounce_time=0.05)
        buttons["RESET"] = Button(BUTTON_PINS["RESET"], pull_up=True, bounce_time=0.05)

        # on press, enqueue the equivalent keycode
        buttons["UP"].when_pressed    = lambda: enqueue(82)          # UP
        buttons["DOWN"].when_pressed  = lambda: enqueue(84)          # DOWN
        buttons["EDIT"].when_pressed  = lambda: enqueue(83)          # RIGHT (toggle edit)
        buttons["SAVE"].when_pressed  = lambda: enqueue(ord('s'))    # save
        buttons["LOAD"].when_pressed  = lambda: enqueue(ord('l'))    # load
        buttons["RESET"].when_pressed = lambda: enqueue(ord('r'))    # reset counter
    else:
        print("gpiozero not available; running with keyboard only.")
except Exception as e:
    print(f"GPIO setup error: {e}")
    buttons = {}

# === Main Loop ===
try:
    while True:
        # Pull current parameter values
        countingline   = int(params["countingline"]["value"])
        multiplier     = float(params["multiplier"]["value"])
        scalar         = int(params["scalar"]["value"])
        min_brightness = int(params["min_brightness"]["value"])
        min_box_size   = int(params["min_box_size"]["value"])
        delta          = int(params["delta"]["value"])

        # Capture and preprocess
        frame = picam2.capture_array()
        frame = cv2.convertScaleAbs(frame, alpha=multiplier, beta=scalar)
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        # Frame differencing
        diff = cv2.subtract(gray, prev_gray)
        _, binary = cv2.threshold(diff, min_brightness, 255, cv2.THRESH_BINARY)
        binary = cv2.dilate(binary, None, iterations=2)

        # Find contours & draw on the full-res frame
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(frame, contours, -1, (0, 255, 0), 3)

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w * h > min_box_size:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cx = x + w // 2

                found = any(cx - delta <= val <= cx + delta for val in memory_current + memory_past)
                if not found and (y + h) > countingline:
                    counter += 1
                    print(counter)
                    memory_current.append(cx)

        # --- Make the preview FIRST ---
        preview_w, preview_h = 960, 540
        preview = cv2.resize(frame, (preview_w, preview_h))

        # Draw the counting line on the PREVIEW so it stays crisp
        sy = preview_h / float(height)
        cv2.line(preview, (0, int(countingline * sy)), (preview_w, int(countingline * sy)), (255, 0, 0), 2)

        # Draw count text & UI AFTER resizing
        cv2.putText(preview, f"Count: {counter}", (30, 50), FONT, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
        draw_param_panel(preview, x=20, y=80, line_h=28)
        draw_status(preview)

        # Show window
        cv2.imshow("Live", preview)

        # Prepare for next iteration
        prev_gray = gray.copy()
        memory_past = memory_current
        memory_current = []

        # Process keyboard
        key = cv2.waitKey(1) & 0xFF
        if key != 255:
            # Handle 'q' and 'r' directly; others through handle_keypress
            if key == ord('r'):
                counter = 0
            elif key == ord('q'):
                break
            else:
                handle_keypress(key)

        # Process queued button events (drain queue)
        while event_queue:
            ev = event_queue.popleft()
            if ev == ord('r'):
                counter = 0
            else:
                handle_keypress(ev)

    cv2.destroyAllWindows()

finally:
    # Clean up buttons (optional, gpiozero cleans up on exit)
    for b in buttons.values():
        try:
            b.close()
        except Exception:
            pass
