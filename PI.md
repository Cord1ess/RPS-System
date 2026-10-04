# Raspberry Pi Branch Testing

This is additive to SETUP_GUIDE.md §3. Use these steps to test eature/rps-cli safely on a Pi without overwriting your running config.

## 1. Push branch from laptop
`ash
git push -u origin feature/rps-cli
`

## 2. On the Raspberry Pi (safe checkout)
Back up the current config first:
`ash
cd ~/RPS-System  # adjust if your path differs
cp config.json ~/rps-config.pi.json
`

Fetch and check out the branch:
`ash
git fetch origin
git checkout -B feature/rps-cli origin/feature/rps-cli
`

Restore your Pi-specific config:
`ash
cp ~/rps-config.pi.json config.json
`

## 3. Run tests (skip training-heavy tests if needed)
Full tests excluding the training subset that needs heavy deps:
`ash
.venv/bin/python -m pytest tests/ -q -k "not training"
`

Training tests (require scikit-learn/matplotlib):
`ash
.venv/bin/python -m pytest tests/test_training.py -q
`

## 4. Quick CLI smoke checks
`ash
.venv/bin/python -m rps --help
.venv/bin/python -m rps config --section robot
.venv/bin/python rps.py setup --mock-camera --seconds 1
.venv/bin/python -m rps play --mock-camera --mock-esp --headless 50
`

## 5. Return to main
`ash
git checkout main
`

## 6. Offline transfer (if Pi has no network)
On laptop:
`ash
git bundle create rps-cli.bundle feature/rps-cli
`
Copy bundle to Pi via USB, then:
`ash
cd ~/RPS-System
git fetch /path/to/rps-cli.bundle feature/rps-cli:feature-rps-cli
git checkout feature-rps-cli
`

Notes: config.json is tracked (branch switches can overwrite it) so always back up on Pi. data/, models/*.pth, models/dextra/ are ignored and survive switches.
## 7. Playing the game (RPS play)

The CLI shares the same runner as the OpenCV window. On a headless Pi over SSH, use --headless or --no-hud. On a Pi with a display, drop --headless to see the live HUD/zone status.

### Check the play zone first
`ash
# Quick check with simulated camera (shows if ROI fits)
.venv/bin/python -m rps setup --mock-camera --seconds 2

# Set the play zone (interactive drag if display available, else give numbers)
.venv/bin/python -m rps setup zone
# Or directly: .venv/bin/python -m rps setup zone X Y SIZE
`

### Mock (no camera/robot) - safe smoke tests
`ash
# Text HUD (shows zone status, last gesture, recognizer, mode)
.venv/bin/python -m rps play --recognizer mediapipe --mock-camera --mock-esp --mode continuous

# One-line decisions only, no live HUD
.venv/bin/python -m rps play --recognizer mediapipe --mock-camera --mock-esp --mode continuous --no-hud

# Headless N frames (good for SSH/CI)
.venv/bin/python -m rps play --recognizer mediapipe --mock-camera --mock-esp --mode continuous --headless 100
`

### Real camera (no robot) - verify hand is inside the bounding box
`ash
# HUD shows: "hand inside zone", "hand near/partial", or "no hand in zone"
# Also shows raw Dextra/Mediapipe readings, fps, and last throw timing
.venv/bin/python -m rps play --recognizer mediapipe

# If you see "no hand in zone", adjust ROI:
.venv/bin/python -m rps setup zone
`

How to know if the hand is within the bounding box:
- In the terminal HUD (default when display+tty): the first line reports zone status explicitly (inside/near-partial/none). 
- The ROI rectangle is the green square; when the hand is fully inside it, status reads "hand inside zone". Partial/edges show "hand near/partial"; no hand visible shows "no hand in zone".
- Windowed mode (no --headless, display): the OpenCV window draws the same ROI and hand landmarks.

### Real camera + robot
`ash
# Uses config.json (robot.mode, host, protocol). HUD gives zone + last command timing
.venv/bin/python -m rps play

# Guided beat mode (matches the app's beat guide timing)
.venv/bin/python -m rps play --mode guided
`

### Tips for Raspberry Pi
- Prefer --recognizer mediapipe for lighter load; Dextra tuned needs the trained model on disk.
- If over SSH with no display, use --headless N or --no-hud (the HUD still updates but may be wide; --no-hud is cleaner).
- Frame rate drops under load will show in HUD fps; aim for ~25-30 fps for reliable timing.
- Always restore your backed-up config.json after checkout (step 2 above).
