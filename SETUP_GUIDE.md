# Setup Guide

How to get the rock-paper-scissors robot running from a fresh clone, on a **Raspberry Pi 5** (the stand-alone demo) or a Windows laptop, and what every page, button and setting does.

The system: a camera watches your hand, the app reads your throw (rock, paper or scissors) and sends the robot hand its move over Wi-Fi as a UDP message, fast enough to look simultaneous. The robot hand is an ESP32 driving servos; its firmware is maintained separately by the hardware team.

> **Status:** the Windows laptop setup is what the team has used and tested. The Raspberry Pi steps follow Raspberry Pi OS "Trixie" defaults, and the Linux-specific code (camera, ping, file manager) is covered by tests, but the full setup has not yet been run on a Pi. Report anything that differs.

**Contents**

1. [What is in the repo](#1-what-is-in-the-repo)
2. [What you need](#2-what-you-need)
3. [Raspberry Pi 5 setup](#3-raspberry-pi-5-setup)
4. [Windows laptop setup](#4-windows-laptop-setup)
5. [First run: camera, robot, play](#5-first-run-camera-robot-play)
6. [The app, page by page](#6-the-app-page-by-page)
7. [The four recognizers](#7-the-four-recognizers)
8. [Game modes and what the robot plays](#8-game-modes-and-what-the-robot-plays)
9. [Reading the Speed card](#9-reading-the-speed-card)
10. [Command line](#10-command-line)
11. [Before the showcase: checklist](#11-before-the-showcase-checklist)
12. [Troubleshooting](#12-troubleshooting)
13. [Every setting](#13-every-setting)

---

## 1. What is in the repo

| Included | Where |
|---|---|
| All the code (desktop app, command-line runtime, tools, tests) | `app.py`, `play.py`, `rps/`, `tools/`, `tests/` |
| The team's saved settings (camera, play zone, robot address, game) | `config.json` |
| Mediapipe's hand model (Google, Apache 2.0) | `models/hand_landmarker.task` |
| A reference ESP32 firmware (not the team's) | `firmware/` |

| Not included | Why | How to get it |
|---|---|---|
| **Dextra Raw** (`models/dextra_roshambo.pth`) | Dextra's weights come from a repository with no license, so they are not redistributed | One click: Play Debug > **Download Dextra**, or `python tools/import_dextra.py` (downloads from Dextra's own repository; needs internet once) |
| **Dextra Tuned** (`models/dextra_tuned.pth`) | Built from Dextra's weights (same reason) | Copy it from the computer that trained it (USB stick or `scp`), or train your own on the Train page |
| Recordings (`data/`) | 180–210 MB each, too large for GitHub | Record your own (Record page) or copy folders from the team laptop |

**Straight after cloning, Mediapipe works:** you can play with the "Mediapipe" recognizer. "Both" also runs, on Mediapipe alone, until Dextra Tuned is present. The repo's `config.json` selects **Dextra Tuned**; while that file is missing, the Play pages show it as "(not available)" and switch to Mediapipe.

## 2. What you need

**For the stand-alone demo**

- Raspberry Pi 5 (4 GB or 8 GB), the **official 27 W USB-C power supply** and the **Active Cooler**. With a weaker supply or no cooler the Pi slows itself down.
- A microSD card (32 GB or more) or an SSD, a screen with HDMI (the Pi uses a micro-HDMI cable), and a keyboard and mouse for setup.
- **Sound:** the Pi 5 has no headphone jack. The beat guide plays through the HDMI screen's speakers or a USB speaker. A Bluetooth speaker works but adds 150–250 ms (see Sound delay).
- **Camera: a USB webcam** (recommended; 640×480 at 30 fps, ideally uncompressed YUY2). An IP camera also works but adds 100 ms or more of delay.
- **A lamp on the play zone.** In dim light webcams slow down to ~15–17 real images a second, which makes every decision slower.
- The robot hand (ESP32 with the team firmware) on the same Wi-Fi as the Pi.

**For training** (making Dextra Tuned): a laptop or desktop; see section 4.

## 3. Raspberry Pi 5 setup

### 3.1 Operating system

Install **Raspberry Pi OS (64-bit)**, version **"Trixie"** (Debian 13), with Raspberry Pi Imager. The desktop version is needed (the app has a window).

> The older "Bookworm" cannot run the app: the screen library (PySide6) needs glibc 2.39 or newer, and Bookworm has 2.36. Check with `ldd --version`.

Then update and add a few tools:

```bash
sudo apt update && sudo apt full-upgrade -y
sudo apt install -y git python3-venv v4l-utils libxcb-cursor0
```

`v4l-utils` lists what your camera can do; `libxcb-cursor0` is needed by Qt when the desktop runs on X11.

### 3.2 Get the code and install

```bash
git clone https://github.com/Cord1ess/RPS-System.git ~/RPS-System
cd ~/RPS-System
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements-run.txt
```

`requirements-run.txt` holds only what running needs (Mediapipe, PyTorch CPU, NumPy, PySide6, pyqtgraph); it takes about 10 minutes the first time. `requirements.txt` adds the training packages.

### 3.3 Models

```bash
.venv/bin/python tools/import_dextra.py        # Dextra Raw, from Dextra's own repository
```

Then copy **Dextra Tuned** from the laptop that trained it into `~/RPS-System/models/`:

```bash
scp USER@LAPTOP:/path/to/RPS-System/models/dextra_tuned.pth ~/RPS-System/models/
```

(or a USB stick). Without it, use the Mediapipe recognizer.

### 3.4 Camera

```bash
ls /dev/video*                                  # the camera should be listed
v4l2-ctl --list-formats-ext -d /dev/video0      # modes it supports (look for YUYV 640x480 30 fps)
```

The repo's settings name a Windows capture driver (Media Foundation); on the Pi it switches to **V4L2** automatically, including exposure locking. If the camera is not `/dev/video0`, set **Camera number** (Setup > Manual camera settings).

**IP camera instead:** put its address in **IP camera address** (Setup > Manual camera settings), e.g. `rtsp://user:password@192.168.1.50:554/stream1` or `http://192.168.1.50:8080/video` (MJPEG). Set its resolution and frame rate in the camera's own web page; 640×480 at 30 fps is enough. Leave the address empty to go back to the USB camera.

### 3.5 Start the app

```bash
cd ~/RPS-System
.venv/bin/python app.py
```

Then follow [section 5](#5-first-run-camera-robot-play).

### 3.6 Check the speed

Copy one recording folder from the team laptop (e.g. `data/recordings/Jonayed/20260927-231916_throws_scissors`) to the same place on the Pi, then:

```bash
.venv/bin/python tools/benchmark.py --recording data/recordings/Jonayed/20260927-231916_throws_scissors
```

It prints each stage's time per camera frame for Dextra Tuned, Mediapipe and Both, says whether each keeps up with 30 fps, and warns if the Pi is throttling (power or heat). Run the same command on the laptop to compare. Expected on a Pi 5: Dextra Tuned alone keeps up easily (~10–15 ms per frame); Mediapipe takes ~30–40 ms, so with Mediapipe or Both it reads about every other frame and its decisions come ~30–60 ms later than on the laptop.

### 3.7 Start the demo when the Pi boots

Create `~/.config/autostart/rps-demo.desktop` (replace `YOUR_USER` with the Pi's user name):

```ini
[Desktop Entry]
Type=Application
Name=RPS demo
Exec=/home/YOUR_USER/RPS-System/.venv/bin/python /home/YOUR_USER/RPS-System/app.py --fullscreen --page play
```

If your desktop session does not pick it up, add the same command followed by ` &` to `~/.config/labwc/autostart`. `--fullscreen` fills the screen; `--page play` opens the Play page. Close the app with Alt+F4 (it asks to save unsaved settings).

### 3.8 Network and sound

- Put the Pi on the same Wi-Fi as the robot. For a venue with no usable Wi-Fi, the Pi can run its own hotspot with the name and password the robot's firmware joins: `sudo nmcli device wifi hotspot ssid NAME password PASSWORD` (or the network menu's hotspot option). The Pi is then `10.42.0.1` and the robot gets an address from it: find it with `ip neigh` and enter it on Bot tuning.
- Choose the speaker in the desktop's volume menu (HDMI or the USB speaker).

### 3.9 Switching the robot between the Pi and a laptop

The robot's firmware takes commands from any computer on its network, so switching needs no change on the robot:

1. **Same network.** The Pi, the laptop and the robot must be on one Wi-Fi: a router, or the Pi's hotspot (then the laptop joins the hotspot too).
2. **Same robot address on both.** Give the robot a fixed address (a DHCP reservation on the router; on the Pi's hotspot it keeps its address) and enter it on Bot tuning on both computers.
3. **Start where you want to play.** When one computer starts a game (or sends a command from Bot tuning), it tells the others on the network, and any game driving the same robot stops there, with "Robot in use by <computer>". Starting again on the first computer takes the robot back.

This is the **Hand over to other computers** setting (`robot.handoff` in `config.json`, on by default). On Windows, the first time the app starts, Windows asks whether Python may use the network: tick **Private networks** and allow it, or the laptop cannot be told to hand over (it can still take over from the Pi). If the robot's Wi-Fi changes, it opens its setup network **ESP32-RPS-Setup** (WiFiManager) to pick the new one; its address may change, so check it on the router or the hotspot (`ip neigh` on the Pi).

## 4. Windows laptop setup

```powershell
git clone https://github.com/Cord1ess/RPS-System.git
cd RPS-System
uv venv --python 3.12 .venv            # or: py -3.12 -m venv .venv
.venv\Scripts\activate
uv pip install -r requirements.txt     # or: pip install -r requirements.txt (includes training)
python tools/import_dextra.py          # Dextra Raw
python app.py
```

- Keep the laptop **plugged in**: on battery, Windows slowed Mediapipe from 13 ms to 47 ms per frame in our test.
- Close OBS and anything else using the camera; turn off camera "AI effects" (Huawei PC Manager, Windows Studio Effects): they change the whole image and look like movement.
- `pytest tests/` runs the test suite (it never talks to the real robot).

## 5. First run: camera, robot, play

1. **Setup page:** choose **Webcam**, press **Start camera**. The **Frame rate** chip should say 27 fps or more; if not, add light and press **Auto-configure**. Press **Set play zone** and drag a square around where your hand plays, **leaving out your face and body**. Press **Save setup**.
2. **Setup > Camera delay tests > Mirror test** (optional but needed for honest delay numbers): hold a mirror so the camera sees the screen; the result is saved as the camera delay.
3. **Bot tuning page:** set **Robot** to *Real robot*, **Robot commands** to *Team firmware*, and the robot's **address** and **port** (4210). Press **Test connection**: the hand should open (the team firmware sends no reply). Press **Rock**, **Paper**, **Scissors** to check each move. Press **Measure Wi-Fi delay**. Press **Save**.
4. **Play page:** choose the recognition, what the robot plays, rounds and pumps, tick **Beat guide** if you want the drum, press **Start match**, and play inside the green square.

No robot at hand? Set **Robot** to *Simulated robot*: a stand-in on this computer receives every command. No camera? Setup's **Simulated camera** or **Recorded session** (a recording replayed in real time).

## 6. The app, page by page

Hover over any control, reading or column title for a short explanation. A **Save** button appears at the bottom right whenever settings changed; closing the app with unsaved changes asks whether to save. The app remembers its window size and last page.

### 1 Setup

| Control | What it does |
|---|---|
| Source | **Webcam**, **Simulated camera** (a moving test pattern) or **Recorded session** (replays a recording in real time). Changing it while the camera runs switches straight away |
| Camera list / Find cameras | Which webcam (built-in or USB), by name; shown when there is more than one. Changing it switches the running camera. **Find cameras** lists them again after plugging one in |
| Start camera | Opens or closes the source. Opening takes about 1.3 s on the demo laptop; closing happens in the background, so the window never freezes |
| Frame rate / Light / Overexposed chips | New camera images per second (images the driver sends twice count once; 27+ is good), average brightness of the play zone (60–200 is good), share of pure-white pixels (under 5% is good). A hint below says what to fix |
| Auto-configure | Measures every camera mode in the current light (~10–15 s) and saves the best: capture driver, exposure lock |
| Manual camera settings | Every camera setting (section 13). **Exposure, colour balance and mirroring apply to the running camera at once**; the others (camera number, IP address, driver, size, frame rate, format) restart it by themselves. **Apply and restart camera** restarts it by hand |
| Set play zone | Click, then drag a square on the video: the only area analysed |
| Camera delay tests | **Mirror test**: the screen flashes, the camera watches it in a mirror; measures how late camera images are (includes the screen's own delay). **LED test**: the robot flashes an LED (reference firmware only) |
| Save setup | Saves the camera and play zone |

### 2 Bot tuning

| Control | What it does |
|---|---|
| Robot | **Real robot** (over Wi-Fi), **Simulated robot** (a stand-in on this computer), **Off** |
| Robot commands | **Team firmware**: `RPS:ROCK`, `RPS:PAPER`, `RPS:SCISSORS`, sent once when the move changes; no replies, no ready position. **Reference firmware**: numbered messages with replies (for `firmware/`) |
| Robot address / port | The ESP32's IP address and UDP port (4210) |
| Reference firmware options | Resend interval and reply timeout (reference firmware only) |
| Test connection | Team firmware: sends `RPS:PAPER` once, so the hand should open. Reference firmware: waits for its reply |
| Rock / Paper / Scissors | Send one command each (Ready too, on the reference firmware). The page keeps one connection while open, so every command comes from the same port; each click goes out at once and appears in the log with anything the robot sends back |
| Finger tuning | One row per servo channel: **0 Pinky + Ring**, **1 Index**, **2 Middle + Point**. Slider or number: 0 = extended, 180 = folded; **Send** sends `ANGLE:<channel>,<angle>` and shows the robot's confirmation (`CONFIRM_ANGLE:...`). **Extend all (0°)**, **Fold all (180°)**; **Send as I move** sends when you let go of a slider. Use it to find each finger's straight and folded angles (then the hardware team puts them in the firmware) |
| Move times | How long the hand takes between each pair of poses (ms). Measure with a 240 fps slow-motion video; the Speed card uses them |
| Hand over (section 3.9) | Sending a command from here, or starting a game, takes the robot from another computer running the app; if another computer takes it, the page shows "Robot in use by <computer>" |
| Measure Wi-Fi delay | Pings the robot 10 times (the ESP32 answers pings whatever its firmware); keeps half the typical round trip as the one-way delay for the Speed card. If it is slow or uneven, the ESP32 is probably power-saving its Wi-Fi: the firmware needs `WiFi.setSleep(false);` in `setup()` |

### 3 Record

Records training and test sessions (raw lossless video plus timestamps, in `data/recordings/<person>/`). Choose **Person** (type a new name), **Type**: *Hold one gesture* (hold it the whole time while moving and turning your hand), *Countdown throws* (pump three times, throw, hold ~1 s, repeat) or *No hand* (move around the play zone without your hand in it); **Gesture**, **Duration**, planned **Throws**, **Hand** and an optional **Lighting** note. **Record** starts after a 3-second countdown; **Stop and save** or **Discard**. Each person should record all 7 sessions in the checklist (3 hold, 3 throws, 1 no hand).

### 4 Dataset

Lists every recording with totals and free disk space. **Build** turns recordings into training images (Dextra views labelled with Mediapipe's help); it warns about sessions where the hand was rarely visible. **Show 48 samples** shows random training images per person and gesture, to check the labels.

### 5 Train

**Tune Dextra** makes Dextra Tuned: Dextra's model keeps what it learned from ~20 people and adapts its last layers to your camera and people. Modes: **Accuracy check** (trains once per person, leaving that person out, to measure accuracy on someone it never saw), **Train, test on one person**, **Train on everyone** (the final model). **Train** / **Cancel**; live accuracy curves. Needs Dextra Raw (Download Dextra) and built training images.

### 6 Evaluate

Replays recordings through the exact live pipeline and scores the recognizers. **Test on**: one person or all recordings. **Models**: **All** or any of Dextra Raw, Dextra Tuned, Mediapipe, Both. **Run** / **Cancel**. Results, one row per model: *Holding right* (share of time a held gesture was read right), *Changes/min* (flicker while holding), *Throws right*, *Throws decided*, *Throws recorded*, *Decides after throw (ms)*, *No-hand moves/min* (false moves with no hand). The best one is named. **Dextra Raw view check** (folded) finds the camera orientation and settings that suit Dextra Raw, with **Use this setting**.

### 7 Play (the demo)

| Part | What it shows or does |
|---|---|
| Scoreboard | **Robot**, **You**, **Draws**, **Round** (of 5 or 10, or the count in an endless match) |
| Big cue | What to do: *Pump!* / the count *3, 2, 1* / *SHOOT*, then the robot's move in its colour, or *Missed* |
| Result line | The last round: your throw, the robot's move, who read it, whose point |
| Camera | The play zone (green), Mediapipe's answer and finger points (cyan), Dextra's answer (violet) |
| **Speed** card | Where the last throw's time went (section 9) |
| Delay graph and chips | Per-frame processing time, Dextra and Mediapipe times, camera interval; camera fps, processing, robot connection |
| Dextra card | Dextra's raw answer and the Dextra view (the movement image it reads) |
| **Match** card | **Recognition** (section 7). **Robot plays**: to win, to draw, to lose (section 8). **Rounds**: 5, 10 or Endless. **Pumps** before the throw (1–4). **Beat guide** on/off, and with it: **Sound** (Drum, Wood block, Beep: all made to be heard on laptop and TV speakers; the wood block and beep cut through a noisy room), **Tempo** (beats per minute; 150 = a beat every 0.4 s), **Count-in** (steady beats before the first round), **Between rounds** (steady beats after SHOOT), **Sound delay** (ms from a beat being played to you hearing it: ~40 on built-in speakers, 150–250 on Bluetooth; the throw window moves by this much), **Play a round** (hear one round of the beat without the camera). **Steady beat** and **Count and SHOOT** volumes can be changed during a match. **Start match** / **Stop** |

Scoring: each round is scored from your throw and the robot's move; a round where no throw was read in time is your point. With the beat guide, pump on the loud hits (the count) and throw on **SHOOT**; the throw is read from 0.35 s before to 0.8 s after SHOOT.

### Play Debug

Everything the recognition and the game rules are doing, for tuning. Status: **Decision** (your gesture after the rules), **Read by** (which reader decided), **Robot** (the move sent), **Round**, **Tempo** (your learned pump speed). Camera, **Speed** card, delay graph. Right column: **Run** (**Recognition**; **Game**: Countdown, Beat guide or Live; **Robot plays**; **Start**; **Test Dextra Raw** runs Dextra Raw alone in Live mode; **Download Dextra**), **Last throws** (each throw's timing: waiting, computing, total), the **Dextra** card (raw answer, confidence per gesture, Dextra's own example images), the **Mediapipe** card, **Orientation and tuning** (turn/mirror the Dextra view, sensitivity, votes, pumps; applied immediately) and the **Log**.

### Settings

**Interface size** (top): how big the whole interface is on this computer, every font, button and panel together. **Auto** (the default) fits the window to the screen it opens on; 60–150% sets it by hand. It is kept per computer (the laptop and the Pi keep their own), and **Restart to apply** reopens the app at the new size. For one run: `python app.py --scale 80`.

Every other setting by section (the robot is on Bot tuning, the match on Play). **Show advanced settings** shows the fine-tuning ones. **Save** writes `config.json`; **Undo changes** reloads the last save; **Reset to defaults** (saved only when you press Save). Section 13 lists every setting.

## 7. The four recognizers

| Name | What reads your hand | Best for |
|---|---|---|
| **Dextra Raw** | Dextra's network (trained on ~20 people with an event camera) as downloaded, reading the Dextra view (movement only) | Showing the original method; weak on webcam scissors |
| **Dextra Tuned** | Dextra's network tuned on your recordings | The fastest option (a few ms per frame; keeps up with 30 fps on a Pi) |
| **Mediapipe** | Google's hand tracker: finger joints, then finger-curl rules | Accuracy, including a still hand; works straight after cloning |
| **Both** (Dextra Tuned + Mediapipe) | Dextra Tuned while the hand moves, Mediapipe when it is still or clearly disagrees | The best overall; falls back to Mediapipe alone without Dextra Tuned |

## 8. Game modes and what the robot plays

| Game | How a round works |
|---|---|
| **Countdown** | Pump at your own pace (pumps are counted from your hand's height and adapt to your pump size and speed), then throw. Paper and scissors are decided as soon as they show; rock once the throwing fist stops (the pumping fist looks the same) |
| **Beat guide** | A drum beat leads each round: steady beats, then the count (3, 2, 1: pump on these), then **SHOOT**: throw. The throw is read in a window around SHOOT |
| **Live** (Play Debug) | The robot answers continuously, whatever it sees |

| Robot plays | Its move against your rock | The round goes to |
|---|---|---|
| **To win** | paper (beats your throw) | the robot |
| **To draw** | rock (copies your throw; in Live mode it mirrors your hand) | nobody |
| **To lose** | scissors (the move your throw beats) | you |

## 9. Reading the Speed card

For every throw, the Play pages show a bar from the throw appearing to the robot's hand having moved, with a line where the UDP command leaves:

| Part | Colour | Meaning | Source |
|---|---|---|---|
| Camera | grey | Light to image in the app | Setup's camera delay test |
| Waiting for camera images | amber | First camera image showing the throw → the image it was decided on. A throw is confirmed on 2–3 images, so this depends on the camera's frame rate (33 ms per image at 30 fps). Rock waits for the hand to stop | measured every throw |
| Computing + UDP send | green | The deciding image arriving → the command leaving: the app's own work | measured every throw |
| Wi-Fi | grey | Command to the robot | Bot tuning's Measure Wi-Fi delay |
| Robot hand moves | grey | The servos moving | Bot tuning's move times |

The headline reads "Computed and sent in X ms · throw -> command Y ms". On the team laptop, computing and sending take ~5–25 ms; the rest is the camera's frame rate and the hardware. For the numbers to be real, run the camera delay test, measure the Wi-Fi delay and enter measured move times. `python play.py` prints the same numbers for each throw.

## 10. Command line

The app runs these same scripts; you can also use them directly (on the Pi, prefix with `.venv/bin/`).

**`python app.py`**: `--fullscreen` fill the screen · `--page NAME` open on a page (`play`, `play-debug`, `setup`, `bot`, ...) · `--scale 80` interface size for this run · `--mock` start with the simulated camera · `--config FILE` · `--data-root DIR` (where recordings live).

**Keys in the app:** **F11** full screen on/off, **Esc** leaves full screen, **F5** starts or stops the match on Play and Play Debug.

**`python play.py`** (the game without the app window: camera view with a readout; `q` quit, `m` switch game, `r` reset counts):

| Option | Meaning |
|---|---|
| `--recognizer dextra_raw / dextra_tuned / mediapipe / both` | Which recognizer (default: the one in `config.json`) |
| `--mode countdown / guided / continuous` | The game (guided = beat guide timing; the beat itself plays only in the app) |
| `--robot-plays win / draw / lose` | What the robot plays |
| `--video DIR` | Replay a recording instead of the camera |
| `--mock-camera` / `--mock-esp` / `--no-robot` | Simulated camera / simulated robot / no robot |
| `--headless N` | Process N frames without a window |
| `--log FILE` | Per-frame timing CSV |
| `--set section.key=value` | Override any setting for this run (repeatable) |

| Tool | What it is for |
|---|---|
| `tools/benchmark.py` | Speed of each stage per camera frame on this computer (compare laptop and Pi) |
| `tools/camera_probe.py --write-config` | What Auto-configure runs: measures camera modes and saves the best |
| `tools/latency_test.py --mode screen` | The mirror test: camera delay |
| `tools/import_dextra.py` | Downloads Dextra Raw |
| `tools/mock_esp.py` | A stand-alone simulated robot that prints every command |
| `tools/dextra_transfer.py` | Finds the camera view that suits Dextra Raw |
| `record_session.py`, `build_dataset.py`, `train.py`, `replay_eval.py` | Record, build training images, tune Dextra, evaluate (what the Record, Dataset, Train and Evaluate pages run) |

## 11. Before the showcase: checklist

- [ ] Pi on its official power supply with the cooler; `tools/benchmark.py` shows no throttling warning.
- [ ] Lamp on the play zone; Setup's Frame rate chip shows 27 fps or more.
- [ ] Play zone set around the hand only (not the face).
- [ ] Dextra Tuned copied to `models/` (or play with Mediapipe).
- [ ] Bot tuning: real robot, team firmware, right address; Test connection and each move work.
- [ ] Mirror test done (camera delay), Wi-Fi delay measured, move times entered.
- [ ] Sound comes out of the right speaker (Play a round).
- [ ] Settings saved; the app starts on Play in full screen at boot.
- [ ] Robot firmware: `WiFi.setSleep(false)`, servos on their own power supply (a power dip resets the ESP32).
- [ ] Recognition set to **Both** (on all 14 recordings: 99% of throws right, decided ~64 ms after the throw; Dextra Tuned alone: 91.5%, ~128 ms).
- [ ] Laptop as backup: same Wi-Fi and robot address as the Pi; on Windows, Python allowed on private networks (section 3.9).
- [ ] Interface size right on the showcase screen (Settings > Interface size).

## 12. Troubleshooting

| Problem | Fix |
|---|---|
| `pip install` fails on PySide6 (Pi) | The OS is Bookworm: install Raspberry Pi OS Trixie (section 3.1) |
| "Camera could not be opened" | Close other programs using it. Pi: `ls /dev/video*` and set Camera number. Windows: camera privacy settings |
| Frame rate chip shows ~15–17 fps | Not enough light (auto exposure slows the camera). Add a lamp, then Auto-configure |
| "Dextra Tuned (not available)" | Copy `models/dextra_tuned.pth` over (section 3.3) or use Mediapipe / Both |
| "Dextra Raw is not downloaded yet" | Play Debug > Download Dextra, or `python tools/import_dextra.py` (needs internet) |
| Robot does not move | Bot tuning: address and port, Test connection. The laptop/Pi and robot must be on the same network. The team firmware never replies, so "Sent" is all the app can show |
| Robot does one move, then ignores the next | A firmware issue: it must not skip a gesture it believes is already showing, and must not save to flash on every move. Check the ESP32 serial monitor: a reboot message means a power dip (give the servos their own supply) |
| Finger tuning says "No confirmation" | Firewall blocking replies to Python (Windows: allow Python on that network), or the wrong address |
| Wi-Fi delay slow or uneven | ESP32 Wi-Fi power saving: `WiFi.setSleep(false);` in the firmware's `setup()`; keep the robot close to the router |
| No beat sound | Pick the right speaker in the system's volume menu; check the volumes on the Play page (Play a round) |
| Everything slow on a laptop | Plug it in (battery mode slows the CPU) |
| The window is too big or too small for the screen | Settings > Interface size (Auto fits the screen), then Restart to apply; or `python app.py --scale 80` |
| A game stopped with "Robot in use by ..." | Another computer started driving the robot (section 3.9). Press Start here to take it back, or turn off Hand over to other computers |
| The laptop does not stop when the Pi takes over | Windows Firewall blocks Python from receiving: allow Python on private networks, and check both use the same robot address |

## 13. Every setting

All settings live in `config.json` (the repo's copy holds the team's values; defaults are in `rps/config.py`). Change them in the app (where noted below) or for one run with `--set section.key=value`. *(advanced)* settings show on the Settings page after ticking **Show advanced settings**.

### Camera (`camera`)

How the webcam is opened and exposed. Where to change it: Setup page (Manual camera settings) or Settings.

| Setting | Key | In the repo's config.json | What it does |
|---|---|---|---|
| Camera number *(advanced)* | `camera.index` | 0 | 0 = built-in webcam. |
| IP camera address *(advanced)* | `camera.url` | "" (empty) | rtsp://... or http://... (MJPEG) to use a network camera instead of the USB one; empty = USB camera. A USB camera is faster: network cameras add 100 ms or more of delay. |
| Capture driver *(advanced)* | `camera.backend` | Media Foundation (Windows) (`msmf`) | Media Foundation: best in dim light. DirectShow: steady 30 fps with a locked exposure. V4L2: Linux (Raspberry Pi); the Windows drivers become V4L2 there. Auto-configure chooses. Choices: `msmf` = Media Foundation (Windows); `dshow` = DirectShow (Windows); `v4l2` = V4L2 (Linux, Raspberry Pi); `any` = Automatic. |
| Width *(advanced)* | `camera.width` | 640 | Pixels. 640 is the fastest uncompressed mode on this camera. |
| Height *(advanced)* | `camera.height` | 480 | Pixels. |
| Frame rate *(advanced)* | `camera.fps` | 30 | Frames per second requested. This camera's maximum is 30. |
| Pixel format *(advanced)* | `camera.fourcc` | YUY2 (uncompressed) (`YUY2`) | YUY2 is uncompressed, so compression noise is not mistaken for movement. Choices: `YUY2` = YUY2 (uncompressed); `MJPG` = MJPG (compressed). |
| Mirror image *(advanced)* | `camera.mirror` | on | Show the camera like a mirror. |
| Lock exposure | `camera.lock_exposure` | off | Fixed brightness and less blur. Needs a lamp on the play zone: this camera cannot boost a dark image. |
| Exposure | `camera.exposure` | -6 | -5 = 31 ms, -6 = 16 ms, -7 = 8 ms per frame. Shorter = less blur, darker image. |
| Lock colour balance *(advanced)* | `camera.lock_white_balance` | off | Stops colour shifts being read as movement. |
| Colour balance (K) *(advanced)* | `camera.wb_temperature` | 4500 | Used when colour balance is locked. |
| Restore camera on exit *(advanced)* | `camera.restore_auto_on_exit` | on | Hand the camera back to other apps with automatic exposure. |

### Play zone (`roi`)

The square the app looks at. Where to change it: Setup page (Set play zone) or Settings.

| Setting | Key | In the repo's config.json | What it does |
|---|---|---|---|
| Play zone left *(advanced)* | `roi.x` | 247 | Pixels from the left edge. Set by dragging on the Setup page. |
| Play zone top *(advanced)* | `roi.y` | 56 | Pixels from the top edge. Set by dragging on the Setup page. |
| Play zone size *(advanced)* | `roi.size` | 317 | Side length in pixels. Set by dragging on the Setup page. |
| Recording margin *(advanced)* | `roi.record_margin` | 0.25 | Extra border saved around the play zone, as a fraction of its size. |

### Dextra view (`dvs`)

How camera frames become the movement pictures Dextra reads. Where to change it: Settings (and Play Debug > Orientation and tuning).

| Setting | Key | In the repo's config.json | What it does |
|---|---|---|---|
| Movement grid *(advanced)* | `dvs.sensor_size` | 128 | The play zone is resized to this many pixels before movement is measured (Dextra: 128). |
| Dextra view size *(advanced)* | `dvs.frame_size` | 64 | Pixels per side of the image Dextra reads. |
| Dark noise damping *(advanced)* | `dvs.log_offset` | 8 | Higher = less false movement from dark, noisy pixels. |
| Movement sensitivity | `dvs.contrast_threshold` | 0.14 | Brightness change counted as movement. Lower = more sensitive, more noise. Default 0.20. |
| Movement per image | `dvs.event_count` | 1500 | How much movement is collected into one Dextra view. Dextra used 1500. |
| Per-pixel cap *(advanced)* | `dvs.clip_count` | 16 | Limit on movement counted in one pixel (Dextra: 16). |
| Noise filter *(advanced)* | `dvs.noise_filter` | on | Ignore single isolated moving pixels. |
| Lighting change limit *(advanced)* | `dvs.global_reset_fraction` | 0.6 | If more than this share of the picture changes at once, treat it as a lighting change, not movement. |
| Partial image *(advanced)* | `dvs.flush_min_fraction` | 0.33 | When the hand stops, still send a Dextra view if it has at least this share of the usual movement. |
| Still threshold *(advanced)* | `dvs.still_events_per_frame` | 40 | Movement per camera frame below which the scene counts as still. |
| Still frames before sending *(advanced)* | `dvs.flush_still_frames` | 2 | When the hand stops, wait this many still camera frames before sending the partial Dextra view. |
| Max collection time (s) *(advanced)* | `dvs.max_accumulation_s` | 0.5 | A Dextra view still not full after this long is sent early, or dropped if it holds too little movement. |

### Dextra (`cnn`)

The model files and how Dextra Raw's image is turned. Where to change it: Settings (and Play Debug > Orientation and tuning).

| Setting | Key | In the repo's config.json | What it does |
|---|---|---|---|
| Dextra Raw file *(advanced)* | `cnn.raw_model` | models/dextra_roshambo.pth | Dextra's model as downloaded (Download Dextra on Play Debug). |
| Dextra Tuned file *(advanced)* | `cnn.tuned_model` | models/dextra_tuned.pth | Dextra tuned on your recordings (made on the Train page). |
| CPU threads *(advanced)* | `cnn.threads` | 1 | 1 is fastest for this small model. |
| Turn Dextra view | `cnn.rotate` | 0° (`0`) | Dextra Raw only: turn its image to match the camera angle Dextra was trained on (side view, fingers pointing left). Dextra Tuned keeps its own. Choices: `0` = 0°; `90` = 90°; `180` = 180°; `270` = 270°. |
| Mirror Dextra view | `cnn.flip` | on | Dextra Raw only: mirror its image left to right. Dextra Tuned keeps its own. |

### Mediapipe (`hand`)

Finger tracking. Where to change it: Settings.

| Setting | Key | In the repo's config.json | What it does |
|---|---|---|---|
| Mediapipe file *(advanced)* | `hand.model_path` | models/hand_landmarker.task | Hand model file (hand_landmarker.task). |
| Use Mediapipe *(advanced)* | `hand.enabled` | on | Allow Mediapipe to run. |
| Find-hand confidence *(advanced)* | `hand.min_detection_confidence` | 0.5 | 0-1. Higher = fewer false hands. |
| Hand-present confidence *(advanced)* | `hand.min_presence_confidence` | 0.5 | 0-1. |
| Keep-tracking confidence *(advanced)* | `hand.min_tracking_confidence` | 0.5 | 0-1. |
| Search margin *(advanced)* | `hand.crop_margin` | 0.25 | Mediapipe also looks this far outside the play zone. |
| Finger straight below *(advanced)* | `hand.extend_threshold` | 0.3 | Finger bend under this counts as straight. |
| Finger bent above *(advanced)* | `hand.curl_threshold` | 0.45 | Finger bend over this counts as bent. |
| Time budget (ms) *(advanced)* | `hand.frame_budget_ms` | 28 | Skip one frame when Mediapipe takes longer than this. |

### Decision rules (`vote`)

How many Dextra answers must agree before the robot moves. Where to change it: Settings (and Play Debug > Orientation and tuning).

| Setting | Key | In the repo's config.json | What it does |
|---|---|---|---|
| Agreement rule *(advanced)* | `vote.method` | Same answer in a row (`sequence`) | How repeated answers from Dextra are combined. Choices: `sequence` = Same answer in a row; `majority` = Majority of recent answers. |
| Answers in a row | `vote.k` | 2 | Dextra must give the same answer this many times before the robot moves. 2 = fast, 3 = safer. |
| Minimum confidence | `vote.min_confidence` | 0.7 | Dextra answers below this confidence are ignored (0-1). |
| Max gap (s) *(advanced)* | `vote.max_gap_s` | 0.25 | Answers further apart than this do not count as in a row. |
| Majority window *(advanced)* | `vote.majority_window` | 5 | Recent answers considered by the majority rule. |

### Game rules (`decision`)

Countdown, beat guide, pumps, throws and timing. Where to change it: Settings; recognition, game and robot plays on Play / Play Debug.

| Setting | Key | In the repo's config.json | What it does |
|---|---|---|---|
| Game | `decision.mode` | Countdown (pumps counted from your hand) (`countdown`) | Countdown: pump, then throw; pumps are counted from your hand. Beat guide: a drum beat leads each round and you throw on SHOOT. Live: the robot answers continuously. Choices: `countdown` = Countdown (pumps counted from your hand); `guided` = Beat guide (throw on SHOOT); `continuous` = Live (answers continuously). |
| Recognition | `decision.recognizer` | Dextra Tuned (`dextra_tuned`) | Dextra Raw, Dextra Tuned, Mediapipe, or Both (Dextra Tuned + Mediapipe). Choices: `dextra_raw` = Dextra Raw; `dextra_tuned` = Dextra Tuned; `mediapipe` = Mediapipe; `both` = Both (Dextra Tuned + Mediapipe). |
| Robot plays | `decision.robot_plays` | To win (beats your throw) (`win`) | To win: the move that beats your throw. To draw: the same move. To lose: the move your throw beats. Choices: `win` = To win (beats your throw); `draw` = To draw (copies your throw); `lose` = To lose (plays what your throw beats). |
| Moving above *(advanced)* | `decision.active_events_per_frame` | 100 | Movement per camera frame that counts as the hand moving. |
| Still below *(advanced)* | `decision.still_events_per_frame` | 40 | Movement per camera frame that counts as still. |
| Still frames | `decision.still_frames` | 2 | Frames without movement before the hand counts as stopped. |
| Mediapipe steady time (ms) *(advanced)* | `decision.mp_stable_ms` | 45 | Mediapipe must see the same gesture for this long before it counts (45 ms = 3 frames at 30 fps). |
| Mediapipe minimum confidence *(advanced)* | `decision.mp_min_confidence` | 0.6 | 0-1. |
| Mediapipe stillness limit *(advanced)* | `decision.mp_still_events_in_hand` | 25 | In Both, Mediapipe decides only when the hand moves less than this. |
| Mediapipe wait (s) *(advanced)* | `decision.mp_after_cnn_commit_s` | 0.15 | In Both, wait this long after a Dextra decision before Mediapipe may change it. |
| Change cooldown (s) *(advanced)* | `decision.switch_dead_time_s` | 0.15 | Minimum time between Mediapipe-driven changes. |
| Idle after (s) *(advanced)* | `decision.idle_timeout_s` | 1 | No hand for this long = idle. |
| When idle *(advanced)* | `decision.idle_action` | Return to ready (`ready`) | What the robot does when nobody is playing. Choices: `ready` = Return to ready; `hold` = Keep last move. |
| Pumps before the throw | `decision.pumps_before_shoot` | 3 | Pumps before the throw. With the beat guide: the loud beats before SHOOT. |
| Missed pumps allowed *(advanced)* | `decision.pump_miss_tolerance` | 1 | A throw still counts if this many pumps were missed. |
| Smallest pump *(advanced)* | `decision.pump_min_rise` | 0.02 | The smallest rise after a low point that counts as a pump, as a share of the play zone height. Grows with the player's own pumps. |
| Fastest pump (s) *(advanced)* | `decision.pump_min_period_s` | 0.18 | Pumps closer together than this are one pump until the player's tempo is learned. |
| Throw window (s) *(advanced)* | `decision.shoot_window_s` | 1.2 | Minimum time allowed for the throw; grows with a slow tempo. |
| Rock fallback (s) *(advanced)* | `decision.rock_settle_fallback_s` | 0.35 | Decide rock after this long if the landing was not seen. |
| Pause after result (s) *(advanced)* | `decision.hold_min_s` | 0.3 | Movement right after a result is ignored for this long. The robot keeps its move until your next pump. |
| Hold at most (s) *(advanced)* | `decision.hold_max_s` | 4 | The robot returns to ready this long after a result if no new round starts. |
| Correction window (s) *(advanced)* | `decision.correction_s` | 0 | Mediapipe may correct a decision this long after it. 0 = off. |
| Throw accepted early (s) *(advanced)* | `decision.guided_early_s` | 0.35 | Beat guide: a throw this much before SHOOT still counts. |
| Throw accepted late (s) *(advanced)* | `decision.guided_late_s` | 0.8 | Beat guide: a throw this much after SHOOT still counts; later is a missed round. |
| Rock after (s) *(advanced)* | `decision.guided_rock_after_s` | 0.3 | Beat guide: a hand still closed this long after SHOOT is rock. |

### Robot (`robot`)

Connection to the robot (Bot tuning page). Where to change it: Bot tuning page.

| Setting | Key | In the repo's config.json | What it does |
|---|---|---|---|
| Robot | `robot.mode` | Real robot (`real`) | Real robot over Wi-Fi, a simulated robot on this computer, or off. Choices: `real` = Real robot; `simulated` = Simulated robot; `off` = Off. |
| Robot commands | `robot.protocol` | Team firmware (RPS:ROCK, RPS:PAPER, RPS:SCISSORS) (`rps_text`) | Must match the robot's firmware. Team firmware: sends RPS:ROCK, RPS:PAPER or RPS:SCISSORS once when the robot's move changes; it has no ready position and does not reply. Reference firmware: numbered messages with replies. Choices: `rps_text` = Team firmware (RPS:ROCK, RPS:PAPER, RPS:SCISSORS); `ack` = Reference firmware (numbered, with replies). |
| Robot address | `robot.host` | 192.168.0.126 | IP address of the ESP32 (team robot: 192.168.0.126; the reference firmware's own Wi-Fi: 192.168.4.1). |
| Robot port | `robot.port` | 4210 | UDP port. Must match the firmware (4210). |
| Resend every (s) *(advanced)* | `robot.heartbeat_s` | 0.1 | Reference firmware only: the current move is resent this often. |
| Reply timeout (s) *(advanced)* | `robot.ack_timeout_s` | 0.5 | Reference firmware only: replies later than this are left out of the robot reply time. |
| Hand over to other computers | `robot.handoff` | on | When another computer running this app (e.g. the Pi or a laptop) starts driving the same robot, stop here so the two never fight over the hand. |
| Finger angles *(advanced)* | `robot.finger_angles` | [0, 0, 0] | Team firmware: the last angle sent to each servo channel on Bot tuning (0 = extended, 180 = folded). |

### Match (`game`)

Rounds and the beat guide (Play page). Where to change it: Play page (Match card).

| Setting | Key | In the repo's config.json | What it does |
|---|---|---|---|
| Rounds | `game.rounds` | 5 | Rounds in a match; 0 = endless (until Stop). |
| Beat tempo (per minute) | `game.beat_bpm` | 150 | Beats per minute of the beat guide. 150 = one beat every 0.4 s, a natural pump pace. |
| Beat sound | `game.sound` | Drum (`drum`) | Drum, wood block or beep. All are made to be heard on laptop speakers; the wood block and beep cut through a noisy room. Choices: `drum` = Drum; `wood` = Wood block; `beep` = Beep. |
| Steady beat volume | `game.beat_volume` | 0.6 | The beat that plays all the time, 0-1. |
| Count volume | `game.cue_volume` | 1 | The count (3, 2, 1) and SHOOT, 0-1. Keep it above the steady beat. |
| Sound delay (ms) | `game.audio_latency_ms` | 40 | Time from a beat being played to it being heard: about 40 ms on laptop speakers, 150-250 ms on Bluetooth. The throw window moves by this much so it matches what you hear. |
| Count-in (beats) | `game.lead_beats` | 4 | Steady beats before the first round, to find the tempo. |
| Beats between rounds | `game.gap_beats` | 4 | Steady beats while the result shows. |

### Timing model (`latency`)

Used on the Evaluate page to estimate when the robot's move is visible. Where to change it: Settings; Setup's camera delay test, Bot tuning's move times and Wi-Fi delay.

| Setting | Key | In the repo's config.json | What it does |
|---|---|---|---|
| Camera delay (ms) *(advanced)* | `latency.camera_latency_ms` | 50 | Measured with the camera delay test on the Setup page. |
| Wi-Fi delay (ms) *(advanced)* | `latency.network_ms` | 0 | One-way delay to the robot: half the ping time, measured on Bot tuning. 0 = not measured. |
| Robot move times (ms) *(advanced)* | `latency.servo_transition_ms` | (table, see config.json) | Time for the hand to move between poses; used by Evaluate to estimate when the robot's move is visible. |
