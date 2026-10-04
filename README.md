# RPS v3: Dextra-style Rock-Paper-Scissors with a Laptop Webcam

A rock-paper-scissors robot hand (3D-printed, servo tendons, ESP32 over UDP) that sees your throw and plays the winning move fast enough to look simultaneous. The perception software copies the method of **Dextra** ([SensorsINI](https://sensors.ini.ch/research/projects/dextra)), but runs on an ordinary laptop webcam instead of a DVS event camera.

- **Four ways to read the hand**, named the same everywhere: **Dextra Raw** (Dextra's model as downloaded), **Dextra Tuned** (Dextra tuned on your recordings), **Mediapipe** (finger positions) and **Both (Dextra Tuned + Mediapipe)**. Dextra reads the *Dextra view*: constant-event-count 64×64 frames from a webcam DVS emulator, made only when something moves, with a Dextra-style sequence vote.
- **Pumps** are counted from the hand's height (the tracked wrist, carried through gaps by the play zone's motion) and adapt to each player's pump size and tempo.
- **Game modes:** *countdown* (pump at your own pace), *beat guide* (a drum beat leads each round: pump on 3, 2, 1, throw on SHOOT) and *live* (Dextra demo style); the robot plays to win, draw or lose.

> **Setting it up?** Read [SETUP_GUIDE.md](SETUP_GUIDE.md): installing on a Raspberry Pi 5 or a laptop, what every page, button and setting does, and a checklist for the showcase.

> **Why v3?** The v2 system (removed; it is in the git history before this version) ran the CNN on every webcam frame. When your hand is still, the frame difference is blank, and blank frames were never in training, so the output jumped randomly. A real DVS produces *no frames* when nothing moves, so Dextra's last decision simply holds. v3 reproduces that behaviour and adds MediaPipe for the still hand.

---

## Architecture

```
camera thread (latest frame, timestamped)
   │
   ▼  one decision thread, fixed order per frame
ROI ─► PseudoDVS ─(Dextra view after N events)─► Dextra ──► SequenceVote ─► decision ─► UDP → ESP32
   └─► MediaPipe HandLandmarker (same frame) ─► finger-curl rules ───────────► decision ─► UDP (if changed)
                                                                                 └─► HUD + latency log
```

| Stage | What it does |
|---|---|
| **PseudoDVS** ([rps/dvs_emulator.py](rps/dvs_emulator.py)) | Per-pixel log-intensity events against a reference level (contrast threshold C), as in v2e/ESIM. Global-gain correction absorbs auto-exposure drift, and a 3×3 filter removes noise. Events accumulate into a 64×64 histogram until **N = 1500** events, with binomial thinning on overshoot and clipping at **K = 16** (Dextra's normalization). A still scene emits nothing. |
| **Dextra** ([model.py](model.py), [rps/cnn.py](rps/cnn.py)) | Dextra's 5-layer network on the Dextra view: **Dextra Raw** as downloaded (98.75% on its own test images), **Dextra Tuned** with its last two layers tuned on your recordings ([train.py](train.py)). |
| **Vote** ([rps/voting.py](rps/voting.py)) | Dextra's default "sequence" filter: k = 2 identical, confident predictions in a row, within 250 ms. |
| **Mediapipe** ([rps/hand_tracker.py](rps/hand_tracker.py)) | 3D joint angles give finger curl, with two thresholds per finger (hysteresis). Rules: rock ≤ 1 finger extended, scissors = index + middle (or the middle hidden behind the index), paper ≥ 3. |
| **Decision** ([rps/decision.py](rps/decision.py)) | Both: Dextra leads while the hand moves; Mediapipe decides when the hand is still or when it steadily disagrees. Background never changes the command. Countdown and beat guide: `IDLE/ARMED → SHOOT → HOLD` per round; pumps from [rps/motion.py](rps/motion.py), beat timing from [rps/game.py](rps/game.py). |
| **Robot link** ([rps/robot_link.py](rps/robot_link.py)) | ASCII UDP. Team firmware: `RPS:ROCK` / `RPS:PAPER` / `RPS:SCISSORS` once per move; reference firmware: send on change plus a 100 ms heartbeat, with acknowledgements. |

## Measured on the demo laptop

Huawei FLMH-XX, Core Ultra 5 125H, "FHD Camera" (USB UVC):

| Item | Result |
|---|---|
| Camera | 30 fps maximum in every mode; uncompressed YUY2 reaches 30 fps only at 640×480 |
| Dim room, auto exposure | About **17 new images a second** either way: DirectShow 16.9 fps; Media Foundation reports 30 fps but sends ~42% of images twice (the copy ~2 ms after the original, measured on the 2026-09-27 night recordings). Auto exposure slows the camera; more light is the fix. The app and the probe skip the copies, so the frame rate shown is new images |
| Locked exposure | DirectShow: 30–31 fps with no jitter, **but the image is black without strong light**. The camera has no gain control, and manual exposure pins gain at its minimum. |
| Focus / depth | No focus control and no IR/depth sensor, so depth-from-focus is impossible (evaluated and rejected) |
| DVS emulator | ~0.9 ms per frame |
| MediaPipe | **13–14 ms median**, p95 ~23 ms with the process boost ([rps/perf.py](rps/perf.py) opts out of Windows EcoQoS throttling). Without the boost: 20–27 ms. |
| CNN | ~3 ms |
| UDP loopback round-trip | < 1 ms |

**Takeaway:** put a lamp on the play zone. More light lets you lock exposure for cleaner events and less motion blur.

---

## Setup

Full instructions, including the Raspberry Pi: [SETUP_GUIDE.md](SETUP_GUIDE.md). In short (Windows):

```powershell
uv venv --python 3.12 .venv
.venv\Scripts\activate
uv pip install -r requirements.txt  # training included; requirements-run.txt = running only
python tools/import_dextra.py       # Dextra Raw (not redistributed here: its repository has no license)
pytest tests/                       # the whole suite, including offscreen UI tests; never uses the real robot
```

Mediapipe's hand model (`models/hand_landmarker.task`, Apache 2.0) is in the repo. Dextra Tuned (`models/dextra_tuned.pth`) is not: copy it from the computer that trained it, or make one on the Train page.

## Desktop app (recommended)

```powershell
python app.py            # or: python app.py --mock   (synthetic camera, no webcam)
```

One window covers the whole workflow. Visible text is kept short and numeric; hover any control, reading or column title for what it does.

| Page | What you do there |
|---|---|
| **1 Setup** | Start the camera and check frame rate, light and overexposure; **Auto-configure**; **Set play zone** by dragging on the video; camera delay tests. Manual camera settings are folded away |
| **2 Bot tuning** | Everything about the robot: **real, simulated or off**, commands (team or reference firmware), address and port; **Test connection**; buttons that send **Rock**, **Paper** and **Scissors** one at a time (and Ready on the reference firmware), with a log; **Finger tuning**: a slider per servo channel (0 extended … 180 folded) that sends the team firmware's `ANGLE:<channel>,<angle>` and shows the robot's confirmation, plus Extend all / Fold all; the hand's move times; **Measure Wi-Fi delay** (ping, for the Speed card) |
| **3 Record** | Choose person, session type and gesture, press **Record** (3-second countdown). A live Dextra view confirms movement is seen; the progress list shows which of the 7 sessions this person still needs |
| **4 Dataset** | All recordings with totals and free disk space; **Build** training images (warns about sessions where the hand is rarely visible); check labels with a grid of random samples per person and gesture |
| **5 Train** | **Tune Dextra** on your recordings: the result is **Dextra Tuned**. Accuracy check (each person left out in turn), tune and test on one person, or tune on everyone; live accuracy curves |
| **6 Evaluate** | Tick **All** or any of Dextra Raw, Dextra Tuned, Mediapipe and Both, pick recordings, **Run**: one row per model (holding right, changes per minute, throws right, how fast after the throw, false moves with no hand) and the best named. The **Dextra Raw view check** (folded) finds the camera view that suits Dextra Raw |
| **7 Play** | The demo. Scoreboard (Robot : You, round of 5, 10 or endless), a big count, the camera with each reader's answer, the delay graph and the Dextra view. A **Speed** card times every throw (laptop computing vs waiting for camera images vs hardware). Options: recognition, **robot plays** to win, draw or lose, rounds, pumps, **beat guide** on/off and its sound (drum, wood block, beep), tempo, count-in, beats between rounds and sound delay; **Play a round** previews the beat without the camera. The steady-beat and count volumes stay adjustable during a match |
| **Play Debug** | Every reading and rule, for tuning: decision, which reader made it, robot move, round, tempo; the Speed card and a table of the last throws' timings; robot plays to win, draw or lose; the Dextra and Mediapipe cards; turn/mirror and sensitivity while playing; the log |
| **Settings** | Every other setting by section (the robot is on Bot tuning, the match on Play); tick "Show advanced settings" for fine-tuning. Save / Undo changes / Reset to defaults |

**Pumps adapt to the player.** A pump is a low point of the hand followed by a real rise: the hand's height comes from the tracked wrist when Mediapipe sees it and is carried through gaps by the play zone's motion, and the rise needed is a fraction of the player's own recent pumps, so small and large pumps both count (on a real 45 s recording: 57 of 57 pumps; the earlier speed-based counter caught 48 and counted throws as pumps). The tempo is learned only within a round. After a decision, the low point the hand reaches while finishing the throw is ignored, and the next pump starts the next round. An open hand after pumping is the throw, whichever beat it comes on; rock is a throw that lands and stays down. The same recording: all 28 throws decided once each, 2 pumps counted in each round (how that player plays).

**Beat guide.** A steady beat keeps the tempo the whole match (with **Endless** rounds, until Stop); before each throw come louder hits (one per pump, shown as 3, 2, 1) and a double hit on **SHOOT**. Every sound has most of its energy above 250 Hz, because laptop speakers play almost nothing lower (a pure bass-drum beat would be silent on them); the steady beat and the count have separate volumes. The throw is read in a window around SHOOT (from 0.35 s before to 0.8 s after; later is a missed round, the player's point), so the robot does not depend on counting pumps at all. The sound, the on-screen count and the throw window all follow one start time; `game.audio_latency_ms` and the camera delay line them up.

Heavy jobs (probe, build, train, evaluate, Dextra import) run the command-line scripts below as background processes and stream their output into the tab, so the app and the CLI always behave the same.

## Workflow (command line)

### The menu, or the commands

Run with no arguments and you get a menu — Setup, Bot, Config, Play, Status, Quit. Type a number,
press Enter. Numbers only, so it works over SSH on a Pi with nothing installed but Python, and
Ctrl-C drops back into the menu rather than out of it.
```bash
python -m rps            # the menu (also: python rps.py)
```
It remembers the recognizer, the mode, what the robot plays to and whether the live HUD is on, in
`data/menu_recent.json` (yours, not tracked by git), so the next visit starts where you left off.

Every entry in the menu is the same code as the typed command, so use whichever you like:
```bash
python -m rps config --section robot     # the settings
python -m rps setup check                # camera fps, light, is a hand in the zone
python -m rps bot ping                   # Wi-Fi delay to the hand
python -m rps play --recognizer mediapipe --no-hud --headless 100
```
Global flags (`--config`, `--data-root`, `--set`, `--json`, `--no-color`, `-q`, `-v`) work before or
after the command name. `--json` prints one object instead of a screen of text.

### 0. Camera and play zone (do this in the real play lighting)
1. Close OBS and anything else using the camera. Turn off Huawei PC Manager "AI camera" effects and any Windows camera effects (auto-framing or background blur change the whole image and flood the emulator).
2. `python tools/camera_probe.py --write-config` picks the backend and exposure that give ≥ 28 fps at usable brightness.
3. In the app's Setup page, **Set play zone**: drag a square around where your hand plays, **excluding your face and body**, like Dextra's camera framing.

### 1. Play right away with MediaPipe (no training needed)
```powershell
python play.py --recognizer mediapipe --mock-esp        # the simulated robot prints every pose it receives
```
Keys: `q` quit · `m` switch countdown/continuous · `r` reset counters. The HUD shows the camera fps, stage latencies, the pseudo-DVS preview, raw CNN/MediaPipe readings, the state machine, and the ESP link.

### 2. Connect the hand (hardware team)
1. Flash [firmware/esp32_rps_receiver/esp32_rps_receiver.ino](firmware/esp32_rps_receiver/esp32_rps_receiver.ino) (Arduino-ESP32 core + ESP32Servo) and calibrate `POSE_ANGLES`.
2. Join the ESP's soft-AP `RPS-HAND` (password `rpsrobot123`); the ESP is at `192.168.4.1` (the default `robot.host`).
3. `python play.py --recognizer mediapipe` (or Bot tuning's Test connection and command buttons).
4. Measure camera latency: `python tools/latency_test.py --mode led`, then put the median into `latency.camera_latency_ms` (the app's Setup page does this for you).
5. Measure servo time between every pair of poses and fill in `latency.servo_transition_ms`. Choose the READY (`N`) pose that minimises the worst case.

### 3. Record data (Dextra's ROSHAMBO17 style: session-level labels; aim for ≥ 5 people)
Record raw lossless video once; frames are generated from it later, so emulator settings can change without re-recording. About 8 minutes per person:
```powershell
python record_session.py --person alice --type show --label rock --duration 75
python record_session.py --person alice --type show --label paper --duration 75
python record_session.py --person alice --type show --label scissors --duration 75
python record_session.py --person alice --type throws --label rock --throws 10 --duration 45
python record_session.py --person alice --type throws --label paper --throws 10 --duration 45
python record_session.py --person alice --type throws --label scissors --throws 10 --duration 45
python record_session.py --person alice --type background --duration 60
```
- **show:** hold the gesture the *whole* time while moving, rotating, changing distance, and switching hands. No pumping.
- **throws:** pump ×3 then throw; hold about 1 s; relax; repeat.
- **background:** keep your hand out of the box; move your body and arm around it; change the lights.
- Start with one 5-minute pilot, then check the pseudo-DVS preview and the output of `build_dataset.py` before the full round.

### 4. Build frames, train, evaluate
```powershell
python build_dataset.py                       # N ∈ {750, 1500, 3000} × frame-skip {1, 2}; MediaPipe label cleaning
python train.py --lopo                        # leave-one-person-out: honest accuracy estimate
python train.py --val_person bob              # tunes Dextra's model -> models/dextra_tuned.pth (--all: everyone)
python train.py --start_from scratch --all    # a new network instead -> models/motion_cnn_v3.pth
python replay_eval.py --recordings data/recordings/bob   # exact live pipeline on held-out recordings
```
`replay_eval.py` compares **Dextra Raw / Dextra Tuned / Mediapipe / Both** (`--sources dextra_raw,dextra_tuned,mediapipe,both`) on:
- hold accuracy and switches per minute (show sessions);
- throw accuracy, and commit time versus the throw (when paper/scissors first shows to Mediapipe, or when a rock throw stops moving down), the same reference for every method;
- false commits (background);
- projected robot-visible time = commit + camera latency + servo transition.

It ends with the best model on those recordings (most throws right, then most time holding right). Check it on people the models did not learn from. Tune any setting without re-recording:
```powershell
python replay_eval.py --recordings data/recordings/bob --sweep vote.k=1,2,3
python replay_eval.py --recordings data/recordings/bob --set decision.still_frames=1
```

### Optional: Dextra's pretrained model (ROSHAMBO17, ~20 people)
```powershell
python tools/import_dextra.py        # downloads Dextra's weights -> models/dextra_roshambo.pth
python tools/dextra_transfer.py --recordings data/recordings/alice   # best orientation/settings for this camera
python play.py --recognizer dextra_raw --mode continuous --set cnn.flip=true
```
- The numpy weights are the complete model: Dextra's float exports (the SavedModel its own code runs, `roshambo.h5`, `modelroshambo.tf`) hold bit-identical values. The port scores 98.8% on ROSHAMBO17 test frames.
- Dextra's camera sees the hand side-on with fingers pointing left. Which orientation matches ours is measured, not assumed: on the first recordings (one person, mirrored webcam) the best was **mirrored back, not turned**, with 5000 events per image (69% of single images correct vs 60% at the defaults); a 90° turn was not in the top 15.
- Why it misreads scissors on the webcam: the implementation is correct (the exact runtime code scores 98.75% on ROSHAMBO17 test images), but the images differ. A real event camera draws thin, dense outlines (9% of pixels, average level 117); the 30 fps webcam gives either thin faint ones (N=1500: 13%, level 46) or thick blobs (N=5000: 21%, level 90). That is a data gap, not a bug: tuning Dextra's last two layers on 1.5 minutes of one person's recordings raised held-out scissors from 65% to 91% (paper 57% to 84%), while a new network trained from scratch on the same images reached only 67-77% overall. Tune on several people (`train.py`, the default `--start_from dextra`) and check it on a person it never saw.
- `cnn.flip` is on by default: the webcam view is mirrored for display, and Dextra reads the un-mirrored view better (60% to 67% of single images). A tuned model always uses the orientation it was tuned with.
- Licensing: ROSHAMBO17 is CC BY-SA 4.0. The Dextra repository has no license file, so its weights are git-ignored here and not redistributed.

### 5. Play
In the app: **Play** for a match (beat guide optional), **Play Debug** to watch every reading. From the command line:
```powershell
python -m rps play                            # the recognition, game and robot saved in config.json
python -m rps play --recognizer both --mode continuous
```
`rps play` shows a text HUD: whether the hand is inside the play zone, the camera fps, the last
gesture and what the robot did, the last throw's timing, the raw Dextra/MediaPipe readings, and the
mode and counters. `--no-hud` prints one line per decision instead, and `--headless N` runs N frames
and prints a summary (good for SSH and CI). `python play.py` still works exactly as before.

---

## Game modes

| Mode | Behaviour |
|---|---|
| **countdown** | `IDLE` → hand appears → `ARMED` (robot shows READY; pumps counted from the hand's height) → after 3 pumps `SHOOT`, or earlier on an open hand → paper/scissors commit as soon as they are seen; **rock commits only after the final downstroke has landed**, because the pumping fist is also "rock" → `HOLD`: the robot keeps its move until the next round's first pump or until the hand leaves. |
| **guided** (beat guide) | Rounds follow the beat: `ARMED` during the count-in (robot READY), `SHOOT` from 0.35 s before to 0.8 s after the SHOOT beat, `HOLD` while the result shows. Paper/scissors commit as soon as they show in the window; a hand still closed 0.3 s after SHOOT is rock; no readable throw is a missed round. |
| **continuous** | The robot always answers the current decision and returns to READY after 1 s with no hand. Good for demos and debugging. |

**What the robot plays** (`decision.robot_plays`, "Robot plays" on Play and Play Debug), in every mode:

| Robot plays | Its move | Your rock → robot | Round goes to |
|---|---|---|---|
| **to win** (default) | the move that beats your throw | paper | the robot |
| **to draw** | the same move as you (in Live mode it mirrors your hand) | rock | nobody (a draw) |
| **to lose** | the move your throw beats | scissors | you |

The Play page scores each round from the two moves (Robot, You and Draws); a round where no throw was read in time is your point.

## ESP32 protocol

Set `robot.protocol` (Bot tuning page: "Robot commands") to match the firmware on the hand. The pose sent is always what the **robot** shows; the counter logic stays on the PC.

**Team firmware (`rps_text`, default)**: address `192.168.0.126`, UDP port **4210**, one ASCII command per datagram:

| Robot shows | Command |
|---|---|
| Rock (all fingers folded) | `RPS:ROCK` |
| Paper (all fingers extended) | `RPS:PAPER` |
| Scissors (index and middle extended) | `RPS:SCISSORS` |

Each command is sent once, when the robot's move changes (so once per countdown round, even if the move repeats). The firmware has no ready position and does not reply: while you pump, the hand keeps its last move; Play shows how many moves were sent instead of a reply time; the LED camera-delay test is not available (use the mirror test). Bot tuning's "Test connection" sends `RPS:PAPER` once, so the hand should open; its Rock, Paper and Scissors buttons send each command on its own. While Bot tuning is open it keeps one connection, so every command (and finger angle) comes from the same port, as the team's own tool sends them; each click is sent at once, in order, and shown in the log with anything the robot sends back.

Finger tuning (Bot tuning page) uses the one command this firmware answers, the same as the team's own tuning tool: `ANGLE:<channel>,<angle>`, with channel 0 = pinky + ring, 1 = index, 2 = middle + point and angle 0 (extended) to 180 (folded). The firmware sends `CONFIRM_ANGLE:ch=<channel>(<label>),deg=<angle>` back to the sender; the page shows it, or "No confirmation" after 1.5 s (then check Windows Firewall). The last angle per channel is kept in `robot.finger_angles`.

**Reference firmware (`ack`)**: [firmware/esp32_rps_receiver](firmware/esp32_rps_receiver/esp32_rps_receiver.ino), ASCII over UDP, port **4210**:

| Direction | Message | Meaning |
|---|---|---|
| PC → ESP | `P,<seq>,<pose>,<pc_ms>` | Pose the **robot** must show: `R`, `P`, `S` or `N` (READY). Sent on change and every 100 ms. |
| PC → ESP | `L,<seq>,<0\|1>` | LED off/on (latency test) |
| ESP → PC | `A,<seq>,<esp_ms>,<reset_reason>` | Acknowledgement; `reset_reason` 9 = brownout (ESP_RST_BROWNOUT) |

Firmware must-dos (either firmware):
- `WiFi.setSleep(false)` — modem sleep adds 100 ms or more.
- Servos jump straight to their target, with no easing.
- Power the servos separately, with a bulk capacitor.
- Reference firmware: the ESP falls back to READY after 2 s without packets.
- Discard the rest of any oversized packet (`udp.flush()`), or the ESP stops receiving.
- If acknowledgements never arrive, check that Windows Firewall allows Python on that network.

The Play page stops the game (robot to READY) when you leave the page or the camera stops, and warns when the robot restarts during play (a power dip, reset reason 9).

`python tools/mock_esp.py` (or "Simulated robot" in the app) stands in for either firmware during development.

## Key settings ([config.json](config.json); override with `--set section.key=value`)

| Setting | Default | Effect |
|---|---|---|
| `dvs.contrast_threshold` | 0.20 | Lower = more events and more noise |
| `decision.recognizer` | both | `dextra_raw`, `dextra_tuned`, `mediapipe` or `both` (Dextra Tuned + Mediapipe) |
| `decision.mode` | countdown | `countdown`, `guided` (beat guide) or `continuous` (live) |
| `decision.robot_plays` | win | `win` (beats your throw), `draw` (copies it) or `lose` (plays what it beats) |
| `dvs.event_count` | 1500 | Movement per Dextra view (Dextra DVS128 value) |
| `vote.k` / `vote.min_confidence` | 2 / 0.70 | Dextra answers needed in a row / minimum confidence |
| `decision.mp_stable_ms` | 45 | Mediapipe gesture must hold this long (3 frames at 30 fps; a time, so it means the same at any frame rate) |
| `decision.pumps_before_shoot` | 3 | Pumps before the throw (beat guide: the loud thumps before SHOOT) |
| `decision.pump_min_rise` | 0.02 | Smallest rise that counts as a pump, as a share of the play zone (grows with the player's pumps) |
| `game.rounds` / `game.beat_bpm` | 5 / 150 | Match length (0 = endless); beat guide tempo |
| `game.sound` / `game.beat_volume` / `game.cue_volume` | drum / 0.6 / 1.0 | Beat sound; steady beat and count volumes |
| `robot.mode` / `robot.protocol` | real / rps_text | Real, simulated or off; team or reference firmware |
| `camera.lock_exposure` / `exposure` | false / −6 | Lock only with a lit play zone (see measurements) |
| `camera.backend` / `camera.url` | msmf / "" | Capture driver (msmf/dshow on Windows become v4l2 on Linux); an IP camera address instead of the USB camera |
| `latency.camera_latency_ms` / `network_ms` | 50 / 0 | Camera delay (Setup's test) and one-way Wi-Fi delay (Bot tuning's ping; 0 = not measured), for the Speed card |

## Speed: where the time goes

Play and Play Debug show a **Speed** card for every throw: a bar from the throw appearing to the robot's hand having moved, with a line where the UDP command leaves the laptop. Play Debug also lists the last throws.

| Part | Colour | Where the number comes from |
|---|---|---|
| Camera | grey (hardware) | Setup's camera delay test (`latency.camera_latency_ms`) |
| Waiting for camera images | amber | Measured: first camera image showing the throw → the image it was decided on. A throw is confirmed on 2–3 images, so this is set by the camera's frame rate (33 ms per image at 30 fps). Rock waits for the hand to stop, because the pumping fist looks the same. |
| Computing + UDP send | green (the laptop) | Measured: the deciding image arriving → the UDP command leaving (Dextra view, Dextra, Mediapipe, the send) |
| Wi-Fi | grey | Bot tuning's **Measure Wi-Fi delay** (ping; the ESP32 answers pings whatever its firmware): half the round trip |
| Robot hand moves | grey | Bot tuning's move times (`latency.servo_transition_ms`); measure them with 240 fps slow motion |

Measured on the demo laptop (Core Ultra 5 125H, 2026-09-29, `python tools/benchmark.py`), per camera frame, median (p90):

| | Dextra view | Dextra | Mediapipe | Total per frame |
|---|---|---|---|---|
| Dextra Tuned | 1.4 ms | 1.2 ms | – | 3.6 ms (5.2) |
| Mediapipe | 1.4 ms | – | 13.2 ms (26) | 16.4 ms (29) |
| Both | 1.5 ms | 1.4 ms | 13.2 ms (26) | 17.9 ms (31) |

Per throw, on the 2026-09-27 recordings (camera at ~17 new images a second): computing + UDP send 4–23 ms (median per recognizer); first image showing paper or scissors → command sent 110–170 ms median, almost all of it waiting for camera images; rock 230–260 ms (it waits for the hand to stop). At a real 30 fps the waiting roughly halves.

Keep the laptop **plugged in**: on battery Windows slowed Mediapipe from 13 ms to 47 ms per frame in the same test. `python play.py` prints each throw's numbers and a summary too.

Final acceptance: film human and robot hands together in 240 fps phone slow-motion and count frames from "human gesture complete" to "robot pose complete".

## Running on a Raspberry Pi 5 (stand-alone)

The running part (the app, `play.py`) runs on a Raspberry Pi 5; training stays on the laptop (copy `models/` and `config.json` across).

1. **OS:** Raspberry Pi OS **Trixie**, 64-bit. PySide6's Linux ARM build needs glibc 2.39 or newer; the older Bookworm (2.36) cannot install it. Every other package (mediapipe, torch, opencv) has a Linux ARM build.
2. **Install:** `python3 -m venv .venv && .venv/bin/pip install -r requirements-run.txt` (the running part only; `requirements.txt` adds the training packages).
3. **Camera:** a USB webcam. The capture driver becomes V4L2 automatically (`camera.backend` msmf/dshow → v4l2), exposure locks are translated, and Setup's Auto-configure probes V4L2. An IP camera works through `camera.url` (`rtsp://...` or `http://...` MJPEG, opened with FFmpeg's buffering off), but expect 100 ms or more of extra delay: prefer USB.
4. **Check the speed:** `python tools/benchmark.py` (the same recording as on the laptop gives a direct comparison; it warns about throttling). Use the official 27 W power supply and the Active Cooler, or the Pi slows itself down.
5. **Demo at boot:** `python app.py --fullscreen --page play`, e.g. from `~/.config/autostart/rps.desktop` (`Exec=/home/pi/RPS-System/.venv/bin/python /home/pi/RPS-System/app.py --fullscreen --page play`).
6. **Sound:** the Pi 5 has no headphone jack: the beat guide plays through the HDMI screen's speakers, or a USB speaker. A Bluetooth speaker adds 150–250 ms: set the beat guide's sound delay to match.
7. **Network:** the Pi and the robot on the same Wi-Fi (or the Pi as a hotspot the ESP32 joins, so no venue Wi-Fi is needed).

Expected speed on the Pi 5 (about 2.5–3.5 times slower per frame than the laptop): Dextra Tuned alone keeps up with 30 fps (~10–15 ms per frame). Mediapipe needs ~30–40 ms per frame (published Pi 5 hand-tracking runs reach ~26–28 fps with one hand), so with Mediapipe or Both it runs on about every other camera frame and decisions that Mediapipe makes come ~30–60 ms later than on the laptop. Measure with `tools/benchmark.py` on the Pi.

## Project structure

```
app.py               desktop app (PySide6): Setup, Bot tuning, Record, Dataset, Train, Evaluate, Play, Play Debug, Settings
                     (--fullscreen --page play for a stand-alone demo)
play.py              live runtime without the app window (camera or --video replay, HUD, robot link)
record_session.py    raw lossless session recorder (FFV1 + timestamps + meta)
build_dataset.py     recordings -> pseudo-DVS frames (multi-N, frame-skip, MediaPipe labels)
train.py             tunes Dextra -> Dextra Tuned, split by person (--lopo / --val_person / --all)
replay_eval.py       offline evaluation of the exact live pipeline, per recognizer
model.py             Dextra's network (and RoshamboNet for train.py --start_from scratch), MajorityVote, counter moves
config.json          all tunables (defaults in rps/config.py)
rps/                 camera, dvs_emulator, cnn, hand_tracker, motion (pumps), game (beat timing), voting,
                     decision, pipeline, robot_link, netping (Wi-Fi delay), recorder, hud, timing, perf, config
rps/ui/              desktop app: camera worker, pages, play_base (shared by Play and Play Debug), delay_view
                     (the Speed card), sound
tools/               benchmark (per-stage speed on any computer), camera_probe, latency_test, mock_esp,
                     import_dextra, dextra_transfer
requirements-run.txt the running part only (e.g. a Raspberry Pi); requirements.txt adds training
firmware/            ESP32 reference receiver
tests/               pytest suite (emulator, pumps, beat guide, voting, rules, decision, protocol, pipeline,
                     Dextra import and tuning, evaluation, UI); fixtures/ holds a real session's numbers
```

## Acknowledgements

- **Dextra** and the RoShamBo demonstrator, Sensors Group, Institute of Neuroinformatics, UZH-ETH Zurich (Tobi Delbruck et al.): [dextra-roshambo-python](https://github.com/SensorsINI/dextra-roshambo-python), [Dextra hand](https://sensorsini.github.io/dextra-robot-hand/).
- X. Deng, S. Weirich, R. Katzschmann, T. Delbruck, *A Rapid and Robust Tendon-Driven Robotic Hand for Human-Robot Interactions Playing Rock-Paper-Scissors*, IEEE RO-MAN 2024.
- I. Lungu, F. Corradi, T. Delbruck, *Live demonstration: Convolutional neural network driven by dynamic vision sensor playing RoShamBo*, ISCAS 2017 (ROSHAMBO17 dataset).
