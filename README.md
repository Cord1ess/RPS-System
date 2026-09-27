# RPS v3: Dextra-style Rock-Paper-Scissors with a Laptop Webcam

A rock-paper-scissors robot hand (3D-printed, servo tendons, ESP32 over UDP) that sees your throw and plays the winning move fast enough to look simultaneous. The perception software copies the method of **Dextra** ([SensorsINI](https://sensors.ini.ch/research/projects/dextra)), but runs on an ordinary laptop webcam instead of a DVS event camera.

- **Primary path:** a webcam **DVS emulator** that emits constant-event-count 64×64 frames only when something moves, feeding **RoshamboNet**, with a Dextra-style sequence vote.
- **Fallback and still-hand authority:** **MediaPipe** hand landmarks with finger-curl rules.
- **Game modes:** *countdown* ("rock, paper, scissors, shoot") and *continuous* (Dextra demo style), switchable live.

> **Why v3?** The v2 system (removed; it is in the git history before this version) ran the CNN on every webcam frame. When your hand is still, the frame difference is blank, and blank frames were never in training, so the output jumped randomly. A real DVS produces *no frames* when nothing moves, so Dextra's last decision simply holds. v3 reproduces that behaviour and adds MediaPipe for the still hand.

---

## Architecture

```
camera thread (latest frame, timestamped)
   │
   ▼  one decision thread, fixed order per frame
ROI ─► PseudoDVS ─(frame only after N events)─► RoshamboNet ─► SequenceVote ─► decision ─► UDP → ESP32
   └─► MediaPipe HandLandmarker (same frame) ─► finger-curl rules ───────────► decision ─► UDP (if changed)
                                                                                 └─► HUD + latency log
```

| Stage | What it does |
|---|---|
| **PseudoDVS** ([rps/dvs_emulator.py](rps/dvs_emulator.py)) | Per-pixel log-intensity events against a reference level (contrast threshold C), as in v2e/ESIM. Global-gain correction absorbs auto-exposure drift, and a 3×3 filter removes noise. Events accumulate into a 64×64 histogram until **N = 1500** events, with binomial thinning on overshoot and clipping at **K = 16** (Dextra's normalization). A still scene emits nothing. |
| **RoshamboNet** ([model.py](model.py)) | Unchanged 115k-parameter CNN (Dextra: ~120k). |
| **Vote** ([rps/voting.py](rps/voting.py)) | Dextra's default "sequence" filter: k = 2 identical, confident predictions in a row, within 250 ms. |
| **MediaPipe** ([rps/hand_tracker.py](rps/hand_tracker.py)) | 3D joint angles give finger curl, with two thresholds per finger (hysteresis). Rules: rock ≤ 1 finger extended, scissors = index + middle, paper ≥ 3. |
| **Decision** ([rps/decision.py](rps/decision.py)) | The CNN leads while the hand moves. MediaPipe decides only when the hand box is still *and* the frame came after the last CNN commit. Background never changes the command. Countdown state machine: `IDLE → ARMED → SHOOT → HOLD`. |
| **Robot link** ([rps/robot_link.py](rps/robot_link.py)) | ASCII UDP: send on change plus a 100 ms heartbeat, with acknowledgements for round-trip time. |

## Measured on the demo laptop

Huawei FLMH-XX, Core Ultra 5 125H, "FHD Camera" (USB UVC):

| Item | Result |
|---|---|
| Camera | 30 fps maximum in every mode; uncompressed YUY2 reaches 30 fps only at 640×480 |
| Dim room, auto exposure | Media Foundation: **30.1 fps** · DirectShow: 16.9 fps (auto-exposure slows the camera) |
| Locked exposure | DirectShow: 30–31 fps with no jitter, **but the image is black without strong light**. The camera has no gain control, and manual exposure pins gain at its minimum. |
| Focus / depth | No focus control and no IR/depth sensor, so depth-from-focus is impossible (evaluated and rejected) |
| DVS emulator | ~0.9 ms per frame |
| MediaPipe | **13–14 ms median**, p95 ~23 ms with the process boost ([rps/perf.py](rps/perf.py) opts out of Windows EcoQoS throttling). Without the boost: 20–27 ms. |
| CNN | ~3 ms |
| UDP loopback round-trip | < 1 ms |

**Takeaway:** put a lamp on the play zone. More light lets you lock exposure for cleaner events and less motion blur.

---

## Setup

```powershell
uv venv --python 3.12 .venv
.venv\Scripts\activate
uv pip install -r requirements.txt
curl -L -o models/hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
pytest tests/                       # 35 tests (incl. an offscreen UI test)
```

## Desktop app (recommended)

```powershell
python app.py            # or: python app.py --mock   (synthetic camera, no webcam)
```

One window covers the whole workflow. Visible text is kept short and numeric; hover any control, reading or column title for what it does.

| Page | What you do there |
|---|---|
| **1 Setup** | Start the camera and check frame rate, light and overexposure; **Auto-configure**; **Set play zone** by dragging on the video; test the robot connection. Manual camera settings and camera delay tests are folded away |
| **2 Record** | Choose person, session type and gesture, press **Record** (3-second countdown). A live motion preview confirms movement is seen; the progress list shows which of the 7 sessions this person still needs |
| **3 Dataset** | All recordings with totals and free disk space; **Build** training images; check labels with a grid of random samples per person and gesture |
| **4 Train** | Accuracy check (each person left out in turn), train and test on one person, or train on everyone; live accuracy curves and the expected accuracy on a new person |
| **5 Evaluate** | **Dextra model check** (no training): ranks rotation, flip, movement per image and zoom, and saves your motion images next to Dextra's. **Compare recognition methods** on recordings, with a verdict |
| **6 Play** | Top: You, Robot, Round (for example "Pump 2 of 3") and your learned Tempo (s per pump). Camera view, the **delay graph** (processing, motion model, hand tracker, camera interval) and health readouts. Right: **Run** (recognition, game, model, robot; shrinks to one line while playing) and **Readings** (motion image beside Dextra's examples, confidence bars, hand tracker, decision). Orientation, tuning and log are folded away |
| **Settings** | Every setting by section; tick "Show advanced settings" for fine-tuning. Save / Undo changes / Reset to defaults |

**Countdown pumps adapt to the player.** Pumps are measured from the hand's up-and-down speed in the play zone (optical flow), so a lost hand track does not lose a pump. The app learns each player's tempo (only from the gaps between pumps of one round, never from the pause after a throw) and stroke size, and scales its thresholds to them. Once enough pumps are counted, an open hand (paper or scissors) is the throw, whichever beat it comes on, so players who throw on the 3rd or the 4th down stroke both work; rock is the stroke that stops instead of reversing. Movement in the first 0.3 s after a result is ignored and the next pump starts a new round. Checked on a real 45 s recording (tests/fixtures): all 28 throws decided once each, none during pumps.

Heavy jobs (probe, build, train, evaluate, Dextra import) run the command-line scripts below as background processes and stream their output into the tab, so the app and the CLI always behave the same.

## Workflow (command line)

### 0. Camera and play zone (do this in the real play lighting)
1. Close OBS and anything else using the camera. Turn off Huawei PC Manager "AI camera" effects and any Windows camera effects (auto-framing or background blur change the whole image and flood the emulator).
2. `python tools/camera_probe.py --write-config` picks the backend and exposure that give ≥ 28 fps at usable brightness.
3. In the app's Setup page, **Set play zone**: drag a square around where your hand plays, **excluding your face and body**, like Dextra's camera framing.

### 1. Play right away with MediaPipe (no training needed)
```powershell
python play.py --source mediapipe --mock-esp            # the mock ESP prints every pose it receives
```
Keys: `q` quit · `m` switch countdown/continuous · `r` reset counters. The HUD shows the camera fps, stage latencies, the pseudo-DVS preview, raw CNN/MediaPipe readings, the state machine, and the ESP link.

### 2. Connect the hand (hardware team)
1. Flash [firmware/esp32_rps_receiver/esp32_rps_receiver.ino](firmware/esp32_rps_receiver/esp32_rps_receiver.ino) (Arduino-ESP32 core + ESP32Servo) and calibrate `POSE_ANGLES`.
2. Join the ESP's soft-AP `RPS-HAND` (password `rpsrobot123`); the ESP is at `192.168.4.1` (the default `robot.host`).
3. `python play.py --source mediapipe`. The HUD should show `ESP: connected, rtt …`.
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
python train.py --val_person bob              # or --all for the final model -> models/motion_cnn_v3.pth
python replay_eval.py --recordings data/recordings/bob   # exact live pipeline on held-out recordings
```
`replay_eval.py` compares **mediapipe / cnn / fused** on:
- hold accuracy and switches per minute (show sessions);
- throw accuracy, and commit time versus the throw (when paper/scissors first shows to the hand tracker, or when a rock throw stops moving down), the same reference for every method;
- false commits (background);
- projected robot-visible time = commit + camera latency + servo transition.

It ends with the **go/no-go rule**: if fused doesn't beat MediaPipe-only on held-out people, ship MediaPipe-only. Tune any setting without re-recording:
```powershell
python replay_eval.py --recordings data/recordings/bob --sweep vote.k=1,2,3
python replay_eval.py --recordings data/recordings/bob --set decision.still_frames=1
```

### Optional: Dextra's pretrained model (ROSHAMBO17, ~20 people)
```powershell
python tools/import_dextra.py        # downloads Dextra's weights -> models/dextra_roshambo.pth
python tools/dextra_transfer.py --recordings data/recordings/alice   # best orientation/settings for this camera
python play.py --source cnn --mode continuous --set cnn.flip=true --set dvs.event_count=5000
```
- The numpy weights are the complete model: Dextra's float exports (the SavedModel its own code runs, `roshambo.h5`, `modelroshambo.tf`) hold bit-identical values. The port scores 98.8% on ROSHAMBO17 test frames.
- Dextra's camera sees the hand side-on with fingers pointing left. Which orientation matches ours is measured, not assumed: on the first recordings (one person, mirrored webcam) the best was **mirrored back, not turned**, with 5000 events per image (69% of single images correct vs 60% at the defaults); a 90° turn was not in the top 15.
- Our 30 fps webcam events are blurrier than a real DVS. Whether the model transfers has to be measured with `replay_eval.py`; if it transfers only partly, fine-tune it on our recordings.
- Licensing: ROSHAMBO17 is CC BY-SA 4.0. The Dextra repository has no license file, so its weights are git-ignored here and not redistributed.

### 5. Play fused
```powershell
python play.py                                # fused, countdown mode, ESP at config address
python play.py --mode continuous
```

---

## Game modes

| Mode | Behaviour |
|---|---|
| **countdown** (default) | `IDLE` → hand appears → `ARMED` (robot shows READY; pumps counted from vertical reversals) → after 3 pumps `SHOOT`, or earlier on an open hand after 2 → paper/scissors commit as soon as they are seen; **rock commits only after the final downstroke has landed**, because the pumping fist is also "rock" → `HOLD`: the robot keeps its move until the next round's first pump (movement in the first 0.3 s is ignored) or until the hand leaves. |
| **continuous** | The robot always shows the counter to the current decision and returns to READY after 1 s with no hand. Good for demos and debugging. |

## ESP32 protocol

ASCII over UDP, one message per datagram, port **4210**:

| Direction | Message | Meaning |
|---|---|---|
| PC → ESP | `P,<seq>,<pose>,<pc_ms>` | Pose the **robot** must show: `R`, `P`, `S` or `N` (READY). Sent on change and every 100 ms. |
| PC → ESP | `L,<seq>,<0\|1>` | LED off/on (latency test) |
| ESP → PC | `A,<seq>,<esp_ms>,<reset_reason>` | Acknowledgement; `reset_reason` 9 = brownout (ESP_RST_BROWNOUT) |

Firmware must-dos:
- `WiFi.setSleep(false)` — modem sleep adds 100 ms or more.
- Servos jump straight to their target, with no easing.
- Power the servos separately, with a bulk capacitor.
- The ESP falls back to READY after 2 s without packets.
- Discard the rest of any oversized packet (`udp.flush()`), or the ESP stops receiving.
- If acknowledgements never arrive, check that Windows Firewall allows Python on that network.

The Play page stops the game (robot to READY) when you leave the page or the camera stops, and warns when the robot restarts during play (a power dip, reset reason 9).

`python tools/mock_esp.py` stands in for the ESP during development.

## Key settings ([config.json](config.json); override with `--set section.key=value`)

| Setting | Default | Effect |
|---|---|---|
| `dvs.contrast_threshold` | 0.20 | Lower = more events and more noise |
| `dvs.event_count` | 1500 | Events per CNN frame (Dextra DVS128 value) |
| `vote.k` / `vote.min_confidence` | 2 / 0.70 | CNN predictions needed in a row / minimum confidence |
| `decision.still_frames` | 2 | Frames of stillness before "settled" (rock commit ≈ 67 ms after landing) |
| `decision.mp_stable_frames` | 3 | MediaPipe gesture must repeat this many frames |
| `decision.pumps_before_shoot` | 3 | Countdown pumps before SHOOT |
| `camera.lock_exposure` / `exposure` | false / −6 | Lock only with a lit play zone (see measurements) |

## Latency budget (target: robot pose visible ≤ ~200 ms after the human's throw, human perceptual latency per Dextra)

| Stage | Estimate |
|---|---|
| Camera (exposure, readout, driver) | 30–70 ms: measure with `tools/latency_test.py` |
| Emulator + CNN | ~4 ms |
| Vote, 2 frames at 30 fps | ~33 ms (paper/scissors often commit *before* the hand lands) |
| UDP over Wi-Fi with sleep off | 2–10 ms |
| Servo and tendon travel | Measure: hobby servos need 150–200 ms at full travel, so pre-positioning (READY pose) matters |

Final acceptance: film human and robot hands together in 240 fps phone slow-motion and count frames from "human gesture complete" to "robot pose complete".

## Project structure

```
app.py               desktop control centre (PySide6): Setup, Record, Dataset, Train, Evaluate, Play, Settings
play.py              live runtime (camera or --video replay, HUD, ESP link)
record_session.py    raw lossless session recorder (FFV1 + timestamps + meta)
build_dataset.py     recordings -> pseudo-DVS frames (multi-N, frame-skip, MediaPipe labels)
train.py             RoshamboNet training, split by person (--lopo / --val_person / --all)
replay_eval.py       offline evaluation of the exact live pipeline + go/no-go
model.py             RoshamboNet, Dextra's network, MajorityVote, counter moves
config.json          all tunables (defaults in rps/config.py)
rps/                 camera, dvs_emulator, cnn, hand_tracker, voting, decision, pipeline,
                     robot_link, recorder, hud, timing, perf, config
rps/ui/              desktop app: camera worker, tabs, shared widgets
tools/               camera_probe, latency_test, mock_esp, import_dextra, dextra_transfer
firmware/            ESP32 reference receiver
tests/               pytest suite (emulator, voting, rules, decision, protocol, pipeline, Dextra import, UI)
```

## Acknowledgements

- **Dextra** and the RoShamBo demonstrator, Sensors Group, Institute of Neuroinformatics, UZH-ETH Zurich (Tobi Delbruck et al.): [dextra-roshambo-python](https://github.com/SensorsINI/dextra-roshambo-python), [Dextra hand](https://sensorsini.github.io/dextra-robot-hand/).
- X. Deng, S. Weirich, R. Katzschmann, T. Delbruck, *A Rapid and Robust Tendon-Driven Robotic Hand for Human-Robot Interactions Playing Rock-Paper-Scissors*, IEEE RO-MAN 2024.
- I. Lungu, F. Corradi, T. Delbruck, *Live demonstration: Convolutional neural network driven by dynamic vision sensor playing RoShamBo*, ISCAS 2017 (ROSHAMBO17 dataset).
