# RPS v3: Dextra-style Rock-Paper-Scissors with a Laptop Webcam

A rock-paper-scissors robot hand (3D-printed, servo tendons, ESP32 over UDP) that sees your throw and plays the winning move fast enough to look simultaneous. The perception software copies the method of **Dextra** ([SensorsINI](https://sensors.ini.ch/research/projects/dextra)), but runs on an ordinary laptop webcam instead of a DVS event camera.

- **Primary path:** a webcam **DVS emulator** that emits constant-event-count 64×64 frames only when something moves, feeding **RoshamboNet**, with a Dextra-style sequence vote.
- **Fallback and still-hand authority:** **MediaPipe** hand landmarks with finger-curl rules.
- **Game modes:** *countdown* ("rock, paper, scissors, shoot") and *continuous* (Dextra demo style), switchable live.

> **Why v3?** The v2 system (now in [`legacy/`](legacy/)) ran the CNN on every webcam frame. When your hand is still, the frame difference is blank, and blank frames were never in training, so the output jumped randomly. A real DVS produces *no frames* when nothing moves, so Dextra's last decision simply holds. v3 reproduces that behaviour and adds MediaPipe for the still hand.

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
pytest tests/                       # 32 tests
```

## Workflow

### 0. Camera and play zone (do this in the real play lighting)
1. Close OBS and anything else using the camera. Turn off Huawei PC Manager "AI camera" effects and any Windows camera effects (auto-framing or background blur change the whole image and flood the emulator).
2. `python tools/camera_probe.py --write-config` picks the backend and exposure that give ≥ 28 fps at usable brightness.
3. `python tools/set_roi.py`: drag a square around where your hand plays, **excluding your face and body**, like Dextra's camera framing.

### 1. Play right away with MediaPipe (no training needed)
```powershell
python play.py --source mediapipe --mock-esp            # the mock ESP prints every pose it receives
```
Keys: `q` quit · `m` switch countdown/continuous · `r` reset counters. The HUD shows the camera fps, stage latencies, the pseudo-DVS preview, raw CNN/MediaPipe readings, the state machine, and the ESP link.

### 2. Connect the hand (hardware team)
1. Flash [firmware/esp32_rps_receiver/esp32_rps_receiver.ino](firmware/esp32_rps_receiver/esp32_rps_receiver.ino) (Arduino-ESP32 core + ESP32Servo) and calibrate `POSE_ANGLES`.
2. Join the ESP's soft-AP `RPS-HAND` (password `rpsrobot123`); the ESP is at `192.168.4.1` (the default `robot.host`).
3. `python play.py --source mediapipe`. The HUD should show `ESP: connected, rtt …`.
4. Measure camera latency: `python tools/latency_test.py --mode led`, then put the median into `latency.camera_latency_ms`.
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
- throw accuracy, and commit time versus when the hand comes to rest (throws);
- false commits (background);
- projected robot-visible time = commit + camera latency + servo transition.

It ends with the **go/no-go rule**: if fused doesn't beat MediaPipe-only on held-out people, ship MediaPipe-only. Tune any setting without re-recording:
```powershell
python replay_eval.py --recordings data/recordings/bob --sweep vote.k=1,2,3
python replay_eval.py --recordings data/recordings/bob --set decision.still_frames=1
```

### 5. Play fused
```powershell
python play.py                                # fused, countdown mode, ESP at config address
python play.py --mode continuous
```

---

## Game modes

| Mode | Behaviour |
|---|---|
| **countdown** (default) | `IDLE` → hand appears → `ARMED` (robot shows READY; pumps counted from vertical reversals) → after 3 pumps `SHOOT` → paper/scissors commit as soon as the vote agrees; **rock commits only after the final downstroke has landed and settled**, because the pumping fist is also "rock" → `HOLD` (≥ 1 s) → a new pump or the hand leaving starts the next round. |
| **continuous** | The robot always shows the counter to the current decision and returns to READY after 1 s with no hand. Good for demos and debugging. |

## ESP32 protocol

ASCII over UDP, one message per datagram, port **4210**:

| Direction | Message | Meaning |
|---|---|---|
| PC → ESP | `P,<seq>,<pose>,<pc_ms>` | Pose the **robot** must show: `R`, `P`, `S` or `N` (READY). Sent on change and every 100 ms. |
| PC → ESP | `L,<seq>,<0\|1>` | LED off/on (latency test) |
| ESP → PC | `A,<seq>,<esp_ms>,<reset_reason>` | Acknowledgement; `reset_reason` 15 = brownout |

Firmware must-dos:
- `WiFi.setSleep(false)` — modem sleep adds 100 ms or more.
- Servos jump straight to their target, with no easing.
- Power the servos separately, with a bulk capacitor.
- The ESP falls back to READY after 2 s without packets.
- Allow Python through Windows Firewall, or acknowledgements never arrive.

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
play.py              live runtime (camera or --video replay, HUD, ESP link)
record_session.py    raw lossless session recorder (FFV1 + timestamps + meta)
build_dataset.py     recordings -> pseudo-DVS frames (multi-N, frame-skip, MediaPipe labels)
train.py             RoshamboNet training, split by person (--lopo / --val_person / --all)
replay_eval.py       offline evaluation of the exact live pipeline + go/no-go
model.py             RoshamboNet, MajorityVote, counter-move table
config.json          all tunables (defaults in rps/config.py)
rps/                 camera, dvs_emulator, cnn, hand_tracker, voting, decision, pipeline,
                     robot_link, hud, timing, perf, config
tools/               camera_probe, set_roi, latency_test, mock_esp
firmware/            ESP32 reference receiver
tests/               pytest suite (emulator, voting, rules, decision, protocol, pipeline)
legacy/              v2 frame-differencing system, dataset and model (reference)
```

## Acknowledgements

- **Dextra** and the RoShamBo demonstrator, Sensors Group, Institute of Neuroinformatics, UZH-ETH Zurich (Tobi Delbruck et al.): [dextra-roshambo-python](https://github.com/SensorsINI/dextra-roshambo-python), [Dextra hand](https://sensorsini.github.io/dextra-robot-hand/).
- X. Deng, S. Weirich, R. Katzschmann, T. Delbruck, *A Rapid and Robust Tendon-Driven Robotic Hand for Human-Robot Interactions Playing Rock-Paper-Scissors*, IEEE RO-MAN 2024.
- I. Lungu, F. Corradi, T. Delbruck, *Live demonstration: Convolutional neural network driven by dynamic vision sensor playing RoShamBo*, ISCAS 2017 (ROSHAMBO17 dataset).
