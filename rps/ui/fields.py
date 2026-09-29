"""
Plain-language name, hover explanation and level (basic / advanced) for every config field, plus
display names for every choice. The app uses one name for each thing:
  Dextra Raw / Dextra Tuned / Mediapipe / Both (Dextra Tuned + Mediapipe)  the four ways to read the hand
  Dextra view   the 64 x 64 movement picture Dextra reads
"""

# (section, key) -> (label, hover explanation, advanced?)
FIELD_INFO = {
    # camera
    ("camera", "index"): ("Camera number", "0 = built-in webcam.", True),
    ("camera", "url"): ("IP camera address", "rtsp://... or http://... (MJPEG) to use a network camera instead of "
                        "the USB one; empty = USB camera. A USB camera is faster: network cameras add 100 ms or "
                        "more of delay.", True),
    ("camera", "backend"): ("Capture driver", "Media Foundation: best in dim light. DirectShow: steady 30 fps with "
                            "a locked exposure. V4L2: Linux (Raspberry Pi); the Windows drivers become V4L2 there. "
                            "Auto-configure chooses.", True),
    ("camera", "width"): ("Width", "Pixels. 640 is the fastest uncompressed mode on this camera.", True),
    ("camera", "height"): ("Height", "Pixels.", True),
    ("camera", "fps"): ("Frame rate", "Frames per second requested. This camera's maximum is 30.", True),
    ("camera", "fourcc"): ("Pixel format", "YUY2 is uncompressed, so compression noise is not mistaken for "
                           "movement.", True),
    ("camera", "mirror"): ("Mirror image", "Show the camera like a mirror.", True),
    ("camera", "lock_exposure"): ("Lock exposure", "Fixed brightness and less blur. Needs a lamp on the play zone: "
                                  "this camera cannot boost a dark image.", False),
    ("camera", "exposure"): ("Exposure", "-5 = 31 ms, -6 = 16 ms, -7 = 8 ms per frame. Shorter = less blur, "
                             "darker image.", False),
    ("camera", "lock_white_balance"): ("Lock colour balance", "Stops colour shifts being read as movement.", True),
    ("camera", "wb_temperature"): ("Colour balance (K)", "Used when colour balance is locked.", True),
    ("camera", "restore_auto_on_exit"): ("Restore camera on exit", "Hand the camera back to other apps with "
                                         "automatic exposure.", True),
    # play zone
    ("roi", "x"): ("Play zone left", "Pixels from the left edge. Set by dragging on the Setup page.", True),
    ("roi", "y"): ("Play zone top", "Pixels from the top edge. Set by dragging on the Setup page.", True),
    ("roi", "size"): ("Play zone size", "Side length in pixels. Set by dragging on the Setup page.", True),
    ("roi", "record_margin"): ("Recording margin", "Extra border saved around the play zone, as a fraction of its "
                               "size.", True),
    # Dextra view
    ("dvs", "sensor_size"): ("Movement grid", "The play zone is resized to this many pixels before movement is "
                             "measured (Dextra: 128).", True),
    ("dvs", "frame_size"): ("Dextra view size", "Pixels per side of the image Dextra reads.", True),
    ("dvs", "log_offset"): ("Dark noise damping", "Higher = less false movement from dark, noisy pixels.", True),
    ("dvs", "contrast_threshold"): ("Movement sensitivity", "Brightness change counted as movement. Lower = more "
                                    "sensitive, more noise. Default 0.20.", False),
    ("dvs", "event_count"): ("Movement per image", "How much movement is collected into one Dextra view. "
                             "Dextra used 1500.", False),
    ("dvs", "clip_count"): ("Per-pixel cap", "Limit on movement counted in one pixel (Dextra: 16).", True),
    ("dvs", "noise_filter"): ("Noise filter", "Ignore single isolated moving pixels.", True),
    ("dvs", "global_reset_fraction"): ("Lighting change limit", "If more than this share of the picture changes at "
                                       "once, treat it as a lighting change, not movement.", True),
    ("dvs", "flush_min_fraction"): ("Partial image", "When the hand stops, still send a Dextra view if it has at "
                                    "least this share of the usual movement.", True),
    ("dvs", "still_events_per_frame"): ("Still threshold", "Movement per camera frame below which the scene counts "
                                        "as still.", True),
    ("dvs", "flush_still_frames"): ("Still frames before sending", "When the hand stops, wait this many still "
                                    "camera frames before sending the partial Dextra view.", True),
    ("dvs", "max_accumulation_s"): ("Max collection time (s)", "A Dextra view still not full after this long is "
                                    "sent early, or dropped if it holds too little movement.", True),
    # Dextra
    ("cnn", "raw_model"): ("Dextra Raw file", "Dextra's model as downloaded (Download Dextra on Play Debug).", True),
    ("cnn", "tuned_model"): ("Dextra Tuned file", "Dextra tuned on your recordings (made on the Train page).", True),
    ("cnn", "threads"): ("CPU threads", "1 is fastest for this small model.", True),
    ("cnn", "rotate"): ("Turn Dextra view", "Dextra Raw only: turn its image to match the camera angle Dextra was "
                        "trained on (side view, fingers pointing left). Dextra Tuned keeps its own.", False),
    ("cnn", "flip"): ("Mirror Dextra view", "Dextra Raw only: mirror its image left to right. Dextra Tuned keeps "
                      "its own.", False),
    # Mediapipe
    ("hand", "model_path"): ("Mediapipe file", "Hand model file (hand_landmarker.task).", True),
    ("hand", "enabled"): ("Use Mediapipe", "Allow Mediapipe to run.", True),
    ("hand", "min_detection_confidence"): ("Find-hand confidence", "0-1. Higher = fewer false hands.", True),
    ("hand", "min_presence_confidence"): ("Hand-present confidence", "0-1.", True),
    ("hand", "min_tracking_confidence"): ("Keep-tracking confidence", "0-1.", True),
    ("hand", "crop_margin"): ("Search margin", "Mediapipe also looks this far outside the play zone.", True),
    ("hand", "extend_threshold"): ("Finger straight below", "Finger bend under this counts as straight.", True),
    ("hand", "curl_threshold"): ("Finger bent above", "Finger bend over this counts as bent.", True),
    ("hand", "frame_budget_ms"): ("Time budget (ms)", "Skip one frame when Mediapipe takes longer than this.", True),
    # decision rules
    ("vote", "method"): ("Agreement rule", "How repeated answers from Dextra are combined.", True),
    ("vote", "k"): ("Answers in a row", "Dextra must give the same answer this many times before the robot moves. "
                    "2 = fast, 3 = safer.", False),
    ("vote", "min_confidence"): ("Minimum confidence", "Dextra answers below this confidence are ignored (0-1).",
                                 False),
    ("vote", "max_gap_s"): ("Max gap (s)", "Answers further apart than this do not count as in a row.", True),
    ("vote", "majority_window"): ("Majority window", "Recent answers considered by the majority rule.", True),
    # game
    ("decision", "mode"): ("Game", "Countdown: pump, then throw; pumps are counted from your hand. Beat guide: a "
                           "drum beat leads each round and you throw on SHOOT. Live: the robot answers "
                           "continuously.", False),
    ("decision", "recognizer"): ("Recognition", "Dextra Raw, Dextra Tuned, Mediapipe, or Both (Dextra Tuned + "
                                 "Mediapipe).", False),
    ("decision", "robot_plays"): ("Robot plays", "To win: the move that beats your throw. To draw: the same move. "
                                  "To lose: the move your throw beats.", False),
    ("decision", "active_events_per_frame"): ("Moving above", "Movement per camera frame that counts as the hand "
                                              "moving.", True),
    ("decision", "still_events_per_frame"): ("Still below", "Movement per camera frame that counts as still.", True),
    ("decision", "still_frames"): ("Still frames", "Frames without movement before the hand counts as stopped.",
                                   False),
    ("decision", "mp_stable_ms"): ("Mediapipe steady time (ms)", "Mediapipe must see the same gesture for this "
                                   "long before it counts (45 ms = 3 frames at 30 fps).", True),
    ("decision", "mp_min_confidence"): ("Mediapipe minimum confidence", "0-1.", True),
    ("decision", "mp_still_events_in_hand"): ("Mediapipe stillness limit", "In Both, Mediapipe decides only when the "
                                              "hand moves less than this.", True),
    ("decision", "mp_after_cnn_commit_s"): ("Mediapipe wait (s)", "In Both, wait this long after a Dextra decision "
                                            "before Mediapipe may change it.", True),
    ("decision", "switch_dead_time_s"): ("Change cooldown (s)", "Minimum time between Mediapipe-driven changes.",
                                         True),
    ("decision", "idle_timeout_s"): ("Idle after (s)", "No hand for this long = idle.", True),
    ("decision", "idle_action"): ("When idle", "What the robot does when nobody is playing.", True),
    ("decision", "pumps_before_shoot"): ("Pumps before the throw", "Pumps before the throw. With the beat guide: "
                                         "the loud beats before SHOOT.", False),
    ("decision", "pump_miss_tolerance"): ("Missed pumps allowed", "A throw still counts if this many pumps were "
                                          "missed.", True),
    ("decision", "pump_min_rise"): ("Smallest pump", "The smallest rise after a low point that counts as a pump, as "
                                    "a share of the play zone height. Grows with the player's own pumps.", True),
    ("decision", "pump_min_period_s"): ("Fastest pump (s)", "Pumps closer together than this are one pump until "
                                        "the player's tempo is learned.", True),
    ("decision", "shoot_window_s"): ("Throw window (s)", "Minimum time allowed for the throw; grows with a slow "
                                     "tempo.", True),
    ("decision", "rock_settle_fallback_s"): ("Rock fallback (s)", "Decide rock after this long if the landing was "
                                             "not seen.", True),
    ("decision", "hold_min_s"): ("Pause after result (s)", "Movement right after a result is ignored for this "
                                 "long. The robot keeps its move until your next pump.", True),
    ("decision", "hold_max_s"): ("Hold at most (s)", "The robot returns to ready this long after a result if no "
                                 "new round starts.", True),
    ("decision", "correction_s"): ("Correction window (s)", "Mediapipe may correct a decision this long after it. "
                                   "0 = off.", True),
    ("decision", "guided_early_s"): ("Throw accepted early (s)", "Beat guide: a throw this much before SHOOT still "
                                     "counts.", True),
    ("decision", "guided_late_s"): ("Throw accepted late (s)", "Beat guide: a throw this much after SHOOT still "
                                    "counts; later is a missed round.", True),
    ("decision", "guided_rock_after_s"): ("Rock after (s)", "Beat guide: a hand still closed this long after "
                                          "SHOOT is rock.", True),
    # robot (Bot tuning page)
    ("robot", "mode"): ("Robot", "Real robot over Wi-Fi, a simulated robot on this computer, or off.", False),
    ("robot", "protocol"): ("Robot commands", "Must match the robot's firmware. Team firmware: sends RPS:ROCK, "
                            "RPS:PAPER or RPS:SCISSORS once when the robot's move changes; it has no ready "
                            "position and does not reply. Reference firmware: numbered messages with replies.",
                            False),
    ("robot", "host"): ("Robot address", "IP address of the ESP32 (team robot: 192.168.0.126; the reference "
                        "firmware's own Wi-Fi: 192.168.4.1).", False),
    ("robot", "port"): ("Robot port", "UDP port. Must match the firmware (4210).", False),
    ("robot", "heartbeat_s"): ("Resend every (s)", "Reference firmware only: the current move is resent this "
                               "often.", True),
    ("robot", "ack_timeout_s"): ("Reply timeout (s)", "Reference firmware only: replies later than this are left "
                                 "out of the robot reply time.", True),
    ("robot", "finger_angles"): ("Finger angles", "Team firmware: the last angle sent to each servo channel on "
                                 "Bot tuning (0 = extended, 180 = folded).", True),
    # game (Play page)
    ("game", "rounds"): ("Rounds", "Rounds in a match; 0 = endless (until Stop).", False),
    ("game", "beat_bpm"): ("Beat tempo (per minute)", "Beats per minute of the beat guide. 150 = one beat every "
                           "0.4 s, a natural pump pace.", False),
    ("game", "sound"): ("Beat sound", "Drum, wood block or beep. All are made to be heard on laptop speakers; the "
                        "wood block and beep cut through a noisy room.", False),
    ("game", "beat_volume"): ("Steady beat volume", "The beat that plays all the time, 0-1.", False),
    ("game", "cue_volume"): ("Count volume", "The count (3, 2, 1) and SHOOT, 0-1. Keep it above the steady beat.",
                             False),
    ("game", "audio_latency_ms"): ("Sound delay (ms)", "Time from a beat being played to it being heard: about 40 "
                                   "ms on laptop speakers, 150-250 ms on Bluetooth. The throw window moves by this "
                                   "much so it matches what you hear.", False),
    ("game", "lead_beats"): ("Count-in (beats)", "Steady beats before the first round, to find the tempo.", False),
    ("game", "gap_beats"): ("Beats between rounds", "Steady beats while the result shows.", False),
    # timing model
    ("latency", "camera_latency_ms"): ("Camera delay (ms)", "Measured with the camera delay test on the Setup "
                                       "page.", True),
    ("latency", "network_ms"): ("Wi-Fi delay (ms)", "One-way delay to the robot: half the ping time, measured on "
                                "Bot tuning. 0 = not measured.", True),
    ("latency", "servo_transition_ms"): ("Robot move times (ms)", "Time for the hand to move between poses; used "
                                         "by Evaluate to estimate when the robot's move is visible.", True),
}

CHOICES = {
    ("camera", "backend"): [("msmf", "Media Foundation (Windows)"), ("dshow", "DirectShow (Windows)"),
                            ("v4l2", "V4L2 (Linux, Raspberry Pi)"), ("any", "Automatic")],
    ("camera", "fourcc"): [("YUY2", "YUY2 (uncompressed)"), ("MJPG", "MJPG (compressed)")],
    ("vote", "method"): [("sequence", "Same answer in a row"), ("majority", "Majority of recent answers")],
    ("decision", "mode"): [("countdown", "Countdown (pumps counted from your hand)"),
                           ("guided", "Beat guide (throw on SHOOT)"),
                           ("continuous", "Live (answers continuously)")],
    ("decision", "recognizer"): [("dextra_raw", "Dextra Raw"), ("dextra_tuned", "Dextra Tuned"),
                                 ("mediapipe", "Mediapipe"), ("both", "Both (Dextra Tuned + Mediapipe)")],
    ("decision", "idle_action"): [("ready", "Return to ready"), ("hold", "Keep last move")],
    ("decision", "robot_plays"): [("win", "To win (beats your throw)"), ("draw", "To draw (copies your throw)"),
                                  ("lose", "To lose (plays what your throw beats)")],
    ("cnn", "rotate"): [(0, "0°"), (90, "90°"), (180, "180°"), (270, "270°")],
    ("robot", "mode"): [("real", "Real robot"), ("simulated", "Simulated robot"), ("off", "Off")],
    ("robot", "protocol"): [("rps_text", "Team firmware (RPS:ROCK, RPS:PAPER, RPS:SCISSORS)"),
                            ("ack", "Reference firmware (numbered, with replies)")],
    ("game", "sound"): [("drum", "Drum"), ("wood", "Wood block"), ("beep", "Beep")],
}

SECTION_INFO = {
    "camera": ("Camera", "How the webcam is opened and exposed."),
    "roi": ("Play zone", "The square the app looks at."),
    "dvs": ("Dextra view", "How camera frames become the movement pictures Dextra reads."),
    "cnn": ("Dextra", "The model files and how Dextra Raw's image is turned."),
    "hand": ("Mediapipe", "Finger tracking."),
    "vote": ("Decision rules", "How many Dextra answers must agree before the robot moves."),
    "decision": ("Game rules", "Countdown, beat guide, pumps, throws and timing."),
    "robot": ("Robot", "Connection to the robot (Bot tuning page)."),
    "game": ("Match", "Rounds and the beat guide (Play page)."),
    "latency": ("Timing model", "Used on the Evaluate page to estimate when the robot's move is visible."),
}


def label_for(section: str, key: str) -> str:
    return FIELD_INFO.get((section, key), (key.replace("_", " ").capitalize(), "", True))[0]


def help_for(section: str, key: str) -> str:
    return FIELD_INFO.get((section, key), ("", "", True))[1]


def is_advanced(section: str, key: str) -> bool:
    return FIELD_INFO.get((section, key), ("", "", True))[2]
