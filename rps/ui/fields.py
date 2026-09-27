"""
Plain-language name, hover explanation and level (basic / advanced) for every config field, plus
display names for every choice. Terms used across the app:
  motion model  = the CNN that reads movement (Dextra's method)
  hand tracker  = MediaPipe finger positions
  motion image  = the 64 x 64 movement picture the motion model reads
"""

# (section, key) -> (label, hover explanation, advanced?)
FIELD_INFO = {
    # camera
    ("camera", "index"): ("Camera number", "0 = built-in webcam.", True),
    ("camera", "backend"): ("Capture driver", "Media Foundation: best in dim light. DirectShow: steady 30 fps with "
                            "a locked exposure. Auto-configure chooses.", True),
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
    # motion image
    ("dvs", "sensor_size"): ("Motion grid", "The play zone is resized to this many pixels before movement is "
                             "measured (Dextra: 128).", True),
    ("dvs", "frame_size"): ("Motion image size", "Pixels per side of the image the motion model reads.", True),
    ("dvs", "log_offset"): ("Dark noise damping", "Higher = less false movement from dark, noisy pixels.", True),
    ("dvs", "contrast_threshold"): ("Motion sensitivity", "Brightness change counted as movement. Lower = more "
                                    "sensitive, more noise. Default 0.20.", False),
    ("dvs", "event_count"): ("Movement per image", "How much movement is collected into one motion image. "
                             "Dextra used 1500.", False),
    ("dvs", "clip_count"): ("Per-pixel cap", "Limit on movement counted in one pixel (Dextra: 16).", True),
    ("dvs", "noise_filter"): ("Noise filter", "Ignore single isolated moving pixels.", True),
    ("dvs", "global_reset_fraction"): ("Lighting change limit", "If more than this share of the picture changes at "
                                       "once, treat it as a lighting change, not movement.", True),
    ("dvs", "flush_min_fraction"): ("Partial image", "When the hand stops, still send a motion image if it has at "
                                    "least this share of the usual movement.", True),
    ("dvs", "still_events_per_frame"): ("Still threshold", "Movement per camera frame below which the scene counts "
                                        "as still.", True),
    ("dvs", "flush_still_frames"): ("Still frames before sending", "", True),
    ("dvs", "max_accumulation_s"): ("Max collection time (s)", "Discard a motion image that takes longer than this "
                                    "to fill.", True),
    # motion model
    ("cnn", "model_path"): ("Model file", "The motion model file (.pth).", False),
    ("cnn", "threads"): ("CPU threads", "1 is fastest for this small model.", True),
    ("cnn", "rotate"): ("Rotate motion image", "Turn the motion image to match the camera angle the model was "
                        "trained on. Dextra's camera saw the hand from the side, fingers pointing left.", False),
    ("cnn", "flip"): ("Mirror motion image", "Mirror the motion image left to right before the model reads it.",
                      False),
    # hand tracker
    ("hand", "model_path"): ("Tracker file", "Hand tracking model file (hand_landmarker.task).", True),
    ("hand", "enabled"): ("Use hand tracker", "Allow the hand tracker to run.", True),
    ("hand", "min_detection_confidence"): ("Find-hand confidence", "0-1. Higher = fewer false hands.", True),
    ("hand", "min_presence_confidence"): ("Hand-present confidence", "0-1.", True),
    ("hand", "min_tracking_confidence"): ("Keep-tracking confidence", "0-1.", True),
    ("hand", "crop_margin"): ("Search margin", "The tracker also looks this far outside the play zone.", True),
    ("hand", "extend_threshold"): ("Finger straight below", "Finger bend under this counts as straight.", True),
    ("hand", "curl_threshold"): ("Finger bent above", "Finger bend over this counts as bent.", True),
    ("hand", "frame_budget_ms"): ("Time budget (ms)", "Skip one frame when the tracker takes longer than this.",
                                  True),
    # decision rules
    ("vote", "method"): ("Agreement rule", "How repeated answers from the motion model are combined.", True),
    ("vote", "k"): ("Answers in a row", "The motion model must give the same answer this many times before the "
                    "robot moves. 2 = fast, 3 = safer.", False),
    ("vote", "min_confidence"): ("Minimum confidence", "Answers below this confidence are ignored (0-1).", False),
    ("vote", "max_gap_s"): ("Max gap (s)", "Answers further apart than this do not count as in a row.", True),
    ("vote", "majority_window"): ("Majority window", "Recent answers considered by the majority rule.", True),
    # game
    ("decision", "mode"): ("Game mode", "Countdown: pump 3 times, then throw. Live: the robot answers "
                           "continuously.", False),
    ("decision", "source"): ("Recognition", "What reads your hand.", False),
    ("decision", "active_events_per_frame"): ("Moving above", "Movement per camera frame that counts as the hand "
                                              "moving.", True),
    ("decision", "still_events_per_frame"): ("Still below", "Movement per camera frame that counts as still.", True),
    ("decision", "still_frames"): ("Still frames", "Frames without movement before the hand counts as stopped.",
                                   False),
    ("decision", "mp_stable_frames"): ("Tracker frames in a row", "The hand tracker must repeat a gesture this "
                                       "many frames.", True),
    ("decision", "mp_min_confidence"): ("Tracker minimum confidence", "0-1.", True),
    ("decision", "mp_still_events_in_hand"): ("Tracker stillness limit", "The tracker decides only when the hand "
                                              "moves less than this.", True),
    ("decision", "mp_after_cnn_commit_s"): ("Tracker wait (s)", "Wait after a motion-model decision before the "
                                            "tracker may change it.", True),
    ("decision", "switch_dead_time_s"): ("Change cooldown (s)", "Minimum time between tracker-driven changes.",
                                         True),
    ("decision", "idle_timeout_s"): ("Idle after (s)", "No hand for this long = idle.", True),
    ("decision", "idle_action"): ("When idle", "What the robot does when nobody is playing.", True),
    ("decision", "pumps_before_shoot"): ("Pumps before the throw", "Countdown pumps before the throw.", False),
    ("decision", "pump_source"): ("Pump tracking", "How pumps are counted.", True),
    ("decision", "pump_miss_tolerance"): ("Missed pumps allowed", "A throw that lands on the beat still counts "
                                          "if this many pumps were missed.", True),
    ("decision", "pump_min_amplitude"): ("Smallest pump", "Minimum up-down movement, as a share of the play zone "
                                         "height, until the player's own size is learned.", True),
    ("decision", "pump_min_period_s"): ("Fastest pump (s)", "Pumps closer together than this are ignored until the "
                                        "player's tempo is learned.", True),
    ("decision", "shoot_window_s"): ("Throw window (s)", "Minimum time allowed for the throw; grows with a slow "
                                     "tempo.", True),
    ("decision", "rock_min_shoot_s"): ("Rock earliest (s)", "", True),
    ("decision", "rock_settle_fallback_s"): ("Rock fallback (s)", "Decide rock after this long if the landing was "
                                             "not seen.", True),
    ("decision", "hold_min_s"): ("Hold at least (s)", "The robot keeps its move at least this long.", True),
    ("decision", "hold_max_s"): ("Hold at most (s)", "", True),
    ("decision", "correction_s"): ("Correction window (s)", "The tracker may correct a decision this long after it. "
                                   "0 = off.", True),
    # robot
    ("robot", "enabled"): ("Robot link", "Send moves to the robot hand.", True),
    ("robot", "host"): ("Robot address", "IP address of the ESP32. Its own Wi-Fi (RPS-HAND) uses 192.168.4.1.",
                        False),
    ("robot", "port"): ("Robot port", "UDP port. Must match the firmware (4210).", False),
    ("robot", "heartbeat_s"): ("Resend every (s)", "The current move is resent this often.", True),
    ("robot", "ack_timeout_s"): ("Reply timeout (s)", "", True),
    # timing model
    ("latency", "camera_latency_ms"): ("Camera delay (ms)", "Measured with the camera delay test on the Setup "
                                       "page.", True),
    ("latency", "servo_transition_ms"): ("Servo move times (ms)", "Time for the hand to move between poses.", True),
}

CHOICES = {
    ("camera", "backend"): [("msmf", "Media Foundation"), ("dshow", "DirectShow"), ("any", "Automatic")],
    ("camera", "fourcc"): [("YUY2", "YUY2 (uncompressed)"), ("MJPG", "MJPG (compressed)")],
    ("vote", "method"): [("sequence", "Same answer in a row"), ("majority", "Majority of recent answers")],
    ("decision", "mode"): [("countdown", "Countdown (3 pumps, then throw)"),
                           ("continuous", "Live (answers continuously)")],
    ("decision", "source"): [("fused", "Motion model + hand tracker"), ("cnn", "Motion model"),
                             ("mediapipe", "Hand tracker")],
    ("decision", "idle_action"): [("ready", "Return to ready"), ("hold", "Keep last move")],
    ("decision", "pump_source"): [("flow", "Movement in play zone"), ("mp", "Tracked wrist")],
    ("cnn", "rotate"): [(0, "0°"), (90, "90°"), (180, "180°"), (270, "270°")],
}

SECTION_INFO = {
    "camera": ("Camera", "How the webcam is opened and exposed."),
    "roi": ("Play zone", "The square the app looks at."),
    "dvs": ("Motion image", "How camera frames become the movement pictures the motion model reads."),
    "cnn": ("Motion model", "The model that reads movement."),
    "hand": ("Hand tracker", "Finger tracking, used when the hand is still."),
    "vote": ("Decision rules", "How many answers must agree before the robot moves."),
    "decision": ("Game", "Countdown, pumps, throws and timing."),
    "robot": ("Robot hand", "Network connection to the ESP32."),
    "latency": ("Timing model", "Used on the Evaluate page to estimate when the robot's move is visible."),
}


def label_for(section: str, key: str) -> str:
    return FIELD_INFO.get((section, key), (key.replace("_", " ").capitalize(), "", True))[0]


def help_for(section: str, key: str) -> str:
    return FIELD_INFO.get((section, key), ("", "", True))[1]


def is_advanced(section: str, key: str) -> bool:
    return FIELD_INFO.get((section, key), ("", "", True))[2]
