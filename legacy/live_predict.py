"""
Real-Time Ultra-Low Latency Invincible Rock-Paper-Scissors System.
Pipeline:
  Webcam Frame -> Temporal Frame Differencing |F_t - F_{t-1}| ->
  Binary Threshold -> 64x64 Mask ->
  PyTorch RoshamboNet (<3ms CPU Inference) ->
  5-Frame Temporal Majority Filter ->
  Invincible Counter-Move Decision Engine
"""

import os
import sys
import time
import argparse
import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root: model.py
from model import RoshamboNet, MajorityVote, CLASS_NAMES, LABEL_TO_SYMBOL, COUNTER_MOVES
from pseudo_event_producer import PseudoEventProducer


class InvincibleRoshamboEngine:
    def __init__(
        self,
        model_path: str = "motion_model.pth",
        camera_index: int = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 60,
        threshold_value: int = 18,
        window_length: int = 5,
        mock_mode: bool = False,
    ):
        self.device = torch.device("cpu")  # Optimized CPU execution
        print(f"[InvincibleRoshamboEngine] Initializing on device: {self.device}")

        # 1. Load Model
        self.model = RoshamboNet(num_classes=len(CLASS_NAMES), pooling="avg")
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"[InvincibleRoshamboEngine] Loaded trained weights from: {model_path}")
        else:
            print(f"[InvincibleRoshamboEngine] Warning: Model weights '{model_path}' not found! Running with untrained initialization.")

        self.model.to(self.device)
        self.model.eval()

        # Warm up JIT/CPU cache
        dummy = torch.randn(1, 1, 64, 64, device=self.device)
        with torch.inference_mode():
            for _ in range(10):
                _ = self.model(dummy)

        # 2. Temporal Majority Voter (ported from dextra-roshambo-python consumer.py)
        self.voter = MajorityVote(window_length=window_length, num_classes=len(CLASS_NAMES))

        # 3. Pseudo-Event Motion Producer
        self.producer = PseudoEventProducer(
            camera_index=camera_index,
            width=width,
            height=height,
            fps=fps,
            threshold_value=threshold_value,
            mock_mode=mock_mode,
        )

        # Performance & Game State Tracking
        self.stats = {
            "inference_times_ms": [],
            "user_wins": 0,
            "ai_wins": 0,
            "ties": 0,
        }

    def predict_mask(self, mask_64: np.ndarray):
        """
        Runs RoshamboNet inference on a single 64x64 uint8 motion mask.
        Returns: (pred_label, confidence, inference_ms)
        """
        # Normalize uint8 [0, 255] to float32 tensor [0.0, 1.0] with shape (1, 1, 64, 64)
        tensor_in = torch.from_numpy(mask_64).float().div_(255.0).unsqueeze(0).unsqueeze(0).to(self.device)

        t_start = time.perf_counter()
        with torch.inference_mode():
            logits = self.model(tensor_in)
            probs = F.softmax(logits, dim=1)[0]
            pred_label = int(torch.argmax(probs).item())
            confidence = float(probs[pred_label].item())
        inference_ms = (time.perf_counter() - t_start) * 1000.0

        return pred_label, confidence, inference_ms

    def run(self, benchmark_frames: int = 0):
        """
        Main real-time loop. If benchmark_frames > 0, runs without GUI and prints latency report.
        """
        window_name = "Invincible Rock-Paper-Scissors (Webcam Pseudo-Events)"
        if benchmark_frames <= 0:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        print("\n========================================================")
        print("  INVINCIBLE ROCK-PAPER-SCISSORS SYSTEM ACTIVE          ")
        print("========================================================")
        print("  Controls:")
        print("    [Q] / ESC -> Exit")
        print("    [R]       -> Reset Win/Streak Counters")
        print("========================================================\n")

        fps_counter = 0
        fps_timer = time.time()
        current_fps = 0.0
        frame_idx = 0

        last_stable_label = 3
        last_announced_move = None
        consecutive_wins = 0

        try:
            while True:
                loop_start = time.perf_counter()
                ret, raw_frame, mask_64 = self.producer.get_motion_mask()
                if not ret or raw_frame is None:
                    time.sleep(0.005)
                    continue

                frame_idx += 1

                # 1. Real-time Inference (<3ms target)
                raw_label, confidence, inf_ms = self.predict_mask(mask_64)
                self.stats["inference_times_ms"].append(inf_ms)
                if len(self.stats["inference_times_ms"]) > 100:
                    self.stats["inference_times_ms"].pop(0)

                # 2. 5-Frame Majority Vote Filter
                voted_label = self.voter.new_prediction_and_vote(raw_label)
                if voted_label is None:
                    active_label = last_stable_label
                else:
                    active_label = voted_label
                    last_stable_label = voted_label

                # 3. Invincible Counter Decision
                counter_info = COUNTER_MOVES[active_label]
                user_symbol = LABEL_TO_SYMBOL[active_label]
                ai_symbol = counter_info["ai_symbol"]
                ai_icon = counter_info["icon"]
                match_desc = counter_info["desc"]

                if active_label != 3 and active_label != last_announced_move:
                    consecutive_wins += 1
                    self.stats["ai_wins"] += 1
                    last_announced_move = active_label

                # FPS calculation
                fps_counter += 1
                if time.time() - fps_timer >= 1.0:
                    current_fps = fps_counter / (time.time() - fps_timer)
                    fps_counter = 0
                    fps_timer = time.time()

                # If benchmarking mode, skip GUI rendering
                if benchmark_frames > 0 and frame_idx >= benchmark_frames:
                    print(f"[Benchmark] Completed {frame_idx} test frames.")
                    break

                if benchmark_frames > 0:
                    continue

                # 4. Render Rich HUD & Visualizer
                display = raw_frame.copy()
                dh, dw, _ = display.shape

                # Center ROI outline
                min_dim = min(dh, dw)
                sy = (dh - min_dim) // 2
                sx = (dw - min_dim) // 2
                cv2.rectangle(display, (sx, sy), (sx + min_dim, sy + min_dim), (80, 80, 80), 1)

                # PiP Preview: Scaled 64x64 Motion Mask
                mask_display = cv2.resize(mask_64, (150, 150), interpolation=cv2.INTER_NEAREST)
                mask_display_bgr = cv2.cvtColor(mask_display, cv2.COLOR_GRAY2BGR)
                # Cyan border around mask preview
                cv2.rectangle(mask_display_bgr, (0, 0), (149, 149), (255, 255, 0), 2)
                display[15:165, dw - 165 : dw - 15] = mask_display_bgr
                cv2.putText(display, "64x64 Motion Mask", (dw - 160, 32),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)

                # Top Banner: System Title & Latency
                avg_inf_ms = np.mean(self.stats["inference_times_ms"]) if self.stats["inference_times_ms"] else 0.0
                cv2.rectangle(display, (0, 0), (dw, 55), (20, 20, 20), -1)
                cv2.putText(display, "ULTRA-LOW LATENCY INVINCIBLE ROSHAMBO", (15, 25),
                            cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
                
                latency_color = (0, 255, 0) if avg_inf_ms < 3.0 else (0, 165, 255)
                cv2.putText(display, f"Inference: {avg_inf_ms:.2f}ms | FPS: {current_fps:.1f}", (15, 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, latency_color, 1)

                # Middle / Bottom Cards: Player vs AI Decision Card
                card_y = dh - 120
                cv2.rectangle(display, (15, card_y), (dw - 15, dh - 15), (15, 15, 15), -1)
                cv2.rectangle(display, (15, card_y), (dw - 15, dh - 15), (60, 60, 60), 1)

                # User Move
                user_text = f"YOU: {user_symbol.upper()} (conf: {confidence*100:.1f}%)"
                cv2.putText(display, user_text, (30, card_y + 35),
                            cv2.FONT_HERSHEY_DUPLEX, 0.65, (200, 200, 200), 1)

                # AI Invincible Counter Move
                if active_label != 3:
                    ai_text = f"AI COUNTER: {ai_symbol.upper()} -> {match_desc}"
                    cv2.putText(display, ai_text, (30, card_y + 70),
                                cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 255, 0), 2)
                    cv2.putText(display, f"INVINCIBLE STREAK: {consecutive_wins}", (30, card_y + 95),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                else:
                    cv2.putText(display, "AI: Waiting for hand motion...", (30, card_y + 70),
                                cv2.FONT_HERSHEY_DUPLEX, 0.65, (120, 120, 120), 1)

                cv2.imshow(window_name, display)
                key = cv2.waitKey(1) & 0xFF

                if key in [ord('q'), 27]:
                    break
                elif key == ord('r'):
                    consecutive_wins = 0
                    self.stats["ai_wins"] = 0
                    print("[InvincibleRoshamboEngine] Streak reset to 0.")

        finally:
            self.producer.release()
            cv2.destroyAllWindows()
            self._print_summary()

    def _print_summary(self):
        times = self.stats["inference_times_ms"]
        if times:
            print("\n================ LATENCY & PERFORMANCE SUMMARY ================")
            print(f"  Mean Inference Latency:   {np.mean(times):.2f} ms")
            print(f"  Median Inference Latency: {np.median(times):.2f} ms")
            print(f"  95th Percentile Latency:  {np.percentile(times, 95):.2f} ms")
            print(f"  Min / Max Latency:        {np.min(times):.2f} ms / {np.max(times):.2f} ms")
            print(f"  Sub-3ms Target Met:       {'YES [PASS]' if np.mean(times) < 3.0 else 'NO'}")
            print(f"  Total Invincible Counter Moves: {self.stats['ai_wins']}")
            print("===============================================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-Time Invincible Rock-Paper-Scissors")
    parser.add_argument("--model", type=str, default="motion_model.pth", help="Path to motion_model.pth")
    parser.add_argument("--camera", type=int, default=0, help="Webcam device index")
    parser.add_argument("--width", type=int, default=640, help="Webcam width")
    parser.add_argument("--height", type=int, default=480, help="Webcam height")
    parser.add_argument("--fps", type=int, default=60, help="Webcam FPS")
    parser.add_argument("--threshold", type=int, default=18, help="Motion differencing threshold (15-20 recommended)")
    parser.add_argument("--window_length", type=int, default=5, help="Majority vote temporal window size")
    parser.add_argument("--mock", action="store_true", help="Simulate camera frames for testing")
    parser.add_argument("--benchmark", type=int, default=0, help="Run headless benchmark for N frames")
    args = parser.parse_args()

    engine = InvincibleRoshamboEngine(
        model_path=args.model,
        camera_index=args.camera,
        width=args.width,
        height=args.height,
        fps=args.fps,
        threshold_value=args.threshold,
        window_length=args.window_length,
        mock_mode=args.mock,
    )
    engine.run(benchmark_frames=args.benchmark)
