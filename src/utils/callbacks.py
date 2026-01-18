import os
import time
import json
import threading
import subprocess
from collections import deque
from transformers import TrainerCallback

class KeyboardPauseCallback(TrainerCallback):
    """
    Control training via flag files in a specified directory.
    - pause.flag: Save checkpoint and exit (can be resumed later).
    - next.flag: Stop training immediately (skip to next task).
    - status.json: Writes current training status/metrics for monitoring.
    """
    
    def __init__(self, flag_dir: str, output_dir: str, trainer_ref=None, status_update_interval: float = 2.0, eta_history: int = 8):
        self.flag_dir = flag_dir
        self.output_dir = output_dir
        os.makedirs(self.flag_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.pause_file = os.path.join(flag_dir, "pause.flag")
        self.next_file = os.path.join(flag_dir, "next.flag")
        self.status_file = os.path.join(self.output_dir, "status.json")
        
        self.trainer_ref = trainer_ref
        self.is_paused = os.path.exists(self.pause_file)
        
        # ETA Calculation
        self.eta_history = eta_history
        self.step_times = deque(maxlen=eta_history)
        
        # Background Status Writer
        self._stop_background = False
        self.status_update_interval = status_update_interval
        self._bg_thread = threading.Thread(target=self._background_worker, daemon=True)
        self._bg_thread_started = False
        
        print(f"🎹 [Control] Initialized. Monitoring '{self.flag_dir}'.")

    def _background_worker(self):
        while not self._stop_background:
            try:
                self._write_status()
            except Exception:
                pass
            time.sleep(self.status_update_interval)

    # --- Hooks ---
    def on_train_begin(self, args, state, control, **kwargs):
        if not self._bg_thread_started:
            self._bg_thread.start()
            self._bg_thread_started = True
        self._write_status(state=state, args=args)

    def on_step_end(self, args, state, control, **kwargs):
        # 1. Update ETA metrics
        try:
            now = time.time()
            step = getattr(state, "global_step", None)
            if step is not None:
                if not self.step_times or self.step_times[-1][0] != step:
                    self.step_times.append((step, now))
        except Exception:
            pass

        # 2. Check PAUSE Flag (Save & Stop)
        if os.path.exists(self.pause_file):
            if not self.is_paused:
                print(f"\n💾 [Control] Pause requested. Saving checkpoint at step {state.global_step} and stopping...")
                self.is_paused = True
                control.should_save = True
                control.should_training_stop = True
        else:
            self.is_paused = False

        # 3. Check NEXT Flag (Stop immediately)
        if os.path.exists(self.next_file):
            print(f"\n⏭️  [Control] Next requested. Stopping training at step {state.global_step}.")
            try:
                os.remove(self.next_file)
            except:
                pass
            control.should_training_stop = True
            
        return control

    def on_train_end(self, args, state, control, **kwargs):
        self._write_status(state=state, args=args)
        self._stop_background = True

    # --- Helpers ---
    def _calc_eta_seconds(self):
        if len(self.step_times) < 2: return None
        first_step, first_time = self.step_times[0]
        last_step, last_time = self.step_times[-1]
        delta_steps = last_step - first_step
        delta_time = last_time - first_time
        if delta_steps <= 0 or delta_time <= 1e-4: return None
        
        sec_per_step = delta_time / delta_steps
        
        total_steps = None
        if self.trainer_ref:
            state = getattr(self.trainer_ref, "state", None) or self.trainer_ref.state
            total_steps = state.max_steps
            
        if total_steps is None: return None
        
        current_step = self.step_times[-1][0]
        return max(0, total_steps - current_step) * sec_per_step

    def _write_status(self, state=None, args=None):
        try:
            # Gather State
            if state is None and self.trainer_ref: state = self.trainer_ref.state
            gs = getattr(state, "global_step", 0) if state else 0
            
            # Temps
            try:
                gpu_t = int(subprocess.check_output(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"], stderr=subprocess.DEVNULL).decode().strip())
            except:
                gpu_t = -1
                
            # ETA
            eta_secs = self._calc_eta_seconds()
            if eta_secs:
                m, s = divmod(int(eta_secs), 60)
                h, m = divmod(m, 60)
                eta_str = f"{h}h{m:02d}m{s:02d}s"
            else:
                eta_str = "Calculating..."

            status = "training"
            if os.path.exists(self.pause_file): status = "paused"
            
            payload = {
                "step": gs,
                "status": status,
                "gpu_temp": gpu_t,
                "eta": eta_str,
                "timestamp": int(time.time())
            }
            
            tmp = self.status_file + ".tmp"
            with open(tmp, "w") as f:
                json.dump(payload, f)
            os.replace(tmp, self.status_file)
            
        except Exception:
            pass
