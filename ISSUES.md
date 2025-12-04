# Known Issues and Workarounds

This file tracks recurring problems we hit while reproducing SafeMeta-RL and the fixes applied. Update it as new issues are found.

## MetaDrive rendering / plotting
- **Symptom:** Plotting trajectories with `scripts/plot_trajectories.py` on macOS caused window/pixel format errors (`Could not open window`, `No graphics pipe`, `Attempt to spawn multiple ShowBase instances`).
- **Fix:** Force offscreen Panda3D (`window-type offscreen`, `p3tinydisplay`, `require-window False`), disable `use_render` in plotting, and avoid passing `seed` to `reset`. Added robust fallback in `metadrive_wrapper.reset` to close the engine if needed.
- **Status:** Plotting runs headless; see `artifacts/figures/*.png`. If errors persist, rerun after closing other MetaDrive processes and ensure `PYOPENGL_PLATFORM=egl`.

- **Symptom (recurring):** Plotting still throws `Attempt to spawn multiple ShowBase instances` / `onscreen window` warnings on macOS.
- **Fix:** Added engine shutdown before env init; plotting now forces `window-type offscreen` with `p3tinydisplay`, disables seeds on reset, and closes MetaDrive engine on reset failure.
- **Status:** Still flaky on macOS; `plot_trajectories.py` now wraps env creation in try/finally and calls `close_engine()` before/after each env, but MetaDrive 0.2.x singleton can persist across runs. If it fails, restart the shell/Python process to clear the singleton and ensure no other MetaDrive instance is alive.

- **Update:** Added `scripts/plot_ppo_ramp.py` to draw PPO ramp trajectories without invoking Panda3D rendering (records positions only), to avoid the ShowBase singleton entirely.

**Current status:** Plotting may still hit the ShowBase singleton if another MetaDrive instance is alive. Before running plotting, close other runs, or restart the shell/Python process. `scripts/plot_trajectories.py` now calls `close_engine()` upfront, but MetaDrive 0.2.x is sensitive to multiple initializations on macOS. The wrapper now only sets supported headless keys (`use_render=False`, `offscreen_render=True`, `headless_machine_render=True`) to avoid config KeyErrors.

## Environment reset robustness
- **Symptom:** Persistent `Attempt to spawn multiple ShowBase instances` even after closing other terminals.
- **Fix:** `LSTCMetaDriveEnv` now closes the engine before instantiation, rebuilds the env on reset failure, and retries up to 3 times. Added `tests/test_env_reset.py` to cover reset/step/close twice in headless mode.
- **Status:** Should reduce singleton issues; if failures persist, fully restart the process to clear Panda3D/MetaDrive state.

## MetaDrive reset/seed compatibility (0.2.x)
- **Symptom:** `reset()` raised `TypeError: unexpected keyword argument 'seed'` or `ValueError: Current MetaDrive instance is broken` during eval/plotting.
- **Fix:** Wrapper now strips unsupported kwargs and retries; on failure closes the MetaDrive engine singleton before reset. Evaluation reuses the training env to avoid multiple engine instances.
- **Status:** Training can resume from checkpoints without crashing; plotting uses the same safeguards.

## numpy deprecations with MetaDrive 0.2.5.3
- **Symptom:** `np.int/np.float/np.bool` removal caused crashes.
- **Fix:** Added compatibility aliases in `metadrive_wrapper.py`.
- **Status:** Warnings may appear, but execution continues.

## Training halted early at eval
- **Symptom:** Training stopped around 200k steps due to eval env initialization errors.
- **Fix:** Evaluation now reuses the training environment and handles reset failures gracefully.
- **Status:** Resuming from checkpoints works; runs can proceed to `total_env_steps`.

## Reward scale vs. paper
- **Symptom:** PPO rewards around tens at 200k steps, success_rate=0; below paper’s curve.
- **Notes:** We have far fewer steps than paper (they use multi-million to 10M); traffic density may be higher than Table I; task is hard early on.
- **Plan:** Continue training to ≥3M steps; optionally lower `traffic_density` toward ~0.12 (≈12 p/km/ln) for a paper-like setup.
