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
