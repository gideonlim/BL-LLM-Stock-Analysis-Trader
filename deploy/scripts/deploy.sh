#!/bin/bash
# deploy.sh — pull latest code, restart daemon, enable timers.
#
# Run from the droplet:
#   ssh deploy@170.64.197.147
#   sudo /opt/day-trader/repo/deploy/scripts/deploy.sh
#
# Or invoke remotely (one line):
#   ssh deploy@170.64.197.147 'sudo /opt/day-trader/repo/deploy/scripts/deploy.sh'
#
# Idempotent — safe to run repeatedly. Each step is a no-op if
# already in the target state.

set -euo pipefail

# ── Pre-flight ───────────────────────────────────────────────────

if [[ $EUID -ne 0 ]]; then
  echo "ERROR: deploy.sh must run as root (use sudo)" >&2
  exit 1
fi

REPO=/opt/day-trader/repo
VENV=$REPO/.venv
SYSTEMD=/etc/systemd/system

# Verify repo exists and is owned by daytrader
if [[ ! -d $REPO/.git ]]; then
  echo "ERROR: repo not found at $REPO — did you clone it as daytrader?" >&2
  exit 1
fi

# Ensure the runtime state dir exists for heartbeat writes
mkdir -p /var/run/day-trader
chown daytrader:daytrader /var/run/day-trader
chmod 750 /var/run/day-trader

echo "=== Day-trader deploy @ $(date) ==="
echo "  repo: $REPO"
echo "  venv: $VENV"

# ── 1. Pull latest code ──────────────────────────────────────────

echo "[1/5] git pull"
sudo -u daytrader bash -c "cd $REPO && git pull --ff-only"

# ── 2. Install runtime deps ──────────────────────────────────────

echo "[2/5] pip install runtime deps"
if [[ ! -d $VENV ]]; then
  echo "  creating venv..."
  sudo -u daytrader bash -c "cd $REPO && python3.12 -m venv $VENV"
  sudo -u daytrader bash -c "$VENV/bin/pip install -q --upgrade pip wheel"
fi
sudo -u daytrader bash -c "$VENV/bin/pip install -q -r $REPO/requirements-daytrade-live.txt"

# ── 3. Install / refresh systemd units ───────────────────────────

echo "[3/5] systemd units"
units=(
  day-trader.service
  day-trader-eod-flatten.service
  day-trader-eod-flatten.timer
  day-trader-watchdog.service
  day-trader-watchdog.timer
)
for unit in "${units[@]}"; do
  src=$REPO/deploy/systemd/$unit
  dst=$SYSTEMD/$unit
  if [[ ! -f $src ]]; then
    echo "  WARN: $src missing, skipping"
    continue
  fi
  if ! cmp -s "$src" "$dst" 2>/dev/null; then
    cp -f "$src" "$dst"
    echo "  updated: $unit"
  fi
done
systemctl daemon-reload

# ── 4. Restart main daemon ───────────────────────────────────────

echo "[4/5] restart day-trader"
systemctl restart day-trader
sleep 2  # give it a beat to either start or fail
if systemctl is-active --quiet day-trader; then
  echo "  day-trader: active"
else
  echo "  day-trader: FAILED — check 'journalctl -u day-trader -n 50'"
  exit 1
fi

# ── 5. Enable timers ─────────────────────────────────────────────

echo "[5/5] enable timers"
for timer in day-trader-eod-flatten.timer day-trader-watchdog.timer; do
  systemctl enable --now "$timer" >/dev/null 2>&1 || true
  if systemctl is-active --quiet "$timer"; then
    echo "  $timer: active"
  else
    echo "  $timer: NOT ACTIVE — check 'systemctl status $timer'"
  fi
done

echo "=== Deploy complete ==="
echo
echo "Useful follow-ups:"
echo "  journalctl -u day-trader -f       # live daemon logs"
echo "  systemctl status day-trader       # service health"
echo "  systemctl list-timers              # next timer fire times"
