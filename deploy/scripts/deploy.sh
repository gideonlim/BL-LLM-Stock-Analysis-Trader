#!/bin/bash
# deploy.sh — pull latest code and restart the day-trader daemon.
#
# Run from the droplet as the deploy user:
#   ssh deploy@170.64.197.147
#   sudo /opt/day-trader/repo/deploy/scripts/deploy.sh
#
# Or invoke remotely:
#   ssh deploy@170.64.197.147 'sudo /opt/day-trader/repo/deploy/scripts/deploy.sh'
#
# What this does:
#   1. git pull (as daytrader user)
#   2. pip install -r requirements-daytrade-live.txt (in the venv)
#   3. systemctl restart day-trader
#   4. Enable the EOD-flatten timer if not already

set -euo pipefail

REPO=/opt/day-trader/repo
VENV=$REPO/.venv

echo "=== Day-trader deploy @ $(date) ==="

echo "[1/4] git pull"
sudo -u daytrader bash -c "cd $REPO && git pull --ff-only"

echo "[2/4] pip install"
sudo -u daytrader bash -c "$VENV/bin/pip install -q -r $REPO/requirements-daytrade-live.txt"

echo "[3/4] restart day-trader"
systemctl restart day-trader
systemctl is-active --quiet day-trader && echo "  day-trader: active" || echo "  day-trader: FAILED"

echo "[4/4] enable EOD-flatten timer"
# Copy systemd units if they've changed
cp -f $REPO/deploy/systemd/day-trader.service /etc/systemd/system/
cp -f $REPO/deploy/systemd/day-trader-eod-flatten.service /etc/systemd/system/
cp -f $REPO/deploy/systemd/day-trader-eod-flatten.timer /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now day-trader-eod-flatten.timer
systemctl is-active --quiet day-trader-eod-flatten.timer && echo "  eod-flatten timer: active" || echo "  eod-flatten timer: FAILED"

echo "=== Deploy complete ==="
