#!/usr/bin/env bash
# Puts the WRC pipeline next to this repo and adds our DM-arm files to it.
#   bash wrc_dm_pipeline/setup_wrc.sh            -> <parent of this repo>/wrc
#   WRC_DIR=/somewhere/wrc bash wrc_dm_pipeline/setup_wrc.sh
# Safe to re-run: it only clones when WRC is missing and re-copies our files.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEAM_REPO="$(cd "$HERE/.." && pwd)"
WRC_DIR="${WRC_DIR:-$TEAM_REPO/../wrc}"
WRC_URL="https://github.com/TheMoonAstronaut/wrc.git"
WRC_COMMIT="723121f"       # the WRC version these files were built and tested against

if [ ! -d "$WRC_DIR/.git" ]; then
  echo ">> Cloning WRC into $WRC_DIR (about 480 MB, takes a few minutes)"
  git clone "$WRC_URL" "$WRC_DIR"
fi
WRC_DIR="$(cd "$WRC_DIR" && pwd)"

if [ "$(git -C "$WRC_DIR" rev-parse --short=7 HEAD)" != "$WRC_COMMIT" ]; then
  echo ">> Switching WRC to commit $WRC_COMMIT"
  git -C "$WRC_DIR" checkout -q "$WRC_COMMIT" || {
    echo "Could not switch WRC to $WRC_COMMIT (local changes?). Fix that in $WRC_DIR and re-run."; exit 1; }
fi

echo ">> Copying our DM-arm files into $WRC_DIR"
cp -R "$HERE/overlay/." "$WRC_DIR/"
chmod +x "$WRC_DIR/START_WRC.sh" "$WRC_DIR/local_dm/install.sh" "$WRC_DIR"/local_dm/*.py

echo
echo "Done. Next:"
echo "  cd $WRC_DIR"
echo "  PYTHONNOUSERSITE=1 conda env create -f environment.yml    # once"
echo "  bash local_dm/install.sh"
if [ "$(cd "$WRC_DIR/.." && pwd)/steel_team" != "$TEAM_REPO" ]; then
  echo
  echo "This repo is not at $(cd "$WRC_DIR/.." && pwd)/steel_team, so tell the scripts where teleoperation is:"
  echo "  export STEEL_TEAM_TELEOP_DIR=$TEAM_REPO/teleoperation"
fi
