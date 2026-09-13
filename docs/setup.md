# Pi environment setup

Target: `pacel@pacel-rbp01`, Raspberry Pi 5 (4GB), Debian 13 (trixie),
Python 3.13.5. `picamera2` and OpenCV come from apt, not pip — the system is
[PEP 668 externally-managed](https://peps.python.org/pep-0668/)
(`/usr/lib/python3.13/EXTERNALLY-MANAGED` exists), so a plain `pip install`
into the system Python is blocked, and a *clean* venv can't see the
apt-installed packages either. The fix is a venv built with
`--system-site-packages`.

## One-time setup

```bash
ssh pacel@pacel-rbp01
sudo apt install -y python3-opencv      # cv2 4.10 into dist-packages
                                          # (python3-picamera2 is already present)

cd ~/misc/greenpro
python3 -m venv --system-site-packages .venv
.venv/bin/pip install onnxruntime pyyaml

# sanity check -- should print nothing and exit 0
.venv/bin/python -c "import picamera2, cv2, onnxruntime, numpy, yaml"
```

## Syncing code from the dev machine

Developed on Windows, run on the Pi. From the repo root:

```bash
rsync -av --delete --exclude .venv --exclude models --exclude .git \
  ./ pacel@pacel-rbp01:~/misc/greenpro/
```

(If `rsync` isn't available on Windows, `scp -r` works but won't prune
deleted files.)

## Running

```bash
ssh pacel@pacel-rbp01
cd ~/misc/greenpro
.venv/bin/python run.py
# from another machine on the same network:
#   http://pacel-rbp01:8000/
```

Camera-free (geometry/shading/server work, useful for iterating without
needing the Pi at all):

```bash
python run.py --source image:picam_snap.jpg   # any local Python 3 + requirements.txt
```

## Model file

`segmenter.kind: neural` or `combo` needs an ONNX model at
`models/selfie_segmentation.onnx` (path configurable via
`segmenter.neural.model_path`). Not yet sourced — see
[status.md](status.md). `models/` is gitignored; copy the file onto the Pi
directly rather than committing it.
