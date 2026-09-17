"""Build H3-grid benchmark crops (17n+5 frames) from the 124-frame dataset clips.

Usage:
  python dataset/make_107_crops.py --frames 107 --n 50   # benchmark set
  python dataset/make_107_crops.py --frames 39  --n 5    # smoke test
  python dataset/make_107_crops.py --frames 73  --n 10   # intermediate check

Crops the FIRST N frames of each clip together with its audio, so video and audio stay in sync.
Output: data/benchmark-<frames>/ with <id>.mp4 (H.264, synced audio), <id>.wav, <id>.txt caption.
"""
import argparse, json, subprocess, wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data' / 'alora-kannada-fleurs-v1'
FPS, SR = 24, 48000
SCENE_WORDS = 1.625, 3.042, 4.458, 5.167


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frames', type=int, default=107, choices=[39, 73, 107, 124])
    ap.add_argument('--n', type=int, default=50)
    ap.add_argument('--split', default='train')
    a = ap.parse_args()
    if a.frames == 124:
        raise SystemExit('124 = training clips themselves; nothing to crop.')
    nsamp = round(a.frames / FPS * SR)
    out = ROOT / 'data' / ('benchmark-%d' % a.frames)
    out.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(x) for x in (DATA / (a.split + '.jsonl')).read_text().splitlines()][:a.n]
    import shutil, sys
    ff = shutil.which('ffmpeg')
    if not ff:
        try:
            import imageio_ffmpeg
            ff = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            r = subprocess.run([sys.executable, '-c', 'import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())'],
                               capture_output=True, text=True)
            ff = r.stdout.strip()
    if not ff:
        raise SystemExit('ffmpeg not found; install ffmpeg or imageio_ffmpeg')
    made = 0
    for r in rows:
        src = DATA / r['collection']
        vp, wp, cap = src / Path(r['video_file']).name, src / Path(r['audio_file']).name, src / (r['id'] + '.txt')
        ov, ow = out / (r['id'] + '.mp4'), out / (r['id'] + '.wav')
        with wave.open(str(wp)) as w:
            pcm = w.readframes(nsamp)
        with wave.open(str(ow), 'wb') as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR); w.writeframes(pcm)
        rc = subprocess.run([ff, '-y', '-v', 'error', '-i', str(vp), '-i', str(ow),
                             '-map', '0:v:0', '-map', '1:a:0', '-frames:v', str(a.frames),
                             '-c:v', 'libx264', '-preset', 'fast', '-crf', '20', '-pix_fmt', 'yuv420p',
                             '-c:a', 'aac', '-ar', str(SR), '-ac', '1', '-movflags', '+faststart', str(ov)],
                            capture_output=True)
        if rc.returncode == 0 and ov.exists() and ow.exists():
            (out / (r['id'] + '.txt')).write_text(cap.read_text() if cap.exists() else (r.get('caption') or ''))
            made += 1
    print('made', made, 'clips at', a.frames, 'frames ->', out)


if __name__ == '__main__':
    main()