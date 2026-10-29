# Fix output video

Video keluaran sekarang adalah `tracking.avi` (MJPG), bukan `tracking.mp4`.

Contoh (jalankan dari folder proyek):

```powershell
python main.py track --device cpu --max-frames 300 --confidence 0.05 --imgsz 960 --tracker config/bytetrack_badminton.yaml --crop-mode court --output outputs/tracking_crop
```

Hasil: `outputs/tracking_crop/tracking.avi`, `tracks.csv`, `tracking_report.json`.
Program memverifikasi video keluaran setelah writer ditutup. Untuk mendapatkan MP4 yang kompatibel gunakan FFmpeg:

```powershell
ffmpeg -i outputs/tracking_crop/tracking.avi -c:v libx264 -pix_fmt yuv420p -movflags +faststart outputs/tracking_crop/tracking_fixed.mp4
```

Perhatikan nilai `crop_bounds_xyxy` di laporan. Jika nilainya `[0,0,476,848]`, crop tidak aktif; periksa parameter `--crop-mode court`.
