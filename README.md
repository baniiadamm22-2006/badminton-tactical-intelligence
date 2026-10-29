# BTI Tahap 07.1 — TrackNetV3 Offline Inference

## Batasan
Adapter ini menjalankan kode upstream TrackNetV3 dalam environment Python terpisah; tidak menyertakan checkpoint dan **belum membuktikan akurasi pada video Anda**. Integrasi tidak memodifikasi YOLO atau ByteTrack BTI.

## 1. Copy file
- `tracknet_bridge.py` -> root project BTI
- `tests/test_tracknet_bridge.py` -> root project BTI `tests/`
- Pastikan `src/shuttle_tool.py` dari Tahap 07 sudah tersedia.

## 2. Clone TrackNetV3
```powershell
git clone https://github.com/qaz812345/TrackNetV3.git external/TrackNetV3
```
Repo upstream: https://github.com/qaz812345/TrackNetV3

## 3. Environment independen
Repository original menyebut Python 3.8.7 dan torch 1.10.0, sedangkan pinned requirements memiliki versi lama. **Jangan jalankan pip install di .venv utama BTI.**

Contoh dengan Miniconda (bila tersedia):
```powershell
conda create -n bti-tracknet python=3.8 -y
conda activate bti-tracknet
python -m pip install -r external/TrackNetV3/requirements.txt
python -m pip install tqdm
python -c "import torch, cv2, pandas, tqdm; print(torch.__version__)"
```
Jika requirements gagal, berhenti dan kirim error pip; jangan mengganti versi sembarangan karena kompatibilitas torch/OpenCV/Python dan perangkat bervariasi.

## 4. Checkpoints
Unduh checkpoint dari README upstream (tautan 'checkpoints' pada bagian Inference), lalu ekstrak ke:
- `external/TrackNetV3/ckpts/TrackNet_best.pt`
- `external/TrackNetV3/ckpts/InpaintNet_best.pt`

Cek nama file sesungguhnya dan gunakan lokasi sesuai perangkat Anda. Jangan unduh executable atau checkpoint dari sumber tidak dikenal.

## 5. Jalankan dari environment BTI (PowerShell)
```powershell
conda activate bti-tracknet
where.exe python
```
Catat lokasi `python.exe` dari environment `bti-tracknet`. Lalu aktifkan kembali `.venv` BTI untuk menjalankan adapter; contoh:

```powershell
conda deactivate
.\.venv\Scripts\Activate.ps1
python -m unittest tests.test_tracknet_bridge -v
python tracknet_bridge.py --python "C:\PATH\TO\bti-tracknet\python.exe" --repo external/TrackNetV3 --video data/raw/match_001.mp4 --tracknet external/TrackNetV3/ckpts/TrackNet_best.pt --inpaintnet external/TrackNetV3/ckpts/InpaintNet_best.pt --ground-truth data/annotations/shuttle_ground_truth.csv --output outputs/shuttle_tracknet --batch-size 2
```
Ganti placeholder `C:\PATH\TO\bti-tracknet\python.exe` dengan path nyata dari langkah `where.exe python`.

Untuk file panjang, tambahkan `--large-video`. Saat pertama menjalankan coba `--batch-size 2` untuk menghemat memori. Kode upstream dapat tetap bermasalah pada kombinasi tertentu, terutama CPU/Windows; simpan traceback lengkap untuk diagnosis.

## 6. Output
```text
outputs/shuttle_tracknet/
  match_001_ball.csv     # prediksi upstream
  predictions.csv        # format BTI
  evaluation.json        # hasil evaluasi jika ground truth tersedia
  run_summary.json       # metadata eksekusi
```

Evaluasi dilakukan hanya pada frame yang dianotasi manual. Angka model dalam paper tidak boleh dianggap sebagai performa pada video BTI Anda.

## Catatan cut
TrackNetV3 dapat memprediksi sambil melewati scene cut bila diberikan video penuh; lintasan dan metrik temporal harus dipisahkan menggunakan `scene_cuts.json` pada tahap validasi berikutnya. Karena itu **jangan menyimpulkan trajectory kontinu** dari prediksi pada batas cut.
