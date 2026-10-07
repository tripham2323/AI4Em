# 17 — Chiến lược kiểm thử

Unit tests dùng `pytest`, fixture tổng hợp deterministic, không phụ thuộc webcam/GPU/download. Integration và hardware smoke tách marker; CI không cố mở camera. **Tests phải kiểm tra hành vi hoặc số đúng**, không chỉ “không exception” hay shape không rỗng.

## Test matrix bắt buộc
| Thành phần | Case và kết quả mong đợi |
|---|---|
| EAR | p1=(0,0), p4=(4,0), hai khoảng dọc bằng 1 → EAR=0.25; phóng to/dịch chuyển không đổi kết quả; mẫu số 0 hoặc NaN → invalid; ảnh không vuông phải đổi normalized sang pixel |
| MAR | Chiều ngang 4, ba khoảng dọc bằng 2 → MAR=0.5; miệng đóng gần 0; chiều ngang 0 → invalid |
| Head pose | Dùng projectPoints tạo góc biết trước ±10° → sai số dưới 1° khi không có nhiễu; resize kèm scale K giữ góc; lỗi chiếu lại lớn → invalid |
| Timestamp/resample | Nguồn 15/24/30 FPS và timestamp không đều → chuỗi 10 Hz chỉ dùng quá khứ; mẫu cũ hơn 100 ms → unknown; timestamp trùng/đi lùi → reject |
| Blink/closure | Nhắm 0.15 giây rồi mở → blink; trên 0.8 giây → prolonged closure, không đếm blink; mất mẫu giữa event → cắt event, không nối |
| PERCLOS proxy | Nhắm 2 giây / quan sát hợp lệ 10 giây = 0.2; thêm 2 giây unknown không vào mẫu số, coverage=10/12; cắt biên 60 giây đúng; lịch sử dưới 30 giây hoặc coverage<0.8 → chưa ready |
| Yawn candidate | MAR delta vượt ngưỡng mở ít nhất 2 giây rồi xuống ngưỡng đóng → candidate; nói ngắn không đủ duration; missing cắt event, không gọi là ngáp thật |
| Split | Tập subject không giao nhau; video cùng người không tách; mỗi người làm outer test đúng một lượt; thiếu fold → reject; calibration prefix không vào window |
| Normalization | EAR 0.25 / baseline 0.25 = 1; baseline mắt trái 0.33, mắt phải 0.25 áp dụng riêng; baseline MAR bằng 0 không gây phép chia; thay test data không đổi scaler/baseline chung |
| Sequence/buffer | Từ t0 đến t0+9.9 giây có 100 mẫu; stride 1 giây tạo đúng index; missing>20%, gap>1 giây hoặc mẫu cuối invalid → không predict; đổi video → reset; mask giữ 0/1 |
| Model bundle | Sai thứ tự feature/schema/hash/mode → reject; reload cùng weights cho probability tương đương; thứ tự class `[0,1,2]` đúng |
| Metrics | Labels `[0,0,1,1,2,2]`, predictions `[0,1,1,1,2,0]` → Accuracy=4/6; recall=[0.5,1,0.5], Macro F1=59/90≈0.65556; support mỗi lớp bằng 2 |
| Smoother | Ba vector probability có mean đúng, tổng bằng 1; thiếu ba mẫu → null; gap≥2 giây → clear; NaN hoặc sai shape → reject |
| Alerts | Fake clock kiểm tra đúng các mốc 2/3/5/10 giây; một prediction cao chưa cảnh báo; unknown hủy timer chờ; strong warning không bị cooldown thường chặn; mute chỉ tắt audio |

Case PERCLOS 10 giây chỉ kiểm tra phép tích phân bên trong, không yêu cầu UI ready trước 30 giây. Kiểm tra ngay trước, đúng và ngay sau mỗi ngưỡng thời gian; không chỉ kiểm tra trường hợp thuận lợi.

## Integration và smoke
1. **Offline mini clip:** đọc video → landmark → features → Parquet → reload → window → cold-load model → prediction. Kiểm tra timestamps/schema và raw/replay equivalence.
2. **Dataset mini run:** người train/val/test thật tách biệt, train một fold; metrics và rejected report xuất được. Không dùng vài frame cùng người ở cả hai tập để smoke “training đúng”.
3. **Webcam:** nhìn thẳng, chớp nhanh, nói, nghiêng/quay đầu, che mặt, kính; low-light và kính râm phải hiện uncertainty. Diễn nhắm mắt chỉ kiểm tra event/warning, không làm ground truth trạng thái sinh lý.
4. **UI/audio:** Start/Stop năm lần, mute, recalibrate, rút camera, đóng cửa sổ; nghe âm thật, camera/thread được giải phóng; UI vẫn phản hồi khi worker bận.
5. **Performance:** chạy 10 phút và ghi latency từng bước, số frame bỏ, memory; đánh giá cảnh báo giả/giờ cần ít nhất một giờ Alert có consent, không suy từ clip ngắn.

## Offline/realtime parity
Replay dùng timestamp nguồn, không dùng tốc độ đọc file. Cùng frame stream/profile/checkpoint phải tạo feature và prediction giống trong sai số số thực cho phép; lịch UI có thể khác nhưng class/alerts theo thời gian nguồn phải khớp. Sai khác là lỗi pipeline, không giải thích chung bằng “khác vì realtime”.

## Lệnh bàn giao hiện tại
Chạy `.venv/Scripts/python.exe -m pytest -q --tb=short` và offline/native commands trong [README](../README.md): `scripts.build_splits`, `scripts.build_derived`, `scripts.train_baseline`, `scripts.train_lstm`, `scripts.webcam_demo`. Phase9–13 đã có real windows/models/cold reload; calibration/realtime/smoothing/replay core có deterministic regressions. `main.py`, audio/Qt UI và full-fold runtime evaluation vẫn thuộc phase sau. Unit suite không mở camera/download; physical acceptance log riêng.

Không cần unit test GUI chi tiết. Không test source text, tên file hay mock forwarding để chứng minh AI hoạt động. Không download dataset trong unit suite. Ghi rõ test skip do hardware/access; không coi skip là evidence pass realtime.

## Bằng chứng Phase6/7 — 2026-10-07
- Final regression394 passed in12.01s; deterministic suite không mở webcam/tải asset. Independent literal projection/temporary OBJ và real AVI kiểm consumer boundaries; thay chỉ native model boundary khi unit test cần isolation.
- Pose: neutral/±10/combined, calibrated/approximate/nonuniform resize/distortion, selected XY/rank/hull/depth/singular/solver/high-RMS và inclusive threshold. No ad-hoc180 patch. Camera/live physical signs phải chứng minh riêng.
- Quality/pipeline: fixed256 optics, inclusive boundaries, independent eyes/mean, zero-vs-missing, pose/optical/manual masks, valid→no-face→valid, source/time/index rejection, reset/terminal close, frozen SHA/value/path checks và native saved-packet parity.12 visual clean/36 degraded evidence không dùng weak class labels.
- Builder: real15FPS15frame expected timestamp sequence0..933, true Arrow null/float32/bool/int64/int32/int8, labels/leading-zero identity,1024 batch transitions, stale fingerprints, source changes, truncated decode, native/write/rename failures, symlink escape, missing/corrupt pairs và interruption giữa replaces. Writable fsync Windows và35/35.0 semantic-policy defects đã reproduce/fix.
- Native3 originals fullEOF/reload/resume:36,755 rows; actual GUI normal/manual/blur/dark/profile đã xem. Preview native close exception được source-report, vẫn attempt remaining cleanup và nguồn tiếp theo (RED/GREEN regression).
- Webcam45s Phase7 all806 no-face/released: không chứng minh live signs, physical sunglasses/face transitions/disconnect. Không chuyển missing hardware evidence thành PASS. Full04_0 raw-gate coverage thấp cần audit trước training, không nâng quality từ face-present hoặc synthetic success.

## Kiểm chứng Phase8 — frozen snapshot và independent audit
- `.venv/Scripts/python.exe -m pytest -q --tb=short`: **441 passed in16.42s** sau full-snapshot acceptance và scoped review fixes. Covering snapshot/audit suites trước cwd fix:46 passed in2.62s.
- Freeze kiểm byte hashes trước/sau parse và copy, source hashes, full manifest identity/program/signature drift; checkpoint failures giữ đủ frozen membership. Successful builder result phải khớp frozen fingerprint và `output_current=true`, không coi stale pair hoặc pending là complete.
- Audit dùng actual small Parquets: time/index disorder qua batch boundaries, first frame0 nhưng gaps hợp lệ, labels/provenance/null/masks/mean, closure/counters/schema. Yaw±120 reject, float32 yaw±90 endpoints pass; pitch/roll±180 giữ đúng estimator range. Ratios không bị arbitrary cap, invalid pose có thể giữ finite RMS.
- Before-fix runtime reproduction chứng minh audit cũ nhận yaw120 và prefix thiếu frame0 với metadata tự nhất quán; regression fixes đã qua46 covering tests và independent scoped re-review. Frame-weighted aggregation dùng unequal row counts, failed support và null zero denominator, không mock echoes.
- Native sequential04_0 replay đến40000ms đã chạy actual overlay, stored masks/pose khớp1e-5. 360ms có foreground hand/brightness rejection;11280/26880ms raw approximate pitch vượt25° khiến eyes/mouth N/A nhưng pose finite;40000ms bên trong envelope là control. Xem `runs/phase8/spotcheck/` và `implementation_verification.json`. Đây là bounded visual QC, không thay full-EOF acceptance hoặc chứng minh physical webcam signs.
- Actual `scripts.process_snapshot` → `scripts.audit_features` trên3 synthetic15FPS blank AVI:45 null rows, literal timestamps `[0,67,133,200,267,333,400,467,533,600,667,733,800,867,933]` mỗi video. Corrupt một synthetic source → CLI exit1/2cached+1failed; old valid pair không che extraction failure trong audit. Restore bytes → CLI exit0/3cached, audit exit0. Original manifests/freeze không đổi; không dùng synthetic labels làm ground truth.
- Actual native blank snapshot audit với absolute input paths từ cwd khác đã reproduce FileNotFoundError trước fix; sau khi resolve project-relative program hashes từ PROJECT_ROOT đạt3/3complete, regression còn kiểm actual code drift. Scoped re-review sạch. Runtime fixtures/reports ở `runs/phase8/native_blank_pipeline/`.
- Additional native low-coverage QC:16_0 có9461/9461face-present nhưng0eye/mouth/pose-valid, tất cả brightness-rejected. Sequential prefix0/3041/6040/9039/12038ms đo brightness63.91–65.48 dưới frozen min84.22; stored masks khớp. Actual window và permitted overlays cho thấy dark/warm illumination/shaded face; không đổi policy hoặc suy class từ missing. Evidence `runs/phase8/spotcheck/16_0_samples.json`, `resume_recovery.json`.
- Full34 working snapshot đã xử lý379,361rows/34EOF pairs và independent row/hash/schema/provenance audit đạt34complete/0failed. Actual full-cache rerun34cached/13.026s, re-audit cũng34complete; original manifest/config bytes và source fields/hash giữ nguyên. Output16,417,481bytes; current audit và coverage ở `runs/phase8/`. Nine videos0eye/mouth/pose-valid vẫn được giữ, subjects18/45 hoàn toàn0feature coverage. Staging cleanup check không còn `.tmp`; không claim training-ready hoặc physical webcam acceptance.

## Kiểm chứng Phase9–13 — 2026-10-07
- Final full suite **604passed27.11s**, sau runtime và clean scoped reviews. Parent-observed RED/GREEN: consumer provenance inventory, Arrow ndarray JSON, nonzero grid origin, prefix endpoint lookahead, positive closure intersection, missing-duration union, bundle/scaler mode binding, immutable native JSON, current-pose stale overlay và CRC32-vs-byte-size source validation. Final portrait surface smoke fits diagnostic/safety lines and renders unknown duration N/A.
- Unchanged Phase8 freeze actual resume34cached/14.881s/audit34complete/15.773s; raw producer/config/schema untouched. All5P0/P1 final datasets generated; complete role/class/source rejection receipts retained. P0outer3 invalid validation and P1all5blocked explicit, not fake metrics.
- Clear04_0 displayed65s/1301raw/651temporal: journal replay exact; raw/cache timestamps/indices/masks exact, numerical error<1e-5. Rejected16_0 displayed5s/100raw/50temporal:100faces but0usable channels, both abstain. Both overlays inspected/resources/window closed; `runs/phase10/`.
- Camera access only: initial60faces brightness-rejected; final56no-face/30temporal, both abstain/released/no image. Correct camera:0 identity; physical Alert calibration/blinks/long closure/mouth/turn/cover/return/disconnect still **NOT_RUN** until safe user observation.
- P0outer0 RF actual fit/evaluation, fresh-process full3861test reload max3.33e-16/atol1e-12;150real predictions independently recomputed. Sequence proof:49400unique train rows,64window exact arithmetic/3multi-video batches/all3labels. `runs/phase11/`, `runs/phase12/`.
- P0outer0 LSTM36epochs/selected28; real64test windows cold-load exact/atol1e-6 with six hashes/mode. RF/LSTM test rows match; current conditional MacroF1s low, no safety/generalization promise. Both P1outer0 CLIs blocked/nonzero/no checkpoint. No Phase14 full-fold/bootstrap/clinical/false-alarm or GPU claims.
