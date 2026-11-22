# Báo cáo Day 08 — 2A202604046 - Hoàng Mạnh Hùng

> Một file duy nhất cho cả bài (đã gộp recipe, QC và ghi chú cá nhân). Mọi con số kèm **mẫu số** và nguồn (file nào). Viết bằng lời của bạn.

## 1. Model và recipe (Part 2–3)

| Lớp | Model + cỡ | Ngưỡng | Vì sao (dẫn `run_id` trong `model_log.csv`) |
|---|---|---|---|
| person / bicycle / motorbike / car / bus / truck | YOLO26m | Mặc định 0.3 cho cả 6 lớp (không lớp nào có ngưỡng riêng) | Nhanh (0.6 s/ảnh ở conf 0.25, r01) và bắt tốt lớp COCO phổ biến. Quét ngưỡng trên 4 ảnh gold: `person` tối đa 71% recall (28 vật), `bicycle` 80% (5 vật), `motorbike` 62% (21 vật), `car` 75% (8 vật), không ngưỡng nào ≤ 0.5 đạt 90% nên giữ mặc định; `bus`, `truck` có 0 vật trong gold nên cũng giữ mặc định. Ở r02 (conf 0.5) số box giảm 215 → 139 nhưng dễ rơi vật nhỏ, nên không dùng một ngưỡng chung 0.5. |
| three_wheeler | YOLOE-26s, prompt: `three_wheeler` | 0.15 | YOLO26 không có lớp này; YOLOE bắt được theo prompt chữ (r03, 8 box / 8 ảnh pilot, 0.34 s/ảnh). Quét ngưỡng: recall 100%, precision 78% trên 7 vật gold. |

- Lệnh `merge_prelabels.py` đã dùng (dán nguyên): `python scripts/merge_prelabels.py work/prelabel/yoloe_all/pre.coco.json work/prelabel/y26_all/pre.coco.json --thr "three_wheeler=0.15" --conf 0.3 --iou 0.6 --split "batch,gold" --out work/prelabel/merged_review/pre.coco.json`
- **Phiên bản SAM bạn chọn** (SAM 1 / 2.1 / 3) và lý do: SAM 2.1 (`sam2.1_b.pt`, r04). Mask bám viền người và xe tốt hơn box chữ nhật, chạy được trên máy 16 GB RAM (MPS). Đánh đổi: 3.84 s/ảnh, chậm hơn detector khoảng 6–11 lần (0.32–0.6 s/ảnh), nên chỉ dùng cho vài ảnh cần mask. Tôi chỉ thử bản này, chưa so sánh với SAM 1 hay SAM 3.
- Lỗi nào **không sửa được bằng đổi ngưỡng** (ví dụ tuk-tuk bị gọi `car`)? Sửa thế nào? Tuk-tuk bị YOLO26 gán nhầm `car`/`truck` hoặc bỏ sót (ví dụ `img_013.jpg`, <<xác nhận lại ảnh bạn thực sự thấy>>), vì mô hình chỉ có 80 lớp COCO. SAM cũng không sửa được vì chỉ vẽ lại viền theo box đầu vào. Cách sửa: khi gộp, đặt `yoloe_all` đứng trước `y26_all` để box `three_wheeler` thắng box `car`/`truck` chồng lên (IoU ≥ 0.6), phần còn lại do người duyệt sửa.
- Box **điểm cao mà sai** hoặc **điểm thấp mà đúng** (ảnh nào): <<điền sau khi duyệt batch ở Pha 4>>
- Giới hạn: chọn ngưỡng trên chính gold (4 ảnh) nên số đo máy trên gold là lạc quan; chỉ `three_wheeler` (7 vật) và `person` (28 vật) có cỡ mẫu tạm đủ, các lớp `bicycle` (5 vật), `car` (8 vật) là mẫu rất nhỏ. Gold đầu tiên của tôi vẽ box `motorbike`/`bicycle` bao cả người ngồi, trái guideline mục 4.1.4, khiến recall `bicycle` chỉ 0% (5 vật) và `motorbike` 67% (9 vật). Sau khi tách thành hai box riêng (`person` + xe), `bicycle` lên 80%; số vật `motorbike` tăng từ 9 lên 21 nhưng recall còn 62%. Gold còn 6 box `motorbike` cao 11–17 px (< 20 px) ở `img_036.jpg` do tôi chủ động không gom vào `ignore_region` để giữ tiến độ (cảnh báo của `validate_labels.py`), nhiều khả năng chúng kéo recall `motorbike` xuống; đây là hạn chế đã biết của số đo.

## 2. AI error profile (từ `qc/review_stats.md`)

| Trạng thái | Số | Tỉ lệ | Mẫu số |
|---|---|---|---|
| accepted | 494 | 94.8% | 521 box máy |
| edited | 2 | 0.4% | 521 box máy |
| deleted | 22 | 4.2% | 521 box máy |
| added | 14 | 2.7% | 510 đối tượng sau duyệt |

- Ghi chú về `edited`: script chỉ xếp 2 box vào nhóm này, trong khi tôi còn xóa 22, thêm 14 và đổi nhãn 7 box (`class_changed_info = 7`); các box chỉ kéo nhẹ cho khít có thể vẫn nằm trong `accepted`. Vì vậy `accepted` 94.8% có thể cao hơn mức "máy đúng thật" và tôi không coi nó là độ chính xác của máy.
- Lớp nào máy tệ nhất? Vì sao (dẫn ảnh)? `three_wheeler`: 11/42 box máy bị xóa (26%), cao nhất trong các lớp có mẫu đủ lớn (`person` 4/271, `motorbike` 1/138, `car` 4/51). Nguyên nhân dự đoán: YOLOE sinh box thừa/trùng trên xe khác hoặc trên cùng chiếc tuk-tuk (r03 trong `model_log.csv`); <<dẫn thêm ảnh cụ thể bạn đã thấy>>. `bus` có 1/3 bị xóa nhưng mẫu quá nhỏ để kết luận.
- So với gold: precision / recall của **máy** 60.3% / 59.4% (tp 41, fp 27, fn 28; mẫu số: 69 đối tượng gold trên 4 ảnh). Đây là bộ nhãn máy trước duyệt (`pre`); `final` chưa tính vì gold không nằm trong lô duyệt.

## 3. Thiên kiến tự động hóa và ảnh bẫy

- Bắt bẫy: **6/12 (50%)**, dưới mốc 10/12 (80%) của bài (`review_stats.json`). Loại nào lọt: `missing` 2/3 (`img_026`, `img_013`), `wrong_class` 2/3 (`img_034`, `img_008`), `extra` 1/3 (`img_005`), `loose` 1/3 (`img_021`); bắt được `extra` 2/3, `loose` 2/3, `missing` 1/3, `wrong_class` 1/3. Vì sao (bạn đã bỏ bước nào của checklist)? Tôi duyệt 24 ảnh trong khoảng 20 phút (~50 giây/ảnh), nhiều khả năng đã làm không đủ bước 1 (ẩn nhãn, đếm vật trước khi nhìn gợi ý) nên lọt `missing`, và bước 3 (soi nhãn và bốn cạnh từng box) nên lọt `wrong_class` và `loose`. <<xác nhận lại đúng bước bạn đã bỏ>>
- Accepted cao mà bắt bẫy thấp (hoặc ngược lại) nói gì về cách bạn duyệt? `accepted` 94.8% đi cùng bắt bẫy 50% cho thấy tôi có xu hướng tin nhãn máy: box trông gọn thì bị lướt qua. Tỉ lệ accepted cao không chứng minh máy tốt mà có thể phản ánh việc tôi duyệt chưa đủ kỹ. Tôi không duyệt lại 6 ảnh lọt vì ưu tiên tiến độ, nên checkpoint Pha 4 (≥ 80%) chưa đạt và tôi ghi nhận trung thực ở đây.

## 4. Máy có đáng dùng không?

- Thời gian (từ `time_log.csv`): gán tay 10 phút / 4 ảnh = **2.5 phút/ảnh** vs duyệt 20 phút / 24 ảnh = **~0.83 phút (~50 giây)/ảnh**, tức nhanh hơn khoảng 3 lần. Thiên lệch (bạn gán gold trước nên đã quen cảnh): tôi gán gold trước nên đã quen cảnh phố khi duyệt batch, và 10 phút của gold có thể chưa tính hết thời gian sửa lại gold theo guideline 4.1.4; hai điều này làm khoảng cách thời gian có thể bị phóng đại. Ngoài ra, thời gian duyệt nhanh đi kèm bắt bẫy chỉ 6/12 nên không so sánh được chất lượng như nhau.
- Kết luận: **chỉ cho lớp phổ biến** (`person`, `motorbike`, `car`), dựa trên tỉ lệ box máy được giữ (person 266/271, motorbike 137/138, car 47/51) và tốc độ duyệt ~3 lần. Với `three_wheeler` thì không nên tin máy (11/42 box bị xóa, 26%), và với toàn bộ lô thì số liệu chưa đủ chắc vì bắt bẫy chỉ 50% và precision/recall của máy trên gold chỉ 60.3% / 59.4%.
- **Ba quy tắc giảm thiên kiến của bạn** (mỗi quy tắc kiểm tra được bằng số hoặc việc làm):
  1. Luôn ẩn nhãn và đếm người/xe bằng mắt trước khi bật nhãn máy (kiểm tra bằng số: bắt bẫy `missing` ≥ 2/3).
  2. Không duyệt nhanh hơn ~1.5 phút/ảnh với ảnh có nhiều vật (kiểm tra bằng `time_log.csv`, so với 0.83 phút/ảnh lần này).
  3. Với `three_wheeler` và box `conf` thấp, lọc riêng bằng Filters và soi từng box, tick `can_review` khi không chắc (kiểm tra bằng số box `three_wheeler` bị xóa/sửa so với lần này).

## 5. Mask SAM cho 3 vật (Part 5)

| Ảnh | SAM (phiên bản) | Số click | Mask gồm người lái / hàng hoá? | Lỗi quan sát | File trong `masks/` |
|---|---|---|---|---|---|
| <<img_0xx>> | <<>> | <<>> | <<>> | <<>> | <<>> |

- Đối chiếu guideline mục 4.2: quy tắc nào giúp, quy tắc nào thiếu? <<>>
- *(Tuỳ chọn)* Interactor SAM trong CVAT: làm được / không (lỗi gì, đã thử gì); khác gì offline: <<>>

## 6. Calibration (Part 6)

- Số quyết định: <<>> (`calibration_decisions.csv`); quan trọng nhất (dẫn `CAL-xx`): <<>>
- Quy tắc mới đã vào `guideline_v1.1_calibrated.md`: <<>>
- Nguyên nhân gốc: <<guideline mơ hồ / chưa nắm quy tắc / máy>>

## 7. QC và quyết định (Part 6)

- Mẫu: <<n>> ảnh (<<m>> ảnh `can_review` + <<k>> ngẫu nhiên, seed 42); người kiểm: bạn (**hạn chế**: bạn vừa là người duyệt).
- Lỗi theo **đối tượng**: <<.../... = ...%>> · theo **ảnh**: <<.../... = ...%>> (ngưỡng Release ≤ 2%)
- Top lỗi (trong: bỏ sót, thừa, sai lớp, sai vị trí/kích thước, sai thuộc tính, sai mask) + một ảnh ví dụ mỗi loại: <<>>
- **Giới hạn mẫu nhỏ** (cận trên 95% trong `qc_report_auto.md`): <<>>
- Script kiểm được / không kiểm được (ví dụ ôm sát, cùng bỏ sót với máy, theo checklist thật): <<>>

**Quyết định: <<Release / Hold / Rework>>**, ngày <<YYYY-MM-DD>>. Lý do (đi theo Hold → Rework → Release): <<>>
Nếu Rework / Hold, việc cần làm tiếp: <<>>

## 8. Ghi chú cá nhân

- **Dùng AI để làm bài** (công cụ nào, việc gì, kiểm tra thế nào; "không dùng" nếu không): Dùng Claude để hướng dẫn từng bước theo file hướng dẫn, đọc kết quả `threshold_sweep.md` và `review_stats.json`, soạn bản nháp `model_log.csv` và mục 1–4 của báo cáo. Tôi kiểm tra bằng cách đối chiếu với file hướng dẫn và guideline gốc, và sửa lại các chỗ Claude ghi theo ví dụ mẫu mà tôi chưa tự xác nhận trên ảnh.
- Gold có bị nhiễm không (bạn có nhìn pre-label hay hỏi AI trước khi gán xong)? Không nhìn pre-label hay hỏi AI trước khi gán xong lần đầu. Tuy nhiên sau khi quét ngưỡng thấy recall thấp (`bicycle` 0%), tôi đọc `guideline_v1.0_label_contract.md` mục 4.1.4 và sửa lại gold (tách box `person` riêng khỏi `motorbike`/`bicycle`). Tôi sửa theo guideline, không sao chép nhãn máy, nhưng việc sửa được gợi ý nhờ kết quả của máy nên gold không hoàn toàn độc lập.
- Máy / GPU / CVAT dùng; vướng mắc cài đặt: Máy 16 GB RAM, chạy trên MPS (`runs.csv`); CVAT dùng để gán và duyệt. <<vướng mắc cài đặt nếu có>>
- Dùng cache `data/cache/` thay vì tự chạy? Không, tôi tự chạy `prelabel.py`.
- Lỗi / bug script đã gặp (nếu có): <<>>
- Một điều bạn làm khác lần sau: Đọc kỹ guideline trước khi gán gold, và dành đủ thời gian cho checklist 4 bước khi duyệt thay vì duyệt nhanh.