# Báo cáo Day 08 — <<2A202604046 - Hoàng Mạnh Hùng>>

> Một file duy nhất cho cả bài (đã gộp recipe, QC và ghi chú cá nhân). Mọi con số kèm **mẫu số** và nguồn (file nào). Viết bằng lời của bạn.

## 1. Model và recipe (Part 2–3)

| Lớp | Model + cỡ | Ngưỡng | Vì sao (dẫn `run_id` trong `model_log.csv`) |
|---|---|---|---|
| person / bicycle / motorbike / car / bus / truck | YOLO26m | Mặc định 0.3 cho cả 6 lớp (không lớp nào có ngưỡng riêng) | Nhanh (0.6 s/ảnh ở conf 0.25, r01) và bắt tốt lớp COCO phổ biến. Quét ngưỡng trên 4 ảnh gold: `person` tối đa 71% recall (28 vật), `bicycle` 80% (5 vật), `motorbike` 62% (21 vật), `car` 75% (8 vật), không ngưỡng nào ≤ 0.5 đạt 90% nên giữ mặc định; `bus`, `truck` có 0 vật trong gold nên cũng giữ mặc định. Ở r02 (conf 0.5) số box giảm 215 → 139 nhưng dễ rơi vật nhỏ, nên không dùng một ngưỡng chung 0.5. |
| three_wheeler | YOLOE-26s, prompt: `three_wheeler` | 0.15 | YOLO26 không có lớp này; YOLOE bắt được theo prompt chữ (r03, 8 box / 8 ảnh pilot, 0.34 s/ảnh). Quét ngưỡng: recall 100%, precision 78% trên 7 vật gold. |

- Lệnh `merge_prelabels.py` đã dùng (dán nguyên): <<điền sau khi chạy ở Pha 4>>
- **Phiên bản SAM bạn chọn** (SAM 1 / 2.1 / 3) và lý do: SAM 2.1 (`sam2.1_b.pt`, r04). Mask bám viền người và xe tốt hơn box chữ nhật, chạy được trên máy 16 GB RAM (MPS). Đánh đổi: 3.84 s/ảnh, chậm hơn detector khoảng 6–11 lần (0.32–0.6 s/ảnh), nên chỉ dùng cho vài ảnh cần mask. Tôi chỉ thử bản này, chưa so sánh với SAM 1 hay SAM 3.
- Lỗi nào **không sửa được bằng đổi ngưỡng** (ví dụ tuk-tuk bị gọi `car`)? Sửa thế nào? Tuk-tuk bị YOLO26 gán nhầm `car`/`truck` hoặc bỏ sót (ví dụ `img_013.jpg`, <<xác nhận lại ảnh bạn thực sự thấy>>), vì mô hình chỉ có 80 lớp COCO. SAM cũng không sửa được vì chỉ vẽ lại viền theo box đầu vào. Cách sửa: khi gộp, đặt `yoloe_all` đứng trước `y26_all` để box `three_wheeler` thắng box `car`/`truck` chồng lên (IoU ≥ 0.6), phần còn lại do người duyệt sửa.
- Box **điểm cao mà sai** hoặc **điểm thấp mà đúng** (ảnh nào): <<điền sau khi duyệt batch ở Pha 4>>
- Giới hạn: chọn ngưỡng trên chính gold (4 ảnh) nên số đo máy trên gold là lạc quan; chỉ `three_wheeler` (7 vật) và `person` (28 vật) có cỡ mẫu tạm đủ, các lớp `bicycle` (5 vật), `car` (8 vật) là mẫu rất nhỏ. Gold đầu tiên của tôi vẽ box `motorbike`/`bicycle` bao cả người ngồi, trái guideline mục 4.1.4, khiến recall `bicycle` chỉ 0% (5 vật) và `motorbike` 67% (9 vật). Sau khi tách thành hai box riêng (`person` + xe), `bicycle` lên 80%; số vật `motorbike` tăng từ 9 lên 21 nhưng recall còn 62%. Gold còn 6 box `motorbike` cao 11–17 px (< 20 px) ở `img_036.jpg` do tôi chủ động không gom vào `ignore_region` để giữ tiến độ (cảnh báo của `validate_labels.py`), nhiều khả năng chúng kéo recall `motorbike` xuống; đây là hạn chế đã biết của số đo.

## 2. AI error profile (từ `qc/review_stats.md`)

| Trạng thái | Số | Tỉ lệ | Mẫu số |
|---|---|---|---|
| accepted | <<>> | <<>> | <<box máy>> |
| edited | <<>> | <<>> | <<>> |
| deleted | <<>> | <<>> | <<>> |
| added | <<>> | <<>> | <<đối tượng sau duyệt>> |

- Lớp nào máy tệ nhất? Vì sao (dẫn ảnh)? <<>>
- So với gold: precision / recall của **máy** <<>> (mẫu số: <<đối tượng gold>>)

## 3. Thiên kiến tự động hóa và ảnh bẫy

- Bắt bẫy: <<.../12>>; loại nào lọt: <<>>; vì sao (bạn đã bỏ bước nào của checklist)? <<>>
- Accepted cao mà bắt bẫy thấp (hoặc ngược lại) nói gì về cách bạn duyệt? <<>>

## 4. Máy có đáng dùng không?

- Thời gian (từ `time_log.csv`): gán tay <<... phút/ảnh>> vs duyệt <<... phút/ảnh>>. Thiên lệch (bạn gán gold trước nên đã quen cảnh): <<>>
- Kết luận: <<có / không / chỉ cho lớp nào>>, dựa trên <<số liệu>>
- **Ba quy tắc giảm thiên kiến của bạn** (mỗi quy tắc kiểm tra được bằng số hoặc việc làm):
  1. <<>>
  2. <<>>
  3. <<>>

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

- **Dùng AI để làm bài** (công cụ nào, việc gì, kiểm tra thế nào; "không dùng" nếu không): <<>>
- Gold có bị nhiễm không (bạn có nhìn pre-label hay hỏi AI trước khi gán xong)? <<không / có, như sau: ...>>
- Máy / GPU / CVAT dùng; vướng mắc cài đặt: <<>>
- Dùng cache `data/cache/` thay vì tự chạy? <<không / có, vì ...>>
- Lỗi / bug script đã gặp (nếu có): <<>>
- Một điều bạn làm khác lần sau: <<>>