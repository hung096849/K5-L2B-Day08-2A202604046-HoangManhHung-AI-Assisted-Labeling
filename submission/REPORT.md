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
| img_013.jpg | SAM 2.1 (`sam2.1_b.pt`) | 9 (5 dương + 4 âm) | Không gồm người lái và khách (áo xanh, áo sọc đỏ); mask 269.847 px (38,6% ảnh), 1 polygon | Còn đốm lẻ trên tay, quần, dép người lái; kính chắn gió bị phủ | `tuktuk_img013_sam2_pos5neg4.jpg` |
| img_013.jpg | MobileSAM (SAM 1, `mobile_sam.pt`) | 9 (5 dương + 4 âm, cùng tọa độ) | Không gồm người, sạch hơn SAM 2.1; mask 185.609 px (26,6% ảnh), 2 polygon | Mái bạc chỉ phủ vài mảng, mất khoảng nửa mái; đầu xe đỏ phủ mỏng | `tuktuk_img013_sam1_pos5neg4.jpg` |
| img_004.jpg | SAM 2.1 (`sam2.1_b.pt`) | 3 (2 dương + 1 âm) | Không dính người đứng gần; không lan sang xe xanh lá bên cạnh | Không có lỗi đáng kể | `tuktuk_img004_sam2.jpg` |
| img_017.jpg | SAM 2.1 (`sam2.1_b.pt`) | 3 (2 dương + 1 âm) | Xe không có người ngồi; mask không dính bàn ghế ở nền | Biển `TAXI` trên mái hơi tách khỏi mask | `tuktuk_img017_sam2.jpg` |

- Quá trình sửa `img_013.jpg` (SAM 2.1): (1) 2 dương + 1 âm (`tuktuk_img013_sam2`): người lái và khách vẫn bị phủ; (2) 2 dương + 4 âm (`tuktuk_img013_sam2_neg4`): người đã tách nhưng mask vỡ, chỉ 80.208 px (11,5% ảnh) và 2 polygon, thiếu thân xe, bánh trước và đầu xe; (3) thêm 3 điểm dương lên bánh trước, cản sau và đầu xe (`tuktuk_img013_sam2_pos5neg4`): 269.847 px (38,6%), 1 polygon, phủ đủ xe. Kết luận: một điểm âm không đủ để tách người, và nhiều điểm âm mà ít điểm dương làm mask vỡ, nên phải cân bằng cả hai.
- So sánh SAM 2.1 và MobileSAM trên cùng 9 click: MobileSAM phủ ít hơn khoảng 31% diện tích (185.609 so với 269.847 px) và làm mất nửa mái xe, nhưng tách người sạch hơn. Tôi chọn SAM 2.1 vì mask liền mạch hơn, hợp mục tiêu "mask ôm cả khung xe và mái" của guideline 4.2. Đây là nhận xét định tính trên một ảnh, tôi không đo IoU; script `sam2_points.py` cũng không in thời gian chạy nên tôi không có số giây để so sánh tốc độ giữa hai bản.
- Đối chiếu guideline mục 4.2: quy tắc giúp: mask chỉ ôm khung xe, mái và bánh còn người tách riêng (điểm `--neg` dùng đúng để thực hiện việc này). Quy tắc thiếu: guideline chưa nói kính chắn gió có thuộc mask xe hay không (mục 6, câu hỏi 5 vẫn mở), nên mask `img_013` phủ kính mà tôi chưa có căn cứ để tính đó là lỗi hay đúng; guideline cũng chưa nói về khách ngồi trong xe ba bánh có mui che ở mức mask.
- *(Tuỳ chọn)* Interactor SAM trong CVAT: không thử; tôi chỉ chạy SAM offline bằng `sam2_points.py`.

## 6. Calibration (Part 6)

- Số quyết định: 4 (`calibration_decisions.csv`); quan trọng nhất (dẫn `CAL-xx`): **CAL-02** (một vật một box, vì `thua` là loại lỗi nhiều nhất trong mẫu QC: 4/7 lỗi, ở `img_003` và `img_013`) và **CAL-03** (kính chắn gió thuộc mask xe, chốt câu hỏi mở số 5 ở mục 6 của guideline, ví dụ `img_013`). CAL-01 (máy gán `car` cho xe máy ở `img_024`, box sai lớp ở xa ở `img_013`) và CAL-04 (lọt 6/12 bẫy) là lỗi của máy và của cách duyệt nên không đổi guideline.
- Quy tắc mới đã vào `guideline_v1.1_calibrated.md`: 4.1.8 "Một vật một box: giữ box ôm khớp nhất, xóa box trùng" và 4.2.3 "Kính chắn gió, cửa sổ xe thuộc mask xe; người nhìn thấy qua kính không thuộc mask xe". Các câu hỏi mở 1–4 và 6–7 của mục 6 chưa chốt vì chưa gặp trong lô, tôi ghi nhận đây là hạn chế.
- Nguyên nhân gốc: chủ yếu **máy và cách duyệt của tôi** (CAL-01, CAL-02, CAL-04: duyệt ~50 giây/ảnh, bắt bẫy 6/12); chỉ 2/4 ca (CAL-02, CAL-03) cần sửa chữ trong guideline. Trước đó gold bị lệch quy ước vì tôi gán khi chưa đọc guideline (xem mục 1).

## 7. QC và quyết định (Part 6)

- Mẫu: 7 ảnh (2 ảnh `can_review` + 5 ngẫu nhiên, seed 42), 172 đối tượng (`qc_sample.csv`); người kiểm: bạn (**hạn chế**: bạn vừa là người duyệt).
- Lỗi theo **đối tượng**: 7/172 = 4,1% · theo **ảnh**: 3/7 = 42,9% (ngưỡng Release ≤ 2%) (`qc_report_auto.md`).
- Top lỗi (trong: bỏ sót, thừa, sai lớp, sai vị trí/kích thước, sai thuộc tính, sai mask) + một ảnh ví dụ mỗi loại: **thừa** 4 lỗi (`img_003`: box #7, #8 trùng với #6; `img_013`: #140, #141 cùng một vật, không box nào khớp); **sai lớp** 3 lỗi (`img_024`: #324 gán `car` cho xe máy; `img_013`: #138, #139 box ở xa không gắn với vật nào). Không có lỗi bỏ sót, sai vị trí/kích thước, sai thuộc tính hay sai mask trong mẫu này. Lỗi `thua` lặp lại nên là lỗi hệ thống, không chỉ sửa từng ảnh.
- **Giới hạn mẫu nhỏ** (cận trên 95% trong `qc_report_auto.md`): 3 ảnh lỗi trong 7 ảnh, cận trên Wilson 95% của tỉ lệ ảnh lỗi thật là **75%**; mẫu 7 ảnh không thể chứng minh ngưỡng 2%. Mẫu cũng chỉ có 2 ca khó và 5 ảnh ngẫu nhiên trên 24 ảnh batch.
- Script kiểm được / không kiểm được (ví dụ ôm sát, cùng bỏ sót với máy, theo checklist thật): script chỉ đếm theo phiếu `qc_review.csv` do tôi điền bằng mắt, nên không kiểm được box có ôm sát vật thật không, lớp có đúng thực tế không, vật mà cả tôi và máy cùng bỏ sót, và việc tôi có làm đúng checklist hay không. Vì tôi vừa duyệt vừa kiểm nên có thể bỏ lọt cùng loại lỗi đã bỏ lọt khi duyệt.

**Quyết định: Rework**, ngày 2026-11-23. Lý do (đi theo Hold → Rework → Release): không chọn Release vì ảnh lỗi 3/7 = 42,9% > 2%, lỗi `thua` lặp (4/7 lỗi) và bắt bẫy 6/12 = 50% < 80% nên lô duyệt chưa đáng tin. Không chọn Hold vì dữ liệu không hỏng và hai ca guideline mơ hồ gặp trong lô đã được chốt ở v1.1 (CAL-02, CAL-03); các câu hỏi mở còn lại chưa gặp trong lô.
Nếu Rework / Hold, việc cần làm tiếp: (1) sửa trong CVAT các box lỗi đã tìm thấy (`img_003` #7, #8; `img_013` #138–#141; `img_024` #324) rồi export `submission/labels/qc_fixed.json`; (2) duyệt lại 6 ảnh bẫy lọt (`img_026`, `img_013`, `img_005`, `img_034`, `img_008`, `img_021`) theo đủ checklist và dành nhiều thời gian hơn; (3) chạy lại `build_release.py` và `qc_sample.py --force`, kiểm lại với mẫu lớn hơn rồi ghi vòng sau vào báo cáo.

## 8. Ghi chú cá nhân

- **Dùng AI để làm bài** (công cụ nào, việc gì, kiểm tra thế nào; "không dùng" nếu không): Dùng Claude để hướng dẫn từng bước theo file hướng dẫn, đọc kết quả `threshold_sweep.md` và `review_stats.json`, soạn bản nháp `model_log.csv` và mục 1–4 của báo cáo. Tôi kiểm tra bằng cách đối chiếu với file hướng dẫn và guideline gốc, và sửa lại các chỗ Claude ghi theo ví dụ mẫu mà tôi chưa tự xác nhận trên ảnh.
- Gold có bị nhiễm không (bạn có nhìn pre-label hay hỏi AI trước khi gán xong)? Không nhìn pre-label hay hỏi AI trước khi gán xong lần đầu. Tuy nhiên sau khi quét ngưỡng thấy recall thấp (`bicycle` 0%), tôi đọc `guideline_v1.0_label_contract.md` mục 4.1.4 và sửa lại gold (tách box `person` riêng khỏi `motorbike`/`bicycle`). Tôi sửa theo guideline, không sao chép nhãn máy, nhưng việc sửa được gợi ý nhờ kết quả của máy nên gold không hoàn toàn độc lập.
- Máy / GPU / CVAT dùng; vướng mắc cài đặt: Máy 16 GB RAM, chạy trên MPS (`runs.csv`); CVAT dùng để gán và duyệt. <<vướng mắc cài đặt nếu có>>
- Dùng cache `data/cache/` thay vì tự chạy? Không, tôi tự chạy `prelabel.py`.
- Lỗi / bug script đã gặp (nếu có): <<>>
- Một điều bạn làm khác lần sau: Đọc kỹ guideline trước khi gán gold, và dành đủ thời gian cho checklist 4 bước khi duyệt thay vì duyệt nhanh.