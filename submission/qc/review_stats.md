# Số đo review (det)

- Ảnh tính: 24 (batch,gold); box máy: 521; đối tượng sau duyệt: 510; ngưỡng accepted: IoU >= 0.9
- Đã **loại 23 đối tượng nằm quanh các bẫy** khỏi bảng này: lỗi nhân tạo không phản ánh chất lượng máy.

## 1. Bốn trạng thái (AI error profile)
| Trạng thái | Số | Tỉ lệ | Mẫu số |
|---|---|---|---|
| accepted (giữ nguyên) | 494 | 494/521 = 94.8% | 521 box máy |
| edited (sửa) | 2 | 2/521 = 0.4% | 521 box máy |
| deleted (xoá) | 22 | 22/521 = 4.2% | 521 box máy |
| added (vẽ thêm) | 14 | 14/510 = 2.7% | 510 đối tượng sau duyệt |

Trong số deleted + added, **7 trường hợp là đổi lớp** (máy gán sai lớp, box đúng chỗ): đó không phải máy bỏ sót. Số vật máy **sót thật** ≈ added − đổi lớp = 7.

### Theo lớp
| Lớp | accepted | edited | deleted | added |
|---|---|---|---|---|
| person | 266 | 1 | 4 | 1 |
| bicycle | 7 | 0 | 1 | 3 |
| motorbike | 137 | 0 | 1 | 4 |
| car | 47 | 0 | 4 | 2 |
| bus | 2 | 0 | 1 | 1 |
| truck | 5 | 0 | 0 | 1 |
| three_wheeler | 30 | 1 | 11 | 2 |

## 2. So với gold (nhãn gán tay từ đầu)
Ảnh gold: 4; đối tượng gold: 69 (mẫu số của recall)
| | TP | FP | FN | Precision | Recall |
|---|---|---|---|---|---|
| Máy (pre-label) | 41 | 27 | 28 | 60.3% (KTC 95%: 48–71%) | 59.4% (KTC 95%: 48–70%) |

*Lạc quan: ngưỡng được chọn trên chính các ảnh gold này (in-sample), và chỉ có 4 ảnh. Đọc khoảng tin cậy, đừng đọc con số đơn.*


(Ảnh gold không nằm trong nhãn sau duyệt: không tính residual / new error. Đó là chủ ý: duyệt lại ảnh bạn vừa tự gán tay sẽ đo trí nhớ, không đo thiên kiến.)

## 3. Ảnh bẫy
- Bắt được 6/12 = 50.0% (KTC 95%: 25–75%) (ngưỡng đạt: 80%). Chỉ 12 bẫy nên khoảng tin cậy rất rộng: đọc cùng danh sách bẫy lọt, đừng kết luận chỉ từ %.
  - extra: 2/3
  - missing: 1/3
  - wrong_class: 1/3
  - loose: 2/3
  - lọt: img_026.jpg (missing), img_005.jpg (extra), img_034.jpg (wrong_class), img_021.jpg (loose), img_008.jpg (wrong_class), img_013.jpg (missing)

## Gợi ý đọc số
- `deleted` cao → ngưỡng thấp hoặc ánh xạ nhãn sai; `added` cao → máy bỏ sót (xem lớp nào);
- `accepted` thấp (< 50%) → sửa mỗi box lâu hơn vẽ mới: đổi model/ngưỡng hoặc vẽ tay;
- bắt bẫy < 80% → duyệt lại lô, siết checklist (quét ảnh TRƯỚC khi nhìn nhãn máy).
