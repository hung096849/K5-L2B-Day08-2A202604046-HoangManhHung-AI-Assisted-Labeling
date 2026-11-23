# Thông tin học viên

- Họ và tên: Hoàng Mạnh Hùng
- MSHV: 
- Lớp (L2 / L3 / khác): L2
- Tài khoản GitHub: hung096849
- Repo bài nộp (URL): https://github.com/hung096849/K5-L2B-Day08-2A202604046-HoangManhHung-AI-Assisted-Labeling.git
- Commit chốt bài (hash): Done lab 
- Máy: Macbook Air M3, 16GB RAM
- CVAT (Docker):  v2.81.0

## Kết quả tóm tắt (điền cuối bài; số lấy từ `submission/qc/*`, kèm mẫu số)

- Máy: accepted / edited / deleted: 494/521 = 94.8% / 2/521 = 0.4% / 22/521 = 4.2% (mẫu số: 521 box máy); added: 14/510 = 2.7% (mẫu số: 510 đối tượng sau duyệt)
- So với gold: precision / recall của **máy** <<chưa có: báo cáo không có số đo so với gold>>
- Bắt bẫy: 6/12 = 50.0%
- Thời gian: gán tay <<chưa có>> phút/ảnh; duyệt có pre-label <<chưa có>> phút/ảnh
- Lỗi QC: 42.9% theo ảnh (3/7), 4.1% theo đối tượng (7/172)
- **Quyết định:** Rework — ảnh có lỗi 42.9% > ngưỡng 2%, lỗi hệ thống "thua" lặp lại (4/7 lỗi) và bắt bẫy chỉ 50% < 80%.
