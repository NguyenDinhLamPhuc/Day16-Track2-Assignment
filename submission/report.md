# Báo Cáo Thực Hành LAB 16: Cloud AI Environment Setup

1. Tôi dùng AWS, us-east-1, t3.micro, source commit `daf02a5`.
2. Dataset Credit Card Fraud có 284,807 dòng, chia train/validation/test 60/20/20, seed 16.
3. Load dữ liệu mất 2.5 giây; training mất 3.52 giây; best iteration của LightGBM là 68.
4. Trên tập test, mô hình đạt AUC: 0.9768, Accuracy: 0.9995, F1: 0.8478, Precision: 0.9070, Recall: 0.7959.
5. Tốc độ suy luận (inference): Latency 1 dòng 1.208 ms; throughput khi chạy batch 1.000 dòng là ~304,832 dòng/giây; cách đo bằng trung vị (`median; warm-up excluded; predict_proba trên pandas input`).
6. CPU, RAM và Network được quan sát trên compute node AWS bằng các lệnh `top`, `free -h` và `ip -s link` trong lúc benchmark chạy. Ảnh minh họa được lưu tại thư mục screenshots trong thư mục submission.
7. Billing trên GCP tại thời điểm kiểm tra 11h30 2/10 ghi nhận chi phí `chưa cập nhật`; ước tính phí duy trì IP tĩnh và NAT là `[~0.04$/giờ]`.
8. Tôi đã tải toàn bộ kết quả về máy và đã chạy lệnh `terraform destroy` xóa tài nguyên lúc `11h35`; bằng chứng dọn dẹp tại ảnh `submission/screenshots/don_dep.png`