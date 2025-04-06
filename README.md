# Project 2 / DL
Repo này nhằm mục đích xây dựng và phát triển project 2 / DL về chủ đề post train mô hình LLM cỡ nhỏ (0.5-1.5B) nhằm tăng khả năng suy luận từ đó cải thiện hiệu quả trên các bộ benchmark như AIME24, 25, GSM8K.
# Phương pháp
Post train sử dụng SFT (cold-start), sau đó tiếp tục train sử dụng RL.
# Cấu trúc
Project hiện có các file sau: 
- `data.py` (chưa hoàn thiệnthiện) chứa code để load các bộ dữ liệu phục vụ cho việc train và đánh giá mô hình. Dữ liệu được xử lý bằng cách đưa về dạng phù hợp cho mục đích (train hoặc đánh giá) sau đó đưa qua tokenizer để tensor hóa và chuyển thành object lớp `Dataset` để có thể load sử dụng PyTorch `DataLoader`. Các bộ dữ liệu hiện có bao gồm Light-R1-SFTData, Dapo-Math-17k, AIME2024, AIME2025.
- `eval.py` (chưa hoàn thiện) chứa code để đánh giá mô hình. Việc lấy kết quả đang sử dụng fixed format, nên thay đổi sang sử dụng các công cụ như `Math-Verify` để tăng độ chính xác.
- `sft.py` chứa code huấn luyện mô hình trong giai đoạn SFT.
- `rl.py` (chưa hoàn thiện) chứa code huấn luyện mô hình trong giai đoạn RL. 
- `model.py` chứa các lớp phụ trợ gồm tham số `Args`, phương thức load mô hình và tokenizer cũng như các phương thức khác. 
# Lưu ý
- Sau khi clone, cần cài đặt các thư viện cần thiết bằng lệnh
```python
pip install -r requirements.txt
```
- Code trong SFT chỉ chạy được trên GPU.