# Project 2 / DL
Repo này nhằm mục đích xây dựng và phát triển project 2 / DL về chủ đề post train mô hình LLM cỡ nhỏ (0.5-1.5B) nhằm tăng khả năng suy luận từ đó cải thiện hiệu quả trên các bộ benchmark như AIME24, 25, GSM8K.
# Phương pháp
Post train sử dụng SFT (cold-start), sau đó tiếp tục train sử dụng RL. 
# Cài đặt
```python
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```
# Lưu ý
-Code trong chỉ chạy được trên GPU.