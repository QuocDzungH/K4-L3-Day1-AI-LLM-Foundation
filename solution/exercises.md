# K4 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 4 tiếng
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `*Câu trả lời của bạn*` bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.5, 1.0 và 1.5 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Việt Nam."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)
> Khi tăng temperature từ 0.0 lên 1.5, câu trả lời chuyển từ tính xác định, quy chuẩn sang đa dạng và sáng tạo hơn. Ở mức 0.0, mô hình luôn lặp lại cùng một sự thật phổ biến nhất (như Việt Nam xuất khẩu cà phê Robusta lớn thứ hai thế giới) với câu từ khuôn mẫu. Khi lên 0.5 - 1.0, góc nhìn phong phú và tự nhiên hơn; nhưng tới 1.5, văn phong trở nên quá bay bổng và có nguy cơ xuất hiện thông tin thiếu chính xác (ảo giác/hallucination).

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**
> Tôi sẽ đặt temperature ở mức **0.0 đến 0.2**. Chatbot hỗ trợ khách hàng đòi hỏi độ chính xác, tính nhất quán và độ tin cậy tuyệt đối về chính sách, giá cả và thông tin kỹ thuật. Mức temperature thấp triệt tiêu tính ngẫu nhiên, giúp câu trả lời bám sát tài liệu hướng dẫn và tránh nguy cơ trả lời sai sự thật gây rủi ro cho doanh nghiệp.

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 3 lần,
mỗi lần trung bình ~350 token đầu ra.

**Ước tính GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này? Nêu một
trường hợp GPT-4o xứng đáng với chi phí và một trường hợp nên dùng mini:**
> - **Ước tính:** Giá output của GPT-4o là $0.010/1K token, còn GPT-4o-mini là $0.0006/1K token. Như vậy, GPT-4o đắt hơn GPT-4o-mini xấp xỉ **16.7 lần** (với 10,5 triệu output token/ngày, GPT-4o tốn khoảng $105/ngày trong khi mini chỉ tốn ~$6.3/ngày).
> - **Nên dùng GPT-4o:** Khi cần suy luận logic phức tạp, tư vấn tài chính/pháp lý chuyên sâu, giải quyết lỗi lập trình khó hoặc đọc hiểu văn bản dài nhiều tầng nghĩa.
> - **Nên dùng GPT-4o-mini:** Chatbot hỗ trợ khách hàng bậc 1 (FAQ), tóm tắt tin tức/email, phân loại ý định (intent classification), hoặc trích xuất thông tin cơ bản.

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích blockchain là gì?"** nhưng hai system prompt khác nhau:
- "Bạn là giáo viên tiểu học, giải thích thật đơn giản cho trẻ 8 tuổi."
- "Bạn là chuyên gia tài chính, trả lời chuyên sâu bằng thuật ngữ kỹ thuật."

**Hai phản hồi khác nhau như thế nào (độ dài, từ vựng, ví dụ)? System prompt
ảnh hưởng đến hành vi model ra sao?** (3–4 câu)
> Hai phản hồi có sự phân hóa rõ rệt: phản hồi cho trẻ em ngắn gọn, dùng từ vựng đơn giản và hình ảnh ví von thân thuộc (như cuốn sổ nhật ký chung của cả lớp mà ai cũng có một bản copy), trong khi phản hồi cho chuyên gia tài chính dài hơn, trang trọng và dày đặc thuật ngữ kỹ thuật (sổ cái phân tán, cơ chế đồng thuận PoW/PoS, mật mã học bất đối xứng). System prompt đóng vai trò như một bộ lọc điều hướng ngữ cảnh (context conditioning), quyết định trực tiếp kho từ vựng, phong cách xưng hô và mức độ trừu tượng của câu trả lời trước khi mô hình bắt đầu sinh token đầu tiên.

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~100 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Vì sao tiếng Việt thường tốn
nhiều token hơn tiếng Anh cùng độ dài?**
> - **So sánh:** Với đoạn văn 100 từ tiếng Việt, công thức ước lượng `100 / 0.75 ≈ 133 token`, trong khi `tiktoken` đếm thực tế lên tới khoảng 175–200 token, chênh lệch khoảng **30% đến 50%**.
> - **Nguyên nhân:** Các bộ tokenizer như Byte-Pair Encoding (BPE) của OpenAI được huấn luyện chủ yếu trên kho dữ liệu tiếng Anh, nơi mỗi từ phổ biến thường chỉ tương ứng với 1 token. Trong khi đó, tiếng Việt có thanh điệu và nhiều ký tự có dấu (ă, â, đ, ê, ô, ơ, ư...), khi mã hóa theo chuẩn UTF-8 sẽ bị băm nhỏ thành nhiều sub-word hoặc byte lẻ, khiến một từ tiếng Việt thường tốn từ 2 đến 4 token.

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì
non-streaming lại phù hợp hơn?** (1 đoạn văn)
> Streaming quan trọng nhất trong các ứng dụng đối thoại trực tiếp thời gian thực (như giao diện chatbot web/mobile, terminal assistant, công cụ hỗ trợ viết code), nơi người dùng cần nhận được phản hồi ngay lập tức để giảm thời gian chờ đợi nhận thức (Time-to-First-Token < 1s). Ngược lại, non-streaming lại phù hợp hơn khi thực hiện các tác vụ xử lý hàng loạt chạy ngầm (batch jobs), các API backend server-to-server cần trích xuất toàn bộ dữ liệu có cấu trúc JSON để nạp vào cơ sở dữ liệu, hoặc các hệ thống bắt buộc phải kiểm duyệt nội dung (moderation filter) toàn diện trước khi hiển thị cho người dùng cuối.

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**So với delay cố định (ví dụ luôn chờ 1 giây), exponential backoff có lợi
thế gì khi API bị quá tải? Điều gì xảy ra nếu hàng nghìn client cùng retry
với delay cố định giống nhau?**
> Khi API bị nghẽn mạch, nguyên nhân thường do server bị quá tải tài nguyên. Nếu hàng nghìn client cùng retry với khoảng thời gian cố định (ví dụ 1 giây), tất cả sẽ cùng ồ ạt gửi lại request đúng vào giây tiếp theo, tạo nên hiện tượng "bão request đồng loạt" (Thundering Herd Problem) khiến máy chủ tiếp tục bị đánh sập theo chu kỳ. Exponential backoff giải quyết triệt để điều này bằng cách kéo giãn thời gian chờ theo cấp số nhân (0.1s -> 0.2s -> 0.4s -> 0.8s...), làm thưa dần mật độ gửi request, tạo khoảng thời gian cần thiết (breathing room) để server giải tỏa hàng đợi và tự phục hồi ổn định.

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Bạn chọn persona gì cho trợ lý của mình? Viết lại system prompt đó và giải
thích 1–2 lựa chọn từ ngữ quan trọng trong prompt (ví dụ: vì sao yêu cầu
"trả lời ngắn gọn", vì sao chỉ định ngôn ngữ...):**
> - **System prompt đã chọn:** `"Bạn là trợ giảng thân thiện của khóa học AI, giải thích các khái niệm kỹ thuật thật dễ hiểu bằng tiếng Việt và luôn trả lời ngắn gọn dưới 3 câu."`
> - **Giải thích từ ngữ quan trọng:**
>   1. *"bằng tiếng Việt"*: Cố định ngôn ngữ đầu ra, ngăn mô hình tự động chuyển sang tiếng Anh khi gặp các thuật ngữ lập trình chuyên ngành (như token, API, latency, streaming).
>   2. *"ngắn gọn dưới 3 câu"*: Giúp câu trả lời súc tích, tối ưu trải nghiệm đọc trên giao diện dòng lệnh (CLI) và tiết kiệm chi phí token cho mỗi lượt trao đổi.
>   3. *"trợ giảng thân thiện"*: Tạo ngữ điệu cởi mở, khuyến khích người học tiếp tục đặt câu hỏi đào sâu.

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn hiện có hạn chế lớn nhất là gì (ví dụ: history chỉ 3 lượt,
không có bộ nhớ dài hạn, không kiểm duyệt nội dung...)? Đề xuất một cải
thiện cụ thể và mô tả ngắn cách triển khai:**
> - **Hạn chế lớn nhất:** Chiến lược cắt tỉa cứng chỉ giữ 3 lượt gần nhất (`history[-6:]`) khiến trợ lý hoàn toàn quên mất các thông tin quan trọng được cung cấp từ đầu phiên (như tên người dùng, sở thích, hoặc đề bài cụ thể được thảo luận từ các lượt trước).
> - **Đề xuất cải thiện:** Triển khai **Bộ nhớ tóm tắt ngữ cảnh (Conversation Summary Buffer Memory)**.
> - **Cách triển khai:** Khi lịch sử vượt quá 3 lượt, thay vì xóa bỏ hoàn toàn các tin nhắn cũ, ta gọi một model phụ siêu nhẹ (`gpt-4o-mini`) để tóm tắt các lượt trao đổi cũ thành một đoạn văn ngắn cô đọng (ví dụ: *"Người dùng tên Hùng, đang học Part 2 về cách tính chi phí token..."*). Đoạn tóm tắt này sẽ được nhúng vào system prompt của các lượt chat tiếp theo, giúp mô hình luôn duy trì trí nhớ dài hạn xuyên suốt phiên mà không làm bùng nổ số lượng token.

---

## Danh Sách Kiểm Tra Nộp Bài

- [ ] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [ ] Cả 4 checkpoint pytest đều pass
- [ ] Tất cả 9 câu trong file này đã được trả lời
- [ ] Đã copy bài làm vào folder `solution/`, push lên fork và dán link trên trang bài Lab ở VLearn trước 23:59 ngày 11/09/2026
