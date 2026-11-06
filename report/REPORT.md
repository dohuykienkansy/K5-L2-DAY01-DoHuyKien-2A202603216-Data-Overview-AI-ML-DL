# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 11/6/2026

**Runtime Colab:** GPU

**Python / PyTorch / Ultralytics:** Python 3.13.16 / PyTorch 2.11.0+cu130 / Ultralytics 8.4.145

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`=468 ; `class_name`="cab" ; `rank`=1 ; `score`=0.510915 ; `taxonomy_name`="ImageNet-1K"
2. Record này nói về cả bức ảnh hay về một vật thể riêng trong ảnh? Vì sao bạn nghĩ vậy?
   Record này mô tả toàn bộ bức ảnh chứ không phải một vật thể riêng lẻ. Vì tác vụ Image Classification chỉ đưa ra các lớp được xếp hạng cho cả hình ảnh, không cung cấp bounding box hoặc polygon để xác định vị trí từng vật thể.
3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?
   Danh sách tên lớp được xác định sẵn bởi taxonomy đi kèm checkpoint. Trong bài thực hành này, mô hình phân loại sử dụng taxonomy ImageNet-1K với các lớp đã được định nghĩa trước.
4. Vì sao cần ghi cả class_id, class_name và taxonomy_name, không chỉ ghi mỗi tên lớp?
   Cần lưu cả ba trường để xác định nhãn chính xác và tránh nhầm lẫn. class_id là mã định danh của lớp, class_name là tên dễ đọc của lớp, taxonomy_name cho biết lớp đó thuộc hệ thống phân loại nào. Việc lưu đầy đủ giúp dữ liệu nhất quán và dễ đối chiếu giữa các hệ thống.
5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn một lớp?
   Guideline cần quy định rõ tiêu chí chọn lớp chính, chẳng hạn ưu tiên chủ thể nổi bật nhất, đối tượng chiếm diện tích lớn nhất hoặc nội dung chính của bức ảnh. Đồng thời cần có quy tắc xử lý khi ảnh chứa nhiều chủ đề hoặc không đủ thông tin để xác định lớp.
6. Vì sao không được lấy score của model làm ground truth?
   Vì Model score chỉ thể hiện mức độ tự tin tương đối của mô hình đối với prediction, không chứng minh prediction đó là chính xác. Ground truth phải do annotator tạo ra dựa trên ảnh thực tế và guideline đã thống nhất, sau đó được reviewer kiểm tra chất lượng.

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`="person" ; `score`=0.912625 ; `bbox_xyxy`=[385.33, 69.24, 498.92, 348.92] ; `bbox_width`=113.58 ; `bbox_height`=279.68
2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?

- x_min = 385.33: tọa độ X của góc trên bên trái.
- y_min = 69.24: tọa độ Y của góc trên bên trái.
- x_max = 498.92: tọa độ X của góc dưới bên phải.
- y_max = 348.92: tọa độ Y của góc dưới bên phải.
  Chiều rộng bounding box là 113.58 pixels và chiều cao là 279.68 pixels.

3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?
   Khi confidence threshold = 0.20, mô hình phát hiện được 17 vật thể.
   Khi confidence threshold = 0.60, mô hình phát hiện được 6 vật thể.

4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người reviewer phải làm thêm việc gì?
   Khi giảm confidence threshold, mô hình chấp nhận nhiều prediction có confidence score thấp hơn, do đó có thể phát hiện được nhiều vật thể hơn. Tuy nhiên, điều này cũng làm tăng nguy cơ nhận diện nhầm vật thể.
   Ngược lại, khi tăng confidence threshold, những prediction có confidence thấp sẽ bị loại bỏ. Điều này giúp hạn chế các kết quả thiếu chắc chắn nhưng cũng có thể khiến mô hình bỏ sót một số vật thể thật.
   Ví dụ trên ảnh kitchen, số lượng prediction giảm từ 17 xuống 6 khi threshold tăng từ 0.20 lên 0.60.
   Vì vậy, annotator cần kiểm tra hình ảnh thực tế, xác minh nhãn và vị trí bounding box thay vì chỉ dựa vào confidence score của mô hình.
5. Viết một quy tắc box để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).
   Khi tạo ground truth bounding box, annotator cần vẽ hình chữ nhật sát với phạm vi hiển thị thực tế của vật thể, hạn chế lấy thêm phần background hoặc các vật thể xung quanh.
   Mỗi vật thể riêng biệt cần có một bounding box tương ứng và được gán đúng class theo taxonomy đã thống nhất.
   Đối với vật thể bị che khuất một phần hoặc bị cắt ở mép ảnh, annotator cần tuân theo quy tắc thống nhất về việc chỉ đánh dấu phần nhìn thấy hay ước lượng toàn bộ vật thể. Không nên để mỗi annotator tự đưa ra cách xử lý khác nhau.
   Các trường hợp không thể xác định rõ ranh giới vật thể cần được đánh dấu để người kiểm duyệt xem xét.
6. Trên ảnh kitchen, chỉ ra một vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?
   Trên ảnh kitchen, em quan sát thấy một phần người xuất hiện ở sát mép trái ảnh và bị cắt khỏi khung hình.
   Em đề xuất xây dựng guideline thống nhất cho trường hợp này: annotator chỉ vẽ bounding box bao quanh phần vật thể thực sự nhìn thấy trong ảnh, không tự ý ước lượng phần nằm ngoài khung hình.
   Cần áp dụng cùng một quy tắc cho tất cả annotator để đảm bảo bounding box nhất quán và chất lượng ground truth. Nếu dataset có quy định riêng về vật thể bị cắt khỏi ảnh, annotator cần tuân thủ quy định đó.

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`="kitchen-003" ; `class_name`="bowl" ; `score`=0.698698 ; số điểm polygon=`polygon_point_count`=70 ; 3–5 điểm đầu của `polygon_xy`=[84.0, 289.0], [83.0, 290.0], [81.0, 290.0]
2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?
   Bounding box chỉ xác định vùng hình chữ nhật bao quanh vật thể thông qua bốn tọa độ góc, nên không thể mô tả chính xác hình dạng và đường biên thực tế của vật thể.
   Trong khi đó, polygon sử dụng nhiều điểm tọa độ [x, y] để mô tả đường viền của từng đối tượng. Nhờ vậy, polygon có thể bám sát hình dạng thực tế của vật thể hơn, phân biệt phần thuộc về vật thể và phần background xung quanh.
3. instance_id dùng để làm gì? Nó không phải class ID và không phải tracking ID — bạn hiểu câu này thế nào?
   instance_id là mã định danh riêng cho từng đối tượng cụ thể trong kết quả dự đoán của một ảnh. Nó giúp phân biệt các đối tượng khác nhau, kể cả khi chúng thuộc cùng một class.
   Ví dụ, trong ảnh kitchen có thể xuất hiện nhiều chiếc bát (bowl). Mặc dù chúng có cùng class_id = 45 và class_name = "bowl", mỗi chiếc bát vẫn có instance_id riêng như kitchen-001, kitchen-002 hoặc kitchen-003.
   instance_id không phải class ID vì nó không đại diện cho loại đối tượng. Class ID mới là mã dùng để xác định lớp, ví dụ 45 tương ứng với bowl trong COCO-80.
   instance_id cũng không phải tracking ID vì nó không theo dõi cùng một đối tượng xuyên suốt nhiều khung hình hoặc video. Đây chỉ là mã định danh trong output của bài lab, được tạo ra để phân biệt các instance trong kết quả hiện tại.
4. Viết một quy tắc vẽ biên mask để người gán nhãn làm theo.
   Annotator cần vẽ đường biên mask bám sát đường viền thực tế của từng vật thể, chỉ bao gồm những pixel thuộc về vật thể và không lấy thêm phần background xung quanh.
   Mỗi đối tượng riêng biệt phải có một mask riêng, kể cả khi nhiều đối tượng có cùng class. Đối với các vùng biên không rõ ràng, cần tuân thủ guideline thống nhất và không tự ý mở rộng hoặc thu hẹp mask theo cảm tính.
5. Trên ảnh kitchen, chỉ ra một chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?
   Trên ảnh kitchen, em quan sát thấy chiếc bát ở khu vực phía trước, bên trái bàn bếp tiếp xúc với mặt bàn. Đây là trường hợp hai vật thể nằm sát nhau, khiến việc xác định chính xác đường biên giữa mask của bowl và dining table có thể gặp khó khăn.
   Em chọn escalation vì tại những vị trí đường tiếp xúc bị che khuất hoặc không rõ ràng, annotator có thể xác định ranh giới khác nhau. Cần yêu cầu Lab Coach hoặc người phụ trách làm rõ quy tắc phân định pixel thuộc về từng vật thể trước khi tiếp tục gán nhãn.
   Điều này giúp đảm bảo mask của từng instance nhất quán và hạn chế sự khác biệt giữa các annotator.

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ                | Ground truth trông như thế nào? (đơn vị nhãn)                                                              | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON                                                                                                                                                                           | Annotator làm gì?                                                                                 | Reviewer xem gì?                                                                                                        |
| --------------------- | ---------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| Phân loại ảnh         | Một nhãn lớp (class label) đại diện cho nội dung chính của toàn bộ ảnh.                                    | Ảnh có nhiều đối tượng hoặc nhiều nội dung khác nhau có thể gây khó khăn khi xác định chủ thể chính và chọn nhãn phù hợp.                                                                                                    | Đọc guideline, xác định chủ thể chính và gán nhãn theo taxonomy thống nhất.                       | Kiểm tra nhãn có phù hợp với nội dung ảnh, đúng taxonomy và nhất quán với guideline hay không.                          |
| Phát hiện vật thể     | Mỗi vật thể có một class label và một bounding box gồm 4 tọa độ pixel [x_min, y_min, x_max, y_max].        | Ảnh kitchen có người xuất hiện một phần ở mép trái. Ngoài ra, kết quả JSON tại threshold 0.35 có 11 prediction, trong khi threshold 0.20 có 17 prediction. Điều này cho thấy cần kiểm tra các prediction có confidence thấp. | Xác định từng vật thể, gán đúng class và vẽ bounding box sát phạm vi vật thể nhìn thấy.           | Kiểm tra nhãn, vị trí và kích thước bounding box; phát hiện vật thể bị bỏ sót, nhận diện nhầm hoặc box không chính xác. |
| Instance segmentation | Mỗi vật thể có một class label và một mask riêng, được biểu diễn bằng polygon gồm nhiều điểm tọa độ pixel. | Trong ảnh kitchen, mask của bowl và dining table tiếp giáp nhau, khiến việc xác định ranh giới giữa hai vật thể có thể khó khăn. Record kitchen-003 có 70 điểm polygon.                                                      | Vẽ mask theo đường biên thực tế của từng instance; không gộp các vật thể cùng lớp thành một mask. | Kiểm tra mask có bám sát đường biên, đúng vật thể, không lấn sang background hoặc mask của đối tượng khác hay không.    |

## 5. An toàn dữ liệu

1. Em chỉ sử dụng dữ liệu được cung cấp chính thức cho bài lab, không tự ý chia sẻ, tải lên nền tảng bên ngoài, tiết lộ thông tin nhạy cảm hoặc sử dụng dữ liệu sai mục đích.
2. Em sẽ dừng xử lý dữ liệu đó, không tự ý chỉnh sửa, chia sẻ hoặc tiếp tục sử dụng. Sau đó, em sẽ báo cho Lab Coach hoặc giảng viên phụ trách để được hướng dẫn cách xử lý phù hợp.

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [ ] `REPORT.md` (file này đã điền)
- [ ] `day1_lab_outputs/classification_predictions.json`
- [ ] `day1_lab_outputs/detection_predictions.json`
- [ ] `day1_lab_outputs/segmentation_predictions.json`
- [ ] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [ ] `day1_lab_outputs/visuals/classification_top5.png`
- [ ] `day1_lab_outputs/visuals/detection_predictions.png`
- [ ] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`
- [ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [ ] Đã nộp **link repository** trên LMS
