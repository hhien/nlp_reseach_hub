
**CPU (central processing unit)**
- quản lý và điều phối các tác vụ giwa các thành phần: đmả bảo GPU, RAM và storage hoạt động cùng nhau
- data preprocessing
- xử lý tuần tự, CPU hiện đại như intel Xeon, AMD EPYC có thể hỗ trợ tốt cho ìnference với mô hình nhỏ
- k phù hợp cho các tác vụ yêu cầu xử lý song song. e.g: training

**GPU (graphcs processing unit):**
- giống đội ngũ công nhân đông đảo và chuyên nghiệp
- ưu điểm:
  + xử lý song song mạnh mẽ
  + tối ưu hóa cho phép toán ma trận
  + hỗ trợ phần mềm phong phú: tensorflow, pytorch,.. đều được tối ưu hóa cho GPU --> k cần viết code để tận dụng sức mạnh của GPU
- nhược điểm:
  + chi phí cao
  + tiêu thụ năng lượng lớn --> bắt buộc yêu cầu làm mát tốt
- các dòng GPU phổ biến cho AI: 
GPU Model	VRAM	CUDA Cores	TDP	Phù hợp cho
NVIDIA RTX 5090	32 GB	21,760	575W	Research, Small training
NVIDIA A100	40/80 GB	6,912	250W	Enterprise training
NVIDIA H100	80 GB	16,896	700W	Large-scale training
AMD MI250X	128 GB	8,192	560W	HPC, Large models

**TPU (Tensor processing unit):**
- giống đội chuyên gia được đào tạo đặc biệt cho 1 công việc cụ thể --> tối ưu hóa hoàn toàn cho các tác vụ AI
- ưu điểm:
    + hiệu suất cao cho tensor operations: nhân ma trận, convolution
    + tiết kiệm năng lượng so với GPU
    + tích hợp với TensorFlow
- nhược điểm:
    + khả năng tiếp cận: TPU chủ yếu chỉ có sẵn qua Google Cloud Plàtorm
    + thiếu tính linh hoạt: TPU ít linh hoạt hơn GPU trong việc chạy các loại workload khác

**FPGA (fiel-programmarble gate array): là mạch tích hợp có thể được lập trình lại sau khi sản xuất, cho phép tùy chỉnh phần cứng theo nhu cầu cụ thể**
- ưu điểm;
    + tùy chỉnh cao
    + độ trễ thấp --> phù hợp cho các ứng dụng real-time
    + hiệu quả năng lượng tốt
- nhược điểm:
    + khó lập trình: y/c kiến thức về HDL: hardware description language
    + thời gian phát triển dài cho quá trình thiết kế và tối ưu hóa

**ASIC (application-specific integrated circuit): là mạch tích hợp được thiết kế riêng cho 1 ứng dụng cụ thể**
ví dụ: Google TPU, intel habana gaudi, Cerebras wafer-scale engine (chip AI lớn nhất tg)
- ưu điểm:
    + hiệu suất cao nhất
    + tiêu thụ năng lượng thấp
- nhược điểm:
    + chi phí phát triển cao, chỉ phù hợp với quy mô lớn
 
**NPU (Neural processing Unit): là bộ xử lý chuyên dụng cho neural networks, thường được tích hợp trong mobile, edge devices**
- phù hợp cho inference trên thiết bị di động
- vi dụ: apple neural engine, qualcomm hexagon NPU, huawei ascend NPU


<img width="1024" height="559" alt="image" src="https://github.com/user-attachments/assets/727805fb-a0f8-4060-a067-9a706eb89d83" />

<img width="1024" height="559" alt="image" src="https://github.com/user-attachments/assets/0c7db1e5-0d5f-45c9-a9a9-d3ec9012f73e" />

khuyến nghị phần cứng theo quy mô dự án:
Quy mô	Training	Inference	Ngân sách ước tính
Cá nhân/Nghiên cứu	RTX 5090 (1-2x)	RTX 5090	2,000 - 4,000 USD
Startup nhỏ	A100 (4-8x)	A100 hoặc T4	50,000 - 150,000 USD
Doanh nghiệp	H100 (8-32x)	A100 cluster	500,000 - 2M+ USD
Hyperscale	Custom ASIC/TPU	Distributed TPU	10M+ USD

- Yêu cầu về bộ nhớ và lưu trữ
    + RAM: training: cần ít nhất 64G RAM, khuyến nghị 128GB+ cho mô hình lớn
    + inference: 32-64GB
- VRAM (GPU memory):
    + 32GB (RTX 5090) train mô hình nhỏ hơn 10B parameters với batch size vừa phải--> phù hợp cho hầu hết các dự án nghiên cứu và phát triển
    + 40GB (A100) : 13B parameters: mô hình lớn
    + công thức tính: VRAM ≈ (Model Parameters × 4 bytes) + (Batch Size × Sequence Length × Hidden Size × 4 bytes) + Overhead
<img width="2816" height="1536" alt="image" src="https://github.com/user-attachments/assets/e09533a7-0b47-4f2e-80d0-a3c39f1abf31" />
cấu trúc hệ thống lưu trữ cho AI

**- có hướng dẫn thiết lập từng bước để kết nối vscode dùng GPU của google colab**
Lưu ý để tận dụng tối đa tài nguyên google colab và tránh mất dữ liệu:
- Lưu dữ liệu trên drive:
    from google.colab import drive
    drive.mount('/content/drive')

    #Lưu model checkpoints
  
    model.save('/content/drive/MyDrive/models/my_model.h5')
- Sử dụng checkpoint thường xuyên:
    ```#Lưu checkpoint mỗi epoch
  
    checkpoint_path = '/content/drive/MyDrive/checkpoints/epoch_{epoch}.ckpt'
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path, 
        save_weights_only=True,
        period=1
    )
- Tối ưu hóa sử dụng VRAM: sử dụng mixed precision training
    #Sử dụng mixed precision training, đặt đoạn code này trước khi định nghĩa mô hình
  
    ```from tensorflow.keras.mixed_precision import set_global_policy
    set_global_policy('mixed_float16')
- Giữ phiên hoạt động: chạy 1 cell nhỏ định kỳ. ví dụ tạo 1 cell và chạy nó mỗi 10-15p:
    ```import time
    from datetime import datetime
    
    while True:
        print("Colab vẫn đang hoạt động | Thời gian hiện tại:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        time.sleep(600)  # 600 giây = 10 phút
    
  lưu ý: Nếu đang train model, training itself đã đủ activity
        Cell này phù hợp khi: Chờ download, Chờ human-in-the-loop, Giữ session để debug

**Giải pháp cho inference production:**
- GPT ìnference server
- TPU inference (Google cloud)
- Edge AI với NPU

