# 1. Giới thiệu chung
Linear Discriminant Analysis và Quadratic Discriminant Analysis là hai bộ phân loại cổ điển, với bề mặt quyết định tuyến tính và bậc hai tương ứng như tên gọi của chúng. 

Các bộ phân loại này hấp dẫn vì chúng có các giải pháp dạng đóng có thể dễ dàng tính toán, vốn có nhiều lớp, đã được chứng minh là hoạt động tốt trong thực tế và không có siêu tham số để điều chỉnh.

Phân tích Phân biệt Tuyến tính chỉ có thể học ranh giới tuyến tính, trong khi Phân tích Phân biệt Bậc hai có thể học ranh giới bậc hai và do đó linh hoạt hơn.
# 2. Linear Discriminant Analysis (LDA)
## 2.1 Đặt vấn đề:
PCA là phương pháp giảm chiều dữ liệu sao cho lượng thông tin về dữ liệu, thể hiện ở tổng phương sai, được giữ lại là nhiều nhất. Tuy nhiên, trong nhiều trường hợp, ta không cần giữ lại lượng thông tin lớn nhất mà chỉ cần giữ lại thông tin cần thiết cho riêng bài toán. Xét ví dụ về bài toán phân lớp với 2 classes được mô tả trong Hình 1.

![GitHub Logo](https://machinelearningcoban.com/assets/29_lda/lda.png)

**Trong hình trên:**
- Chiếu dữ liệu lên các đường thẳng khác nhau. 
- Có hai lớp dữ liệu minh hoạ bởi các điểm màu xanh và đỏ. 
- Dữ liệu được giảm số chiều về 1 bằng cách chiếu chúng lên các đường thẳng khác nhau d1 và d2. 
- Trong hai cách chiều này, phương của d1 gần giống với phương của thành phần chính thứ nhất của dữ liệu, phương của d2 gần với thành phần phụ của dữ liệu nếu dùng PCA. 
- Khi chiếu lên d1, các điểm màu đỏ và xanh bị chồng lấn lên nhau, khiến cho việc phân loại dữ liệu là không khả thi trên đường thẳng này. 
- Ngược lại, khi được chiếu lên d2, dữ liệu của hai class được chia thành các cụm tương ứng tách biệt nhau, khiến cho việc classification trở nên đơn giản hơn và hiệu quả hơn. 
- Các đường cong hình chuông thể hiện xấp xỉ phân bố xác suất của dữ liệu hình chiếu trong mỗi class.

**Nhận xét:**
- PCA (Principal Component Analysis): Là phương pháp giảm chiều không giám sát (unsupervised). Mục tiêu là tìm phương chiếu để giữ lại lượng phương sai (thông tin) lớn nhất của dữ liệu.

- Hạn chế của PCA trong Classification: Trong nhiều trường hợp, việc giữ lại phương sai lớn nhất (như chiếu lên $d_1$ trong Hình 1) lại khiến các điểm dữ liệu của các lớp khác nhau bị chồng lấn lên nhau, làm cho việc phân loại trở nên khó khăn hoặc không khả thi.

- Mục tiêu mới: Ta không cần giữ lại lượng thông tin lớn nhất, mà chỉ cần giữ lại thông tin cần thiết nhất cho bài toán phân lớp. Tức là, tìm phương chiếu ($d_2$ trong Hình 1) sao cho các lớp dữ liệu sau khi chiếu được tách biệt rõ ràng nhất.
## 2.2 Giới thiệu về Linear Discriminant Analysis:
-	Định nghĩa: LDA là một phương pháp giảm chiều dữ liệu có giám sát (supervised learning), sử dụng labels của dữ liệu.
-	Mục tiêu của LDA: Giảm số chiều dữ liệu sao cho khả năng phân lớp (discrimination) là hiệu quả nhất.
-	Tính chất:
    -  Discriminant: Tìm kiếm thông tin đặc trưng của mỗi lớp, giúp lớp này không bị lẫn với các lớp khác.
    -	 Linear: Phép giảm chiều được thực hiện thông qua một phép biến đổi tuyến tính (ma trận chiếu).
-	Ứng dụng:
    -  Là phương pháp giảm chiều dữ liệu (Dimensionality Reduction).
    -  Là phương pháp phân lớp (Classification).
    -  Có thể áp dụng đồng thời cả hai.
-	Giới hạn về số chiều mới: Số chiều tối đa của dữ liệu sau khi giảm là $C - 1$, trong đó $C$ là số lượng lớp (classes).
