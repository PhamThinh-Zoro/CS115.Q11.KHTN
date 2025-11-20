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
## 2.3 Linear Discriminant Analysis cho bài toán 2 lớp:
### a) Các tiêu chuẩn khi phân tách:
- Within-class variances: **Phương sai nhỏ** thể hiện việc dữ liệu ít bị phân tán. Điều này có nghĩa là dữ liệu trong mỗi class có xu hướng giống nhau. Được ký hiệu là $s^2$
- Between-class variances: **Khoảng cách giữa các kỳ vọng lớn** chứng tỏ rằng hai classes nằm xa nhau, tức dữ liệu giữa các classes là khác nhau nhiều. Được tính bằng phép tính bình phương khoảng cách giữa 2 kỳ vọng $(m_1 - m_2)^2$

Hai classes được gọi là discriminative nếu hai class đó cách xa nhau và dữ liệu trong mỗi class có xu hướng giống nhau . Nói cách khác thì between-class variance lớn và within-class variance nhỏ. Linear Discriminant Analysis là thuật toán đi tìm một phép chiếu sao cho tỉ lệ giữa between-class variance và within-class variance lớn nhất có thể.
### b) Xây dựng hàm mục tiêu:
Giả sử rằng bài toán yêu cầu phân loại 2 lớp.

Ta có n điểm dữ liệu được gắn nhãn (lớp 1 hoặc lớp 2). Mỗi điểm dữ liệu được mô tả bằng một vector $x_i(1\le i\le n)$

Mỗi điểm dữ liệu được chiếu lên không gian mới bởi vector đích w.

$$y_i = w^Tx_i, 1\le i\le n$$

Do bài toán có 2 lớp nên số chiều được giảm xuống còn 1.

Vector kỳ vọng của class 1 và class 2:

$$m_1 = \frac{1}{N_1} \sum_{i \in C_1} x_i $$

$$m_2 = \frac{1}{N_2} \sum_{j \in C_2} x_j $$

Giá trị kỳ vọng sau khi giảm chiều:

$$e_1 = \frac{1}{N_1} \sum_{i \in C_1} y_i = w^T m_1$$

$$e_2 = \frac{1}{N_2} \sum_{j \in C_2} y_j = w^T m_2$$

$$\Rightarrow (e_1 - e_2) = w^T(m_1 - m_2)$$

Within-class variance:

$$s_1^2 = \sum_{i \in C_1} (y_i - e_1)^2 $$
$$s_2^2 = \sum_{j \in C_2} (y_j - e_2)^2 $$

$$J(w) = \frac{(e_1 - e_2)^2}{s_1^2 + s_2^2}$$

LDA là thuật toán tìm giá trị lớn nhất của hàm $J(w)$

$$(e_1 - e_2)^2 = w^T(m_1 - m_2)(m_1 - m_2)^T w = w^T S_B w$$

Với $S_B=(m_1 - m_2)(m_1 - m_2)^T$ là một ma trận đối xứng nửa xác định dương. Còn được gọi là between-class covariance matrix
$s_1^2 + s_2^2 = \sum_{k=1}^{2}\sum_{i=1}^{N_k} (y_i - e_k)^2$

mà $(y_i - e_k)^2 = (w^T (x_i - m_k))^2$

$$
\begin{aligned}
\Rightarrow s_1^2 + s_2^2 &= \sum_{k=1}^{2}\sum_{i \in C_k} (w^T (x_i - m_k))^2  \\
&= w^T \sum_{k=1}^{2}\sum_{i \in C_k} (x_i - m_k)(x_i - m_k)^T w \\
&= w^T S_W w
\end{aligned}
$$

Với $S_W = \sum_{k=1}^{2}\sum_{i \in C_k} (x_i - m_k)(x_i - m_k)^T $

$S_W$ cũng là một ma trận đối xứng nửa xác định dương. Còn được gọi là within-class covariance matrix.

$$\Rightarrow J(w) = \frac{w^T S_B w}{w^T S_W w} $$
### c) Nghiệm tối ưu của bài toán:
Nghiệm $w$ để $J(w)$ đạt giá trị lớn nhất chính là nghiệm w của phương trình

$$\frac{\partial J(\mathbf{W})}{\partial \mathbf{W}} = 0 $$

$$
\begin{aligned}
\Leftrightarrow \mathbf{S}_B \mathbf{w} &= \frac{\mathbf{w}^T \mathbf{S}_B \mathbf{w}}{\mathbf{w}^T \mathbf{S}_W \mathbf{w}} \mathbf{S}_W \mathbf{w} \\
\Leftrightarrow \mathbf{S}_W^{-1} \mathbf{S}_B \mathbf{w} &= J(\mathbf{w}) \mathbf{w} \\
\text{Với } \mathbf{A} &= \mathbf{S}_W^{-1} \mathbf{S}_B \text{ và } \lambda = J(\mathbf{w}), \text{ ta có: } \mathbf{A} \mathbf{w} = \lambda \mathbf{w} \\
\text{Thay } \mathbf{S}_B &= (\mathbf{m}_1 - \mathbf{m}_2)(\mathbf{m}_1 - \mathbf{m}_2)^T \text{ (cho 2 lớp) được: } \\
\mathbf{S}_W^{-1} (\mathbf{m}_1 - \mathbf{m}_2)(\mathbf{m}_1 - \mathbf{m}_2)^T \mathbf{w} &= \lambda \mathbf{w} \\
\text{Đặt } k &= (\mathbf{m}_1 - \mathbf{m}_2)^T \mathbf{w} \text{ (là một số vô hướng)} \\
\Rightarrow w &= \frac{k}{\lambda} {S_W}^{-1} (m_1 - m_2)\\
\Rightarrow \mathbf{w} &\propto \mathbf{S}_W^{-1} (\mathbf{m}_1 - \mathbf{m}_2)
\end{aligned}
$$
## 2.4 Linear Discriminant Analysis cho bài toán nhiều lớp
### a) Xây dựng hàm thành phần:
Giả sử rằng chiều mà chúng ta muốn giảm về là D′<D và dữ liệu mới ứng với mỗi điểm dữ liệu x là:

**$$ y=W^T x $$**

với $W^T \in R^ {D \times D'}$

Ma trận dữ liệu trong không gian mới có D' chiều:

$$Y_k = W^T X_k $$

Với $X_k$ là ma trận dữ liệu trong không gian ban đầu

Vector kỳ vọng của class k trong không gian ban đầu với số chiều D:

$$m_k = \frac{1}{N_k} \sum_{i \in C_k} x_i $$

Vector kỳ vọng của class k trong không gian mới sau khi giảm chiều còn D':

$$e_k = \frac{1}{N_k} \sum_{i \in C_k} y_i = W^T m_k $$

### b) Xây dựng hàm mất mát
Within_class variance:

${\sigma_k}^2 = \sum_{i \in C_k} \lVert y_n - e_k\rVert_2^2 $

$=\lVert Y_k - E_k \rVert_2^2 $

$=\lVert W^T(X_k - M_k) \rVert_2^2 $

$=trace(W^T(X_k - M_k)(X_k - M_k)^T W) $

Với $E_k$ là một ma trận có các cột giống hệt nhau và bằng với vector kỳ vọng $e_k$. Có thể thấy $E_k=W^T M_k$ với $M_k$ là ma trận có các cột giống hệt nhau và bằng với vector kỳ vọng $m_k$ trong không gian ban đầu.

Đại lượng đo within-class trong multi-class LDA có thể được đo bằng:

$$s_W = \sum_{k=1}^{C}{\sigma_k}^2 = \sum_{k=1}^{C} trace(W^T(X_k - M_k)(X_k - M_k)^T W) \\
= trace(W^T S_W W)
$$

với:

$$
S_W = \sum_{k=1}^{C}(X_k - M_k)(X_k - M_k)^T = \sum_{k=1}^{C}\sum_{n \in C_k} (x_n - m_k)(x_n - m_k)^T
$$

$S_W$ có thể được coi là **within-class covariance matrix của  multi-class LDA.** Ma trận $S_W$ này là một ma trận nửa xác định dương.

Tương tự, between-class variance được tính theo công thức:

$$
s_B = \sum_{k=1}^{C}N_k \lVert e_k - e\rVert_2^2 = \sum_{k=1}^{C}\lVert E_k - E\rVert_2^2
$$

$N_k$ làm trọng số vì có những class có nhiều phần tử so với các classes còn lại.

Ma trận E và $E_k$ có số cột biến thiên và bằng với $N_k$.

Tương tự như within-class variance:

$$
s_B = trace(W^T S_B W)
$$

với:

$$
S_B = \sum_{k=1}^{C}(M_k - M)(M_k - M)^T = \sum_{k=1}^{C}N_k (m_k - m)(m_k - m)^T
$$

Số cột của **M** cũng linh hoạt theo số cột của **$M_k** và bằng với $N_k$

### c) Tìm nghiệm tối ưu:

Nghiệm tối ưu cho bài toán LDA multi-class là tìm W sao cho tỉ lệ giữa $s_B$ và $s_W$ lớn nhất

$$
W = \underset{W}{\text{arg max}} \ J(W) = \underset{W}{\text{arg max}} \ \frac{\text{trace}(W^T S_B W)}{\text{trace}(W^T S_W W)}
$$

Điều đó có nghĩa là tìm nghiệm W cho phương trình

$$\frac{\partial J(\mathbf{W})}{\partial \mathbf{W}} = 0 $$

$$
\Leftrightarrow \frac{2S_B W trace(W^T S_W W) - trace(W^T S_B W) 2 S_W W}{(trace(W^T S_W W))^2} = 0 
$$

$$
\Leftrightarrow {S_W}^{-1} S_B W = J W
$$

Ma trận W tối ưu chính là ma trận chứa các vector riêng của ${S_W}^{-1} S_B$ tương ứng với các giá trị riêng lớn nhất J

Số lượng các vector độc lập tuyến tính ứng với 1 trị riêng chính là rank của không gian riêng ứng với trị riêng đó, và không được vượt quá C-1, với C là số lớp của bài toán.

# 3. Quadratic Discriminant Analysis:
