# Introduction

# Show-o: ONE SINGLE TRANSFORMER TO UNIFY MULTIMODAL UNDERSTANDING AND GENERATION
## 离散去噪过程
### 扩散过程：
从初始图像 x_0 开始，通过每个时间步 t 使用过渡矩阵 transition_matrix[t] 对图像进行腐蚀（加噪）。
transition_matrix[t] 用于确定图像每个像素的状态更新（如变为噪声、保持不变或变为掩码）。
### 逆扩散过程：
从时间步 T 的噪声图像 x_T 开始，使用训练好的模型逐步去除噪声，恢复到接近原始图像的状态。模型通过学习如何从腐蚀后的图像预测出原始图像标记。文中提到，这个过程类似于MaskGIT中使用的掩码标记预测，通过逐步恢复被掩码的部分来完成图像重建。
### 扩散过程的分析
文中提到的离散去噪扩散与传统的连续扩散模型（如基于高斯噪声的扩散模型）有所不同。在离散扩散模型中，图像被表示为离散的标记（tokens），每个标记代表一个特定的图像像素状态。扩散过程通过一个过渡矩阵进行控制，这个矩阵定义了图像每个标记在不同时间步的状态转移概率。

此扩散过程的核心是通过离散状态转换实现图像标记的逐步腐蚀与恢复，从而生成或重建图像。

###  离散去噪过程的伪代码
```python
# 参数
T = 总时间步数  # 扩散的步数
K = 图像的token数量  # token的离散类别数
alpha_t = mask的概率  # 转换为[MASK]的概率
beta_t = 扩散概率  # 产生均匀噪声的概率

# 定义转移矩阵 Q_t
def transition_matrix(Q_t, alpha_t, beta_t, K):
    mask_token = K  # [MASK] token的索引
    I = identity_matrix(K + 1)  # 大小为(K+1)的单位矩阵
    em = one_hot(mask_token, K + 1)  # [MASK] token的one-hot向量

    # 定义 Qa 和 Qu 组成转移矩阵
    Qa = (1 - alpha_t) * I + alpha_t * em @ em.T
    Qu = I - beta_t * (I - em @ em.T) + (beta_t / (K + 1)) * (1 - em) @ (1 - em).T

    # 最终转移矩阵 Q_t
    return Qa @ Qu

# 前向扩散过程
def forward_process(x0, Q_t, T):
    xt = x0  # 初始图像的tokens
    for t in range(1, T + 1):
        xt = corrupt_tokens(xt, Q_t)  # 基于 Q_t 扰乱tokens
    return xt

# 逆向去噪过程
def reverse_process(xt_corrupted, Q_t, T):
    xt_denoised = xt_corrupted
    for t in range(T, 0, -1):
        xt_denoised = denoise(xt_denoised, Q_t)  # 逐步去噪
    return xt_denoised

# 训练目标：Mask Token 预测
def train_model(x0, T):
    Q_t = transition_matrix(alpha_t, beta_t, K)  # 获取转移矩阵
    xt_corrupted = forward_process(x0, Q_t, T)  # 扩散过程
    xt_predicted = reverse_process(xt_corrupted, Q_t, T)  # 预测并恢复

    # 计算原始图像和预测图像之间的损失
    loss = compute_loss(x0, xt_predicted)
    return loss

# 推理过程：生成图像
def generate_image(T):
    xt_noise = sample_random_noise()  # 采样随机噪声
    xt_denoised = reverse_process(xt_noise, Q_t, T)  # 去噪生成新图像
    return xt_denoised

``` 

### Qt 转移矩阵

#### 基本定义
Qt矩阵定义了扰动的转移概率，表示在时间步 𝑡中数据的扰动方式。Qt矩阵的公式如下：

$$
𝑄_𝑡 = 𝑄_𝑡^𝑎 * 𝑄_𝑡^𝑢​
$$
其中：𝑄_𝑡^𝑎  控制图像token是否被转为[MASK] token。
𝑄_𝑡^u 控制图像token的噪声扰动过程(以均匀概率转换其token的类别）。

$$
Q_t^a = (1 - \alpha_t) \cdot I + \alpha_t \cdot \mathbf{1} \cdot e_m^T
$$

- \( I \)：单位矩阵。
- \( \alpha_t \)：token 被 `[MASK]` 的概率。
- \( e_m \)：`[MASK]` token 的 one-hot 向量。
- \( \mathbf{1} \)：全1向量，表示其他 tokens 被转换为 `[MASK]` 的部分。


$$
Q_t^u = I - \beta_t \cdot (I - e_m e_m^T) + \frac{\beta_t}{K+1} \cdot (1 - e_m)(1 - e_m)^T
$$

- \( \beta_t \)：控制 token 被扰动为均匀噪声的概率。
- \( K \)：token 的类别数。
- \( e_m e_m^T \)：将 token 扩散为 `[MASK]`。

$$
Q_t = \begin{pmatrix}
\omega_t + \nu_t & \nu_t & \nu_t & \alpha_t \\
\nu_t & \omega_t + \nu_t & \nu_t & \alpha_t \\
\nu_t & \nu_t & \omega_t + \nu_t & \alpha_t \\
0 & 0 & 0 & 1
\end{pmatrix}
$$

其中：
- \( \omega_t = 1 - \alpha_t - \beta_t \)。
- \( \nu_t = \frac{\beta_t}{K + 1} \)。

- **\( \omega_t \)**: 表示 token 保持不变的概率。
- **\( \nu_t \)**: 表示 token 被均匀扰动的概率。
- **\( \alpha_t \)**: 表示 token 被转化为 `[MASK]` token 的概率。


#### Q t的学习与定义
在离散去噪扩散模型中，过渡矩阵通常是事先定义好的，不是通过学习获得的。具体来说：

**事先定义**：在模型设计阶段， Qt的结构由设计者指定，基于图像标记的腐蚀过程要求。这个矩阵的构造是根据扩散过程的需求，例如，我们希望在早期时间步较高概率地将标记替换为掩码 [MASK]，然后逐渐降低这种概率。
**概率控制**： 𝛼𝑡，𝛽𝑡是两个时间步相关的参数，定义了每个时间步中标记被掩码和扩散的概率。通常这些参数会随着时间变化，早期时间步中腐蚀强度较高（即较大概率替换为 [MASK]），而在后期时间步中，腐蚀变得较小，保留更多原始信息。

#### Qt的演变过程 
扩散过程通过 Qt矩阵定义了每个标记在时间步 t 时的状态变化。这个过程可以分为两部分：
- **前向扩散**：在每个时间步 𝑡，使用 𝑄𝑡对图像标记进行腐蚀，增加噪声，使得标记逐渐变得随机或被掩码。
在这个过程中，早期的 𝑄𝑡会有较高的 𝛼𝑡值，这意味着标记更有可能被替换为 [MASK] 标记。

- **反向扩散**：模型学习通过逆扩散过程逐步去除噪声，恢复原始图像。反向扩散过程中，模型通过对已掩码的标记进行预测，逐步恢复图像标记的真实值。

#### 示例：吸收-均匀扩散的解释
假设有一个图像被离散为若干个标记，每个标记可以是 1 到 𝐾之间的整数，并且我们有一个 [MASK] 状态。在某个时间步 𝑡假设 
𝛼𝑡 = 0.2； 𝛽𝑡 = 0.1； 标记 𝑥𝑡可以有以下几种可能的状态变化：
- 20% 概率被替换为 [MASK]；
- 10% 概率扩散到其他标记类别，均匀分布；
- 70% 概率保持不变。
这种设计使得模型在扩散过程中逐渐丧失原始图像信息，而在逆扩散过程中通过去噪逐渐恢复图像。