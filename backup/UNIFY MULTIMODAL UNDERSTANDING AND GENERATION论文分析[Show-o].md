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

## Qt 转移矩阵

### 基本定义
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
- \( $\alpha_t$ \)：token 被 `[MASK]` 的概率。
- \( $e_m$ \)：`[MASK]` token 的 one-hot 向量。
- \( $\mathbf{1}$ \)：全1向量，表示其他 tokens 被转换为 `[MASK]` 的部分。


$$
Q_t^u = I - \beta_t \cdot (I - e_m e_m^T) + \frac{\beta_t}{K+1} \cdot (1 - e_m)(1 - e_m)^T
$$

- \( $\beta_t$ \)：控制 token 被扰动为均匀噪声的概率。
- \( K \)：token 的类别数。
- \( $e_m e_m^T$ \)：将 token 扩散为 `[MASK]`。

$$
Q_t = \begin{pmatrix}
\omega_t + \nu_t & \nu_t & \nu_t & \alpha_t \\
\nu_t & \omega_t + \nu_t & \nu_t & \alpha_t \\
\nu_t & \nu_t & \omega_t + \nu_t & \alpha_t \\
0 & 0 & 0 & 1
\end{pmatrix}
$$

其中：
- \( $\omega_t = 1 - \alpha_t - \beta_t $ \)。
- \( $\nu_t = \frac{\beta_t}{K + 1} $ \)。

- **\( $\omega_t$ \)**: 表示 token 保持不变的概率。
- **\( $\nu_t$ \)**: 表示 token 被均匀扰动的概率。
- **\( $\alpha_t$ \)**: 表示 token 被转化为 `[MASK]` token 的概率。


### Qt的学习
在离散去噪扩散模型中，过渡矩阵通常是事先定义好的，不是通过学习获得的。具体来说：

**事先定义**：在模型设计阶段， Qt的结构由设计者指定，基于图像标记的腐蚀过程要求。这个矩阵的构造是根据扩散过程的需求，例如，我们希望在早期时间步较高概率地将标记替换为掩码 [MASK]，然后逐渐降低这种概率。
**概率控制**： 𝛼𝑡，𝛽𝑡是两个时间步相关的参数，定义了每个时间步中标记被掩码和扩散的概率。通常这些参数会随着时间变化，早期时间步中腐蚀强度较高（即较大概率替换为 [MASK]），而在后期时间步中，腐蚀变得较小，保留更多原始信息。

### Qt的演变过程 
扩散过程通过 Qt矩阵定义了每个标记在时间步 t 时的状态变化。这个过程可以分为两部分：
- **前向扩散**：在每个时间步 𝑡，使用 𝑄𝑡对图像标记进行腐蚀，增加噪声，使得标记逐渐变得随机或被掩码。
在这个过程中，早期的 𝑄𝑡会有较高的 𝛼𝑡值，这意味着标记更有可能被替换为 [MASK] 标记。

- **反向扩散**：模型学习通过逆扩散过程逐步去除噪声，恢复原始图像。反向扩散过程中，模型通过对已掩码的标记进行预测，逐步恢复图像标记的真实值。

### 示例：吸收-均匀扩散的解释
假设有一个图像被离散为若干个标记，每个标记可以是 1 到 𝐾之间的整数，并且我们有一个 [MASK] 状态。在某个时间步 𝑡假设 𝛼𝑡 = 0.2； 𝛽𝑡 = 0.1； 标记 𝑥𝑡可以有以下几种可能的状态变化：
- 20% 概率被替换为 [MASK]；
- 10% 概率扩散到其他标记类别，均匀分布；
- 70% 概率保持不变。
这种设计使得模型在扩散过程中逐渐丧失原始图像信息，而在逆扩散过程中通过去噪逐渐恢复图像。

## 网络学习的目标

show-o的反向去噪过程参考了maskgit文章的实现，先来看基本的maskgit是如何生成图片，再来看show-o中网络学习的目标是什么？以及从理论推导出发，如何保证该去噪过程可以做适当简化，最终和maskgit一致；

 ### MaskGIT

**MaskGIT 学习目标**：

在 MaskGIT 中，网络的学习目标是通过从时间步 \(x_t\)（即包含一定比例掩码的图像）预测出原始图像的标记 \(x_0\)。MaskGIT 模型学习的是如何在给定部分掩码信息的情况下，利用上下文恢复被掩码的图像部分。因此，**每次预测的目标确实是原始图像 \(x_0\)**，但这种恢复是基于部分标记已经被填充的条件下进行的。

MaskGIT 采用的是 **Masked Visual Token Modeling (MVTM)**，这一策略与 BERT 中的 masked language modeling (MLM) 类似。具体来说，在训练过程中，随机选择一部分图像标记并替换为特殊的 `[MASK]` 标记，模型的任务是利用其余未掩码的部分来恢复这些掩码标记；

在推理时，MaskGIT 并不是一次性预测整个图像，而是采用 **迭代解码**（iterative decoding），即每次只填充一部分最有信心的标记，其余部分仍保持掩码，直到所有标记都被填充完毕。

**监督机制：交叉熵损失**

在训练过程中，MaskGIT 使用 **交叉熵损失（Cross Entropy Loss）** 来监督模型的预测结果。具体地说，模型会根据已知的部分图像标记预测被掩码的标记，而交叉熵损失度量的是模型预测出的标记分布与真实标记 \(x_0\) 之间的差异。公式为：

$$ L_{\text{mask}} = -\sum_{i \in \text{masked}} \log p_\theta(x_0^i | x_t) $$

其中：
- \( $x_0^i $\) 是第 \(i\) 个被掩码的标记的真实类别（即原始图像中的标记）。
- \( $p_\theta(x_0^i | x_t) $\) 是模型预测该标记属于真实类别的概率分布。

**与 Show-O 文章的联系**:

从 Show-O 文章的简化扩散模型推导来看，MaskGIT 的方法与其密切相关，特别是关于如何从扩散后的噪声图像 \(x_t\) 直接预测 \(x_0\)。Show-O 的模型通过引入 `[MASK]` 标记并简化状态转移，使得模型在每一步扩散过程中都只需考虑两种情况：**保持不变或被掩码**。而 MaskGIT 的设计恰好也采用了类似的 `[MASK]` 标记策略，通过交叉熵损失对掩码部分进行监督，最终恢复出 \(x_0\)。在这种设置下，无论输入的图像包含什么样的噪声或被掩码的部分，**输出的目标始终是恢复出原始图像 \(x_0\)**。两者是一致的；

### Qt矩阵简化：简化状态转移

文中公式 (8) 给出的是variational lower bound：

$$ \[ \mathbb{E}_{q(x_0)} [\log p_\theta(x_0)] \geq \mathbb{E}_{q(x_0)}[-L_{\text{ELBO}}(x_0, \theta)] \geq \sum_{t=1}^{T}\mathbb{E}_{q(x_0) q(x_t | x_0)} [\log p_\theta(x_0 | x_t)] + C \] $$

$$ \mathbb{E}_{q(x_0)} [\log p_\theta(x_0)] \geq \mathbb{E}_{q(x_0)}[-L_{\text{ELBO}}(x_0, \theta)] \geq \sum_{t=1}^{T} \mathbb{E}_{q(x_0) q(x_t | x_0)} [\log p_\theta(x_0 | x_t)] + C $$

$$
\mathbb{E}_{q(x_0)} [\log p_\theta(x_0)] \geq \mathbb{E}_{q(x_0)}[-L_{\text{ELBO}}(x_0, \theta)] \geq \sum_{t=1}^{T} \mathbb{E}_{q(x_0) q(x_t | x_0)} [\log p_\theta(x_0 | x_t)] + C
$$

其中：
- \(  $p_\theta(x_0 | x_t) $ \) 是从时间步 \( t \) 恢复到初始图像 \( $x_0$ \) 的概率，描述模型的去噪能力；
- \( $q(x_t | x_0) $ \) 是前向扩散过程中从 \( $x_0$ \) 到 \( $x_t$ \) 的概率分布；
- \( C \) 是与模型无关的常数项。


**简化结论的推导动机**:

从公式 (8) 的推导可以看出，模型的目标是通过最大化去噪恢复概率 \( $p_\theta(x_0 | x_t)$ \) 来优化去噪过程。在去噪任务中，模型需要预测每个时间步 \( t \) 的图像如何恢复原始状态 \( $x_0$ \)。因此，为了简化这种预测任务，最大化$p_\theta(x_0 | x_t)$前提下, 我们可以减少模型在每个时间步中需要处理的标记转移情况，从而提出“标记要么保持不变，要么变为 `[MASK]`”的简化策略。

**为什么简化为“保持不变或替换为 `[MASK]`”**

1. **减少状态空间**：如果我们允许标记可以转移到多个不同的类别，那么模型需要学习如何从多个噪声干扰的类别中恢复出原始标记。这会显著增加模型的复杂性，导致较大的计算开销。通过将状态转移简化为“保持不变或变为 `[MASK]`”，我们大幅减少了状态空间，模型只需关注恢复被掩码的部分即可。

2. **优化计算效率**：通过限制标记状态的变化，KL 散度项的计算也被简化，因为模型只需考虑两种状态（保持不变或被掩码）。这样，公式 (8) 中的损失项可以在优化中变得更加简洁，从而提升计算效率。

3. **去噪任务的集中化**：简化后，模型的去噪任务集中于恢复 `[MASK]` 部分。这与公式 (8) 的推导结果一致：模型的目标是从噪声干扰的图像 \( $x_t$ \) 中最大化恢复原始图像 \( $x_0$ \) 的概率。通过限制标记转移为两种状态，去噪过程可以更有效地进行，模型无需处理复杂的类别转移。



