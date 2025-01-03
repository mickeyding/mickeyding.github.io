# **JanusFlow 实现详解**

JanusFlow 是一个结合 **整流流（Rectified Flow）** 和 **任务解耦策略** 的多模态统一框架，以下是其实现的关键内容。

---

## **1. 架构设计**

### (1) **双任务解耦编码器**
- **理解编码器（Understanding Encoder, \(f_{enc}\)**）：  
  使用预训练视觉编码器（如 SigLIP-Large）提取高层语义特征，适合多模态理解任务。
  
- **生成编码器（Generation Encoder, \(g_{enc}\)**）：  
  使用轻量卷积模块（ConvNeXt Block），生成细粒度视觉信息，适合生成任务。

- **生成解码器（Generation Decoder, \(g_{dec}\)**）：  
  将生成编码器的特征映射回图像空间，通过像素上采样和线性变换生成高质量图像。

![image](https://github.com/user-attachments/assets/acb16891-aff8-44e8-a766-d2df6322f66a)

### (2) **整流流生成模块**
通过建模时间上的微分方程逐步将高斯噪声转换为目标数据分布：
- **时间步预测**：
  \[
  z_{t+\Delta t} = z_t + v(z_t, t) \cdot \Delta t
  \]

- **分类器自由引导（Classifier-Free Guidance, CFG）**：
  \[
  v(z_t, t) = w \cdot v(z_t, t | x_{con}) + (1-w) \cdot v(z_t, t | \emptyset)
  \]

### (3) **轻量化自回归语言模型**
- **多模态理解**：通过自回归生成下一个文本 token。
- **图像生成**：利用整流流生成图像特征，最后通过 VAE 解码回像素空间。

---

## **2. 训练流程**

### **阶段 1：随机初始化组件的适配**
- 训练随机初始化的模块（如线性变换层、生成编码器和解码器）。
- 冻结预训练的 LLM 和理解编码器。

### **阶段 2：联合预训练**
- 使用多模态数据进行预训练：
  - **多模态理解数据**：图像与文本对。
  - **图像生成数据**：文本到图像数据。
  - **纯文本数据**：增强语言能力。
- 数据比例动态调整：初期多用理解数据，后期增加生成数据比例。

### **阶段 3：监督微调**
- 解冻所有模块，使用高质量的指令微调数据。
- 数据包括：
  - **对话任务**：多轮多模态对话。
  - **生成任务**：高质量的文本到图像数据。

---

## **3. 训练目标**

### (1) **自回归损失（理解任务）**
\[
L_{AR}(\theta) = - \mathbb{E}_{x \sim D_{und}} \left[ \sum_{i=\ell_{con}}^{\ell-1} \log P_\theta(x_{i+1} | x_1, \ldots, x_i) \right]
\]

### (2) **整流流损失（生成任务）**
\[
L_{RF}(\theta) = \mathbb{E}_{x \sim D_{gen}, t \sim P(t), z_0 \sim \mathcal{N}(0, I)} \left[ \| v_\theta(z_t, t | x_{con}) - (x_{res} - z_0) \|^2 \right]
\]

### (3) **表示对齐损失**
\[
L_{REPA}(\theta, \phi) = - \mathbb{E}_{x \sim D_{gen}} \left[ \text{sim}(\text{stop\_grad}(f_{enc}(x_{res})), h_\phi(q_\theta(z_t))) \right]
\]
其中，\(\text{sim}(\cdot, \cdot)\) 表示余弦相似度。

### (4) **总损失**
理解任务使用：
\[
L_{AR}
\]
生成任务使用：
\[
L_{RF} + L_{REPA}
\]

---

## **4. 性能与创新**

1. **生成任务**：通过整流流实现高质量的图像生成，性能超越传统扩散模型。
2. **理解任务**：解耦策略避免任务冲突，显著提升了多模态理解性能。
3. **轻量化设计**：仅使用 1.3B 参数即可超越许多更大规模模型。

---
