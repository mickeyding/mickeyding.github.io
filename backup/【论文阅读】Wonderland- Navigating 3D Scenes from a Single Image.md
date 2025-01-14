
![image](https://github.com/user-attachments/assets/be7fb372-01ec-453f-9289-ebd2b91270af)

# 目标

这篇文章《Wonderland: Navigating 3D Scenes from a Single Image》提出了一种**基于单张图片的高效三维场景重建方法**，结合视频扩散模型和高斯点渲染技术，能够在无需逐场景优化的情况下快速生成高质量、宽范围的三维场景。

# 方法

## **基本方法概述**
1. **输入**：单张图片及其摄像机位姿信息。
2. **目标**：生成包含高几何一致性和高保真度的三维场景表示，并支持从任意视角生成2D图像。
3. **核心技术**：
   - **视频扩散模型（Video Diffusion Model）**：用于从单张图片生成多视角一致的潜空间特征。
   - **3D高斯点（3D Gaussian Splatting, 3DGS）**：用于表达三维场景，通过稀疏高斯点的集合描述场景的几何结构和外观。
   - **Latent Large Reconstruction Model（LaLRM）**：直接从潜特征生成三维高斯点。

---
![image](https://github.com/user-attachments/assets/faf3d1a5-67d3-487b-9912-462d10e0e3b1)

## **详细步骤**
### **1. 视频潜空间生成（Video Latent Generation）**
**目标**：将单张图片转化为包含多视角信息的紧凑视频潜特征（Video Latents）。

1. **视频扩散模型的使用**：
   - 通过预训练的**视频扩散模型**，从单张图片生成与摄像机轨迹相关的视频潜特征。
   - 视频扩散模型使用大规模视频数据进行训练，包含丰富的三维场景理解能力，因此能够生成具有3D一致性的视频潜空间。

2. **摄像机条件嵌入**：
   - 通过**Plücker坐标嵌入**表示摄像机的位姿信息，结合旋转矩阵、平移向量和内参矩阵，生成6维的空间-时间相机嵌入。
   - 设计了**双分支摄像头控制模块（Dual-branch Camera Conditioning）**，融合了ControlNet和LoRA技术：
     - **ControlNet分支**：保证对相机运动的精确控制。
     - **LoRA分支**：提供轻量化训练，适应静态场景。

3. **生成潜特征**：
   - 扩散模型通过输入的图片和相机轨迹生成一段潜视频，编码了场景的多视角信息（如几何、外观）。

---

### **2. 三维重建（Latent-to-3D Reconstruction）**
**目标**：从视频潜空间生成三维场景的稀疏表示——**3D高斯点（3DGS）**。

1. **潜特征离散化为Token**：
   - 将潜特征进行空间和时间维度上的分块（Patchify），生成潜特征Token。
   - 同时，将相机位姿嵌入生成相机Token。

2. **Token的融合与建模**：
   - 使用Transformer架构融合潜特征Token和相机Token，捕捉多视角之间的几何一致性。
   - Transformer输出3D高斯点的属性，包括：
     - **位置**：3D点的空间坐标。
     - **颜色**：每个点的RGB值。
     - **尺度**：点的大小及空间覆盖范围。
     - **旋转**：高斯点的方向，由四元数表示。
     - **透明度**：高斯点的透明程度。
     - **射线距离**：点与摄像机之间的距离（用于深度排序）。

3. **生成高斯点云**：
   - 输出结果是一组稀疏的3D高斯点（Gaussian Splatting），用于描述场景的几何结构和外观。

---

### **3. 渲染为2D图像**
**目标**：从生成的3D高斯点云中渲染出目标视角的图像。

1. **虚拟相机的设定**：
   - 使用输入的目标相机参数，设置虚拟相机的位姿和视角。

2. **高斯点的投影与体积渲染**：
   - 将3D高斯点投影到虚拟相机视角下的2D平面。
   - 使用体积渲染技术，将高斯点的颜色、透明度和深度信息叠加，生成目标图像。

3. **图像生成**：
   - 融合所有点的渲染结果，生成目标视角下的最终图像。

---

### **4. 训练优化**
为了提升生成质量，模型通过以下方式进行优化：

1. **损失函数**：
   - **均方误差（MSE）损失**：比较渲染图片与真实图片的像素差异。
   - **感知损失（Perceptual Loss）**：通过预训练的VGG网络，提取高层语义特征，保证感知一致性。
   - **正则化损失**：对高斯点的尺度和透明度进行正则化，避免生成的点云过于密集或离散。

2. **逐步训练策略**：
   - 初始阶段使用低分辨率数据训练，捕捉场景的基本几何结构。
   - 后期引入高分辨率数据（包括真实视频和扩散模型生成的视频）进行微调，提高生成的细节和场景范围。

---

## **创新与优势**
1. **效率高**：Feed-forward架构避免了逐场景优化，直接生成三维场景表示。
2. **一致性强**：利用视频潜空间和高斯点云的特性，保证多视角一致性。
3. **通用性强**：模型在大规模数据上预训练，能生成宽范围、复杂的三维场景。
4. **内存友好**：通过潜空间的压缩表示减少内存需求，同时加速推理。

--



# Q&A

1.那既然多视一致依赖于生成模型视频潜空间，那为什么不直接用生成模型生成多视角图片，而是要加上Feed-forward重建过程（并没有统一的场景3DGS的表示，所以加上重建这个阶段也很难保证多视一致性）？

---

## **1. 为什么不直接用生成模型生成多视角图片？**
直接使用视频生成模型生成多视角图片看似简单，但在实际场景中存在以下问题：

### **(1) 多视角一致性问题**
- **生成模型的限制**：
  视频生成模型（如扩散模型）生成的潜特征可以包含多视角信息，但在实际输出时，其多视角生成并不完全保证**三维一致性**。
  - 例如，从两个新视角生成的图像可能在重叠区域出现几何和纹理不一致（如物体位置偏移、形状扭曲、颜色不同）。
  - 这是因为扩散模型的核心是基于逐像素生成（2D纹理生成），而非对场景的显式三维理解。

- **无法对遮挡区域建模**：
  直接生成的新视角图片依赖于输入图片和潜特征的推断，而不是构建显式的三维结构。遮挡区域的生成可能缺乏几何逻辑一致性。

### **(2) 扩散模型的复杂性与高计算成本**
- **高计算成本**：
  视频扩散模型直接生成一组多视角图片时，需要在高维像素空间中进行大量采样，计算代价非常高。
  - 相比之下，生成一个潜特征（latent space）的成本要低得多。
  - Feed-forward过程利用潜特征直接重建3DGS，渲染新视角时效率更高。

- **存储成本**：
  视频扩散模型生成的是一组高分辨率的2D图片，而Feed-forward方法生成的是稀疏点云表示（3DGS），存储效率更高，适合更大范围的场景。

### **(3) 缺乏明确的三维场景表示**
- **生成模型无法提供显式三维结构**：
  直接生成多视角图片无法提取场景的几何信息和深度信息，这使得后续的视角调整、场景编辑、渲染变得非常困难。

---

## **2. 为什么要加入Feed-forward重建过程？**
通过引入Feed-forward 3D Reconstruction，文章解决了上述问题，并显著提升了多视图一致性和效率。其优势如下：

### **(1) 明确的三维表示**
- **3DGS作为三维表示**：
  Feed-forward重建过程中生成的3D Gaussian Splatting（3DGS）是一种显式的三维表示，能够同时描述几何、外观和透明度等属性。
  - 这为不同视角的渲染提供了清晰的几何参考，确保了遮挡关系和物体位置的一致性。
  - 相比直接生成图片，显式三维表示可以更好地对多视角重叠区域进行建模。

- **统一的生成管道**：
  虽然3DGS是为单张图片生成的，但由于共享的LaLRM（Latent Large Reconstruction Model）捕捉了多视角一致性，因此不同视角下的3DGS表示在渲染时能保持较高的一致性。
  
  **核心在于：多视生成视频是在隐空间学习一致性，表现在像素的一致性上，不一定是3D一致，但加上FeedForward重建，则是在3D空间学习一致性；**

### **(2) 高效的渲染过程**
- **稀疏点云的高效渲染**：
  3DGS是一种稀疏表示，渲染效率高，不需要像扩散模型那样逐像素生成高分辨率图片。
  - 渲染时只需要投影和混合稀疏高斯点即可，计算和存储需求显著降低。

- **支持动态视角调整**：
  有了3DGS表示，用户可以任意调整视角进行实时渲染，而无需为每个目标视角重新生成图片。

### **(3) 更好的遮挡区域推断**
- **结合生成先验和几何建模**：
  在Feed-forward过程中，模型利用视频潜空间中的生成先验结合3DGS的几何建模能力，能够更好地推断出遮挡区域和未见区域的合理几何与纹理。
  - 例如：在极端视角偏移时，Feed-forward生成的3DGS可以通过推断生成隐藏区域，而直接生成的新视角图片可能会丢失细节或逻辑不一致。

### **(4) 更适合大场景和多视图扩展**
- **分块式建模与渲染**：
  由于3DGS是稀疏的点云表示，可以分块生成和渲染，不需要一次性在像素空间生成整幅图像。
  - 这使得Feed-forward方法在处理大场景时更高效，同时适合多视图扩展。

---

## **3. Feed-forward方法如何解决多视一致性？**
虽然Feed-forward方法生成的是局部3DGS表示，但以下设计保证了跨视角的一致性：

### **(1) 基于共享模型的多视角一致性**
- LaLRM（Latent Large Reconstruction Model）是一个训练好的全局模型，对所有图片共享，其设计能够捕捉多视角一致性。
- 在训练时，通过多视角损失（如感知损失和像素级对比）优化模型，使其能够生成不同视角下几何一致的3DGS。

### **(2) 潜空间的一致性建模**
- 视频潜空间编码了多视角信息，Feed-forward过程以此为基础，生成的3DGS天然具有跨视角一致性。

### **(3) 渲染阶段的深度排序**
- 通过渲染阶段对3DGS高斯点进行深度排序和混合，确保遮挡关系正确，从而提升视觉一致性。

---

2.如何适应大场景呢？新视角有较大偏移，如何脑补出原图中不在的区域呢？

## **适应大场景的能力**
文中方法通过以下机制支持大场景：

### 1. **视频潜空间的“全局上下文”**
- **视频扩散模型**在生成视频潜特征时，会模拟摄像机轨迹，生成包含多个视角的潜视频。这意味着潜特征中编码了更广范围的场景信息，而不仅仅是原始图片的视角。
- 在推理阶段，虽然输入仅为单张图片，但潜特征中已经包含了对大场景的“隐式理解”。

### 2. **多视角一致性建模**
- 通过训练过程中的损失设计（例如感知损失和像素级重建损失），模型学习到在新视角下生成与输入一致的场景几何和外观。
- 这种一致性建模使得3DGS在大场景下能保持更高的全局几何一致性。

### 3. **高斯点云的稀疏表示**
- 3DGS表示通过稀疏点云（而非逐像素的表示）描述场景，大幅减少了对存储和计算的需求，使得模型可以处理更大范围的场景。
- 点云的稀疏性和可扩展性为大场景表示提供了天然优势。

---

## **新视角大偏移的渲染**
当目标视角相较原始图片偏移较大时（例如视角旋转超过90度或观察完全遮挡的区域），模型需要“脑补”出输入图片中没有的信息。文章中的方法通过以下机制解决这一问题：

### 1. **基于生成的“新视角信息”**
- 视频扩散模型训练时会模拟摄像机轨迹生成潜特征，因此它能够在潜空间中编码从未观察到区域的可能几何和纹理信息。
- 在推理阶段，目标视角的信息是基于这些潜特征生成的，而不是完全依赖输入图片。

### 2. **使用大规模多视角数据训练**
- 视频扩散模型在大规模视频数据集上训练，包含了丰富的三维场景先验知识。这种训练使得模型对大偏移的新视角具有较强的泛化能力。
- 即使输入图片中没有的区域，模型可以根据训练中学到的全局场景先验进行“合理推测”。

### 3. **渐进式训练策略**
- 在训练阶段，模型逐步扩展到更高分辨率的场景表示，同时在真实视频数据和合成视频数据上训练。
- 通过这种渐进式训练策略，模型可以更好地适应“未见区域”的重建和新视角生成。

---

## **文章能否完全解决大视角偏移和新区域生成的问题？**
文章提供了较好的方法来**缓解**这一问题，但可能还无法完全解决，尤其是以下场景：

1. **极端偏移视角**：
   - 如果目标视角与输入图片相差太大，且没有任何先验几何信息可参考（例如输入图片是正面视角，目标视角完全是背面），模型可能会生成缺乏细节或不完全正确的结果。
   - 此时的生成完全依赖于训练中学到的先验，结果可能带有一定的不确定性。

2. **高复杂度大场景**：
   - 对于高度复杂的场景（如遮挡多、纹理丰富），由于视频潜空间的压缩性，可能会丢失部分细节，从而影响新视角的生成效果。

3. **训练数据依赖**：
   - 模型的生成能力高度依赖于训练数据的多样性。如果训练集中缺乏某些类型的场景或大视角偏移的数据，生成的结果可能会受到影响。

---

**总结**：
这篇文章提出的方法可以**在一定程度上解决新视角大偏移问题**，通过视频潜空间捕捉多视角一致性和利用生成先验，能够脑补出输入图片中未观察到的区域。对于大场景，该方法通过稀疏的3DGS表示和渐进式训练策略提升适应能力。然而，对于极端偏移视角和高度复杂场景，可能仍存在一定的挑战。