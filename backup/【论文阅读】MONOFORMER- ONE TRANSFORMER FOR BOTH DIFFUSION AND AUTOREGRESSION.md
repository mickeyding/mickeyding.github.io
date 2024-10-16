MonoFormer 是一种多模态生成模型，基本想法和Transfusion类似，它通过共享一个Transformer模型实现了文本自回归和图像扩散两种生成任务。MonoFormer提出了一种统一的框架，使同一个Transformer架构同时用于多种模态的生成任务。

# 与Transfusion的比较总结
- 1.与Transfusion高度类似，都是共享Transformer模型，都是同时使用因果attention和双向attention完成文本自回归任务和图像扩散任务，都是连续特征，都是文本和图像共享特征空间
- 2.MonoFormer 整体而言没有Transfusion的效果好，表现在：
     -  a.模型scale up的能力（MonoFormer 1B vs Transfusion 7B）
     -  b.多模态任务质量（MonoFormer 部分指标下降，文章提到We believe the performance can be further improved when more language data are incorporated in training，且图像生成质量没有和SD系列模型比较）
     -  c.图像编辑能力（Transfusion展现了文本控制图像编辑，MonoFormer没有）

以下是MonoFormer的多模态实现方法的详细介绍：

# 1. 模型架构设计
MonoFormer的核心思想是使用单一Transformer架构同时处理自回归（文本生成）和扩散（图像生成）两种不同任务。在现有的多模态模型中，通常会使用两个不同的网络，一个用于离散数据（例如文本）的自回归建模，另一个用于连续数据（例如图像）的扩散建模。而MonoFormer则通过共享Transformer模型来同时处理离散（文本）和连续（图像）数据，减少了模型的复杂性。

- 自回归（Autoregression）部分：自回归模型用于文本生成，采用因果注意力（Causal Attention）机制来确保每个文本token只能看到它之前的token，从而保证顺序生成。这与经典的语言模型（如GPT）类似。

- 扩散（Diffusion）部分：扩散模型用于图像生成，采用双向注意力（Bidirectional Attention），允许每个图像patch可以与同一图像中的其他patch相互作用。这与图像生成中的扩散模型类似（如DDPM）。

# 2. 多模态处理方法
MonoFormer的关键创新在于如何在同一个Transformer中处理离散的文本生成和连续的图像生成。这种多模态处理的实现方式包括以下几个方面：

## a. 不同的注意力机制
MonoFormer针对文本和图像任务使用不同的注意力机制：

- 文本任务：文本生成任务使用因果注意力掩码（Causal Attention Mask），这确保了生成下一个文本token时，当前token只能关注它之前的token，而不能看到未来的token。这样保证了文本生成过程是自回归的，即每次生成一个token。

- 图像任务：图像生成任务使用双向注意力掩码（Bidirectional Attention Mask），允许图像的所有patch相互之间进行交互。与文本生成不同，图像生成不需要因果性，因此双向注意力可以提高图像生成的质量和效率。

## b. 输入处理
MonoFormer接收的输入既可以是文本token序列，也可以是图像的嵌入。具体来说：

- 文本输入：文本生成任务的输入是文本的token嵌入，Transformer在自回归模式下生成下一个token的嵌入，最终将这些嵌入解码成具体的文本输出。
- 图像输入：图像生成任务则是通过扩散模型的方式进行，输入是加入噪声的图像latent表示，模型的任务是逐步去除这些噪声，生成高质量的图像。

## c. 共享Transformer权重
MonoFormer的Transformer结构是共享的，即文本生成和图像生成任务共用同一组Transformer参数。Transformer通过不同的注意力掩码来区分文本和图像任务，因此可以同时处理两种模态的数据。

# 3. 训练过程
MonoFormer的训练过程分为两个部分：文本生成任务使用自回归训练，图像生成任务使用扩散训练。

- 文本生成的损失：文本生成部分的损失是标准的自回归负对数似然损失（Negative Log-Likelihood Loss）。模型根据已经生成的文本token预测下一个token。

- 图像生成的损失：图像生成部分的损失是扩散模型常用的L2损失（均方误差损失）。模型通过逐步去噪的方式还原图像，损失衡量的是模型预测的噪声与实际噪声之间的差异。

# 4. 推理过程
在推理过程中，MonoFormer可以根据输入的不同模态灵活切换生成模式：

- 文本生成：当输入为文本时，模型按照自回归的方式逐步生成文本，每次预测下一个token。
- 图像生成：当开始图像生成时，模型按照扩散模型的推理过程，输入随机噪声，并通过多次迭代生成高质量图像。

# 5. 实验与性能
实验结果表明，MonoFormer在生成图像和文本的任务上均取得了接近于当前最先进方法的表现。在图像生成方面，MonoFormer在ImageNet 256x256上的FID为2.57，接近最优的扩散模型（DiT XL/2的FID为2.27）。在文本生成方面，它保持了预训练语言模型的能力，并在多个文本生成任务上表现良好。

