# StableDiffusion3Pipeline
## 步骤和算法
**输入检查：**
调用 [check_inputs]方法，验证输入参数的有效性。如果输入参数不符合要求，则抛出相应的错误。
**设置内部参数：**
设置 [guidance_scale]\、[clip_skip]和 [joint_attention_kwargs]等内部参数。
**定义调用参数：**
根据 [prompt]的类型（字符串或列表），定义调用参数。
**准备潜变量：**
调用 [prepare_latents] 方法，准备潜变量（latents），如果未提供，则生成随机潜变量。
**编码提示：**
调用 [encode_prompt]方法，编码文本提示（prompt）和负面提示（negative_prompt），生成相应的嵌入（embeds）。
**设置时间步长：**
调用 [retrieve_timesteps]方法，从调度器中获取时间步长（timesteps）。
**生成图像：**
通过循环迭代时间步长，逐步去噪潜变量，生成图像。
在每一步中，调用调度器的 [step]方法，更新潜变量。
**后处理：**
调用 [vae.decode]方法，将潜变量解码为图像。
根据 [output_type]参数，返回图像的不同格式（如 PIL 图像或 NumPy 数组）。

```python
def __call__(self, ...):
    # 1. Check inputs
    self.check_inputs(...)

    # 2. Set internal parameters
    self._guidance_scale = guidance_scale
    self._clip_skip = clip_skip
    self._joint_attention_kwargs = joint_attention_kwargs
    self._interrupt = False

    # 3. Define call parameters
    if isinstance(prompt, str):
        ...
    elif isinstance(prompt, list):
        ...
    else:
        ...

    device = self._execution_device

    # 4. Prepare latents
    latents = self.prepare_latents(...)

    # 5. Encode prompt
    prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = self.encode_prompt(...)

    # 6. Retrieve timesteps
    timesteps, num_inference_steps = retrieve_timesteps(...)

    # 7. Generate images
    for t in timesteps:
        latents = scheduler.step(...)

    # 8. Post-process
    images = vae.decode(latents)

    # 9. Return results
    if return_dict:
        return StableDiffusion3PipelineOutput(images=images)
    else:
        return images
``` 

# SD3Transformer2DModel

# FlowMatchEulerDiscreteScheduler
## scale_noise
前向加噪过程,遵循rectified flow前向过程; 
``` math
 z_{t} = t * \epsilon + (1.0 - t ) * z_{t - 1} 
``` 

``` Python
sample = sigma * noise + (1.0 - sigma) * sample
``` 

## step
反向去噪过程
``` Python
prev_sample = sample + (sigma_next - sigma) * model_output
``` 
``` math
 z_{t -1} = z_t + dt * s_{\theta}(x_t, t) ;

 \frac{d x_t}{dt} = s_{\theta}(x_t, t)
``` 


