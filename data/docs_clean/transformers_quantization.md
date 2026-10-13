# Overview

Quantization lowers the memory requirements of loading and using a model by storing the weights in a lower precision while trying to preserve as much accuracy as possible. Weights are typically stored in full-precision (fp32) floating point representations, but half-precision (fp16 or bf16) are increasingly popular data types given the large size of models today. Some quantization methods can reduce the precision even further to integer representations, like int8 or int4.

Transformers supports many quantization methods, each with their pros and cons, so you can pick the best one for your specific use case. Some methods require calibration for greater accuracy and extreme compression (1-2 bits), while other methods work out of the box with on-the-fly quantization.

Use the quantization picker to find a method for your model and hardware. Pick a model, your hardware, and what you need the method to do: load the model on the fly, fine-tune it with PEFT, compile it with `torch.compile()`, or save the quantized result. The picker shows whether the model fits in memory at each precision, including the KV cache, and the **Compare** view lists the methods that match.

Full compatibility table

| Quantization Method                       | On the fly quantization | CPU             | CUDA GPU | ROCm GPU  | Metal (Apple Silicon)              | Intel GPU       | Torch compile() | Bits         | PEFT Fine Tuning | Serializable with 🤗Transformers | 🤗Transformers Support  | Link to library                             |
|-------------------------------------------|----------------------|-----------------|----------|-----------|------------------------------------|-----------------|-----------------|--------------|------------------|-----------------------------|-------------------------|---------------------------------------------|
| AQLM                            | 🔴                   | 🟢              |     🟢     | 🔴        | 🔴                                 | 🟢              | 🟢              | 1/2          | 🟢               | 🟢                          | 🟢                      | https://github.com/Vahe1994/AQLM            |
| AutoRound                 | 🔴                   | 🟢               | 🟢          |   🔴        |   🔴                                |   🟢              |   🔴               | 2/3/4/8      |    🔴              |       🟢                      |    🟢                       |      https://github.com/intel/auto-round                                       |
| AWQ                              | 🔴                   | 🟢              | 🟢        | 🟢        | 🔴                                 | 🟢              | ?               | 4            | 🟢               | 🟢                          | 🟢                      | https://github.com/casper-hansen/AutoAWQ    |
| bitsandbytes            | 🟢                   | 🟢 |     🟢     | 🟡 | 🟡                    | 🟢 | 🟢 | 4/8          | 🟢               | 🟢                          | 🟢                      | https://github.com/bitsandbytes-foundation/bitsandbytes |
| compressed-tensors | 🔴                   | 🟢              |     🟢     | 🟢        | 🔴                                 | 🟢              | 🔴              | 1/8          | 🟢               | 🟢                          | 🟢                      | https://github.com/neuralmagic/compressed-tensors |
| EETQ                            | 🟢                   | 🔴              | 🟢        | 🔴        | 🔴                                 | 🔴              | ?               | 8            | 🟢               | 🟢                          | 🟢                      | https://github.com/NetEase-FuXi/EETQ        |
| Four Over Six            | 🟢                   | 🟢              | 🟢        | 🔴        | 🔴                                 | 🔴              | 🟢              | 4            | 🔴               | 🟢                          | 🟢                      | https://github.com/mit-han-lab/fouroversix |
| FP-Quant                          | 🟢                   | 🔴              | 🟢        | 🔴        | 🔴                                 | 🔴              | 🟢              | 4           | 🔴               | 🟢                          | 🟢                      | https://github.com/IST-DASLab/FP-Quant      |
| GGUF / GGML (llama.cpp)        | 🔴                   | 🟢              | 🟢        | 🔴        | 🟢                                 | 🟢              | 🟢              | 1/8          | 🔴               | 🔴                          | See Notes | https://github.com/ggerganov/llama.cpp      |
| GPT-QModel                     | 🔴                   | 🟢 | 🟢        | 🟢        | 🟢                                 | 🟢 | 🔴              | 2/3/4/8      | 🟢               | 🟢                          | 🟢                      | https://github.com/ModelCloud/GPTQModel        |
| HIGGS                          | 🟢                   | 🔴              | 🟢        | 🔴        | 🔴                                 | 🔴              | 🟢              | 2/4          | 🔴               | 🟢                          | 🟢                      | https://github.com/HanGuo97/flute           |
| HQQ                              | 🟢                   | 🟢              | 🟢        | 🔴        | 🔴                                 | 🟢              | 🟢              | 1/8          | 🟢               | 🔴                          | 🟢                      | https://github.com/mobiusml/hqq/            |
| Metal                          | 🟢                   | 🔴              | 🔴        | 🔴        | 🟢                                 | 🔴              | 🔴              | 2/4/8        | 🔴               | 🟢                          | 🟢                      | Hub Kernels |
| NVFP4                          | 🟢                   | 🔴              | 🟢        | 🔴        | 🔴                                 | 🔴              | 🟢              | 4            | 🔴               | 🔴                          | 🟢                      | Hub Kernels |
| optimum-quanto                | 🟢                   | 🟢              | 🟢        | 🔴        | 🟢                                 | 🟢              | 🟢              | 2/4/8        | 🔴               | 🔴                          | 🟢                      | https://github.com/huggingface/optimum-quanto       |
| SINQ                          | 🟢                     | 🟢 | 🟢      | 🟡      | 🟡                   | 🟡       | 🟡               | 2/3/4/6/8 | 🔴                | 🟢                               | 🟢                       | https://github.com/huawei-csl/SINQ                      |
| FBGEMM_FP8                | 🟢                   | 🔴              | 🟢        | 🔴        | 🔴                                 | 🔴              | 🔴              | 8            | 🔴               | 🟢                          | 🟢                      | https://github.com/pytorch/FBGEMM       |
| torchao                      | 🟢                   | 🟢               | 🟢        | 🔴        | 🟡 | 🟢              |                 | 4/8          |                  | 🟢🔴                        | 🟢                      | https://github.com/pytorch/ao       |
| VPTQ                            | 🔴                   | 🔴              |     🟢     | 🟡        | 🔴                                 | 🔴              | 🟢              | 1/8          | 🔴               | 🟢                          | 🟢                      | https://github.com/microsoft/VPTQ            |
| FINEGRAINED_FP8      | 🟢                   | 🔴              | 🟢        | 🔴        | 🔴                                 | 🟢              | 🔴              | 8            | 🔴               | 🟢                          | 🟢                      | Built-in |
| SpQR                            | 🔴                     |  🔴   | 🟢        | 🔴              |    🔴    | 🔴         |         🟢              | 3            |              🔴                     | 🟢           | 🟢                      | https://github.com/Vahe1994/SpQR/       |
| Quark                          | 🔴                     | 🟢 | 🟢      | 🟢      | 🟢                   | 🟢       | ?               | 2/4/6/8/9/16 | 🔴                | 🔴                               | 🟢                       | https://quark.docs.amd.com/latest/                      |

## Resources

If you are new to quantization, we recommend checking out these beginner-friendly quantization courses in collaboration with DeepLearning.AI.

* Quantization Fundamentals with Hugging Face
* Quantization in Depth

## User-Friendly Quantization Tools

If you are looking for a user-friendly quantization experience, you can use the following community spaces and notebooks:

* Bitsandbytes Space
* GGUF Space
* MLX Space
* AutoQuant Notebook
