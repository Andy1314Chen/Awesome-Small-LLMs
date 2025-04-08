# 🚀 Awesome Small LLMs (<3B) 

A curated list of high-performance, efficient small language models (≤3B parameters), including language models and multimodal models.

## Table of Contents

* [Language Models](#language-models)
* [Multimodal Models](#multimodal-models)
* [Contributing](#contributing)
* [Related Awesome Repositories](#Related-Awesome-Repositories)
* [License](#license)

## Language Models (<3B)

* SmolLM ([SmolLM](https://huggingface.co/HuggingFaceTB/SmolLM-1.7B), [SmolLM2](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B))
    - Paper/Blog: https://huggingface.co/blog/smollm, https://simonwillison.net/2024/Nov/2/smollm2
    - Size: 135M, 360M, 1.7B
    - Summary: SmolLM models come in sizes of 135M, 360M, and 1.7B parameters, trained on a high-quality dataset called SmolLM-Corpus, which includes educational web pages and synthetic textbooks. These models are noted for their ability to run locally on various devices, offering a balance between size and performance, and outperforming other models in their respective categories. SmolLM2, an updated version, continues this trend with similar model sizes and is trained on an even larger dataset, including new mathematics and coding datasets, making it suitable for on-device deployment. (Hugging Face)

* Phi ([Phi-1](https://huggingface.co/microsoft/phi-1), [Phi-1.5](https://huggingface.co/microsoft/phi-1_5), [Phi-2](https://huggingface.co/microsoft/phi-2))
    - Paper/Blog: https://arxiv.org/pdf/2309.05463
    - Size: 1.3B, 2.7B
    - Summary: The Phi-1, Phi-1.5, and Phi-2 models are a series of Transformer-based language models developed by Microsoft. Phi-1 is specialized for basic Python coding with 1.3 billion parameters, focusing on generating code. Phi-1.5 also has 1.3 billion parameters and is designed for broader language tasks like QA, chat, and code generation, showing nearly state-of-the-art performance in its class. Phi-2 is larger with 2.7 billion parameters, offering enhanced capabilities in common sense, language understanding, and logical reasoning, while being suitable for QA, chat, and code formats. (Microsoft)

* Qwen ([Qwen-1_8B](https://huggingface.co/Qwen/Qwen-1_8B), [Qwen1.5-0.5B](https://huggingface.co/Qwen/Qwen1.5-0.5B), [Qwen1.5-1.8B](https://huggingface.co/Qwen/Qwen1.5-1.8B), [Qwen2-0.5B](https://huggingface.co/Qwen/Qwen2-0.5B), [Qwen2-1.5B](https://huggingface.co/Qwen/Qwen2-1.5B), [Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B), [Qwen2.5-1.5B](https://huggingface.co/Qwen/Qwen2.5-1.5B), [Qwen2.5-3B](https://huggingface.co/Qwen/Qwen2.5-3B))
    - Paper/Blog: https://qwenlm.github.io/blog/qwen2.5/
    - Size: 0.5B, 1.5B, 1.8B, 3B
    - Summary: The Qwen series includes several models developed by Alibaba Cloud, each with different parameter sizes and versions. Qwen-1.8B is a 1.8 billion parameter model known for its low-cost deployment and high-quality training data, covering multiple languages and tasks like common sense, reasoning, and code generation. The Qwen1.5 and Qwen2 series offer models with varying sizes (e.g., 0.5B, 1.8B) and are designed for efficiency and performance across different tasks. The Qwen2.5 series further expands this range with models up to 3 billion parameters, enhancing capabilities in language understanding and generation. These models are optimized for multilingual support and efficient deployment, making them versatile tools for various applications.

* MobileLLM ([MobileLLM-125M](https://huggingface.co/facebook/MobileLLM-125M), [MobileLLM-350M](https://huggingface.co/facebook/MobileLLM-350M), [MobileLLM-600M](https://huggingface.co/facebook/MobileLLM-600M), [MobileLLM-1B](https://huggingface.co/facebook/MobileLLM-1B), [MobileLLM-1.5B](https://huggingface.co/facebook/MobileLLM-1.5B))
    - Paper/Blog: 
    - Size: 125M, 350M, 600M, 1B, 1.5B
    - Summary: The MobileLLM series addresses the need for efficient, on-device large language models (LLMs) with fewer than a billion parameters. The paper finds that model architecture is critical for sub-billion scale LLMs and introduces MobileLLM, which uses deep and thin architectures, embedding sharing, and grouped-query attention. MobileLLM outperforms existing 125M/350M models and performs comparably to the much larger LLaMA-v2 7B in API calling tasks. (FaceBook)

* Gemma ([Gemma-2B](https://huggingface.co/google/gemma-2b), [Gemma2-2B](https://huggingface.co/google/gemma-2-2b), [Gemma3-1B](https://huggingface.co/google/gemma-3-1b-it))
    - Paper/Blog: https://ai.google.dev/gemma/docs/core/model_card_2, https://huggingface.co/google/gemma-3-1b-it
    - Size: 1B, 2B
    - Summary: The Gemma family from Google consists of lightweight, open-source models built using the same technology as the Gemini models. These models are designed to be efficient and can be deployed on devices with limited resources, democratizing access to advanced AI. They support multilingual capabilities and are suitable for a variety of tasks like text generation, image understanding, question answering, and summarization. They are trained on diverse datasets including web documents, code, mathematics, and images. (Google)

* Llama3.2 ([Llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B), [Llama-3.2-3B](https://huggingface.co/meta-llama/Llama-3.2-3B))
    - Paper/Blog: https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/
    - Size: 1B, 3B
    - Summary: Llama 3.2 introduces 1B and 3B parameter models designed for on-device applications. These lightweight, text-only models enable developers to build personalized agentic applications with strong privacy, processing data locally without sending it to the cloud. They are highly capable in multilingual text generation and tool calling. These models were created using pruning and knowledge distillation techniques, leveraging larger Llama 3.1 models to improve performance. They support a context length of 128K tokens and have undergone extensive post-training alignment, including supervised fine-tuning, rejection sampling, and direct preference optimization, to ensure high quality across multiple capabilities like summarization, instruction following, and tool use. (FaceBook)

* [TinyLlama](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)
    - Paper/Blog: https://github.com/jzhang38/TinyLlama
    - Size: 1.1B
    - Summary: The TinyLlama project aims to pretrain a 1.1B Llama model on 3 trillion tokens. They adopted exactly the same architecture and tokenizer as Llama 2. This means TinyLlama can be plugged and played in many open-source projects built upon Llama. Besides, TinyLlama is compact with only 1.1B parameters. This compactness allows it to cater to a multitude of applications demanding a restricted computation and memory footprint. (TinyLlama)

## Multimodal Models  (<3B)

* [TinyLLaVA](https://huggingface.co/tinyllava/TinyLLaVA-Phi-2-SigLIP-3.1B)(TinyLLaVA-1.5B, TinyLLaVA-2.0B, TinyLLaVA-3.1B)
    - Paper/Blog: https://arxiv.org/abs/2402.14289
    - Size: 1.5B, 2.0B, 3.1B
    - Summary: TinyLLaVA has released a family of small-scale Large Multimodel Models(LMMs), ranging from 1.4B to 3.1B. Our best model, TinyLLaVA-Phi-2-SigLIP-3.1B, achieves better overall performance against existing 7B models such as LLaVA-1.5 and Qwen-VL.

* [Qwen2-VL-2B](https://huggingface.co/Qwen/Qwen2-VL-2B)
    - Paper/Blog: https://qwenlm.github.io/blog/qwen2-vl
    - Size: 2B
    - Summary: Qwen2-VL is a cutting-edge vision language model that excels in image and video understanding. It supports multiple languages and offers three versions: Qwen2-VL-2B, Qwen2-VL-7B, and Qwen2-VL-72B. The model enhances object recognition, handwritten text interpretation, and mathematical problem-solving. It is particularly effective in document understanding and outperforms several other models in various tasks. The 2B version is optimized for mobile devices, while the 72B version showcases exceptional performance.

* [Qwen2.5-VL-3B](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)
    - Paper/Blog: https://qwenlm.github.io/blog/qwen2.5-vl
    - Size: 3B
    - Summary: Qwen2.5-VL is Qwen's new flagship vision-language model, improving significantly upon Qwen2-VL. Key features include enhanced visual understanding for objects, texts, charts, and layouts, as well as the ability to act as a visual agent for computer and phone use. It can comprehend videos longer than 1 hour and pinpoint relevant segments. Qwen2.5-VL can accurately localize objects and generate structured outputs for documents like invoices and forms. The model comes in 3B, 7B, and 72B sizes, with the 72B version achieving competitive performance in various benchmarks, especially in document and diagram understanding. Smaller models like the 7B outperform GPT-4o-mini in several tasks, while the 3B is designed for edge AI applications.

* Mini-InternVL ([Mini-InternVL-2B-1.5](https://huggingface.co/OpenGVLab/Mini-InternVL-Chat-2B-V1-5))
    - Paper/Blog: https://internvl.github.io/blog/2024-05-25-Mini-InternVL-1.5
    - Size: 2.2B
    - Summary: 

* InternVL ([InternVL2-1B](https://huggingface.co/OpenGVLab/InternVL2-1B), [InternVL2-2B](https://huggingface.co/OpenGVLab/InternVL2-2B), [InternVL2_5-1B](https://huggingface.co/OpenGVLab/InternVL2_5-1B), [InternVL2_5-2B](https://huggingface.co/OpenGVLab/InternVL2_5-2B))
    - Paper/Blog: https://internvl.github.io/blog/2024-07-02-InternVL-2.0, https://internvl.github.io/blog/2024-12-05-InternVL-2.5
    - Size: 0.9B, 2.2B
    - Summary: 

* DeepSeek-VL ([DeepSeek-VL-1.3B](https://huggingface.co/deepseek-ai/deepseek-vl-1.3b-chat), [DeepSeek-VL2-Tiny](https://huggingface.co/deepseek-ai/deepseek-vl2-tiny), [DeepSeek-VL2-Small](https://huggingface.co/deepseek-ai/deepseek-vl2-small))
    - Paper/Blog: https://arxiv.org/abs/2403.05525, https://arxiv.org/abs/2412.10302
    - Size: 1.0B, 2.0B, 2.8B
    - Summary: 

* MobileVLM ([MobileVLM-1.7B](https://huggingface.co/mtgv/MobileVLM-1.7B), [MobileVLM-3B](https://huggingface.co/mtgv/MobileVLM-3B), [MobileVLM_V2-1.7B](https://huggingface.co/mtgv/MobileVLM_V2-1.7B), [MobileVLM_V2-3B](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct))
    - Paper/Blog: https://arxiv.org/abs/2312.16886, https://arxiv.org/abs/2402.03766
    - Size: 1.7B, 3.0B
    - Summary: 

* [PaliGemma-3B](https://huggingface.co/google/paligemma-3b-pt-224)
    - Paper/Blog: https://ai.google.dev/gemma/docs/paligemma
    - Size: 3B
    - Summary:     

## Contributing

1. Fork the repository
2. Add new models via Pull Request (include: name, link, and key specs)
3. Verify benchmarks with reproducible code

## Related Awesome Repositories

If you want to read more about related topics, here are some tangential awesome repositories to visit:

* [NexaAI/Awesome-LLMs-on-device](https://github.com/NexaAI/Awesome-LLMs-on-device) on LLMs on Device
* [stevelaskaridis/awesome-mobile-llm](https://github.com/stevelaskaridis/awesome-mobile-llm) on Mobile Large Language Models
* [csarron/awesome-emdl](https://github.com/csarron/awesome-emdl) on Embedded and Mobile Deep Learning

## License

Apache License 2.0. See [LICENSE](LICENSE).
