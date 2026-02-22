# Retrieval Results

**Query:** attention mechanism transformer
**Run:** 2026-02-21 17:48:06
**Pipeline:** L1 BM25+GIST→RRF | L2 BM25-triplet+Dense-centroid→RRF | L3 ColBERT+Cross-Encoder+GIST-diversity
**Time:** 27.5s
**Papers:** 13

## Breadth / Depth Summary

| Metric | Value |
|--------|-------|
| Unique papers | 13 |
| Total sections | 17 |
| Avg sections / paper | 1.3 |
| Score range | 6.3564 – 6.7501 |

> Scores are ColBERT late-interaction composites computed at **section level**.
> Paper score = avg of its retrieved section scores.

## Paper Rankings

### [1] 2312_17482

**Paper score:** 6.7501 &nbsp;|&nbsp; **Sections retrieved:** 1

**Section 1** (section_idx=3, ColBERT score=6.7501)

> The basic transformer block used in BERT models consists of (1) the attention mechanism and (2) the feed forward layers. This block is then repeated depending on the model size; BERT-Base has 12 repeated transformer blocks, while BERT-Large has 24. For our baseline BERT-Base, we used the exact archi…

### [2] 1911_09886

**Paper score:** 6.6302 &nbsp;|&nbsp; **Sections retrieved:** 1

**Section 1** (section_idx=14, ColBERT score=6.6302)

> We include the performance of different attention mechanisms with our WordDecoding model, effects of our masking-based copy mechanism, and ablation results of three variants of the single attention mechanism with our PtrNetDecoding model in Table 4. WordDecoding with single attention achieves the hi…

### [3] 2410_05258

**Paper score:** 6.5751 &nbsp;|&nbsp; **Sections retrieved:** 1

**Section 1** (section_idx=14, ColBERT score=6.5751)

> In this work, we introduce Differential Transformer (a.k.a. DIFF Transformer), which amplifies attention to the relevant context while canceling noise. Experimental results on language modeling show that DIFF Transformer outperforms Transformer in terms of scaling properties, long-context modeling, …

### [4] 2401_02038

**Paper score:** 6.5176 &nbsp;|&nbsp; **Sections retrieved:** 1

**Section 1** (section_idx=3, ColBERT score=6.5176)

> Transformer is a deep learning model based on an attention mechanism for processing sequence data that can effectively solve complex natural language processing problems. This model was first proposed in 2017 [6], and replaced the traditional recurrent neural network architecture [30] in machine tra…

### [5] 2305_13817

**Paper score:** 6.4853 &nbsp;|&nbsp; **Sections retrieved:** 1

**Section 1** (section_idx=9, ColBERT score=6.4853)

> The architecture consists of a non-pretrained Transformer that performs classification on extracted lines from the PDF. Each line is represented by a sum of a textual and a layout embedding encoded with 96 dimensions. The textual embedding is a pooled convolution window of 3, 4 and 5 tokens over som…

### [6] 2409_04431

**Paper score:** 6.4796 &nbsp;|&nbsp; **Sections retrieved:** 1

**Section 1** (section_idx=15, ColBERT score=6.4796)

> In this work, we present a comprehensive theoretical and empirical study of sigmoid attention as an alternative to softmax attention in transformers. We prove that transformers with sigmoid attention are universal function approximators with improved regularity, and identify LayerScale and preventio…

### [7] 2407_09777

**Paper score:** 6.4743 &nbsp;|&nbsp; **Sections retrieved:** 3

**Section 1** (section_idx=22, ColBERT score=6.4382)

> Graph transformers often face challenges when it comes to generalizing to graphs that they have not encountered before or that fall outside of their usual distribution. This is especially true for graphs that have different sizes, structures, features, and domains. Additionally, graph transformers c…

**Section 2** (section_idx=23, ColBERT score=6.4708)

> Graph transformers commonly regarded as black box models present significant challenges in terms of interpretability and explainability. The lack of sufficient justification and evidence for their decisions can undermine their credibility and transparency. To address this issue, several approaches c…

**Section 3** (section_idx=25, ColBERT score=6.5138)

> Graph transformers require a significant quantity of diverse and high-quality data to achieve effective learning. Nevertheless, in the real world, data is frequently limited, noisy, incomplete, imbalanced, and biased. This has a detrimental effect on the performance and fairness of graph transformer…

### [8] 2504_00927

**Paper score:** 6.4515 &nbsp;|&nbsp; **Sections retrieved:** 1

**Section 1** (section_idx=14, ColBERT score=6.4515)

> In this paper, we focused on a limitation of the standard soft attention mechanism that stems from conditioning on the similarity of a single vector pairs. This makes it challenging for Transformers to precisely locate relevant information based on richer distinguishing information. As a remedy, we …

### [9] 2310_03025

**Paper score:** 6.4333 &nbsp;|&nbsp; **Sections retrieved:** 1

**Section 1** (section_idx=1, ColBERT score=6.4333)

> The long context large language models (LLM) have recently received a lot of attention in production (e.g., Anthropic, 2023; OpenAI, 2023b), research community (e.g., Chen et al., 2023; Liu et al., 2023; Tworkowski et al., 2023), and open source community (e.g., Kaiokendev, 2023). Although the appro…

### [10] 2405_04517

**Paper score:** 6.4260 &nbsp;|&nbsp; **Sections retrieved:** 1

**Section 1** (section_idx=8, ColBERT score=6.4260)

> Linear Attention. Several methods have been suggested to overcome the quadratic complexity in terms of context length of the Transformer and make attention linear in the context length. The Synthesizer learns synthetic attention weights without token-token interactions (Tay et al., 2020). Linformer …

### [11] 2410_13276

**Paper score:** 6.4254 &nbsp;|&nbsp; **Sections retrieved:** 1

**Section 1** (section_idx=0, ColBERT score=6.4254)

> Yizhao Gao * 1 Zhichen Zeng * 2 Dayou Du 3 Shijie Cao 4 Peiyuan Zhou 5 Jiaxing Qi 5 Junjie Lai 5 Hayden Kwok-Hay So 1 Ting Cao 4 Fan Yang 4 Mao Yang 4 Attention is the cornerstone of modern Large Language Models (LLMs). Yet its quadratic complexity hinders efficiency and scalability, especially for …

### [12] 2306_07303

**Paper score:** 6.3967 &nbsp;|&nbsp; **Sections retrieved:** 3

**Section 1** (section_idx=4, ColBERT score=6.3839)

> Before delving into the literature of transformers, let us describe some concepts that will be used throughout this article. Figure 1: Multi-head attention &amp; scaled dot product attention (Vaswani et al., 2017) <image 47> <description>A diagram of the structure of a micro micro micro micro micro …

**Section 2** (section_idx=5, ColBERT score=6.4286)

> The attention mechanism has garnered significant recognition since its introduction in the 1990s, owing to its ability to concentrate on critical pieces of information. In image processing, certain regions of images were found to be more pertinent than others. Consequently, the attention mechanism w…

**Section 3** (section_idx=7, ColBERT score=6.3775)

> The transformer model was primarily developed based on the attention mechanism (Vaswani et al., 2017), with the aim of processing sequential data. Its outstanding performance, especially in achieving state-of-the-art benchmarks for NLP translation models, has led to the widespread use of transformer…

### [13] 2410_04780

**Paper score:** 6.3564 &nbsp;|&nbsp; **Sections retrieved:** 1

**Section 1** (section_idx=1, ColBERT score=6.3564)

> Recent research on Multimodal Large Language Models (MLLMs) has achieved great progress in diverse applications (Yin et al., 2023; Jin et al., 2024; Yan et al., 2024; Zou et al., 2024b), particularly due to their reliance on Transformer models (Vaswani, 2017), where performance is driven by the atte…
