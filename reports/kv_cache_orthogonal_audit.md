# Audit of Lossless Cross-LLM KV-Cache Transfer via Orthogonal Transformation

**Answer:** The paper does not prove that the orthogonal transformation yields exactly the same target-model attention outputs, hidden states, logits, and generation as native target-model prefill; it only proves that the transformation preserves vector norms and enables exact inversion, but it does not establish identity of hidden states, logits, or generation [1, 2].

## Abstract

The paper establishes that an orthogonal transformation preserves vector norms and enables exact inversion via the transpose, but it does not prove that the transformed cache yields exactly the same target-model attention outputs, hidden states, logits, or generation as native target-model prefill [3, 4].

## Previous Work

An orthogonal transformation is a linear transformation that preserves a symmetric inner product and consequently preserves vector lengths and angles between vectors, as defined in mathematical literature [3]. Such transformations can be represented by orthogonal matrices and include rotations and improper rotations, which are characterized by the property that the matrix multiplied by its transpose yields the identity [3]. When identical Hadamard transforms are applied to both query and key matrices, the attention logits remain unchanged because the orthogonality condition ensures that the transformed inner products equal the original ones [1]. The norm preservation property guarantees that the Euclidean norm of any vector is unchanged after transformation, which underpins the lossless nature of the operation [1]. The inverse of the transformation equals its transpose, allowing exact reversal without loss of information, as demonstrated by the identity HdHdT=Id [1]. The paper describes a practical implementation where the key transform is fused into the key projection weight matrix offline, while the query transform is applied online per decoding step, enabling efficient computation [1].

## Experiments

The experiments evaluate SplitZip as a GPU-friendly lossless compression scheme for KV-cache transfer in disaggregated LLM serving, demonstrating bitwise preservation of KV tensors and exact model output match across context lengths from 128 to 2840 tokens [4]. All listed speculative decoding algorithms (Eagle-3, P-EAGLE, DFlash, DFlash2, DSpark, and MTP) are reported to be lossless, producing output from the same distribution as the target model [5]. The invariance property guarantees mathematically identical model output to the original unprotected model, ensuring lossless accuracy in privacy-preserving KV-cache schemes [6]. Applying identical Hadamard transforms to both query and key matrices is proven lossless, leaving attention logits unchanged due to the orthogonality condition HdHdT=Id [1]. The orthogonal transformation preserves vector norms and enables exact inversion via the transpose, supporting efficient fused key projection and online query transformation in practical implementations. Cross-model KV cache transfer achieves 2.7 to 25 times speedup over re-prefilling from scratch while maintaining 73–98% performance retention [2].

## Comparison

| Item | What is transformed | What is recomputed | Strict requirement met? | Key evidence |
|---|---|---|---|---|
| transformed cache | applies orthogonal transform to KV cache [3] | not established | no | F1, F2, F3, F4, F5, F6, F10, F11, F12, F15, F18, F21, F23, F25 [3] |
| native target-model prefill | native prefill process [4, 5, 6] | not established | no | F7, F8, F9, F13, F14, F15, F16, F17, F19, F20, F22, F24, F26 [4, 5, 6] |

## Discussion

The orthogonal transformation is proven to preserve vector norms and enable exact inversion via the transpose, but the paper does not establish that the transformed cache yields exactly the same target-model attention outputs, hidden states, logits, or generation as native target-model prefill [3, 4].

## References

[1] [OScaR: Occam's Razor for Extreme KV Cache Quantization | Zhongzhu (Charlie) Zhou](https://www.zhongzhuzhou.org/blog/2026-06-17-oscar-technical-review-en)
[2] [Transferring KV Cache to Models of Different Sizes](https://note.com/okssusucha/n/n2f229ce1b782?hl=en)
[3] [Orthogonal Transformation -- from Wolfram MathWorld](https://mathworld.wolfram.com/OrthogonalTransformation.html)
[4] [SplitZip: Ultra Fast Lossless KV Compression for Disaggregated LLM Serving](https://arxiv.org/abs/2605.01708)
[5] [Decision Guide - Speculators Docs](https://docs.vllm.ai/projects/speculators/en/latest/user_guide/algorithms/decision_guide)
[6] [Unveiling and Mitigating Privacy Risks of KV-cache in LLM ...](https://www.ndss-symposium.org/wp-content/uploads/2026-f258-paper.pdf)

## Methodology

Decompose the claim into definitional and proof components, focusing on precise mathematical and empirical verification of losslessness and identity of outputs across all specified dimensions.

## Limitations

The paper does not address whether the transformed cache produces identical hidden states, logits, or generation; it only discusses norm preservation and inversion.
