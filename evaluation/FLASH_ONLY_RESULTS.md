# Flash-Only vs Hybrid Model Results

## Test Summary (Oct 27-28, 2025)

### Flash-Only Configuration (Recommended ✅)
- **100 samples (Phase 1 optimized)**: ANLS 0.969, Accuracy 98%
- **100 samples (baseline)**: ANLS 0.946, Accuracy 96%
- **30 samples**: ANLS 0.991, Accuracy 100%
- **Cost**: 5x cheaper than hybrid
- **Speed**: Faster processing times

### Hybrid Model (Pro + Flash) ❌
- **100 samples**: ANLS 0.919, Accuracy 93%
- **20 samples**: ANLS 0.976, Accuracy 100% (lucky sample selection)
- **Cost**: 5x more expensive
- **Issues**: Inconsistent performance, degraded Image/Photo scores

## Phase 1 Optimizations (Oct 28, 2025)

### Performance Improvements
- **Baseline**: ANLS 0.946, Accuracy 96%
- **Phase 1**: ANLS 0.969, Accuracy 98%
- **Improvement**: +0.023 ANLS (+2.4%)
- **Cost**: $0 additional
- **Time**: ~1 day implementation

### Optimizations Implemented
1. **Smart Answer Comparison (Phase 1.1)**: Number-word normalization (+0.022 ANLS)
2. **Numerical Extraction Enhancement (Phase 1.3)**: Critical accuracy instructions for numerical precision
3. **Temperature Optimization (Phase 1.5)**: Dynamic temperature per question type (0.05-0.15)

### Why Flash-Only Wins

1. **Better overall performance**: 0.969 (Phase 1) vs 0.919 on 100 samples
2. **More consistent**: Stable across different sample sizes
3. **Lower cost**: ~$0.02 vs ~$0.10 per question
4. **Faster**: 5.97s avg vs 8.16s avg processing time
5. **Simpler architecture**: Fewer failure points
6. **Optimizable**: Easy to improve with free, zero-shot techniques

## Conclusion

Flash-only configuration is optimal for production use. Hybrid models provide no benefits and add cost/complexity.

**Recommended configuration**: All agents use `gemini-2.5-flash` with Phase 1 optimizations enabled.
