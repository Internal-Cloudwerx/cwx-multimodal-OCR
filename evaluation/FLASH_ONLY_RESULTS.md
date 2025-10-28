# Flash-Only vs Hybrid Model Results

## Test Summary (Oct 27, 2025)

### Flash-Only Configuration (Recommended ✅)
- **100 samples**: ANLS 0.946, Accuracy 96%
- **30 samples**: ANLS 0.991, Accuracy 100%
- **Cost**: 5x cheaper than hybrid
- **Speed**: Faster processing times

### Hybrid Model (Pro + Flash) ❌
- **100 samples**: ANLS 0.919, Accuracy 93%
- **20 samples**: ANLS 0.976, Accuracy 100% (lucky sample selection)
- **Cost**: 5x more expensive
- **Issues**: Inconsistent performance, degraded Image/Photo scores

## Why Flash-Only Wins

1. **Better overall performance**: 0.946 vs 0.919 on 100 samples
2. **More consistent**: Stable across different sample sizes
3. **Lower cost**: ~$0.02 vs ~$0.10 per question
4. **Faster**: 5.97s avg vs 8.16s avg processing time
5. **Simpler architecture**: Fewer failure points

## Conclusion

Flash-only configuration is optimal for production use. Hybrid models provide no benefits and add cost/complexity.

**Recommended configuration**: All agents use `gemini-2.5-flash`
