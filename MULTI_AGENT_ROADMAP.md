# Multi-Agentic System Roadmap

## Vision: Specialist Agent Team for 93-95% ANLS

Current single-agent system: **90.81% ANLS**  
Target multi-agent system: **93-95% ANLS**

---

## Architecture Overview

### Agent Hierarchy

```
Orchestrator Agent (Main Entry Point)
├── OCR Specialist Agent
│   ├── Tools: Document AI OCR, Table Parser
│   ├── Specializes in: Tables, Forms, Structured text
│   └── Best for: table/list, form questions
│
├── Vision Specialist Agent  
│   ├── Tools: Gemini Vision, Image Analysis
│   ├── Specializes in: Photos, Images, Visual content
│   └── Best for: image/photo, figure/diagram questions
│
├── Layout Specialist Agent
│   ├── Tools: Document AI Layout Parser
│   ├── Specializes in: Document structure, spatial relationships
│   └── Best for: layout, abstract/opinion questions
│
└── Answer Validator Agent
    ├── Tools: Confidence scoring, Format checking
    ├── Specializes in: Answer refinement, validation
    └── Always runs: Final stage for all answers
```

---

## Phase 1: Core Specialist Agents (Week 1)

### Task 1.1: Create OCR Specialist Agent
**Files to create:**
- `agents/ocr_specialist.py`
- `tools/table_parser.py` (enhanced table handling)

**Capabilities:**
- Specialized prompts for structured data
- Enhanced table understanding
- Form field extraction
- High-confidence scoring for table/form questions

**Expected Impact:** +5-10% on table/form questions

---

### Task 1.2: Create Vision Specialist Agent
**Files to create:**
- `agents/vision_specialist.py`
- `tools/image_analyzer.py`

**Capabilities:**
- Specialized prompts for visual content
- Brand/product recognition focus
- Photo interpretation
- Multi-resolution analysis

**Expected Impact:** +15-20% on image/photo questions (65% → 80-85%)

---

### Task 1.3: Create Layout Specialist Agent
**Files to create:**
- `agents/layout_specialist.py`
- `tools/layout_analyzer.py`

**Capabilities:**
- Document structure understanding
- Spatial relationship analysis
- Section identification
- Context-aware answering

**Expected Impact:** +2-3% on layout questions

---

## Phase 2: Orchestrator & Routing (Week 2)

### Task 2.1: Build Orchestrator Agent
**Files to modify:**
- `agent.py` → `agents/orchestrator.py`

**Routing Logic:**
```python
def route_question(question: str, question_types: List[str]) -> List[Agent]:
    """Route to specialist agent(s) based on question analysis"""
    
    if 'table' in question_types or 'list' in question_types:
        return [ocr_specialist, answer_validator]
    
    elif 'image' in question_types or 'photo' in question_types:
        return [vision_specialist, answer_validator]
    
    elif 'layout' in question_types:
        return [layout_specialist, answer_validator]
    
    elif 'form' in question_types:
        return [ocr_specialist, layout_specialist, answer_validator]
    
    else:  # General/handwritten/abstract
        return [vision_specialist, ocr_specialist, answer_validator]
```

**Expected Impact:** Optimal routing = +1-2% overall

---

### Task 2.2: Create Answer Validator Agent
**Files to create:**
- `agents/answer_validator.py`
- `tools/answer_scorer.py`

**Capabilities:**
- Confidence scoring for each answer
- Format validation (dates, numbers, etc.)
- Answer length checking
- Fallback to alternative agent if low confidence

**Expected Impact:** +1-2% by catching errors

---

## Phase 3: Ensemble & Advanced Strategies (Week 3)

### Task 3.1: Multi-Agent Voting
**Strategy:**
- For challenging questions, query multiple specialists
- Agents vote on answer with confidence scores
- Weighted ensemble based on question type

**Expected Impact:** +0.5-1% on edge cases

---

### Task 3.2: Few-Shot Examples per Agent
**Strategy:**
- Each specialist gets domain-specific few-shot examples
- Vision specialist: Brand recognition examples
- OCR specialist: Complex table examples
- Layout specialist: Unusual layout examples

**Expected Impact:** +1-2%

---

### Task 3.3: Confidence-Based Fallback
**Strategy:**
```python
# If primary agent has low confidence, try alternative
if primary_confidence < 0.7:
    backup_answer = fallback_agent.run(question)
    if backup_confidence > primary_confidence:
        return backup_answer
```

**Expected Impact:** +0.5-1%

---

## Phase 4: Optimization & Testing (Week 4)

### Task 4.1: A/B Testing Framework
- Test single vs multi-agent on same samples
- Compare per-question-type performance
- Measure latency and cost tradeoffs

### Task 4.2: Cost Optimization
- Cache OCR results across agents
- Selective specialist invocation
- Parallel agent execution where possible

### Task 4.3: Full Benchmark Evaluation
- Run 1000+ samples
- Statistical significance testing
- Per-question-type analysis

---

## Expected Outcomes

### Performance Targets

| Question Type | Current ANLS | Target ANLS | Strategy |
|--------------|--------------|-------------|----------|
| Table/List | 99.52% | 99.8% | OCR Specialist |
| Layout | 95.78% | 97% | Layout Specialist |
| Form | 89.98% | 93% | OCR + Layout |
| General | 100% | 100% | Vision + Validator |
| Handwritten | 100% | 100% | Vision Specialist |
| Free Text | 80.36% | 85% | Vision + Layout |
| Figure/Diagram | 75% | 82% | Vision Specialist |
| **Image/Photo** | **65%** | **82-85%** | **Vision Specialist (focused)** |

**Overall Target: 93-95% ANLS** (up from 90.81%)

---

## Cost Analysis

### Per-Document Costs

| Configuration | API Calls | Cost/Doc | Speed |
|--------------|-----------|----------|-------|
| Current (single) | 1-2 | $0.0016 | 6.3s |
| Multi-agent (selective) | 2-3 | $0.0024 | 8-10s |
| Multi-agent (ensemble) | 3-5 | $0.0040 | 12-15s |

**Recommendation:** Selective routing (2-3 agents) for best cost/performance balance

---

## Implementation Priority

### High Priority (Do First)
1. ✅ Vision Specialist Agent - Biggest impact on weakest area (Image/Photo)
2. ✅ Orchestrator Agent - Required for routing
3. ✅ Answer Validator - Catch errors across all types

### Medium Priority
4. OCR Specialist Agent - Current performance already excellent
5. Layout Specialist Agent - Refinement of good performance

### Low Priority (Polish)
6. Ensemble voting
7. Advanced caching
8. Few-shot examples

---

## Next Steps

1. **Decision Point:** Does team want to proceed with multi-agent architecture?
2. **Resource Allocation:** How much time/budget for development?
3. **Success Criteria:** What ANLS score justifies the added complexity?
4. **Timeline:** Iterative (build one agent at a time) vs. full build?

---

## Questions to Consider

- **Latency:** Is 8-12s per document acceptable (vs current 6.3s)?
- **Cost:** Is 2-3x cost increase worth 3-5% ANLS improvement?
- **Complexity:** Multi-agent = more code to maintain
- **ROI:** For what use cases is 93-95% significantly better than 90.81%?

---

## Recommended Approach

**Start Small, Iterate Fast:**

1. **Week 1:** Build Vision Specialist for Image/Photo (biggest gain)
2. **Test:** Evaluate impact on 100 samples
3. **If successful:** Add Orchestrator + Validator
4. **Test:** Re-evaluate on 100 samples  
5. **If 92%+:** Continue with remaining specialists
6. **If < 92%:** Revisit strategy

**This way, you validate the approach before committing to full build.**

---

## Success Metrics

| Milestone | ANLS Target | Status |
|-----------|-------------|--------|
| Vision Specialist added | 92% | Not started |
| + Orchestrator/Validator | 93% | Not started |
| + All specialists | 94%+ | Not started |
| Full optimization | 95%+ | Not started |

---

Let's discuss with team and decide on next steps!

