# Simply Estimate

A project planning tool implementing PERT (Program Evaluation and Review Technique) and CPM (Critical Path Method) for simplify project estimation.

## Overview

Simply Estimate helps project managers and development teams create realistic project estimates using proven statistical methods. Inspired by Robert C. Martin's "The Clean Coder: A Code of Conduct for Professional Programmers" (Chapter 10: Estimation).

## Understanding Estimates vs. Commitments

**Estimates** are probabilistic assessments based on available information:
- Represent a range of possible outcomes, not fixed promises
- Account for uncertainty and risk factors
- Should not be treated as binding commitments

**Commitments** are firm promises with specific deliverables and deadlines:
- Require certainty and accountability
- Include buffer time for unforeseen circumstances
- Should only be made after careful analysis

### The Distinction in Practice

Common conversations that illustrate the difference:

**Business**: "Can we say four days then?"  
**Developer**: "No, it could be five or six days."

**Business**: "Can you try to make it no more than six days?"  
**Developer**: "No, 'try' implies success, which means working more than eight hours, working weekends, or skipping family time."

Professional developers maintain clear boundaries between estimates (uncertainty ranges) and commitments (firm promises).

## Estimation Techniques

### 1. PERT (Program Evaluation and Review Technique)

Uses three-point estimation to model uncertainty:

- **Optimistic (O)**: Best-case scenario (~1% probability)
- **Most Likely (M)**: Most realistic estimate
- **Pessimistic (P)**: Worst-case scenario (~1% probability)

**Formula**: Expected Time = (O + 4M + P) / 6

### 2. Wideband Delphi

Collaborative estimation technique for team consensus:

- **Benefits**: Leverages collective expertise and reduces individual bias
- **Process**: Anonymous rounds of estimation with discussion between rounds
- **Considerations**: Requires time investment and skilled facilitation

### 3. Task Decomposition (Law of Large Numbers)

Breaking large tasks into smaller, more predictable components:

- **Benefits**: Reduces estimation error through statistical averaging
- **Process**: Decompose complex tasks into 1-4 hour work units
- **Considerations**: Account for integration complexity and dependencies

