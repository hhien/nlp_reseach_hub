1. https://sourcebae.com/blog/data-labeling-annotation-guide/?utm_source=chatgpt.com
this site includes all aspects of annotations: from foundations to strategy of various domains: CV, NLP,

# Foundation
A data label is a tag, category, bounding box, annotation, transcription, or any form of metadata applied to a raw data point to describe its content, context, or meaning. 
Labels are the ground truth against which model predictions are evaluated.

In professional data labeling workflows, labelers operate within structured quality assurance pipelines. 
They undergo domain-specific training, follow detailed annotation guidelines, and are evaluated continuously through 
benchmark tasks and inter-annotator agreement metrics. For domain-sensitive tasks (medical, legal, scientific), 
labelers may require professional credentials or specialized subject matter expertise.

--> what is needed for a labeling workflow:
- labeling pipeline
- domain-specific training (if necessary)
- annotation guidelines
- testing benchmark

## Data annotation vs data labeling
Although 2 terms are often used interchangeably, there are key differences:
data labeling mainly focuses on assigning categorical tags or class labels to data while
data annotation focuses on adding descriptive metadata, attributes, or contextual information.
Therefore, their typical outputs, complexity, use cases, and industry usage are different.

2023 Snorkel AI sủvey, data scientít spend an average of 45% of their time on data preparation including labelling and curation

the ‘data-centric AI’ philosophy argues that systematically improving data quality produces more reliable model performance gains than iterating on model architectures - Andrew Ng.

# Best Practices for High-Quality Annotations
1. Start with a pilot batch (200–500 samples) to validate instructions, calibrate labeler performance, and identify ambiguities before scaling.
2. Write unambiguous, comprehensive annotation guidelines with positive and negative examples for every edge case.
3. Use benchmark (gold standard) tasks: embed known-correct tasks into labeler queues and use performance on these to gate labeler inclusion.
4. Establish a consensus pipeline for subjective tasks: collect 3–5 annotations per item and use majority vote or weighted consensus (by labeler historical accuracy).
5. Implement hierarchical review: primary annotator → QA reviewer → subject matter expert for ambiguous or high-stakes items.
6. Deploy AI-assisted pre-labeling to reduce human annotation time, then route only corrections and edge cases to human reviewers.
7. Continuously audit random samples from completed batches even after a project is complete to detect quality drift.
8. Track and retrain or remove underperforming annotators. Performance on benchmark tasks is the most reliable signal.
9. Curate your dataset using model-assisted tools: use model predictions on the labeled training set to surface likely mislabeled examples for human review.
10. Version your labels: maintain audit trails of label changes and annotator provenance for reproducibility and compliance.
11. Calibrate instructions as you encounter edge cases update guidelines and communicate changes to the full labeling team.
12. Leverage diverse annotator demographics for subjective tasks (sentiment, cultural references) to reduce systematic bias.

# Data labeling for specific domains
## NLP & text:
- use native speakers
- handle linguistic ambiguity explicitly: annotation guidelines must address ambiguous cases
- build consensus pipelines for subjective tasks: collect multiple annotations and resolve disagreements via explicit adjudication protocols
- levarate rule-based pre-labeling: Known named entities, domain-specific terminology, and regex patterns can be pre-labeled automatically, reserving human attention for novel or ambiguous cases.

# Data labeling framework
## data collection and pipeline design

## annotation guideline design:
1. write for your worst-case annotator
2. use worked examples for every edge case: annoted examples of ambiguious cases, common mistakes, borderline decision
3. version control your guidelines
   
## workforce & quality management:
1. screen annotators before production: use calibration batches and benchmark tasks to qualify annotators
2. implement tiered review: annototer --> QA reviewer --> expert auditor
3. incentivize quality over speed

## technology and tooling
