source: https://www.sapien.io/blog/labeling-data-for-machine-learning-best-practices-and-quality-control
Concise: short and cover foundation problems in labeling process

## Defining clear and comprehensive labeling instructions:
The guidelines should cover:
1. Definitions of labels: precise, with examples, clarify scope and boundary of each label
2. Labeling criteria: e.g the minimum threshold for a positive label or the specific attributes that determine the label.
3. Edge cases and exceptions: potential edge cases and exceptions, guidance on how to handle them consistently.
4. Visual aids: to illustrate the labeling process and provide a reference for annotators.

## Strategies to handle edge cases and ambiguous Examples:
1. Collaborative decision-making
2. Escalation process: for difficult cases, ask senior annotators or domain experts who can provide guidance and make final decisions.
3. Uncertainty labeling: add additional labels or confidence scores for ambiguous examples, enabling downstream analysis and potential refinement of the labels.
4. Continuous feedback and updates: Regularly review and update the annotation guidelines based on the feedback and insights gained from handling edge cases

## maintaining consistency across annotators:
1. Training and calibration (hiệu chuẩn)
2. Regular quality control checks
3. Collaborative annotation
4. Automated cónistency checks

## IAA: inter-annotator agreement
### metrics:
1. Cohen's Kappa
2. Fleiss's Kappa
### resolve dissagreements:
1. Majority vote
2. Adjudication: assign a senior annotator or domain expert to review and resolve disagreements
3. collaborative resolution
4. weighted vote
### IAA thresholds:
0.6 - sufficient
0.8: perfect
