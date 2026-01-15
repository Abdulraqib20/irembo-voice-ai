

# Machine Learning Engineer Take-Home Assignment

**Role:** Machine Learning Engineer – Voice AI

## Overview

1. All applicants are required to complete and submit this take-home assignment.
2. Submissions will be reviewed offline by the hiring team.
3. Only shortlisted candidates will be invited to present their work to a technical panel
4. Submission timeline: *5 days after receiving this assignment*

## Context

Irembo is building a Voice AI assistant to help citizens access and complete government services using natural-language voice interactions.

The system must:

- Understand user intent from spoken input
- Handle multilingual and low-resource language contexts
- Operate reliably in a public-sector, citizen-facing environment

This assignment focuses on designing and prototyping **one core ML component** of that system.

## Assignment Objective

Design and prototype an ***Intent Classification ML solution*** for a Voice AI system, and clearly explain your technical decisions, assumptions, and trade-offs.

## Problem Statement

You are working with data from a Voice AI system that includes:

- `utterance_text` – transcribed speech from users
- `language` – e.g., `rw`, `en`, or `mixed`
- `intent` – the user's goal (e.g., `check_application_status`, `apply_new_service`, `check_traffic_fines`)
- Optional or noisy metadata (e.g., confidence scores, incomplete labels)

## Your Task

Build an ML solution that predicts the *user's intent* from the utterance text.

You may:

- Use the [provided dataset](#) or
- Create a realistic synthetic dataset and clearly document your assumptions

### Multilingual & Low-Resource Language Context (Kinyarwanda Example)

In practice, the Voice AI system must handle Kinyarwanda, often mixed with English or informal phrasing.

Example User Utterances

| User Utterance                                          | Notes                  |
|---------------------------------------------------------|------------------------|
| <i>"Ndashaka kureba status ya application yanije."</i>  | Kinyarwanda + English  |
| <i>"Ese passport yanije igeze he?"</i>                  | Informal phrasing      |
| <i>"Nashaka gufata rendez-vous yo gufata passport."</i> | Borrowed French term   |
| <i>"Natanze application ariko sinzi aho igeze."</i>     | Paraphrased intent     |
| <i>"Checka niba application yanije yarangiye."</i>      | Code-switching + slang |

All of the above may map to the same intent:

[check\\_application\\_status](#)

When handling such examples, consider and explain:

- How your model handles **code-switching**
- How does it generalize across formal and informal language
- Strategies for **limited labeled Kinyarwanda data**
- How would you evaluate performance specifically for low-resource language utterances ( in this case, **Kinyarwanda**)

## What to Submit

### 1. Model & Approach (Core)

- Implement one primary modeling approach
  - e.g., a multilingual transformer, a fine-tuned model, or a justified baseline
- Clearly explain:
  - Why this model fits a Voice AI use case
  - How it supports multilingual / low-resource scenarios
  - Trade-offs between accuracy, latency, and complexity

### 2. Data Strategy: Explain how you would:

- Clean and preprocess noisy voice transcripts
- Handle transcription errors and code-switching
- Deal with small or imbalanced datasets
- Improve data quality over time (e.g., labeling, feedback loops)

### 3. Evaluation

- Define the metrics you chose and why they matter
- Describe how you would:
  - Test robustness and edge cases
  - Identify common failure modes
  - Assess suitability for a citizen-facing system

### 4. Production & MLOps Consideration ;Conceptually describe:

- Model versioning and deployment strategy
- Monitoring in production (performance, drift, degradation)
- How the model integrates into a real-time Voice AI pipeline
- Fallback strategies for low-confidence predictions

*Full production deployment is not required; it's a good to have. We care about your critical thinking, tradeoffs, and design judgment.*

### 5. Ethics & Public-Sector Considerations ; Briefly address:

- Bias and fairness risks
- Explainability expectations
- Responsible AI considerations for government services

### 6. This is Optional (bonus)

You may include **one** of the following if time allows:

- Confidence-based fallback or escalation logic
- A simple inference API design
- A diagram of the Voice AI → Intent → Action flow
- A multilingual expansion strategy

## **Submission Instructions**

Please submit **one (1) PDF document** that contains everything listed below.

1. **Link to your Git repository (must be public)**
  - The repository should contain all code related to your solution
  - Include clear instructions on how to run your work (locally or via notebook)
2. **Written summary (max 4 pages)** covering:
  - Your overall approach
  - Key assumptions you made
  - Trade-offs and limitations
  - Any notes you think are important for reviewers
  - Responses to parts 2, 3, 4, and 5 - ensure to label them accurately in the document

## **Important Notes**

- Do **not** submit separate README files or documents
- All explanations should be included in the **single PDF**
- The Git repository must be **public** and accessible via the link in the PDF

How we will evaluate your submission (High-Level): we will assess the following;

- Sound ML engineering judgment
- Understanding of Voice AI and NLP constraints
- Practical production and MLOps thinking
- Data realism and maturity
- Clarity of written communication
- Awareness of ethical and public-sector implications

*We understand that candidates may use AI-assisted tools as part of their workflow. However, during the review and presentation stages, we will be assessing your critical thinking, technical judgment, and ability to explain and defend your decisions.*