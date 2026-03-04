# Headline Sentiment Intelligence (Zero-Shot NLP)

## Executive Summary
This project demonstrates a data-first approach to market sentiment. By leveraging Zero-Shot Classification (BART-large-mnli), this tool bypasses the high error rates of legacy, dictionary-based analyzers (like TextBlob) in favor of deep semantic context.

## Why This Matters for 2026 Marketing
Traditional SEO is dead. Founders must now optimize for **GEO (Generative Engine Optimization)**. This tool serves as a foundational component for monitoring how AI models perceive brand headlines and market trends in real-time.

## High-Level Capabilities
* **Zero-Shot Logic:** Classifies headlines into any category without pre-training or hard-coded keyword lists.
* **Confidence Scoring:** Outputs a mathematical probability for every label, allowing for high-precision data analysis.
* **Tech Stack:** Python 3.12, Hugging Face Transformers, BeautifulSoup4.

## Deployment
This repository is optimized for quick deployment in virtual environments to ensure dependency isolation and system hygiene.

Sample Output
The following data was generated during a live execution of the analyzer. Note the confidence scores, which highlight the model's ability to handle linguistic nuance in real-time headlines.

Classification complete in 32.0s (1.6s per headline)

                          HEADLINE SENTIMENT ANALYSIS (Zero-Shot)                          
                              Model: facebook/bart-large-mnli                              
                       Source: TechCrunch | March 04, 2026 03:54 AM        
```text                   
+------+------------------------------------------------------+------------+--------------+
|    # | Headline                                             | Sentiment  |   Confidence |
+------+------------------------------------------------------+------------+--------------+
|    1 | The hidden cost of early hires with Yuri Sagalov,... | Negative   |        71.2% |
|    2 | More from TechCrunch                                 | Positive   |        60.7% |
|    3 | Why AI startups are selling the same equity at tw... | Neutral    |        41.1% |
|    4 | Claude Code rolls out a voice mode capability        | Neutral    |        41.0% |
|    5 | A suite of government hacking tools targeting iPh... | Negative   |        68.3% |
|    6 | Alibaba’s Qwen tech lead steps down after major A... | Negative   |        47.0% |
|    7 | Hacked traffic cams and hijacked TVs: How cyber o... | Negative   |        60.9% |
|    8 | X says it will suspend creators from revenue-shar... | Negative   |        89.2% |
|    9 | The candidate that Silicon Valley built is now th... | Negative   |        81.8% |
|   10 | Apple unveils new MacBook Air and MacBook Pro wit... | Positive   |        50.5% |
|   11 | Just three companies dominated the $189B in VC in... | Positive   |        44.1% |
|   12 | TikTok down for some in US, thanks to second Orac... | Negative   |        85.5% |
|   13 | AI companies are spending millions to thwart this... | Negative   |        74.9% |
|   14 | ChatGPT’s new GPT-5.3 Instant model will stop tel... | Positive   |        35.5% |
|   15 | Anduril aims at $60 billion valuation in new fund... | Positive   |        85.3% |
|   16 | Android users can now share tracker tag info with... | Positive   |        42.2% |
|   17 | ChatGPT uninstalls surged by 295% after DoD deal     | Positive   |        78.6% |
|   18 | MyFitnessPal has acquired Cal AI, the viral calor... | Negative   |        44.3% |
|   19 | Anthropic’s Claude reports widespread outage         | Negative   |        83.7% |
|   20 | The trap Anthropic built for itself                  | Negative   |        92.1% |
+------+------------------------------------------------------+------------+--------------+
```
