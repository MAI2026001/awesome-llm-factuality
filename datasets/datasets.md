# Datasets

This section contains datasets that can be used to evaluate factuality, hallucination, scientific claim verification, biomedical question answering, and evidence-grounded generation.

---

## 1. SciFact

- **Source:** Allen Institute for AI (AI2)
- **Domain:** Scientific / Biomedical
- **Description:** SciFact is a scientific claim verification dataset containing expert-written scientific claims paired with evidence from scientific paper abstracts. Claims are labelled as supported or contradicted, with evidence rationales.
- **Application:** Scientific claim verification, evidence retrieval, factuality evaluation, and LLM evaluation.
- **Dataset Size:** Approximately 1,409 claims and more than 5,000 scientific abstracts.
- **Link:** https://github.com/allenai/scifact
- **License:** Claims and evidence annotations are released under CC BY 4.0; the corpus abstracts come from S2ORC and have their own license.

---

## 2. SciFact-Open

- **Source:** SciFact-Open GitHub repository
- **Domain:** Scientific
- **Description:** SciFact-Open extends scientific claim verification toward an open-domain setting. It contains claims with evidence and a large research-paper corpus.
- **Application:** Open-domain scientific claim verification, evidence retrieval, and factuality evaluation.
- **Dataset Size:** The corpus contains approximately 500,000 research abstracts from S2ORC.
- **Link:** https://github.com/dwadden/scifact-open
- **License:** Check the repository and underlying data sources before redistribution.

---

## 3. PubHealth

- **Source:** Public health fact-checking resources
- **Domain:** Public Health
- **Description:** PubHealth contains public-health-related claims collected from fact-checking and news websites, together with verification labels and explanations.
- **Application:** Public health misinformation detection, claim verification, and factuality evaluation.
- **Dataset Size:** Approximately 11,832 claims.
- **Link:** https://github.com/sarrouti/PUBHEALTH
- **License:** Check the original repository for the current dataset license and usage conditions.

---

## 4. HealthVer

- **Source:** HealthVer dataset
- **Domain:** Healthcare / Medical
- **Description:** HealthVer is a medical-domain claim verification dataset focused on health-related claims. Claims are verified against scientific evidence and classified according to their relationship with the evidence.
- **Application:** Medical fact checking, health misinformation detection, scientific evidence verification, and LLM factuality evaluation.
- **Dataset Size:** Approximately 14,330 claims.
- **Link:** https://github.com/NLPatVCU/HealthVer
- **License:** Check the original repository for licensing and redistribution conditions.

---

## 5. COVID-Fact

- **Source:** COVID-Fact dataset
- **Domain:** COVID-19 / Healthcare
- **Description:** COVID-Fact contains real-world claims related to the COVID-19 pandemic and evidence used to verify those claims.
- **Application:** COVID-19 misinformation detection, scientific claim verification, evidence retrieval, and healthcare LLM evaluation.
- **Dataset Size:** Approximately 4,086 claims.
- **Link:** https://github.com/THU-KEG/COVID-Fact
- **License:** Check the original repository for the applicable dataset license.

---

## 6. CLIMATE-FEVER

- **Source:** CLIMATE-FEVER dataset
- **Domain:** Climate Science
- **Description:** CLIMATE-FEVER is a dataset for verifying real-world climate-related claims. Claims are paired with evidence from Wikipedia and are labelled according to their factual status.
- **Application:** Climate claim verification, misinformation detection, evidence retrieval, and factuality evaluation.
- **Dataset Size:** Approximately 1,535 claims.
- **Link:** https://github.com/tdiggelm/Climate-FEVER
- **License:** Check the original repository for licensing information.

---

## 7. PubMedQA

- **Source:** PubMedQA
- **Domain:** Biomedical Research
- **Description:** PubMedQA is a biomedical question-answering dataset based on research articles and PubMed abstracts. It contains questions that require understanding scientific and biomedical evidence.
- **Application:** Biomedical question answering, scientific knowledge evaluation, and evaluation of LLM factuality in healthcare.
- **Dataset Size:** The dataset contains approximately 1,000 expert-labelled questions in its labelled setting, with additional data available in the full dataset.
- **Link:** https://pubmedqa.github.io/
- **Paper:** https://aclanthology.org/D19-1259/
- **License:** Check the official dataset page and source data terms before redistribution.

---

## 8. MedMCQA

- **Source:** MedMCQA
- **Domain:** Medical / Healthcare
- **Description:** MedMCQA is a large-scale multiple-choice medical question-answering dataset covering many medical subjects and healthcare topics.
- **Application:** Medical knowledge evaluation, question answering, reasoning evaluation, and testing factual knowledge of LLMs.
- **Dataset Size:** More than 194,000 multiple-choice questions covering 21 medical subjects.
- **Link:** https://medmcqa.github.io/
- **Paper:** https://proceedings.mlr.press/v174/pal22a.html
- **License:** Check the official dataset repository for the current license and usage conditions.

---

## Dataset Comparison

| Dataset | Domain | Main Task | Approx. Size |
|---|---|---|---:|
| SciFact | Scientific / Biomedical | Claim verification | 1,409 claims |
| SciFact-Open | Scientific | Open-domain claim verification | 500K abstracts |
| PubHealth | Public Health | Health claim verification | 11,832 claims |
| HealthVer | Healthcare | Medical claim verification | 14,330 claims |
| COVID-Fact | COVID-19 | Claim verification | 4,086 claims |
| CLIMATE-FEVER | Climate Science | Climate claim verification | 1,535 claims |
| PubMedQA | Biomedical | Question answering | 1,000 labelled questions |
| MedMCQA | Medical | Multiple-choice QA | 194K+ questions |

---

## Why These Datasets Are Relevant

These datasets provide coverage across several scientific and healthcare areas:

- **Scientific research:** SciFact and SciFact-Open
- **Public health:** PubHealth
- **Medical claims:** HealthVer
- **COVID-19:** COVID-Fact
- **Climate science:** CLIMATE-FEVER
- **Biomedical research:** PubMedQA
- **Medical knowledge:** MedMCQA

Together, they provide resources for evaluating whether LLM-generated information is factually correct, supported by evidence, and appropriate for scientific or healthcare applications.
