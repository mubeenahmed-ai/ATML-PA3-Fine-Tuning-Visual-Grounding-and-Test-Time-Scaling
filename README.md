# ATML PA3: Fine-Tuning, Visual Grounding and Test-Time Compute

Code, configurations, split IDs and machine-readable results for Programming Assignment 3 of
**EE-5102/CS-6304 Advanced Topics in Machine Learning (LUMS, Fall 2026)**.

**Author:** Mubeen Ahmed  
**Report:** [`ATML_PA3_Report.pdf`](ATML_PA3_Report.pdf)

| Task | Notebook | What it does |
|---|---|---|
| 1 | [`ATML_PA3_Task1_LLM_FineTuning.ipynb`](ATML_PA3_Task1_LLM_FineTuning.ipynb) | Fine-tunes Qwen3-0.6B on 40k MetaMathQA examples with Full FT, Full FT + L2-SP, LoRA-4, LoRA-16 and OPLoRA-16. Evaluates GSM8K, HellaSwag, BoolQ and ARC-Easy, and runs weight-space, activation and qualitative analyses. |
| 2 | [`ATML_PA3_Task2_VLM_Grounding.ipynb`](ATML_PA3_Task2_VLM_Grounding.ipynb) | Builds a SigLIP → MLP → Qwen3-0.6B vision-language model and trains it through a three-stage curriculum: caption alignment, template visual questions, then natural VQAv2. Tests grounding with complementary-swap, mismatched and no-image controls. |
| 3 | [`ATML_PA3_Task3_TestTime_Compute.ipynb`](ATML_PA3_Task3_TestTime_Compute.ipynb) | Studies repeated sampling, self-certainty, VisConf and rank-weighted consensus, plus adaptive stopping, on the supplied Qwen2.5-3B (MATH) and Qwen2.5-VL-3B (MathVista) rollouts. Also runs a live comparison of one long response, parallel sampling and sequential refinement. |

---

## Repository structure

```
.
├── README.md
├── ATML_PA3_Report.pdf
├── ATML_PA3_Task1_LLM_FineTuning.ipynb
├── ATML_PA3_Task2_VLM_Grounding.ipynb
├── ATML_PA3_Task3_TestTime_Compute.ipynb
├── report/              # LaTeX source of the report (NeurIPS 2024 preprint style)
├── task1/               # Task 1 results
│   ├── data/
│   ├── eval/
│   ├── runs/
│   ├── diagnostics/
│   ├── qualitative/
│   └── results/
├── task2/               # Task 2 results
│   ├── data/
│   └── runs/tok196/
└── task3/               # Task 3 results
    ├── cache/
    ├── offline/
    ├── live/
    ├── adaptive/
    └── figures/
```

`results/` mirrors the folders the notebooks write to Google Drive (`MyDrive/ATML_PA3/task{1,2,3}/`). The section [What is (and is not) in `results/`](#what-is-and-is-not-in-results) lists exactly which files are included.

---

## How to reproduce

All three notebooks are designed for **Google Colab** with Google Drive mounted.

1. Open a notebook in Colab, select a **GPU runtime**, and choose **Run all**.
2. The first cells mount Drive and install the dependencies.
3. Every stage saves its outputs to Drive and skips work that is already finished. After a disconnect, reconnect and choose **Run all** again. Training resumes from the latest checkpoint, which stores the model or adapters, optimizer, scheduler, gradient scaler, random-number state and data position.

### Task 1: LLM fine-tuning

- **Data:** the notebook downloads `meta-math/MetaMathQA`, `openai/gsm8k` (to exclude test questions from training) and the evaluation datasets through lm-evaluation-harness. `first_plot_questions.yaml` is downloaded from the emergent-misalignment repository if it is not already on Drive.
- **Hardware:** training needs about 12 GB of GPU memory for Full FT and about 3.5 GB for the adapter runs. We trained on a Tesla T4 (about 2.5 h per Full FT run and 3–4.3 h per adapter run) and evaluated on an A100. Each condition takes about 15 minutes of GSM8K generation plus about 4 minutes for the other benchmarks.
- **Drive space:** a Full FT resume state is about 7 GB, and only one is kept at a time. Files deleted through the Drive mount go to Drive's Trash. With `PURGE_DRIVE_TRASH = True`, the notebook permanently deletes only its own trashed checkpoints (`resume_uNNNN.pt`, `ckpt_NNNN.pt`).
- **Splitting the work:** set `RUNS_TO_TRAIN` or `CONDITIONS_TO_EVAL` to spread the work across sessions.
- **Extra analysis:** the last two cells (the BoolQ "yes"-rate and the TPR/TNR/balanced-accuracy/AUC analysis) were added after the main run and reproduce the BoolQ decomposition in the report.

### Task 2: VLM and visual grounding

- **Data:** the notebook downloads the COCO 2014 caption and instance annotations, the VQAv2 questions, annotations and complementary pairs, and only the COCO images it needs (9,279 images). Images are deleted after their SigLIP features are cached. The features (about 3 GB in FP16) live on Drive.
- **Hardware:** a Tesla T4 is enough. Feature caching takes about 6 minutes and training about 98 minutes (46, 35 and 17 minutes per stage), with a 2.2 GB peak.
- **Optional experiment:** set `N_VIS = 49` in the configuration cell to run the 49-token compression experiment. Its outputs go to `runs/tok49/`. This run was not performed for the report.

### Task 3: Test-time compute

- **Data:** add the course-supplied **ATML-PA3-Data** folder to your Drive as a shortcut. To do this, open the shared folder, choose Organize → Add shortcut → My Drive. The notebook finds the folder automatically and checks the loaded data against the totals in its README. MathVista question metadata comes from the released [`testmini.json`](https://github.com/lupantech/MathVista/blob/main/data/testmini.json).
- **Hardware:** the offline analyses (Steps 1–3 and 5–6) run on CPU. On a runtime without a GPU, the live step (Step 4) is skipped automatically.
- **Live step:** Step 4 needs a GPU (about 6 GB in FP16). It uses `Qwen/Qwen2.5-3B-Instruct` at revision `aa8e72537993ba99e69dfaafa59ed015b17504d1` and took about 1.5 hours for 50 questions. Every generation call is logged to `live/calls.jsonl` as it finishes, so an interrupted run resumes.

---

## Configuration and seeds

All hyperparameters are set in the configuration cell at the top of each notebook and match the assignment specification. The seed is **6304** throughout.

**Task 1**
- **Training data:** the first 40,000 eligible MetaMathQA examples in source order.
- **Batching and schedule:** effective batch 32 (microbatch 2 × 16 accumulation steps), 1,250 updates, cosine schedule with 38 warm-up updates.
- **Optimizer:** AdamW with betas (0.9, 0.999), weight decay 0.01, gradient clipping at 1.0, FP16 autocast with dynamic loss scaling.
- **Learning rates:** 2e-5 for Full FT and L2-SP (λ = 1e-3); 2e-4 for LoRA (r = 4, α = 8 and r = 16, α = 32) and OPLoRA (r = 16, α = 32, k = 128).

**Task 2**
- **Batching:** effective batch 16 (microbatch 8 × 2 accumulation steps).
- **Stages:** 2,250, 1,500 and 750 updates. Connector learning rates are 1e-3, 1e-4 and 5e-5; LoRA learning rates (r = 16 on q/k/v/o) are 2e-5 and 1e-5 for Stages 2 and 3.
- **Schedule:** 5% warm-up, then cosine decay to 10% of the peak learning rate.

**Task 3**
- **Splits:** 200 development and 800 held-out questions, drawn with `default_rng(6304)`.
- **Orders:** ten seeded rollout orders per question, shared by every method.
- **Budgets and bootstrap:** N ∈ {1, 2, 4, 8, 16, 32}; 1,000 paired question-level bootstrap resamples.

The split and example IDs are saved by the notebooks and included in `results/`:
- **Task 1:** `task1/data/selected_source_indices.txt`, `probe_set.json` and `data_stats.json`.
- **Task 2:** `task2/data/selection.json`, `pair_ids.json`, `mismatch_map.json` and `curriculum_sequence.npz`.
- **Task 3:** `task3/cache/*_splits.json` and `*_orders.json`.

---
