# Erowid × Mistral-7B: fine-tuning and creativity evaluation

Group 2: Selim Can Mutlu, Le Phuong Thao Hoàng, Deren Serçe.
NLP take-home assignment, Tilburg University, fall 2026.

We reconstructed a LoRA fine-tuning pipeline from a written project report, training Mistral-7B on drug-experience reports from the Erowid experience vaults. We then compared the base and fine-tuned models on three operationalisations of creativity at temperatures 0.5, 1.0 and 1.5:

- **DAT** (Divergent Association Task): 10 maximally different nouns, scored by SBERT distance (Nakajima et al., 2026).
- **CDAT** (Conditional DAT): diverse nouns that must stay associated with a cue word. It reports novelty, appropriateness and an appropriateness gate.
- **Novel words**: invented words for five new concepts (inspired by Ciaccio et al., 2025), scored by the group with a rubric.

## Repository contents

| Path | What it is |
|---|---|
| `group2_nlp_assignment_fall2026.ipynb` | The full pipeline and analysis, with all outputs and our answers to the questions |
| `stimuli&outputs/group2_DAT_scores.tsv` | DAT results: model, temperature, repetition, valid words, score |
| `stimuli&outputs/group2_CDAT_scores.tsv` | CDAT results: adds novelty, appropriateness and whether each list passed the gate |
| `stimuli&outputs/group2_novelWord_scoring.tsv` | The 30 invented words with the rubric scores |
| `stimuli&outputs/group2_novelWord_autochecks.tsv` | Automatic checks supporting the rubric: dictionary lookup, nearest existing word, edit distance |
| `stimuli&outputs/group2_reproducibility_log.txt` | Every implementation decision, whether the report specified it, and its expected impact |
| `stimuli&outputs/raw_generations/` | Every raw model output (JSONL), including the decoding settings |
| `artifacts/finetuned_adapter/` | The trained LoRA adapter (for `mistralai/Mistral-7B-Instruct-v0.2`) and its training log |
| `artifacts/*.json`, `train_progress.log` | Validation loss before and after training, the model check, step-by-step training log |
| `tools/benchmark_train.py` | Training-throughput benchmark used to choose batch settings on a 36 GB Apple-silicon laptop |

## Main findings

- The report's model (base Mistral-7B v0.2) cannot follow the evaluation prompts: only 28% of its answers were scorable. We switched to Mistral-7B-Instruct-v0.2 (97%) and fine-tuned that.
- Fine-tuning (LoRA r=8 on q/k/v/o, 2 epochs, effective batch 16) lowered the held-out loss on the reports from 2.48 to 2.31.
- **DAT:** temperature barely matters (base 77.1 / 77.1 / 76.2). Fine-tuning lowers scores slightly (−1 to −1.4 points). Every condition scores below a list of random nouns (81.1), as Nakajima et al. also found.
- **CDAT:** higher temperature buys a little novelty at a cost in appropriateness. The fine-tuned model loses appropriateness faster.
- **Decoding matters as much as temperature:** with top-k 50, 92% of lists at T = 1.5 are scorable; with untruncated sampling only 2–10% are, and the rest is degenerate text.

## Running it

The notebook is written for Google Colab with a GPU. On an A100 (≥40 GB) it trains plain bf16 LoRA in about 30 minutes; on smaller GPUs it falls back to 4-bit QLoRA.

1. Put this repository in Google Drive as `MyDrive/NLP Take Home Assignment/`.
2. Add `erowid_experiences.zip`: the `Experiences/` folder of the Erowid corpus, zipped. It is not included here because the corpus belongs to its owners.
3. Add `nltk_data.zip`, containing WordNet, the word list and the POS tagger. NLTK refuses to download through Colab's proxy, so the data has to be supplied. Create it locally:
   ```
   python -m nltk.downloader -d nltk_data wordnet words averaged_perceptron_tagger_eng
   cd nltk_data && zip -r ../nltk_data.zip corpora taggers
   ```
4. Open the notebook in Colab, choose a GPU runtime and run all cells.

Because the adapter, the model check and the raw generations are included, a re-run reuses them instead of retraining or re-sampling. Delete `artifacts/` and `stimuli&outputs/raw_generations/` to reproduce everything from scratch (seed 111). To run locally instead: `pip install -r requirements.txt`, then execute the notebook from the repository folder.

## References

- Nakajima, Zuiderveld & Pezzelle (2026). *Beyond Divergent Creativity: A Human-Based Evaluation of Creativity in Large Language Models.*
- Ciaccio, Miaschi & Dell'Orletta (2025). *Evaluating Lexical Proficiency in Neural Language Models.*
- Jiang et al. (2023). *Mistral 7B.*
- Hu et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models.*
