# First Review - Slide Outline (slides 12 onwards)

Your existing 11 slides (introduction, motivation, literature, gap, problem, objectives, methodology, evaluation, references) stay as they are.
Add the slides below after the **Drift-Aware Evaluation** slide (slide 10) and before **References**. All figures are in `results/figures/`.
Every number comes from the untuned, default-setting runs in this repository. **Say "preliminary" and "untuned" out loud.**

---

## Slide 12 - Data and Drift Evidence
**Figures:** `03_ethanol_drift_heatmap.png`, `04_pca_batch_gas.png`

* 13,910 samples, 10 chronological batches, 16 sensors x 8 features = 128 features, 6 gases
* Same gas (Ethanol), different months: 36% of sensor-batch cells respond more than 2x differently from batch 1
* Sensor S01 falls to about 1/24 of its batch-1 response by batch 8
* In the PCA plot, Ethanol samples separate by batch

**Say:** "The same gas gives different sensor readings as the sensors age. This is the drift we want to handle. Features also differ by orders of magnitude, so scaling is mandatory."

## Slide 13 - Experimental Setup (all untuned)
**Figure:** `08_protocols.png`

* Models: kNN, SVM, Decision Tree, Random Forest, Naive Bayes, AdaBoost, Gradient Boosting, stacking (all scikit-learn defaults)
* Pipelines: scaling only / PCA (95% variance) / LDA (5 axes), always fitted on training batches only
* Protocols: **P1** train batches 1-3, test 4-10 (headline); P2 train batch 1; P3 rolling retraining
* Metrics: accuracy, macro-precision, macro-recall, **macro-F1**, confusion matrix

**Say:** "Macro-F1 gives every gas equal weight, which matters because some gases are rare in some batches. We never train on the future."

## Slide 14 - Why a Random Split Misleads
**Figure:** `21_random_vs_timeaware.png`

* A Random Forest predicts the batch (time period) from the readings with 99.2% accuracy; chance is 26%
* Random split vs time-aware macro-F1: SVM 97.6% to 69.0%; kNN 99.0% to 60.9%; Random Forest 99.5% to 51.8%
* The gap is 10 to 56 points across the seven models

**Say:** "Samples from the same batch are near-identical twins, so a random split lets the model memorise them. This inflates accuracy and hides drift - the concern raised by Dennler et al., 2022."

## Slide 15 - Preliminary Results (P1)
**Figure:** `14_baselines_heatmap.png` (optional) and this table

| Model (best pipeline) | Accuracy | Macro-F1 |
|---|---|---|
| **SVM (scaling only)** | **70.5%** | **69.0%** |
| Random Forest (LDA) | 64.7% | 62.5% |
| Naive Bayes (LDA) | 63.2% | 62.1% |
| kNN (LDA) | 64.0% | 61.0% |
| AdaBoost (PCA) | 65.7% | 59.4% |
| Gradient Boosting (PCA) | 62.4% | 57.0% |
| Decision Tree (LDA) | 59.3% | 56.7% |
| *No-skill floor* | *18.4%* | *5.5%* |

* Rolling retraining (P3): SVM again best, 82.4% accuracy / 79.7% macro-F1
* Training on batch 1 only (P2): every model weak; no model above 49% accuracy

**Say:** "These are default settings. Rankings are descriptive, and some test batches are small, so differences of a few points are within noise."

## Slide 16 - Performance Degradation Over Time
**Figures:** `22_degradation_top_models.png`, `16_near_vs_far.png`

* Skill falls as test batches move away from the training period, unevenly
* Near (batches 4-5) vs far (batches 8-10) macro-F1: SVM 93.0% to 56.6%; kNN 77.6% to 51.1%; Naive Bayes 82.8% to 20.2%
* Distance-based models (SVM, kNN) hold up best

**Say:** "A model that looks excellent on near-term data can be the worst on distant data."

## Slide 17 - PCA vs LDA
**Figures:** `23_pca_lda_effect.png`, `11_subspace_drift.png`

* PCA keeps 95% of the variance with 11 of 128 components
* On P1, PCA/LDA raise trees, Naive Bayes and boosting by up to 16 macro-F1 points but lower SVM
* LDA's gas clusters blur over time (silhouette 0.60-0.82 on training batches, -0.07 on batch 10) and LDA collapses when trained on one batch
* No universal winner: the effect depends on the model and the protocol

**Say:** "PCA ignores the gas labels; LDA uses them to separate the gases. LDA fits the training period more tightly, so it is less robust to drift."

## Slide 18 - Stacking Ensemble (preliminary, negative)
**Figures:** `17_stacking_heatmap.png`, `20_stacking_confusions.png`

* Stacking = five models + a logistic-regression meta-learner, trained on leave-one-batch-out predictions to avoid leakage
* P1 macro-F1: leakage-safe stacking 54.0% (scaling only; best pipeline 60.0%) vs SVM 69.0%
* Diagnostic: out-of-fold predictions are very uneven (SVM 53.7% accurate when batch 1 is held out vs 97.3% for batch 2)
* Next: forward-chaining folds, tuning, the rolling protocol

**Say:** "With default settings the ensemble does not beat the best single model. We understand why, and that tells us how to redesign it."

## Slide 19 - A Hidden Factor: Toluene
**Figures:** `26_toluene_effect.png`, `25_svm_per_gas_recall.png`

* Toluene is only 79 of 3,275 training samples (2.4%), 74 of them from batch 1
* No model recognises Toluene on later batches (recall at most 12%; SVM, kNN, Random Forest never predict it)
* Excluding it from the average lifts SVM from 69.0% to 77.4% macro-F1: part of the P1 penalty is data scarcity, not drift
* Drift is still real: SVM accuracy on non-Toluene samples is 51.7% on batch 10

**Say:** "We found this ourselves by looking at per-gas results. The final evaluation will report per gas and include a protocol where Toluene is present in training."

## Slide 20 - Limitations and Next Steps
* Limitations: untuned defaults; one seed; small, imbalanced batches (4, 5, 8); no confidence intervals yet; this dataset copy has no raw baselines, so the leakage check is indirect
* Next (final phase): hyperparameter tuning with validation inside the training batches; drift mitigation (sliding window, per-batch standardisation, recency weighting, CORAL); sensor/feature interpretability;
  redesigned stacking; bootstrap confidence intervals and significance tests; final report

---

### Questions your Miss may ask (and honest answers)

* **Why is accuracy so much lower than papers that report 95%+?** Those usually use random splits or train on much more recent data. Our random-split numbers are also above 97%; the time-aware ones are the realistic ones.
* **Why no tuning?** Preliminary phase by design; tuning must use validation inside the training batches, never the test batches.
* **Is SVM really the best?** It is best on average here, but kNN beats it on batch 10 and the ranking changes under P2. Significance tests come in the final phase.
* **Does stacking work?** Not yet with defaults; we know why and have a concrete redesign.
