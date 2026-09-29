# Reproduction details and remaining gaps

## Source mapping

| Repository component | Original source |
|---|---|
| Model, FX16 simulation, preprocessing | `Ekdum_final_for_paper/vit_base/ Kendall's Tau_base_model.ipynb`, code cells 6/7/10/14/16/19 (zero-based); tiny/small share the architecture with different dimensions |
| Final accuracy records | Selected 10,000-image final records in `Ekdum_final_for_paper/vit_{base,small,tiny}/*.txt`; relative source paths and SHA-256 hashes retained in CSV |
| Center-bias / translation code | `revision_kendaall_tau/revision_kendall_tau.ipynb`, cells 28 and 30 |
| Revision summary data | `revision_kendaall_tau/center_bias_analysis/imagenet_kta_center_bias_summary.csv` and `kta_translation_center_bias/kta_translation_summary.csv` |
| Gradient attribution formula | `VISUALIZATION/gredient_for_inference.ipynb`, cell 14: absolute gradient×token sum over embedding dimensions |
| Paper tables and source figures | Supplied `paper.pdf`, SHA-256 in `figures/paper/provenance.json` |

The repository is self-contained after installing dependencies and supplying evaluation inputs. None of its commands depends on the original workspace directories.

## Behavioral choices

1. **Kendall statistic:** notebook code computes `(concordant − discordant) / choose(D,2)`, whereas the paper describes tau-b. The default `--tau notebook` preserves the source formula. `--tau tau-b` enables tie-corrected scores but retains the notebook's threshold selection. It is an explicitly different experiment, not a claim to have replicated all pseudocode edge cases. All-tied tau-b inputs use a documented neutral score of zero.
2. **Selection boundary:** the notebook retains scores `<= kthvalue(floor(N*keep_ratio)+1)`, including ties; actual retention can exceed the nominal ratio. The paper instead uses a strict comparison to a descending threshold. Full retention previously requested an out-of-range kth value; this repository fixes that boundary. At keep ratio zero the historical rule still retains at least one patch.
3. **ATS:** the quantile includes CLS attention; CLS is always retained. Tokens are pruned after the block's MLP. Both behaviors preserve the source notebooks.
4. **Preprocessing:** nearest-neighbor resize to 224×224, BGR→RGB, and conversion to `[0,1]`, with no mean/std normalization. This is deliberately the notebook pipeline, even though timm's pretrained configuration recommends different transforms. `nn.LayerNorm` retains the notebook's default epsilon. Models now run in explicit evaluation mode with inference autograd disabled.
5. **Quantization:** QPyTorch `FixedPoint(16,8)` rounds module weights and hooked activations, input, and output. This is floating-point simulation of fixed-point values, not integer storage. Original notebook labels saying INT8 were corrected. The learned standalone CLS/position parameters are not explicitly quantized by the original module traversal.
6. **Memory:** upper-triangle feature comparisons are evaluated in patch chunks; tests verify equivalence to the full notebook matrix calculation.
7. **Reproducibility:** random seeds, sorted image order, explicit manifests, package versions and checkpoint identity are recorded for new evaluation runs. Revision scripts retain their sorted source traversal. Their new random seed may differ from the historical unrecorded random baseline.

## What is still needed for exact paper accuracy

- The ordered image identifiers for the original 10,000-image evaluation subset and 1,000-image revision subset.
- The exact pretrained checkpoint(s) or their hashes and original timm version. The supplied notebooks specify untagged model names. The repository pins the currently inspected timm version, not an unverified historical version.
- Confirmation of whether the published runs used the saved notebook statistic or the paper's tau-b variant, and the exact rounding/tie policy.
- The missing matching-budget configurations for Table III and original machine-readable arrays for all Figure 6/7 curves. New sweeps produce measurable curves; existing paper Figures 6/7 are extracted exactly rather than tracing or fabricating missing values.
- ToMe and ToFu reimplementation code is not present in this workspace. Their Table VI numbers are labelled as reported comparisons; their experiments are not implemented here.

## Observed inconsistencies

- Thirty selected notebook configurations match Table IV by model and pruning ratios. Twenty-nine accuracy values agree; the ViT-Small baseline log ends at **78.15%**, while the paper reports **78.14%**. Both values are preserved with their provenance.
- Table II reports 9.7M / 30.1M / 86.0M parameters for Tiny / Small / Base. These are preserved as reported values rather than asserted to be counts of the constructed timm models.
- The translation prose mentions offsets as far as ±64 pixels, but the available notebook and summary evaluate **−48, −32, −16, 0, 16, 32, 48**. These observed settings are retained.
- Figure 7's exact underlying per-block arrays are not available for all configurations. `plot_runs.py` labels *tokens entering each block* explicitly to avoid silently mixing pre-block and post-block counts.

## Verification scope

Five unit tests passed, as did an offline pretrained-shaped ViT-Tiny FP32 forward, a small FX16 forward, and CLI smoke runs using a random local ViT-Tiny checkpoint with a synthetic image. Full ImageNet inference, numerical equality to the reported accuracies, a clean pip installation, and GPU execution were not performed. No synthetic outputs are included as scientific results.
