"""Cross-Modal Reasoning (CMR): training and evaluation code for NTU FYP CCDS25-1134.

Package map:

    cmr.data        -> A-OKVQA items and the image-grouped dev split
    cmr.eval        -> prompts, image conditions, metrics, statistics
    cmr.models      -> letter scorers behind one protocol: the bridge model (SigLIP -> bridge
                       -> Qwen2), other Hugging Face models, and fake models for tests
    cmr.train       -> bridge training: training targets, loss and loop, generation
    cmr.runners     -> command-line entry points: cmr-stage-a (score a model on A-OKVQA
                       multiple choice), cmr-stage-b (train a bridge, then score it the same
                       way), cmr-transfer (direct answers and other benchmarks)
    cmr.analysis    -> per-question records -> summaries and tables
    cmr.fsutil      -> crash-safe file writes and JSONL reads
    cmr.provenance  -> run manifest: code, packages, data hashes, weights, threads, CPU
"""

__version__ = "0.1.0"
