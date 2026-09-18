# Data Governance

Every dataset must have a card with: source, ownership/license, collection/generation method, intended use, exclusions, sensitive-data assessment, preprocessing, train/validation/test split strategy and known limitations.

Raw evidence hashes and provenance are preserved where legally permitted. Training data derived from model outputs must identify the teacher/model and generation configuration. Evaluation data must not be silently reused for training.
