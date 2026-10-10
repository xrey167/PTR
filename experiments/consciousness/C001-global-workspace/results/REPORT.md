# C001 global workspace: NOT_SHOWN

Seeds: 4001, 4002, 4003, 4004, 4005, 4006, 4007, 4008, 4009, 4010, 4011, 4012, 4013, 4014, 4015, 4016, 4017, 4018, 4019, 4020. Every seed must hold every coverage and falsifier rule; the paired rule must hold across the seeds.

- seed 4001: valid; 0 coverage rules unmet, 0 falsifier rules tripped, 0 integrity counters above 0
- seed 4002: valid; 0 coverage rules unmet, 0 falsifier rules tripped, 0 integrity counters above 0
- seed 4003: valid; 0 coverage rules unmet, 0 falsifier rules tripped, 0 integrity counters above 0
- seed 4004: valid; 0 coverage rules unmet, 0 falsifier rules tripped, 0 integrity counters above 0
- seed 4005: valid; 0 coverage rules unmet, 0 falsifier rules tripped, 0 integrity counters above 0
- seed 4006: valid; 0 coverage rules unmet, 0 falsifier rules tripped, 0 integrity counters above 0
- seed 4007: valid; 0 coverage rules unmet, 0 falsifier rules tripped, 0 integrity counters above 0
- seed 4008: valid; 0 coverage rules unmet, 0 falsifier rules tripped, 0 integrity counters above 0
- seed 4009: valid; 0 coverage rules unmet, 0 falsifier rules tripped, 0 integrity counters above 0
- seed 4010: valid; 0 coverage rules unmet, 0 falsifier rules tripped, 0 integrity counters above 0
- seed 4011: valid; 0 coverage rules unmet, 0 falsifier rules tripped, 0 integrity counters above 0
- seed 4012: valid; 0 coverage rules unmet, 0 falsifier rules tripped, 0 integrity counters above 0
- seed 4013: valid; 0 coverage rules unmet, 0 falsifier rules tripped, 0 integrity counters above 0
- seed 4014: valid; 0 coverage rules unmet, 0 falsifier rules tripped, 0 integrity counters above 0
- seed 4015: valid; 0 coverage rules unmet, 0 falsifier rules tripped, 0 integrity counters above 0
- seed 4016: valid; 0 coverage rules unmet, 0 falsifier rules tripped, 0 integrity counters above 0
- seed 4017: valid; 0 coverage rules unmet, 0 falsifier rules tripped, 0 integrity counters above 0
- seed 4018: valid; 0 coverage rules unmet, 0 falsifier rules tripped, 0 integrity counters above 0
- seed 4019: valid; 0 coverage rules unmet, 0 falsifier rules tripped, 0 integrity counters above 0
- seed 4020: valid; 0 coverage rules unmet, 0 falsifier rules tripped, 0 integrity counters above 0


Paired comparisons (workspace minus comparator, per mille; one-sided 95% t bounds against the margin):
- ood16 vs all_to_all: mean 27.3, sd 109.6, lower -15.0, upper 69.7, margin 30: straddles
- ood16 vs bw_all_to_all: mean 18.6, sd 107.2, lower -22.9, upper 60.1, margin 30: straddles
- ood16 vs sparse_all_to_all: mean 132.3, sd 173.6, lower 65.2, upper 199.4, margin 30: supports
- gross sensitivity control (all_to_all minus crippled_all_to_all): mean 394.0, lower 384.6: resolves
- graded sensitivity control (all_to_all minus weak_all_to_all): mean 61.5, lower 42.1: resolves
- attribution contrast vs no_broadcast (margin 30): mean 13.8, median 34.7, 13 of 20 seeds positive, half-width 40.5, lower -26.7, upper 54.3: UNRESOLVED BY NOISE
- attribution contrast vs capacity_off (margin 30): mean -14.1, median 2.0, 11 of 20 seeds positive, half-width 36.5, lower -50.6, upper 22.4: NOT ATTRIBUTED
- attribution contrast vs competition_off (margin 30): mean -29.2, median -11.2, 7 of 20 seeds positive, half-width 39.7, lower -68.8, upper 10.5: NOT ATTRIBUTED
- (descriptive) id vs all_to_all: mean -18.5, lower -60.5, upper 23.6
- (descriptive) id vs bw_all_to_all: mean -6.9, lower -50.0, upper 36.1
- (descriptive) id vs sparse_all_to_all: mean 122.3, lower 48.6, upper 196.1
- (descriptive) id vs no_comm: mean 482.3, lower 439.4, upper 525.1
- (descriptive) id vs no_broadcast: mean 14.6, lower -25.0, upper 54.1
- (descriptive) ood24 vs all_to_all: mean 80.5, lower 34.1, upper 126.9
- (descriptive) ood24 vs bw_all_to_all: mean 51.0, lower 7.5, upper 94.6
- (descriptive) ood24 vs sparse_all_to_all: mean 130.8, lower 67.3, upper 194.3
- (descriptive) ood24 vs no_comm: mean 391.2, lower 348.8, upper 433.7
- (descriptive) ood24 vs no_broadcast: mean 17.2, lower -24.7, upper 59.0
- (descriptive) auc vs all_to_all: mean -65.5, lower -94.4, upper -36.7
- (descriptive) auc vs bw_all_to_all: mean -42.6, lower -75.1, upper -10.1
- (descriptive) auc vs sparse_all_to_all: mean 129.5, lower 83.9, upper 175.0
- (descriptive) auc vs no_comm: mean 292.1, lower 260.6, upper 323.6
- (descriptive) auc vs no_broadcast: mean 23.0, lower -8.6, upper 54.5
- (descriptive) drop_irrelevant vs all_to_all: mean -19.0, lower -59.0, upper 21.0
- (descriptive) drop_irrelevant vs bw_all_to_all: mean -7.3, lower -48.6, upper 34.1
- (descriptive) drop_irrelevant vs sparse_all_to_all: mean 119.1, lower 46.4, upper 191.9
- (descriptive) drop_irrelevant vs no_comm: mean 488.3, lower 449.1, upper 527.5
- (descriptive) drop_irrelevant vs no_broadcast: mean 10.9, lower -25.0, upper 46.8
- (descriptive) drop_relevant vs all_to_all: mean -2.5, lower -8.8, upper 3.7
- (descriptive) drop_relevant vs bw_all_to_all: mean -4.2, lower -9.8, upper 1.5
- (descriptive) drop_relevant vs sparse_all_to_all: mean -1.3, lower -7.3, upper 4.6
- (descriptive) drop_relevant vs no_comm: mean -12.2, lower -19.0, upper -5.4
- (descriptive) drop_relevant vs no_broadcast: mean -2.2, lower -7.8, upper 3.4
- mutation check: 38 of 38 killed

Limitations: a result applies to this task family, its size (eight training modules, d=48, three rounds), the training-set size chosen on pilot seeds and these hyperparameters (the same learning-rate x width grid for every arm, selected on the primary metric over five pilot seeds disjoint from the declared ones; the grid was extended by lr 0.006 at d 48 for every arm after every arm had selected the edge 0.003, and by lr 0.012 for the four arms that then selected 0.006 (the pairwise comparators and competition_off), both extensions run on pilot seeds after the pilot comparisons had been seen; 0.012 won for none of them (the networks did not train), so every arm's selected rate is interior to the tested grid at d 48, but the grid is geometric and a rate between two tested ones or at d 32 was not tried, so an arm may still be under-tuned; d = 48 is the upper edge of the width grid {32, 48} for six of seven arms and d = 64 was not tried, the workspace gaining most from width, direction of the bias unknown); the workspace had extra structural choices (slots, top-k, rounds) made from pilot observations; arms are trained to a validation plateau and a converged arm is assumed to be at its optimum, which a bounded patience cannot prove; the plateau protocol can stop an arm prematurely (a run that stalls for patience evaluations before a late jump is flagged converged): a stalled comparator inflates the workspace advantage, a stalled workspace deflates it; the guards catch only the extreme cases (band, untrained workspace), and the per-seed differences are heavy-tailed, so the median and the number of seeds with a positive difference are reported beside the t bounds; the t bounds assume roughly normal paired differences, which twenty points cannot check; per-seed differences and both bounds are reported; the decision uses the upper bound for refutation and the lower bound for support; the 30 per mille margin is justified by the cost of building the learned workspace but was fixed before the freeze and after the pilots had been seen (not fitted to them; an earlier version used the pilot gap, 50); disclosed; the ablations differ from the workspace arm in more than one way at matched parameter counts (update-network width); the no-broadcast arm keeps one reader (the reporter) so that the readout path is shared; the bandwidth-matched arm is one definition of matched bandwidth; a learned-attention network is not the deterministic bid-ordered workspace of ptr-pods::workspace: a supportive result does not validate that contract and a refuting one does not touch it; in distribution (eight modules) the pairwise arms can express everything the workspace can; the claim tested is an inductive-bias one (finite data, more modules), not an expressivity one; sample efficiency (area under the learning curve), module dropout, ID accuracy, the 24-module accuracy, ignition and communication-cost proxies are reported with bounds where they are differences and are never judged; mutation evidence comes from run_mutations.py (scripts/mutation_check.py plants Cargo packages only) and was run at a reduced scale with the locked uv environment's interpreter (the c001-venv python of the preregistration, section 8); experiment_records tracks only Rust and Cargo files as code, so this aggregator separately refuses when research/consciousness or this experiment's files differ from the mutation commit, and when the mutation total is not the plan's size; COMPLETED shows an advantage over pairwise attention at 2048 training examples and 16 modules; it does not show that the broadcast read, the capacity limit or the competition is responsible (the attribution contrasts, which are not in the support conjunction, may read not attributed or unresolved), and FALSIFIED is scoped to the same setting (the pilot advantage grows with the training-set size); the workspace stalled (16-module accuracy near chance) on 1 of 17 pilot seeds without being flagged by the guard's harness rule (in-distribution accuracy at least the no-communication arm's plus 100 per mille); such a seed stays in the paired differences and counts against the workspace, and the guard tolerates up to 4 flagged seeds of 20 (more make the run INVALID), so the outcome probabilities are wide (P(COMPLETED) 0.02 to 0.69 depending on how the pilots are read; preregistration section 4); an indicator property here is a functional property: the experiment says nothing about whether any system is conscious.
