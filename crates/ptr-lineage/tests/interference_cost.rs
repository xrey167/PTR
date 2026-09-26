//! The cost of measuring interference, pinned by the heap it takes rather
//! than by wall time, which depends on the machine. This file holds a single
//! test, so the counting allocator below sees no other test's allocations.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use ptr_lineage::{
    column_basis, measure_interference, subspace_overlap, AdapterId, LayerUpdate, Matrix,
};

/// The system allocator, counting the bytes live and the most live at once.
struct Counting;

static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

// SAFETY: every call is forwarded unchanged to the system allocator, which
// upholds the `GlobalAlloc` contract; the counters only observe sizes.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: the caller's obligations for `layout` are passed on as is.
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() {
            let live = LIVE.fetch_add(layout.size(), Ordering::SeqCst) + layout.size();
            PEAK.fetch_max(live, Ordering::SeqCst);
        }
        pointer
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        // SAFETY: `pointer` was returned by `alloc` above for this `layout`.
        unsafe { System.dealloc(pointer, layout) };
        LIVE.fetch_sub(layout.size(), Ordering::SeqCst);
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

/// Deterministic entries in `[-1, 1)`, as trained low-rank factors look.
fn entries(count: usize, seed: u64) -> Vec<f64> {
    let mut state = seed;
    (0..count)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 11) as f64 / (1u64 << 52) as f64 - 1.0
        })
        .collect()
}

fn update(d_out: usize, d_in: usize, rank: usize, seed: u64) -> LayerUpdate {
    LayerUpdate::new(
        "q_proj",
        Matrix::new(d_out, rank, entries(d_out * rank, seed)).unwrap(),
        Matrix::new(rank, d_in, entries(rank * d_in, seed + 1)).unwrap(),
    )
    .unwrap()
}

#[test]
fn the_bases_of_a_large_layer_are_computed_without_forming_its_product() {
    // A 2048 x 3072 layer at rank 4: the product B A alone is 48 MiB, and
    // taking bases from the formed product needs about twice that at peak.
    // From the factors, 160 KiB per update, the bases of the candidate and
    // both earlier updates take about 1 MiB at peak.
    let (d_out, d_in, rank) = (2048, 3072, 4);
    let candidate = update(d_out, d_in, rank, 1);
    let earlier = vec![
        (AdapterId::from("e1"), vec![update(d_out, d_in, rank, 3)]),
        (AdapterId::from("e2"), vec![candidate.clone()]),
    ];
    let product_bytes = d_out * d_in * std::mem::size_of::<f64>();

    let baseline = LIVE.load(Ordering::SeqCst);
    PEAK.store(baseline, Ordering::SeqCst);
    let (output, input) = (candidate.output_basis(), candidate.input_basis());
    let report = measure_interference(
        &AdapterId::from("candidate"),
        std::slice::from_ref(&candidate),
        &earlier,
    )
    .unwrap();
    let peak = PEAK.load(Ordering::SeqCst) - baseline;

    assert!(
        peak < product_bytes / 16,
        "{peak} bytes live at peak; the product alone takes {product_bytes}"
    );
    // The bases are those of the product, formed here only to check them.
    let product = candidate.delta_weight().unwrap();
    let transposed: Vec<f64> = (0..d_in * d_out)
        .map(|index| product.get(index % d_out, index / d_out))
        .collect();
    let transposed = Matrix::new(d_in, d_out, transposed).unwrap();
    for (basis, of_product) in [
        (&output, column_basis(&product)),
        (&input, column_basis(&transposed)),
    ] {
        assert_eq!((basis.rank(), of_product.rank()), (rank, rank));
        let overlap = subspace_overlap(basis, &of_product).unwrap();
        assert!((overlap - 1.0).abs() < 1e-12, "{overlap}");
    }
    // The candidate overlaps itself fully, and an unrelated update of the
    // same rank near the chance level of 4 / 2048.
    let layer = &report.layers[0];
    assert!((layer.output_overlap - 1.0).abs() < 1e-12, "{layer:?}");
    assert!((layer.input_overlap - 1.0).abs() < 1e-12, "{layer:?}");
    assert_eq!(layer.worst, Some(AdapterId::from("e2")));
    let unrelated = subspace_overlap(&output, &earlier[0].1[0].output_basis()).unwrap();
    assert!(unrelated < 0.01, "{unrelated}");
}
