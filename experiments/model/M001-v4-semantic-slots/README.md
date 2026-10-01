# M001-v4-semantic-slots

Clean same-freeze evidence run for semantic-slot fidelity against the matched
plain cross-attention baseline. M001-v2 remains historical and is not counted
as v4 evidence.

The `paired` Burn entrypoint trains the full A0 arm and the plain baseline in
one process with identical seed, optimizer, batch, learning rate and steps.
