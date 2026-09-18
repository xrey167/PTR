# PTR Fuzzing

Fuzz targets live outside the main Cargo workspace so fuzz-only dependencies do not affect production builds. The first target covers arbitrary PodWire bytes through Prost decoding and validated domain conversion.
