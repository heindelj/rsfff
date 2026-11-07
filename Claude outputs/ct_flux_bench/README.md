# Flux-CT cost benchmark (2026-11-05)

CPU (2 threads, float64), untrained model (test_complete_flux settings + xpol_perp, scalefree kappa), jittered
rock-salt NaCl clusters, CT channel cutoff 8 A (params pf2p). Run from this folder with rsfff + bb importable:

    python bench_flux.py [--atomic] [--ct direct] --sizes 2,8,27,64,125 --modes fwd,train,force
    python bench_batch.py 12,15,18,20,22,25     # a 30-frame training batch, channel vs atomic solve
    python check_atomic.py                      # atomic solve == channel solve (E, q, z, forces, param grads)
    python channels.py 5,5,5                    # per-channel energies by type/distance, cutoff sweep

flux_atomic.py: drop-in flux_operator / flux_solve that eliminate the channels exactly and solve an N x N system in
atomic-charge space (see its docstring). Agrees with the channel-space solve to 1e-14 relative.

Single frame, seconds (fwd = no-grad energy; force = energy + autograd forces):

| n | channels | fwd channel | fwd atomic | fwd direct-CT | force channel | force atomic | force direct-CT |
|---|---|---|---|---|---|---|---|
| 27 | 626 | 0.056 | 0.035 | 0.026 | 0.104 | 0.073 | 0.061 |
| 64 | 2244 | 0.297 | 0.123 | 0.106 | 0.642 | 0.275 | 0.223 |
| 100 | 4046 | 1.448 | 0.296 | 0.232 | 2.830 | 0.538 | 0.495 |
| 125 | 5434 | 3.023 | 0.385 | 0.347 | 6.236 | 0.866 | 0.728 |

30-frame training batch (2-25 ions, Cmax 532 after padding): energy step 0.62 -> 0.16 s, force step 2.25 -> 0.60 s.
