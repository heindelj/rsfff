| structure | ff_backend | frames | n_atoms | n_pairs | forward_ms | forward_forces_ms | train_step_ms | train_peak_gb | neighbor_list | projector_features | parameter_network | gates | elst_pairs | pauli_pairs | disp_pairs | bonded | coupled_solve |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| w6_mp2_avtz | torchff | 128 | 18 | 19584 | 100.8 | 226.6 | 248.7 | 0.9 | 9.0 | 23.3 | 19.4 | 1.3 | 16.4 | 4.4 | 0.1 | 0.5 | 54.2 |
| w21_mp2_avtz | torchff | 128 | 63 | 249984 | 200.3 | 333.4 | 354.3 | 18.3 | 23.7 | 54.4 | 19.7 | 0.5 | 23.1 | 6.9 | 0.0 | 0.1 | 65.7 |

Forward split column unit: cuda_ms per call (forward only; backward is in the whole-call columns). train_step = energy-only loss + backward (no create_graph; the --loss energy ablation).

### w6_mp2_avtz training step (torchff, 128 frames): wall per phase, kernel-busy ms by forward region

| region | forward | second_backward | total |
|---|---|---|---|
| **wall (synced)** | **107.3** | **138.2** | **245.6** |
| kernel busy | 36.2 | 44.5 | 80.7 |
| neighbor_list | 6.2 | 0.2 | 6.4 |
| projector_features | 9.5 | 4.5 | 14.0 |
| parameter_network | 3.3 | 13.6 | 16.9 |
| gates | 0.4 | 1.2 | 1.5 |
| elst_pairs | 3.6 | 8.5 | 12.1 |
| pauli_pairs | 0.9 | 2.0 | 2.9 |
| disp_pairs | 0.0 | 0.0 | 0.0 |
| bonded | 0.1 | 0.1 | 0.1 |
| coupled_solve | 10.0 | 12.4 | 22.4 |
| other | 2.2 | 2.1 | 4.4 |

second backward, top autograd nodes (kernel-busy ms, inclusive): MulBackward0 12.8, IndexBackward0 10.5, AddmmBackward0 3.7, _CoupledSolveBackward 3.1, BmmBackward0 3.0, EmbeddingBackward0 2.1, NegBackward0 2.0, StackBackward0 1.5, SelectBackward0 1.4, SubBackward0 0.7, _FieldBackward 0.6, SliceBackward0 0.5

### w21_mp2_avtz training step (torchff, 128 frames): wall per phase, kernel-busy ms by forward region

| region | forward | second_backward | total |
|---|---|---|---|
| **wall (synced)** | **200.8** | **154.5** | **355.2** |
| kernel busy | 161.5 | 116.2 | 277.7 |
| neighbor_list | 19.9 | 0.3 | 20.3 |
| projector_features | 50.0 | 12.2 | 62.2 |
| parameter_network | 6.0 | 24.9 | 30.9 |
| gates | 0.4 | 2.4 | 2.8 |
| elst_pairs | 19.9 | 25.7 | 45.6 |
| pauli_pairs | 4.7 | 6.1 | 10.8 |
| disp_pairs | 0.0 | 0.1 | 0.1 |
| bonded | 0.1 | 0.0 | 0.1 |
| coupled_solve | 31.9 | 40.6 | 72.4 |
| other | 28.7 | 3.9 | 32.5 |

second backward, top autograd nodes (kernel-busy ms, inclusive): IndexBackward0 25.6, MulBackward0 22.8, BmmBackward0 16.6, _CoupledSolveBackward 13.8, StackBackward0 10.8, NegBackward0 8.9, _FieldBackward 7.2, AddmmBackward0 5.8, SelectBackward0 2.1, SubBackward0 1.4, torch::autograd::CopySlices 1.3, SliceBackward0 1.2

wall - kernel busy = launch/dispatch/sync idle in that phase. Region attribution follows autograd sequence numbers (see attribute_train_step).
