| structure | ff_backend | frames | n_atoms | n_pairs | forward_ms | forward_forces_ms | train_step_ms | train_peak_gb | neighbor_list | projector_features | parameter_network | gates | elst_pairs | pauli_pairs | disp_pairs | bonded | coupled_solve |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| w6_mp2_avtz | torchff | 128 | 18 | 19584 | 100.6 | 229.4 | 251.7 | 0.9 | 9.0 | 23.1 | 19.2 | 1.4 | 16.2 | 4.3 | 0.1 | 0.5 | 53.4 |
| w21_mp2_avtz | torchff | 128 | 63 | 249984 | 197.3 | 325.1 | 351.0 | 18.3 | 23.4 | 54.1 | 19.5 | 0.5 | 23.1 | 6.9 | 0.0 | 0.1 | 65.8 |

Forward split column unit: cuda_ms per call (forward only; backward is in the whole-call columns). train_step = energy-only loss + backward (no create_graph; the --loss energy ablation).

### w6_mp2_avtz training step (torchff, 128 frames): wall per phase, kernel-busy ms by forward region

| region | forward | second_backward | total |
|---|---|---|---|
| **wall (synced)** | **109.1** | **140.3** | **249.4** |
| kernel busy | 37.1 | 45.5 | 82.6 |
| neighbor_list | 6.4 | 0.2 | 6.5 |
| projector_features | 9.7 | 4.6 | 14.3 |
| parameter_network | 3.4 | 13.8 | 17.3 |
| gates | 0.4 | 1.2 | 1.5 |
| elst_pairs | 3.6 | 8.7 | 12.4 |
| pauli_pairs | 0.9 | 2.1 | 3.0 |
| disp_pairs | 0.0 | 0.0 | 0.0 |
| bonded | 0.1 | 0.1 | 0.1 |
| coupled_solve | 10.3 | 12.7 | 23.0 |
| other | 2.3 | 2.2 | 4.4 |

second backward, top autograd nodes (kernel-busy ms, inclusive): MulBackward0 13.1, IndexBackward0 10.7, AddmmBackward0 3.8, _CoupledSolveBackward 3.2, BmmBackward0 3.0, EmbeddingBackward0 2.1, NegBackward0 2.1, _StackBackward 1.5, SelectBackward0 1.4, SubBackward0 0.7, _FieldBackward 0.6, SliceBackward0 0.5

### w21_mp2_avtz training step (torchff, 128 frames): wall per phase, kernel-busy ms by forward region

| region | forward | second_backward | total |
|---|---|---|---|
| **wall (synced)** | **200.8** | **154.3** | **355.2** |
| kernel busy | 161.5 | 116.0 | 277.5 |
| neighbor_list | 19.9 | 0.3 | 20.3 |
| projector_features | 50.0 | 12.2 | 62.2 |
| parameter_network | 6.0 | 24.8 | 30.8 |
| gates | 0.4 | 2.4 | 2.8 |
| elst_pairs | 19.8 | 25.7 | 45.6 |
| pauli_pairs | 4.6 | 6.1 | 10.8 |
| disp_pairs | 0.0 | 0.1 | 0.1 |
| bonded | 0.1 | 0.0 | 0.1 |
| coupled_solve | 31.9 | 40.6 | 72.4 |
| other | 28.7 | 3.8 | 32.5 |

second backward, top autograd nodes (kernel-busy ms, inclusive): IndexBackward0 25.6, MulBackward0 22.8, BmmBackward0 16.6, _CoupledSolveBackward 13.8, _StackBackward 10.8, NegBackward0 8.9, _FieldBackward 7.2, AddmmBackward0 5.8, SelectBackward0 2.1, SubBackward0 1.4, torch::autograd::CopySlices 1.3, SliceBackward0 1.2

wall - kernel busy = launch/dispatch/sync idle in that phase. Region attribution follows autograd sequence numbers (see attribute_train_step).
