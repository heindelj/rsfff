| structure | ff_backend | frames | n_atoms | n_pairs | forward_ms | forward_forces_ms | train_step_ms | train_peak_gb | neighbor_list | projector_features | parameter_network | gates | elst_pairs | pauli_pairs | disp_pairs | bonded | coupled_solve |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| w6_mp2_avtz | torchff | 128 | 18 | 19584 | 73.5 | 132.9 | 267.5 | 1.2 | 5.0 | 20.2 | 20.9 | 1.4 | 0.2 | 0.1 | 0.0 | 0.5 | 31.6 |
| w21_mp2_avtz | torchff | 128 | 63 | 249984 | 136.4 | 204.4 | 361.3 | 18.3 | 4.9 | 46.1 | 22.0 | 0.6 | 0.2 | 0.1 | 0.3 | 0.1 | 36.6 |

Forward split column unit: cuda_ms per call (forward only; backward is in the whole-call columns). train_step = E+F loss with create_graph=True + backward.

### w6_mp2_avtz training step (torchff, 128 frames): wall per phase, kernel-busy ms by forward region

| region | forward | first_backward | second_backward | total |
|---|---|---|---|---|
| **wall (synced)** | **74.8** | **48.3** | **141.4** | **264.5** |
| kernel busy | 12.6 | 9.8 | 37.1 | 59.5 |
| neighbor_list | 0.4 | 0.1 | 0.1 | 0.6 |
| projector_features | 3.5 | 4.0 | 9.4 | 16.8 |
| parameter_network | 2.9 | 3.8 | 18.3 | 25.0 |
| gates | 0.3 | 0.2 | 1.4 | 1.9 |
| elst_pairs | 0.0 | 0.1 | 0.7 | 0.8 |
| pauli_pairs | 0.0 | 0.0 | 0.2 | 0.2 |
| disp_pairs | 0.0 | 0.0 | 0.0 | 0.1 |
| bonded | 0.1 | 0.0 | 0.1 | 0.2 |
| coupled_solve | 3.4 | 0.4 | 4.7 | 8.5 |
| other | 2.0 | 1.1 | 2.3 | 5.4 |

second backward, top autograd nodes (kernel-busy ms, inclusive): IndexBackward0 7.2, MulBackward0 4.9, BmmBackward0 4.9, AddmmBackward0 3.5, _CoupledSolveBackward 3.5, EmbeddingBackward0 1.9, MmBackward0 1.4, _EnergyGradBackward 1.1, DivBackward0 1.0, SelectBackward0 1.0, SliceBackward0 0.9, torch::autograd::CopySlices 0.8

### w21_mp2_avtz training step (torchff, 128 frames): wall per phase, kernel-busy ms by forward region

| region | forward | first_backward | second_backward | total |
|---|---|---|---|---|
| **wall (synced)** | **140.1** | **51.6** | **165.7** | **357.4** |
| kernel busy | 91.3 | 27.0 | 146.1 | 264.5 |
| neighbor_list | 0.6 | 0.2 | 0.3 | 1.1 |
| projector_features | 38.1 | 13.2 | 34.3 | 85.6 |
| parameter_network | 6.1 | 8.2 | 54.3 | 68.6 |
| gates | 0.4 | 0.3 | 4.4 | 5.2 |
| elst_pairs | 0.1 | 1.1 | 11.2 | 12.4 |
| pauli_pairs | 0.1 | 0.2 | 2.6 | 2.9 |
| disp_pairs | 0.0 | 0.1 | 0.1 | 0.2 |
| bonded | 0.1 | 0.1 | 0.1 | 0.2 |
| coupled_solve | 9.3 | 1.5 | 32.7 | 43.5 |
| other | 36.6 | 2.1 | 6.1 | 44.8 |

second backward, top autograd nodes (kernel-busy ms, inclusive): IndexBackward0 33.9, _CoupledSolveBackward 21.3, _EnergyGradBackward 19.7, BmmBackward0 15.7, _FieldBackward 13.4, MulBackward0 12.6, AddmmBackward0 10.5, torch::autograd::CopySlices 4.0, MmBackward0 2.8, SelectBackward0 2.5, SoftplusBackward0 2.4, _PauliGradBackward 2.3

wall - kernel busy = launch/dispatch/sync idle in that phase. Region attribution follows autograd sequence numbers (see attribute_train_step).
