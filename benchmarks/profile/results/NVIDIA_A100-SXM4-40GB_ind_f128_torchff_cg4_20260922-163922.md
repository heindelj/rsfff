| structure | ff_backend | frames | n_atoms | n_pairs | forward_ms | forward_forces_ms | train_step_ms | train_peak_gb | neighbor_list | projector_features | parameter_network | gates | elst_pairs | pauli_pairs | disp_pairs | bonded | coupled_solve |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| w6_mp2_avtz | torchff | 128 | 18 | 19584 | 91.4 | 164.9 | 317.6 | 1.2 | 5.3 | 20.5 | 21.6 | 1.7 | 0.2 | 0.1 | 0.0 | 0.5 | 48.7 |
| w21_mp2_avtz | torchff | 128 | 63 | 249984 | 159.6 | 236.6 | 404.2 | 18.3 | 5.7 | 45.0 | 23.5 | 0.5 | 0.2 | 0.1 | 0.0 | 0.6 | 62.5 |

Forward split column unit: cuda_ms per call (forward only; backward is in the whole-call columns). train_step = E+F loss with create_graph=True + backward.

### w6_mp2_avtz training step (torchff, 128 frames): wall per phase, kernel-busy ms by forward region

| region | forward | first_backward | second_backward | total |
|---|---|---|---|---|
| **wall (synced)** | **92.6** | **52.2** | **162.1** | **307.0** |
| kernel busy | 15.4 | 9.8 | 40.6 | 65.8 |
| neighbor_list | 0.4 | 0.1 | 0.1 | 0.6 |
| projector_features | 3.5 | 3.9 | 9.6 | 17.1 |
| parameter_network | 2.9 | 3.8 | 18.3 | 25.0 |
| gates | 0.3 | 0.2 | 1.4 | 1.9 |
| elst_pairs | 0.0 | 0.1 | 0.7 | 0.8 |
| pauli_pairs | 0.0 | 0.0 | 0.2 | 0.2 |
| disp_pairs | 0.0 | 0.0 | 0.0 | 0.1 |
| bonded | 0.1 | 0.0 | 0.1 | 0.2 |
| coupled_solve | 6.1 | 0.4 | 7.9 | 14.4 |
| other | 2.0 | 1.1 | 2.3 | 5.4 |

second backward, top autograd nodes (kernel-busy ms, inclusive): IndexBackward0 7.3, _CoupledSolveBackward 6.7, BmmBackward0 5.2, MulBackward0 4.9, AddmmBackward0 3.5, EmbeddingBackward0 1.8, MmBackward0 1.4, _EnergyGradBackward 1.1, DivBackward0 1.0, SelectBackward0 1.0, SliceBackward0 0.8, torch::autograd::CopySlices 0.8

### w21_mp2_avtz training step (torchff, 128 frames): wall per phase, kernel-busy ms by forward region

| region | forward | first_backward | second_backward | total |
|---|---|---|---|---|
| **wall (synced)** | **160.4** | **55.0** | **185.2** | **400.6** |
| kernel busy | 94.9 | 26.5 | 140.4 | 261.8 |
| neighbor_list | 0.6 | 0.2 | 0.3 | 1.1 |
| projector_features | 38.0 | 12.7 | 34.2 | 84.9 |
| parameter_network | 6.1 | 8.2 | 48.0 | 62.3 |
| gates | 0.4 | 0.3 | 3.1 | 3.8 |
| elst_pairs | 0.1 | 1.1 | 10.6 | 11.8 |
| pauli_pairs | 0.1 | 0.2 | 2.7 | 3.0 |
| disp_pairs | 0.0 | 0.1 | 0.1 | 0.2 |
| bonded | 0.1 | 0.1 | 0.1 | 0.2 |
| coupled_solve | 14.9 | 1.5 | 35.3 | 51.7 |
| other | 34.7 | 2.2 | 5.9 | 42.8 |

second backward, top autograd nodes (kernel-busy ms, inclusive): IndexBackward0 29.0, _CoupledSolveBackward 22.8, _EnergyGradBackward 20.1, BmmBackward0 19.0, MulBackward0 12.5, _FieldBackward 11.0, AddmmBackward0 6.1, MmBackward0 2.8, torch::autograd::CopySlices 2.5, SelectBackward0 2.5, _PauliGradBackward 2.4, _EnergyBackward 2.3

wall - kernel busy = launch/dispatch/sync idle in that phase. Region attribution follows autograd sequence numbers (see attribute_train_step).
