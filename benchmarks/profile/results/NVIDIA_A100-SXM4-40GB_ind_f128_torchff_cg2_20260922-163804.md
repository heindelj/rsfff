| structure | ff_backend | frames | n_atoms | n_pairs | forward_ms | forward_forces_ms | train_step_ms | train_peak_gb | neighbor_list | projector_features | parameter_network | gates | elst_pairs | pauli_pairs | disp_pairs | bonded | coupled_solve |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| w6_mp2_avtz | torchff | 128 | 18 | 19584 | 90.6 | 156.7 | 305.4 | 1.2 | 5.6 | 20.2 | 22.1 | 1.6 | 0.2 | 0.1 | 0.2 | 0.7 | 49.8 |
| w21_mp2_avtz | torchff | 128 | 63 | 249984 | 156.4 | 230.5 | 392.8 | 18.3 | 5.4 | 46.0 | 21.9 | 0.6 | 0.2 | 0.1 | 0.1 | 0.6 | 56.5 |

Forward split column unit: cuda_ms per call (forward only; backward is in the whole-call columns). train_step = E+F loss with create_graph=True + backward.

### w6_mp2_avtz training step (torchff, 128 frames): wall per phase, kernel-busy ms by forward region

| region | forward | first_backward | second_backward | total |
|---|---|---|---|---|
| **wall (synced)** | **93.9** | **50.1** | **153.0** | **297.0** |
| kernel busy | 15.2 | 9.8 | 39.9 | 64.9 |
| neighbor_list | 0.4 | 0.1 | 0.1 | 0.6 |
| projector_features | 3.5 | 4.0 | 9.7 | 17.2 |
| parameter_network | 2.9 | 3.8 | 18.3 | 25.0 |
| gates | 0.3 | 0.2 | 1.4 | 1.9 |
| elst_pairs | 0.0 | 0.1 | 0.7 | 0.8 |
| pauli_pairs | 0.0 | 0.0 | 0.2 | 0.2 |
| disp_pairs | 0.0 | 0.0 | 0.0 | 0.1 |
| bonded | 0.1 | 0.0 | 0.1 | 0.2 |
| coupled_solve | 5.9 | 0.4 | 7.1 | 13.4 |
| other | 2.0 | 1.1 | 2.3 | 5.4 |

second backward, top autograd nodes (kernel-busy ms, inclusive): IndexBackward0 7.3, _CoupledSolveBackward 5.9, BmmBackward0 5.3, MulBackward0 4.9, AddmmBackward0 3.5, EmbeddingBackward0 1.8, MmBackward0 1.4, _EnergyGradBackward 1.1, DivBackward0 1.0, SelectBackward0 1.0, SliceBackward0 0.9, torch::autograd::CopySlices 0.8

### w21_mp2_avtz training step (torchff, 128 frames): wall per phase, kernel-busy ms by forward region

| region | forward | first_backward | second_backward | total |
|---|---|---|---|---|
| **wall (synced)** | **152.5** | **53.0** | **183.2** | **388.7** |
| kernel busy | 96.0 | 26.0 | 125.8 | 247.7 |
| neighbor_list | 0.6 | 0.2 | 0.3 | 1.1 |
| projector_features | 37.6 | 12.1 | 34.7 | 84.5 |
| parameter_network | 6.1 | 8.3 | 40.2 | 54.5 |
| gates | 0.4 | 0.3 | 3.1 | 3.8 |
| elst_pairs | 0.1 | 1.1 | 9.9 | 11.1 |
| pauli_pairs | 0.1 | 0.2 | 2.2 | 2.5 |
| disp_pairs | 0.0 | 0.1 | 0.1 | 0.2 |
| bonded | 0.1 | 0.1 | 0.1 | 0.2 |
| coupled_solve | 13.3 | 1.5 | 29.6 | 44.3 |
| other | 37.8 | 2.2 | 5.5 | 45.5 |

second backward, top autograd nodes (kernel-busy ms, inclusive): IndexBackward0 24.4, _CoupledSolveBackward 19.4, _EnergyGradBackward 16.5, BmmBackward0 16.4, MulBackward0 12.7, _FieldBackward 9.1, AddmmBackward0 6.1, torch::autograd::CopySlices 3.8, _EnergyBackward 2.9, MmBackward0 2.8, SelectBackward0 2.5, AsStridedBackward0 2.3

wall - kernel busy = launch/dispatch/sync idle in that phase. Region attribution follows autograd sequence numbers (see attribute_train_step).
