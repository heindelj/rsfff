| structure | ff_backend | frames | n_atoms | n_pairs | forward_ms | forward_forces_ms | train_step_ms | train_peak_gb | neighbor_list | projector_features | parameter_network | gates | elst_pairs | pauli_pairs | disp_pairs | bonded | coupled_solve |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| w6_mp2_avtz | torchff | 128 | 18 | 19584 | 63.9 | 135.8 | 343.7 | 1.4 | 9.0 | 23.1 | 19.1 | 1.3 | 16.6 | 4.8 | 0.1 | 0.2 | nan |
| w21_mp2_avtz | torchff | 128 | 63 | 249984 | 146.3 | 217.7 | 694.5 | 18.3 | 23.4 | 54.6 | 19.2 | 0.5 | 22.9 | 6.9 | 0.0 | 0.0 | nan |

Forward split column unit: cuda_ms per call (forward only; backward is in the whole-call columns). train_step = E+F loss with create_graph=True + backward.

### w6_mp2_avtz training step (torchff, 128 frames): wall per phase, kernel-busy ms by forward region

| region | forward | first_backward | second_backward | total |
|---|---|---|---|---|
| **wall (synced)** | **66.8** | **76.2** | **196.1** | **339.1** |
| kernel busy | 22.7 | 16.5 | 68.6 | 107.8 |
| neighbor_list | 5.3 | 0.1 | 0.2 | 5.7 |
| projector_features | 8.3 | 3.9 | 9.2 | 21.5 |
| parameter_network | 2.9 | 2.1 | 11.5 | 16.5 |
| gates | 0.3 | 0.2 | 1.5 | 2.0 |
| elst_pairs | 3.1 | 7.2 | 34.5 | 44.8 |
| pauli_pairs | 0.8 | 1.7 | 8.8 | 11.3 |
| disp_pairs | 0.0 | 0.0 | 0.0 | 0.1 |
| bonded | 0.0 | 0.0 | 0.1 | 0.1 |
| other | 1.9 | 1.2 | 2.8 | 5.9 |

second backward, top autograd nodes (kernel-busy ms, inclusive): MulBackward0 21.1, SelectBackward0 20.5, BmmBackward0 6.6, IndexBackward0 6.4, AddmmBackward0 2.8, NegBackward0 2.0, EmbeddingBackward0 1.1, MmBackward0 1.1, torch::autograd::CopySlices 0.8, StackBackward0 0.8, DivBackward0 0.7, SliceBackward0 0.6

### w21_mp2_avtz training step (torchff, 128 frames): wall per phase, kernel-busy ms by forward region

| region | forward | first_backward | second_backward | total |
|---|---|---|---|---|
| **wall (synced)** | **149.2** | **76.8** | **466.5** | **692.5** |
| kernel busy | 129.4 | 50.1 | 437.1 | 616.6 |
| neighbor_list | 19.9 | 0.3 | 0.5 | 20.7 |
| projector_features | 50.0 | 11.8 | 28.7 | 90.6 |
| parameter_network | 6.0 | 4.9 | 24.0 | 35.0 |
| gates | 0.4 | 0.3 | 3.1 | 3.8 |
| elst_pairs | 19.9 | 24.2 | 299.9 | 343.9 |
| pauli_pairs | 4.6 | 6.2 | 75.1 | 85.9 |
| disp_pairs | 0.0 | 0.1 | 0.1 | 0.2 |
| bonded | 0.0 | 0.0 | 0.1 | 0.1 |
| other | 28.4 | 2.3 | 5.6 | 36.3 |

second backward, top autograd nodes (kernel-busy ms, inclusive): SelectBackward0 306.9, MulBackward0 40.6, BmmBackward0 35.4, IndexBackward0 17.3, NegBackward0 6.5, StackBackward0 6.0, AddmmBackward0 4.6, torch::autograd::CopySlices 2.5, IndexPutImplBackward0 2.2, MmBackward0 2.2, ViewBackward0 1.7, AsStridedBackward0 1.7

wall - kernel busy = launch/dispatch/sync idle in that phase. Region attribution follows autograd sequence numbers (see attribute_train_step).
