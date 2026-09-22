| structure | ff_backend | frames | n_atoms | n_pairs | forward_ms | forward_forces_ms | train_step_ms | train_peak_gb | neighbor_list | projector_features | parameter_network | gates | elst_pairs | pauli_pairs | disp_pairs | bonded | coupled_solve |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| w6_mp2_avtz | torchff | 128 | 18 | 19584 | 62.3 | 133.9 | 315.2 | 1.4 | 8.8 | 22.9 | 18.8 | 1.2 | 16.3 | 4.7 | 0.1 | 0.2 | nan |
| w21_mp2_avtz | torchff | 128 | 63 | 249984 | 146.5 | 220.8 | 420.2 | 18.3 | 23.4 | 54.1 | 18.6 | 0.5 | 22.8 | 6.9 | 0.0 | 0.0 | nan |

Forward split column unit: cuda_ms per call (forward only; backward is in the whole-call columns). train_step = E+F loss with create_graph=True + backward.

### w6_mp2_avtz training step (torchff, 128 frames): wall per phase, kernel-busy ms by forward region

| region | forward | first_backward | second_backward | total |
|---|---|---|---|---|
| **wall (synced)** | **68.0** | **74.5** | **172.9** | **315.3** |
| kernel busy | 27.3 | 19.3 | 57.2 | 103.8 |
| neighbor_list | 6.5 | 0.2 | 0.2 | 6.9 |
| projector_features | 9.9 | 4.5 | 10.7 | 25.2 |
| parameter_network | 3.5 | 2.4 | 13.4 | 19.3 |
| gates | 0.4 | 0.2 | 1.8 | 2.4 |
| elst_pairs | 3.7 | 8.4 | 22.4 | 34.5 |
| pauli_pairs | 0.9 | 2.1 | 5.7 | 8.8 |
| disp_pairs | 0.0 | 0.0 | 0.0 | 0.1 |
| bonded | 0.0 | 0.0 | 0.1 | 0.1 |
| other | 2.2 | 1.4 | 2.9 | 6.6 |

second backward, top autograd nodes (kernel-busy ms, inclusive): MulBackward0 24.3, BmmBackward0 7.6, IndexBackward0 7.6, AddmmBackward0 3.3, NegBackward0 2.3, EmbeddingBackward0 1.3, MmBackward0 1.3, SelectBackward0 1.1, torch::autograd::CopySlices 0.9, _StackBackward 0.8, DivBackward0 0.8, SliceBackward0 0.7

### w21_mp2_avtz training step (torchff, 128 frames): wall per phase, kernel-busy ms by forward region

| region | forward | first_backward | second_backward | total |
|---|---|---|---|---|
| **wall (synced)** | **148.9** | **76.2** | **180.9** | **405.9** |
| kernel busy | 129.3 | 50.0 | 146.9 | 326.2 |
| neighbor_list | 19.9 | 0.3 | 0.5 | 20.7 |
| projector_features | 50.0 | 11.8 | 28.7 | 90.5 |
| parameter_network | 6.0 | 4.9 | 24.0 | 34.9 |
| gates | 0.4 | 0.3 | 3.1 | 3.8 |
| elst_pairs | 19.8 | 24.2 | 68.0 | 112.0 |
| pauli_pairs | 4.7 | 6.2 | 17.2 | 28.0 |
| disp_pairs | 0.0 | 0.1 | 0.1 | 0.2 |
| bonded | 0.0 | 0.0 | 0.1 | 0.1 |
| other | 28.4 | 2.2 | 5.3 | 35.9 |

second backward, top autograd nodes (kernel-busy ms, inclusive): MulBackward0 40.5, BmmBackward0 35.4, IndexBackward0 17.2, UnbindBackward0 14.1, NegBackward0 6.5, _StackBackward 6.0, AddmmBackward0 4.6, SelectBackward0 2.8, torch::autograd::CopySlices 2.5, IndexPutImplBackward0 2.2, MmBackward0 2.1, ViewBackward0 1.7

wall - kernel busy = launch/dispatch/sync idle in that phase. Region attribution follows autograd sequence numbers (see attribute_train_step).
