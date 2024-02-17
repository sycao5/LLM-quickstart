- The training t5_3b using 1 gpu (A100) on colab completed.
- The training t5_11b using 1 gpu (A100) reported error when the used RAM close to 83.5 GB. I think the insufficient GPU memory because AMP requires additional GPU memory to store the half-precision values. If the GPU does not have enough memory, the training script may fail.

![Snapshot for training t5_11b on 1 gpu A100](oom_error.png "Snapshot for training t5_11b on 1 gpu A100")