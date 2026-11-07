# SUNDIALS working-tree evidence

The pinned SUNDIALS commit remains checked out. Its working tree is not clean because long generated ARKODE answer-file paths could not be materialized in this Windows checkout.

At the SUNDIALS root, the tracked file `examples/arkode/C_serial/ark_kepler_--stepper_SPRK_--step-mode_fixed_--method_ARKODE_SPRK_PSEUDO_LEAPFROG_2_2_--tf_50_--check-order_--nout_1.out` is absent. Its committed Git blob is `ad149a869c66561625bf998b4234f3a3693890c1`; no working-tree SHA256 exists because the file is missing. Git reports `Filename too long` when attempting to read it.

The nested `test/answers` repository remains at commit `779e974c8095d8b3c2837edf56632a990714d669`, but Git reports many missing generated `.out` answer files with similarly long names. This is a checkout filesystem limitation, not a scientific source edit. No file was reset, deleted, or modified by this port after the issue was detected. Future generated output is kept under `outputs/cantera-python`, never under `cantera-original`.
