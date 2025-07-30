
## Physics Alignment

### Robot joint control data generation

You can try this script to record real arm motion data under some different commands.

```bash
trajectory_type='sin'  # one of step, sin, valid, combined, policy
output="csv/test_joint"
for joint_idx in {1..6}
do
    python sysid_traj_gen.py \
        --ctrl_freq 10 \
        --trajectory_type "${trajectory_type}" \
        --joint_idx "${joint_idx}" \
        --output "${output}"\
done
```