
## Physics Alignment

```bash
pip install open3d
```

### Robot joint control data generation

You can try this script to record real arm motion data under some different commands.

```bash
cd physics_datagen
trajectory_type='sine'  # one of step, sine, valid, combined, policy
output="csv/test_joint"
for joint_idx in 6
do
    python sysid_joint_traj_gen.py \
        --ctrl_freq 10 \
        --trajectory_type "${trajectory_type}" \
        --joint_idx "${joint_idx}" \
        --output "${output}"
done



trajectory_type='sine'  # one of step, sine, valid, combined, policy
output="csv/test_joint"
python sysid_joint_traj_gen.py \
    --ctrl_freq 10 \
    --trajectory_type "${trajectory_type}" \
    --joint_idx 6 \
    --output "${output}"
```