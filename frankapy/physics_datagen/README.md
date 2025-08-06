
## Physics Alignment

```bash
pip install open3d
```

### Robot joint control data generation

You can try this script to record real arm motion data under some different commands.

```bash
cd physics_datagen

# sine is the best for our panda robot
trajectory_type='sine'  # one of step, sine, triangle, combined
output="csv/test_joint"
for joint_idx in {0..6}
do
    python sysid_joint_traj_gen.py \
        --ctrl_freq 10 \
        --trajectory_type "${trajectory_type}" \
        --joint_idx "${joint_idx}" \
        --output "${output}"
done


# sine is the best for our panda robot
trajectory_type='sine'  # one of step, sine, triangle, combined
output="csv/test_joint"
python sysid_joint_traj_gen.py \
    --ctrl_freq 10 \
    --trajectory_type "${trajectory_type}" \
    --joint_idx 0 \
    --output "${output}"
```