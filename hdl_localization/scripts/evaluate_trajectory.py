#!/usr/bin/env python3
"""
Trajectory Evaluation Script
Compares estimated trajectory with ground truth using Umeyama alignment
and generates visualization plots.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import sys
import os

def load_tum_trajectory(filepath):
    """Load trajectory in TUM format: timestamp tx ty tz qx qy qz qw"""
    data = np.loadtxt(filepath)
    timestamps = data[:, 0]
    positions = data[:, 1:4]
    quaternions = data[:, 4:8]  # qx qy qz qw
    return timestamps, positions, quaternions

def umeyama_alignment(source, target):
    """
    Computes the SE(3) transformation that aligns source to target
    using the Umeyama method.
    Returns: scale, rotation (3x3), translation (3,)
    """
    assert source.shape == target.shape
    n = source.shape[0]

    # Centroids
    mu_source = source.mean(axis=0)
    mu_target = target.mean(axis=0)

    # Centered points
    source_centered = source - mu_source
    target_centered = target - mu_target

    # Covariance matrix
    H = source_centered.T @ target_centered / n

    # SVD
    U, S, Vt = np.linalg.svd(H)

    # Rotation
    R_mat = Vt.T @ U.T

    # Handle reflection case
    if np.linalg.det(R_mat) < 0:
        Vt[-1, :] *= -1
        R_mat = Vt.T @ U.T

    # Scale
    var_source = np.sum(source_centered ** 2) / n
    scale = np.sum(S) / var_source

    # Translation
    t = mu_target - scale * R_mat @ mu_source

    return scale, R_mat, t

def align_trajectories(est_pos, gt_pos):
    """Align estimated trajectory to GT using Umeyama"""
    scale, R_mat, t = umeyama_alignment(est_pos, gt_pos)
    aligned = scale * (est_pos @ R_mat.T) + t
    return aligned, scale, R_mat, t

def compute_ate(aligned_pos, gt_pos):
    """Compute Absolute Trajectory Error"""
    errors = np.linalg.norm(aligned_pos - gt_pos, axis=1)
    rmse = np.sqrt(np.mean(errors ** 2))
    mean = np.mean(errors)
    median = np.median(errors)
    std = np.std(errors)
    max_err = np.max(errors)
    return {
        'rmse': rmse,
        'mean': mean,
        'median': median,
        'std': std,
        'max': max_err,
        'errors': errors
    }

def compute_rotation_error(est_quats, gt_quats):
    """Compute rotation errors in degrees"""
    errors = []
    for eq, gq in zip(est_quats, gt_quats):
        # Convert to rotation objects (qx, qy, qz, qw format)
        r_est = R.from_quat(eq)
        r_gt = R.from_quat(gq)
        # Relative rotation
        r_diff = r_gt.inv() * r_est
        # Angle in degrees
        angle = np.abs(r_diff.magnitude()) * 180 / np.pi
        errors.append(angle)
    errors = np.array(errors)
    return {
        'rmse': np.sqrt(np.mean(errors ** 2)),
        'mean': np.mean(errors),
        'median': np.median(errors),
        'std': np.std(errors),
        'max': np.max(errors),
        'errors': errors
    }

def associate_trajectories(est_ts, est_pos, est_quat, gt_ts, gt_pos, gt_quat, max_diff=0.05):
    """Associate trajectories by timestamp"""
    matches_est = []
    matches_gt = []

    gt_idx = 0
    for i, t in enumerate(est_ts):
        # Find closest GT timestamp
        while gt_idx < len(gt_ts) - 1 and gt_ts[gt_idx] < t - max_diff:
            gt_idx += 1

        if gt_idx < len(gt_ts):
            diff = abs(gt_ts[gt_idx] - t)
            if diff < max_diff:
                matches_est.append(i)
                matches_gt.append(gt_idx)

    return (est_pos[matches_est], est_quat[matches_est], est_ts[matches_est],
            gt_pos[matches_gt], gt_quat[matches_gt], gt_ts[matches_gt])

def main():
    # File paths
    est_file = "/home/q/colcon_ws/src/hdl-localization-ROS2/hdl_localization/Log/traj_lidar.txt"
    gt_file = "/home/q/colcon_ws/src/glim_ros2/src/glim_ros/Log/map_20251217_093040/traj_lidar.txt"
    output_dir = "/home/q/colcon_ws/src/hdl-localization-ROS2/hdl_localization/Log"

    print("=" * 60)
    print("Trajectory Evaluation")
    print("=" * 60)

    # Load trajectories
    print(f"\nLoading estimated trajectory: {est_file}")
    est_ts, est_pos, est_quat = load_tum_trajectory(est_file)
    print(f"  Loaded {len(est_ts)} poses")
    print(f"  Time range: {est_ts[0]:.3f} - {est_ts[-1]:.3f} ({est_ts[-1] - est_ts[0]:.1f}s)")

    print(f"\nLoading GT trajectory: {gt_file}")
    gt_ts, gt_pos, gt_quat = load_tum_trajectory(gt_file)
    print(f"  Loaded {len(gt_ts)} poses")
    print(f"  Time range: {gt_ts[0]:.3f} - {gt_ts[-1]:.3f} ({gt_ts[-1] - gt_ts[0]:.1f}s)")

    # Associate by timestamp
    print("\nAssociating trajectories by timestamp...")
    (est_pos_matched, est_quat_matched, est_ts_matched,
     gt_pos_matched, gt_quat_matched, gt_ts_matched) = associate_trajectories(
        est_ts, est_pos, est_quat, gt_ts, gt_pos, gt_quat)
    print(f"  Matched {len(est_pos_matched)} pose pairs")

    if len(est_pos_matched) < 10:
        print("ERROR: Not enough matched poses!")
        return

    # Align using Umeyama
    print("\nPerforming Umeyama SE(3) alignment...")
    aligned_pos, scale, R_mat, t = align_trajectories(est_pos_matched, gt_pos_matched)
    print(f"  Scale: {scale:.6f}")
    print(f"  Translation: [{t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f}]")

    # Compute ATE
    print("\n" + "=" * 60)
    print("Position Error (ATE)")
    print("=" * 60)
    ate = compute_ate(aligned_pos, gt_pos_matched)
    print(f"  RMSE:   {ate['rmse']*100:.2f} cm")
    print(f"  Mean:   {ate['mean']*100:.2f} cm")
    print(f"  Median: {ate['median']*100:.2f} cm")
    print(f"  Std:    {ate['std']*100:.2f} cm")
    print(f"  Max:    {ate['max']*100:.2f} cm")

    # Compute rotation error
    print("\n" + "=" * 60)
    print("Rotation Error")
    print("=" * 60)
    rot_err = compute_rotation_error(est_quat_matched, gt_quat_matched)
    print(f"  RMSE:   {rot_err['rmse']:.3f} deg")
    print(f"  Mean:   {rot_err['mean']:.3f} deg")
    print(f"  Median: {rot_err['median']:.3f} deg")
    print(f"  Std:    {rot_err['std']:.3f} deg")
    print(f"  Max:    {rot_err['max']:.3f} deg")

    # Evaluation summary
    print("\n" + "=" * 60)
    print("Evaluation Summary")
    print("=" * 60)
    pos_pass = ate['rmse'] < 0.10  # 10cm target
    rot_pass = rot_err['rmse'] < 1.0  # 1 deg target
    print(f"  Position RMSE: {ate['rmse']*100:.2f} cm {'PASS' if pos_pass else 'FAIL'} (target: <10cm)")
    print(f"  Rotation RMSE: {rot_err['rmse']:.3f} deg {'PASS' if rot_pass else 'FAIL'} (target: <1deg)")

    # Percentile analysis
    print("\n" + "=" * 60)
    print("Position Error Percentiles")
    print("=" * 60)
    percentiles = [50, 75, 90, 95, 99]
    for p in percentiles:
        val = np.percentile(ate['errors'], p) * 100
        print(f"  P{p}: {val:.2f} cm")

    # Error distribution by time
    time_offset = est_ts_matched - est_ts_matched[0]

    # Create visualization
    print("\nGenerating visualization plots...")

    fig = plt.figure(figsize=(16, 12))

    # 1. 2D Trajectory comparison (top-down view)
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(gt_pos_matched[:, 0], gt_pos_matched[:, 1], 'b-', label='GT', linewidth=1.5, alpha=0.7)
    ax1.plot(aligned_pos[:, 0], aligned_pos[:, 1], 'r-', label='Estimated (aligned)', linewidth=1.5, alpha=0.7)
    ax1.scatter(gt_pos_matched[0, 0], gt_pos_matched[0, 1], c='green', s=100, marker='o', label='Start', zorder=5)
    ax1.scatter(gt_pos_matched[-1, 0], gt_pos_matched[-1, 1], c='purple', s=100, marker='s', label='End', zorder=5)
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_title('2D Trajectory Comparison (Top-Down View)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')

    # 2. Position error over time
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(time_offset, ate['errors'] * 100, 'b-', linewidth=0.8, alpha=0.7)
    ax2.axhline(y=ate['rmse']*100, color='r', linestyle='--', label=f'RMSE: {ate["rmse"]*100:.2f} cm')
    ax2.axhline(y=10, color='g', linestyle=':', label='Target: 10 cm')
    ax2.fill_between(time_offset, 0, ate['errors'] * 100, alpha=0.3)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Position Error (cm)')
    ax2.set_title('Position Error Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, min(ate['max']*100*1.1, 100))

    # 3. Rotation error over time
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(time_offset, rot_err['errors'], 'b-', linewidth=0.8, alpha=0.7)
    ax3.axhline(y=rot_err['rmse'], color='r', linestyle='--', label=f'RMSE: {rot_err["rmse"]:.3f} deg')
    ax3.axhline(y=1.0, color='g', linestyle=':', label='Target: 1 deg')
    ax3.fill_between(time_offset, 0, rot_err['errors'], alpha=0.3)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Rotation Error (deg)')
    ax3.set_title('Rotation Error Over Time')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, min(rot_err['max']*1.1, 10))

    # 4. Error histogram
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.hist(ate['errors'] * 100, bins=50, color='blue', alpha=0.7, edgecolor='black')
    ax4.axvline(x=ate['rmse']*100, color='r', linestyle='--', linewidth=2, label=f'RMSE: {ate["rmse"]*100:.2f} cm')
    ax4.axvline(x=ate['median']*100, color='orange', linestyle='--', linewidth=2, label=f'Median: {ate["median"]*100:.2f} cm')
    ax4.axvline(x=10, color='g', linestyle=':', linewidth=2, label='Target: 10 cm')
    ax4.set_xlabel('Position Error (cm)')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Position Error Distribution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    output_file = os.path.join(output_dir, "trajectory_evaluation.png")
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_file}")

    # Also create 3D plot
    fig2 = plt.figure(figsize=(10, 8))
    ax = fig2.add_subplot(111, projection='3d')
    ax.plot(gt_pos_matched[:, 0], gt_pos_matched[:, 1], gt_pos_matched[:, 2],
            'b-', label='GT', linewidth=1.5, alpha=0.7)
    ax.plot(aligned_pos[:, 0], aligned_pos[:, 1], aligned_pos[:, 2],
            'r-', label='Estimated (aligned)', linewidth=1.5, alpha=0.7)
    ax.scatter(gt_pos_matched[0, 0], gt_pos_matched[0, 1], gt_pos_matched[0, 2],
               c='green', s=100, marker='o', label='Start')
    ax.scatter(gt_pos_matched[-1, 0], gt_pos_matched[-1, 1], gt_pos_matched[-1, 2],
               c='purple', s=100, marker='s', label='End')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('3D Trajectory Comparison')
    ax.legend()

    output_file_3d = os.path.join(output_dir, "trajectory_3d.png")
    plt.savefig(output_file_3d, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_file_3d}")

    plt.show()

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)

if __name__ == "__main__":
    main()
