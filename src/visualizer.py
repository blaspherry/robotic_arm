import numpy as np
import matplotlib.pyplot as plt


def plot_arm(joint_positions, ax=None, show=True, title="Robotic Arm"):
    """
    参数:
        joint_positions: list[np.ndarray]，每个元素是 shape(3,) 的 xyz
    """
    pts = np.asarray(joint_positions, dtype=float)

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

    ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], marker="o")
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # 自动等比例显示（避免机械臂看起来被压扁）
    _set_axes_equal(ax, pts)

    if show:
        plt.show()
    return ax


def _set_axes_equal(ax, pts):
    # Matplotlib 3D 等比例缩放小技巧
    x_limits = [np.min(pts[:, 0]), np.max(pts[:, 0])]
    y_limits = [np.min(pts[:, 1]), np.max(pts[:, 1])]
    z_limits = [np.min(pts[:, 2]), np.max(pts[:, 2])]

    x_range = x_limits[1] - x_limits[0]
    y_range = y_limits[1] - y_limits[0]
    z_range = z_limits[1] - z_limits[0]
    max_range = max(x_range, y_range, z_range)

    x_mid = np.mean(x_limits)
    y_mid = np.mean(y_limits)
    z_mid = np.mean(z_limits)

    half = max_range / 2 if max_range > 0 else 1.0
    ax.set_xlim(x_mid - half, x_mid + half)
    ax.set_ylim(y_mid - half, y_mid + half)
    ax.set_zlim(z_mid - half, z_mid + half)
