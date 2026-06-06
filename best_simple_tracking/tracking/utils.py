import torch

from torch import tensor


def SE2_kinematics(x: tensor, action: tensor, tau: float) -> tensor:
    wt_2 = action[1] * tau / 2
    t_v_sinc_term = tau * action[0] * torch.sinc(wt_2 / torch.pi)
    ret_x = torch.empty(3)
    ret_x[0] = x[0] + t_v_sinc_term * torch.cos(x[2] + wt_2)
    ret_x[1] = x[1] + t_v_sinc_term * torch.sin(x[2] + wt_2)
    ret_x[2] = x[2] + 2 * wt_2
    return ret_x


def landmark_motion(mu: tensor, v: tensor, A: tensor, B: tensor) -> tensor:
    return mu @ A.T + v @ B.T

def landmark_motion_real(mu: tensor, v: tensor, A: tensor, B: tensor, W: tensor) -> tensor:
    # print(mu, A.T, torch.normal(mean=torch.zeros(mu.size()), std=torch.sqrt(W)), "\n\n\n")
    return mu @ A.T + v @ B.T + torch.normal(mean=torch.zeros(mu.size()), std=torch.sqrt(W))

def triangle_SDF(q: tensor, psi: float, r: float) -> tensor:
    original_shape = q.shape[:-1]
    q_flat = q.reshape(-1, 2)
    x, y = q_flat[:, 0], q_flat[:, 1]

    psi_val = torch.as_tensor(psi, device=q_flat.device, dtype=q_flat.dtype).reshape(())
    r_val = torch.as_tensor(r, device=q_flat.device, dtype=q_flat.dtype).reshape(())
    tan_psi = torch.tan(psi_val)
    p_x = r_val / (1 + torch.sin(psi_val))

    a_1 = torch.stack((-torch.ones_like(tan_psi), 1 / tan_psi))
    a_2 = torch.stack((-torch.ones_like(tan_psi), -1 / tan_psi))
    a_3 = q_flat.new_tensor([1.0, 0.0])
    b_1, b_2, b_3 = 0.0, 0.0, -r_val

    q_1 = torch.stack((r_val, r_val * tan_psi))
    q_2 = torch.stack((r_val, -r_val * tan_psi))
    q_3 = q_flat.new_tensor([0.0, 0.0])

    l_1_low, l_1_up, l_2_low, l_2_up = l_function(x, psi_val, r_val, p_x)

    sdf = torch.empty_like(x)

    cond_1 = y >= l_1_up
    sdf[cond_1] = torch.linalg.norm(q_flat[cond_1] - q_1, dim=1)

    cond_2 = (~cond_1) & (y >= l_1_low) & (y < l_1_up)
    sdf[cond_2] = (q_flat[cond_2] @ a_1 + b_1) / torch.linalg.norm(a_1)

    cond_3 = (~cond_1) & (~cond_2) & (x < 0) & (y >= l_2_up) & (y < l_1_low)
    sdf[cond_3] = torch.linalg.norm(q_flat[cond_3] - q_3, dim=1)

    cond_4 = (~cond_1) & (~cond_2) & (~cond_3) & (x > p_x) & (y >= l_2_up) & (y < l_1_low)
    sdf[cond_4] = (q_flat[cond_4] @ a_3 + b_3) / torch.linalg.norm(a_3)

    cond_5 = (~cond_1) & (~cond_2) & (~cond_3) & (~cond_4) & (y > l_2_low) & (y < l_2_up)
    sdf[cond_5] = (q_flat[cond_5] @ a_2 + b_2) / torch.linalg.norm(a_2)

    cond_6 = ~(cond_1 | cond_2 | cond_3 | cond_4 | cond_5)
    sdf[cond_6] = torch.linalg.norm(q_flat[cond_6] - q_2, dim=1)

    return sdf.reshape(original_shape)


def l_function(x, psi, r, p_x):
    ones = torch.ones_like(x)
    l_1_low, l_2_up = r * torch.tan(psi) * ones, -r * torch.tan(psi) * ones

    inds_1 = torch.nonzero(x < 0)
    l_1_low[inds_1], l_2_up[inds_1] = - x[inds_1] / torch.tan(psi), x[inds_1] / torch.tan(psi)

    inds_2 = torch.nonzero(torch.logical_and(0 <= x, x < p_x))
    l_1_low[inds_2], l_2_up[inds_2] = 0, 0

    inds_3 = torch.nonzero(torch.logical_and(p_x <= x, x < r))
    l_1_low[inds_3] = torch.tan(torch.pi / 4 + psi / 2) * x[inds_3] - r / torch.cos(psi)
    l_2_up[inds_3] = - torch.tan(torch.pi / 4 + psi / 2) * x[inds_3] + r / torch.cos(psi)

    l_1_up, l_2_low = r * torch.tan(psi) * ones, -r * torch.tan(psi) * ones

    inds_4 = torch.nonzero(x < r)
    l_1_up[inds_4] = - (x[inds_4] - r) / torch.tan(psi) + r * torch.tan(psi)
    l_2_low[inds_4] = (x[inds_4] - r) / torch.tan(psi) - r * torch.tan(psi)

    return l_1_low, l_1_up, l_2_low, l_2_up


def get_transformation(x: tensor) -> tensor:
    cos_term = torch.cos(x[2])
    sin_term = torch.sin(x[2])
    # transformation = torch.zeros((3, 3), requires_grad=False)
    # transformation[0, :] = tensor([cos_term, - sin_term, x[0]])
    # transformation[1, :] = tensor([sin_term, cos_term, x[1]])
    # transformation[2, 2] = 1
    return tensor([[cos_term, - sin_term, x[0]],
                   [sin_term, cos_term, x[1]],
                   [0, 0, 1]], requires_grad=True)


def phi(SDF: tensor, kappa: float) -> tensor:
    return 0.5 * (1 + torch.erf(SDF / (2**0.5 * kappa) - 2))
