import torch

from torch import tensor
from torch.optim import Adam

from best_simple_tracking.tracking.policy_net import PolicyNet
from best_simple_tracking.tracking.policy_net_att import PolicyNetAtt
from best_simple_tracking.tracking.utils import landmark_motion, triangle_SDF, phi


class ModelBasedAgent:
    """Model-based agent with linear Kalman filter in information form.

    Predict step (1:1 port of simple_env / best_simple_reward):
        mu_predict = clip(A·mu + B·v, -env_half, +env_half)
        info_new   = (info^-1 + W)^-1
    Update step: standard information-form Kalman update with FoV-gated
    measurements.
    """

    def __init__(self, num_landmarks, init_info, A, B, W, radius, psi, kappa, V, lr,
                 num_robots=2, reward_weights=None, reward_clip=1.0,
                 uncertainty_threshold=10.0, tau: float = 1.0):
        self._init_info = init_info
        self._info = None

        self._num_robots = num_robots
        self._num_landmarks = num_landmarks
        self._A = A
        self._B = B
        self._W = W
        self._psi = psi
        self._radius = radius
        self._kappa = kappa
        self._V = V
        self._inv_V = V ** (-1)
        self._tau = float(tau)

        # Reward weights — coverage + persistence tuning.
        weights = {
            'info_gain': 0.1,
            'persistence': 4.0,
            'loss': 8.0,
            'overlap': 2.5,
            'coverage': 4.0,
            'tracking_continuity': 3.0,
        }
        if reward_weights is not None:
            weights.update(reward_weights)
        self._info_gain_weight = float(weights['info_gain'])
        self._persistence_weight = float(weights['persistence'])
        self._loss_weight = float(weights['loss'])
        self._overlap_weight = float(weights['overlap'])
        self._coverage_weight = float(weights['coverage'])
        self._tracking_continuity_weight = float(weights['tracking_continuity'])

        self._reward_clip = float(reward_clip)
        self._log_epsilon = 1e-6

        # Retained for API compatibility; coverage is computed from FoV visibility only.
        self._uncertainty_threshold = uncertainty_threshold

        # Tracking state
        self._prev_visible = None
        self._prev_owner = None
        self._episode_reward = None
        self._reward_steps = 0
        self._accumulated_info_gain = None
        self._tracking_history = None
        self._consecutive_tracking = None

        input_dim = num_landmarks * 4 + 3 * self._num_robots
        self._policy = PolicyNet(input_dim=input_dim, policy_dim=2 * self._num_robots,
                                 num_robots=self._num_robots)
        self._policy_optimizer = Adam(self._policy.parameters(), lr=lr)

    def reset_agent_info(self):
        self._info = self._init_info * torch.ones((self._num_landmarks, 2))
        self._reset_reward_tracking(self._info.device)

    def reset_estimate_mu(self, mu_real):
        self._mu_update = mu_real + torch.normal(
            mean=torch.zeros(self._num_landmarks, 2),
            std=torch.sqrt(self._V),
        )

    def _reset_reward_tracking(self, device):
        self._prev_visible = None
        self._prev_owner = None
        self._episode_reward = torch.tensor(0.0, device=device)
        self._reward_steps = 0
        self._accumulated_info_gain = torch.tensor(0.0, device=device)
        self._tracking_history = torch.zeros(self._num_landmarks, device=device)
        self._consecutive_tracking = torch.zeros(self._num_landmarks, device=device)

    def _assign_target_owners(self, mu_real, x, visible, prev_owner):
        num_robots = x.size(0)
        num_landmarks = mu_real.size(0)
        device = mu_real.device
        owner = torch.full((num_landmarks,), -1, dtype=torch.long, device=device)

        if prev_owner is not None:
            prev_owner = prev_owner.to(device)
            valid_prev = prev_owner >= 0
            if valid_prev.any():
                target_idx = torch.arange(num_landmarks, device=device)[valid_prev]
                prev_r = prev_owner[valid_prev]
                still_visible = visible[prev_r, target_idx]
                if still_visible.any():
                    owner[target_idx[still_visible]] = prev_r[still_visible]

        unassigned = owner < 0
        if unassigned.any() and num_robots > 0:
            target_idx = torch.arange(num_landmarks, device=device)[unassigned]
            dx = mu_real[target_idx, 0].unsqueeze(0) - x[:, 0].unsqueeze(1)
            dy = mu_real[target_idx, 1].unsqueeze(0) - x[:, 1].unsqueeze(1)
            dist2 = dx * dx + dy * dy
            vis_sub = visible[:, target_idx]
            dist2 = dist2.masked_fill(~vis_sub, float('inf'))
            min_dist, min_idx = dist2.min(dim=0)
            visible_any = torch.isfinite(min_dist)
            if visible_any.any():
                owner[target_idx[visible_any]] = min_idx[visible_any]

        return owner

    def _update_reward_tracking(self, mu_real, x, visible):
        visible_d = visible.detach()
        mu_real_d = mu_real.detach()
        x_d = x.detach()
        num_robots = x_d.size(0)
        num_landmarks = mu_real_d.size(0)
        device = mu_real_d.device

        if self._prev_visible is None:
            self._prev_visible = visible_d
            self._prev_owner = self._assign_target_owners(mu_real_d, x_d, visible_d, None).detach()
            currently_visible = visible_d.any(dim=0)
            self._tracking_history[currently_visible] += 1
            self._consecutive_tracking[currently_visible] += 1
            return

        curr_owner = self._assign_target_owners(mu_real_d, x_d, visible_d, self._prev_owner).detach()
        curr_any = visible_d.any(dim=0)
        prev_any = self._prev_visible.any(dim=0)

        lost = prev_any & (~curr_any)
        persistence = (curr_owner >= 0) & (self._prev_owner == curr_owner)
        overlap_excess = (visible_d.sum(dim=0) - 1).clamp(min=0)

        self._tracking_history[curr_any] += 1
        self._tracking_history[~curr_any] = 0

        self._consecutive_tracking[persistence] += 1
        self._consecutive_tracking[~persistence] = 0
        newly_visible = curr_any & (~prev_any)
        self._consecutive_tracking[newly_visible] = 1

        num_visible = curr_any.to(mu_real_d.dtype).sum()
        coverage_ratio = num_visible / max(num_landmarks, 1)

        continuity_bonus = torch.sum(1.0 - torch.exp(-self._consecutive_tracking / 5.0))
        continuity_bonus = continuity_bonus / max(num_landmarks, 1)

        persist_frac = persistence.to(mu_real_d.dtype).sum() / max(num_landmarks, 1)
        loss_frac = lost.to(mu_real_d.dtype).sum() / max(num_landmarks, 1)
        overlap_den = max(num_landmarks * max(num_robots - 1, 1), 1)
        overlap_frac = overlap_excess.to(mu_real_d.dtype).sum() / overlap_den

        step_reward = (
            self._persistence_weight * persist_frac +
            self._coverage_weight * coverage_ratio +
            self._tracking_continuity_weight * continuity_bonus -
            self._loss_weight * loss_frac -
            self._overlap_weight * overlap_frac
        )
        step_reward = step_reward.clamp(min=-self._reward_clip, max=self._reward_clip)

        self._episode_reward = self._episode_reward + step_reward
        self._reward_steps += 1

        self._prev_visible = visible_d
        self._prev_owner = curr_owner

    def eval_policy(self):
        self._policy.eval()

    def train_policy(self):
        self._policy.train()

    def plan(self, v, x):
        # Linear KF predict (information form):
        #   mu_predict = clip(A·mu + B·v, -env_half, +env_half)
        #   info_new   = (info^-1 + W)^-1
        env_half = tensor([self._num_landmarks, self._num_landmarks])
        self._mu_predict = torch.clip(
            landmark_motion(self._mu_update, v, self._A, self._B),
            min=-env_half, max=env_half,
        )
        self._info = (self._info ** (-1) + self._W) ** (-1)

        if len(x.size()) == 1:
            x = x[None, :]

        x_ref = x[0]
        q_predict = torch.vstack((
            (self._mu_predict[:, 0] - x_ref[0]) * torch.cos(x_ref[2]) + (self._mu_predict[:, 1] - x_ref[1]) * torch.sin(x_ref[2]),
            (x_ref[0] - self._mu_predict[:, 0]) * torch.sin(x_ref[2]) + (self._mu_predict[:, 1] - x_ref[1]) * torch.cos(x_ref[2]),
        )).T

        agent_pos_local = torch.zeros(3, device=x.device, dtype=x.dtype)
        other_robots = []
        max_other = min(self._num_robots - 1, x.size(0) - 1)
        for i in range(max_other):
            other_pose = x[i + 1]
            dx = other_pose[0] - x_ref[0]
            dy = other_pose[1] - x_ref[1]
            theta = x_ref[2]
            rel_x = dx * torch.cos(theta) + dy * torch.sin(theta)
            rel_y = -dx * torch.sin(theta) + dy * torch.cos(theta)
            rel_theta = other_pose[2] - theta
            other_robots.append(torch.stack((rel_x, rel_y, rel_theta)))
        if len(other_robots) > 0:
            other_robots_input = torch.cat(other_robots)
        else:
            other_robots_input = torch.zeros(0, device=x.device, dtype=x.dtype)
        missing = (self._num_robots - 1) - max_other
        if missing > 0:
            other_robots_input = torch.cat((other_robots_input,
                                            torch.zeros(3 * missing, device=x.device, dtype=x.dtype)))

        net_input = torch.hstack((agent_pos_local, other_robots_input,
                                  self._info.flatten(), q_predict.flatten()))
        action = self._policy.forward(net_input)
        return action

    def update_info_mu(self, mu_real, x):
        if len(x.size()) == 1:
            x = x[None, :]

        num_robots = x.size(0)
        if num_robots == 0:
            return

        dx = mu_real[:, 0].unsqueeze(0) - x[:, 0].unsqueeze(1)
        dy = mu_real[:, 1].unsqueeze(0) - x[:, 1].unsqueeze(1)
        c = torch.cos(x[:, 2]).unsqueeze(1)
        s = torch.sin(x[:, 2]).unsqueeze(1)
        q_real = torch.stack((dx * c + dy * s,
                              -dx * s + dy * c), dim=2)
        sdf_real = triangle_SDF(q_real, self._psi, self._radius).reshape(num_robots, self._num_landmarks)
        visible = (sdf_real <= 0)

        # IMPORTANT: Update reward tracking BEFORE info update so we can track uncertainty changes.
        self._update_reward_tracking(mu_real, x, visible)

        noise = torch.normal(
            mean=torch.zeros((num_robots, self._num_landmarks, 2),
                             device=mu_real.device, dtype=mu_real.dtype),
            std=torch.sqrt(self._V),
        )
        z = mu_real.unsqueeze(0) + noise

        weights = visible.unsqueeze(-1).to(mu_real.dtype)
        sum_weights = weights.sum(dim=0)

        info_prior = self._info
        y_prior = info_prior * self._mu_predict
        meas_sum = (weights * z).sum(dim=0)

        info_post = info_prior + sum_weights * self._inv_V
        info_post_safe = info_post.clamp_min(1e-8)
        y_post = y_prior + meas_sum * self._inv_V
        self._mu_update = y_post / info_post_safe

        dx_u = self._mu_update[:, 0].unsqueeze(0) - x[:, 0].unsqueeze(1)
        dy_u = self._mu_update[:, 1].unsqueeze(0) - x[:, 1].unsqueeze(1)
        q_update = torch.stack((dx_u * c + dy_u * s,
                                -dx_u * s + dy_u * c), dim=2)
        sdf_update = triangle_SDF(q_update, self._psi, self._radius).reshape(num_robots, self._num_landmarks)
        weights_info = (1 - phi(sdf_update, self._kappa))
        M_total = weights_info.sum(dim=0).unsqueeze(1) * self._inv_V
        self._info = self._info + M_total

        info_gain = torch.sum(torch.log(self._info.clamp_min(self._log_epsilon))) - torch.sum(
            torch.log(info_prior.clamp_min(self._log_epsilon)))
        self._accumulated_info_gain = self._accumulated_info_gain + info_gain
        self._mu_predict = self._mu_update

    def set_policy_grad_to_zero(self):
        self._policy_optimizer.zero_grad()

    def update_policy_grad(self, train=True):
        avg_reward = self._episode_reward
        if self._reward_steps > 0:
            avg_reward = self._episode_reward / self._reward_steps
        total_objective = (self._info_gain_weight * self._accumulated_info_gain) + avg_reward
        loss = -total_objective
        if train and total_objective.requires_grad:
            loss.backward()
        return total_objective.item()

    def policy_step(self, debug=False, clip_grad_norm: float | None = None):
        if debug:
            param_list = []
            for i, p in enumerate(self._policy.parameters()):
                param_list.append(p.data.detach().clone())

        # Skip if any gradient is non-finite (NaN/Inf). One NaN in the policy
        # weights would poison all future steps.
        for p in self._policy.parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all():
                self._policy_optimizer.zero_grad()
                return

        if clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self._policy.parameters(), max_norm=clip_grad_norm)

        self._policy_optimizer.step()

        if debug:
            total_param_rssd = 0
            grad_power = 0
            for i, p in enumerate(self._policy.parameters()):
                if p.grad is not None:
                    grad_power += (p.grad ** 2).sum()
                else:
                    grad_power += 0
                total_param_rssd += ((param_list[i] - p.data) ** 2).sum().sqrt()

            print("Gradient power after backward: {}".format(grad_power))
            print("RSSD of weights after applying the gradient: {}".format(total_param_rssd))

    def get_policy_state_dict(self):
        return self._policy.state_dict()

    def load_policy_state_dict(self, load_model):
        self._policy.load_state_dict(torch.load(load_model))


class ModelBasedAgentAtt:
    """Attention-based variant of ModelBasedAgent (uses PolicyNetAtt).
    Supports a max_num_landmarks budget so the policy network can be shared
    across episodes with varying real landmark count (padding + mask).
    """

    def __init__(self, max_num_landmarks, init_info, A, B, W, radius, psi, kappa, V, lr,
                 num_robots=2, reward_weights=None, reward_clip=1.0,
                 uncertainty_threshold=10.0, tau: float = 1.0):
        self._init_info = init_info
        self._info = None

        self._num_robots = num_robots
        self._max_num_landmarks = max_num_landmarks
        self._A = A
        self._B = B
        self._W = W
        self._psi = psi
        self._radius = radius
        self._kappa = kappa
        self._V = V
        self._inv_V = V ** (-1)
        self._tau = float(tau)

        weights = {
            'info_gain': 0.3,
            'persistence': 2.0,
            'loss': 2.0,
            'overlap': 0.5,
            'coverage': 3.0,
            'tracking_continuity': 1.0,
        }
        if reward_weights is not None:
            weights.update(reward_weights)
        self._info_gain_weight = float(weights['info_gain'])
        self._persistence_weight = float(weights['persistence'])
        self._loss_weight = float(weights['loss'])
        self._overlap_weight = float(weights['overlap'])
        self._coverage_weight = float(weights['coverage'])
        self._tracking_continuity_weight = float(weights['tracking_continuity'])

        self._reward_clip = float(reward_clip)
        self._log_epsilon = 1e-6
        # Retained for API compatibility; coverage is computed from FoV visibility only.
        self._uncertainty_threshold = uncertainty_threshold

        self._prev_visible = None
        self._prev_owner = None
        self._episode_reward = None
        self._reward_steps = 0
        self._accumulated_info_gain = None
        self._tracking_history = None
        self._consecutive_tracking = None

        input_dim = max_num_landmarks * 5 + 3 * self._num_robots
        self._policy = PolicyNetAtt(input_dim=input_dim, policy_dim=2,
                                    num_other_robots=self._num_robots - 1,
                                    num_robots=self._num_robots)
        self._policy_optimizer = Adam(self._policy.parameters(), lr=lr)

    def reset_agent_info(self):
        self._info = self._init_info * torch.ones((self._num_landmarks, 2))
        self._reset_reward_tracking(self._info.device)

    def reset_estimate_mu(self, mu_real):
        self._num_landmarks = mu_real.size()[0]
        self._mu_update = mu_real + torch.normal(
            mean=torch.zeros(self._num_landmarks, 2),
            std=torch.sqrt(self._V),
        )
        self._padding = torch.zeros(2 * (self._max_num_landmarks - self._num_landmarks))
        self._mask = torch.tensor(
            [True] * self._num_landmarks +
            [False] * (self._max_num_landmarks - self._num_landmarks)
        )

    def _reset_reward_tracking(self, device):
        self._prev_visible = None
        self._prev_owner = None
        self._episode_reward = torch.tensor(0.0, device=device)
        self._reward_steps = 0
        self._accumulated_info_gain = torch.tensor(0.0, device=device)
        self._tracking_history = torch.zeros(self._num_landmarks, device=device)
        self._consecutive_tracking = torch.zeros(self._num_landmarks, device=device)

    _assign_target_owners = ModelBasedAgent._assign_target_owners
    _update_reward_tracking = ModelBasedAgent._update_reward_tracking
    eval_policy = ModelBasedAgent.eval_policy
    train_policy = ModelBasedAgent.train_policy
    set_policy_grad_to_zero = ModelBasedAgent.set_policy_grad_to_zero
    update_policy_grad = ModelBasedAgent.update_policy_grad
    policy_step = ModelBasedAgent.policy_step
    get_policy_state_dict = ModelBasedAgent.get_policy_state_dict
    load_policy_state_dict = ModelBasedAgent.load_policy_state_dict

    def plan(self, v, x):
        # Linear KF predict (Att variant — no clip, matches source).
        self._mu_predict = landmark_motion(self._mu_update, v, self._A, self._B)
        self._info = (self._info ** (-1) + self._W) ** (-1)

        if len(x.size()) == 1:
            x = x[None, :]

        num_robots = x.size(0)
        target_other_len = 3 * (self._num_robots - 1)
        observations = []

        for i in range(num_robots):
            other_robots_rel = []
            for j in range(num_robots):
                if i == j:
                    continue
                dx = x[j, 0] - x[i, 0]
                dy = x[j, 1] - x[i, 1]
                theta = x[i, 2]
                rel_x = dx * torch.cos(theta) + dy * torch.sin(theta)
                rel_y = -dx * torch.sin(theta) + dy * torch.cos(theta)
                rel_theta = x[j, 2] - theta
                other_robots_rel.append(torch.stack([rel_x, rel_y, rel_theta]))

            if len(other_robots_rel) > 0:
                other_robots_input = torch.cat(other_robots_rel)
            else:
                other_robots_input = torch.zeros(0, device=x.device, dtype=x.dtype)

            pad_len = target_other_len - other_robots_input.numel()
            if pad_len > 0:
                other_robots_input = torch.cat((other_robots_input,
                                                torch.zeros(pad_len, device=x.device, dtype=x.dtype)))

            q_predict = torch.vstack((
                (self._mu_predict[:, 0] - x[i, 0]) * torch.cos(x[i, 2]) + (self._mu_predict[:, 1] - x[i, 1]) * torch.sin(x[i, 2]),
                (x[i, 0] - self._mu_predict[:, 0]) * torch.sin(x[i, 2]) + (self._mu_predict[:, 1] - x[i, 1]) * torch.cos(x[i, 2]),
            )).T

            agent_pos_local = torch.zeros(3, device=x.device, dtype=x.dtype)
            net_input = torch.hstack((agent_pos_local, other_robots_input,
                                      self._info.flatten(), self._padding,
                                      q_predict.flatten(), self._padding,
                                      self._mask))
            observations.append(net_input)

        batch_input = torch.stack(observations)
        actions = self._policy.forward(batch_input)
        return actions

    def update_info_mu(self, mu_real, x):
        if len(x.size()) == 1:
            x = x[None, :]

        num_robots = x.size(0)
        if num_robots == 0:
            return

        dx = mu_real[:, 0].unsqueeze(0) - x[:, 0].unsqueeze(1)
        dy = mu_real[:, 1].unsqueeze(0) - x[:, 1].unsqueeze(1)
        c = torch.cos(x[:, 2]).unsqueeze(1)
        s = torch.sin(x[:, 2]).unsqueeze(1)
        q_real = torch.stack((dx * c + dy * s,
                              -dx * s + dy * c), dim=2)
        sdf_real = triangle_SDF(q_real, self._psi, self._radius).reshape(num_robots, self._num_landmarks)
        visible = (sdf_real <= 0)

        self._update_reward_tracking(mu_real, x, visible)

        noise = torch.normal(
            mean=torch.zeros((num_robots, self._num_landmarks, 2),
                             device=mu_real.device, dtype=mu_real.dtype),
            std=torch.sqrt(self._V),
        )
        z = mu_real.unsqueeze(0) + noise

        weights = visible.unsqueeze(-1).to(mu_real.dtype)
        sum_weights = weights.sum(dim=0)

        info_prior = self._info
        y_prior = info_prior * self._mu_predict
        meas_sum = (weights * z).sum(dim=0)

        info_post = info_prior + sum_weights * self._inv_V
        info_post_safe = info_post.clamp_min(1e-8)
        y_post = y_prior + meas_sum * self._inv_V
        self._mu_update = y_post / info_post_safe

        dx_u = self._mu_update[:, 0].unsqueeze(0) - x[:, 0].unsqueeze(1)
        dy_u = self._mu_update[:, 1].unsqueeze(0) - x[:, 1].unsqueeze(1)
        q_update = torch.stack((dx_u * c + dy_u * s,
                                -dx_u * s + dy_u * c), dim=2)
        sdf_update = triangle_SDF(q_update, self._psi, self._radius).reshape(num_robots, self._num_landmarks)
        weights_info = (1 - phi(sdf_update, self._kappa))
        M_total = weights_info.sum(dim=0).unsqueeze(1) * self._inv_V
        self._info = self._info + M_total
        info_gain = torch.sum(torch.log(self._info.clamp_min(self._log_epsilon))) - torch.sum(
            torch.log(info_prior.clamp_min(self._log_epsilon)))
        self._accumulated_info_gain = self._accumulated_info_gain + info_gain
        self._mu_predict = self._mu_update
