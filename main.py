import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable


# --- Physical parameters ---

U0 = 1.2
n = 1
k = 2.0 * torch.pi * n
nu = 0.6 / (2 * k**2)


# --- Neural network model ---

class Sine(nn.Module):
    def forward(self, x):
        return torch.sin(x)


class MLP(nn.Module):
    def __init__(self, input_dim=3, width=64, depth=4):
        super().__init__()

        trunk = [nn.Linear(input_dim, width), Sine()]
        for _ in range(depth - 1):
            trunk += [nn.Linear(width, width), Sine()]
        self.trunk = nn.Sequential(*trunk)

        self.head_w = nn.Linear(width, 1)
        self.head_u = nn.Linear(width, 1, bias=False)
        self.head_v = nn.Linear(width, 1, bias=False)

        for head in (self.head_w, self.head_u, self.head_v):
            nn.init.uniform_(head.weight, -1e-3, 1e-3)
            if head.bias is not None:
                nn.init.zeros_(head.bias)

    def forward(self, x):
        features = self.trunk(x)
        w = self.head_w(features)
        u = self.head_u(features)
        v = self.head_v(features)
        return w, u, v


# --- Taylor-Green vortex initial condition ---

def taylor_green_ic(xyt0, U0=1.0, n=1):
    x = xyt0[:, 0:1]
    y = xyt0[:, 1:2]
    k = 2.0 * torch.pi * n

    w_0 = 2.0 * U0 * k * torch.sin(k * x) * torch.sin(k * y)
    u_0 = U0 * torch.sin(k * x) * torch.cos(k * y)
    v_0 = -U0 * torch.cos(k * x) * torch.sin(k * y)

    return w_0, u_0, v_0


# --- Autograd helpers for PDE residuals ---

def first_derivatives(output, inputs):
    grads = torch.autograd.grad(
        output,
        inputs,
        torch.ones_like(output),
        create_graph=True,
    )[0]
    return grads[:, 0:1], grads[:, 1:2], grads[:, 2:3]


def second_derivatives(first_grad, inputs, idx):
    return torch.autograd.grad(
        first_grad,
        inputs,
        torch.ones_like(first_grad),
        create_graph=True,
    )[0][:, idx:idx + 1]


# --- PINN training loop ---

model = MLP()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

for step in range(20001):
    batch_size = 256

    x = torch.rand(batch_size, 1)
    y = torch.rand(batch_size, 1)
    t = torch.rand(batch_size, 1)

    xyt = torch.cat([x, y, t], dim=1)
    xyt.requires_grad_(True)

    w, u, v = model(xyt)

    w_x, w_y, w_t = first_derivatives(w, xyt)
    w_xx = second_derivatives(w_x, xyt, 0)
    w_yy = second_derivatives(w_y, xyt, 1)

    u_x, u_y, _ = first_derivatives(u, xyt)
    v_x, v_y, _ = first_derivatives(v, xyt)

    R_w = ((w_t + (u * w_x + v * w_y) - nu * (w_xx + w_yy)) ** 2).mean()
    R_U = ((w - v_x + u_y) ** 2).mean()
    R_incompress = ((u_x + v_y) ** 2).mean()
    R_eqns = R_w + R_U + R_incompress

    x0yt = torch.cat([torch.zeros_like(x), y, t], dim=1).requires_grad_()
    x1yt = torch.cat([torch.ones_like(x), y, t], dim=1).requires_grad_()
    xy0t = torch.cat([x, torch.zeros_like(y), t], dim=1).requires_grad_()
    xy1t = torch.cat([x, torch.ones_like(y), t], dim=1).requires_grad_()
    xyt0 = torch.cat([x, y, torch.zeros_like(t)], dim=1)

    w_0y, u_0y, v_0y = model(x0yt)
    w_1y, u_1y, v_1y = model(x1yt)
    w_x0, u_x0, v_x0 = model(xy0t)
    w_x1, u_x1, v_x1 = model(xy1t)
    w_t0, u_t0, v_t0 = model(xyt0)

    R_BC_w = ((w_0y - w_1y) ** 2 + (w_x0 - w_x1) ** 2).mean()
    R_BC_u = ((u_0y - u_1y) ** 2 + (u_x0 - u_x1) ** 2).mean()
    R_BC_v = ((v_0y - v_1y) ** 2 + (v_x0 - v_x1) ** 2).mean()
    R_BC = R_BC_w + R_BC_u + R_BC_v

    w_IC, u_IC, v_IC = taylor_green_ic(xyt0, U0=U0, n=n)
    R_IC = ((w_t0 - w_IC) ** 2 + (u_t0 - u_IC) ** 2 + (v_t0 - v_IC) ** 2).mean()

    fixed_times = torch.rand(batch_size // 16, 1).squeeze().tolist()
    fixed_times.sort()

    R_gauge = 0
    for tval in fixed_times:
        xyt_input = torch.cat([x, y, tval * torch.ones_like(x)], dim=1)
        _, u_sample, v_sample = model(xyt_input)
        R_gauge += (u_sample.mean()) ** 2 + (v_sample.mean()) ** 2

    R_total = R_eqns + 10 * (R_BC + R_IC) + R_gauge

    opt.zero_grad()
    R_total.backward()
    opt.step()

    if step % 500 == 0:
        print(f"Step {step}, Loss {R_total.item():.4e}")
        print(
            f"R_w: {R_w.item():.2e}  "
            f"R_U: {R_U.item():.2e}  "
            f"R_incompress: {R_incompress.item():.2e}  "
            f"R_BC: {R_BC.item():.2e}  "
            f"R_IC: {R_IC.item():.2e}  "
            f"R_gauge: {R_gauge.item():.2e}"
        )


# --- Analytic Taylor-Green solution ---

def tg_w_psi(x, y, t, nu=nu, k=k, U=U0):
    x = torch.as_tensor(x)
    y = torch.as_tensor(y)
    t = torch.as_tensor(t, dtype=x.dtype, device=x.device)

    decay = torch.exp(-2 * nu * k**2 * t)
    kx = k * x
    ky = k * y

    w = 2.0 * U * k * torch.sin(kx) * torch.sin(ky) * decay
    u = U * torch.sin(kx) * torch.cos(ky) * decay
    v = -U * torch.cos(kx) * torch.sin(ky) * decay

    return w, u, v


# --- Plotting grid ---

n_plot = 80
xv = torch.linspace(0, 1, n_plot)
yv = torch.linspace(0, 1, n_plot)
X, Y = torch.meshgrid(xv, yv, indexing="ij")
xy = torch.stack([X.flatten(), Y.flatten()], dim=1)


# --- Particle animation setup ---

n_frames = 144
t_end = 2.0
t_vals = torch.linspace(0, t_end, n_frames)

n_particles = 400
arrow_scale = 25

px, py = np.meshgrid(
    np.linspace(0, 1, int(np.sqrt(n_particles))),
    np.linspace(0, 1, int(np.sqrt(n_particles))),
)
px = px.flatten()
py = py.flatten()
particle_pos = np.stack([px, py], axis=1)

dt = float(t_vals[1] - t_vals[0])

fig2, (axp_ml, axp_an) = plt.subplots(
    1,
    2,
    figsize=(14, 8),
    constrained_layout=False,
)
fig2.subplots_adjust(left=0.08, right=0.92, bottom=0.10, top=0.88, wspace=0.35)

axp_ml.set_title(r'$\mathrm{ML\ Vortex\ Simulation}$', fontsize=17)
axp_ml.set_xlabel(r'$x$', fontsize=15)
axp_ml.set_ylabel(r'$y$', fontsize=15)

axp_an.set_title(r'$\mathrm{Analytic\ Solution}$', fontsize=17)
axp_an.set_xlabel(r'$x$', fontsize=15)
axp_an.set_ylabel(r'$y$', fontsize=15)

axp_ml.tick_params(axis='both', labelsize=12)
axp_an.tick_params(axis='both', labelsize=12)


# --- Initial vorticity and velocity fields ---

with torch.no_grad():
    t0 = t_vals[0]
    xyt0_grid = torch.cat([xy, t0.repeat(xy.shape[0], 1)], dim=1)

    w_ml, u_ml, v_ml = model(xyt0_grid)
    w_ml = w_ml.reshape(n_plot, n_plot).cpu().numpy()
    u_ml = u_ml.reshape(n_plot, n_plot).cpu().numpy()
    v_ml = v_ml.reshape(n_plot, n_plot).cpu().numpy()

    w_an, u_an, v_an = tg_w_psi(X, Y, t0, nu, k=k, U=U0)
    w_an = w_an.cpu().numpy()
    u_an = u_an.cpu().numpy()
    v_an = v_an.cpu().numpy()

cmap = 'Spectral'
bg_ml = axp_ml.pcolormesh(X, Y, w_ml, cmap=cmap, shading='auto')
bg_an = axp_an.pcolormesh(X, Y, w_an, cmap=cmap, shading='auto')

divider_ml = make_axes_locatable(axp_ml)
cax_ml = divider_ml.append_axes("right", size="4%", pad=0.08)
fig2.colorbar(bg_ml, cax=cax_ml, orientation='vertical')

divider_an = make_axes_locatable(axp_an)
cax_an = divider_an.append_axes("right", size="4%", pad=0.08)
fig2.colorbar(bg_an, cax=cax_an, orientation='vertical')


# --- Velocity interpolation for moving tracers ---

def interpolate_velocity(u_field, v_field, X_grid, Y_grid, px_vals, py_vals):
    u_interp = np.zeros_like(px_vals)
    v_interp = np.zeros_like(py_vals)

    for idx in range(len(px_vals)):
        x_val = px_vals[idx]
        y_val = py_vals[idx]

        ix = np.searchsorted(X_grid[:, 0], x_val) - 1
        iy = np.searchsorted(Y_grid[0, :], y_val) - 1
        ix = np.clip(ix, 0, X_grid.shape[0] - 2)
        iy = np.clip(iy, 0, Y_grid.shape[1] - 2)

        x1, x2 = X_grid[ix, 0], X_grid[ix + 1, 0]
        y1, y2 = Y_grid[0, iy], Y_grid[0, iy + 1]

        wx2 = (x_val - x1) / (x2 - x1) if x2 != x1 else 0
        wx1 = 1 - wx2
        wy2 = (y_val - y1) / (y2 - y1) if y2 != y1 else 0
        wy1 = 1 - wy2

        u_interp[idx] = (
            u_field[ix, iy] * wx1 * wy1
            + u_field[ix + 1, iy] * wx2 * wy1
            + u_field[ix, iy + 1] * wx1 * wy2
            + u_field[ix + 1, iy + 1] * wx2 * wy2
        )
        v_interp[idx] = (
            v_field[ix, iy] * wx1 * wy1
            + v_field[ix + 1, iy] * wx2 * wy1
            + v_field[ix, iy + 1] * wx1 * wy2
            + v_field[ix + 1, iy + 1] * wx2 * wy2
        )

    return u_interp, v_interp


# --- Initial tracer arrows ---

particle_pos_ml = particle_pos.copy()
particle_pos_an = particle_pos.copy()

u_p_ml, v_p_ml = interpolate_velocity(u_ml, v_ml, X.numpy(), Y.numpy(), px, py)
u_p_an, v_p_an = interpolate_velocity(u_an, v_an, X.numpy(), Y.numpy(), px, py)

quiv_ml = axp_ml.quiver(px, py, u_p_ml, v_p_ml, color='k', scale=arrow_scale)
quiv_an = axp_an.quiver(px, py, u_p_an, v_p_an, color='k', scale=arrow_scale)


# --- Animation frame update ---

def update_particles(frame):
    global particle_pos_ml, particle_pos_an, quiv_ml, quiv_an

    if frame == 0:
        particle_pos_ml[:] = particle_pos.copy()
        particle_pos_an[:] = particle_pos.copy()

    t = t_vals[frame]
    xyt = torch.cat([xy, t.repeat(xy.shape[0], 1)], dim=1)

    with torch.no_grad():
        w_ml_frame, u_ml_frame, v_ml_frame = model(xyt)
        w_ml_frame = w_ml_frame.reshape(n_plot, n_plot).cpu().numpy()
        u_ml_frame = u_ml_frame.reshape(n_plot, n_plot).cpu().numpy()
        v_ml_frame = v_ml_frame.reshape(n_plot, n_plot).cpu().numpy()

        w_an_frame, u_an_frame, v_an_frame = tg_w_psi(
            X,
            Y,
            t,
            nu,
            k=k,
            U=U0,
        )
        w_an_frame = w_an_frame.cpu().numpy()
        u_an_frame = u_an_frame.cpu().numpy()
        v_an_frame = v_an_frame.cpu().numpy()

    bg_ml.set_array(w_ml_frame.ravel())
    bg_an.set_array(w_an_frame.ravel())

    u_p_ml, v_p_ml = interpolate_velocity(
        u_ml_frame,
        v_ml_frame,
        X.numpy(),
        Y.numpy(),
        particle_pos_ml[:, 0],
        particle_pos_ml[:, 1],
    )
    u_p_an, v_p_an = interpolate_velocity(
        u_an_frame,
        v_an_frame,
        X.numpy(),
        Y.numpy(),
        particle_pos_an[:, 0],
        particle_pos_an[:, 1],
    )

    particle_pos_ml[:, 0] += u_p_ml * dt
    particle_pos_ml[:, 1] += v_p_ml * dt
    particle_pos_an[:, 0] += u_p_an * dt
    particle_pos_an[:, 1] += v_p_an * dt

    particle_pos_ml = np.mod(particle_pos_ml, 1.0)
    particle_pos_an = np.mod(particle_pos_an, 1.0)

    quiv_ml.remove()
    quiv_an.remove()

    quiv_ml = axp_ml.quiver(
        particle_pos_ml[:, 0],
        particle_pos_ml[:, 1],
        u_p_ml,
        v_p_ml,
        color='k',
        scale=arrow_scale,
    )
    quiv_an = axp_an.quiver(
        particle_pos_an[:, 0],
        particle_pos_an[:, 1],
        u_p_an,
        v_p_an,
        color='k',
        scale=arrow_scale,
    )

    return bg_ml, bg_an, quiv_ml, quiv_an


# --- Save animation as MP4 ---

ani_particles = animation.FuncAnimation(
    fig2,
    update_particles,
    frames=n_frames,
    interval=33,
    blit=False,
)

fig2.subplots_adjust(left=0.06, right=0.94, bottom=0.09, top=0.91, wspace=0.25)

ani_particles.save(
    'vortex_comparison.mp4',
    writer='ffmpeg',
    fps=24,
    dpi=150,
)

plt.show()