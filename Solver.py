import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# === PARAMETERS ===
omega_s = 10.0 * 2 * np.pi
alpha_s = 20.0 * 2 * np.pi
A = 0.01
U_inf = 2.0
nu = 0.01
rho = 1.2  # Nondimensional density
T_total = 0.1
fps = 300

# === GRID ===
x = np.linspace(0, 0.2, 200)
y = np.linspace(0, 0.1, 200)
X, Y = np.meshgrid(x, y)
t_vals = np.linspace(0, T_total, int(T_total * fps))

# === DERIVED CONSTANTS ===
lambda_ = omega_s - alpha_s * U_inf

# === HANDLE RESONANCE SPECIAL CASE ===
resonance = np.isclose(lambda_, 0, atol=1e-10)

if resonance:
    def v_hat(y):
        return -A * omega_s * np.exp(-alpha_s * y)


    def dv_hat(y):
        return A * omega_s * alpha_s * np.exp(-alpha_s * y)
else:
    beta = np.sqrt(alpha_s ** 2 + lambda_ / (1j * nu))
    beta_r, beta_i = np.real(beta), np.imag(beta)
    denom = (beta_r - alpha_s) ** 2 + beta_i ** 2
    a4 = A * omega_s * alpha_s * (beta_r - alpha_s) / denom
    b4 = -A * omega_s * alpha_s * beta_i / denom
    C4 = a4 + 1j * b4
    C2 = - (beta / alpha_s) * C4


    def v_hat(y):
        term1 = C2 * np.exp(-alpha_s * y)
        term2 = C4 * np.exp(-beta_r * y) * np.exp(-1j * beta_i * y)
        return term1 + term2


    def dv_hat(y):
        term1 = -alpha_s * C2 * np.exp(-alpha_s * y)
        term2 = -beta * C4 * np.exp(-beta * y)
        return term1 + term2


def u_hat(y):
    return -1 / (1j * alpha_s) * dv_hat(y)


def velocity_magnitude(x, y, t):
    Vhat = v_hat(y)
    Uhat = u_hat(y)
    phase = alpha_s * x - omega_s * t

    # shape correction for broadcasting
    V = np.real(Vhat[:, np.newaxis] * np.exp(1j * phase))
    U = np.real(Uhat[:, np.newaxis] * np.exp(1j * phase))

    # ADD base flow
    U_total = U_inf + U
    return np.sqrt(U_total ** 2 + V ** 2)


def get_wall_forces(x, t):
    V0 = v_hat(0)
    dVdy0 = dv_hat(0)
    d2Vdy2 = (
        -alpha_s ** 2 * C2 * np.exp(-alpha_s * 0)
        + (-beta ** 2 * C4 * np.exp(-beta * 0)) if not resonance else 0
    )
    d3Vdy3 = (
        -alpha_s ** 3 * C2 * np.exp(-alpha_s * 0)
        + (-beta ** 3 * C4 * np.exp(-beta * 0)) if not resonance else 0
    )

    # u_hat and derivatives
    U0 = -1 / (1j * alpha_s) * dVdy0
    dUdy0 = -1 / (1j * alpha_s) * d2Vdy2

    # Horizontal force
    shear_hat = nu * (dUdy0 + 1j * alpha_s * V0)
    f_x = np.real(shear_hat * np.exp(1j * (alpha_s * x - omega_s * t)))

    # Vertical force from pressure
    pressure_hat = -(rho / (1j * alpha_s ** 2)) * (
            lambda_ * dVdy0 + nu * (d3Vdy3 - alpha_s ** 2 * dVdy0)
    )
    f_y = -np.real(pressure_hat * np.exp(1j * (alpha_s * x - omega_s * t)))

    return f_x, f_y


# === PLOT ===
fig, ax = plt.subplots(figsize=(8, 4))
contour = ax.contourf(X, Y, velocity_magnitude(x, y, 0), levels=100, cmap='viridis')
fig.colorbar(contour)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Velocity Magnitude $|\mathbf{u}\'(x,y,t)|$')


# === ANIMATION FUNCTION ===
def update(frame):
    ax.clear()
    t = t_vals[frame]
    Z = velocity_magnitude(x, y, t)
    contour = ax.contourf(X, Y, Z, levels=100, cmap='viridis')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'$|\mathbf{{u}}\'(x, y, t={t:.2f}s)|$')


ani = animation.FuncAnimation(fig, update, frames=len(t_vals), blit=False, interval=1000 / fps)
plt.tight_layout()
plt.show()

# === ANIMATE FORCES ===
fig, ax = plt.subplots(2, 1, figsize=(8, 6))

line_fx, = ax[0].plot(x, np.zeros_like(x), lw=2, label="Shear force $f_x$")
line_fy, = ax[1].plot(x, np.zeros_like(x), lw=2, label="Normal force $f_y$")
for a in ax:
    a.set_xlim(x.min(), x.max())
    a.set_ylim(-2, 2)
    a.grid(True)
    a.legend()

ax[0].set_ylabel('$f_x(x,t)$')
ax[1].set_ylabel('$f_y(x,t)$')
ax[1].set_xlabel('$x$')
fig.suptitle("Forces on Wall Over Time")


def update_force(frame):
    t = t_vals[frame]
    fx, fy = get_wall_forces(x, t)
    line_fx.set_ydata(fx)
    line_fy.set_ydata(fy)
    ax[0].set_title(f"$t = {t:.2f}$ s")
    return line_fx, line_fy


ani = animation.FuncAnimation(fig, update_force, frames=len(t_vals), blit=True, interval=1000 / fps)
plt.tight_layout()
plt.show()
