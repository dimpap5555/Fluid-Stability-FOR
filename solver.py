import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class LinearNavierStokesSolver:
    """Solve the linearized Navier–Stokes equations for a uniform base flow."""

    def __init__(self,
                 omega_s=20.0 * 2 * np.pi,
                 alpha_s=20.0 * 2 * np.pi,
                 amplitude=0.01,
                 u_inf=2.0,
                 nu=0.01,
                 rho=1.2,
                 t_total=0.1,
                 fps=300,
                 x_bounds=(0.0, 0.2),
                 y_bounds=(0.0, 0.1),
                 x_points=200,
                 y_points=200):
        self.omega_s = omega_s
        self.alpha_s = alpha_s
        self.amplitude = amplitude
        self.free_stream_velocity = u_inf
        self.viscosity = nu
        self.density = rho
        self.total_time = t_total
        self.frames_per_second = fps

        self.x = np.linspace(*x_bounds, x_points)
        self.y = np.linspace(*y_bounds, y_points)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        self.t_vals = np.linspace(0, self.total_time,
                                 int(self.total_time * self.frames_per_second))

        self.frequency_offset = omega_s - alpha_s * self.free_stream_velocity
        self.resonance = np.isclose(self.frequency_offset, 0, atol=1e-10)

        self._compute_coefficients()

    def _compute_coefficients(self):
        if self.resonance:
            self.beta = None
            self.C2 = None
            self.C4 = None
        else:
            beta = np.sqrt(self.alpha_s ** 2 + self.frequency_offset / (1j * self.viscosity))
            self.beta = beta
            beta_r, beta_i = np.real(beta), np.imag(beta)
            denom = (beta_r - self.alpha_s) ** 2 + beta_i ** 2
            a4 = self.amplitude * self.omega_s * self.alpha_s * (beta_r - self.alpha_s) / denom
            b4 = -self.amplitude * self.omega_s * self.alpha_s * beta_i / denom
            self.C4 = a4 + 1j * b4
            self.C2 = -(beta / self.alpha_s) * self.C4

    # === Profile functions ===
    def v_hat(self, y):
        if self.resonance:
            return -self.amplitude * self.omega_s * np.exp(-self.alpha_s * y)
        term1 = self.C2 * np.exp(-self.alpha_s * y)
        term2 = self.C4 * np.exp(-self.beta * y)
        return term1 + term2

    def dv_hat(self, y):
        if self.resonance:
            return self.amplitude * self.omega_s * self.alpha_s * np.exp(-self.alpha_s * y)
        term1 = -self.alpha_s * self.C2 * np.exp(-self.alpha_s * y)
        term2 = -self.beta * self.C4 * np.exp(-self.beta * y)
        return term1 + term2

    def d2v_hat(self, y):
        if self.resonance:
            return -self.amplitude * self.omega_s * self.alpha_s ** 2 * np.exp(-self.alpha_s * y)
        term1 = self.alpha_s ** 2 * self.C2 * np.exp(-self.alpha_s * y)
        term2 = self.beta ** 2 * self.C4 * np.exp(-self.beta * y)
        return term1 + term2

    def d3v_hat(self, y):
        if self.resonance:
            return self.amplitude * self.omega_s * self.alpha_s ** 3 * np.exp(-self.alpha_s * y)
        term1 = -self.alpha_s ** 3 * self.C2 * np.exp(-self.alpha_s * y)
        term2 = -self.beta ** 3 * self.C4 * np.exp(-self.beta * y)
        return term1 + term2

    def u_hat(self, y):
        return -1 / (1j * self.alpha_s) * self.dv_hat(y)

    # === Utility computations ===
    def velocity_magnitude(self, t):
        Vhat = self.v_hat(self.y)
        Uhat = self.u_hat(self.y)
        phase = self.alpha_s * self.x - self.omega_s * t
        V = np.real(Vhat[:, np.newaxis] * np.exp(1j * phase))
        U = np.real(Uhat[:, np.newaxis] * np.exp(1j * phase))
        U_total = self.free_stream_velocity + U
        return np.sqrt(U_total ** 2 + V ** 2)

    def get_wall_forces(self, t):
        V0 = self.v_hat(0)
        dVdy0 = self.dv_hat(0)
        d2Vdy2 = self.d2v_hat(0)
        d3Vdy3 = self.d3v_hat(0)

        U0 = -1 / (1j * self.alpha_s) * dVdy0
        dUdy0 = -1 / (1j * self.alpha_s) * d2Vdy2

        shear_hat = self.viscosity * (dUdy0 + 1j * self.alpha_s * V0)
        fx = np.real(shear_hat * np.exp(1j * (self.alpha_s * self.x - self.omega_s * t)))

        pressure_hat = -(self.density / (1j * self.alpha_s ** 2)) * (
            self.frequency_offset * dVdy0
            + self.viscosity * (d3Vdy3 - self.alpha_s ** 2 * dVdy0)
        )
        fy = -np.real(pressure_hat * np.exp(1j * (self.alpha_s * self.x - self.omega_s * t)))
        return fx, fy

    # === Plotting helpers ===
    def animate_velocity(self):
        fig, ax = plt.subplots(figsize=(8, 4))
        contour = ax.contourf(self.X, self.Y, self.velocity_magnitude(0), levels=100, cmap="viridis")
        fig.colorbar(contour)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Velocity Magnitude $|\\mathbf{u}'(x,y,t)|$")

        def update(frame):
            ax.clear()
            t = self.t_vals[frame]
            Z = self.velocity_magnitude(t)
            contour = ax.contourf(self.X, self.Y, Z, levels=100, cmap="viridis")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_title(f"$|\\mathbf{{u}}'(x, y, t={t:.2f}s)|$")

        ani = animation.FuncAnimation(fig, update, frames=len(self.t_vals),
                                      interval=1000 / self.frames_per_second)
        plt.tight_layout()
        plt.show()
        return ani

    def animate_wall_forces(self):
        fig, ax = plt.subplots(2, 1, figsize=(8, 6))
        line_fx, = ax[0].plot(self.x, np.zeros_like(self.x), lw=2, label="Shear force $f_x$")
        line_fy, = ax[1].plot(self.x, np.zeros_like(self.x), lw=2, label="Normal force $f_y$")
        for a in ax:
            a.set_xlim(self.x.min(), self.x.max())
            a.set_ylim(-2, 2)
            a.grid(True)
            a.legend()
        ax[0].set_ylabel("$f_x(x,t)$")
        ax[1].set_ylabel("$f_y(x,t)$")
        ax[1].set_xlabel("$x$")
        fig.suptitle("Forces on Wall Over Time")

        def update_force(frame):
            t = self.t_vals[frame]
            fx, fy = self.get_wall_forces(t)
            line_fx.set_ydata(fx)
            line_fy.set_ydata(fy)
            ax[0].set_title(f"$t = {t:.2f}$ s")
            return line_fx, line_fy

        ani = animation.FuncAnimation(fig, update_force, frames=len(self.t_vals),
                                      blit=True, interval=1000 / self.frames_per_second)
        plt.tight_layout()
        plt.show()
        return ani
