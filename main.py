from solver import LinearNavierStokesSolver


def main() -> None:
    """Run a demo of the linearized Navier–Stokes solver."""
    solver = LinearNavierStokesSolver()
    solver.animate_velocity()
    solver.animate_wall_forces()


if __name__ == "__main__":
    main()
