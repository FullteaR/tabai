"""Single-GPU versus multi-GPU comparison benchmark; use --help for shared CLI options."""
if __package__:
    from .runner import main as run_cli
else:
    from runner import main as run_cli


def main(argv=None):
    return run_cli(argv, operations=['mul', 'square', 'pow', 'div'], gpu_mode='both', backends=['tabai'], bits=[1_000_000, 10_000_000, 100_000_000], exponents=[3])


if __name__ == '__main__':
    raise SystemExit(main())
