"""Floor division and modulo benchmark; use --help for shared CLI options."""
if __package__:
    from .runner import main as run_cli
else:
    from runner import main as run_cli


def main(argv=None):
    return run_cli(argv, operations=['div', 'mod'])


if __name__ == '__main__':
    raise SystemExit(main())
