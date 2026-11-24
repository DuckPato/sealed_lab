# Contributing

Thanks for wanting to improve sealed_lab! Fixes, new report ideas and better wording are all welcome.

## Sending a change

1. Fork this repository on GitHub and make your change on a branch of your fork.
2. Run the app from source and the tests (see [DEVELOPING.md](DEVELOPING.md)):
   ```
   uv run --extra dev pytest
   ```
   The full suite takes about half an hour because some tests fit models. Set `SEALED_LAB_QUICK_TESTS=1` to run only the quick part,
   which is what GitHub runs.
3. Open a pull request against `main`. Say what the change does and why, and add a screenshot if it changes what the app looks like.
   GitHub then runs the quick tests on Windows and Ubuntu (for a first contribution, after the maintainer approves the run).

## What happens next

This repository is published from the maintainer's own development repository. When your pull request is accepted, its commits
are imported there (you stay the author) and reach GitHub with the next release. The pull request is then closed with a link to that
release instead of being merged with the button, so don't worry when it shows as "closed".

Every release is signed by the maintainer, and the app installs only signed releases, so a change reaches other people's apps only after
it has been reviewed and released.

## Please

- Keep a pull request to one topic. Small ones are reviewed faster.
- Never include personal data: Arena logs, 17Lands data files, folder paths with your user name, tokens or passwords.
- Ask before adding a dependency, since everything in `uv.lock` is installed on everyone's computer. If one is agreed, update the lock
  file with `uv lock`.
- Changes to the updater, release signing, install or update scripts, GitHub workflows or downloads get a particularly careful review,
  so expect those to take longer.
- Found a security problem? Report it privately, as described in [SECURITY.md](SECURITY.md), not in a pull request or issue.

By contributing, you agree that your contribution is released under this repository's [MIT license](LICENSE).
