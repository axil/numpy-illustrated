# Publishing releases

Releases use PyPI Trusted Publishing through GitHub Actions. No PyPI API token
is needed. The package metadata remains in `setup.py`; `pyproject.toml` declares
the build backend.

## One-time setup

1. Create the `pypi` environment in the
   [repository environment settings](https://github.com/axil/numpy-illustrated/settings/environments).
   Configure a required reviewer if you want approval before each upload.
2. Open the existing package's
   [PyPI publishing settings](https://pypi.org/manage/project/numpy-illustrated/settings/publishing/)
   as a project owner and add a GitHub Trusted Publisher with these values:

   | Field | Value |
   | --- | --- |
   | Owner | `axil` |
   | Repository | `numpy-illustrated` |
   | Workflow filename | `publish.yml` |
   | Environment | `pypi` |

   The workflow filename is just `publish.yml`, without `.github/workflows/`.
   See the [PyPI setup guide](https://docs.pypi.org/trusted-publishers/adding-a-publisher/).
3. Commit and push the workflow and build configuration before making a release.

## Release procedure

1. Update the version in both `setup.py` and `npi/__init__.py` to a new version
   that has not already been published to PyPI. Commit and push the changes.
2. Optionally run **Publish to PyPI** manually from GitHub Actions. This runs the
   tests and builds downloadable distributions without uploading to PyPI.
3. Create and publish a GitHub release at the release commit, with a tag matching
   the package version, such as `v0.3.2` (or `0.3.2`). A draft does not trigger
   publishing.
4. The workflow runs the existing Python/NumPy test matrix, builds a wheel and
   source distribution, checks their metadata, and tests the installed wheel.
   It checks that the package version, `npi.__version__`, and release tag agree.
5. Approve the `pypi` environment job if required. The publishing job downloads
   the validated distributions and uploads them using short-lived credentials.

The build runs separately from the publishing job. Only the publishing job has
the `id-token: write` permission required for Trusted Publishing. See the
[PyPA publishing guide](https://packaging.python.org/en/latest/guides/publishing-package-distribution-releases-using-github-actions-ci-cd-workflows/).
