# Release process

1. Run the unit tests and the real-data worked example.
2. Validate the complete dataset and its checksums.
3. Confirm that no manuscript files, publication graphics or credentials are tracked.
4. Build and install the Python package in a clean environment.
5. Tag the repository with a semantic version such as `v1.0.0`.
6. Create a GitHub release from that tag.
7. Archive the tagged software release through Zenodo.
8. Publish the separately curated research-data record.
9. Record the version-specific software and dataset DOIs in the release notes.

Published Zenodo files are immutable. Scientific or file changes require a new software or dataset version; metadata-only corrections should be documented in the release notes.
