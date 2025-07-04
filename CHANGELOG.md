# Changelog

<!--

Changelog follow the https://keepachangelog.com/ standard (at least the headers)

This allow to:

* auto-parsing release notes during the automated releases from github-action:
  https://github.com/marketplace/actions/pypi-github-auto-release
* Have clickable headers in the rendered markdown

To release a new version (e.g. from `1.0.0` -> `2.0.0`):

* Create a new `# [2.0.0] - YYYY-MM-DD` header and add the current
  `[Unreleased]` notes.
* At the end of the file:
  * Define the new link url:
  `[2.0.0]: https://github.com/google-research/my_project/compare/v1.0.0...v2.0.0`
  * Update the `[Unreleased]` url: `v1.0.0...HEAD` -> `v2.0.0...HEAD`
-->

## [Unreleased]

## [1.0.0] - 2025-07-01

* Initial release: support evaluation of a mock model and a [Hugging Face VideoMAE] model on SciVid benchmarks.

[Unreleased]: https://github.com/google-deepmind/scivid/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/google-deepmind/scivid/compare/v1.0.0
[Hugging Face VideoMAE]: https://huggingface.co/MCG-NJU/videomae-base
