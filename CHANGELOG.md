# Changelog
All notable changes to this packages will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
- Prediction intervals method
### Fixed
- Jacobian calculation for 4PL model
### Changed
- Modified 4PL formula to avoid warning associated with complex numbers
- Renamed std dev method to confidence interval to clarify meaning
- Split fit based on whether model is positively or negatively sloping
### Deprecated
### Removed
### Security