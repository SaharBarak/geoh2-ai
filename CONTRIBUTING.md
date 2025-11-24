# Contributing to H2 Seep Detection

Thank you for your interest in contributing to the H2 Seep Detection project!

## Development Setup

1. **Fork and Clone**
   ```bash
   git fork https://github.com/yourusername/h2-seep-detection.git
   cd h2-seep-detection
   ```

2. **Create Virtual Environment**
   ```bash
   conda create -n h2detect python=3.10
   conda activate h2detect
   ```

3. **Install Dependencies**
   ```bash
   make install-dev
   ```

## Code Style

- Follow PEP 8 guidelines
- Use type hints throughout
- Write docstrings for all public functions/classes
- Format code with Black: `make format`
- Run linting: `make lint`

## Testing

- Write tests for new features
- Maintain test coverage > 80%
- Run tests: `make test`

## Pull Request Process

1. Create a feature branch: `git checkout -b feature/your-feature`
2. Make your changes with clear commit messages
3. Add tests for new functionality
4. Ensure all tests pass: `make test`
5. Format code: `make format`
6. Push and create PR

## Commit Messages

Follow conventional commits format:
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation
- `test:` Tests
- `refactor:` Code refactoring

Example: `feat: add sentinel-2 cloud masking`

## Questions?

Open an issue for discussions or questions.
