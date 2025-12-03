# Contributing to Convert and Quantize Library

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing to the project.

## Code of Conduct

- Be respectful and inclusive
- Report issues and provide constructive feedback
- Help others and be open to learning

## Getting Started

### 1. Fork and Clone

```bash
# Fork the repository on GitHub
# Clone your fork
git clone https://github.com/YOUR_USERNAME/ConvertAndQuantizeLibrary.git
cd ConvertAndQuantizeLibrary

# Add upstream remote
git remote add upstream https://github.com/silveroxides/ConvertAndQuantizeLibrary.git
```

### 2. Set Up Development Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode with all dependencies
pip install -e ".[dev]"

# Install pre-commit hooks (optional)
pip install pre-commit
pre-commit install
```

### 3. Create Feature Branch

```bash
# Update main branch
git fetch upstream
git rebase upstream/main

# Create feature branch
git checkout -b feature/your-feature-name
```

## Development Workflow

### Code Style

We follow PEP 8 and use tools to enforce consistency:

```bash
# Format code with black
black convert_and_quantize/ tests/ examples/

# Sort imports with isort
isort convert_and_quantize/ tests/ examples/

# Check with flake8
flake8 convert_and_quantize/ tests/ examples/

# Type checking with mypy
mypy convert_and_quantize/
```

### Testing

Write tests for new features in the `tests/` directory:

```bash
# Run tests
pytest tests/

# Run with coverage
pytest --cov=convert_and_quantize tests/

# Run specific test
pytest tests/test_converter.py::test_basic_conversion
```

### Documentation

- Add docstrings to all classes and functions
- Follow Google-style docstrings
- Update README.md if adding features
- Add examples in `examples/` for new functionality

#### Docstring Example

```python
def my_function(param1: str, param2: int) -> bool:
    """
    Short description of what the function does.
    
    Longer description if needed. Explain the purpose,
    behavior, and any important details.
    
    Args:
        param1: Description of param1
        param2: Description of param2
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When something is invalid
        
    Example:
        >>> result = my_function("test", 42)
        >>> print(result)
        True
    """
    pass
```

## Types of Contributions

### Bug Reports

Report bugs by opening an issue with:

- Clear title describing the bug
- Steps to reproduce
- Expected behavior
- Actual behavior
- Python and PyTorch versions
- Relevant code snippets

### Feature Requests

Propose features with:

- Clear description of the feature
- Use case and motivation
- Example usage
- Any implementation ideas

### Code Contributions

#### Adding New Features

1. **Check existing issues** - See if someone is already working on it
2. **Create an issue** - Describe the feature and get feedback
3. **Implement** - Follow the code style guidelines
4. **Add tests** - Ensure good test coverage
5. **Update docs** - Add docstrings and update README if needed
6. **Submit PR** - Reference the issue in your PR

#### Fixing Bugs

1. **Create an issue** - If one doesn't exist
2. **Create a test** - That reproduces the bug
3. **Fix the bug** - Make the test pass
4. **Update docs** - If the bug fix changes behavior
5. **Submit PR** - Reference the issue

### Documentation Improvements

- Fix typos and grammar
- Improve clarity
- Add missing sections
- Update examples

## Pull Request Process

### Before Submitting

```bash
# Update your branch with latest changes
git fetch upstream
git rebase upstream/main

# Run tests and checks
pytest tests/
black --check convert_and_quantize/ tests/ examples/
isort --check convert_and_quantize/ tests/ examples/
flake8 convert_and_quantize/ tests/ examples/
```

### Submitting PR

1. **Push to your fork**

   ```bash
   git push origin feature/your-feature-name
   ```

2. **Create PR on GitHub**
   - Reference related issue: `Fixes #123`
   - Describe changes clearly
   - Include any breaking changes
   - Add tests added/modified
   - Update documentation links

3. **PR Description Template**

   ```markdown
   ## Description
   Brief description of changes
   
   ## Type of Change
   - [ ] Bug fix
   - [ ] New feature
   - [ ] Documentation update
   - [ ] Performance improvement
   
   ## Related Issue
   Fixes #(issue number)
   
   ## Testing
   Describe testing performed
   
   ## Checklist
   - [ ] Tests added/updated
   - [ ] Documentation updated
   - [ ] Code follows style guidelines
   - [ ] No breaking changes
   ```

### Code Review

- Be open to feedback
- Respond to comments promptly
- Make requested changes in new commits (don't force push unless asked)
- Thank reviewers

## Project Structure

```markdown
convert_and_quantize/
├── __init__.py              # Package initialization
├── constants.py             # Constants and model configs
├── core/
│   ├── __init__.py
│   └── converter.py         # Main converter class
├── optimizers/
│   └── __init__.py          # Optimization implementations
└── utils/
    └── __init__.py          # Utility functions

tests/
├── __init__.py
├── test_converter.py        # Converter tests
├── test_optimizers.py       # Optimizer tests
└── test_utils.py            # Utility function tests

examples/
├── 01_basic_quantization.py
├── 02_compare_optimizers.py
├── 03_block_vs_tensor_scaling.py
└── 04_quantize_safetensors.py

docs/
├── INSTALLATION.md
├── QUICKSTART.md
├── API.md
└── CONTRIBUTING.md
```

## Tips for Successful Contributions

1. **Start small** - Begin with small fixes or improvements
2. **Communicate** - Discuss major changes in issues first
3. **Write tests** - Good tests ensure code quality
4. **Document well** - Clear documentation helps others understand
5. **Be patient** - Reviews take time; maintainers have limited bandwidth

## Development Tips

### Running a Single Test

```bash
pytest tests/test_converter.py::test_function_name -v
```

### Debugging with print statements

```python
import sys
print(f"Debug: {variable}", file=sys.stderr)
```

### Profiling Performance

```python
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# Code to profile
your_function()

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(10)
```

### Building Distribution

```bash
# Build wheel and source distribution
pip install build
python -m build

# Check distribution
twine check dist/*
```

## Getting Help

- Check existing issues and discussions
- Read the documentation
- Look at examples
- Ask in issue comments
- Join discussions if available

## Recognition

Contributors will be:

- Added to CONTRIBUTORS file
- Credited in release notes
- Recognized in documentation

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Additional Resources

- [GitHub Help](https://help.github.com)
- [How to Contribute to Open Source](https://opensource.guide/how-to-contribute/)
- [PEP 8 Style Guide](https://www.python.org/dev/peps/pep-0008/)
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)

---

Thank you for contributing to Convert and Quantize Library! 🎉
