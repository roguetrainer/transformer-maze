# Contributing to Transformer Maze

Thank you for your interest in contributing! This project aims to make transformer architecture accessible through progressive implementation and clear pedagogy.

## Ways to Contribute

### 1. Educational Content
- **New notebooks**: Additional architectures or variants
- **Exercises**: Practice problems for learners
- **Solutions**: Answer keys for exercises
- **Translations**: Documentation in other languages

### 2. Code Improvements
- **Optimization**: Faster implementations
- **New features**: Additional maze types, visualizations
- **Bug fixes**: Issues with current code
- **Tests**: Expand test coverage

### 3. Documentation
- **Clarifications**: Improve unclear explanations
- **Examples**: Additional usage examples
- **Architecture guide**: More detailed derivations
- **FAQs**: Common questions and answers

### 4. Visualization
- **Interactive demos**: Web-based visualizations
- **Animations**: Training process visualizations
- **Diagrams**: Architecture illustrations
- **Comparisons**: Side-by-side model behavior

## Getting Started

### Development Setup

```bash
# Fork and clone the repository
git clone https://github.com/YOUR_USERNAME/transformer-maze.git
cd transformer-maze

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest tests/
```

### Code Style

We follow standard Python conventions:

- **PEP 8**: Style guide for Python code
- **Type hints**: Use where appropriate
- **Docstrings**: Google style for all public functions
- **Comments**: Explain *why*, not *what*

Format code with:
```bash
black src/ tests/
isort src/ tests/
flake8 src/ tests/
```

### Commit Messages

Follow conventional commits:
- `feat:` New features
- `fix:` Bug fixes
- `docs:` Documentation changes
- `style:` Formatting, missing semicolons, etc.
- `refactor:` Code restructuring
- `test:` Adding tests
- `chore:` Maintenance tasks

Example: `feat: add sparse attention implementation to notebook 5`

## Contribution Guidelines

### For New Notebooks

1. **Follow the progression**: Build on previous notebooks
2. **Start simple**: Basic example before complexity
3. **Visualize heavily**: Every concept needs a plot
4. **Explain why**: Not just what the code does
5. **Test thoroughly**: Verify on multiple examples
6. **Add to README**: Update learning path section

Template structure:
```markdown
# Notebook Title

## Learning Objectives
- [ ] Objective 1
- [ ] Objective 2

## Prerequisites
- Notebook X
- Concept Y

## Sections
1. Introduction & Motivation
2. Theory (with math)
3. Implementation
4. Visualization
5. Exercises
6. Summary

## Estimated Time
~X hours
```

### For Code Contributions

1. **Create issue first**: Discuss before implementing
2. **Single purpose**: One feature/fix per PR
3. **Add tests**: For new functionality
4. **Update docs**: If API changes
5. **Check compatibility**: Test on Python 3.8+

### For Documentation

1. **Clear examples**: Show don't just tell
2. **Correct math**: Double-check equations
3. **Cite sources**: Proper attribution
4. **Accessible language**: Avoid unnecessary jargon
5. **Proofread**: Check spelling and grammar

## Pull Request Process

1. **Fork** the repository
2. **Create branch**: `git checkout -b feature/your-feature-name`
3. **Make changes**: Following guidelines above
4. **Test**: Run full test suite
5. **Commit**: With clear messages
6. **Push**: To your fork
7. **Open PR**: With description of changes

### PR Description Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation
- [ ] Other (specify)

## Testing
How was this tested?

## Related Issues
Fixes #(issue number)

## Checklist
- [ ] Code follows style guidelines
- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] All tests pass
```

## Code Review Process

Maintainers will review PRs considering:
- **Correctness**: Does it work as intended?
- **Clarity**: Is it understandable?
- **Completeness**: Tests and docs included?
- **Consistency**: Matches project style?

You may be asked to:
- Make changes
- Add tests
- Clarify documentation
- Rebase on main

## Architecture Decisions

Major changes should be discussed via issues first:
- New module structure
- API changes
- Additional dependencies
- Framework changes

## Testing Requirements

All code should include:
- **Unit tests**: For individual functions
- **Integration tests**: For module interactions
- **Notebook tests**: Verify notebooks run end-to-end

Run tests:
```bash
# All tests
pytest

# With coverage
pytest --cov=src tests/

# Specific test file
pytest tests/test_maze_envs.py

# Notebook tests
pytest --nbmake notebooks/
```

## Documentation Standards

### Docstrings (Google Style)

```python
def function_name(arg1: type, arg2: type) -> return_type:
    """Brief description.
    
    Longer description if needed.
    
    Args:
        arg1: Description of arg1
        arg2: Description of arg2
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When this happens
        
    Example:
        >>> result = function_name(1, 2)
        >>> print(result)
        3
    """
```

### README Updates

If adding features, update:
- Feature list
- Installation instructions
- Usage examples
- Learning path (if new notebook)

## Community

### Code of Conduct

- Be respectful and inclusive
- Welcome newcomers
- Accept constructive criticism
- Focus on what's best for the project
- Show empathy towards others

### Getting Help

- **Issues**: For bugs and feature requests
- **Discussions**: For questions and ideas
- **Email**: For private concerns

### Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md
- Mentioned in release notes
- Acknowledged in citations

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Questions?

Open an issue with the `question` label or start a discussion. We're happy to help!

---

Thank you for helping make transformer architecture more accessible! 🎉
