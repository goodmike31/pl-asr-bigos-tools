# BIGOS Developer Guide

<div align="center">
  <h3>A comprehensive guide for contributors to the BIGOS ASR Evaluation Framework</h3>
</div>

## Introduction

The BIGOS framework represents a significant contribution to the field of speech recognition evaluation, particularly for Polish language resources. As you embark on extending or modifying this codebase, this guide will serve as your map through its architecture and design principles.

At its core, BIGOS exemplifies what we might call an **evaluation orchestration system** – a carefully designed infrastructure that coordinates multiple components to produce standardized assessments of ASR performance. Unlike simpler evaluation scripts, BIGOS manages the complete workflow from dataset handling to metric calculation, creating a reproducible evaluation environment.

> **Architectural Philosophy**: BIGOS follows a modular design that separates concerns between dataset management, ASR system interfaces, evaluation pipelines, and results analysis. This separation allows independent evolution of components while maintaining system coherence.

## Core Architecture 

The framework's architecture follows a layered approach, with each layer building upon the capabilities of those beneath it. Understanding this structure is essential for effective contribution.

### Conceptual Layers

1. **Foundation Layer**: Configuration management, utility functions, and data structures
2. **System Layer**: ASR system wrappers and interfaces
3. **Processing Layer**: Data preparation and transformation 
4. **Evaluation Layer**: Metric calculation and results aggregation
5. **Presentation Layer**: Visualization and reporting capabilities

This layering isn't merely conceptual—it's reflected in the code organization and dependencies. Changes to lower layers ripple upward, while higher layers can be modified without affecting foundational components.

### Data Flow Architecture

The data flows through BIGOS in a distinct pattern that mirrors the scientific method:

1. **Hypothesis Generation**: Audio samples are processed by ASR systems, producing transcription hypotheses
2. **Reference Comparison**: Hypotheses are compared against reference transcriptions
3. **Measurement**: Discrepancies are quantified using established metrics
4. **Analysis**: Results are aggregated and examined across dimensions

This flow can be visualized as both a directed acyclic graph (for a single evaluation) and as a cyclical process (for iterative improvement of ASR systems). Your contributions should maintain this logical progression.

## Key Components

### ASR System Interface

The `BaseASRSystem` class forms the core abstraction for all ASR system implementations. It defines a common protocol through which the framework interacts with diverse ASR engines:

```python
class BaseASRSystem:
    def __init__(self, system, model, language_code) -> None:
        # Initialize the system
        
    def process_audio(self, speech_file, force_hyps) -> str:
        # Process audio file and handle caching logic
        
    def generate_asr_hyp(self, speech_file) -> str:
        # Must be implemented by subclasses
```

This design applies the **adapter pattern** – each specific ASR system implementation adapts its unique API to conform to this standard interface. This pattern enables the framework to treat all ASR systems uniformly despite their underlying differences.

> **Design Pattern**: The adapter pattern creates a unified interface for disparate systems. In BIGOS, it allows us to abstract away the complexities of various ASR APIs, presenting a simplified, consistent interface to the rest of the framework.

### Prefect Workflows

BIGOS uses Prefect for workflow orchestration, which brings several advantages:

1. **Reproducibility**: Workflows are defined declaratively, ensuring consistent execution
2. **Resilience**: Failed tasks can be retried or resumed without rerunning the entire pipeline
3. **Observability**: Execution is tracked, providing visibility into the evaluation process
4. **Parallelism**: Independent tasks can be executed concurrently

The main workflows are defined in the `prefect_flows` directory, with each flow representing a major processing stage. Understanding these flows is critical for comprehending how BIGOS sequences its operations.

### Configuration System

The configuration system in BIGOS employs a hierarchical approach:

```
config/
  ├── common/            # Shared configuration across all runs
  ├── eval-run-specific/ # Configuration for specific evaluation scenarios  
  ├── tts-set-specific/  # Configuration for TTS data generation
  └── user-specific/     # User-specific settings (API keys, paths)
```

This separation creates a clear distinction between:

- **Structural configuration**: How the system is organized (common)
- **Task configuration**: What specific evaluation to perform (eval-run-specific)
- **Resource configuration**: Where to find necessary resources (user-specific)

When extending BIGOS, respect these boundaries to maintain configuration coherence.

## Extending the Framework

### Adding a New ASR System

Adding support for a new ASR system is perhaps the most common extension. Follow these steps:

1. **Create Implementation**: Implement a new class that inherits from `BaseASRSystem`
2. **Register System**: Add your implementation to the factory function in `__init__.py`
3. **Update Configuration**: Add relevant parameters to the configuration files
4. **Test Integration**: Verify your implementation works within the evaluation pipeline

Your implementation should handle all system-specific details while conforming to the common interface. This might include:

- Authentication mechanisms
- API rate limiting
- Error handling specific to the ASR service
- Format conversions between the service and BIGOS

Here's a skeleton implementation:

```python
from .base_asr_system import BaseASRSystem

class NewASRSystem(BaseASRSystem):
    def __init__(self, system, model, credentials, language_code="pl-PL"):
        super().__init__(system, model, language_code)
        # Initialize connection to the ASR service
        
    def generate_asr_hyp(self, speech_file):
        try:
            # System-specific implementation
            # 1. Prepare audio data
            # 2. Call ASR service API
            # 3. Process response
            # 4. Return transcription
            return transcription
        except ServiceSpecificException as e:
            # Handle service-specific errors
            print(f"Error from service: {e}")
            return ""
```

### Adding a New Metric

Extending the evaluation metrics requires:

1. Implementing the metric calculation in `eval_utils`
2. Updating the metric aggregation functions to include your new metric
3. Adding the metric to the visualization components

Metrics should conform to existing patterns:

- Accept references and hypotheses as inputs
- Return normalized values (typically as percentages)
- Handle edge cases (empty strings, mismatched counts, etc.)
- Provide clear documentation of the metric's meaning and interpretation

### Adding a New Dataset

New datasets must conform to the BIGOS format specification to integrate properly. The process involves:

1. Converting the dataset to BIGOS format
2. Creating a configuration file in `config/eval-run-specific/`
3. Testing dataset loading and processing
4. Verifying evaluation results

## Development Practices

### Code Style

BIGOS follows these coding practices:

- **PEP 8** compliant formatting
- **Type annotations** for function signatures
- **Docstrings** in a consistent format
- **Exception handling** with specific error types
- **Clear variable naming** that reflects purpose

Your contributions should maintain these standards for consistency and maintainability.

### Testing Strategy

The framework employs several testing approaches:

1. **Unit tests** for individual components
2. **Integration tests** for interaction between components
3. **End-to-end tests** for complete workflows
4. **Validation tests** for configuration and dataset integrity

When adding features or fixing bugs, ensure appropriate test coverage at all relevant levels.

### Error Handling Philosophy

BIGOS takes a pragmatic approach to error handling:

- **Recoverable errors** (e.g., temporary service unavailability) should be caught and handled
- **Configuration errors** should be detected early and reported clearly
- **Data inconsistencies** should be logged with sufficient context for diagnosis
- **Programming errors** (e.g., type mismatches) should propagate for immediate detection

This philosophy balances robustness against transparency, ensuring both reliability and debuggability.

## Advanced Topics

### Caching Strategy

The framework employs a sophisticated caching mechanism for ASR hypotheses, balancing performance against storage requirements:

- Hypotheses are cached based on audio file content
- Cache entries include metadata for traceability
- Cache invalidation occurs when forcing regeneration
- Cache is persisted to disk for reuse across runs

Understanding this caching behavior is important when diagnosing performance or storage issues.

### Normalization Techniques

Text normalization plays a crucial role in fair evaluation. BIGOS supports several normalization strategies:

- **Case normalization**: Converting all text to lowercase
- **Whitespace normalization**: Standardizing spacing between words
- **Punctuation removal**: Eliminating non-alphanumeric characters
- **Lexical normalization**: Standardizing word forms and spellings

Each strategy addresses specific aspects of textual variation, and combining them produces increasingly lenient evaluation criteria.

### Metadata Exploitation

The BIGOS format includes rich metadata about audio samples, speakers, and recording conditions. This metadata enables sophisticated analyses:

- Performance across demographic groups (age, gender)
- Accuracy variations by acoustic conditions
- Sensitivity to speech characteristics (rate, accent, etc.)

When extending the framework, consider how to leverage this metadata for deeper insights.

## Contribution Workflow

1. **Fork** the repository to your account
2. **Clone** your fork locally
3. **Create** a feature branch
4. **Implement** your changes with appropriate tests
5. **Document** your changes thoroughly
6. **Submit** a pull request with a clear description
7. **Address** review feedback
8. **Celebrate** your contribution!

Remember that good contributions aren't just about code—improvements to documentation, configuration, or examples are equally valuable.

## Conclusion
BIGOS represents both a technical framework and a scientific methodology for ASR evaluation. As you contribute to its evolution, maintain this dual perspective—every technical choice has implications for the validity and reproducibility of evaluation results.

Your contributions, whether large or small, advance our collective understanding of ASR performance and help drive progress in speech recognition technology. Welcome to the BIGOS development community!

> **Final Thought**: The most profound contributions often come not from adding features, but from enhancing clarity, robustness, or accessibility. A simpler interface, more comprehensive documentation, or more intuitive visualization can impact more users than the cleverest algorithm.
