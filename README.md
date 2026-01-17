<p align="center">
  <a href="https://ai.tamer.work/vesper.ai/">
    <img src="https://github.com/TamerDotWork/vesper/blob/main/cover.png" alt="Vesper Banner" width="100%">
  </a>
</p>

<div align="center">

# Vesper

### Agentic data intelligence using LangChain & Pandas for dataset cleaning, governance, and quality analysis

<br/>


</div>

<br/>

<div align="center">
  <img src="https://github.com/TamerDotWork/vesper/blob/main/screenshot.png" alt="Vesper Demo" width="900" style="border-radius: 16px;">
</div>

<br>

> [!TIP]
> **New!** Vesper transforms raw messy datasets into governed, analysis-ready data using only natural language instructions.

---

## Vesper Overview

Vesper is an autonomous agentic data analyst built with:

- LangChain agent orchestration  
- Pandas DataFrame Agent  
- Python execution tool  
- Dataset governance layer  
- Quality measurement engine  

It performs real operations on real data:  
load → clean → analyze → validate → score → explain

**Key Capabilities**

- True Pandas execution  
- Automated cleaning workflows  
- Measurable quality scoring  
- Explainable transformations  
- Reproducible lineage

---

## Use Cases

- Automated data cleaning  
- Dataset quality governance  
- Exploratory analysis  
- Pre-ML preparation  
- BI readiness

---

## Quick Start

**Prerequisites**

- Python 3.10+  
- LLM provider key

### Install

Use one of the following commands:

```bash
pipx install vesper-agent
```

```bash
pip install vesper-agent
```

### First Run

```bash
export VESPER_LLM="openai/gpt-4o"
export LLM_API_KEY="your-api-key"

vesper --file dataset.csv --goal "clean dataset and measure overall quality"
```

> [!NOTE]
> Results are stored in vesper_runs with full logs.

---

## Run Vesper in Cloud

Use the hosted platform at **app.vesper.ai** with:

- Quality scorecard  
- Explainable actions  
- Clean dataset download  
- Team dashboards  
- API automation  

Analyze your dataset at the platform website.

---

## Features

### Agentic Data Tools
- Pandas runtime execution  
- Safe Python sandbox  
- Profiling engine  
- Rule validation  
- Audit logs

### Quality Detection
- Missing values  
- Duplicates  
- Type conflicts  
- Outliers  
- Schema drift  
- Inconsistent categories

### Multi-Agent Flow
- Planner → strategy  
- Pandas → execution  
- Validator → quality  
- Reporter → insights

---

## Usage Examples

### Basic

```bash
vesper --file sales.csv
```

### With Instruction

```bash
vesper --file customers.xlsx --instruction "normalize columns and compute quality score"
```

### Multi File

```bash
vesper -f jan.csv -f feb.csv --instruction "merge and validate schema"
```

### Headless

```bash
vesper -n --file data.csv
```

---

## Configuration

```bash
export VESPER_LLM="openai/gpt-4o"
export LLM_API_KEY="your-api-key"
export LLM_API_BASE="local-endpoint"
export VESPER_REASONING="high"
```

Recommended models:

- OpenAI GPT-4o  
- Claude Sonnet  
- Gemini Pro  
- Local Llama 3

---

## Documentation

See the documentation website for full guides.

---

## Contributing

Contributions are welcome through pull requests.

---

## Join Our Community

Community links are available above.

---

## Support

Give us a star on GitHub if Vesper helps you.

---

## Acknowledgements

Built with:

- LangChain  
- Pandas  
- OpenAI  
- LiteLLM  
- DuckDB  

> [!WARNING]
> Always review AI-applied transformations before production use.

</div>
