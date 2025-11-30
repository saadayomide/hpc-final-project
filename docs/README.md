# Documentation

This directory contains the main project documentation and deliverables.

## Deliverables

### 1. Research Paper (paper.md)

**4-6 page paper** covering:
- Introduction and motivation
- Background (DCRNN, distributed training)
- System design and implementation
- Experimental setup and methodology
- Results (scaling, profiling, accuracy)
- Discussion and limitations
- Conclusion and future work

**Format:** Markdown (can be converted to PDF using pandoc)

```bash
# Convert to PDF (requires pandoc and LaTeX)
pandoc paper.md -o paper.pdf --pdf-engine=pdflatex
```

### 2. EuroHPC Proposal (eurohpc_proposal.md)

**6-8 page EuroHPC Development Access proposal** including:
- Abstract and objectives
- State of the art
- Current code and TRL
- Target EuroHPC machine (LUMI-G/Leonardo)
- Work plan with milestones
- Resource justification (512 GPU-node-hours)
- Data management and FAIR compliance
- Expected impact

### 3. Pitch Presentation (pitch.md)

**5-slide pitch** covering:
1. Problem & Impact
2. Approach & Prototype  
3. Scaling & Profiling Results
4. EuroHPC Target & Resource Ask
5. Risks, Milestones & Support

**Format:** Markdown outline (can be converted to slides)

```bash
# Convert to slides using Marp
npx @marp-team/marp-cli pitch.md -o pitch.pdf

# Or use reveal.js, Google Slides, etc.
```

## Additional Documentation

### In Project Root

- **README.md** - Project overview and quick start
- **reproduce.md** - Step-by-step reproduction instructions
- **SYSTEM.md** - Cluster configuration and requirements
- **FILESYSTEM.md** - HPC filesystem guidelines

### In Other Directories

- **data/README.md** - Dataset documentation
- **results/README.md** - Results format and interpretation
- **results/profiling/README.md** - Profiling methodology

## Document Conversion

### To PDF

```bash
# Install pandoc
sudo apt install pandoc texlive-latex-recommended

# Convert documents
pandoc docs/paper.md -o docs/paper.pdf \
    --pdf-engine=pdflatex \
    -V geometry:margin=1in \
    -V fontsize=11pt

pandoc docs/eurohpc_proposal.md -o docs/eurohpc_proposal.pdf \
    --pdf-engine=pdflatex \
    -V geometry:margin=1in \
    -V fontsize=11pt
```

### To Slides

```bash
# Using Marp
npm install -g @marp-team/marp-cli
marp docs/pitch.md -o docs/pitch.pdf

# Using reveal-md
npm install -g reveal-md
reveal-md docs/pitch.md --print docs/pitch.pdf
```

## Templates Used

The documents follow these templates:
- **Paper**: Academic conference format
- **Proposal**: EuroHPC Development Access template
- **Pitch**: 5-minute technical presentation format

## Word/Page Counts

| Document | Target | Approximate |
|----------|--------|-------------|
| Paper | 4-6 pages | ~3000 words |
| Proposal | 6-8 pages | ~4000 words |
| Pitch | 5 slides | ~500 words |
