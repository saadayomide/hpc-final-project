# Submission Checklist

## Code & Repository
- [ ] `src/` contains all source code
- [ ] `env/project.def` Apptainer recipe is complete
- [ ] `slurm/` contains working job scripts
- [ ] `run.sh` wrapper script works
- [ ] Code runs on ≥2 nodes under SLURM

## Reproducibility
- [ ] `reproduce.md` has exact commands
- [ ] Random seeds are fixed (42)
- [ ] Versions documented in `SYSTEM.md`
- [ ] Container recipe (`env/project.def`) is complete

## Performance Evidence
- [ ] Strong scaling results (1, 2, 4+ nodes)
- [ ] Weak scaling results
- [ ] Plots in `results/scaling/` (PNG + SVG)
- [ ] `sacct` logs saved for jobs
- [ ] Profiling data collected

## Documentation
- [ ] Paper (4-6 pages) in `docs/paper.pdf`
- [ ] EuroHPC proposal (6-8 pages) in `docs/eurohpc_proposal.pdf`
- [ ] 5-slide pitch in `docs/` (PDF or slides)
- [ ] `SYSTEM.md` with node types, modules, versions

## Final Steps
- [ ] All changes committed
- [ ] Release tag created: `git tag -a v1.0-course-teamname -m "Final submission"`
- [ ] Tag pushed: `git push origin v1.0-course-teamname`
- [ ] Verify tag on GitHub/GitLab

## Optional but Recommended
- [ ] Container `.sif` file built and tested
- [ ] Sample data in `data/` for quick testing
- [ ] Screenshots of successful runs
