ROBUST BAYESIAN POWER-MOMENT COMPUTATIONAL PACKAGE

Files:
1. rbpm_colab_analysis.py
   - Upload to Google Colab and run as a single Python script/cell.
   - Default FAST_MODE=True gives a relatively quick diagnostic run.
   - Set FAST_MODE=False before producing final manuscript numbers.
   - All tables/plots/raw results are saved under rbpm_results/.
   - A ZIP bundle is created automatically and, in Colab, downloaded automatically.

2. computational_study_section.tex
   - Replace the manuscript section beginning
       \section{A reproducible computational study design}
     with this file.
   - It contains only design/methodology statements; no uncomputed results are claimed.

3. wine_dataset_reference.bib
   - Merge this verified UCI dataset entry into bayesian_power_moment_refs.bib.
   - DOI: 10.24432/C5PC7J.

Important modeling note:
The default computational implementation uses a fixed broad Student-t_3 working
contaminant on robust-standardized coordinates (scale 8). This is the fixed-xi/
degenerate-prior special case of the manuscript's general contamination model.
It is chosen for speed, identifiability, and direct alignment with the theorem
requiring a heavier-tailed contaminant independent of the clean scale.
Sensitivity to this working choice should be reported in the final paper.
