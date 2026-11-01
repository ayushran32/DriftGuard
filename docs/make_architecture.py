import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(13, 7.4)); ax.set_xlim(0, 13.2); ax.set_ylim(0.1, 7.4); ax.axis("off")
def box(x, y, w, h, title, sub="", fc="#DCE6F5", ec="#1F3864"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.12", fc=fc, ec=ec, lw=1.4))
    ax.text(x + w/2, y + h - 0.28, title, ha="center", va="top", fontsize=9.5, fontweight="bold", color=ec)
    if sub: ax.text(x + w/2, y + h - 0.62, sub, ha="center", va="top", fontsize=7.6, color="#222", linespacing=1.35)
def arrow(a, b, c="#1F3864", style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=14, lw=1.4, color=c, linestyle=ls))

ax.text(6.5, 7.15, "DriftGuard: System Architecture", ha="center", fontsize=14, fontweight="bold", color="#1F3864")
# row 1: data
box(0.3, 5.3, 2.6, 1.5, "1. Data Module", "AI4I 2020 dataset (10,000 rows)\nsplit: Reference | Calibration | Pool\n+ drift / fault injection", fc="#E8F0E0", ec="#3B6B2A")
box(3.4, 5.3, 2.6, 1.5, "2. Baseline Model", "Random Forest\n(5 sensors + product type)\nfailure / no-failure", fc="#E8F0E0", ec="#3B6B2A")
box(6.5, 5.3, 2.8, 1.5, "Live Sensor Window", "500 readings per window\n(air temp, process temp, rpm,\ntorque, tool wear)", fc="#F5F5F5", ec="#555")
arrow((2.9, 6.05), (3.4, 6.05)); arrow((6.0, 6.05), (6.5, 6.05), "#555")
# row 2: detection + triage
box(0.3, 3.1, 4.2, 1.7, "3. Detection Module", "KS test + PSI (per sensor)\nk-means occupancy / error (multivariate)\nthresholds calibrated by bootstrap on\nno-drift windows of the same size", fc="#FFF1D6", ec="#9A6A00")
box(5.0, 3.1, 4.3, 1.7, "4. Triage Module (cause diagnosis)", "variance signatures: stuck / noisy sensor\ntemp-pair consistency: offset sensor\nonly tool wear moves: fleet ageing\nlabels fail, inputs normal: concept drift", fc="#FFE0DC", ec="#A12A1F")
arrow((7.9, 5.3), (7.9, 4.8), "#555"); arrow((6.0, 5.6), (2.4, 4.8), "#555")
arrow((4.5, 3.95), (5.0, 3.95), "#9A6A00")
# row 3: actions
box(0.3, 0.9, 3.9, 1.6, "5a. Online Sensor Repair", "sensor fault  ->  subtract bias / regress\nfrom healthy correlated sensors\nNO retraining", fc="#E0F3F3", ec="#1B6F72")
box(4.6, 0.9, 4.2, 1.6, "5b. Guarded Retraining", "regime shift / concept drift  ->  wait for\ndelayed labels, check real harm,\ntrain challenger, bootstrap-compare,\ndeploy only if better", fc="#E0F3F3", ec="#1B6F72")
arrow((6.2, 3.1), (2.5, 2.5), "#A12A1F"); arrow((7.0, 3.1), (6.7, 2.5), "#A12A1F")
ax.text(3.7, 2.95, "sensor fault", fontsize=7.5, color="#A12A1F", style="italic")
ax.text(7.05, 2.78, "regime / concept", fontsize=7.5, color="#A12A1F", style="italic")
# outputs
box(9.9, 3.1, 3.0, 1.7, "6. Prediction", "failure prediction from\nrepaired input + deployed model\n(champion or validated challenger)", fc="#EDE4F5", ec="#5B3A8C")
box(9.9, 0.9, 3.0, 1.6, "7. Dashboard + Incident Log", "drift score vs threshold, per-sensor\ncontribution, triage decision,\nactions taken", fc="#EDE4F5", ec="#5B3A8C")
arrow((9.3, 3.95), (9.9, 3.95), "#A12A1F")
ax.text(9.6, 4.12, "diagnosis", ha="center", fontsize=7, color="#A12A1F", style="italic")
# repaired input (5a) and validated model (5b) feed prediction via a bus below the boxes
ax.plot([2.25, 2.25, 9.6, 9.6], [0.9, 0.5, 0.5, 3.45], color="#1B6F72", lw=1.4, ls="--")
ax.plot([6.7, 6.7], [0.9, 0.5], color="#1B6F72", lw=1.4, ls="--")
arrow((9.6, 3.45), (9.9, 3.45), "#1B6F72")
ax.text(5.0, 0.3, "repaired input (5a)  /  validated challenger model (5b)  ->  prediction", fontsize=7.5, color="#1B6F72", style="italic")
arrow((11.4, 3.1), (11.4, 2.5), "#5B3A8C")
ax.text(11.4, 5.55, "delayed failure labels\n(arrive ~2 windows late)", ha="center", fontsize=7.5, color="#555", style="italic")
ax.add_patch(FancyArrowPatch((11.4, 5.1), (11.4, 4.82), arrowstyle="-|>", mutation_scale=12, lw=1.2, color="#555"))
plt.savefig("docs/DriftGuard_architecture.png", dpi=220, bbox_inches="tight", facecolor="white")
