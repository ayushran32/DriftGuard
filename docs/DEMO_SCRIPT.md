# Partial demo (about 6 minutes)
Start: `streamlit run app.py` (or `python demo_cli.py --scenario <name>` if the browser is a problem).
1. control: no drift -> no alerts (false alarms are controlled).
2. ambient_warming: sensors drift but F1 barely moves -> flagged, not blindly retrained (harmless drift).
3. process_temp_offset: triage says SENSOR_FAULT, input repaired, no retraining; compare F1 curves of the three strategies.
4. heavy_load: real regime shift -> challenger trained and validated before deployment (show the incident log).
5. new_tool_material: inputs look normal, labels reveal the model is wrong -> CONCEPT_DRIFT (shows why delayed labels matter).
6. Show `results_summary.csv` (3 seeds, simulated drift) and state the limits.
Run every scenario once before the demo and keep screenshots as a backup.
