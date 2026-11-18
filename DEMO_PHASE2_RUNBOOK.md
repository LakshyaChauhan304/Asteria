# Asteria Demo Phase 2

Run from the isolated demo worktree with the isolated environment:

```powershell
Set-Location C:\tmp\Asteria-demo-worktree
& C:\Users\phila\AppData\Local\Asteria\demo-venv\Scripts\python.exe student2_demo_response_smoke.py
```

This runs actual Phase-1 model inference over real UNSW-NB15 rows, then only
in-memory graph simulation, explicit approval, simulated control state change,
and independent verification. It does not scan, exploit, execute tools, or
modify live infrastructure.
