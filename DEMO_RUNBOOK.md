# Asteria SOC Demo

Docker Desktop is installed but its daemon was unavailable during validation, so the supported local route is:

```powershell
Set-Location C:\tmp\Asteria-demo-worktree
.\run_demo.ps1
```

Open `http://localhost:8501`. The app starts with no model activity. Use the workflow buttons in order; every transition calls the per-session `DemoOrchestrator`. Graphs, particles, approvals, remediation and verification are visualizations of returned backend state only.
