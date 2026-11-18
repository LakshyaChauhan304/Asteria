$venv = 'C:\Users\phila\AppData\Local\Asteria\demo-venv\Scripts\python.exe'
$required = @('models\xgboost_asteria_pipeline.pkl','models\model_metadata.json','data\UNSW_NB15_training-set.csv','data\UNSW_NB15_testing-set.csv')
if (-not (Test-Path -LiteralPath $venv)) { throw "Demo venv is missing: $venv" }
foreach ($path in $required) { if (-not (Test-Path -LiteralPath $path)) { throw "Required demo file is missing: $path" } }
& $venv -c "import streamlit, xgboost, sklearn"; if ($LASTEXITCODE -ne 0) { throw 'Install demo_requirements.txt in the isolated demo venv.' }
Write-Host 'Asteria SOC: http://localhost:8501'
& $venv -m streamlit run demo_app.py
