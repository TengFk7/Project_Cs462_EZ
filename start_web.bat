@echo off
echo Checking environment...

:: ถ้ายังไม่มี .venv ให้สร้างและลง Library อัตโนมัติ
if not exist ".venv" (
    echo Creating Virtual Environment...
    python -m venv .venv
    call .venv\Scripts\activate
    echo Installing Libraries... This may take a few minutes.
    pip install -r requirements.txt
) else (
    call .venv\Scripts\activate
)

echo Starting Thai Digit Prediction System...
python app.py
pause