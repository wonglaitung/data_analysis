@echo off

set PATH="C:\Users\User\Documents\My Tools\Python312";"C:\Users\User\Documents\My Tools\Python312\Scripts";%PATH%

set "current_dir=%~dp0"
echo 当前脚本所在路径：%current_dir%
cd %current_dir%

pyinstaller --onefile --icon=images\ai.ico --exclude-module=torch --clean convert_train_data.py
pyinstaller --onefile --icon=images\ai.ico --exclude-module=torch --clean add_train_label.py
pyinstaller --onefile --icon=images\ai.ico --exclude-module=torch --clean train_model.py
pyinstaller --onefile --icon=images\ai.ico --exclude-module=torch --clean convert_predict_data.py
pyinstaller --onefile --icon=images\ai.ico --exclude-module=torch --clean predict.py
pyinstaller --onefile --icon=images\ai.ico --clean train_model_dl.py
pyinstaller --onefile --icon=images\ai.ico --exclude-module=torch --clean check_model_fairness.py
pyinstaller --onefile --icon=images\ai.ico --exclude-module=torch --clean gui_app.py

del .\*.exe
del .\*.spec
copy dist\*.exe .


rmdir /s /q ".\build"
rmdir /s /q ".\dist"
rmdir /s /q ".\__pycache__"

rmdir /s /q ".\output"
mkdir .\output

pause
