@echo off

echo "Creating environment..."
python -m venv asr-env

echo "Activating Virtual Environment..."
call asr-env\Script\activate

echo "Installing Dependencies..."
python -m pip install --upgrade pip
pip install -r requirements.txt

echo "Done"