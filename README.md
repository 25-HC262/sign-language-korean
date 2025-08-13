# sign-language-korean

## 가상 환경 생성(Windows 기준)
```
python -m venv [가상환경 이름]
.[가상환경 이름]/Scripts/activate

pip install -r requirements-win.txt

```

## 가상 환경 생성(MacOS 기준)
```
python3.11 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip cache purge
pip install -r requirements-mac.txt
pip install tensorflow-metal==1.1.0
```

드라이브 참고해서 root 경로에 /data 만들어서 /train과 /val을 만들고 그대로 넣은 뒤 학습을 진행할 수 있음