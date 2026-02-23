# sign-language-korean

## 가상 환경 생성(Windows 기준)
```
python -m venv venv
.venv/Scripts/activate
pip install -r requirements-win.txt

```

## 가상 환경 생성(MacOS 기준)
```
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r documents/requirements-mac.txt
```

## 가상 환경 생성(Linux 기준)
```
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements-linux.txt
```

드라이브 참고해서 root 경로에 /data 만들어서 /train과 /val을 만들고 그대로 넣은 뒤 학습을 진행할 수 있음

## 학습 진행
```
python3 -m train.gloss_transformer_train
```