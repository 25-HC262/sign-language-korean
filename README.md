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
```bash
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements-linux.txt
```

드라이브 참고해서 root 경로에 /data 만들어서 /train과 /val을 만들고 그대로 넣은 뒤 학습을 진행할 수 있음

## 학습 진행
- download/upload options: `-s`(storage), `-u`(upload)
- model options: `--gm`(gloss_model), `--gmt`(gloss_model_type), `-u`(umap)
- training options: `--bs`(batch_size), `--lr`(learning_rate), `--epochs`(epochs), `--wd`(weight_decay), `--msl`(max_sequence_len)
```
python -m train.gloss_transformer_train -s L -u G --lr 0.0001178136471332758 --bs 32 --epochs 158 --wd 0.23042807878441396 --msl 281
```
