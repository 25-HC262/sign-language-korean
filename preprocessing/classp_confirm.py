#!/usr/bin/env python3
import sys, os, io, json, gzip, bz2, lzma, pickle, types
from pprint import pprint

def try_joblib_load(path):
    try:
        import joblib
        return joblib.load(path), "joblib.load"
    except ImportError:
        return None, "joblib (미설치)"
    except Exception:
        return None, "joblib.load 실패"

def try_numpy_load(path):
    try:
        import numpy as np
        obj = np.load(path, allow_pickle=True)
        return obj, "numpy.load"
    except ImportError:
        return None, "numpy (미설치)"
    except Exception:
        return None, "numpy.load 실패"

def try_json_load_bytes(b):
    for enc in ("utf-8", "utf-8-sig", "cp949", "latin-1"):
        try:
            s = b.decode(enc)
            s_strip = s.strip()
            if (s_strip.startswith("{") and s_strip.endswith("}")) or \
                    (s_strip.startswith("[") and s_strip.endswith("]")):
                return json.loads(s_strip), f"json.loads (encoding={enc})"
        except Exception:
            continue
    return None, "json 실패"

def try_pickle_stream(fobj, label):
    try:
        return pickle.load(fobj), label
    except Exception:
        return None, f"{label} 실패"

def summarize(obj, max_items=10):
    try:
        import numpy as np
    except Exception:
        np = None
    try:
        import pandas as pd
    except Exception:
        pd = None

    def short(x, n=120):
        s = repr(x)
        return s if len(s) <= n else s[:n] + "...(truncated)"

    print("\n=== 객체 요약 ===")
    print(f"type: {type(obj)}")
    if isinstance(obj, dict):
        print(f"dict 길이: {len(obj)} / 키 샘플:")
        keys = list(obj.keys())
        for k in keys[:max_items]:
            print(f"  - {short(k)} : {short(obj[k])}")
        if len(keys) > max_items:
            print(f"  ... (+{len(keys)-max_items} more keys)")
    elif isinstance(obj, (list, tuple, set, frozenset)):
        seq = list(obj) if not isinstance(obj, tuple) else list(obj)
        print(f"{type(obj).__name__} 길이: {len(seq)} / 앞 부분 샘플:")
        for i, v in enumerate(seq[:max_items]):
            print(f"  [{i}] {short(v)}")
        if len(seq) > max_items:
            print(f"  ... (+{len(seq)-max_items} more items)")
    elif np is not None and isinstance(obj, np.ndarray):
        print(f"NumPy 배열 shape={obj.shape}, dtype={obj.dtype}")
        if obj.size > 0:
            print("샘플:", short(obj.flat[0]))
    elif pd is not None and isinstance(obj, pd.DataFrame):
        print(f"pandas DataFrame shape={obj.shape}")
        print("컬럼:", list(obj.columns)[:max_items], "...")
        try:
            print(obj.head(5).to_string())
        except Exception:
            pass
    elif isinstance(obj, types.ModuleType):
        print("모듈 객체입니다. dir() 샘플:")
        names = dir(obj)[:max_items]
        for n in names:
            print(" ", n)
    else:
        print(short(obj))

def save_as_json(obj, json_path):
    """객체를 JSON 파일로 저장"""
    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2, default=str)
        print(f"JSON 파일 저장 완료: {json_path}")
    except Exception as e:
        print(f"JSON 저장 실패: {e}")

def load_p_file(path):
    with open(path, "rb") as f:
        raw = f.read()

    obj, how = try_json_load_bytes(raw)
    if obj is not None:
        return obj, how

    try:
        with gzip.GzipFile(fileobj=io.BytesIO(raw)) as gf:
            obj, how = try_pickle_stream(gf, "gzip+pickle")
            if obj is not None:
                return obj, how
    except Exception:
        pass

    try:
        obj, how = try_pickle_stream(bz2.BZ2File(io.BytesIO(raw)), "bz2+pickle")
        if obj is not None:
            return obj, how
    except Exception:
        pass

    try:
        obj, how = try_pickle_stream(lzma.LZMAFile(io.BytesIO(raw)), "lzma+pickle")
        if obj is not None:
            return obj, how
    except Exception:
        pass

    try:
        obj = pickle.loads(raw)
        return obj, "pickle.loads"
    except Exception:
        pass

    obj, how = try_joblib_load(path)
    if obj is not None:
        return obj, how

    obj, how = try_numpy_load(path)
    if obj is not None:
        return obj, how

    raise RuntimeError("열기 실패: pickle/json/joblib/numpy 방식 모두 실패.")

def main():
    if len(sys.argv) < 2:
        print("사용법: python view_p_file.py <파일경로> [--to-json json경로]")
        sys.exit(1)

    path = sys.argv[1]
    to_json_path = None
    if len(sys.argv) == 4 and sys.argv[2] == "--to-json":
        to_json_path = sys.argv[3]

    if not os.path.exists(path):
        print(f"파일 없음: {path}")
        sys.exit(1)

    print(f"파일: {path}")
    print("경고: pickle은 **신뢰할 수 없는 파일에 절대 사용하면 안 됩니다.**\n"
          "(임의 코드 실행 위험)")

    try:
        obj, how = load_p_file(path)
        print(f"\n열기 성공 — 방법: {how}")
        summarize(obj)

        if to_json_path:
            save_as_json(obj, to_json_path)

    except Exception as e:
        print("\n열기 실패:")
        print(e)

if __name__ == "__main__":
    main()
