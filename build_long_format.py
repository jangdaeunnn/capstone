import os
import pandas as pd


def build_and_save_long_df(base_root: str):

    print(f"\n============================")
    print(f"📂 build_long_df 실행: {base_root}")
    print(f"============================\n")

    # 1) 숫자로 된 subfolder 탐색 (예: 1,2,3…)
    sample_ids = sorted(
        int(d) for d in os.listdir(base_root)
        if d.isdigit() and os.path.isdir(os.path.join(base_root, d))
    )

    if not sample_ids:
        raise FileNotFoundError(f"⚠ 샘플 폴더가 없습니다: {base_root}")

    print(f"🔍 발견된 샘플 폴더: {sample_ids}")

    long_rows = []
    col_ref = None

    # 2) 각 샘플별 CSV → long rows 변환
    for sample_id in sample_ids:
        sample_folder = os.path.join(base_root, str(sample_id))

        # con100_ 폴더 찾기
        subdirs = [
            d for d in os.listdir(sample_folder)
            if os.path.isdir(os.path.join(sample_folder, d)) and d.startswith("con100_")
        ]
        if not subdirs:
            print(f"[WARN] con100_* 폴더 없음 → sample {sample_id} 스킵")
            continue

        con_dir = os.path.join(sample_folder, subdirs[0])

        csv_path = os.path.join(con_dir, "ALL_sensors_time_matrix.csv")
        if not os.path.exists(csv_path):
            print(f"[WARN] ALL_sensors_time_matrix.csv 없음 → sample {sample_id} 스킵")
            continue

        # CSV 로드
        df = pd.read_csv(csv_path, index_col=0)
        print(f"[LOAD] sample {sample_id}: {csv_path}, shape={df.shape}")

        # 센서 이름 기준 설정 (첫 sample 기준)
        if col_ref is None:
            col_ref = list(df.columns)
            print(f"📌 기준 센서 이름 세트(col_ref) 저장 ({len(col_ref)}개)")
        else:
            if list(df.columns) != col_ref:
                print(f"[WARN] 센서 컬럼 순서가 기준과 다름 → 정렬 강제 적용")
                df = df.reindex(columns=col_ref)

        # long format 생성
        for t in range(df.shape[0]):
            for s_idx, sensor in enumerate(col_ref):
                long_rows.append({
                    "sample_id": sample_id,
                    "time_index": t,
                    "sensor": sensor,
                    "intensity": df.iat[t, s_idx],
                })

    # 3) long dataframe 생성 + 저장
    long_df = pd.DataFrame(long_rows)

    out_csv = os.path.join(base_root, "long_df.csv")
    long_df.to_csv(out_csv, index=False)

    print(f"\n🎉 long_df.csv 생성 완료!")
    print(f"📁 저장 위치: {out_csv}")
    print(f"📏 최종 shape: {long_df.shape} (rows = sample × T × sensor)")
    print(f"============================\n")

    return long_df


# 단독 실행 테스트용
if __name__ == "__main__":
    base_root = "/home/gracejang42/CP/final_data/voc1"
    build_and_save_long_df(base_root)
