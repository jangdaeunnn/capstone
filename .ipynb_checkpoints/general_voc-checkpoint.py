import os
import sys
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tifffile import imread, imwrite

# labeling_sensor 모듈 import 경로
sys.path.append("/home/gracejang42/CP/251129/VOC3")
from labeling_sensor import extract_time_sensor_dataframe, plot_frame_with_crops

# -----------------------------
# 기본 상수 (기본값은 voc1)
# -----------------------------
BASE_ROOT = "/home/gracejang42/CP/final_data/voc1"
COORD_ROOT = "/home/gracejang42/CP/final_data/coord"
STEP = 9  # T축 주기 간격

POS_TO_WELL = {
    1: "A1", 2: "A2", 3: "A3",
    4: "B1", 5: "B2", 6: "B3",
    7: "C1", 8: "C2", 9: "C3",
}


# -----------------------------
# 유틸: 폴더 안 con100_* 디렉토리 찾기
# -----------------------------
def get_con_dir(folder_num: int, base_root: str = BASE_ROOT) -> str:
    folder_path = os.path.join(base_root, str(folder_num))
    subdirs = [
        d for d in os.listdir(folder_path)
        if os.path.isdir(os.path.join(folder_path, d)) and d.startswith("con100_")
    ]
    if not subdirs:
        raise FileNotFoundError(f"con100_* 폴더가 없음: {folder_path}")

    con_dir = os.path.join(folder_path, subdirs[0])
    print(f"[INFO] con_dir: {con_dir}")
    return con_dir


# -----------------------------
# 1) NDTiffStack → pos1~pos9_TXX_TYX.tif 생성
# -----------------------------
def generate_pos_tyx_stacks(folder_num: int,
                            base_root: str = BASE_ROOT,
                            step: int = STEP):
    con_dir = get_con_dir(folder_num, base_root=base_root)

    # NDTiffStack 또는 첫 번째 tif 선택
    tifs = [f for f in os.listdir(con_dir) if f.endswith(".tif")]
    if not tifs:
        raise FileNotFoundError(f"{con_dir} 안에 tif 파일 없음")

    ndtiff_candidates = [f for f in tifs if "NDTiffStack" in f]
    tif_name = ndtiff_candidates[0] if ndtiff_candidates else tifs[0]
    tif_path = os.path.join(con_dir, tif_name)

    print(f"[INFO] 원본 TIF: {tif_path}")
    stack = imread(tif_path)  # 예상 shape: (T, C, H, W)
    print("[INFO] 원본 shape:", stack.shape)

    if stack.ndim != 4:
        raise ValueError(f"예상 shape (T,C,H,W)가 아님. ndim={stack.ndim}")

    T, C, H, W = stack.shape

    use_frames = (T // step) * step
    if use_frames == 0:
        raise ValueError(f"사용 가능한 프레임 없음 (T={T})")

    valid_stack = stack[:use_frames]
    base_indices = np.arange(0, use_frames, step)

    print(f"[INFO] 사용 프레임 수: {use_frames} (cycles={len(base_indices)})")
    print(f"[INFO] 각 pos당 선택될 T 인덱스 수: {len(base_indices)}")

    for pos_idx in range(C):  # 보통 C=9
        indices = base_indices + pos_idx
        indices = indices[indices < use_frames]

        sub_stack = valid_stack[indices, pos_idx, :, :]  # (N, H, W)
        save_name = f"pos{pos_idx+1}_T{sub_stack.shape[0]}_TYX.tif"
        save_path = os.path.join(con_dir, save_name)

        imwrite(
            save_path,
            sub_stack,
            imagej=True,
            metadata={"axes": "TYX"},
        )
        print(f"[SAVE] pos{pos_idx+1}: {save_path}  shape={sub_stack.shape}")

    print(f"\n🎉 Folder {folder_num}: pos1~pos{C} T-stack 생성 완료! (base_root={base_root})\n")


# -----------------------------
# 2) pos*_TXX_TYX.tif → pos*_time_sensor.csv 생성
# -----------------------------
def generate_time_sensor_csvs(folder_num: int,
                              base_root: str = BASE_ROOT,
                              coord_root: str = COORD_ROOT):
    con_dir = get_con_dir(folder_num, base_root=base_root)

    for pos_idx in range(1, 10):
        # posX_T*_TYX.tif 찾기
        tif_candidates = [
            f for f in os.listdir(con_dir)
            if f.startswith(f"pos{pos_idx}_T") and f.endswith("_TYX.tif")
        ]
        if not tif_candidates:
            print(f"[WARN] pos{pos_idx} TIF 없음, 스킵.")
            continue

        pos_tif = os.path.join(con_dir, tif_candidates[0])
        well = POS_TO_WELL[pos_idx]
        crop_dir = os.path.join(coord_root, well)

        if not os.path.isdir(crop_dir):
            print(f"[WARN] crop_dir 없음, 스킵: {crop_dir}")
            continue

        print(f"\n[INFO] Folder {folder_num} | pos{pos_idx} (well={well})")
        print(f"  TIF : {pos_tif}")
        print(f"  CROP: {crop_dir}")

        # labeling_sensor가 이해할 수 있는 dummy 파일명 생성
        dummy_name = f"VOC_folder{folder_num}_pos{pos_idx}_MMStack_{well}-Site_0.ome.tif"
        dummy_path = os.path.join(con_dir, dummy_name)

        if not os.path.exists(dummy_path):
            shutil.copy(pos_tif, dummy_path)
            print(f"[COPY] {pos_tif} → {dummy_path}")
        else:
            print(f"[INFO] dummy TIF 이미 존재: {dummy_path}")

        # 시간 x 센서 DataFrame 계산
        df = extract_time_sensor_dataframe(dummy_path, crop_dir)

        out_csv = os.path.join(con_dir, f"pos{pos_idx}_time_sensor.csv")
        df.to_csv(out_csv)
        print(f"[SAVE] {out_csv} (shape={df.shape})")

    print(f"\n🎉 Folder {folder_num}: pos1~9 time_sensor CSV 생성 완료! (base_root={base_root})\n")


# -----------------------------
# 3) crop boundary 오버레이 시각화
# -----------------------------
def plot_crops_for_folder(folder_num: int,
                          frame_index: int = 0,
                          base_root: str = BASE_ROOT,
                          coord_root: str = COORD_ROOT):
    con_dir = get_con_dir(folder_num, base_root=base_root)

    for pos_idx in range(1, 10):
        tif_candidates = [
            f for f in os.listdir(con_dir)
            if f.startswith(f"pos{pos_idx}_T") and f.endswith("_TYX.tif")
        ]
        if not tif_candidates:
            print(f"[WARN] pos{pos_idx} TIF 없음, 스킵.")
            continue

        pos_tif = os.path.join(con_dir, tif_candidates[0])
        well = POS_TO_WELL[pos_idx]
        crop_dir = os.path.join(coord_root, well)

        if not os.path.isdir(crop_dir):
            print(f"[WARN] crop_dir 없음, 스킵: {crop_dir}")
            continue

        print(f"\n=== 📌 Folder {folder_num} | pos{pos_idx} (well={well}) | frame={frame_index} ===")
        print(f"  TIF : {pos_tif}")
        print(f"  CROP: {crop_dir}")

        plot_frame_with_crops(pos_tif, crop_dir, frame_index=frame_index)


# -----------------------------
# 4) pos1~9 모든 센서 time-series 한 플롯에
# -----------------------------
def plot_all_sensors(folder_num: int,
                     base_root: str = BASE_ROOT):
    con_dir = get_con_dir(folder_num, base_root=base_root)
    print(f"\n📁 Plotting data from folder: {con_dir}\n")

    plt.figure(figsize=(14, 7))

    for pos_idx in range(1, 10):
        csv_path = os.path.join(con_dir, f"pos{pos_idx}_time_sensor.csv")
        if not os.path.exists(csv_path):
            print(f"[WARN] CSV 없음 (스킵): {csv_path}")
            continue

        df = pd.read_csv(csv_path, index_col=0)

        for sensor in df.columns:
            plt.plot(df.index, df[sensor],
                     label=f"pos{pos_idx}-{sensor}",
                     alpha=0.6,
                     linewidth=1)

    plt.xlabel("Time index (frame)")
    plt.ylabel("Mean ROI Intensity")
    plt.title(f"📈 Sensor Trends — Folder {folder_num} (pos1~9)")
    plt.grid(alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.show()


# -----------------------------
# 5) big_df: 모든 pos 센서를 열로 합친 DataFrame 생성 + 저장
# -----------------------------
def build_and_save_big_df(folder_num: int,
                          base_root: str = BASE_ROOT) -> pd.DataFrame:
    con_dir = get_con_dir(folder_num, base_root=base_root)
    df_list = []

    for pos_idx in range(1, 10):
        csv_path = os.path.join(con_dir, f"pos{pos_idx}_time_sensor.csv")
        if not os.path.exists(csv_path):
            print(f"[WARN] CSV 없음 (스킵): {csv_path}")
            continue

        df = pd.read_csv(csv_path, index_col=0)
        df.columns = [f"pos{pos_idx}_{col}" for col in df.columns]
        df_list.append(df)
        print(f"[LOAD] pos{pos_idx}: shape={df.shape}")

    if not df_list:
        raise ValueError("병합할 DF가 없습니다. CSV 생성이 되었는지 확인하세요.")

    big_df = pd.concat(df_list, axis=1)
    big_df.index.name = "time_index"

    out_csv = os.path.join(con_dir, "ALL_sensors_time_matrix.csv")
    big_df.to_csv(out_csv)
    print(f"\n🎉 Big DataFrame 저장 완료: {out_csv}  shape={big_df.shape}")

    return big_df


# -----------------------------
# 6) 여러 폴더에 대해 전체 파이프라인 실행
# -----------------------------
def run_full_pipeline_for_folders(folder_nums,
                                  base_root: str = BASE_ROOT):
    for folder_num in folder_nums:
        print(f"\n===== VOC Pipeline for folder {folder_num} (base_root={base_root}) =====\n")
        generate_pos_tyx_stacks(folder_num, base_root=base_root)
        generate_time_sensor_csvs(folder_num, base_root=base_root)
        _ = build_and_save_big_df(folder_num, base_root=base_root)
        # 시각화는 필요할 때만 수동 호출 권장
        # plot_crops_for_folder(folder_num, frame_index=0, base_root=base_root)
        # plot_all_sensors(folder_num, base_root=base_root)


# -----------------------------
# 예시 실행 (직접 모듈 실행 시)
# -----------------------------
if __name__ == "__main__":
    # 예: voc1의 folder 1만 돌리기
    folder_num = 1
    run_full_pipeline_for_folders([folder_num], base_root=BASE_ROOT)
