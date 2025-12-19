"""
voc3_labeling.py

VOC3 TIF + crop CSV로부터
1) 시간(행) × 센서(열) 평균 intensity DataFrame 생성
2) 특정 프레임 + crop boundary 오버레이 plotting

"""

import os
import re

import numpy as np
import pandas as pd
from tifffile import imread
from matplotlib.path import Path
import matplotlib.pyplot as plt  # 🔹 새로 추가


# -----------------------------
# 1. well(A1, A2, ...) → 센서 라벨 매핑 테이블
# -----------------------------
WELL_TO_SENSORS = {
    "A1": ["Sensor2",  "Sensor1",  "Sensor11", "Sensor12"],
    "A2": ["Sensor4",  "Sensor3",  "Sensor9",  "Sensor10"],
    "A3": ["Sensor6",  "Sensor5",  "Sensor7",  "Sensor8"],
    "B1": ["Sensor18", "Sensor17", "Sensor19", "Sensor20"],
    "B2": ["Sensor16", "Sensor15", "Sensor21", "Sensor22"],
    "B3": ["Sensor14", "Sensor13", "Sensor23", "Sensor24"],
    "C1": ["Sensor26", "Sensor25", "Sensor35", "Sensor36"],
    "C2": ["Sensor28", "Sensor27", "Sensor33", "Sensor34"],
    "C3": ["Sensor30", "Sensor29", "Sensor31", "Sensor32"],
}


def sensor_ids_from_filename(tif_path: str):
    """
    TIF 파일 이름에서 well 정보를 읽어와서
    crop1~4에 대응하는 센서 라벨 리스트를 반환한다.

    예:
        VOC3_testtime_5050200_1_MMStack_A2-Site_0.ome.tif
        → well = 'A2'
        → ['Sensor4', 'Sensor3', 'Sensor9', 'Sensor10']
    """
    fname = os.path.basename(tif_path)
    # MMStack_A1, MMStack_B3 이런 패턴에서 A1~C3 추출
    m = re.search(r"MMStack_([A-C][1-3])", fname)
    if not m:
        raise ValueError(f"파일명에서 well 정보를 찾지 못했습니다: {fname}")

    well = m.group(1)  # 'A1', 'A2', ..., 'C3'
    if well not in WELL_TO_SENSORS:
        raise ValueError(f"지원하지 않는 well: {well}")

    return WELL_TO_SENSORS[well]


def load_crop_mask(csv_path: str, image_shape):
    """
    crop CSV (다각형 좌표) → (H, W) boolean mask로 변환.

    Parameters
    ----------
    csv_path : str
        crop1.csv 같은 파일 경로
    image_shape : tuple
        (H, W)

    Returns
    -------
    mask : np.ndarray (H, W), dtype=bool
        ROI 안이 True인 마스크
    """
    df = pd.read_csv(csv_path)

    # 좌표 컬럼 이름이 정확치 않으니, 앞의 두 컬럼을 x, y로 사용
    x_col, y_col = df.columns[:2]
    xs = df[x_col].values
    ys = df[y_col].values

    # (N, 2) 폴리곤 좌표 (x, y)
    poly = np.vstack([xs, ys]).T

    H, W = image_shape

    # 이미지 전체 픽셀 좌표 생성
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")  # yy=row, xx=col
    points = np.vstack([xx.ravel(), yy.ravel()]).T  # (num_pixels, 2) = (x, y)

    path = Path(poly)
    mask_flat = path.contains_points(points)
    mask = mask_flat.reshape(H, W)

    return mask


# =========================
# 🔹 새로 추가: polygon만 불러오는 헬퍼
# =========================
def load_polygon(csv_path: str) -> np.ndarray:
    """
    crop CSV에서 (N,2) 형태의 polygon 좌표(x,y)를 반환.
    """
    df = pd.read_csv(csv_path)
    x_col, y_col = df.columns[:2]  # 앞 2개 column을 x,y로 사용
    xs = df[x_col].values
    ys = df[y_col].values
    return np.vstack([xs, ys]).T  # (N,2)


# =========================
# 🔹 새로 추가: frame + crop boundary plotting
# =========================
def plot_frame_with_crops(tif_path: str, crop_dir: str, frame_index: int = 0):
    """
    TIF의 특정 프레임에 crop1~4 polygon boundary를 오버레이해서 플롯.

    Parameters
    ----------
    tif_path : str
        TIF 경로
    crop_dir : str
        crop1.csv ~ crop4.csv가 들어있는 폴더
    frame_index : int, optional
        표시할 시간 프레임 인덱스 (기본 0)
    """
    # 1) 이미지 로드
    img = imread(tif_path)
    if img.ndim != 3:
        raise ValueError(f"예상과 다른 차원입니다. img.ndim={img.ndim}, shape={img.shape}")

    T, H, W = img.shape
    if not (0 <= frame_index < T):
        raise IndexError(f"frame_index={frame_index} 가 범위를 벗어났습니다. (0 ~ {T-1})")

    frame0 = img[frame_index]

    print(f"[PLOT] Loaded TIFF shape: {img.shape}, plotting frame_index={frame_index}")

    # 2) 플롯
    plt.figure(figsize=(8, 8))
    plt.imshow(frame0, cmap="gray")
    plt.title(f"frame {frame_index} with crop boundaries")
    plt.axis("off")

    colors = ["red", "cyan", "yellow", "lime"]  # crop별 구분 색상

    for i in range(1, 5):
        crop_csv = os.path.join(crop_dir, f"crop{i}.csv")
        if not os.path.exists(crop_csv):
            print(f"[WARN] CSV not found: {crop_csv}")
            continue
        polygon = load_polygon(crop_csv)
        plt.plot(polygon[:, 0], polygon[:, 1],
                 color=colors[i - 1],
                 linewidth=2,
                 label=f"crop{i}")

    plt.legend(loc="upper right")
    plt.show()


def extract_time_sensor_dataframe(tif_path: str, crop_dir: str) -> pd.DataFrame:
    """
    하나의 TIF 스택과 crop1~4 CSV를 이용해
    '시간(행) × 센서(열)' 평균 intensity DataFrame을 생성한다.

    Parameters
    ----------
    tif_path : str
        예: "/home/.../VOC3_testtime_5050200_1_MMStack_A1-Site_0.ome.tif"
    crop_dir : str
        crop1.csv, crop2.csv, crop3.csv, crop4.csv가 들어있는 폴더 경로

    Returns
    -------
    df_time_sensor : pd.DataFrame
        index: time_index (0 ~ T-1)
        columns: Sensor 라벨들 (예: 'Sensor2', 'Sensor1', ...)
    """
    print(f"[INFO] TIF 경로: {tif_path}")
    print(f"[INFO] crop 폴더: {crop_dir}")

    # 1) 파일명에서 센서 라벨 리스트 얻기
    sensor_labels = sensor_ids_from_filename(tif_path)
    print("이 TIF에서 crop1~4에 대응하는 센서 라벨:", sensor_labels)

    # 2) TIF 데이터 불러오기
    img = imread(tif_path)
    print("원본 img shape:", img.shape)  # (T, H, W) 예상

    if img.ndim != 3:
        raise ValueError(f"예상과 다른 차원입니다. img.ndim={img.ndim}, shape={img.shape}")

    T, H, W = img.shape
    print(f"T={T}, H={H}, W={W}")

    # 3) crop 마스크 기반으로 시간 x 센서 평균세기 계산
    sensor_time_dict = {}  # key: sensor 라벨, value: 길이 T인 1D array

    for crop_idx in range(1, 5):  # crop1~4
        crop_csv_path = os.path.join(crop_dir, f"crop{crop_idx}.csv")
        print(f"[INFO] crop{crop_idx} CSV 경로: {crop_csv_path}")

        if not os.path.exists(crop_csv_path):
            raise FileNotFoundError(f"CSV가 없습니다: {crop_csv_path}")

        mask = load_crop_mask(crop_csv_path, (H, W))  # (H, W) bool

        # img: (T, H, W), mask: (H, W)
        roi_values = img[:, mask]              # shape: (T, num_pixels)
        mean_over_time = roi_values.mean(axis=1)  # shape: (T,)

        sensor_label = sensor_labels[crop_idx - 1]
        sensor_time_dict[sensor_label] = mean_over_time

        print(
            f"  → {sensor_label}: ROI 픽셀 수 = {mask.sum()}, "
            f"첫 3프레임 mean = {mean_over_time[:3]}"
        )

    # 4) 시간(행) × 센서(열) DataFrame 생성
    df_time_sensor = pd.DataFrame(sensor_time_dict)
    df_time_sensor.index.name = "time_index"

    print("\n=== 시간 x 센서 평균 intensity DataFrame (head) ===")
    print(df_time_sensor.head())

    return df_time_sensor


# -----------------------------
# 모듈을 직접 실행했을 때 테스트용 예시
# -----------------------------
if __name__ == "__main__":
    # ✅ 이 부분만 너의 실제 경로에 맞게 수정해 줘
    example_tif = r"/home/gracejang42/CP/251129/VOC3/VOC3_testtime_5050200_1/VOC3_testtime_5050200_1_MMStack_A1-Site_0.ome.tif"
    example_crop_dir = r"/home/gracejang42/CP/251129/VOC3/coord"

    # 1) DataFrame 추출 테스트
    df = extract_time_sensor_dataframe(example_tif, example_crop_dir)

    # 2) crop boundary 플롯 테스트
    plot_frame_with_crops(example_tif, example_crop_dir, frame_index=0)
