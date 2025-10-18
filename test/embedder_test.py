import json

from src.config import POINT_LANDMARKS
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.manifold import trustworthiness
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score, pairwise_distances
from scipy.stats import spearmanr
from scipy.spatial.distance import pdist, squareform

# def _json_to_numpy(data) -> np.ndarray:
#     person = data
#     # Extract pose keypoints (25 points, 3 values each)
#     pose_kp = np.array(person.get('pose_keypoints_2d', [])).reshape(-1, 3)
#     assert pose_kp.shape[0] == 25, f"pose keypoint shape differs: {pose_kp.shape[0]}"
#
#     # Extract face keypoints (70 points)
#     face_kp = np.array(person.get('face_keypoints_2d', [])).reshape(-1, 3)
#     assert face_kp.shape[0] == 70, f"face keypoint shape differs: {face_kp.shape[0]}"
#
#     # Extract hand keypoints (21 points each)
#     left_hand_kp = np.array(person.get(
#         'hand_left_keypoints_2d', [])).reshape(-1, 3)
#     assert left_hand_kp.shape[0] == 21, f"left hand keypoint shape differs: {left_hand_kp.shape[0]}"
#
#     right_hand_kp = np.array(person.get(
#         'hand_right_keypoints_2d', [])).reshape(-1, 3)
#     assert right_hand_kp.shape[0] == 21, f"right hand keypoint shape differs: {right_hand_kp.shape[0]}"
#
#     # Take x, y coordinates and confidence scores
#     pose_xy = pose_kp[:, :2]
#     pose_conf = pose_kp[:, 2:3]
#
#     face_xy = face_kp[:, :2]
#     face_conf = face_kp[:, 2:3]
#
#     left_hand_xy = left_hand_kp[:, :2]
#     left_hand_conf = left_hand_kp[:, 2:3]
#
#     right_hand_xy = right_hand_kp[:, :2]
#     right_hand_conf = right_hand_kp[:, 2:3]
#
#     # [-1, 1] 정규화
#     # First, find the bounding box of all valid points
#     all_x = np.concatenate([
#         pose_xy[pose_conf[:, 0] > 0.1, 0],
#         face_xy[face_conf[:, 0] > 0.1, 0],
#         left_hand_xy[left_hand_conf[:, 0] > 0.1, 0],
#         right_hand_xy[right_hand_conf[:, 0] > 0.1, 0]
#     ])
#     all_y = np.concatenate([
#         pose_xy[pose_conf[:, 0] > 0.1, 1],
#         face_xy[face_conf[:, 0] > 0.1, 1],
#         left_hand_xy[left_hand_conf[:, 0] > 0.1, 1],
#         right_hand_xy[right_hand_conf[:, 0] > 0.1, 1]
#     ])
#
#     if len(all_x) > 0 and len(all_y) > 0:
#         # Calculate center and scale
#         center_x = (np.max(all_x) + np.min(all_x)) / 2
#         center_y = (np.max(all_y) + np.min(all_y)) / 2
#         scale = max(
#             np.max(all_x) - np.min(all_x),
#             np.max(all_y) - np.min(all_y)) / 2
#
#         # Avoid division by zero
#         if scale < 1:
#             scale = 1
#
#         # [-1, 1] 정규화
#         pose_xy = (pose_xy - [center_x, center_y]) / scale
#         face_xy = (face_xy - [center_x, center_y]) / scale
#         left_hand_xy = (left_hand_xy - [center_x, center_y]) / scale
#         right_hand_xy = (right_hand_xy - [center_x, center_y]) / scale
#
#         # Clip
#         pose_xy = np.clip(pose_xy, -1, 1) # -2, 2)
#         face_xy = np.clip(face_xy, -1, 1) # -2, 2)
#         left_hand_xy = np.clip(left_hand_xy, -1, 1) #  -2, 2)
#         right_hand_xy = np.clip(right_hand_xy, -1, 1) #  -2, 2)
#
#     # Concatenate all keypoints: body(25) + face(70) + left_hand(21) + right_hand(21) = 137
#     all_keypoints = np.concatenate([
#         pose_xy,
#         face_xy,
#         left_hand_xy,
#         right_hand_xy
#     ], axis=0)
#
#     # 선택된 키포인트 추출
#     selected_keypoints = all_keypoints[POINT_LANDMARKS, :]
#
#     # 목 기준 정규화
#     neck_pos = all_keypoints[NECK:NECK + 1, :]
#     neck_mean = np.nanmean(neck_pos, axis=0, keepdims=True)
#     if np.isnan(neck_mean).any():
#         neck_mean = np.array([[0.5, 0.5]])
#
#     std = np.nanstd(selected_keypoints)
#     if std == 0:
#         std = 1.0
#     selected_keypoints = selected_keypoints - neck_mean
#     selected_keypoints = np.nan_to_num(selected_keypoints, 0)
#     print(selected_keypoints.shape)
#
#     return selected_keypoints.astype(np.float32) # (98, 2)

def _json_to_numpy(data) -> np.ndarray:
    person = data
    # Extract pose keypoints (25 points, 3 values each)
    pose_kp = np.array(person.get('pose_keypoints_2d', [])).reshape(-1, 3)
    assert pose_kp.shape[0] == 25, f"pose keypoint shape differs: {pose_kp.shape[0]}"

    # Extract face keypoints (70 points)
    face_kp = np.array(person.get('face_keypoints_2d', [])).reshape(-1, 3)
    assert face_kp.shape[0] == 70, f"face keypoint shape differs: {face_kp.shape[0]}"

    # Extract hand keypoints (21 points each)
    left_hand_kp = np.array(person.get(
        'hand_left_keypoints_2d', [])).reshape(-1, 3)
    assert left_hand_kp.shape[0] == 21, f"left hand keypoint shape differs: {left_hand_kp.shape[0]}"

    right_hand_kp = np.array(person.get(
        'hand_right_keypoints_2d', [])).reshape(-1, 3)
    assert right_hand_kp.shape[0] == 21, f"right hand keypoint shape differs: {right_hand_kp.shape[0]}"

    # Take x, y coordinates and confidence scores
    pose_xy = pose_kp[:, :2]
    pose_conf = pose_kp[:, 2:3]

    face_xy = face_kp[:, :2]
    face_conf = face_kp[:, 2:3]

    left_hand_xy = left_hand_kp[:, :2]
    left_hand_conf = left_hand_kp[:, 2:3]

    right_hand_xy = right_hand_kp[:, :2]
    right_hand_conf = right_hand_kp[:, 2:3]

    all_keypoints_0 = np.concatenate([
        pose_xy,
        face_xy,
        left_hand_xy,
        right_hand_xy
    ], axis=0)

    # [-1, 1] 정규화
    # First, find the bounding box of all valid points
    all_x = np.concatenate([
        pose_xy[pose_conf[:, 0] > 0.1, 0],
        face_xy[face_conf[:, 0] > 0.1, 0],
        left_hand_xy[left_hand_conf[:, 0] > 0.1, 0],
        right_hand_xy[right_hand_conf[:, 0] > 0.1, 0]
    ])
    all_y = np.concatenate([
        pose_xy[pose_conf[:, 0] > 0.1, 1],
        face_xy[face_conf[:, 0] > 0.1, 1],
        left_hand_xy[left_hand_conf[:, 0] > 0.1, 1],
        right_hand_xy[right_hand_conf[:, 0] > 0.1, 1]
    ])

    # print("="*60)
    # print(f"이전 키포인트: \n{all_keypoints_0[POINT_LANDMARKS, :].reshape(-1)}")
    # print("="*60)

    if len(all_x) > 0 and len(all_y) > 0:
        # Calculate center and scale
        center_x = (np.max(all_x) + np.min(all_x)) / 2
        center_y = (np.max(all_y) + np.min(all_y)) / 2
        scale = max(
            np.max(all_x) - np.min(all_x),
            np.max(all_y) - np.min(all_y)) / 2

        # Avoid division by zero
        if scale < 1:
            scale = 1

        # [-1, 1] 정규화
        pose_xy = (pose_xy - [center_x, center_y]) / scale
        face_xy = (face_xy - [center_x, center_y]) / scale
        left_hand_xy = (left_hand_xy - [center_x, center_y]) / scale
        right_hand_xy = (right_hand_xy - [center_x, center_y]) / scale

        # Clip
        pose_xy = np.clip(pose_xy, -2, 2)
        face_xy = np.clip(face_xy, -2, 2)
        left_hand_xy = np.clip(left_hand_xy, -2, 2)
        right_hand_xy = np.clip(right_hand_xy, -2, 2)

    # Concatenate all keypoints: body(25) + face(70) + left_hand(21) + right_hand(21) = 137
    all_keypoints = np.concatenate([
        pose_xy,
        face_xy,
        left_hand_xy,
        right_hand_xy
    ], axis=0)

    # 선택된 키포인트 추출
    selected_keypoints = all_keypoints[POINT_LANDMARKS, :]
    selected_keypoints = np.nan_to_num(selected_keypoints, 0)

    return selected_keypoints.astype(np.float32) # (98, 2)

def visualize_reconstruction(original, reconstructed, n_samples=10, save_path='reconstruction.png'):
    """
    원본 vs 복원 키포인트 비교 시각화

    Parameters:
    -----------
    original : array, shape (n_samples, n_keypoints * 2)
        원본 키포인트 (x1, y1, x2, y2, ...)
    reconstructed : array, shape (n_samples, n_keypoints * 2)
        복원된 키포인트
    n_samples : int
        시각화할 샘플 수
    """
    print(f"\n[시각화] 입력 shape 확인:")
    print(f"  Original: {original.shape}")
    print(f"  Reconstructed: {reconstructed.shape}")
    # Shape 검증
    if original.shape != reconstructed.shape:
        raise ValueError(f"Shape 불일치! Original: {original.shape}, Reconstructed: {reconstructed.shape}")

    n_samples = min(n_samples, original.shape[0])
    n_keypoints = original.shape[1] // 2

    # 원본과 복원 reshape: (n_samples, n_keypoints, 2)
    orig_points = original[:n_samples].reshape(n_samples, n_keypoints, 2)
    recon_points = reconstructed[:n_samples].reshape(n_samples, n_keypoints, 2)

    # 서브플롯 생성
    n_cols = min(5, n_samples)
    n_rows = (n_samples + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
    if n_samples == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx in range(n_samples):
        ax = axes[idx]

        # 원본 (파란색)
        ax.scatter(orig_points[idx, :, 0], orig_points[idx, :, 1],
                   c='blue', s=100, alpha=0.6, label='Original', marker='o')

        # 복원 (빨간색)
        ax.scatter(recon_points[idx, :, 0], recon_points[idx, :, 1],
                   c='red', s=100, alpha=0.6, label='Reconstructed', marker='x')

        # 연결선 (오차 시각화)
        for kp_idx in range(n_keypoints):
            ax.plot([orig_points[idx, kp_idx, 0], recon_points[idx, kp_idx, 0]],
                    [orig_points[idx, kp_idx, 1], recon_points[idx, kp_idx, 1]],
                    'gray', alpha=0.3, linestyle='--')
        # 키포인트 번호 표시
        for kp_idx in range(n_keypoints):
            ax.text(orig_points[idx, kp_idx, 0], orig_points[idx, kp_idx, 1],
                    str(kp_idx), fontsize=8, ha='right')

        # 오차 계산
        error = np.mean(np.sqrt(np.sum((orig_points[idx] - recon_points[idx])**2, axis=1)))

        ax.set_title(f'Sample {idx+1}\nAvg Error: {error:.4f}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.invert_yaxis()  # 이미지 좌표계 (y축 반전)
        ax.set_aspect('equal')

    # 남은 서브플롯 숨기기
    for idx in range(n_samples, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"✅ 시각화 저장: {save_path}")
    plt.close()

def evaluate_reconstruction_quality(original, reconstructed):
    """
    복원 품질 정량 평가
    """
    print("\n" + "="*60)
    print("복원 품질 평가")
    print("="*60)

    # 전체 오차
    mse = np.mean((original - reconstructed) ** 2)
    mae = np.mean(np.abs(original - reconstructed))
    rmse = np.sqrt(mse)

    print(f"\n전체 지표:")
    print(f"  MSE:  {mse:.6f}")
    print(f"  MAE:  {mae:.6f}")
    print(f"  RMSE: {rmse:.6f}")

    # 오차 평가
    if rmse < 0.1:
        print("  ✅ 매우 우수한 복원 품질!")
    elif rmse < 0.3:
        print("  🟢 양호한 복원 품질")
    elif rmse < 0.5:
        print("  🟡 보통 수준")
    else:
        print("  ❌ 나쁜 복원 품질 - 모델 재학습 필요!")

    # 키포인트별 오차
    n_keypoints = original.shape[1] // 2
    orig_reshaped = original.reshape(-1, n_keypoints, 2)
    recon_reshaped = reconstructed.reshape(-1, n_keypoints, 2)

    # 각 키포인트의 Euclidean distance
    keypoint_errors = np.sqrt(np.sum((orig_reshaped - recon_reshaped)**2, axis=2))
    avg_keypoint_error = np.mean(keypoint_errors, axis=0)

    print(f"\n키포인트별 평균 오차 (상위 10개):")
    sorted_indices = np.argsort(avg_keypoint_error)[::-1][:10]
    for rank, kp_idx in enumerate(sorted_indices, 1):
        print(f"  {rank}. Keypoint {kp_idx:2d}: {avg_keypoint_error[kp_idx]:.4f}")

    print(f"\n전체 평균: {np.mean(avg_keypoint_error):.4f}")
    print(f"최대 오차: {np.max(avg_keypoint_error):.4f} (Keypoint {np.argmax(avg_keypoint_error)})")
    print(f"최소 오차: {np.min(avg_keypoint_error):.4f} (Keypoint {np.argmin(avg_keypoint_error)})")

    # 정규화된 좌표 (0-1 범위) 가정 시 픽셀 오차 추정
    if np.max(original) <= 1.0:
        print(f"\n※ 정규화 좌표 가정 (이미지 크기 1920x1080):")
        pixel_error_x = np.mean(avg_keypoint_error) * 1920
        pixel_error_y = np.mean(avg_keypoint_error) * 1080
        print(f"  추정 픽셀 오차: X={pixel_error_x:.2f}px, Y={pixel_error_y:.2f}px")

    # 데이터 범위 자동 감지
    data_min = np.min(original)
    data_max = np.max(original)
    print(f"\n데이터 범위: [{data_min:.2f}, {data_max:.2f}]")

    if data_max <= 1.0 and data_min >= -1.0:
        print("  → [-1, 1] 정규화 감지")
        scale = 1920  # 예시
    elif data_max <= 1.0 and data_min >= 0.0:
        print("  → [0, 1] 정규화 감지")
        scale = 1920
    else:
        print("  → 픽셀 좌표 또는 z-score 정규화")
        scale = 1

    return {
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'keypoint_errors': avg_keypoint_error
    }

# 32차원이라 무의미할 듯;;
# def visualize_embedding_space(embeddings, labels=None, save_path='embedding_space.png'):
#     """
#     저차원 임베딩 공간 시각화 (2D 또는 3D)
#     """
#     print(f"\n[임베딩 시각화] 입력 shape: {embeddings.shape}")
#
#     # Shape 자동 수정
#     if len(embeddings.shape) == 3:
#         print(f"  ⚠️ 3D shape 감지: {embeddings.shape} → squeeze")
#         embeddings = np.squeeze(embeddings, axis=1)
#         print(f"  ✅ 수정 후: {embeddings.shape}")
#
#     n_dims = embeddings.shape[1]
#     print(f"  임베딩 차원: {n_dims}D") # 32차원
#
#     if n_dims == 2:
#         # 2D 시각화
#         plt.figure(figsize=(10, 8))
#
#         if labels is not None:
#             scatter = plt.scatter(embeddings[:, 0], embeddings[:, 1],
#                                   c=labels, cmap='tab10', s=50, alpha=0.7)
#             plt.colorbar(scatter, label='Class')
#         else:
#             plt.scatter(embeddings[:, 0], embeddings[:, 1],
#                         s=50, alpha=0.7, c='steelblue')
#
#             # 점에 번호 표시 (샘플 수가 적을 때)
#             if len(embeddings) <= 20:
#                 for i, (x, y) in enumerate(embeddings):
#                     plt.text(x, y, str(i), fontsize=9, ha='right')
#
#         plt.title('UMAP Embedding Space (2D)', fontsize=14, fontweight='bold')
#         plt.xlabel('UMAP Dimension 1')
#         plt.ylabel('UMAP Dimension 2')
#         plt.grid(True, alpha=0.3)
#
#     elif n_dims >= 3:
#         # 3D 시각화
#         from mpl_toolkits.mplot3d import Axes3D
#
#         fig = plt.figure(figsize=(12, 10))
#         ax = fig.add_subplot(111, projection='3d')
#
#         if labels is not None:
#             scatter = ax.scatter(embeddings[:, 0], embeddings[:, 1], embeddings[:, 2],
#                                  c=labels, cmap='tab10', s=50, alpha=0.7)
#             plt.colorbar(scatter, label='Class')
#         else:
#             ax.scatter(embeddings[:, 0], embeddings[:, 1], embeddings[:, 2],
#                        s=50, alpha=0.7, c='steelblue')
#
#         ax.set_title('UMAP Embedding Space (3D)', fontsize=14, fontweight='bold')
#         ax.set_xlabel('UMAP Dimension 1')
#         ax.set_ylabel('UMAP Dimension 2')
#         ax.set_zlabel('UMAP Dimension 3')
#
#     else:
#         print("⚠️ 임베딩 차원이 1차원입니다. 시각화를 생략합니다.")
#         return
#
#     plt.tight_layout()
#     plt.savefig(save_path, dpi=300, bbox_inches='tight')
#     plt.show()
#
#     print(f"✅ 임베딩 공간 시각화 저장: {save_path}")
#     plt.close()

def quick_quality_check(X_original, X_embedded):
    """
    빠른 품질 검사
    ✅ 수정: shape 자동 수정
    """
    print("="*60)
    print("⚡ 빠른 품질 검사 (핵심 지표만)")
    print("="*60)

    print(f"\n입력 shape:")
    print(f"  Original: {X_original.shape}")
    print(f"  Embedded: {X_embedded.shape}")

    # Shape 자동 수정
    if len(X_embedded.shape) == 3:
        print(f"  ⚠️ Embedded shape 3D 감지 → squeeze")
        X_embedded = np.squeeze(X_embedded, axis=1)
        print(f"  ✅ 수정 후: {X_embedded.shape}")

    # 최소 샘플 수 확인
    if X_original.shape[0] < 10:
        print(f"\n⚠️ 샘플 수가 너무 적습니다: {X_original.shape[0]}개")
        print("  신뢰성 있는 평가를 위해 최소 100개 이상 권장")
        return None

    # Trustworthiness
    k = min(15, X_original.shape[0] - 1)
    trust = trustworthiness(X_original, X_embedded, n_neighbors=k)

    # Distance correlation (샘플 수가 적으면 전체 사용)
    sample_size = min(1000, X_original.shape[0])
    if sample_size < X_original.shape[0]:
        indices = np.random.choice(X_original.shape[0], sample_size, replace=False)
        dist_orig = pdist(X_original[indices])
        dist_emb = pdist(X_embedded[indices])
    else:
        dist_orig = pdist(X_original)
        dist_emb = pdist(X_embedded)

    corr, _ = spearmanr(dist_orig, dist_emb)

    print(f"\n✓ Trustworthiness (k={k}): {trust:.4f}")
    print(f"✓ Distance Correlation: {corr:.4f}")

    # 종합 판단
    print("\n" + "="*60)
    if trust > 0.9 and corr > 0.7:
        print("🏆 매우 우수한 축소 품질!")
        print("   → 프로덕션 사용 가능")
    elif trust > 0.7 and corr > 0.5:
        print("✅ 양호한 축소 품질")
        print("   → 대부분의 경우 충분")
    else:
        print("⚠️ 개선 필요")
        print("   → 하이퍼파라미터 조정 권장")
    print("="*60)

    return {'trustworthiness': trust, 'correlation': corr}

# ========================================
# 방법 1: Trustworthiness & Continuity (가장 중요!)
# ========================================
def method1_neighbor_preservation(X_original, X_embedded, k_values=[5, 10, 15, 20]):
    """
    이웃 관계 보존율 - UMAP의 핵심 목표
    """
    print("="*60)
    print("방법 1: 이웃 관계 보존율 (Trustworthiness & Continuity)")
    print("="*60)

    for k in k_values:
        # Trustworthiness: 임베딩에서 가까운 점이 원본에서도 가까운가?
        trust = trustworthiness(X_original, X_embedded, n_neighbors=k)

        # Continuity 계산
        n_samples = X_original.shape[0]
        nbrs_orig = NearestNeighbors(n_neighbors=k+1).fit(X_original)
        nbrs_emb = NearestNeighbors(n_neighbors=k+1).fit(X_embedded)

        _, idx_orig = nbrs_orig.kneighbors(X_original)
        _, idx_emb = nbrs_emb.kneighbors(X_embedded)

        cont = 0
        for i in range(n_samples):
            preserved = len(set(idx_orig[i, 1:]) & set(idx_emb[i, 1:]))
            cont += preserved / k
        cont /= n_samples

        print(f"\nk={k:2d}:")
        print(f"  Trustworthiness: {trust:.4f} {'✅' if trust > 0.9 else '🟢' if trust > 0.7 else '⚠️'}")
        print(f"  Continuity:      {cont:.4f} {'✅' if cont > 0.9 else '🟢' if cont > 0.7 else '⚠️'}")

    print("\n해석:")
    print("  0.9+: 매우 우수 (이웃 관계 거의 완벽 보존)")
    print("  0.7-0.9: 양호 (대부분의 이웃 관계 보존)")
    print("  0.5-0.7: 보통 (일부 이웃 관계 손실)")
    print("  0.5 미만: 나쁨 (많은 이웃 관계 왜곡)")

# ========================================
# 방법 2: 거리 상관관계 (Distance Correlation)
# ========================================
def method2_distance_correlation(X_original, X_embedded, sample_size=1000):
    """
    원본 거리와 임베딩 거리의 상관관계
    """
    print("\n" + "="*60)
    print("방법 2: 거리 상관관계")
    print("="*60)

    # 샘플링 (전체는 너무 느림)
    n_samples = min(sample_size, X_original.shape[0])
    indices = np.random.choice(X_original.shape[0], n_samples, replace=False)

    X_orig_sample = X_original[indices]
    X_emb_sample = X_embedded[indices]

    # Pairwise distances
    dist_orig = pdist(X_orig_sample)
    dist_emb = pdist(X_emb_sample)

    # Pearson correlation (절대 거리)
    pearson_corr = np.corrcoef(dist_orig, dist_emb)[0, 1]

    # Spearman correlation (순위)
    spearman_corr, pvalue = spearmanr(dist_orig, dist_emb)

    print(f"\nPearson correlation:  {pearson_corr:.4f}")
    print(f"Spearman correlation: {spearman_corr:.4f} (p={pvalue:.6f})")

    # 시각화
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(dist_orig, dist_emb, alpha=0.3, s=5)
    plt.xlabel('Original Distance')
    plt.ylabel('Embedded Distance')
    plt.title(f'Distance Correlation (Pearson: {pearson_corr:.3f})')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.hist2d(dist_orig, dist_emb, bins=50, cmap='Blues')
    plt.xlabel('Original Distance')
    plt.ylabel('Embedded Distance')
    plt.title('Distance Density')
    plt.colorbar(label='Count')

    plt.tight_layout()
    plt.savefig('distance_correlation.png', dpi=300)
    plt.show()

    print("\n해석:")
    print("  0.7+: 거리 관계가 잘 보존됨")
    print("  0.5-0.7: 대략적인 거리 관계 유지")
    print("  0.5 미만: 거리 관계 많이 왜곡")

# ========================================
# 방법 3: Stress (Kruskal's Stress)
# ========================================
def method3_stress(X_original, X_embedded, sample_size=1000):
    """
    MDS-style stress: 거리 차이의 normalized sum
    """
    print("\n" + "="*60)
    print("방법 3: Kruskal's Stress")
    print("="*60)

    n_samples = min(sample_size, X_original.shape[0])
    indices = np.random.choice(X_original.shape[0], n_samples, replace=False)

    # Distance matrices
    D_orig = squareform(pdist(X_original[indices]))
    D_emb = squareform(pdist(X_embedded[indices]))

    # Kruskal's stress formula 1
    numerator = np.sum((D_orig - D_emb) ** 2)
    denominator = np.sum(D_orig ** 2)
    stress = np.sqrt(numerator / denominator)

    print(f"\nKruskal's Stress: {stress:.4f}")

    print("\n해석:")
    print("  0.00-0.05: 완벽")
    print("  0.05-0.10: 우수")
    print("  0.10-0.20: 양호")
    print("  0.20+: 나쁨")

    if stress < 0.05:
        print("  ✅ 거의 완벽한 거리 보존!")
    elif stress < 0.10:
        print("  ✅ 우수한 거리 보존")
    elif stress < 0.20:
        print("  🟢 양호한 수준")
    else:
        print("  ⚠️ 개선 필요")

# ========================================
# 방법 4: Local Structure Preservation
# ========================================
def method4_local_structure(X_original, X_embedded, k=15):
    """
    각 점에서 local structure 보존율
    """
    print("\n" + "="*60)
    print("방법 4: Local Structure Preservation")
    print("="*60)

    n_samples = X_original.shape[0]

    # k-NN in both spaces
    nbrs_orig = NearestNeighbors(n_neighbors=k+1).fit(X_original)
    nbrs_emb = NearestNeighbors(n_neighbors=k+1).fit(X_embedded)

    dist_orig, idx_orig = nbrs_orig.kneighbors(X_original)
    dist_emb, idx_emb = nbrs_emb.kneighbors(X_embedded)

    # Local structure similarity
    preservation_scores = []
    for i in range(n_samples):
        # Neighbor overlap
        overlap = len(set(idx_orig[i, 1:]) & set(idx_emb[i, 1:]))
        preservation_scores.append(overlap / k)

    preservation_scores = np.array(preservation_scores)

    print(f"\nLocal Structure Preservation:")
    print(f"  평균: {np.mean(preservation_scores):.4f}")
    print(f"  중앙값: {np.median(preservation_scores):.4f}")
    print(f"  표준편차: {np.std(preservation_scores):.4f}")
    print(f"  최소: {np.min(preservation_scores):.4f}")
    print(f"  최대: {np.max(preservation_scores):.4f}")

    # 분포 시각화
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.hist(preservation_scores, bins=30, edgecolor='black', alpha=0.7)
    plt.xlabel('Local Preservation Score')
    plt.ylabel('Frequency')
    plt.title('Distribution of Local Structure Preservation')
    plt.axvline(np.mean(preservation_scores), color='red',
                linestyle='--', label=f'Mean: {np.mean(preservation_scores):.3f}')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.boxplot(preservation_scores)
    plt.ylabel('Preservation Score')
    plt.title('Local Structure Preservation')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('local_structure_preservation.png', dpi=300)
    plt.show()

    # 문제 있는 점들 찾기
    poor_preservation = np.where(preservation_scores < 0.5)[0]
    if len(poor_preservation) > 0:
        print(f"\n⚠️ 보존율 낮은 점: {len(poor_preservation)}개 ({len(poor_preservation)/n_samples*100:.1f}%)")
    else:
        print(f"\n✅ 모든 점이 양호한 보존율!")

if __name__=="__main__":
    LOCAL_UMAP_SAVE_PATH='../models/umap_models'
    encoder_path = os.path.join(LOCAL_UMAP_SAVE_PATH,'encoder_98.keras')
    decoder_path = os.path.join(LOCAL_UMAP_SAVE_PATH, 'decoder_98.keras')
    encoder = tf.keras.models.load_model(encoder_path)
    decoder = tf.keras.models.load_model(decoder_path)

    print("="*60)
    print("Encoder Summary:")
    print("="*60)
    encoder.summary()

    print("\n" + "="*60)
    print("Decoder Summary:")
    print("="*60)
    decoder.summary()

    directory_path = Path('../data/openpose_keypoints/NIA_SL_SEN0181/NIA_SL_SEN0181_D/NIA_SL_SEN0181_REAL01_D')
    json_files = list(directory_path.glob('*.json'))[:10]

    seq = []
    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)
        assert 'people' in data, "데이터에 키포인트 없음"
        people_data = data['people']

        assert isinstance(people_data, dict), "이상한 데이터 형식"
        person = people_data
        assert len(people_data) != 0, "데이터 길이 0"
        keypoint = _json_to_numpy(person)
        keypoint = keypoint.reshape(-1)
        seq.append(keypoint)

    # keypoints_array = np.stack(seq) # (10, 98)
    print("\n" + "="*60)
    print("데이터 로드 완료")
    print("="*60)
    # print(f"키포인트 배열 shape: {keypoints_array.shape}")
    # print(f"첫 번째 샘플:\n{keypoints_array[0]}")
    keypoints_array = seq
    seq = np.array(seq)
    print(f"seq 크기: {seq.shape}") # (10, 98)

    print("\n" + "="*60)
    print("Encoding & Decoding...")
    print("="*60)

    # embeddings = []
    # reconstruted = []
    # for s in keypoints_array:
    #     s = np.expand_dims(s, axis=0)           # (98,) → (1, 98)
    #     e = encoder.predict(s)                  # (1, 32)
    #     embeddings.append(e)                    # [(1,32), (1,32), ...]
    #
    #     reconstruct = decoder.predict(e)        # (1, 98)
    #     reconstruted.append(reconstruct)        # [(1,98), (1,98), ...]

    embeddings = encoder.predict(seq)
    print(f"embedding 차원: {embeddings.shape}") # (10,32)
    print(f"first embedding: \n{embeddings[0]}")

    # embs = np.concatenate(embeddings)
    # recs = np.concatenate(reconstruted)

    # 내부 데이터 확인
    # for i, e in enumerate(embeddings):
    #     print("="*60)
    #     print(f"{i+1} 번째 임베딩 샘플: \n{embeddings[i]}")
    #     print("="*60)
    #     print(f"{i+1} 번째 재구성 샘플: \n{reconstruted[i]}")
    #     print("="*60)
    #
    # print(f"\n결과 shape:")
    # print(f"  원본: {keypoints_array.shape}") # (10, 98)
    # print(f"  임베딩: {embs.shape}")           # (10, 32)
    # print(f"  복원: {recs.shape}")            # (10, 98)
    #
    # if embs.shape[0] != keypoints_array.shape[0]:
    #     print(f"⚠️ Shape 불일치 감지!")
    #
    # if recs.shape != keypoints_array.shape:
    #     print(f"⚠️ Reconstruction shape 불일치!")
    #     print(f"  Expected: {keypoints_array.shape}, Got: {recs.shape}")
    #
    #     # 자동 수정 시도
    #     if len(recs.shape) == 3:
    #         recs = np.squeeze(recs, axis=1)
    #         print(f"  ✅ Squeeze 후: {recs.shape}")
    #
    # print("\n" + "="*60)
    # print("시각화 생성 중...")
    # print("="*60)
    #
    # try:
    #     visualize_reconstruction(
    #         original=keypoints_array,
    #         reconstructed=recs,
    #         n_samples=10,
    #         save_path='reconstruction_comparison.png'
    #     )
    # except Exception as e:
    #     print(f"❌ 복원 시각화 실패: {e}")
    #
    # # try:
    # #     visualize_embedding_space(
    # #         embeddings=embs,
    # #         save_path='embedding_space.png'
    # #     )
    # # except Exception as e:
    # #     print(f"❌ 임베딩 시각화 실패: {e}")
    #
    # # 품질 평가
    # try:
    #     quality = evaluate_reconstruction_quality(
    #         original=keypoints_array,
    #         reconstructed=recs
    #     )
    # except Exception as e:
    #     print(f"❌ 품질 평가 실패: {e}")

    # # 빠른 검사 (샘플이 10개라 생략 권장)
    # print("\n" + "="*60)
    # print("참고: 샘플이 10개뿐이라 Trustworthiness는 신뢰성이 낮습니다.")
    # print("      최소 100개 이상의 테스트 데이터로 재평가 권장")
    # print("="*60)