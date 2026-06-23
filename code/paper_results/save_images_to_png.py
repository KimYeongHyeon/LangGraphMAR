"""
PNG 이미지 저장 함수 - PPT 발표용
Overlay 버전과 Non-overlay 버전 두 가지 제공
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path


def save_images_as_png(
    images_dict,
    roi=None,
    vmin=-1000,
    vmax=2000,
    save_dir='results',
    filename='comparison',
    overlay=True,
    dpi=300,
    figsize_per_image=(6, 6)
):
    """
    여러 이미지를 PNG로 저장 (PPT 발표용)
    
    Parameters:
    -----------
    images_dict : dict
        {'title': image_array} 형태의 딕셔너리
        예: {'GT': gt_image, 'FBP': fbp_image, ...}
    roi : tuple, optional
        (x, y, width, height) 형태의 ROI 좌표
    vmin : float
        colormap 최소값 (HU)
    vmax : float
        colormap 최대값 (HU)
    save_dir : str
        저장 디렉토리
    filename : str
        파일명 (확장자 제외)
    overlay : bool
        True: ROI 오버레이 버전, False: ROI 없는 버전
    dpi : int
        저장 해상도 (기본값: 300 - PPT에 적합)
    figsize_per_image : tuple
        각 이미지당 크기 (inch)
    
    Returns:
    --------
    str : 저장된 파일 경로
    """
    
    # 저장 디렉토리 생성
    os.makedirs(save_dir, exist_ok=True)
    
    # 이미지 개수
    n_images = len(images_dict)
    
    # Figure 생성 - 가로로 나열
    fig, axes = plt.subplots(
        1, n_images, 
        figsize=(figsize_per_image[0] * n_images, figsize_per_image[1])
    )
    
    # 단일 이미지인 경우 처리
    if n_images == 1:
        axes = [axes]
    
    # 각 이미지 플롯
    for idx, (title, image) in enumerate(images_dict.items()):
        ax = axes[idx]
        
        # 이미지가 3D인 경우 2D로 변환
        if image.ndim == 3:
            image = image.squeeze()
        
        # 이미지 표시
        im = ax.imshow(image, cmap='gray', vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=16, fontweight='bold', pad=10)
        ax.axis('off')
        
        # ROI 오버레이
        if overlay and roi is not None:
            x, y, w, h = roi
            rect = patches.Rectangle(
                (x, y), w, h,
                linewidth=3,
                edgecolor='red',
                facecolor='none',
                linestyle='--'
            )
            ax.add_patch(rect)
    
    # 레이아웃 조정
    plt.tight_layout(pad=2.0)
    
    # 파일명 생성
    if overlay:
        output_filename = f"{filename}_overlay.png"
    else:
        output_filename = f"{filename}_no_overlay.png"
    
    output_path = os.path.join(save_dir, output_filename)
    
    # PNG 저장 - PPT에 최적화된 설정
    plt.savefig(
        output_path,
        dpi=dpi,
        bbox_inches='tight',
        pad_inches=0.1,
        facecolor='white',
        edgecolor='none',
        format='png',
        transparent=False
    )
    
    plt.close(fig)
    
    print(f"✓ Saved: {output_path}")
    return output_path


def save_roi_comparison(
    images_dict,
    roi,
    vmin=-1000,
    vmax=2000,
    save_dir='results',
    filename='roi_comparison',
    dpi=300,
    figsize_per_roi=(4, 4)
):
    """
    ROI 영역만 확대하여 비교 이미지 저장
    
    Parameters:
    -----------
    images_dict : dict
        {'title': image_array} 형태
    roi : tuple
        (x, y, width, height)
    vmin, vmax : float
        colormap 범위
    save_dir : str
        저장 디렉토리
    filename : str
        파일명
    dpi : int
        해상도
    figsize_per_roi : tuple
        각 ROI당 크기
    
    Returns:
    --------
    str : 저장된 파일 경로
    """
    
    os.makedirs(save_dir, exist_ok=True)
    
    x, y, w, h = roi
    n_images = len(images_dict)
    
    # Figure 생성
    fig, axes = plt.subplots(
        1, n_images,
        figsize=(figsize_per_roi[0] * n_images, figsize_per_roi[1])
    )
    
    if n_images == 1:
        axes = [axes]
    
    # 각 ROI 영역 플롯
    for idx, (title, image) in enumerate(images_dict.items()):
        ax = axes[idx]
        
        # 이미지가 3D인 경우 2D로 변환
        if image.ndim == 3:
            image = image.squeeze()
        
        # ROI 영역 추출
        roi_image = image[y:y+h, x:x+w]
        
        # 표시
        im = ax.imshow(roi_image, cmap='gray', vmin=vmin, vmax=vmax)
        ax.set_title(f"{title} (ROI)", fontsize=14, fontweight='bold', pad=8)
        ax.axis('off')
    
    plt.tight_layout(pad=1.5)
    
    output_path = os.path.join(save_dir, f"{filename}_roi_only.png")
    
    plt.savefig(
        output_path,
        dpi=dpi,
        bbox_inches='tight',
        pad_inches=0.1,
        facecolor='white',
        edgecolor='none',
        format='png',
        transparent=False
    )
    
    plt.close(fig)
    
    print(f"✓ Saved ROI: {output_path}")
    return output_path


def save_all_versions(
    images_dict,
    roi=None,
    vmin=-1000,
    vmax=2000,
    save_dir='results',
    base_filename='comparison',
    dpi=300
):
    """
    Overlay, Non-overlay, ROI 확대 버전 모두 저장
    
    Parameters:
    -----------
    images_dict : dict
        이미지 딕셔너리
    roi : tuple, optional
        ROI 좌표
    vmin, vmax : float
        colormap 범위
    save_dir : str
        저장 디렉토리
    base_filename : str
        기본 파일명
    dpi : int
        해상도 (기본 300 - PPT 적합)
    
    Returns:
    --------
    dict : 저장된 파일 경로들
    """
    
    saved_files = {}
    
    # 1. Overlay 버전
    saved_files['overlay'] = save_images_as_png(
        images_dict,
        roi=roi,
        vmin=vmin,
        vmax=vmax,
        save_dir=save_dir,
        filename=base_filename,
        overlay=True,
        dpi=dpi
    )
    
    # 2. Non-overlay 버전
    saved_files['no_overlay'] = save_images_as_png(
        images_dict,
        roi=roi,
        vmin=vmin,
        vmax=vmax,
        save_dir=save_dir,
        filename=base_filename,
        overlay=False,
        dpi=dpi
    )
    
    # 3. ROI 확대 버전 (roi가 제공된 경우)
    if roi is not None:
        saved_files['roi_only'] = save_roi_comparison(
            images_dict,
            roi=roi,
            vmin=vmin,
            vmax=vmax,
            save_dir=save_dir,
            filename=base_filename,
            dpi=dpi
        )
    
    print(f"\n✅ All versions saved to: {save_dir}/")
    return saved_files


# 사용 예제
if __name__ == "__main__":
    # 예제 데이터 생성
    sample_num = 483
    anatomy = 'body'
    data_dir = f'data/{anatomy}_{sample_num}'
    
    # 데이터 로드
    gt_image = np.load(os.path.join(data_dir, 'gt_image.npy'))
    fbp_image = np.load(os.path.join(data_dir, 'FBP_image.npy'))
    proposed_image = np.load(os.path.join(data_dir, 'proposed_image.npy'))
    unet_image = np.load(os.path.join(data_dir, 'unet_image.npy'))
    
    # 이미지 딕셔너리 구성
    images_dict = {
        'Ground Truth': gt_image,
        'FBP': fbp_image,
        'U-Net': unet_image,
        'Proposed': proposed_image
    }
    
    # ROI 설정
    roi = (254, 348, 93, 64)  # (x, y, width, height)
    
    # 저장 디렉토리
    save_dir = f'results_png/{anatomy}_{sample_num}'
    
    # 모든 버전 저장
    saved_files = save_all_versions(
        images_dict=images_dict,
        roi=roi,
        vmin=-135,
        vmax=215,
        save_dir=save_dir,
        base_filename=f'{anatomy}_{sample_num}',
        dpi=300
    )
    
    print("\n저장된 파일들:")
    for key, path in saved_files.items():
        print(f"  - {key}: {path}")
