import numpy as np
import cv2
import os
import pandas as pd
from phasepack import phasecong
from tqdm import tqdm

def normalize_image(image, vmin, vmax):
    """Normalize and scale an image to uint8 range."""
    tmp_image = np.clip(image, vmin, vmax)
    tmp_image = (tmp_image - vmin) / (vmax - vmin) * 255
    return tmp_image.astype(np.uint8)

def calculate_fsim(gray1, gray2):
    # Calculate phase congruency (PC) and gradient magnitude (GM)
    pc1_data = phasecong(gray1)
    pc2_data = phasecong(gray2)
    pc1 = np.array(pc1_data[0])
    pc2 = np.array(pc2_data[0])
    
    gm1 = cv2.Sobel(gray1, cv2.CV_64F, 1, 1, ksize=3)
    gm2 = cv2.Sobel(gray2, cv2.CV_64F, 1, 1, ksize=3)
    gm1 = np.sqrt(gm1**2 + gm1**2)
    gm2 = np.sqrt(gm2**2 + gm2**2)

    # Define constants T1 and T2
    T1 = 0.85
    T2 = 160

    # Calculate similarity
    pc_sim = 2 * pc1 * pc2 / (pc1**2 + pc2**2 + T1)
    gm_sim = 2 * gm1 * gm2 / (gm1**2 + gm2**2 + T2)
    
    # Handle potential division by zero
    pc_sim = np.where((pc1**2 + pc2**2 + T1) > 0, pc_sim, 0)
    gm_sim = np.where((gm1**2 + gm2**2 + T2) > 0, gm_sim, 0)
    
    sim = pc_sim * gm_sim

    # Calculate FSIM
    fsim = np.mean(sim)
    return fsim

def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2
    
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return ssim_map.mean()

def calculate_rmse(image1, image2):
    image1 = np.array(image1)
    image2 = np.array(image2)
    mse = np.mean((image1 - image2) ** 2)
    rmse = np.sqrt(mse)
    return rmse

def process_folder(folder_path):
    folder_name = os.path.basename(folder_path)
    anatomy = folder_name.split('_')[0]
    
    # Body와 Head 모두 동일한 윈도우 사용 (FSIM 일관성 확보)
    vmin, vmax = -150, 400
        
    mask_path = os.path.join(folder_path, 'gt_metal_mask.npy')
    gt_path = os.path.join(folder_path, 'gt_image.npy')
    
    if not os.path.exists(mask_path) or not os.path.exists(gt_path):
        return []
        
    img_mask = np.load(mask_path)
    pos_mask = np.where(img_mask == 1)
    
    img_gt = np.load(gt_path)
    img_gt[pos_mask] = 0
    
    model_files = {
        'FBP': 'FBP_image.npy',
        'UNet': 'unet_image.npy',
        'Uformer': 'uformer_image.npy',
        'Restormer': 'restormer_image.npy',
        'NAFNet': 'nafnet_image.npy',
        'Proposed': 'proposed_image.npy'
    }
    
    results = []
    nor_img_gt = normalize_image(img_gt, vmin, vmax)
    
    for model_name, file_name in model_files.items():
        file_path = os.path.join(folder_path, file_name)
        if not os.path.exists(file_path):
            continue
            
        img = np.load(file_path)
        img[pos_mask] = 0
        
        s_score = ssim(img_gt, img)
        r_score = calculate_rmse(img_gt, img)
        
        nor_img = normalize_image(img, vmin, vmax)
        f_score = calculate_fsim(nor_img_gt, nor_img)
        
        results.append({
            'folder': folder_name,
            'anatomy': anatomy,
            'model': model_name,
            'ssim': s_score,
            'fsim': f_score * 100,
            'rmse': r_score
        })
    return results

def main():
    data_dir = 'code/paper_results/data/'
    csv_output = 'code/paper_results/all_metrics_sequential.csv'
    
    folders = sorted([os.path.join(data_dir, d) for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    
    # Check for existing progress
    if os.path.exists(csv_output):
        existing_df = pd.read_csv(csv_output)
        processed_folders = set(existing_df['folder'].unique())
        print(f"Resuming from {len(processed_folders)} processed folders.")
    else:
        processed_folders = set()

    for folder in tqdm(folders):
        folder_name = os.path.basename(folder)
        if folder_name in processed_folders:
            continue
            
        folder_results = process_folder(folder)
        if folder_results:
            df_new = pd.DataFrame(folder_results)
            header = not os.path.exists(csv_output)
            df_new.to_csv(csv_output, mode='a', index=False, header=header)
            
    # Final aggregation
    print("\nCalculating final statistics...")
    df = pd.read_csv(csv_output)
    stats = df.groupby(['anatomy', 'model'])[['ssim', 'fsim', 'rmse']].agg(['mean', 'std']).reset_index()
    stats.columns = ['_'.join(col).strip('_') for col in stats.columns.values]
    
    print("\nMetric Stats (Mean and Std):")
    print(stats.to_string())
    
    stats.to_csv('code/paper_results/metrics_stats_sequential.csv', index=False)
    print("\nResults saved to code/paper_results/metrics_stats_sequential.csv")

if __name__ == '__main__':
    main()


