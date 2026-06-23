import numpy as np
import torch
import numpy as np
from skimage.metrics import structural_similarity as SKssim
from skimage.metrics import peak_signal_noise_ratio as SKpsnr
from skimage.metrics import normalized_root_mse as SKnrmse

try:
    from utils.ct import transform_image_unit_mm_to_HU
except ModuleNotFoundError:
    from ct import transform_image_unit_mm_to_HU
from icecream import ic


import copy
import numpy as np
from skimage.feature import graycomatrix
from scipy.linalg import eig  # Corrected import: use eig instead of eigs
from skimage.feature import graycomatrix
import warnings


import numpy as np

def GLCMFeaturesInvariant(glcm, homogeneityConstant=1, inverseDifferenceConstant=1):
    """
    Calculate gray-level invariant Haralick features from one or several GLCMs.

    Parameters
    ----------
    glcm : numpy.ndarray
        2D or 3D array representing one (2D) or several (3D, last dim = number of GLCMs) GLCMs.
        Each individual GLCM must be square.
    homogeneityConstant : float, optional
        Constant for the homogeneity feature (default is 1).
    inverseDifferenceConstant : float, optional
        Constant for the inverse difference feature (default is 1).

    Returns
    -------
    out : dict
        Dictionary containing all calculated Haralick features as 1D numpy arrays.
    """
    # 2D이면 3D로 확장
    if glcm.ndim == 2:
        glcm = glcm[:, :, np.newaxis]
    elif glcm.ndim != 3:
        raise ValueError("The GLCM should be a 2-D or 3-D numpy array.")
    
    nGrayLevels, nGrayLevels2, p = glcm.shape
    if nGrayLevels != nGrayLevels2:
        raise ValueError("Each GLCM should be square (NumLevels x NumLevels).")
    
    # Differential step sizes
    dA = 1 / (nGrayLevels ** 2)
    dL = 1 / nGrayLevels
    dXplusY = 1 / (2 * nGrayLevels - 1)
    dXminusY = 1 / nGrayLevels
    dkdiag = 1 / nGrayLevels

    # 정규화: 각 GLCM 슬라이스별로 전체합과 dA를 사용
    glcm = glcm.astype(np.float64).copy()
    for k in range(p):
        total = np.sum(glcm[:, :, k])
        if total != 0:
            glcm[:, :, k] = glcm[:, :, k] / (total * dA)
    
    # 피처 결과용 배열들 (각 피처는 길이 p인 1D numpy array)
    autoCorrelation = np.zeros(p)
    clusterProminence = np.zeros(p)
    clusterShade = np.zeros(p)
    contrast = np.zeros(p)
    correlation = np.zeros(p)
    differenceAverage = np.zeros(p)
    differenceEntropy = np.zeros(p)
    differenceVariance = np.zeros(p)
    dissimilarity = np.zeros(p)
    energy = np.zeros(p)
    entropy_arr = np.zeros(p)
    homogeneity = np.zeros(p)
    informationMeasureOfCorrelation1 = np.zeros(p)
    informationMeasureOfCorrelation2 = np.zeros(p)
    inverseDifference = np.zeros(p)
    maximalCorrelationCoefficient = np.zeros(p)
    maximumProbability = np.zeros(p)
    sumAverage = np.zeros(p)
    sumEntropy = np.zeros(p)
    sumOfSquaresVariance = np.zeros(p)
    sumVariance = np.zeros(p)
    
    # 중간 계산용 배열들
    glcmMean = np.zeros(p)
    uX = np.zeros(p)
    uY = np.zeros(p)
    sX = np.zeros(p)
    sY = np.zeros(p)
    pX = np.zeros((nGrayLevels, p))
    pY = np.zeros((nGrayLevels, p))
    pXplusY = np.zeros((2 * nGrayLevels - 1, p))
    pXminusY = np.zeros((nGrayLevels, p))
    HX = np.zeros(p)
    HY = np.zeros(p)
    HXY1 = np.zeros(p)
    HXY2 = np.zeros(p)
    
    # 인덱스 생성 (MATLAB은 1-indexed이므로 +1)
    I_arr, J_arr = np.indices((nGrayLevels, nGrayLevels))
    I_arr = (I_arr.flatten() + 1).astype(np.float64)
    J_arr = (J_arr.flatten() + 1).astype(np.float64)
    nI = I_arr / nGrayLevels
    nJ = J_arr / nGrayLevels

    eps_val = np.finfo(float).eps

    for k in range(p):
        currentGLCM = glcm[:, :, k]
        current_flat = currentGLCM.flatten()
        glcmMean[k] = np.mean(currentGLCM)
        uX[k] = np.sum(nI * current_flat) * dA
        uY[k] = np.sum(nJ * current_flat) * dA
        sX[k] = np.sum(((nI - uX[k]) ** 2) * current_flat) * dA
        sY[k] = np.sum(((nJ - uY[k]) ** 2) * current_flat) * dA

        # pXplusY: 각 대각선 (행+열의 합 = s)에 대해 합산
        for s in range(2, 2 * nGrayLevels + 1):
            idx = np.where((I_arr + J_arr) == s)[0]
            pXplusY[s - 2, k] = np.sum(current_flat[idx]) * dkdiag

        # pXminusY: 각 차이(행과 열의 차의 절대값)에 대해 합산
        for d in range(nGrayLevels):
            idx = np.where(np.abs(I_arr - J_arr) == d)[0]
            pXminusY[d, k] = np.sum(current_flat[idx]) * dXminusY

        # pX, pY: 행합과 열합
        pX[:, k] = np.sum(currentGLCM, axis=1) * dL
        pY[:, k] = np.sum(currentGLCM, axis=0) * dL

        # 로그 계산 시 0인 값은 1로 대체(로그 1 = 0)
        HX[k] = -np.sum(pX[:, k] * np.log(np.where(pX[:, k] > 0, pX[:, k], 1))) * dL
        HY[k] = -np.sum(pY[:, k] * np.log(np.where(pY[:, k] > 0, pY[:, k], 1))) * dL

        product = pX[(I_arr - 1).astype(int), k] * pY[(J_arr - 1).astype(int), k]
        HXY1[k] = -np.sum(current_flat * np.log(np.where(product > 0, product, 1))) * dA
        HXY2[k] = -np.sum(product * np.log(np.where(product > 0, product, 1))) * dA

        # Haralick 피처 계산
        energy[k] = np.sum(current_flat ** 2) * dA
        contrast[k] = np.sum(((nI - nJ) ** 2) * current_flat) * dA
        
        autoCorr = np.sum(nI * nJ * current_flat) * dA
        autoCorrelation[k] = autoCorr
        
        if sX[k] < eps_val or sY[k] < eps_val:
            correlation[k] = np.clip(autoCorr - uX[k] * uY[k], -1, 1)
        else:
            correlation[k] = (autoCorr - uX[k] * uY[k]) / np.sqrt(sX[k] * sY[k])
        
        sumOfSquaresVariance[k] = np.sum(current_flat * ((nI - uX[k]) ** 2)) * dA
        
        homogeneity[k] = np.sum(current_flat / (1 + homogeneityConstant * ((nI - nJ) ** 2))) * dA

        sumIndices = np.arange(2, 2 * nGrayLevels + 1)
        factor = (2 * (sumIndices - 1)) / (2 * nGrayLevels - 1)
        sumAverage[k] = np.sum(factor * pXplusY[sumIndices - 2, k]) * dXplusY
        sumVariance[k] = np.sum(((factor - sumAverage[k]) ** 2) * pXplusY[sumIndices - 2, k]) * dXplusY
        sumEntropy[k] = -np.sum(pXplusY[sumIndices - 2, k] * 
                                np.log(np.where(pXplusY[sumIndices - 2, k] > 0, pXplusY[sumIndices - 2, k], 1))) * dXplusY
        
        entropy_arr[k] = -np.sum(current_flat * np.log(np.where(current_flat > 0, current_flat, 1))) * dA
        
        idx2 = np.arange(nGrayLevels)
        diff_avg = np.sum(((idx2 + 1) / nGrayLevels) * pXminusY[:, k]) * dXminusY
        differenceAverage[k] = diff_avg
        differenceVariance[k] = np.sum((((idx2 + 1) / nGrayLevels) - diff_avg) ** 2 * pXminusY[:, k]) * dXminusY
        differenceEntropy[k] = -np.sum(pXminusY[:, k] * 
                                      np.log(np.where(pXminusY[:, k] > 0, pXminusY[:, k], 1))) * dXminusY
        
        if max(HX[k], HY[k]) > 0:
            informationMeasureOfCorrelation1[k] = (entropy_arr[k] - HXY1[k]) / max(HX[k], HY[k])
        else:
            informationMeasureOfCorrelation1[k] = 0
        informationMeasureOfCorrelation2[k] = np.sqrt(1 - np.exp(-2 * (HXY2[k] - entropy_arr[k])))
        
        # Maximal Correlation Coefficient 계산
        P_mat = currentGLCM
        pX_ = pX[:, k].copy()
        if np.any(pX_ < eps_val):
            pX_ = pX_ + eps_val
            pX_ = pX_ / (np.sum(pX_) * dL)
        pY_ = pY[:, k].copy()
        if np.any(pY_ < eps_val):
            pY_ = pY_ + eps_val
            pY_ = pY_ / (np.sum(pY_) * dL)
        
        Q = np.zeros((nGrayLevels, nGrayLevels))
        for i in range(nGrayLevels):
            Pi = P_mat[i, :]
            pXi = pX_[i]
            for j in range(nGrayLevels):
                Pj = P_mat[j, :]
                denom = pXi * pY_
                # 분모의 0 또는 매우 작은 값을 eps_val로 대체
                denom = np.where(denom < eps_val, eps_val, denom)
                Q[i, j] = dA * np.sum((Pi * Pj) / denom)
        if np.any(np.isinf(Q)) or np.any(np.isnan(Q)):
            e2 = np.nan
        else:
            try:
                eigvals = np.linalg.eigvals(Q)
            except Exception:
                eigvals = np.array([np.nan])
            eigvals = np.real(eigvals)
            eigvals_sorted = np.sort(eigvals)[::-1]
            e2 = eigvals_sorted[1] if eigvals_sorted.size >= 2 else np.nan
        maximalCorrelationCoefficient[k] = e2
        
        dissimilarity[k] = np.sum(np.abs(nI - nJ) * current_flat) * dA
        clusterShade[k] = np.sum(((nI + nJ - uX[k] - uY[k]) ** 3) * current_flat) * dA
        clusterProminence[k] = np.sum(((nI + nJ - uX[k] - uY[k]) ** 4) * current_flat) * dA
        maximumProbability[k] = np.max(currentGLCM)
        inverseDifference[k] = np.sum(current_flat / (1 + inverseDifferenceConstant * np.abs(nI - nJ))) * dA

    out = {
        'autoCorrelation': autoCorrelation,
        'clusterProminence': clusterProminence,
        'clusterShade': clusterShade,
        'contrast': contrast,
        'correlation': correlation,
        'differenceAverage': differenceAverage,
        'differenceEntropy': differenceEntropy,
        'differenceVariance': differenceVariance,
        'dissimilarity': dissimilarity,
        'energy': energy,
        'entropy': entropy_arr,
        'homogeneity': homogeneity,
        'informationMeasureOfCorrelation1': informationMeasureOfCorrelation1,
        'informationMeasureOfCorrelation2': informationMeasureOfCorrelation2,
        'inverseDifference': inverseDifference,
        'maximalCorrelationCoefficient': maximalCorrelationCoefficient,
        'maximumProbability': maximumProbability,
        'sumAverage': sumAverage,
        'sumEntropy': sumEntropy,
        'sumOfSquaresVariance': sumOfSquaresVariance,
        'sumVariance': sumVariance
    }
    return out
def GLCMFeaturesInvariant2(glcm, homogeneityConstant=1, inverseDifferenceConstant=1):
    """
    정규화된 GLCM으로부터 21개의 Haralick 피처를 계산한다.
    
    Parameters
    ----------
    glcm : numpy.ndarray
        2D 또는 3D 배열. 2D면 단일 GLCM, 3D면 각 슬라이스가 하나의 GLCM을 의미.
    homogeneityConstant : float, optional
        동질성 피처에 사용되는 상수 (기본값 1).
    inverseDifferenceConstant : float, optional
        역차분 피처에 사용되는 상수 (기본값 1).
    
    Returns
    -------
    out : dict
        계산된 피처들을 키(key)와 1D numpy 배열의 값으로 담은 딕셔너리.
    """
    # GLCM이 2D면 3D로 확장
    if glcm.ndim == 2:
        glcm = glcm[:, :, np.newaxis]
    elif glcm.ndim != 3:
        raise ValueError("GLCM은 2D 또는 3D numpy 배열이어야 합니다.")
        
    nLevels, nLevels2, numGLCM = glcm.shape
    if nLevels != nLevels2:
        raise ValueError("각 GLCM은 정방행렬(NumLevels x NumLevels)이어야 합니다.")
    
    # 결과 피처 배열 초기화
    autoCorrelation                = np.zeros(numGLCM)
    clusterProminence              = np.zeros(numGLCM)
    clusterShade                   = np.zeros(numGLCM)
    contrast                       = np.zeros(numGLCM)
    correlation                    = np.zeros(numGLCM)
    differenceAverage              = np.zeros(numGLCM)
    differenceEntropy              = np.zeros(numGLCM)
    differenceVariance             = np.zeros(numGLCM)
    dissimilarity                  = np.zeros(numGLCM)
    energy                         = np.zeros(numGLCM)
    entropy_arr                    = np.zeros(numGLCM)
    homogeneity                    = np.zeros(numGLCM)
    informationMeasureOfCorrelation1 = np.zeros(numGLCM)
    informationMeasureOfCorrelation2 = np.zeros(numGLCM)
    inverseDifference              = np.zeros(numGLCM)
    maximalCorrelationCoefficient  = np.zeros(numGLCM)
    maximumProbability             = np.zeros(numGLCM)
    sumAverage                     = np.zeros(numGLCM)
    sumEntropy                     = np.zeros(numGLCM)
    sumOfSquaresVariance           = np.zeros(numGLCM)
    sumVariance                    = np.zeros(numGLCM)
    
    eps = np.finfo(float).eps  # 수치적 안정성을 위한 작은 값
    
    # 인덱스 생성: I_arr, J_arr는 1부터 nLevels까지의 값을 가지며, 정규화(nI, nJ)는 0~1 범위
    I_arr, J_arr = np.indices((nLevels, nLevels))
    I_arr = I_arr.astype(np.float64) + 1
    J_arr = J_arr.astype(np.float64) + 1
    nI = I_arr / nLevels
    nJ = J_arr / nLevels
    
    # 행, 열에 대한 1D 인덱스 (정규화된 값)
    rows = np.arange(1, nLevels+1) / nLevels
    cols = np.arange(1, nLevels+1) / nLevels
    
    for k in range(numGLCM):
        # 현재 슬라이스 정규화 (총합 1)
        currentGLCM = glcm[:, :, k].astype(np.float64).copy()
        total = np.sum(currentGLCM)
        if total > 0:
            currentGLCM /= total
        
        # 마진 확률 분포
        p_x = np.sum(currentGLCM, axis=1)  # 각 행의 합
        p_y = np.sum(currentGLCM, axis=0)  # 각 열의 합
        
        # Joint 엔트로피: HXY = - sum_{i,j} p(i,j)*log(p(i,j))
        HXY = -np.sum(currentGLCM * np.log(np.where(currentGLCM > 0, currentGLCM, 1)))
        entropy_arr[k] = HXY
        
        # 마진 엔트로피 HX, HY
        HX = -np.sum(p_x * np.log(np.where(p_x > 0, p_x, 1)))
        HY = -np.sum(p_y * np.log(np.where(p_y > 0, p_y, 1)))
        
        # HXY1 = - sum_{i,j} p(i,j)*log(p_x(i)*p_y(j))
        p_x_y = np.outer(p_x, p_y)
        HXY1 = -np.sum(currentGLCM * np.log(np.where(p_x_y > 0, p_x_y, 1)))
        
        # HXY2 = - sum_{i,j} p_x(i)*p_y(j)*log(p_x(i)*p_y(j))
        HXY2 = -np.sum(p_x_y * np.log(np.where(p_x_y > 0, p_x_y, 1)))
        
        # 정보 척도 IMC1, IMC2
        if max(HX, HY) > 0:
            informationMeasureOfCorrelation1[k] = (HXY - HXY1) / max(HX, HY)
        else:
            informationMeasureOfCorrelation1[k] = 0
        diff = HXY2 - HXY
        if diff < 0:
            diff = 0
        informationMeasureOfCorrelation2[k] = np.sqrt(1 - np.exp(-2 * diff))
        
        # Auto-correlation: sum(nI * nJ * p(i,j))
        autoCorrelation[k] = np.sum(nI * nJ * currentGLCM)
        
        # Contrast: sum((nI - nJ)^2 * p(i,j))
        contrast[k] = np.sum((nI - nJ)**2 * currentGLCM)
        
        # Correlation: (sum(nI*nJ*p) - mu_x*mu_y) / (sigma_x * sigma_y)
        mu_x = np.sum(rows * p_x)
        mu_y = np.sum(cols * p_y)
        sigma_x = np.sqrt(np.sum(((rows - mu_x)**2) * p_x))
        sigma_y = np.sqrt(np.sum(((cols - mu_y)**2) * p_y))
        if sigma_x < eps or sigma_y < eps:
            correlation[k] = np.clip(autoCorrelation[k] - mu_x * mu_y, -1, 1)
        else:
            correlation[k] = (autoCorrelation[k] - mu_x * mu_y) / (sigma_x * sigma_y)
        
        # Sum of Squares Variance: sum((nI - mu_x)^2 * p(i,j))
        sumOfSquaresVariance[k] = np.sum((nI - mu_x)**2 * currentGLCM)
        
        # Homogeneity: sum(p(i,j) / (1 + homogeneityConstant*(nI - nJ)^2))
        homogeneity[k] = np.sum(currentGLCM / (1 + homogeneityConstant * (nI - nJ)**2))
        
        # Inverse Difference: sum(p(i,j) / (1 + inverseDifferenceConstant*|nI - nJ|))
        inverseDifference[k] = np.sum(currentGLCM / (1 + inverseDifferenceConstant * np.abs(nI - nJ)))
        
        # Dissimilarity: sum(|nI - nJ| * p(i,j))
        dissimilarity[k] = np.sum(np.abs(nI - nJ) * currentGLCM)
        
        # Maximum Probability: max(p(i,j))
        maximumProbability[k] = np.max(currentGLCM)
        
        # Sum Average, Sum Entropy, Sum Variance (p_x+y: 합 인덱스 확률)
        pXplusY = np.zeros(2 * nLevels - 1)
        for s in range(2, 2 * nLevels + 1):
            mask = (I_arr + J_arr) == s
            pXplusY[s - 2] = np.sum(currentGLCM[mask])
        sumIndices = np.arange(2, 2 * nLevels + 1)
        # factor는 원래 (2*(s-1))/(2*nLevels - 1)로 정의됨
        factor = (2 * (sumIndices - 1)) / (2 * nLevels - 1)
        sumAverage[k] = np.sum(factor * pXplusY)
        sumEntropy[k] = -np.sum(pXplusY * np.log(np.where(pXplusY > 0, pXplusY, 1)))
        sumVariance[k] = np.sum(((factor - sumAverage[k])**2) * pXplusY)
        
        # Difference features: p_x-y: 차이 인덱스 확률
        pXminusY = np.zeros(nLevels)
        for d in range(nLevels):
            mask = np.abs(I_arr - J_arr) == d
            pXminusY[d] = np.sum(currentGLCM[mask])
        diff_vals = np.arange(1, nLevels + 1) / nLevels
        differenceAverage[k] = np.sum(diff_vals * pXminusY)
        differenceVariance[k] = np.sum(((diff_vals - differenceAverage[k])**2) * pXminusY)
        differenceEntropy[k] = -np.sum(pXminusY * np.log(np.where(pXminusY > 0, pXminusY, 1)))
        
        # Cluster Shade & Cluster Prominence:
        # ((i+j)/nLevels - (mu_x+mu_y))의 3제곱, 4제곱 가중치
        clusterShade[k] = np.sum((((I_arr + J_arr) / nLevels) - (mu_x + mu_y))**3 * currentGLCM)
        clusterProminence[k] = np.sum((((I_arr + J_arr) / nLevels) - (mu_x + mu_y))**4 * currentGLCM)
        
        # Maximal Correlation Coefficient 계산
        P_mat = currentGLCM
        pX_ = p_x.copy()
        if np.any(pX_ < eps):
            pX_ = pX_ + eps
            pX_ = pX_ / np.sum(pX_)
        pY_ = p_y.copy()
        if np.any(pY_ < eps):
            pY_ = pY_ + eps
            pY_ = pY_ / np.sum(pY_)
        Q = np.zeros((nLevels, nLevels))
        for i in range(nLevels):
            for j in range(nLevels):
                denom = pX_[i] * pY_
                denom = np.where(denom < eps, eps, denom)
                Q[i, j] = np.sum((P_mat[i, :] * P_mat[j, :]) / denom)
        try:
            eigvals = np.linalg.eigvals(Q)
            eigvals = np.real(eigvals)
            eigvals_sorted = np.sort(eigvals)[::-1]
            if eigvals_sorted.size >= 2:
                maximalCorrelationCoefficient[k] = eigvals_sorted[1]
            else:
                maximalCorrelationCoefficient[k] = np.nan
        except Exception:
            maximalCorrelationCoefficient[k] = np.nan
        
        # Energy: sum(p(i,j)^2)
        energy[k] = np.sum(currentGLCM**2)
    
    out = {
        'autoCorrelation': autoCorrelation,
        'clusterProminence': clusterProminence,
        'clusterShade': clusterShade,
        'contrast': contrast,
        'correlation': correlation,
        'differenceAverage': differenceAverage,
        'differenceEntropy': differenceEntropy,
        'differenceVariance': differenceVariance,
        'dissimilarity': dissimilarity,
        'energy': energy,
        'entropy': entropy_arr,
        'homogeneity': homogeneity,
        'informationMeasureOfCorrelation1': informationMeasureOfCorrelation1,
        'informationMeasureOfCorrelation2': informationMeasureOfCorrelation2,
        'inverseDifference': inverseDifference,
        'maximalCorrelationCoefficient': maximalCorrelationCoefficient,
        'maximumProbability': maximumProbability,
        'sumAverage': sumAverage,
        'sumEntropy': sumEntropy,
        'sumOfSquaresVariance': sumOfSquaresVariance,
        'sumVariance': sumVariance
    }
    
    return out
def glcm_feature(image, levels=64):
    image = (image - np.min(image)) / (np.max(image) - np.min(image))  # 0~1 정규화
    image = (image * (levels-1)).astype(np.uint8)  # 0~levels-1 범위의 정수로 변환

    distance = [1]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]

    glcm = graycomatrix(image, distances=distance, angles=angles, 
                        levels=levels, symmetric=True, normed=False)
    glcm = np.squeeze(glcm, axis=2)
    return glcm    

def compute_relative_texture_distance(gt_image, pred_image):
    """
    GT 이미지와 예측된 이미지 간의 Relative Texture Feature Distance 계산
    """
    gt_glcm = glcm_feature(gt_image)
    pred_glcm = glcm_feature(pred_image)

    gt_features = GLCMFeaturesInvariant2(gt_glcm)
    gt_features_invariance = {key: value.mean() for key, value in gt_features.items()}
    
    pred_features = GLCMFeaturesInvariant2(pred_glcm)
    pred_features_invariance = {key: value.mean() for key, value in pred_features.items()}
    
    FEATURES = gt_features_invariance.keys()
    # 상대적 거리 계산: | GT - Prediction | / GT
    relative_distances = {feature: abs(gt_features_invariance[feature] - pred_features_invariance[feature]) / (gt_features_invariance[feature] + 1e-8)
                          for feature in FEATURES}
    return relative_distances, gt_features_invariance, pred_features_invariance

    
class ImageQualityEvaluator():
    def __init__(self):
        self.FOVmask = self.get_ring_mask

    @property
    def get_ring_mask(self):
        #mask = np.zeros((512, 512), dtype=int)
        mask_diam = 470 # in mm
        mask_diam_pix = mask_diam/(400/512) # in pixels
        indice = np.indices((512, 512))
        mask = (indice[0]-255.5)**2 + (indice[1]-255.5)**2 < mask_diam_pix**2/4.
        return mask

    def remove_artifact(self, gt, pred, gt_metal_mask):
        myrecon = copy.deepcopy(pred)
        gtrecon = copy.deepcopy(gt)
            
        myrecon += 1000
        gtrecon += 1000
        gt_metal_mask = gt_metal_mask > 0.5

        myrecon[gt_metal_mask] = 0
        gtrecon[gt_metal_mask] = 0
        return gtrecon, myrecon
    
    def normalize_image(self, gtrecon, myrecon, anatomy):
        min2 = -2000
        max2 = 6000 
        gtrecon = (gtrecon-min2)/(max2-min2)
        myrecon = (myrecon-min2)/(max2-min2)

        gtrecon = np.clip(gtrecon, 0, 1)[:,:,0]
        myrecon = np.clip(myrecon, 0, 1)[:,:,0]
        
        # only body has max FOV limit
        if anatomy == 'body':
            myrecon[self.FOVmask<0.5] = 0
            gtrecon[self.FOVmask<0.5] = 0
        return gtrecon, myrecon
    
    def calculate_metrics(self, gtrecon, myrecon, data_range=1):
        psnr = SKpsnr(gtrecon, myrecon, data_range=1)
        ssim = SKssim(gtrecon, myrecon, win_size=11, data_range=1, gaussian_weights=True)           
        nrmse = SKnrmse(gtrecon, myrecon)
        rmse = np.sqrt(np.mean((gtrecon-myrecon)**2))
        # relative_texture_distance = compute_relative_texture_distance(gtrecon.squeeze(), myrecon.squeeze())

        return psnr, ssim, nrmse#, relative_texture_distance
    
    def __call__(self, gt, pred, gt_metal_mask, anatomy):
        """지정된 경로의 Ground Truth 및 예측 이미지에 대해 PSNR, SSIM 및 RMSE 값을 계산합니다.

        이 메서드는 입력으로 주어진 두 이미지 파일 경로에서 이미지를 읽고, 전처리를 수행한 후,
        PSNR(Peak Signal-to-Noise Ratio), SSIM(Structural Similarity Index Measure) 및 RMSE(Root Mean Square Error)
        지표를 계산하여 반환합니다. 이미지는 metal artifact가 있는 영역을 제외하고 평가됩니다.
        'body'가 경로에 포함된 경우, 이미지의 최대 FOV(Field of View) 한계를 적용합니다.

        Args:
            gt_path (str): Ground Truth 이미지의 파일 경로.
            pred_path (str): 예측 이미지의 파일 경로.

        Returns:
            tuple:
                float: 계산된 PSNR 값.
                float: 계산된 SSIM 값.
                float: 계산된 RMSE 값.

        Raises:
            FileNotFoundError: 지정된 경로의 파일이 존재하지 않을 때 발생.
            IOError: 파일 읽기 중 문제가 발생했을 때 발생.
        """
        assert anatomy in ['body', 'head']
        
        try:
            gtrecon, myrecon = self.remove_artifact(gt, pred, gt_metal_mask)
            gtrecon, myrecon = self.normalize_image(gtrecon, myrecon, anatomy)
            # psnr, ssim, nrmse, relative_texture_distance = self.calculate_metrics(gtrecon, myrecon)
            psnr, ssim, nrmse = self.calculate_metrics(gtrecon, myrecon)
        except Exception as e:
            raise Exception(f"Error during metric calculation: {e}")
        return psnr, ssim, nrmse
    
import piq
def to_torch_tensor(img: np.ndarray) -> torch.Tensor:
    """
    NumPy 이미지를 PyTorch 텐서로 변환.
    - 입력 img의 shape이 (H, W)이면 => (1, 1, H, W)
    - 입력 img의 shape이 (H, W, C)이면 => (1, C, H, W)
    """
    if img.ndim == 2:
        img_tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    elif img.ndim == 3:
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]
    else:
        raise ValueError(f"Unsupported image shape: {img.shape}")
    return img_tensor.float()

class ImageQualityEvaluator2():
    """
    A class to evaluate the quality of reconstructed medical images using various metrics.

    This class calculates image quality metrics such as PSNR, SSIM, RMSE, MS-SSIM, GMSD, FSIM, and VIF.
    It also considers anatomical contexts ('body' or 'head') and excludes regions with metal artifacts during evaluation.
    The class incorporates a field-of-view (FOV) mask for limiting evaluation to a specific area.

    Attributes:
        FOVmask (np.ndarray): A binary mask defining the field of view (FOV) for 'body' anatomy.
    
    Methods:
        __call__(gt, pred, gt_metal_mask, anatomy):
            Computes various quality metrics for the given ground truth and predicted images.

    Examples:
        evaluator = ImageQualityEvaluator2()
        metrics = evaluator(gt_image, pred_image, metal_mask, anatomy='body')
        print(metrics)
    """

    def __init__(self):
        self.FOVmask = self.get_ring_mask

    @property
    def get_ring_mask(self):
        #mask = np.zeros((512, 512), dtype=int)
        mask_diam = 470 # in mm
        mask_diam_pix = mask_diam/(400/512) # in pixels
        indice = np.indices((512, 512))
        mask = (indice[0]-255.5)**2 + (indice[1]-255.5)**2 < mask_diam_pix**2/4.
        return mask

    def _to_torch_tensor(self, img: np.ndarray) -> torch.Tensor:
        """
        NumPy 이미지를 PyTorch 텐서로 변환합니다.
        """
        if img.ndim == 2:  # 흑백 이미지 (H, W)
            img_tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        else:
            raise ValueError(f"Unsupported image shape: {img.shape}")
        return img_tensor.float()

    def __call__(self, gt, pred, gt_metal_mask, anatomy):
        """지정된 경로의 Ground Truth 및 예측 이미지에 대해 PSNR, SSIM 및 RMSE 값을 계산합니다.

        이 메서드는 입력으로 주어진 두 이미지 파일 경로에서 이미지를 읽고, 전처리를 수행한 후,
        PSNR(Peak Signal-to-Noise Ratio), SSIM(Structural Similarity Index Measure) 및 RMSE(Root Mean Square Error)
        지표를 계산하여 반환합니다. 이미지는 metal artifact가 있는 영역을 제외하고 평가됩니다.
        'body'가 경로에 포함된 경우, 이미지의 최대 FOV(Field of View) 한계를 적용합니다.

        Args:
            gt_path (str): Ground Truth 이미지의 파일 경로.
            pred_path (str): 예측 이미지의 파일 경로.

        Returns:
            tuple:
                float: 계산된 PSNR 값.
                float: 계산된 SSIM 값.
                float: 계산된 RMSE 값.

        Raises:
            FileNotFoundError: 지정된 경로의 파일이 존재하지 않을 때 발생.
            IOError: 파일 읽기 중 문제가 발생했을 때 발생.
        """
        assert anatomy in ['body', 'head']
        
        # Convert torch tensors to numpy if needed
        import torch
        if torch.is_tensor(pred):
            pred = pred.detach().cpu().numpy()
        if torch.is_tensor(gt):
            gt = gt.detach().cpu().numpy()
        if torch.is_tensor(gt_metal_mask):
            gt_metal_mask = gt_metal_mask.detach().cpu().numpy()
        
        # Squeeze to remove batch and channel dimensions if needed
        pred = np.squeeze(pred)
        gt = np.squeeze(gt)
        gt_metal_mask = np.squeeze(gt_metal_mask)
        
        myrecon = copy.deepcopy(pred)
        # myrecon = transform_image_unit_mm_to_HU(myrecon)
        gtrecon = copy.deepcopy(gt)
            
        myrecon += 1000
        gtrecon += 1000
        gt_metal_mask = gt_metal_mask > 0.5

        try:           
            # Calculate PSNR
            myrecon[gt_metal_mask] = 0
            gtrecon[gt_metal_mask] = 0
            min2 = -2000
            max2 = 6000 
            gtrecon = (gtrecon-min2)/(max2-min2)
            myrecon = (myrecon-min2)/(max2-min2)
            data_range = 1
            gtrecon = np.clip(gtrecon, 0, 1)
            myrecon = np.clip(myrecon, 0, 1)
            
            # Ensure 2D arrays
            if gtrecon.ndim == 3:
                gtrecon = gtrecon[:,:,0]
            if myrecon.ndim == 3:
                myrecon = myrecon[:,:,0]
            
            # only body has max FOV limit
            if anatomy == 'body':
                myrecon[self.FOVmask<0.5] = 0
                gtrecon[self.FOVmask<0.5] = 0

            psnr = SKpsnr(gtrecon, myrecon, data_range=data_range)
            ssim = SKssim(gtrecon, myrecon, win_size=11, data_range=data_range, gaussian_weights=True)           
            nrmse = SKnrmse(gtrecon, myrecon)
            rmse = np.sqrt(np.mean((gtrecon-myrecon)**2))
            
            # PyTorch 텐서 변환
            gt_tensor = self._to_torch_tensor(gtrecon)
            pred_tensor = self._to_torch_tensor(myrecon)

            # 추가 품질 지표 계산
            ms_ssim = piq.multi_scale_ssim(gt_tensor, pred_tensor, data_range=data_range).item()
            gmsd = piq.gmsd(gt_tensor, pred_tensor, data_range=data_range).item()
            fsim = piq.fsim(gt_tensor, pred_tensor, data_range=data_range, chromatic=False).item()
            vif = piq.vif_p(gt_tensor, pred_tensor, data_range=data_range).item()

            
        except Exception as e:
            raise Exception(f"Error during metric calculation: {e}")
        return {
            "PSNR": psnr,
            "SSIM": ssim,
            "NRMSE": nrmse,
            "RMSE": rmse,
            "MS-SSIM": ms_ssim,
            "GMSD": gmsd,
            "FSIM": fsim,
            "VIF": vif,
        }

# class ImageQualityEvaluator():
#     def __init__(self):
#         self.FOVmask = self.get_ring_mask

#     @property
#     def get_ring_mask(self):
#         #mask = np.zeros((512, 512), dtype=int)
#         mask_diam = 470 # in mm
#         mask_diam_pix = mask_diam/(400/512) # in pixels
#         indice = np.indices((512, 512))
#         mask = (indice[0]-255.5)**2 + (indice[1]-255.5)**2 < mask_diam_pix**2/4.
#         return mask


#     def __call__(self, gt_path, pred_path):
#         """지정된 경로의 Ground Truth 및 예측 이미지에 대해 PSNR, SSIM 및 RMSE 값을 계산합니다.

#         이 메서드는 입력으로 주어진 두 이미지 파일 경로에서 이미지를 읽고, 전처리를 수행한 후,
#         PSNR(Peak Signal-to-Noise Ratio), SSIM(Structural Similarity Index Measure) 및 RMSE(Root Mean Square Error)
#         지표를 계산하여 반환합니다. 이미지는 metal artifact가 있는 영역을 제외하고 평가됩니다.
#         'body'가 경로에 포함된 경우, 이미지의 최대 FOV(Field of View) 한계를 적용합니다.

#         Args:
#             gt_path (str): Ground Truth 이미지의 파일 경로.
#             pred_path (str): 예측 이미지의 파일 경로.

#         Returns:
#             tuple:
#                 float: 계산된 PSNR 값.
#                 float: 계산된 SSIM 값.
#                 float: 계산된 RMSE 값.

#         Raises:
#             FileNotFoundError: 지정된 경로의 파일이 존재하지 않을 때 발생.
#             IOError: 파일 읽기 중 문제가 발생했을 때 발생.
#         """
#         try:
#             if isinstance(pred_path, str):
#                 myrecon = xc.rawread(pred_path, [512, 512, 1], 'float')
#             else:
#                 myrecon = pred_path
#             # myrecon = transform_image_unit_mm_to_HU(myrecon)
#             if isinstance(gt_path, str):
#                 gtrecon = xc.rawread(gt_path, [512, 512, 1], 'float')
#             else:
#                 gtrecon = gt_path
                
#             ic(gtrecon.min(), gtrecon.max())
#             ic(myrecon.min(), myrecon.max())
#             myrecon += 1000
#             gtrecon += 1000
#         except FileNotFoundError as e:
#             raise FileNotFoundError(f"The specified file was not found: {e}")
#         except IOError as e:
#             raise IOError(f"Error occurred while reading the file: {e}")

#         try:
#             gt_metal_mask_path = gt_path.replace("Target", "Mask").replace("nometal", "metalonlymask")
#             gt_metal_mask = xc.rawread(gt_metal_mask_path, [512, 512, 1], 'float')
#             gt_metal_mask = gt_metal_mask > 0.5
#         except FileNotFoundError as e:
#             raise FileNotFoundError(f"Mask file not found: {e}")
#         except Exception as e:
#             raise Exception(f"Error processing mask file: {e}")

#         try:
#             ic(gtrecon.min(), gtrecon.max())
#             ic(myrecon.min(), myrecon.max())
            
#             # Calculate PSNR
#             myrecon[gt_metal_mask] = 0
#             gtrecon[gt_metal_mask] = 0
#             min2 = -2000
#             max2 = 6000
#             gtrecon = (gtrecon-min2)/(max2-min2)
#             myrecon = (myrecon-min2)/(max2-min2)
#             ic(gtrecon.min(), gtrecon.max())
#             ic(myrecon.min(), myrecon.max())

#             gtrecon = np.clip(gtrecon, 0, 1)[:,:,0]
#             myrecon = np.clip(myrecon, 0, 1)[:,:,0]
            
#             # only body has max FOV limit
#             if 'body' in gt_path:
#                 myrecon[self.FOVmask<0.5] = 0
#                 gtrecon[self.FOVmask<0.5] = 0
#             ic(gtrecon.min(), gtrecon.max())
#             ic(myrecon.min(), myrecon.max())

#             psnr = SKpsnr(gtrecon, myrecon, data_range=1)
#             ssim = SKssim(gtrecon, myrecon, win_size=11, data_range=1, gaussian_weights=True)
#             rmse = np.sqrt(np.mean((gtrecon-myrecon)**2))
#         except Exception as e:
#             raise Exception(f"Error during metric calculation: {e}")
#         return psnr, ssim, rmse
    

import torch
import torch.nn.functional as F
from torchmetrics.image import StructuralSimilarityIndexMeasure
ssim_measure = StructuralSimilarityIndexMeasure(data_range=1, kernel_size=11, gaussian_kernel=True, reduction='none')

class BatchImageQualityEvaluator():
    def __init__(self):
        self.FOVmask = self.get_ring_mask
    @property
    def get_ring_mask(self):
        # 마스크 생성
        mask_diam = 470 # in mm
        mask_diam_pix = mask_diam/(400/512) # in pixels
        indices = torch.stack(torch.meshgrid(torch.arange(512), torch.arange(512)), 0)
        mask = torch.sum((indices - 255.5) ** 2, dim=0) < (mask_diam_pix ** 2/4.)
        return mask.float()

    def __call__(self, gt_batch, pred_batch, gt_metal_mask_batch, anatomy):
        """
        Args:
            gt_batch (torch.Tensor): Ground Truth 배치 텐서, 크기는 (N, 512, 512).
            pred_batch (torch.Tensor): 예측 이미지 배치 텐서, 크기는 (N, 512, 512).

        Returns:
            tuple: PSNR 값, SSIM 값, RMSE 값 각각의 배치에 대한 평균
        """
        assert anatomy in ['head', 'body'], f"Invalid anatomy: {anatomy}"

        ssim_measure.to(gt_batch.device)
        self.FOVmask.to(gt_batch.device)
        
        # 값 조정
        gt_batch = gt_batch.clone()
        pred_batch = pred_batch.clone()
        
        gt_batch += 1000
        pred_batch += 1000
        gt_metal_mask_batch = gt_metal_mask_batch > 0.5
        
        # 마스크의 채널 차원을 gt_batch와 맞춤
        if gt_metal_mask_batch.shape[1] != gt_batch.shape[1]:
            gt_metal_mask_batch = gt_metal_mask_batch.expand_as(gt_batch)
        
        gt_batch[gt_metal_mask_batch] = 0
        pred_batch[gt_metal_mask_batch] = 0

        min_val, max_val = -2000, 6000
        gt_batch = torch.clamp((gt_batch - min_val) / (max_val - min_val), 0, 1)
        pred_batch = torch.clamp((pred_batch - min_val) / (max_val - min_val), 0, 1)

        # FOV 마스크 적용
        # gt_batch = gt_batch * self.FOVmask
        # pred_batch = pred_batch * self.FOVmask

        if 'body' == anatomy:
            fov_mask = self.FOVmask.to(gt_batch.device)
            gt_batch[:,:, fov_mask < 0.5] = 0
            pred_batch[:,:, fov_mask < 0.5] = 0
        # 계산
        mse = torch.mean((gt_batch - pred_batch) ** 2, dim=[2, 3])
        rmse = torch.sqrt(mse)
    
        # PSNR 계산
        data_range = 1
        psnr = 10 * torch.log10(data_range**2 / mse)

        ssim = ssim_measure(pred_batch, gt_batch)
        # return psnr.mean().item(), ssim.mean().item(), rmse.mean().item()
        metrics = {
    "PSNR": psnr.mean().item(),
    "SSIM": ssim.mean().item(),
    "RMSE": rmse.mean().item()
}
        return metrics

    
def min_std(preds, targets):
    """ radial direciton으로 std 계산

    Args:
        preds (torch.Tensor): The predicted outputs from the model.
        targets (torch.Tensor): The ground truth labels.

    Returns:
        torch.Tensor: The computed loss value.
    """
    preds = preds.squeeze() # n x 1 x h x w -> n x h x w
    

    # remove padding n x 1024 x 928 -> n x 1000 x 900
    preds = preds[:, 12:-12, 14:-14] # n x h x w -> n x h-24 x w-28
    # targets = targets.squeeze() # n x 1 x h x w -> n x h x w

    # sum over every width
    preds = preds.sum(dim=2) # n x h x w -> n x h
    # targets = targets.sum(dim=2) # n x h x w -> n x h
    
    # std for preds for each sample
    loss = preds.std(axis=1) # n x w -> n

    loss = torch.mean(loss)

    return loss


if __name__ == '__main__':
    # Example Usage (similar to MATLAB example)
    # Create a dummy GLCM for demonstration (replace with actual GLCM from graycomatrix equivalent in Python if needed)
    dummy_glcm = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]) # Example 2x2x2 GLCM
    # 4x4 GLCM 
    dummy_glcm = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
    # Calculate all invariant Haralick features
    all_features = glcm_features_invariant(dummy_glcm)
    print("All Features:", all_features)

    # Calculate energy and entropy only
    selected_features = glcm_features_invariant(dummy_glcm, features=['all'])

    print("\nSelected Features (Energy and Entropy):", selected_features)
    
    