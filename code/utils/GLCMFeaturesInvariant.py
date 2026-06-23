
import copy
import numpy as np
from skimage.feature import graycomatrix
from scipy.linalg import eig  # Corrected import: use eig instead of eigs
import warnings


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

    gt_features = GLCMFeaturesInvariant(gt_glcm)
    gt_features_invariance = {key: value.mean() for key, value in gt_features.items()}
    pred_features = GLCMFeaturesInvariant(pred_glcm)
    pred_features_invariance = {key: value.mean() for key, value in pred_features.items()}
    
    FEATURES = gt_features_invariance.keys()
    # 상대적 거리 계산: | GT - Prediction | / GT
    relative_distances = {feature: abs(gt_features_invariance[feature] - pred_features_invariance[feature]) / (gt_features_invariance[feature] + 1e-8)
                          for feature in FEATURES}
    return relative_distances
