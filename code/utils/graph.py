"""
최적의 gc를 찾은 이후 추가 시도 횟수를 증가시키고, 최대 추가 시도 횟수를 초과할 경우 종료
"""


from typing import List, Dict, Any
from typing_extensions import TypedDict
import numpy as np
import copy
import cv2
from utils.algorithm import IterativeReconstruction, Inpainting, merge_masked_prediction
from langgraph.graph import StateGraph, START, END
from utils.projection import filtering, fp, bp
from utils.algorithm import inference_classifier, inference_enhancement
from utils.models import SinogramInpainting, ImageClassifier, ImageEnhancement
from utils.ct import transform_image_unit_mm_to_HU
from utils.logging import log_info, log_debug, log_decision, log_param
class MAR_Workflow_State(TypedDict):
    """ 
    Represents the state of the Metal Artifact Reduction (MAR) workflow.

    Attributes:
        original_sinogram (np.ndarray): 원본 CT sinogram 데이터로, 금속 아티팩트가 포함된 상태
        original_image (np.ndarray): 원본 CT 이미지 데이터

        total_iterations (int): 전체 워크플로우에서 허용된 최대 반복 횟수
        current_iteration (int): 현재 진행 중인 반복 횟수
        
        total_additional_trial (int): GC(Ghost Correction)에서 통과하더라도 추가 시도할 수 있는 최대 횟수
        current_additional_trial (int): 현재 진행 중인 추가 시도 횟수

        ct_param (Dict[str, Any]): CT 스캐닝 및 복원과 관련된 다양한 매개변수 모음
        metal_striking_level (float): 금속 아티팩트의 심각도를 나타내는 정량적 지표으로 gc에서 생성된 값이 들어감
        
        reconstruction_params (Dict[str, float]): CT 재구성 과정에서 활용되는 매개변수 집합
            - image_mask_threshold: 금속 부위를 분할하기 위한 임계값
            - num_whole_iterations: 전체 프로세스에서 수행할 반복 횟수
            - kernel_size: 가우시안 블러링에서 사용되는 커널 크기
            - sigmaX: 가우시안 블러링의 표준 편차 값
            - ir_num_iterations: 반복적 재구성(iterative reconstruction)에서 수행할 반복 횟수
            - soft_thresholding: 소프트 임계값 지정 (노이즈 억제용)
            - metal_striking_threshold: 금속 아티팩트 심각도를 판단하기 위한 임계값

        inpainting_model (SinogramInpainting): 금속 제거 후 보간(inpainting)을 수행하는 모델
        classifier_model (ImageClassifier): 금속 아티팩트 여부를 판단하는 이미지 분류 모델

        metalart_image (np.ndarray): 금속 아티팩트 보정 후의 CT 이미지
        metalart_sinogram (np.ndarray): 금속 아티팩트 보정 후의 sinogram 데이터

        img_b (np.ndarray): 보정 후의 CT background 이미지
        img_m (np.ndarray): 보정 후 CT Metal 이미지
        
        gc_dict (Dict[str, float]): ground checking 관련 결과 저장용 딕셔너리
        best_params (Dict[str, float]): 최적의 재구성 매개변수 저장
        best_gc (Dict[str, float]): 최적의 GC 관련 지표 저장
        best_img_b (np.ndarray): 최적의 CT background 이미지
        
        anatomy (str): 스캔된 신체 부위 정보 (예: "head", "chest")
        
        additional_trial_decision (str): 추가 시도 여부를 결정하는 플래그 (예: "pass", "retry")
    """

    original_sinogram: np.ndarray
    original_image: np.ndarray

    total_iterations: int
    current_iteration: int
    total_additional_trial: int
    current_additional_trial: int
    ct_param: Dict[str, Any]
    metal_striking_level: float
    reconstruction_params: Dict[str, float]
    
    inpainting_model: SinogramInpainting
    classifier_model: ImageClassifier
    enhancement_model: ImageEnhancement

    metalart_image: np.ndarray
    metalart_sinogram: np.ndarray
    
    mask_i: np.ndarray
    mask_s: np.ndarray
    
    img_b: np.ndarray
    img_m: np.ndarray
    
    sino_b: np.ndarray
    
    gc_dict: Dict[str, float]    
    best_params: Dict[str, float]
    best_gc: Dict[str, float]
    best_img_b: np.ndarray
    best_img_b_original: np.ndarray
    enhanced_img_b: np.ndarray
    anatomy: str
    
    additional_trial_decision: str
    
class Reconstruction_Pipeline_State(TypedDict):
    """
    Represents the state of the Iterative Reconstruction (IR) subgraph.

    Attributes:
        original_sinogram (np.ndarray): 원본 CT sinogram 데이터로, 금속 아티팩트가 포함된 상태
        original_image (np.ndarray): 원본 CT 이미지 데이터

        metalart_sinogram (np.ndarray): 금속 아티팩트 보정 후의 sinogram 데이터
        anatomy (str): 스캔된 신체 부위 정보 (예: "head", "chest")
        
        ct_param (Dict[str, float]): CT 스캐닝 및 복원과 관련된 다양한 매개변수 모음

        mask_i (np.ndarray): 금속 마스크 이미지 (input domain)
        mask_s (np.ndarray): 금속 마스크 이미지 (sinogram domain)
        
        img_b (np.ndarray): 보정 후의 CT background 이미지
        img_m (np.ndarray): 보정 후 CT Metal 이미지 
        
        sino_b (np.ndarray): 보정 후의 sinogram background 데이터

        current_IR_iteration (int): 현재 진행 중인 반복 재구성(IR) 단계의 반복 횟수
        total_IR_iterations (int): 전체 IR 과정에서 수행할 반복 횟수
        
        inpainting_model (SinogramInpainting): 금속 제거 후 보간(inpainting)을 수행하는 모델
        classifier_model (ImageClassifier): 금속 아티팩트 여부를 판단하는 이미지 분류 모델

        reconstruction_params (Dict[str, float]): CT 재구성 과정에서 활용되는 매개변수 집합
        best_img_b (np.ndarray): 최적의 CT background 이미지
        
        total_additional_trial (int): GC(Ghost Correction)에서 통과하더라도 추가 시도할 수 있는 최대 횟수
        current_additional_trial (int): 현재 진행 중인 추가 시도 횟수

        gc_dict (Dict[str, float]): ground checking 관련 결과 저장용 딕셔너리

        image (np.ndarray): 현재 처리 중인 CT 이미지 데이터
        sinogram (np.ndarray): 현재 처리 중인 CT sinogram 데이터
    """
    original_sinogram: np.ndarray
    original_image: np.ndarray
    
    metalart_image: np.ndarray
    metalart_sinogram: np.ndarray
    anatomy: str
    ct_param: Dict[str, float]
    
    mask_i: np.ndarray
    mask_s: np.ndarray
    
    img_b: np.ndarray
    img_m: np.ndarray
    
    sino_b: np.ndarray

    
    current_IR_iteration: int
    total_IR_iterations: int
    
    inpainting_model: SinogramInpainting
    classifier_model: ImageClassifier
    enhancement_model: ImageEnhancement
    
    reconstruction_params: Dict[str, float]
    best_img_b: np.ndarray
    best_img_b_original: np.ndarray
    enhanced_img_b: np.ndarray
    
    total_additional_trial: int
    current_additional_trial: int
    gc_dict: Dict[str, float]
    image: np.ndarray
    sinogram: np.ndarray


def thresholding(state: Reconstruction_Pipeline_State) -> Reconstruction_Pipeline_State:
    """ metal을 찾기위한 image thresholding
    """
    if state['metalart_image'] is not None:
        metalart_image = state['metalart_image']
    else:
        metalart_image = state['original_image']
    image_mask_threshold = state['reconstruction_params']['image_mask_threshold']
        
    mask_i = copy.deepcopy(metalart_image)
    mask_i[mask_i<=image_mask_threshold] = 0
    mask_i[mask_i>image_mask_threshold] = 1
    
    return {"mask_i": mask_i}

def generate_mask_s(state: Reconstruction_Pipeline_State) -> Reconstruction_Pipeline_State:
    """ 앞서 수행된 mask_i로부터 mask_s생성
    """
    mask_i = state['mask_i']
    ct_param = state['ct_param']
    mask_s = fp(mask_i, ct_param).astype('float32')
    mask_s[mask_s!=0] = 1
    mask_s[mask_s==0] = 0
    
    return {"mask_s": mask_s}

def blurring(state: Reconstruction_Pipeline_State) -> Reconstruction_Pipeline_State:
    """ sinogram에 blurring을 적용해 메탈 부분을 약간 더 키우는 작업 수행
    """
    mask_s = state['mask_s']
    ksize = state['reconstruction_params']['kernel_size']
    sigmaX = state['reconstruction_params']['sigmaX']
        
    mask_s = cv2.GaussianBlur(mask_s, ksize=ksize, sigmaX=sigmaX, sigmaY=0)
    mask_s[mask_s>0.1] = 1
    mask_s[mask_s<=0.1] = 0

    return {"mask_s": mask_s}
    
def inpainting(state: Reconstruction_Pipeline_State) -> Reconstruction_Pipeline_State:
    metalart_sinogram = copy.deepcopy(state['original_sinogram'])
    # metalart_sinogram = state['metalart_sinogram']
    mask_s = state['mask_s']
    inpainting_model = state['inpainting_model']
    
    sinogram_without_metal = metalart_sinogram  * (1 - mask_s)
    
    sino_b = Inpainting(sinogram_without_metal, 
                        mask_s, inpainting_model)
    
    return {"sino_b": sino_b}
def generate_img_b(state: Reconstruction_Pipeline_State) -> Reconstruction_Pipeline_State:
    """ Inpainting """
    sino_b = state['sino_b']
    ct_param = state['ct_param']
    ct_param['filter'] = 'shepp-logan'
    
    sino_filt = filtering(sino_b, ct_param)
    img_b = bp(sino_filt, ct_param).astype('float32')        
    img_b = np.maximum(img_b, 0)
    
    return {"img_b": img_b}
def iterative_reconstruction(state: Reconstruction_Pipeline_State) -> Reconstruction_Pipeline_State:
    original_sinogram = state['original_sinogram']
    sino_b = state['sino_b']
    anatomy = state['anatomy']
    
    ir_num_iterations = state['reconstruction_params']['ir_num_iterations']
    soft_thresholding = state['reconstruction_params']['soft_thresholding']
    
    sino_m = original_sinogram - sino_b     
    ir = IterativeReconstruction(anatomy)
    img_m = ir.perform_iterative_reconstruction(sino_m, num_iterations=ir_num_iterations, soft_thresholding=soft_thresholding)
    
    state['reconstruction_params']['current_IR_iteration'] += 1
    return {"img_m": img_m,
            "reconstruction_params": state['reconstruction_params']}
    

def generate_result(state: Reconstruction_Pipeline_State) -> Reconstruction_Pipeline_State:
    """ 결과물을 생성하는 노드 """
    img_b = state['img_b']
    img_m = state['img_m']
    ct_param = state['ct_param']
    metalart_image = img_m + img_b
    
    metalart_sinogram = fp(metalart_image, ct_param).astype('float32')
    
    return {"metalart_sinogram": metalart_sinogram,
            "img_b": img_b,
            "img_m": img_m}

# edge
def check_iteration_completion(state: Reconstruction_Pipeline_State) -> str:
    """ 전체 횟수를 다 돌았는지 체크"""
    current_iteration = state['reconstruction_params']['current_IR_iteration']
    num_whole_iterations = state['reconstruction_params']['num_whole_iterations']
    if current_iteration < num_whole_iterations:
        return "go"
    else:
        return "finish"


def get_data(state: MAR_Workflow_State) -> MAR_Workflow_State:
    """ START NODE. Get data.
    """
    original_metalart_sinogram = state['original_sinogram']
    
    ct_param = state['ct_param']
    ct_param['filter'] = 'ram-lak'
    
    sino_filt = filtering(original_metalart_sinogram, ct_param)
    original_metalart_image = bp(sino_filt, ct_param).astype('float32')
    original_metalart_image = np.maximum(original_metalart_image, 0)
    log_debug(f"ori_metalart_image: intensity- max:{np.max(original_metalart_image):.4f}, min:{np.min(original_metalart_image):.4f}")
    
    # ori_metalart_image = reconstruction(ori_metalart_sinogram, 
    #                         anatomy=anatomy, 
    #                         kernelType='standard', 
    #                         unit='/mm')
    metalart_sinogram = copy.deepcopy(original_metalart_sinogram)
    
    state.update({"original_image": original_metalart_image,
                  "metalart_sinogram": metalart_sinogram})
    return state

def adjust_params(state: MAR_Workflow_State) -> MAR_Workflow_State:
    """adjust reconstruction parameters

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): state
    """
    
    log_info("파라미터 조정 중...", level=1)
    reconstruction_params = state['reconstruction_params']
    image_mask_threshold = reconstruction_params['image_mask_threshold'] 
    image_mask_threshold -= 0.01
    image_mask_threshold = round(image_mask_threshold, 2)
    
    current_IR_iteration = state['reconstruction_params']['current_IR_iteration']
    current_IR_iteration = 0
    
    current_iteration = state['current_iteration']
    current_iteration += 1
    
    reconstruction_params.update({"image_mask_threshold": image_mask_threshold})
    reconstruction_params.update({"current_IR_iteration": current_IR_iteration})
    state.update({"current_iteration": current_iteration,
                  "reconstruction_params": reconstruction_params,})
    return state
def ground_checking(state: MAR_Workflow_State) -> MAR_Workflow_State:
    """ground checking
    Args:
        state (dict): The current graph state
    Returns:
        state (dict): state
    """
    img_b = state["enhanced_img_b"]
    classifier_model = state['classifier_model']

    output = inference_classifier(img_b, classifier_model)
    output = output.item()
    gc_dict = state["gc_dict"]
    
    gc_dict["current_gc"] = output

    state.update({"metal_striking_level": output,
                  "gc_dict": gc_dict})
    return state

def enhancement(state: MAR_Workflow_State) -> MAR_Workflow_State:
    """CT Image enhancement
    Args:
        state (dict): The current graph state
    Returns:
        state (dict): state
    """
    enhancement_model = state.get('enhancement_model', None)
    if enhancement_model is not None:
        img_b = transform_image_unit_mm_to_HU(state["img_b"])
        
        enhanced_img_b = inference_enhancement(img_b, enhancement_model).squeeze()
        state.update({"enhanced_img_b":enhanced_img_b})
    else:
        state.update({"enhanced_img_b":None})
    # Put enhancement code
    return state

# def additional_trial(state: MAR_Workflow_State) -> MAR_Workflow_State:
#     """
def additional_trial(state: MAR_Workflow_State) -> MAR_Workflow_State:
    """ 최적의 GC를 찾았는지 판단하는 노드
    """
    gc_dict = state["gc_dict"]
    current_gc = gc_dict["current_gc"]
    prev_gc = gc_dict["prev_gc"]
    
    total_additional_trial = state["total_additional_trial"]
    current_additional_trial = state["current_additional_trial"]
    
    log_info(f"Ground Checking - current: {current_gc:.6f}, prev: {prev_gc:.6f}", level=1)
    # 기본값 설정
    additional_trial_decision = 'end'
    best_params = state["best_params"]
    best_gc = state["best_gc"]
    best_img_b = state["best_img_b"]
    best_img_b_original = state["best_img_b_original"]
    # 현재 GC가 이전 GC보다 개선되었거나, prev_gc가 초기값(np.inf)인 경우 최적값 갱신
    if prev_gc == np.inf or current_gc < prev_gc:
        log_param(f"최적값 갱신: {state['reconstruction_params']}", level=2)
        best_params = copy.deepcopy(state["reconstruction_params"])
        best_gc = current_gc
        best_img_b_original = state["img_b"]
        best_img_b = state["enhanced_img_b"]
        gc_dict["prev_gc"] = gc_dict["current_gc"]
        # current_additional_trial 초기화 (새로운 최적 값 발견)
        current_additional_trial = 0
        additional_trial_decision = 'go'  # 최적 값이 갱신되었으면 추가 시도 가능
    else:
        # 기존의 current_additional_trial 증가 (최적값 갱신 X)
        current_additional_trial += 1
        
        # 최대 추가 시도 횟수 이하이면 계속 진행
        if current_additional_trial < total_additional_trial:
            additional_trial_decision = 'go'
        else:
            additional_trial_decision = 'end'

    state.update({"best_params": best_params,
            "best_gc": best_gc,
            "current_additional_trial": current_additional_trial,
            "additional_trial_decision": additional_trial_decision,
            "gc_dict": gc_dict,
            "best_img_b": best_img_b,
            "best_img_b_original": best_img_b_original})
    return state

def decide_to_retry(state): # TODO: change function name
    """self-evaluation code
    Args:
        state (dict): The current graph state
    Returns:
        str: Binary decision for next node to call
    """
    output = state["metal_striking_level"]
    THRESHOLD = state["reconstruction_params"]["metal_striking_threshold"]
    if output < THRESHOLD:
        log_decision(f"금속 아티팩트 수준 낮음: {output:.6f}", level=1)
        return "good"
    else:
        log_decision(f"파라미터 변경 필요: {output:.6f}", level=1)
        return "bad"
            
def decide_to_finish(state):
    """전체 횟수를 다 돌았는지 체크하는 엣지
    Args:
        state (dict): The current graph state
    Returns:
        str: Binary decision for next node to call 
    """ 
    total_iterations = state["total_iterations"]
    current_iteration = state["current_iteration"]
    
    if current_iteration < total_iterations:
        log_decision(f"진행 중: {current_iteration}/{total_iterations}", level=1)
        return "go"
    else:
        log_decision("전체 반복 완료", level=1)
        return "finish"
def decide_to_additional_try(state):
    """ 최적의 GC에서 더 수행할지 판단하는 엣지
    """
    additional_trial_decision = state["additional_trial_decision"]
    if additional_trial_decision == 'go':
        return "go"
    else:
        return "finish"

def get_reconstruction_graph():
            
    reconstruction_workflow = StateGraph(Reconstruction_Pipeline_State)
    reconstruction_workflow.add_node('thresholding', thresholding)
    reconstruction_workflow.add_node('generate_mask_s', generate_mask_s)
    reconstruction_workflow.add_node('blurring', blurring)
    reconstruction_workflow.add_node('inpainting', inpainting)
    reconstruction_workflow.add_node('generate_img_b', generate_img_b)
    reconstruction_workflow.add_node('iterative_reconstruction', iterative_reconstruction)
    reconstruction_workflow.add_node('generate_result', generate_result)

    reconstruction_workflow.add_edge(START, "thresholding")
    reconstruction_workflow.add_edge("thresholding", "generate_mask_s")
    reconstruction_workflow.add_edge("generate_mask_s", "blurring")
    reconstruction_workflow.add_edge("blurring", "inpainting")
    reconstruction_workflow.add_edge("inpainting", "generate_img_b")
    reconstruction_workflow.add_edge("generate_img_b", "iterative_reconstruction")
    reconstruction_workflow.add_conditional_edges("iterative_reconstruction",
        check_iteration_completion,
        {
            "go": "thresholding",
            "finish": "generate_result"
        }
    )
    reconstruction_workflow.add_edge("generate_result", END)
    reconstruction_graph = reconstruction_workflow.compile()
    return reconstruction_graph

def get_workflow_graph(mode='experiment'):
    """
    MAR 워크플로우 그래프를 생성합니다.
    
    Args:
        mode (str): 'inference' 또는 'experiment'
            - 'inference': 조기 종료 방식 (빠름, 첫 번째 good 결과에서 종료)
            - 'experiment': 전체 탐색 방식 (느림, 최적값 탐색, best 추적)
    
    Returns:
        CompiledGraph: 컴파일된 Langgraph 워크플로우
    """
    if mode not in ['inference', 'experiment']:
        raise ValueError(f"mode는 'inference' 또는 'experiment'여야 합니다. 현재: {mode}")
    
    reconstruction_graph = get_reconstruction_graph()
    
    workflow = StateGraph(MAR_Workflow_State)

    # 공통 노드 추가
    workflow.add_node("get_data", get_data)
    workflow.add_node("reconstruction_pipeline", reconstruction_graph)
    workflow.add_node("adjust_params", adjust_params)
    workflow.add_node("ground_checking", ground_checking)
    workflow.add_node("enhancement", enhancement)

    # 공통 엣지
    workflow.add_edge(START, "get_data")
    workflow.add_edge("get_data", "reconstruction_pipeline")
    workflow.add_edge("reconstruction_pipeline", "enhancement")
    workflow.add_edge("enhancement", "ground_checking")

    if mode == 'inference':
        # Inference 모드: 조기 종료 (빠름)
        log_info(f"워크플로우 모드: INFERENCE (조기 종료)", level=1)
        workflow.add_conditional_edges("ground_checking",
            decide_to_retry,
            {
                "bad": "adjust_params",
                "good": END  # GC 좋으면 즉시 종료
            }
        )
        
    else:  # mode == 'experiment'
        # Experiment 모드: 최적값 탐색 (느림, best 추적)
        log_info(f"워크플로우 모드: EXPERIMENT (전체 탐색)", level=1)
        workflow.add_node("additional_trial", additional_trial)
        workflow.add_edge("ground_checking", "additional_trial")
        workflow.add_conditional_edges("additional_trial",
            decide_to_additional_try,
            {
                "go": "adjust_params",
                "finish": END
            }
        )

    # adjust_params 후 조건부 분기 (공통)
    workflow.add_conditional_edges("adjust_params",
        decide_to_finish,
        {
            "go": "get_data",
            "finish": END
        }
    )

    # Compile
    return workflow.compile()
