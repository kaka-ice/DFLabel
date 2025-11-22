import time
import cv2
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import numpy as np
import torch
import gc
import psutil  # 用于监控内存使用
import sys
from sam_modeling import ImageEncoderViT,MaskDecoder,PromptEncoder,Sam,TwoWayTransformer
from functools import partial
from collections import OrderedDict
import threading

from sam_predictor import SamPredictor

# 禁用梯度计算，大幅减少显存
torch.set_grad_enabled(False)


def _build_sam(
    encoder_embed_dim,
    encoder_depth,
    encoder_num_heads,
    encoder_global_attn_indexes,
    checkpoint=None,
):
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    sam = Sam(
        image_encoder=ImageEncoderViT(
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
        ),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )
    sam.eval()
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)
        sam.load_state_dict(state_dict)
    return sam

def build_sam_vit_b(checkpoint=None):
    return _build_sam(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        checkpoint=checkpoint,
    )


class SAMSingleton:
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SAMSingleton, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.sam = None
            self.predictor = None
            self.device = None
            self.current_image_path = None
            self.image_embedding_cache = {}  # 简单的缓存，只存当前图片
            self._initialized = True
            
            # 打印初始内存状态
            self.print_memory_status("初始化前")
    
    def print_memory_status(self, stage=""):
        """打印内存和显存使用情况"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            cached = torch.cuda.memory_reserved() / 1024**3
            print(f"{stage} - GPU显存: 已分配 {allocated:.2f}GB, 缓存 {cached:.2f}GB", file=sys.stderr)
        
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_gb = memory_info.rss / 1024**3
        print(f"{stage} - 系统内存: {memory_gb:.2f}GB", file=sys.stderr)
    
    def load_model(self):
        """加载模型，确保只加载一次"""
        if self.sam is not None:
            return True
            
        try:
            print("首次加载SAM模型...", file=sys.stderr)
            self.print_memory_status("加载模型前")
            
            # from sam_modeling import build_sam_vit_b
            
            # 获取模型路径
            script_dir = os.path.dirname(os.path.abspath(__file__))
            weight_path = os.path.join(script_dir, "sam_vit_b_01ec64.pth")
            
            # 加载模型到CPU
            self.sam = build_sam_vit_b(checkpoint=weight_path)
            
            # 选择设备
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"使用设备: {self.device}", file=sys.stderr)
            
            # 移到目标设备
            self.sam.to(self.device)
            self.sam.eval()
            
            # 创建predictor
            self.predictor = SamPredictor(self.sam)
            
            self.print_memory_status("加载模型后")
            print("SAM模型加载完成", file=sys.stderr)
            return True
            
        except Exception as e:
            print(f"模型加载失败: {str(e)}", file=sys.stderr)
            return False
    
    def init_image(self, img_path):
        """初始化图片"""
        try:
            if not self.load_model():
                return False
            
            self.print_memory_status(f"初始化图片 {os.path.basename(img_path)} 前")
            
            # 检查缓存
            if img_path in self.image_embedding_cache:
                print(f"使用缓存图片: {img_path}", file=sys.stderr)
                embedding_data = self.image_embedding_cache[img_path]
                
                self.predictor.reset_image()
                self.predictor.features = embedding_data['features']
                self.predictor.original_size = embedding_data['original_size']
                self.predictor.input_size = embedding_data['input_size']
                self.predictor.is_image_set = True
            else:
                # 加载并编码新图片
                print(f"编码新图片: {img_path}", file=sys.stderr)
                
                # 使用imdecode避免路径编码问题
                image = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
                if image is None:
                    print(f"无法加载图像: {img_path}", file=sys.stderr)
                    return False
                
                # 编码图片
                start_time = time.time()
                self.predictor.set_image(image)
                encoding_time = time.time() - start_time
                
                # 缓存编码结果（只缓存当前图片）
                self.image_embedding_cache.clear()  # 清空旧缓存
                embedding_data = {
                    'features': self.predictor.features,
                    'original_size': self.predictor.original_size,
                    'input_size': self.predictor.input_size
                }
                self.image_embedding_cache[img_path] = embedding_data
                
                print(f"图片编码耗时: {encoding_time:.2f}秒", file=sys.stderr)
            
            self.current_image_path = img_path
            self.print_memory_status(f"初始化图片 {os.path.basename(img_path)} 后")
            return True
            
        except Exception as e:
            print(f"图片初始化失败: {str(e)}", file=sys.stderr)
            return False
    
    def predict(self, points, labels):
        """预测掩码"""
        try:
            if self.predictor is None or not self.predictor.is_image_set:
                return None
            
            self.print_memory_status("预测前")
            
            input_points = np.array(points, dtype=np.float32)
            input_labels = np.array(labels, dtype=np.int32)
            
            start_time = time.time()
            
            # 单次预测，不进行refine
            masks, scores, logits = self.predictor.predict(
                point_coords=input_points,
                point_labels=input_labels,
                multimask_output=False  # 只输出一个mask
            )
            
            predict_time = time.time() - start_time
            print(f"预测耗时: {predict_time:.3f}秒", file=sys.stderr)
            
            # 提取轮廓
            mask_uint8 = (masks[0] * 255).astype(np.uint8)
            contours, _ = cv2.findContours(
                mask_uint8, 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            if len(contours) == 0:
                return None
            
            # 取最大轮廓
            max_contour = max(contours, key=cv2.contourArea)
            contour_points = max_contour.reshape(-1, 2).astype(np.float32)
            
            # 立即清理临时变量
            del masks, scores, logits, mask_uint8
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self.print_memory_status("预测后")
            return contour_points
            
        except Exception as e:
            print(f"预测失败: {str(e)}", file=sys.stderr)
            return None
    
    def cleanup(self):
        """清理资源"""
        print("清理SAM资源...", file=sys.stderr)
        
        # 清理缓存
        self.image_embedding_cache.clear()
        
        # 清理predictor
        if self.predictor is not None:
            self.predictor.reset_image()
        
        # 强制垃圾回收
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        self.print_memory_status("清理后")

# 全局单例实例
sam_manager = SAMSingleton()

# -------------------------- 兼容原有接口 --------------------------
def init_sam(img_path):
    return sam_manager.init_image(img_path)

def predict_mask(points, labels):
    return sam_manager.predict(points, labels)

def handle_memory_cleanup():
    sam_manager.cleanup()
    return True

# -------------------------- 主循环 --------------------------
def main():
    print("SAM进程启动（单例优化版）...", file=sys.stderr)
    
    while True:
        cmd = sys.stdin.readline().strip()
        if not cmd:
            continue

        if cmd.startswith("INIT:"):
            img_path = cmd[len("INIT:"):]
            success = init_sam(img_path)
            if success:
                print("INIT_SUCCESS")
            else:
                print("INIT_FAIL")
            sys.stdout.flush()

        elif cmd.startswith("PREDICT:"):
            param_str = cmd[len("PREDICT:"):]
            try:
                points = []
                labels = []
                for item in param_str.split(';'):
                    if not item:
                        continue
                    x, y, label = item.split(',')
                    points.append([float(x), float(y)])
                    labels.append(int(label))

                contour = predict_mask(points, labels)
                if contour is not None:
                    contour_str = ';'.join([f"{x},{y}" for x, y in contour])
                    print(f"PREDICT_SUCCESS:{contour_str}")
                else:
                    print("PREDICT_FAIL")
            except Exception as e:
                print("PREDICT_FAIL")
            sys.stdout.flush()

        elif cmd == "CLEANUP":
            handle_memory_cleanup()
            print("CLEANUP_SUCCESS")
            sys.stdout.flush()

        elif cmd == "EXIT":
            handle_memory_cleanup()
            print("EXIT_SUCCESS")
            sys.stdout.flush()
            break

        else:
            print(f"UNKNOWN_COMMAND:{cmd}")
            sys.stdout.flush()

if __name__ == "__main__":
    main()