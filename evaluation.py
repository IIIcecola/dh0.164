"""
audio2face测试集推理与评估脚本 - 更新版
根据补充信息调整数据格式处理
"""

import torch
import numpy as np
import librosa
import json
import os
import time
import logging
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import matplotlib
matplotlib.use('Agg')  # 使用Agg后端，无需图形界面
import matplotlib.pyplot as plt
import pandas as pd

# 导入模型相关模块
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from ModelDecoder import TransformerStackedDecoder
from AudioDataset import pack_exp
from PreProcess import ctrl_expressions as ctrl_expressions_list

# 设置日志
def setup_logging(log_dir: str = "./logs"):
    """配置日志系统"""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"inference_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

class Audio2FaceTester:
    def __init__(
        self, 
        model: dict,
        model_weights_path: str,
        wav2vec_path: str = "./wav2vec2-base-960h",
        device: str = None,
        output_dir: str = "./test_results",
        param_mapping_path= "./ctrl_expressions_map.json",
        max_params_per_category= 10,
        fps: int = 25  # 帧率
    ):
        """
        初始化测试器
        
        Args:
            model_weights_path: 解码器模型权重路径
            wav2vec_path: wav2vec2模型路径
            device: 设备 ('cuda' 或 'cpu')
            output_dir: 输出目录
            fps: 输出帧率
        """
        self.logger = setup_logging()
        self.logger.info("初始化Audio2Face测试器...")
        
        # 设置设备
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.logger.info(f"使用设备: {self.device}")
        
        # 设置输出目录
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建子目录
        self.pred_json_dir = self.output_dir / "pred_json"
        self.visualization_dir = self.output_dir / "visualization"
        self.metrics_dir = self.output_dir / "metrics"
        self.summary_dir = self.output_dir / "summary"
        
        for dir_path in [self.pred_json_dir, self.visualization_dir, 
                        self.metrics_dir, self.summary_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # 帧率设置
        self.fps = fps
        
        # 加载模型
        self._load_models(model_weights_path, wav2vec_path, model)
        
        # 性能统计
        self.inference_times = []
        self.audio_lengths = []

        # 初始化可视化工具
        self.logger.info(f"尝试初始化FacialParamVisualizer可视化工具...")
        self.visualizer = FacialParamVisualizer(
          param_mapping_path=Path(param_mapping_path),
          fps=self.fps,
          max_params_per_category=max_params_per_category,  # 每个类别最多显示多少参数
          output_dir=self.visualization_dir
        )
        
    def _load_models(self, model_weights_path: str, wav2vec_path: str, model: dict):
        """加载所有需要的模型"""
        try:
            self.logger.info("加载wav2vec2模型...")
            self.processor = Wav2Vec2Processor.from_pretrained(wav2vec_path)
            self.wav2vec_model = Wav2Vec2Model.from_pretrained(wav2vec_path).to(self.device)
            self.wav2vec_model.eval()
            
            self.logger.info("加载Transformer解码器...")
            self.decoder = TransformerStackedDecoder(
                input_dim=model.input_dim,
                output_dim=model.output_dim, 
                num_heads=model.num_heads,
                num_layers=model.num_layers
            ).to(self.device)
            
            # 加载权重
            state_dict = torch.load(model_weights_path, map_location=self.device)
            self.decoder.load_state_dict(state_dict)
            self.decoder.eval()
            
            self.logger.info("模型加载成功")
            
        except Exception as e:
            self.logger.error(f"模型加载失败: {str(e)}")
            raise
    
    def _process_audio_segment(self, audio_segment: np.ndarray, sr: int = 16000) -> np.ndarray:
        """
        处理单个音频片段
        
        Returns:
            预测的表情参数序列 (T, 136)
        """
        # 确保音频长度正确
        target_length = sr * 5  # 5秒
        if len(audio_segment) < target_length:
            audio_segment = np.pad(
                audio_segment, 
                (0, target_length - len(audio_segment)), 
                mode='constant', 
                constant_values=0
            )
        
        # 提取特征
        with torch.no_grad():
            inputs = self.processor(
                audio_segment, 
                sampling_rate=sr, 
                return_tensors="pt", 
                padding=True
            )
            
            # 移动到设备
            input_values = inputs.input_values.to(self.device)
            attention_mask = inputs.attention_mask.to(self.device) if 'attention_mask' in inputs else None
            
            # wav2vec2前向传播
            wav_features = self.wav2vec_model(
                input_values, 
                attention_mask=attention_mask
            ).last_hidden_state
            
            # 解码器前向传播
            pred = self.decoder(wav_features)
            pred = pred.squeeze(0).cpu().numpy()
            
        return pred
    
    def merge_segments(self, segments: List[np.ndarray]) -> List[List[float]]:
        """
        合并处理后的音频片段
        
        Args:
            segments: 各个片段的预测结果列表
        
        Returns:
            合并后的完整预测序列
        """
        merged = []
        for segment in segments:
            for frame in segment:
                merged.append(frame.tolist())
        
        return merged
    
    def inference_single_audio(
        self, 
        wav_path: str, 
        segment_length: int = 5
    ) -> Tuple[List[List[float]], Dict[str, float]]:
        """
        对单个音频文件进行推理
        
        Returns:
            pred_sequence: 预测的表情参数序列 (face_pred)
            metrics: 推理性能指标
        """
        start_time = time.time()
        
        try:
            # 加载音频
            self.logger.info(f"处理音频: {wav_path}")
            wave_data, sr = librosa.load(wav_path, sr=16000)
            audio_duration = len(wave_data) / sr
            
            # 计算总帧数
            total_frames = int(audio_duration * self.fps)
            
            # 分割音频
            segment_len = sr * segment_length
            segment_num = int(np.ceil(len(wave_data) / segment_len))
            
            self.logger.info(f"音频时长: {audio_duration:.2f}s, 分割为 {segment_num} 个片段")
            
            # 处理每个片段
            segments = []
            for i in range(segment_num):
                self.logger.debug(f"处理片段 {i+1}/{segment_num}")
                start_point = segment_len * i
                end_point = min(start_point + segment_len, len(wave_data))
                wav_segment = wave_data[start_point:end_point]
                
                # 处理片段
                pred_segment = self._process_audio_segment(wav_segment, sr)
                segments.append(pred_segment)
            
            # 合并片段
            merged_pred = self.merge_segments(segments)
            
            # 调整帧数匹配音频时长
            if len(merged_pred) > total_frames:
                merged_pred = merged_pred[:total_frames]
            elif len(merged_pred) < total_frames:
                # 重复最后一帧
                last_frame = merged_pred[-1]
                merged_pred.extend([last_frame] * (total_frames - len(merged_pred)))
            
            # 计算性能指标
            inference_time = time.time() - start_time
            real_time_factor = inference_time / audio_duration
            
            metrics = {
                'audio_duration': audio_duration,
                'inference_time': inference_time,
                'real_time_factor': real_time_factor,
                'segment_count': segment_num,
                'frame_count': len(merged_pred),
                'fps': self.fps
            }
            
            self.logger.info(
                f"推理完成 - 时长: {audio_duration:.2f}s, "
                f"推理时间: {inference_time:.2f}s, "
                f"实时比: {real_time_factor:.2f}, "
                f"帧数: {len(merged_pred)}"
            )
            
            return merged_pred, metrics
            
        except Exception as e:
            self.logger.error(f"音频推理失败 {wav_path}: {str(e)}")
            raise
    
    def save_predictions(
        self, 
        face_pred: List[List[float]], 
        output_path: str,
        motion_pred: Optional[List[List[float]]] = None
    ):
        """
        保存预测结果到JSON文件，按照指定格式
        
        Args:
            face_pred: 面部表情预测序列 (T, D)
            output_path: 输出文件路径
            motion_pred: 运动预测序列 (T, 3)，可选
        """
        try:
            total_frames = len(face_pred)
            
            # 创建motion_pred（如果未提供）
            if motion_pred is None:
                # 假设motion_pred是3维，初始化为0
                motion_pred = [[0.0, 0.0, 0.0] for _ in range(total_frames)]
            
            # 构建输出数据
            output_data = {
                "params_type": "set_face_animation",
                "motion_pred": motion_pred,
                "face_pred": face_pred,
                "fps": self.fps,
                "frames": total_frames,
                "timestamp": datetime.now().isoformat(),
                "version": "1.0"
            }
            
            # 保存JSON文件
            with open(output_path, 'w') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"预测结果已保存: {output_path}")
            
        except Exception as e:
            self.logger.error(f"保存预测结果失败: {str(e)}")
            raise
    
    def load_gt_json(self, gt_path: str) -> Optional[Dict[str, Any]]:
        """
        加载真实值JSON文件
        
        Args:
            gt_path: GT文件路径
            
        Returns:
            GT数据字典，包含face_pred等字段
        """
        try:
            if not os.path.exists(gt_path):
                self.logger.warning(f"GT文件不存在: {gt_path}")
                return None
            
            with open(gt_path, 'r') as f:
                gt_data = json.load(f)
            
            # 验证必要字段
            if 'face_pred' not in gt_data:
                self.logger.warning(f"GT文件缺少face_pred字段: {gt_path}")
                return None
            
            self.logger.info(f"GT文件加载成功: {gt_path}")
            return gt_data
            
        except Exception as e:
            self.logger.error(f"加载GT文件失败 {gt_path}: {str(e)}")
            return None
    
    def extract_gt_face_pred(self, gt_data: Dict[str, Any]) -> Optional[np.ndarray]:
        """
        从GT数据中提取face_pred序列
        
        Returns:
            face_pred序列数组 (T, D)
        """
        try:
            face_pred = gt_data.get('face_pred', [])
            if not face_pred:
                return None
            
            return np.array(face_pred)
            
        except Exception as e:
            self.logger.error(f"提取face_pred失败: {str(e)}")
            return None
    
    def visualize_predictions(
        self, 
        pred_face: np.ndarray, 
        gt_face: Optional[np.ndarray] = None,
        output_path: str = None,
        title: str = "Audio2Face预测结果",
        max_params_to_plot: int = 6
    ):
        """
        可视化预测结果
        
        Args:
            pred_face: 预测的face_pred序列 (T, D)
            gt_face: 真实的face_pred序列 (T, D)，可选
            output_path: 保存路径
            title: 图表标题
            max_params_to_plot: 最多可视化的参数数量
        """
        try:
            # 确定要绘制的参数数量
            num_params = pred_face.shape[1]
            params_to_plot = min(num_params, max_params_to_plot)
            
            # 选择关键参数索引
            # 可以均匀选择，或者选择方差最大的参数
            if num_params > max_params_to_plot:
                # 选择方差最大的参数
                variances = np.var(pred_face, axis=0)
                param_indices = np.argsort(variances)[-params_to_plot:]
            else:
                param_indices = range(params_to_plot)
            
            # 创建图形
            rows = int(np.ceil(params_to_plot / 2))
            cols = 2
            fig, axes = plt.subplots(rows, cols, figsize=(cols*6, rows*4))
            
            if params_to_plot == 1:
                axes = np.array([axes])
            
            fig.suptitle(title, fontsize=16)
            
            # 绘制每个参数
            for idx, param_idx in enumerate(param_indices):
                ax = axes.flat[idx] if params_to_plot > 1 else axes[0]
                
                # 绘制预测
                ax.plot(pred_face[:, param_idx], 'b-', label='预测', alpha=0.7, linewidth=1.5)
                
                # 绘制真实值（如果存在）
                if gt_face is not None and len(gt_face) == len(pred_face):
                    ax.plot(gt_face[:, param_idx], 'r--', label='真实', alpha=0.7, linewidth=1.5)
                
                ax.set_title(f'参数 {param_idx}')
                ax.set_xlabel('帧')
                ax.set_ylabel('值')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            # 隐藏多余的子图
            for idx in range(params_to_plot, rows*cols):
                axes.flat[idx].set_visible(False)
            
            plt.tight_layout()
            
            if output_path:
                plt.savefig(output_path, dpi=150, bbox_inches='tight')
                self.logger.info(f"可视化结果已保存: {output_path}")
            
            plt.close(fig)
            
        except Exception as e:
            self.logger.error(f"可视化失败: {str(e)}")
            plt.close('all')
    
    def compute_metrics(
        self, 
        pred_face: np.ndarray, 
        gt_face: np.ndarray
    ) -> Dict[str, float]:
        """
        计算预测与真实值之间的评估指标
        
        Returns:
            包含各项指标的字典
        """
        try:
            # 确保长度一致
            min_len = min(len(pred_face), len(gt_face))
            pred_face = pred_face[:min_len]
            gt_face = gt_face[:min_len]
            
            metrics = {}
            
            # 均方误差 (MSE)
            mse = np.mean((pred_face - gt_face) ** 2)
            metrics['mse'] = float(mse)
            
            # 平均绝对误差 (MAE)
            mae = np.mean(np.abs(pred_face - gt_face))
            metrics['mae'] = float(mae)
            
            # 相关系数（逐参数计算）
            correlations = []
            for i in range(pred_face.shape[1]):
                if (np.std(pred_face[:, i]) > 1e-6 and 
                    np.std(gt_face[:, i]) > 1e-6):
                    corr = np.corrcoef(pred_face[:, i], gt_face[:, i])[0, 1]
                    if not np.isnan(corr):
                        correlations.append(corr)
            
            if correlations:
                metrics['avg_correlation'] = float(np.mean(correlations))
                metrics['max_correlation'] = float(np.max(correlations))
                metrics['min_correlation'] = float(np.min(correlations))
            else:
                metrics['avg_correlation'] = 0.0
                metrics['max_correlation'] = 0.0
                metrics['min_correlation'] = 0.0
            
            # RMSE
            metrics['rmse'] = float(np.sqrt(mse))
            
            # 对称平均绝对百分比误差 (SMAPE)
            denominator = np.abs(pred_face) + np.abs(gt_face) + 1e-8
            smape = 200 * np.mean(np.abs(pred_face - gt_face) / denominator)
            metrics['smape'] = float(smape)
            
            self.logger.info(f"评估指标计算完成: MSE={mse:.4f}, MAE={mae:.4f}, 平均相关系数={metrics.get('avg_correlation', 0):.4f}")
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"计算评估指标失败: {str(e)}")
            return {}
    
    def run_test_suite(
        self, 
        test_data_dir: str,
        wav_dir: str,
        gt_json_dir: str,
        result_name: str = "test_run"
    ) -> pd.DataFrame:
        """
        运行完整的测试套件
        
        Args:
            test_data_dir: 测试数据目录，包含wav/和json/子目录
            result_name: 测试运行的名称
            
        Returns:
            包含所有测试结果的DataFrame
        """
        test_dir = Path(test_data_dir)
        wav_dir = Path(wav_dir)
        gt_json_dir = Path(gt_json_dir)
        
        # 检查目录
        if not wav_dir.exists():
            raise FileNotFoundError(f"wav目录不存在: {wav_dir}")
        
        # 获取所有wav文件
        wav_files = list(wav_dir.glob("*.wav"))
        if not wav_files:
            raise FileNotFoundError(f"在{wav_dir}中未找到wav文件")
        
        self.logger.info(f"开始测试套件，共{len(wav_files)}个音频文件")
        
        # 存储所有结果
        all_results = []
        
        for wav_path in wav_files:
            audio_name = wav_path.stem
            self.logger.info(f"处理测试样本: {audio_name}")
            
            try:
                # 1. 推理
                pred_face, perf_metrics = self.inference_single_audio(str(wav_path))
                
                # 2. 保存预测结果（按照指定格式）
                pred_output_path = self.pred_json_dir / f"{audio_name}_pred.json"
                self.save_predictions(pred_face, str(pred_output_path))
                
                # 3. 加载真实值（GT）
                gt_data = None
                gt_face = None
                
                # 查找GT文件（支持__converted.json后缀）
                possible_gt_files = [
                    gt_json_dir / f"apply_{audio_name}.json",
                    gt_json_dir / f"CD_{audio_name}_1_converted.json"
                ]
                
                gt_path = None
                for gt_file in possible_gt_files:
                    if gt_file.exists():
                        gt_path = gt_file
                        break
                
                if gt_path:
                    gt_data = self.load_gt_json(str(gt_path))
                    if gt_data:
                        gt_face = self.extract_gt_face_pred(gt_data)
                
                # 4. 可视化
                '''
                viz_output_path = self.visualization_dir / f"{audio_name}_viz.png"
                self.visualize_predictions(
                    np.array(pred_face), 
                    gt_face,
                    str(viz_output_path),
                    f"音频: {audio_name}"
                )
                '''
                saved_paths = self.visualizer.plot_all_categories(
                    np.array(pred_face),
                    gt_face,
                    audio_name,
                    output_dir=str(self.visualization_dir / audio_name)
                )
                if gt_face is not None:
                    dashboard_path = self.visualization_dir / f"{audio_name}_dashboard.png"
                    category_errors, category_correlations = self.visualizer.plot_summary_dashboard(
                        np.array(pred_face),
                        gt_face,
                        audio_name,
                        str(dashboard_path)
                    )
                
                # 5. 计算评估指标（如果有GT）
                eval_metrics = {}
                if gt_face is not None:
                    # 确保维度匹配
                    pred_face_array = np.array(pred_face)
                    
                    # 如果维度不一致，尝试截断或填充
                    if pred_face_array.shape[1] != gt_face.shape[1]:
                        self.logger.warning(
                            f"预测维度({pred_face_array.shape[1]})与GT维度({gt_face.shape[1]})不匹配，"
                            f"样本: {audio_name}"
                        )
                        # 取最小维度
                        min_dim = min(pred_face_array.shape[1], gt_face.shape[1])
                        pred_face_array = pred_face_array[:, :min_dim]
                        gt_face = gt_face[:, :min_dim]
                    
                    eval_metrics = self.compute_metrics(pred_face_array, gt_face)
                    
                    # 保存评估指标
                    if eval_metrics:
                        metrics_path = self.metrics_dir / f"{audio_name}_metrics.json"
                        with open(metrics_path, 'w') as f:
                            json.dump(eval_metrics, f, indent=2)
                
                # 6. 收集结果
                result = {
                    'audio_name': audio_name,
                    'audio_duration': perf_metrics['audio_duration'],
                    'inference_time': perf_metrics['inference_time'],
                    'real_time_factor': perf_metrics['real_time_factor'],
                    'frame_count': len(pred_face),
                    'fps': self.fps,
                    'has_gt': gt_face is not None
                }
                result.update(eval_metrics)  # 添加评估指标
                if gt_face is not None:
                    for category, error in category_errors.items():
                        result[f'{category}_mae'] = error
                    for category, corr in category_correlations.items():
                        result[f'{category}_corr'] = corr
                
                all_results.append(result)
                
                # 记录性能数据
                self.inference_times.append(perf_metrics['inference_time'])
                self.audio_lengths.append(perf_metrics['audio_duration'])
                
                self.logger.info(f"完成样本: {audio_name}")
                
            except Exception as e:
                self.logger.error(f"处理测试样本失败 {wav_path}: {str(e)}")
                self.logger.error(traceback.format_exc())
                continue
        
        # 生成测试报告
        report = self._generate_test_report(all_results, result_name)
        
        return report
    
    def _generate_test_report(self, all_results: List[Dict], result_name: str) -> pd.DataFrame:
        """生成测试报告"""
        # 转换为DataFrame
        df = pd.DataFrame(all_results)
        
        if df.empty:
            self.logger.warning("没有有效的测试结果")
            return df
        
        # 汇总统计
        summary = {
            'total_samples': len(df),
            'samples_with_gt': df['has_gt'].sum() if 'has_gt' in df.columns else 0,
            'avg_inference_time': df['inference_time'].mean(),
            'avg_real_time_factor': df['real_time_factor'].mean(),
            'total_inference_time': df['inference_time'].sum(),
            'total_audio_duration': df['audio_duration'].sum(),
        }
        
        # 添加评估指标统计（如果有GT）
        if 'mse' in df.columns and df['has_gt'].any():
            summary['avg_mse'] = df[df['has_gt']]['mse'].mean()
            summary['avg_mae'] = df[df['has_gt']]['mae'].mean()
            summary['avg_rmse'] = df[df['has_gt']]['rmse'].mean()
            summary['avg_correlation'] = df[df['has_gt']]['avg_correlation'].mean()
        
        # 保存汇总报告
        summary_path = self.summary_dir / f"{result_name}_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # 保存详细结果CSV
        csv_path = self.summary_dir / f"{result_name}_detailed.csv"
        df.to_csv(csv_path, index=False)
        
        # 创建可视化报告
        self._create_summary_visualization(df, summary, result_name)
        
        self.logger.info(f"测试报告已保存到: {self.summary_dir}")
        self.logger.info(f"汇总结果: {summary}")
        
        return df
    
    def _create_summary_visualization(self, df: pd.DataFrame, summary: Dict, result_name: str):
        """创建汇总可视化"""
        try:
            # 确定子图数量
            has_gt = 'has_gt' in df.columns and df['has_gt'].any()
            num_plots = 3 if has_gt else 2
            
            fig, axes = plt.subplots(1, num_plots, figsize=(5*num_plots, 4))
            if num_plots == 1:
                axes = [axes]
            
            # 1. 推理时间分布
            ax = axes[0]
            ax.hist(df['inference_time'], bins=10, alpha=0.7, color='skyblue', edgecolor='black')
            ax.axvline(df['inference_time'].mean(), color='red', linestyle='--', 
                      label=f'均值: {df["inference_time"].mean():.2f}s')
            ax.set_xlabel('推理时间 (秒)')
            ax.set_ylabel('样本数')
            ax.set_title('推理时间分布')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # 2. 实时比分布
            ax = axes[1]
            ax.hist(df['real_time_factor'], bins=10, alpha=0.7, color='lightgreen', edgecolor='black')
            ax.axvline(df['real_time_factor'].mean(), color='red', linestyle='--',
                      label=f'均值: {df["real_time_factor"].mean():.2f}')
            ax.set_xlabel('实时比')
            ax.set_ylabel('样本数')
            ax.set_title('实时比分布 (越低越好)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # 3. 如果有GT，显示MSE分布
            if has_gt and 'mse' in df.columns:
                ax = axes[2]
                valid_mse = df[df['has_gt']]['mse']
                ax.hist(valid_mse, bins=10, alpha=0.7, color='salmon', edgecolor='black')
                ax.axvline(valid_mse.mean(), color='red', linestyle='--',
                          label=f'均值: {valid_mse.mean():.4f}')
                ax.set_xlabel('MSE')
                ax.set_ylabel('样本数')
                ax.set_title('均方误差分布 (越低越好)')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            plt.suptitle(f'测试套件汇总 - {result_name}', fontsize=14)
            plt.tight_layout()
            
            summary_viz_path = self.summary_dir / f"{result_name}_summary.png"
            plt.savefig(summary_viz_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            # 创建性能对比图
            if len(df) > 1:
                self._create_performance_comparison(df, result_name)
            
        except Exception as e:
            self.logger.warning(f"创建汇总可视化失败: {str(e)}")
    
    def _create_performance_comparison(self, df: pd.DataFrame, result_name: str):
        """创建性能对比图"""
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # 按推理时间排序
            df_sorted = df.sort_values('inference_time')
            
            x = range(len(df_sorted))
            width = 0.35
            
            # 绘制推理时间
            bars1 = ax.bar(x, df_sorted['inference_time'], width, 
                          label='推理时间 (秒)', color='skyblue')
            
            # 绘制实时比（使用次坐标轴）
            ax2 = ax.twinx()
            bars2 = ax2.bar([i + width for i in x], df_sorted['real_time_factor'], width,
                           label='实时比', color='lightgreen')
            
            # 设置标签
            ax.set_xlabel('测试样本')
            ax.set_ylabel('推理时间 (秒)', color='skyblue')
            ax2.set_ylabel('实时比', color='lightgreen')
            
            # 设置x轴刻度
            ax.set_xticks([i + width/2 for i in x])
            ax.set_xticklabels(df_sorted['audio_name'], rotation=45, ha='right')
            
            # 添加图例
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
            
            plt.title('测试样本性能对比')
            plt.tight_layout()
            
            comparison_path = self.summary_dir / f"{result_name}_performance_comparison.png"
            plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
        except Exception as e:
            self.logger.warning(f"创建性能对比图失败: {str(e)}")


"""
面部参数可视化器
根据参数映射关系进行智能可视化
"""
class FacialParamVisualizer:
    """面部参数可视化器 - 支持按类别分组可视化"""
    
    def __init__(
        self, 
        param_mapping_path: Optional[str] = None,
        fps: int = 25,
        max_params_per_category: int = 10,  # 每个类别最多显示多少参数
        output_dir: str = "./visualizations"
    ):
        """
        初始化可视化器
        
        Args:
            param_mapping_path: 参数映射JSON文件路径
            fps: 视频帧率
            max_params_per_category: 每个类别最大显示参数数
            output_dir: 输出目录
        """
        self.fps = fps
        self.max_params_per_category = max_params_per_category
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # 设置日志
        self.logger = logging.getLogger(__name__)
        
        # 颜色方案
        self.colors = {
            'pred': '#1f77b4',  # 蓝色
            'gt': '#d62728',     # 红色
            'diff': '#2ca02c',   # 绿色
            'background': '#f5f5f5'
        }
        
        # 加载参数映射
        self.param_groups = self._load_param_mapping(param_mapping_path)
        
        # 参数重要性记录
        self.param_importance = {}
        
    def _load_param_mapping(self, mapping_path: Optional[str]) -> Dict[str, List[int]]:
        """
        加载参数映射关系
        
        Returns:
            参数分组字典
        """
        default_mapping = {
            "brow": [0, 1, 2, 3],
            "eye": [4, 5],
            "nose": [6, 7],
            "mouth": list(range(8, 136))  # 示例，实际需要根据您的映射调整
        }
        
        if not mapping_path:
            self.logger.warning("未提供参数映射文件，使用默认映射")
            return default_mapping
        
        try:
            with open(mapping_path, 'r') as f:
                mapping = json.load(f)
            
            # 确保所有值都是整数列表
            for key, value in mapping.items():
                if isinstance(value, list):
                    mapping[key] = [int(v) for v in value]
            
            self.logger.info(f"加载参数映射成功，共{len(mapping)}个类别")
            return mapping
            
        except Exception as e:
            self.logger.error(f"加载参数映射失败: {str(e)}")
            return default_mapping
    
    def select_key_params(
        self, 
        pred_face: np.ndarray, 
        gt_face: Optional[np.ndarray] = None,
        category: Optional[str] = None,
        num_params: int = 6,
        selection_method: str = "variance"
    ) -> List[int]:
        """
        智能选择关键参数
        
        Args:
            pred_face: 预测的面部参数
            gt_face: 真实的面部参数（可选）
            category: 指定类别，None表示所有参数
            num_params: 选择的参数数量
            selection_method: 选择方法
                - "variance": 基于方差选择（变化最大的参数）
                - "error": 基于误差选择（误差最大的参数）
                - "correlation": 基于相关性选择（相关性最低的参数）
                - "combined": 综合评分
        
        Returns:
            选择的参数索引列表
        """
        if category and category in self.param_groups:
            candidate_indices = self.param_groups[category]
        else:
            # 所有参数
            candidate_indices = list(range(pred_face.shape[1]))
        
        # 确保不超过可用参数数
        num_params = min(num_params, len(candidate_indices))
        
        if selection_method == "variance":
            # 基于方差选择
            variances = []
            for idx in candidate_indices:
                var = np.var(pred_face[:, idx])
                variances.append((idx, var))
            
            # 按方差降序排序
            variances.sort(key=lambda x: x[1], reverse=True)
            selected = [idx for idx, _ in variances[:num_params]]
            
        elif selection_method == "error" and gt_face is not None:
            # 基于绝对误差选择
            errors = []
            for idx in candidate_indices:
                error = np.mean(np.abs(pred_face[:, idx] - gt_face[:, idx]))
                errors.append((idx, error))
            
            errors.sort(key=lambda x: x[1], reverse=True)
            selected = [idx for idx, _ in errors[:num_params]]
            
        elif selection_method == "correlation" and gt_face is not None:
            # 基于相关性选择
            correlations = []
            for idx in candidate_indices:
                if (np.std(pred_face[:, idx]) > 1e-6 and 
                    np.std(gt_face[:, idx]) > 1e-6):
                    corr = np.corrcoef(pred_face[:, idx], gt_face[:, idx])[0, 1]
                    if not np.isnan(corr):
                        correlations.append((idx, abs(corr)))
            
            # 相关性越低，问题可能越大，优先展示
            correlations.sort(key=lambda x: x[1])
            selected = [idx for idx, _ in correlations[:num_params]]
            
        else:
            # 默认使用方差
            variances = []
            for idx in candidate_indices:
                var = np.var(pred_face[:, idx])
                variances.append((idx, var))
            
            variances.sort(key=lambda x: x[1], reverse=True)
            selected = [idx for idx, _ in variances[:num_params]]
        
        return selected
    
    def plot_category_comparison(
        self,
        pred_face: np.ndarray,
        gt_face: Optional[np.ndarray] = None,
        category: str = "mouth",
        output_path: Optional[str] = None,
        title_suffix: str = ""
    ):
        """
        绘制单个类别的参数对比图
        
        Args:
            pred_face: 预测参数序列 (T, D)
            gt_face: 真实参数序列 (T, D)，可选
            category: 参数类别
            output_path: 输出路径
            title_suffix: 标题后缀
        """
        if category not in self.param_groups:
            self.logger.warning(f"未知的参数类别: {category}")
            return
        
        param_indices = self.param_groups[category]
        num_params = len(param_indices)
        
        # 如果参数太多，智能选择关键参数
        if num_params > self.max_params_per_category:
            selected_indices = self.select_key_params(
                pred_face, gt_face, category, 
                self.max_params_per_category, "variance"
            )
            self.logger.info(f"类别 '{category}' 参数过多 ({num_params})，选择 {len(selected_indices)} 个关键参数")
        else:
            selected_indices = param_indices
        
        # 计算子图布局
        num_plots = len(selected_indices)
        cols = 2
        rows = (num_plots + cols - 1) // cols
        
        # 创建图形
        fig, axes = plt.subplots(rows, cols, figsize=(cols*6, rows*3))
        if rows == 1 and cols == 1:
            axes = np.array([axes])
        elif rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)
        
        # 展平axes数组以便迭代
        axes_flat = axes.flatten()
        
        # 时间轴（秒）
        time_axis = np.arange(len(pred_face)) / self.fps
        
        for idx, (ax, param_idx) in enumerate(zip(axes_flat, selected_indices)):
            # 绘制预测值
            ax.plot(time_axis, pred_face[:, param_idx], 
                   color=self.colors['pred'], label='预测', linewidth=2, alpha=0.8)
            
            # 绘制真实值（如果存在）
            if gt_face is not None and len(gt_face) == len(pred_face):
                ax.plot(time_axis, gt_face[:, param_idx], 
                       color=self.colors['gt'], label='真实', linewidth=1.5, 
                       linestyle='--', alpha=0.7)
            
            # 计算并显示误差指标
            if gt_face is not None:
                mae = np.mean(np.abs(pred_face[:, param_idx] - gt_face[:, param_idx]))
                ax.text(0.02, 0.98, f'MAE: {mae:.4f}', 
                       transform=ax.transAxes, fontsize=9,
                       verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            ax.set_title(f'参数 {param_idx} ({category})')
            ax.set_xlabel('时间 (秒)')
            ax.set_ylabel('参数值')
            ax.legend(loc='upper right', fontsize=9)
            ax.grid(True, alpha=0.3)
            
            # 设置合适的Y轴范围
            all_values = pred_face[:, param_idx]
            if gt_face is not None:
                all_values = np.concatenate([all_values, gt_face[:, param_idx]])
            
            y_min, y_max = all_values.min(), all_values.max()
            y_range = y_max - y_min
            ax.set_ylim(y_min - 0.1*y_range, y_max + 0.1*y_range)
        
        # 隐藏多余的子图
        for idx in range(num_plots, len(axes_flat)):
            axes_flat[idx].set_visible(False)
        
        # 主标题
        main_title = f'{category}参数对比 {title_suffix}'.strip()
        if len(selected_indices) < num_params:
            main_title += f' (显示{len(selected_indices)}/{num_params}个参数)'
        
        plt.suptitle(main_title, fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])  # 为suptitle留出空间
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            self.logger.info(f"类别对比图已保存: {output_path}")
        
        plt.close(fig)
        
        return selected_indices  # 返回实际显示的参数索引
    
    def plot_all_categories(
        self,
        pred_face: np.ndarray,
        gt_face: Optional[np.ndarray] = None,
        audio_name: str = "unknown",
        output_dir: Optional[str] = None
    ):
        """
        为所有类别分别绘制对比图
        
        Returns:
            保存的所有图表路径列表
        """
        if output_dir is None:
            output_dir = self.output_dir / audio_name
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(exist_ok=True, parents=True)
        
        saved_paths = []
        
        for category in self.param_groups.keys():
            output_path = output_dir / f"{audio_name}_{category}_comparison.png"
            
            try:
                # 绘制单个类别
                shown_indices = self.plot_category_comparison(
                    pred_face, gt_face, category,
                    str(output_path),
                    f"- {audio_name}"
                )
                
                saved_paths.append(output_path)
                
                # 如果有真实值，为这个类别额外生成统计图表
                if gt_face is not None:
                    self.plot_category_statistics(
                        pred_face, gt_face, category, shown_indices,
                        str(output_dir / f"{audio_name}_{category}_stats.png"),
                        audio_name
                    )
                    
            except Exception as e:
                self.logger.error(f"绘制类别 '{category}' 失败: {str(e)}")
                continue
        
        return saved_paths
    
    def plot_category_statistics(
        self,
        pred_face: np.ndarray,
        gt_face: np.ndarray,
        category: str,
        param_indices: List[int],
        output_path: str,
        audio_name: str = ""
    ):
        """
        绘制单个类别的统计信息图
        """
        if category not in self.param_groups:
            return
        
        # 计算统计指标
        errors = []
        correlations = []
        variances_pred = []
        variances_gt = []
        
        for param_idx in param_indices:
            # 绝对误差
            mae = np.mean(np.abs(pred_face[:, param_idx] - gt_face[:, param_idx]))
            errors.append(mae)
            
            # 相关系数
            if (np.std(pred_face[:, param_idx]) > 1e-6 and 
                np.std(gt_face[:, param_idx]) > 1e-6):
                corr = np.corrcoef(pred_face[:, param_idx], gt_face[:, param_idx])[0, 1]
                correlations.append(corr if not np.isnan(corr) else 0)
            else:
                correlations.append(0)
            
            # 方差
            variances_pred.append(np.var(pred_face[:, param_idx]))
            variances_gt.append(np.var(gt_face[:, param_idx]))
        
        # 创建统计图表
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. 误差条形图
        bars1 = axes[0, 0].bar(range(len(errors)), errors, color='salmon')
        axes[0, 0].set_title(f'{category}参数 - 平均绝对误差(MAE)')
        axes[0, 0].set_xlabel('参数索引')
        axes[0, 0].set_ylabel('MAE')
        axes[0, 0].set_xticks(range(len(param_indices)))
        axes[0, 0].set_xticklabels(param_indices)
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        for bar, error in zip(bars1, errors):
            height = bar.get_height()
            axes[0, 0].text(bar.get_x() + bar.get_width()/2., height,
                           f'{error:.4f}', ha='center', va='bottom', fontsize=8)
        
        # 2. 相关系数条形图
        bars2 = axes[0, 1].bar(range(len(correlations)), correlations, 
                              color=['green' if c > 0.8 else 'orange' if c > 0.5 else 'red' 
                                    for c in correlations])
        axes[0, 1].set_title(f'{category}参数 - 预测相关性')
        axes[0, 1].set_xlabel('参数索引')
        axes[0, 1].set_ylabel('相关系数')
        axes[0, 1].set_xticks(range(len(param_indices)))
        axes[0, 1].set_xticklabels(param_indices)
        axes[0, 1].axhline(y=0.8, color='gray', linestyle='--', alpha=0.5)
        axes[0, 1].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # 3. 方差对比散点图
        axes[1, 0].scatter(variances_pred, variances_gt, alpha=0.6, edgecolors='k')
        
        # 添加对角线（理想情况）
        max_var = max(max(variances_pred), max(variances_gt))
        axes[1, 0].plot([0, max_var], [0, max_var], 'r--', alpha=0.5, label='理想匹配')
        
        axes[1, 0].set_title(f'{category}参数 - 方差对比')
        axes[1, 0].set_xlabel('预测方差')
        axes[1, 0].set_ylabel('真实方差')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 误差分布直方图
        axes[1, 1].hist(errors, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
        axes[1, 1].axvline(np.mean(errors), color='red', linestyle='--',
                          label=f'均值: {np.mean(errors):.4f}')
        axes[1, 1].set_title(f'{category}参数 - 误差分布')
        axes[1, 1].set_xlabel('MAE')
        axes[1, 1].set_ylabel('频数')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # 主标题
        title = f'{category}参数统计 - {audio_name}' if audio_name else f'{category}参数统计'
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        self.logger.info(f"类别统计图已保存: {output_path}")
    
    def plot_temporal_analysis(
        self,
        pred_face: np.ndarray,
        gt_face: Optional[np.ndarray] = None,
        param_indices: List[int] = None,
        audio_name: str = "",
        output_path: Optional[str] = None
    ):
        """
        针对指定参数的详细时序分析
        
        Args:
            param_indices: 要分析的参数索引列表
        """
        if param_indices is None:
            # 如果没有指定，选择变化最大的6个参数
            param_indices = self.select_key_params(pred_face, gt_face, 
                                                  num_params=6, 
                                                  selection_method="variance")
        
        num_params = len(param_indices)
        fig, axes = plt.subplots(num_params, 3, figsize=(18, 4*num_params))
        
        if num_params == 1:
            axes = axes.reshape(1, 3)
        
        time_axis = np.arange(len(pred_face)) / self.fps
        
        for row, param_idx in enumerate(param_indices):
            ax1 = axes[row, 0] if num_params > 1 else axes[0]
            ax2 = axes[row, 1] if num_params > 1 else axes[1]
            ax3 = axes[row, 2] if num_params > 1 else axes[2]
            
            # 子图1: 时序对比
            ax1.plot(time_axis, pred_face[:, param_idx], 
                    color=self.colors['pred'], label='预测', linewidth=2)
            
            if gt_face is not None:
                ax1.plot(time_axis, gt_face[:, param_idx], 
                        color=self.colors['gt'], label='真实', linewidth=1.5, linestyle='--')
            
            ax1.set_title(f'参数 {param_idx} - 时序对比')
            ax1.set_xlabel('时间 (秒)')
            ax1.set_ylabel('参数值')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 子图2: 误差时序
            if gt_face is not None:
                error = pred_face[:, param_idx] - gt_face[:, param_idx]
                ax2.plot(time_axis, error, color=self.colors['diff'], linewidth=1.5)
                ax2.fill_between(time_axis, 0, error, alpha=0.3, color=self.colors['diff'])
                
                # 零线
                ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                
                # 误差统计
                mae = np.mean(np.abs(error))
                rmse = np.sqrt(np.mean(error**2))
                
                stats_text = f'MAE: {mae:.4f}\nRMSE: {rmse:.4f}'
                ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes,
                        fontsize=10, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                ax2.set_title(f'参数 {param_idx} - 误差时序')
                ax2.set_xlabel('时间 (秒)')
                ax2.set_ylabel('预测误差')
                ax2.grid(True, alpha=0.3)
            
            # 子图3: 频谱分析（展示参数变化频率）
            if len(pred_face) > 1:
                # 计算FFT
                signal = pred_face[:, param_idx]
                signal = signal - np.mean(signal)  # 去直流
                
                n = len(signal)
                freqs = np.fft.rfftfreq(n, d=1.0/self.fps)
                fft_values = np.abs(np.fft.rfft(signal))
                
                # 只显示有意义的部分（0-10Hz，因为面部动作通常低频）
                max_freq = min(10, freqs[-1])
                mask = freqs <= max_freq
                
                ax3.plot(freqs[mask], fft_values[mask], color='purple', linewidth=1.5)
                ax3.set_title(f'参数 {param_idx} - 频谱分析')
                ax3.set_xlabel('频率 (Hz)')
                ax3.set_ylabel('幅度')
                ax3.grid(True, alpha=0.3)
                
                # 标记主要频率
                if len(fft_values[mask]) > 0:
                    main_freq_idx = np.argmax(fft_values[mask])
                    main_freq = freqs[mask][main_freq_idx]
                    main_amp = fft_values[mask][main_freq_idx]
                    
                    ax3.plot(main_freq, main_amp, 'ro', markersize=8)
                    ax3.text(main_freq, main_amp*1.1, 
                            f'{main_freq:.1f} Hz', 
                            ha='center', fontsize=9)
        
        # 主标题
        main_title = f'参数时序分析'
        if audio_name:
            main_title += f' - {audio_name}'
        
        plt.suptitle(main_title, fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            self.logger.info(f"时序分析图已保存: {output_path}")
        
        plt.close(fig)
    
    def plot_summary_dashboard(
        self,
        pred_face: np.ndarray,
        gt_face: np.ndarray,
        audio_name: str = "",
        output_path: Optional[str] = None
    ):
        """
        创建综合仪表板，展示整体评估结果
        """
        # 计算每个类别的平均误差
        category_errors = {}
        category_correlations = {}
        
        for category, param_indices in self.param_groups.items():
            errors = []
            correlations = []
            
            for param_idx in param_indices:
                if param_idx < pred_face.shape[1] and param_idx < gt_face.shape[1]:
                    # 误差
                    mae = np.mean(np.abs(pred_face[:, param_idx] - gt_face[:, param_idx]))
                    errors.append(mae)
                    
                    # 相关性
                    if (np.std(pred_face[:, param_idx]) > 1e-6 and 
                        np.std(gt_face[:, param_idx]) > 1e-6):
                        corr = np.corrcoef(pred_face[:, param_idx], gt_face[:, param_idx])[0, 1]
                        if not np.isnan(corr):
                            correlations.append(abs(corr))
            
            if errors:
                category_errors[category] = np.mean(errors)
            
            if correlations:
                category_correlations[category] = np.mean(correlations)
        
        # 创建仪表板
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. 各类别平均误差
        categories = list(category_errors.keys())
        errors = [category_errors[c] for c in categories]
        
        bars = axes[0, 0].bar(categories, errors, color='lightcoral')
        axes[0, 0].set_title('各类别平均绝对误差(MAE)')
        axes[0, 0].set_xlabel('参数类别')
        axes[0, 0].set_ylabel('平均MAE')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        
        # 误差标签
        for bar, error in zip(bars, errors):
            height = bar.get_height()
            axes[0, 0].text(bar.get_x() + bar.get_width()/2., height,
                           f'{error:.4f}', ha='center', va='bottom')
        
        # 2. 各类别平均相关性
        categories_corr = list(category_correlations.keys())
        corrs = [category_correlations[c] for c in categories_corr]
        
        colors_corr = ['green' if c > 0.8 else 'orange' if c > 0.5 else 'red' 
                      for c in corrs]
        
        bars = axes[0, 1].bar(categories_corr, corrs, color=colors_corr)
        axes[0, 1].set_title('各类别平均相关系数')
        axes[0, 1].set_xlabel('参数类别')
        axes[0, 1].set_ylabel('平均相关系数')
        axes[0, 1].axhline(y=0.8, color='gray', linestyle='--', alpha=0.5, label='优秀(>0.8)')
        axes[0, 1].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='合格(>0.5)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].legend(loc='upper right', fontsize=9)
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # 3. 误差分布直方图（所有参数）
        all_errors = []
        for param_idx in range(min(pred_face.shape[1], gt_face.shape[1])):
            mae = np.mean(np.abs(pred_face[:, param_idx] - gt_face[:, param_idx]))
            all_errors.append(mae)
        
        axes[1, 0].hist(all_errors, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[1, 0].axvline(np.mean(all_errors), color='red', linestyle='--',
                          label=f'均值: {np.mean(all_errors):.4f}')
        axes[1, 0].set_title('所有参数误差分布')
        axes[1, 0].set_xlabel('MAE')
        axes[1, 0].set_ylabel('参数数量')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 误差vs方差散点图
        all_var_pred = []
        all_var_gt = []
        all_mae = []
        
        for param_idx in range(min(pred_face.shape[1], gt_face.shape[1])):
            var_pred = np.var(pred_face[:, param_idx])
            var_gt = np.var(gt_face[:, param_idx])
            mae = np.mean(np.abs(pred_face[:, param_idx] - gt_face[:, param_idx]))
            
            all_var_pred.append(var_pred)
            all_var_gt.append(var_gt)
            all_mae.append(mae)
        
        scatter = axes[1, 1].scatter(all_var_gt, all_mae, c=all_var_pred, 
                                    cmap='viridis', alpha=0.6, s=30)
        axes[1, 1].set_title('参数误差 vs 真实方差')
        axes[1, 1].set_xlabel('真实方差')
        axes[1, 1].set_ylabel('MAE')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 添加颜色条
        plt.colorbar(scatter, ax=axes[1, 1], label='预测方差')
        
        # 主标题
        main_title = f'模型性能综合仪表板'
        if audio_name:
            main_title += f' - {audio_name}'
        
        plt.suptitle(main_title, fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            self.logger.info(f"综合仪表板已保存: {output_path}")
        
        plt.close(fig)
        
        return category_errors, category_correlations


class Audio2FaceResultEvaluator:
    """
    专门处理已有推理结果的评估器
    输入: 推理结果目录 + 真值目录，直接进行对比评估
    """
    
    def __init__(
        self,
        pred_dir: str,  # 推理结果目录，包含 transformer_pred_*.json 文件
        gt_dir: str,    # 真值目录，包含 CD_*_1_converted.json 文件
        output_dir: str = "./result_evaluation",
        param_mapping_path: str = "./ctrl_expressions_map.json",
        max_params_per_category: int = 10,
        fps: int = 25
    ):
        """
        初始化结果评估器
        
        Args:
            pred_dir: 预测结果JSON文件目录
            gt_dir: 真实值JSON文件目录
            output_dir: 输出目录
            param_mapping_path: 参数映射文件路径
            max_params_per_category: 每个类别最大显示参数数
            fps: 帧率
        """
        self.logger = setup_logging()
        self.logger.info("初始化Audio2Face结果评估器...")
        
        self.pred_dir = Path(pred_dir)
        self.gt_dir = Path(gt_dir)
        self.output_dir = Path(output_dir)
        
        # 创建子目录
        self.visualization_dir = self.output_dir / "visualization"
        self.metrics_dir = self.output_dir / "metrics"
        self.summary_dir = self.output_dir / "summary"
        
        for dir_path in [self.visualization_dir, self.metrics_dir, self.summary_dir]:
            dir_path.mkdir(exist_ok=True, parents=True)
        
        self.fps = fps
        
        # 检查目录存在性
        if not self.pred_dir.exists():
            raise FileNotFoundError(f"预测结果目录不存在: {self.pred_dir}")
        if not self.gt_dir.exists():
            self.logger.warning(f"真值目录不存在: {self.gt_dir}, 将只进行可视化")
        
        # 初始化可视化器（复用原有可视化器）
        self.visualizer = FacialParamVisualizer(
            param_mapping_path=param_mapping_path,
            fps=fps,
            max_params_per_category=max_params_per_category,
            output_dir=str(self.visualization_dir)
        )
        
        # 结果存储
        self.all_results = []
        self.logger.info("结果评估器初始化完成")
    
    def extract_filename_from_pred(self, pred_filename: str) -> str:
        """
        从预测文件名提取原始音频名称
        
        Args:
            pred_filename: 如 "transformer_pred_test1.json"
        
        Returns:
            如 "test1"
        """
        # 移除前缀 "transformer_pred_" 和后缀 ".json"
        if pred_filename.startswith("transformer_pred_"):
            base_name = pred_filename[len("transformer_pred_"):]
        else:
            base_name = pred_filename
        
        if base_name.endswith(".json"):
            base_name = base_name[:-5]
        
        return base_name
    
    def find_matching_gt_file(self, audio_name: str) -> Optional[Path]:
        """
        根据音频名称查找匹配的真值文件
        
        支持多种真值文件命名格式:
        1. CD_{audio_name}_1_converted.json (主要)
        2. apply_{audio_name}.json (备选)
        """
        possible_files = [
            self.gt_dir / f"CD_{audio_name}_1_converted.json",
            self.gt_dir / f"apply_{audio_name}.json",
            self.gt_dir / f"{audio_name}.json",  # 纯音频名
        ]
        
        for gt_file in possible_files:
            if gt_file.exists():
                return gt_file
        
        return None
    
    def load_face_pred_from_json(self, json_path: Path) -> Optional[np.ndarray]:
        """
        从JSON文件加载face_pred字段
        
        Returns:
            face_pred序列数组 (T, 136)
        """
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            face_pred = data.get('face_pred', [])
            if not face_pred:
                self.logger.warning(f"JSON文件没有face_pred字段: {json_path}")
                return None
            
            # 转换为numpy数组
            face_pred_array = np.array(face_pred)
            
            # 验证维度
            if len(face_pred_array.shape) != 2:
                self.logger.warning(f"face_pred维度异常: {face_pred_array.shape}, 文件: {json_path}")
                return None
            
            self.logger.debug(f"加载 {json_path.name}: {face_pred_array.shape}")
            return face_pred_array
            
        except Exception as e:
            self.logger.error(f"加载JSON文件失败 {json_path}: {str(e)}")
            return None

    @classmethod
    def from_config(cls, config: OmegaConf):
        """
        从OmegaConf配置创建评估器
        
        Args:
            config: OmegaConf配置对象
        
        Returns:
            Audio2FaceResultEvaluator实例
        """
        return cls(
            pred_dir=config.evaluator.pred_dir,
            gt_dir=config.evaluator.gt_dir,
            output_dir=config.evaluator.output_dir,
            param_mapping_path=config.visualizer.param_mapping_path,
            max_params_per_category=config.visualizer.max_params_per_category,
            fps=config.visualizer.fps
        )
    
    def compute_metrics(self, pred_face: np.ndarray, gt_face: np.ndarray) -> Dict[str, float]:
        """
        计算评估指标（复用原有逻辑，可以导入或直接复制）
        """
        # 这里可以直接从原脚本导入compute_metrics函数
        # 或者复制实现逻辑
        try:
            min_len = min(len(pred_face), len(gt_face))
            pred_face = pred_face[:min_len]
            gt_face = gt_face[:min_len]
            
            metrics = {}
            
            # MSE
            mse = np.mean((pred_face - gt_face) ** 2)
            metrics['mse'] = float(mse)
            
            # MAE
            mae = np.mean(np.abs(pred_face - gt_face))
            metrics['mae'] = float(mae)
            
            # 相关系数
            correlations = []
            for i in range(pred_face.shape[1]):
                if (np.std(pred_face[:, i]) > 1e-6 and 
                    np.std(gt_face[:, i]) > 1e-6):
                    corr = np.corrcoef(pred_face[:, i], gt_face[:, i])[0, 1]
                    if not np.isnan(corr):
                        correlations.append(corr)
            
            if correlations:
                metrics['avg_correlation'] = float(np.mean(correlations))
                metrics['max_correlation'] = float(np.max(correlations))
                metrics['min_correlation'] = float(np.min(correlations))
            else:
                metrics['avg_correlation'] = 0.0
                metrics['max_correlation'] = 0.0
                metrics['min_correlation'] = 0.0
            
            # RMSE
            metrics['rmse'] = float(np.sqrt(mse))
            
            # SMAPE
            denominator = np.abs(pred_face) + np.abs(gt_face) + 1e-8
            smape = 200 * np.mean(np.abs(pred_face - gt_face) / denominator)
            metrics['smape'] = float(smape)
            
            self.logger.info(f"评估指标: MSE={mse:.4f}, MAE={mae:.4f}, 平均相关={metrics.get('avg_correlation', 0):.4f}")
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"计算评估指标失败: {str(e)}")
            return {}
    
    def evaluate_single_pair(
        self, 
        pred_path: Path, 
        gt_path: Optional[Path] = None
    ) -> Optional[Dict]:
        """
        评估单个预测-真值对
        
        Returns:
            包含评估结果的字典
        """
        audio_name = pred_path.stem
        self.logger.info(f"评估样本: {audio_name}")
        
        try:
            # 1. 加载预测结果
            pred_face = self.load_face_pred_from_json(pred_path)
            if pred_face is None:
                return None
            
            # 2. 加载真值（如果存在）
            gt_face = None
            if gt_path and gt_path.exists():
                gt_face = self.load_face_pred_from_json(gt_path)
            
            # 3. 可视化
            saved_paths = []
            if gt_face is not None:
                # 有真值：进行对比可视化
                saved_paths = self.visualizer.plot_all_categories(
                    pred_face,
                    gt_face,
                    audio_name,
                    output_dir=str(self.visualization_dir / audio_name)
                )
                
                # 创建仪表板
                dashboard_path = self.visualization_dir / f"{audio_name}_dashboard.png"
                category_errors, category_correlations = self.visualizer.plot_summary_dashboard(
                    pred_face,
                    gt_face,
                    audio_name,
                    str(dashboard_path)
                )
            else:
                # 只有预测：单序列可视化
                self.visualizer.plot_all_categories(
                    pred_face,
                    None,  # 没有真值
                    audio_name,
                    output_dir=str(self.visualization_dir / audio_name)
                )
            
            # 4. 计算评估指标（如果有真值）
            eval_metrics = {}
            category_errors = {}
            category_correlations = {}
            
            if gt_face is not None:
                # 确保维度匹配
                if pred_face.shape[1] != gt_face.shape[1]:
                    self.logger.warning(
                        f"维度不匹配: 预测{pred_face.shape[1]}, 真值{gt_face.shape[1]}"
                    )
                    min_dim = min(pred_face.shape[1], gt_face.shape[1])
                    pred_face = pred_face[:, :min_dim]
                    gt_face = gt_face[:, :min_dim]
                
                # 计算全局指标
                eval_metrics = self.compute_metrics(pred_face, gt_face)
                
                # 保存指标
                if eval_metrics:
                    metrics_path = self.metrics_dir / f"{audio_name}_metrics.json"
                    with open(metrics_path, 'w') as f:
                        json.dump(eval_metrics, f, indent=2)
            
            # 5. 收集结果
            result = {
                'audio_name': audio_name,
                'pred_file': pred_path.name,
                'gt_file': gt_path.name if gt_path else None,
                'has_gt': gt_face is not None,
                'pred_shape': list(pred_face.shape),
                'gt_shape': list(gt_face.shape) if gt_face is not None else None,
                'frame_count': len(pred_face),
                'fps': self.fps
            }
            
            # 添加评估指标
            result.update(eval_metrics)
            
            # 添加类别指标
            if gt_face is not None:
                for category, error in category_errors.items():
                    result[f'{category}_mae'] = error
                for category, corr in category_correlations.items():
                    result[f'{category}_corr'] = corr
            
            self.logger.info(f"完成评估: {audio_name}")
            return result
            
        except Exception as e:
            self.logger.error(f"评估样本失败 {audio_name}: {str(e)}")
            self.logger.error(traceback.format_exc())
            return None
    
    def run_evaluation(self, result_name: str = "result_evaluation") -> pd.DataFrame:
        """
        运行完整的结果评估
        
        Returns:
            包含所有评估结果的DataFrame
        """
        # 获取所有预测文件
        pred_files = list(self.pred_dir.glob("transformer_pred_*.json"))
        if not pred_files:
            # 也支持没有前缀的文件
            pred_files = list(self.pred_dir.glob("*.json"))
        
        if not pred_files:
            raise FileNotFoundError(f"在 {self.pred_dir} 中没有找到JSON文件")
        
        self.logger.info(f"找到 {len(pred_files)} 个预测文件")
        
        # 处理每个预测文件
        all_results = []
        
        for pred_path in pred_files:
            # 提取音频名称
            audio_name = self.extract_filename_from_pred(pred_path.name)
            
            # 查找匹配的真值文件
            gt_path = None
            if self.gt_dir.exists():
                gt_path = self.find_matching_gt_file(audio_name)
                if not gt_path:
                    self.logger.warning(f"未找到 {audio_name} 的真值文件")
            
            # 评估单个样本
            result = self.evaluate_single_pair(pred_path, gt_path)
            if result:
                all_results.append(result)
        
        # 生成报告
        report = self._generate_evaluation_report(all_results, result_name)
        
        return report
    
    def _generate_evaluation_report(self, all_results: List[Dict], result_name: str) -> pd.DataFrame:
        """生成评估报告"""
        # 转换为DataFrame
        df = pd.DataFrame(all_results)
        
        if df.empty:
            self.logger.warning("没有有效的评估结果")
            return df
        
        # 汇总统计
        summary = {
            'total_samples': len(df),
            'samples_with_gt': df['has_gt'].sum() if 'has_gt' in df.columns else 0,
            'total_frame_count': df['frame_count'].sum() if 'frame_count' in df.columns else 0,
            'avg_frame_count': df['frame_count'].mean() if 'frame_count' in df.columns else 0,
        }
        
        # 添加评估指标统计（如果有GT）
        if 'mse' in df.columns and df['has_gt'].any():
            gt_samples = df[df['has_gt']]
            summary['avg_mse'] = gt_samples['mse'].mean()
            summary['avg_mae'] = gt_samples['mae'].mean()
            summary['avg_rmse'] = gt_samples['rmse'].mean()
            summary['avg_correlation'] = gt_samples['avg_correlation'].mean()
            summary['samples_with_gt'] = len(gt_samples)
        
        # 保存汇总报告
        summary_path = self.summary_dir / f"{result_name}_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # 保存详细结果CSV
        csv_path = self.summary_dir / f"{result_name}_detailed.csv"
        df.to_csv(csv_path, index=False)
        
        # 创建可视化报告
        self._create_evaluation_summary_visualization(df, summary, result_name)
        
        self.logger.info(f"评估报告已保存到: {self.summary_dir}")
        self.logger.info(f"汇总结果: {summary}")
        
        return df
    
    def _create_evaluation_summary_visualization(self, df: pd.DataFrame, summary: Dict, result_name: str):
        """创建评估汇总可视化"""
        try:
            # 确定子图数量
            has_gt = 'has_gt' in df.columns and df['has_gt'].any()
            num_plots = 2 if has_gt else 1
            
            fig, axes = plt.subplots(1, num_plots, figsize=(5*num_plots, 4))
            if num_plots == 1:
                axes = [axes]
            
            # 1. 帧数分布
            ax = axes[0]
            if 'frame_count' in df.columns:
                ax.hist(df['frame_count'], bins=10, alpha=0.7, color='skyblue', edgecolor='black')
                ax.axvline(df['frame_count'].mean(), color='red', linestyle='--',
                          label=f'均值: {df["frame_count"].mean():.0f}')
                ax.set_xlabel('帧数')
                ax.set_ylabel('样本数')
                ax.set_title('帧数分布')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            # 2. 如果有GT，显示MSE分布
            if has_gt and 'mse' in df.columns:
                ax = axes[1] if num_plots > 1 else axes[0]
                valid_mse = df[df['has_gt']]['mse']
                ax.hist(valid_mse, bins=10, alpha=0.7, color='salmon', edgecolor='black')
                ax.axvline(valid_mse.mean(), color='red', linestyle='--',
                          label=f'均值: {valid_mse.mean():.4f}')
                ax.set_xlabel('MSE')
                ax.set_ylabel('样本数')
                ax.set_title('均方误差分布 (越低越好)')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            plt.suptitle(f'结果评估汇总 - {result_name}', fontsize=14)
            plt.tight_layout()
            
            summary_viz_path = self.summary_dir / f"{result_name}_summary.png"
            plt.savefig(summary_viz_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
        except Exception as e:
            self.logger.warning(f"创建评估汇总可视化失败: {str(e)}")
    
    def compare_multiple_results(
        self, 
        result_dirs: Dict[str, str], 
        gt_dir: str,
        comparison_name: str = "multi_model_comparison"
    ):
        """
        比较多个不同模型/配置的推理结果
        
        Args:
            result_dirs: 字典，key为模型名称，value为结果目录路径
            gt_dir: 真值目录
            comparison_name: 比较实验名称
        """
        self.logger.info(f"开始多结果比较: {list(result_dirs.keys())}")
        
        comparison_dir = self.output_dir / "comparisons" / comparison_name
        comparison_dir.mkdir(exist_ok=True, parents=True)
        
        all_comparison_results = []
        
        for model_name, result_dir in result_dirs.items():
            self.logger.info(f"处理模型: {model_name}")
            
            # 为该模型创建评估器
            model_evaluator = Audio2FaceResultEvaluator(
                pred_dir=result_dir,
                gt_dir=gt_dir,
                output_dir=str(comparison_dir / model_name),
                param_mapping_path=self.visualizer.param_mapping_path,
                max_params_per_category=self.visualizer.max_params_per_category,
                fps=self.fps
            )
            
            # 运行评估
            results_df = model_evaluator.run_evaluation(result_name=model_name)
            
            # 添加模型名称列
            if not results_df.empty:
                results_df['model'] = model_name
                all_comparison_results.append(results_df)
        
        # 合并所有结果
        if all_comparison_results:
            combined_df = pd.concat(all_comparison_results, ignore_index=True)
            
            # 保存合并结果
            combined_csv = comparison_dir / f"{comparison_name}_combined.csv"
            combined_df.to_csv(combined_csv, index=False)
            
            # 创建比较图表
            self._create_comparison_visualization(combined_df, comparison_dir, comparison_name)
            
            self.logger.info(f"多模型比较完成，结果保存在: {comparison_dir}")
            
            return combined_df
        
        return pd.DataFrame()
    
    def _create_comparison_visualization(self, df: pd.DataFrame, output_dir: Path, comparison_name: str):
        """创建多模型比较可视化"""
        try:
            if 'model' not in df.columns or 'mse' not in df.columns:
                return
            
            # 模型性能对比（箱线图）
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # 1. MSE对比
            models = df['model'].unique()
            mse_data = [df[df['model'] == model]['mse'].dropna() for model in models]
            
            bp1 = axes[0].boxplot(mse_data, labels=models, patch_artist=True)
            for patch, color in zip(bp1['boxes'], plt.cm.Set3.colors):
                patch.set_facecolor(color)
            axes[0].set_title('MSE对比 (越低越好)')
            axes[0].set_ylabel('MSE')
            axes[0].grid(True, alpha=0.3, axis='y')
            axes[0].tick_params(axis='x', rotation=45)
            
            # 2. 相关系数对比
            if 'avg_correlation' in df.columns:
                corr_data = [df[df['model'] == model]['avg_correlation'].dropna() for model in models]
                bp2 = axes[1].boxplot(corr_data, labels=models, patch_artist=True)
                for patch, color in zip(bp2['boxes'], plt.cm.Set3.colors):
                    patch.set_facecolor(color)
                axes[1].set_title('平均相关系数对比 (越高越好)')
                axes[1].set_ylabel('相关系数')
                axes[1].grid(True, alpha=0.3, axis='y')
                axes[1].tick_params(axis='x', rotation=45)
            
            plt.suptitle(f'多模型性能对比 - {comparison_name}', fontsize=14)
            plt.tight_layout()
            
            comparison_path = output_dir / f"{comparison_name}_performance_comparison.png"
            plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
        except Exception as e:
            self.logger.warning(f"创建比较可视化失败: {str(e)}")


def main_backup():
    """主函数"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="path to config file")
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    
    # 配置参数
    CONFIG = {
        'model': {"input_dim": 768, "output_dim": 136, "num_layers": 11, "num_heads": 24},
        'model_weights': "./Weights/transformer_decoder_V3.pth",  # 模型权重路径
        'wav2vec_path': "./wav2vec-base-960h",
        'test_data_dir': "./test",  # 测试数据目录
        'output_dir': "./test_results",  # 输出目录
        'device': None,  # 设备: "cuda" 或 "cpu"，None表示自动选择
        'param_mapping_path': "./ctrl_expressions_map.json",
        'max_params_per_category': 10,
        'fps': 25,  # 输出帧率
        'result_name': "audio2face_evaluation"  # 测试结果名称
    }
    
    try:
        # 创建测试器
        '''
        tester = Audio2FaceTester(
            model=model,
            model_weights_path=CONFIG['model_weights'],
            wav2vec_path=CONFIG['wav2vec_path'],
            device=CONFIG['device'],
            output_dir=CONFIG['output_dir'],
            param_mapping_path=CONFIG['param_mapping_path'],
            max_params_per_category=CONFIG['max_params_per_category'],
            fps=CONFIG['fps']
        )
        
        # 运行测试套件
        results_df = tester.run_test_suite(
            test_data_dir=CONFIG['test_data_dir'],
            result_name=CONFIG['result_name']
        )
        '''
        tester = Audio2FaceTester(
            model=config.model,
            model_weights_path=config.model_weights,
            wav2vec_path=config.wav2vec.path,
            device=config.device,
            output_dir=config.tester.output_dir,
            param_mapping_path=config.visualizer.param_mapping_path,
            max_params_per_category=config.visualizer.max_params_per_category,
            fps=config.visualizer.fps
        )
        
        # 运行测试套件
        results_df = tester.run_test_suite(
            test_data_dir=config.tester.test_data_dir,
            wav_dir=config.tester.wav_dir,
            gt_json_dir=config.tester.gt_json_dir,
            result_name=config.visualizer.result_name
        )
        
        # 打印总结
        print("\n" + "="*60)
        print("测试完成!")
        print("="*60)
        
        if not results_df.empty:
            print(f"总测试样本数: {len(results_df)}")
            print(f"有GT的样本数: {results_df['has_gt'].sum() if 'has_gt' in results_df.columns else 0}")
            print(f"平均推理时间: {results_df['inference_time'].mean():.2f}s")
            print(f"平均实时比: {results_df['real_time_factor'].mean():.2f}")
            
            if 'mse' in results_df.columns and results_df['has_gt'].any():
                gt_samples = results_df[results_df['has_gt']]
                print(f"平均MSE: {gt_samples['mse'].mean():.6f}")
                print(f"平均MAE: {gt_samples['mae'].mean():.6f}")
                print(f"平均相关系数: {gt_samples['avg_correlation'].mean():.4f}")
        
        print(f"详细结果保存在: {CONFIG['output_dir']}")
        print("="*60)
        
    except Exception as e:
        print(f"测试失败: {str(e)}")
        traceback.print_exc()

def main():
    """统一的主函数，根据配置选择模式"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="path to config file")
    parser.add_argument("--mode", type=str, choices=["inference", "evaluate", "compare"], nargs='?', const=None,
                       help="override mode in config file")
    args = parser.parse_args()

    # 加载配置
    config = OmegaConf.load(args.config)
    
    # 命令行参数覆盖配置文件
    if args.mode:
        config.mode = args.mode
    
    # 根据模式执行不同流程
    if config.evaluate_mode == "inference":
        run_inference_mode(config)
    elif config.evaluate_mode == "evaluate":
        run_evaluate_mode(config)
    elif config.evaluate_mode == "compare":
        run_compare_mode(config)
    else:
        print(f"未知的模式: {config.mode}")
        print("可用模式: inference, evaluate, compare")
        sys.exit(1)

def run_inference_mode(config):
    """运行推理模式"""
    print("=" * 60)
    print("运行推理模式")
    print("=" * 60)

    try:
        # 创建测试器
        tester = Audio2FaceTester(
            model=config.model,
            model_weights_path=config.model_weights,
            wav2vec_path=config.wav2vec.path,
            device=config.device,
            output_dir=config.tester.output_dir,
            param_mapping_path=config.visualizer.param_mapping_path,
            max_params_per_category=config.visualizer.max_params_per_category,
            fps=config.visualizer.fps
        )
        
        # 运行测试套件
        results_df = tester.run_test_suite(
            test_data_dir=config.tester.test_data_dir,
            wav_dir=config.tester.wav_dir,
            gt_json_dir=config.tester.gt_json_dir,
            result_name=config.visualizer.result_name
        )
        
        # 打印总结
        print_results_summary(results_df, "推理模式")
        
    except Exception as e:
        print(f"推理模式失败: {str(e)}")
        traceback.print_exc()

def run_evaluate_mode(config):
    """运行结果评估模式"""
    print("=" * 60)
    print("运行结果评估模式")
    print("=" * 60)
    
    try:
        # 创建结果评估器
        evaluator = Audio2FaceResultEvaluator(
            pred_dir=config.evaluator.pred_dir,
            gt_dir=config.evaluator.gt_dir,
            output_dir=config.evaluator.output_dir,
            param_mapping_path=config.visualizer.param_mapping_path,
            max_params_per_category=config.visualizer.max_params_per_category,
            fps=config.visualizer.fps
        )
        
        # 运行评估
        results_df = evaluator.run_evaluation(
            result_name=config.visualizer.result_name
        )
        
        # 打印总结
        print_results_summary(results_df, "结果评估模式")
        
    except Exception as e:
        print(f"结果评估模式失败: {str(e)}")
        traceback.print_exc()

def run_compare_mode(config):
    """运行多模型比较模式"""
    print("=" * 60)
    print("运行多模型比较模式")
    print("=" * 60)
    
    try:
        # 使用第一个模型目录初始化评估器（用于获取公共配置）
        first_model_dir = list(config.comparison.result_dirs.values())[0]
        
        evaluator = Audio2FaceResultEvaluator(
            pred_dir=first_model_dir,
            gt_dir=config.comparison.gt_dir,
            output_dir=config.comparison.output_dir,
            param_mapping_path=config.visualizer.param_mapping_path,
            max_params_per_category=config.visualizer.max_params_per_category,
            fps=config.visualizer.fps
        )
        
        # 运行多模型比较
        combined_results = evaluator.compare_multiple_results(
            result_dirs=config.comparison.result_dirs,
            gt_dir=config.comparison.gt_dir,
            comparison_name=config.comparison.comparison_name
        )
        
        print("\n多模型比较完成!")
        print(f"比较结果保存在: {config.comparison.output_dir}/comparisons/{config.comparison.comparison_name}")
        
    except Exception as e:
        print(f"多模型比较失败: {str(e)}")
        traceback.print_exc()

def print_results_summary(df, mode_name):
    """打印结果总结"""
    if df.empty:
        print("没有有效的测试结果")
        return
    
    print(f"\n{mode_name}完成!")
    print("-" * 40)
    print(f"总样本数: {len(df)}")
    print(f"有GT的样本数: {df['has_gt'].sum() if 'has_gt' in df.columns else 0}")
    
    if 'inference_time' in df.columns:
        print(f"平均推理时间: {df['inference_time'].mean():.2f}s")
        print(f"平均实时比: {df['real_time_factor'].mean():.2f}")
    
    if 'mse' in df.columns and 'has_gt' in df.columns and df['has_gt'].any():
        gt_samples = df[df['has_gt']]
        print(f"平均MSE: {gt_samples['mse'].mean():.6f}")
        print(f"平均MAE: {gt_samples['mae'].mean():.6f}")
        print(f"平均相关系数: {gt_samples['avg_correlation'].mean():.4f}")
    
    print("-" * 40)

if __name__ == "__main__":
    main()
