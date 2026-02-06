from dataclasses import asdict
from typing import List, Dict, Any
from loguru import logger
from torch.utils.data import DataLoader
import gc

from npuslim.utils.backend import bh
from npuslim.utils.config_parser import GlobalConfig
from npuslim.utils.factory import ModelFactory, DatasetFactory, TaskFactory
from npuslim.tasks.base_task import BaseTask


class SlimEngine:
    def __init__(self):
        # 1. 获取全局配置
        self.cfg = GlobalConfig.get_config()

        # 2. 初始化资源池 (Resource Pool)
        # 这里存放所有任务可能共享的对象，避免重复加载
        self.resources: Dict[str, Any] = {
            "main_model": None,   # 老师模型 / 待量化模型 / 待剪枝模型
            "draft_model": None,  # 投机采样草稿模型
            "student_model": None,# 蒸馏学生模型
            "dataloader": None,   # 校准/通用数据加载器
            "tokenizer": None,
            "engine": self,       # 允许 Task 反向访问 Engine
        }

        # 3. 任务流水线
        self.pipeline: List[BaseTask] = []
        
        # === 启动流程 ===
        self.prepare_resources()
        self.build_pipeline()

    def prepare_resources(self):
        """
        准备全局共享资源。
        策略：加载 Config 中定义的所有模型和数据。
        """
        logger.info("🛠️ [Resource] Preparing global resources...")

        # --- 1. 加载主模型 (Target/Teacher Model) ---
        # 这里的 main_model 是核心对象
        self.resources["main_model"] = self._init_model(self.cfg.model)

        # 提取 Tokenizer (供 Dataset 使用)
        model_instance = self.resources["main_model"]
        self.resources["tokenizer"] = getattr(model_instance, "processor", model_instance.tokenizer)

        # --- 2. 加载辅助模型 (Draft/Student) ---
        
        # [投机采样资源]
        if self.cfg.speculative:
            logger.info("🛰️ [Resource] Speculative config detected.")
            draft_cfg_dict = self.cfg.speculative.get("draft_model")
            if draft_cfg_dict:
                # 投机采样需要草稿模型
                self.resources["draft_model"] = self._init_model(draft_cfg_dict)
        
        # [蒸馏/微调资源]
        # 如果是蒸馏任务，main_model 通常充当 Teacher，这里加载 Student
        if self.cfg.distillation:
            logger.info("🧪 [Resource] Distillation config detected.")
            student_cfg_dict = self.cfg.distillation.get("student_model")
            if student_cfg_dict:
                self.resources["student_model"] = self._init_model(student_cfg_dict)

        # --- 3. 加载数据集 ---
        # 通常用于 PTQ 校准或通用评估
        if self.cfg.calib_dataset:
            self._prepare_dataloader()

    def _init_model(self, model_cfg):
        """通用模型初始化辅助函数"""
        # ModelFactory 负责处理 Config 对象或字典
        model = ModelFactory.create(config=model_cfg)
        model.prepare()
        return model

    def _prepare_dataloader(self):
        """初始化统一的数据加载器"""
        logger.info("📊 [Resource] Loading calibration dataset...")
        
        dataset_cfg = self.cfg.calib_dataset.dataset
        loader_cfg = self.cfg.calib_dataset.dataloader
        
        # 创建 Dataset
        dataset = DatasetFactory.create(
            processor=self.resources["tokenizer"], 
            config=dataset_cfg
        )
        
        # 兼容 loader_cfg 是 dataclass 的情况
        loader_kwargs = loader_cfg if isinstance(loader_cfg, dict) else asdict(loader_cfg)
        
        self.resources["dataloader"] = DataLoader(
            dataset, 
            collate_fn=dataset.collate_fn, 
            **loader_kwargs
        )

    def build_pipeline(self):
        """
        核心调度逻辑：基于 Config 中的 pipeline 列表构建任务链。
        """
        logger.info("🏗️ [Pipeline] Building execution sequence...")
        
        # 获取 pipeline 列表 (List[Dict])
        pipeline_configs = self.cfg.pipeline
        
        if not pipeline_configs:
            logger.warning("⚠️ Pipeline is empty in configuration.")
            return

        for i, task_cfg in enumerate(pipeline_configs):
            # 获取任务类型标识 (如 'ptq', 'sparse', 'sft', 'speculative_eval')
            task_type = task_cfg.get("type")
            if not task_type:
                logger.error(f"❌ Pipeline item #{i} missing 'type' field.")
                continue

            logger.info(f"➕ Adding Step {i+1}: {task_type}")
            
            # 使用工厂创建任务实例
            try:
                task = TaskFactory.create(
                    task_key=task_type,
                    raw_config=task_cfg,
                    resources=self.resources
                )
                self.pipeline.append(task)
            except Exception as e:
                logger.error(f"Failed to create task '{task_type}': {e}")
                raise e

    def run(self):
        """执行流水线"""
        if not self.pipeline:
            logger.warning("🚫 Pipeline is empty. Nothing to run.")
            return

        logger.info(f"🏁 Starting execution of {len(self.pipeline)} tasks...")

        for idx, task in enumerate(self.pipeline):
            task_name = task.__class__.__name__
            logger.info(f"▶️ [Step {idx+1}/{len(self.pipeline)}] Running {task_name}...")
            
            # 任务执行
            task.execute()
            # 任务间隙显存清理
            self._clear_memory()

        logger.success("✨ All tasks completed successfully.")
    
    def _clear_memory(self):
        """强制垃圾回收和显存释放"""
        gc.collect()
        bh.empty_cache()