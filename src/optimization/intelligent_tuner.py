"""
智能调优系统
自动优化超参数、模型选择、特征工程等
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
import json
import time
import logging
from dataclasses import dataclass, asdict
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed
import itertools
import random
from abc import ABC, abstractmethod

# 优化算法导入
try:
    from scipy.optimize import minimize, differential_evolution, basinhopping
    from scipy.stats import uniform, randint
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

try:
    from sklearn.model_selection import cross_val_score, ParameterGrid, RandomizedSearchCV
    from sklearn.metrics import mean_squared_error, accuracy_score, f1_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

logger = logging.getLogger(__name__)


class OptimizationMethod(Enum):
    """优化方法"""
    GRID_SEARCH = "grid_search"
    RANDOM_SEARCH = "random_search"
    BAYESIAN = "bayesian"
    GENETIC = "genetic"
    PARTICLE_SWARM = "particle_swarm"
    SIMULATED_ANNEALING = "simulated_annealing"
    DIFFERENTIAL_EVOLUTION = "differential_evolution"
    OPTUNA = "optuna"


class OptimizationObjective(Enum):
    """优化目标"""
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    ROC_AUC = "roc_auc"
    MSE = "mse"
    RMSE = "rmse"
    MAE = "mae"
    R2_SCORE = "r2_score"
    CUSTOM = "custom"


@dataclass
class OptimizationResult:
    """优化结果"""
    best_params: Dict[str, Any]
    best_score: float
    optimization_history: List[Dict[str, Any]]
    method: str
    objective: str
    total_evaluations: int
    optimization_time: float
    convergence_info: Dict[str, Any]


@dataclass
class ParameterSpace:
    """参数空间定义"""
    name: str
    param_type: str  # 'int', 'float', 'categorical', 'bool'
    bounds: Optional[Tuple[Any, Any]] = None  # (min, max) for numeric
    choices: Optional[List[Any]] = None  # for categorical
    log_scale: bool = False  # 是否使用对数尺度


class BaseOptimizer(ABC):
    """优化器基类"""
    
    def __init__(self, objective_function: Callable, parameter_spaces: List[ParameterSpace]):
        self.objective_function = objective_function
        self.parameter_spaces = parameter_spaces
        self.optimization_history = []
        self.best_params = None
        self.best_score = float('-inf')
        self.evaluation_count = 0
    
    @abstractmethod
    def optimize(self, max_evaluations: int = 100, **kwargs) -> OptimizationResult:
        """执行优化"""
        pass
    
    def _evaluate(self, params: Dict[str, Any]) -> float:
        """评估参数组合"""
        try:
            score = self.objective_function(params)
            self.evaluation_count += 1
            
            # 记录历史
            self.optimization_history.append({
                'evaluation': self.evaluation_count,
                'params': params.copy(),
                'score': score,
                'timestamp': time.time()
            })
            
            # 更新最佳结果
            if score > self.best_score:
                self.best_score = score
                self.best_params = params.copy()
            
            return score
            
        except Exception as e:
            logger.error(f"参数评估失败: {e}")
            return float('-inf')
    
    def _generate_random_params(self) -> Dict[str, Any]:
        """生成随机参数"""
        params = {}
        
        for space in self.parameter_spaces:
            if space.param_type == 'int':
                if space.log_scale:
                    log_min = np.log10(space.bounds[0])
                    log_max = np.log10(space.bounds[1])
                    value = int(10 ** np.random.uniform(log_min, log_max))
                else:
                    value = np.random.randint(space.bounds[0], space.bounds[1] + 1)
                params[space.name] = value
                
            elif space.param_type == 'float':
                if space.log_scale:
                    log_min = np.log10(space.bounds[0])
                    log_max = np.log10(space.bounds[1])
                    value = 10 ** np.random.uniform(log_min, log_max)
                else:
                    value = np.random.uniform(space.bounds[0], space.bounds[1])
                params[space.name] = value
                
            elif space.param_type == 'categorical':
                params[space.name] = np.random.choice(space.choices)
                
            elif space.param_type == 'bool':
                params[space.name] = np.random.choice([True, False])
        
        return params


class GridSearchOptimizer(BaseOptimizer):
    """网格搜索优化器"""
    
    def optimize(self, max_evaluations: int = 100, **kwargs) -> OptimizationResult:
        """执行网格搜索"""
        start_time = time.time()
        
        # 生成参数网格
        param_grid = {}
        for space in self.parameter_spaces:
            if space.param_type == 'int':
                step = max(1, (space.bounds[1] - space.bounds[0]) // 10)
                param_grid[space.name] = list(range(space.bounds[0], space.bounds[1] + 1, step))
            elif space.param_type == 'float':
                param_grid[space.name] = np.linspace(space.bounds[0], space.bounds[1], 10).tolist()
            elif space.param_type == 'categorical':
                param_grid[space.name] = space.choices
            elif space.param_type == 'bool':
                param_grid[space.name] = [True, False]
        
        # 生成所有参数组合
        keys = param_grid.keys()
        values = param_grid.values()
        param_combinations = [dict(zip(keys, combination)) for combination in itertools.product(*values)]
        
        # 限制评估次数
        if len(param_combinations) > max_evaluations:
            param_combinations = random.sample(param_combinations, max_evaluations)
        
        # 评估所有组合
        for params in param_combinations:
            self._evaluate(params)
        
        optimization_time = time.time() - start_time
        
        return OptimizationResult(
            best_params=self.best_params,
            best_score=self.best_score,
            optimization_history=self.optimization_history,
            method="grid_search",
            objective="custom",
            total_evaluations=self.evaluation_count,
            optimization_time=optimization_time,
            convergence_info={'total_combinations': len(param_combinations)}
        )


class RandomSearchOptimizer(BaseOptimizer):
    """随机搜索优化器"""
    
    def optimize(self, max_evaluations: int = 100, **kwargs) -> OptimizationResult:
        """执行随机搜索"""
        start_time = time.time()
        
        for _ in range(max_evaluations):
            params = self._generate_random_params()
            self._evaluate(params)
        
        optimization_time = time.time() - start_time
        
        return OptimizationResult(
            best_params=self.best_params,
            best_score=self.best_score,
            optimization_history=self.optimization_history,
            method="random_search",
            objective="custom",
            total_evaluations=self.evaluation_count,
            optimization_time=optimization_time,
            convergence_info={'max_evaluations': max_evaluations}
        )


class BayesianOptimizer(BaseOptimizer):
    """贝叶斯优化器（使用Optuna）"""
    
    def optimize(self, max_evaluations: int = 100, **kwargs) -> OptimizationResult:
        """执行贝叶斯优化"""
        if not OPTUNA_AVAILABLE:
            logger.error("Optuna不可用，无法执行贝叶斯优化")
            return self._fallback_to_random_search(max_evaluations)
        
        start_time = time.time()
        
        def objective(trial):
            params = {}
            for space in self.parameter_spaces:
                if space.param_type == 'int':
                    if space.log_scale:
                        params[space.name] = trial.suggest_int(
                            space.name, space.bounds[0], space.bounds[1], log=True
                        )
                    else:
                        params[space.name] = trial.suggest_int(
                            space.name, space.bounds[0], space.bounds[1]
                        )
                elif space.param_type == 'float':
                    if space.log_scale:
                        params[space.name] = trial.suggest_float(
                            space.name, space.bounds[0], space.bounds[1], log=True
                        )
                    else:
                        params[space.name] = trial.suggest_float(
                            space.name, space.bounds[0], space.bounds[1]
                        )
                elif space.param_type == 'categorical':
                    params[space.name] = trial.suggest_categorical(space.name, space.choices)
                elif space.param_type == 'bool':
                    params[space.name] = trial.suggest_categorical(space.name, [True, False])
            
            return self._evaluate(params)
        
        # 创建研究
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=max_evaluations)
        
        optimization_time = time.time() - start_time
        
        return OptimizationResult(
            best_params=study.best_params,
            best_score=study.best_value,
            optimization_history=self.optimization_history,
            method="bayesian",
            objective="custom",
            total_evaluations=len(study.trials),
            optimization_time=optimization_time,
            convergence_info={
                'n_trials': len(study.trials),
                'best_trial': study.best_trial.number
            }
        )
    
    def _fallback_to_random_search(self, max_evaluations: int) -> OptimizationResult:
        """回退到随机搜索"""
        logger.warning("回退到随机搜索优化")
        random_optimizer = RandomSearchOptimizer(self.objective_function, self.parameter_spaces)
        return random_optimizer.optimize(max_evaluations)


class GeneticOptimizer(BaseOptimizer):
    """遗传算法优化器"""
    
    def optimize(self, max_evaluations: int = 100, population_size: int = 20, 
                mutation_rate: float = 0.1, crossover_rate: float = 0.8, **kwargs) -> OptimizationResult:
        """执行遗传算法优化"""
        start_time = time.time()
        
        # 初始化种群
        population = [self._generate_random_params() for _ in range(population_size)]
        fitness_scores = [self._evaluate(params) for params in population]
        
        generations = max_evaluations // population_size
        
        for generation in range(generations):
            # 选择
            selected = self._selection(population, fitness_scores, population_size // 2)
            
            # 交叉和变异
            offspring = []
            for i in range(0, len(selected), 2):
                parent1 = selected[i]
                parent2 = selected[i + 1] if i + 1 < len(selected) else selected[0]
                
                if random.random() < crossover_rate:
                    child1, child2 = self._crossover(parent1, parent2)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()
                
                if random.random() < mutation_rate:
                    child1 = self._mutate(child1)
                if random.random() < mutation_rate:
                    child2 = self._mutate(child2)
                
                offspring.extend([child1, child2])
            
            # 评估后代
            offspring_fitness = [self._evaluate(params) for params in offspring]
            
            # 生成新一代
            all_individuals = population + offspring
            all_fitness = fitness_scores + offspring_fitness
            
            # 选择最佳个体
            sorted_indices = sorted(range(len(all_fitness)), key=lambda i: all_fitness[i], reverse=True)
            population = [all_individuals[i] for i in sorted_indices[:population_size]]
            fitness_scores = [all_fitness[i] for i in sorted_indices[:population_size]]
        
        optimization_time = time.time() - start_time
        
        return OptimizationResult(
            best_params=self.best_params,
            best_score=self.best_score,
            optimization_history=self.optimization_history,
            method="genetic",
            objective="custom",
            total_evaluations=self.evaluation_count,
            optimization_time=optimization_time,
            convergence_info={
                'generations': generations,
                'population_size': population_size,
                'final_best_fitness': max(fitness_scores)
            }
        )
    
    def _selection(self, population: List[Dict], fitness_scores: List[float], num_selected: int) -> List[Dict]:
        """选择操作（锦标赛选择）"""
        selected = []
        tournament_size = 3
        
        for _ in range(num_selected):
            tournament_indices = random.sample(range(len(population)), min(tournament_size, len(population)))
            winner_idx = max(tournament_indices, key=lambda i: fitness_scores[i])
            selected.append(population[winner_idx].copy())
        
        return selected
    
    def _crossover(self, parent1: Dict, parent2: Dict) -> Tuple[Dict, Dict]:
        """交叉操作"""
        child1 = parent1.copy()
        child2 = parent2.copy()
        
        # 随机选择一些参数进行交换
        for space in self.parameter_spaces:
            if random.random() < 0.5:
                child1[space.name], child2[space.name] = child2[space.name], child1[space.name]
        
        return child1, child2
    
    def _mutate(self, individual: Dict) -> Dict:
        """变异操作"""
        mutated = individual.copy()
        
        # 随机变异一些参数
        for space in self.parameter_spaces:
            if random.random() < 0.3:  # 30%的参数变异概率
                if space.param_type == 'int':
                    # 在原值附近变异
                    current_val = mutated[space.name]
                    mutation_range = max(1, (space.bounds[1] - space.bounds[0]) // 10)
                    new_val = current_val + random.randint(-mutation_range, mutation_range)
                    mutated[space.name] = max(space.bounds[0], min(space.bounds[1], new_val))
                    
                elif space.param_type == 'float':
                    # 在原值附近变异
                    current_val = mutated[space.name]
                    mutation_range = (space.bounds[1] - space.bounds[0]) * 0.1
                    new_val = current_val + random.uniform(-mutation_range, mutation_range)
                    mutated[space.name] = max(space.bounds[0], min(space.bounds[1], new_val))
                    
                elif space.param_type == 'categorical':
                    mutated[space.name] = random.choice(space.choices)
                    
                elif space.param_type == 'bool':
                    mutated[space.name] = not mutated[space.name]
        
        return mutated


class ParticleSwarmOptimizer(BaseOptimizer):
    """粒子群优化器"""
    
    def optimize(self, max_evaluations: int = 100, swarm_size: int = 20, 
                w: float = 0.5, c1: float = 1.5, c2: float = 1.5, **kwargs) -> OptimizationResult:
        """执行粒子群优化"""
        start_time = time.time()
        
        # 初始化粒子群
        particles = []
        for _ in range(swarm_size):
            position = self._generate_random_params()
            velocity = {}
            
            # 初始化速度
            for space in self.parameter_spaces:
                if space.param_type in ['int', 'float']:
                    velocity[space.name] = 0.0
                else:
                    velocity[space.name] = None
            
            particles.append({
                'position': position,
                'velocity': velocity,
                'best_position': position.copy(),
                'best_score': self._evaluate(position)
            })
        
        # 全局最佳位置
        global_best_particle = max(particles, key=lambda p: p['best_score'])
        global_best_position = global_best_particle['best_position'].copy()
        global_best_score = global_best_particle['best_score']
        
        iterations = max_evaluations // swarm_size
        
        for iteration in range(iterations):
            for particle in particles:
                # 更新速度和位置
                self._update_particle(particle, global_best_position, w, c1, c2)
                
                # 评估新位置
                score = self._evaluate(particle['position'])
                
                # 更新个体最佳
                if score > particle['best_score']:
                    particle['best_score'] = score
                    particle['best_position'] = particle['position'].copy()
                
                # 更新全局最佳
                if score > global_best_score:
                    global_best_score = score
                    global_best_position = particle['position'].copy()
        
        optimization_time = time.time() - start_time
        
        return OptimizationResult(
            best_params=global_best_position,
            best_score=global_best_score,
            optimization_history=self.optimization_history,
            method="particle_swarm",
            objective="custom",
            total_evaluations=self.evaluation_count,
            optimization_time=optimization_time,
            convergence_info={
                'iterations': iterations,
                'swarm_size': swarm_size,
                'final_global_best': global_best_score
            }
        )
    
    def _update_particle(self, particle: Dict, global_best_position: Dict, 
                        w: float, c1: float, c2: float):
        """更新粒子位置和速度"""
        for space in self.parameter_spaces:
            if space.param_type in ['int', 'float']:
                # 更新速度
                r1, r2 = random.random(), random.random()
                
                cognitive = c1 * r1 * (particle['best_position'][space.name] - particle['position'][space.name])
                social = c2 * r2 * (global_best_position[space.name] - particle['position'][space.name])
                
                particle['velocity'][space.name] = (
                    w * particle['velocity'][space.name] + cognitive + social
                )
                
                # 更新位置
                new_position = particle['position'][space.name] + particle['velocity'][space.name]
                
                # 边界处理
                if space.param_type == 'int':
                    new_position = int(round(new_position))
                    new_position = max(space.bounds[0], min(space.bounds[1], new_position))
                else:  # float
                    new_position = max(space.bounds[0], min(space.bounds[1], new_position))
                
                particle['position'][space.name] = new_position
            
            else:  # categorical or bool
                # 对于离散变量，随机选择
                if random.random() < 0.1:  # 10%的概率改变
                    if space.param_type == 'categorical':
                        particle['position'][space.name] = random.choice(space.choices)
                    else:  # bool
                        particle['position'][space.name] = not particle['position'][space.name]


class IntelligentTuner:
    """智能调优主类"""
    
    def __init__(self):
        self.optimizers = {
            OptimizationMethod.GRID_SEARCH: GridSearchOptimizer,
            OptimizationMethod.RANDOM_SEARCH: RandomSearchOptimizer,
            OptimizationMethod.BAYESIAN: BayesianOptimizer,
            OptimizationMethod.GENETIC: GeneticOptimizer,
            OptimizationMethod.PARTICLE_SWARM: ParticleSwarmOptimizer
        }
        
        self.optimization_history = []
        self.best_configurations = {}
        
        logger.info("智能调优系统初始化完成")
    
    def optimize_model_parameters(self, model_class, parameter_spaces: List[ParameterSpace],
                                 X_train, y_train, X_val, y_val,
                                 method: OptimizationMethod = OptimizationMethod.BAYESIAN,
                                 max_evaluations: int = 50,
                                 objective: OptimizationObjective = OptimizationObjective.ACCURACY,
                                 **kwargs) -> OptimizationResult:
        """优化模型参数"""
        
        def objective_function(params: Dict[str, Any]) -> float:
            """目标函数"""
            try:
                # 创建模型实例
                model = model_class(**params)
                
                # 训练模型
                model.fit(X_train, y_train)
                
                # 预测
                y_pred = model.predict(X_val)
                
                # 计算评分
                if objective == OptimizationObjective.ACCURACY:
                    score = accuracy_score(y_val, y_pred)
                elif objective == OptimizationObjective.MSE:
                    score = -mean_squared_error(y_val, y_pred)  # 负号因为要最大化
                elif objective == OptimizationObjective.RMSE:
                    score = -np.sqrt(mean_squared_error(y_val, y_pred))
                elif objective == OptimizationObjective.F1_SCORE:
                    score = f1_score(y_val, y_pred, average='weighted')
                else:
                    score = accuracy_score(y_val, y_pred)  # 默认
                
                return score
                
            except Exception as e:
                logger.error(f"模型评估失败: {e}")
                return float('-inf')
        
        # 创建优化器
        optimizer_class = self.optimizers.get(method, RandomSearchOptimizer)
        optimizer = optimizer_class(objective_function, parameter_spaces)
        
        # 执行优化
        result = optimizer.optimize(max_evaluations, **kwargs)
        
        # 记录历史
        self.optimization_history.append({
            'timestamp': time.time(),
            'method': method.value,
            'objective': objective.value,
            'result': result
        })
        
        return result
    
    def auto_feature_selection(self, X: pd.DataFrame, y: pd.Series,
                              feature_selection_methods: List[str] = None,
                              max_features: int = None) -> Dict[str, Any]:
        """自动特征选择"""
        if not SKLEARN_AVAILABLE:
            logger.error("sklearn不可用，无法执行特征选择")
            return {'selected_features': list(X.columns)}
        
        from sklearn.feature_selection import SelectKBest, RFE, SelectFromModel
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.linear_model import LassoCV
        
        feature_selection_methods = feature_selection_methods or ['univariate', 'rfe', 'lasso']
        max_features = max_features or min(20, X.shape[1])
        
        results = {}
        
        try:
            # 单变量特征选择
            if 'univariate' in feature_selection_methods:
                selector = SelectKBest(k=max_features)
                X_selected = selector.fit_transform(X, y)
                selected_features = X.columns[selector.get_support()].tolist()
                results['univariate'] = {
                    'features': selected_features,
                    'scores': selector.scores_.tolist()
                }
            
            # 递归特征消除
            if 'rfe' in feature_selection_methods:
                estimator = RandomForestRegressor(n_estimators=10, random_state=42)
                selector = RFE(estimator, n_features_to_select=max_features)
                selector.fit(X, y)
                selected_features = X.columns[selector.support_].tolist()
                results['rfe'] = {
                    'features': selected_features,
                    'ranking': selector.ranking_.tolist()
                }
            
            # 基于模型的特征选择（Lasso）
            if 'lasso' in feature_selection_methods:
                lasso = LassoCV(cv=5, random_state=42)
                lasso.fit(X, y)
                selector = SelectFromModel(lasso, prefit=True, max_features=max_features)
                selected_features = X.columns[selector.get_support()].tolist()
                results['lasso'] = {
                    'features': selected_features,
                    'coefficients': lasso.coef_.tolist()
                }
            
            # 综合特征选择
            all_features = set()
            for method_result in results.values():
                all_features.update(method_result['features'])
            
            # 按出现频率排序
            feature_counts = {}
            for method_result in results.values():
                for feature in method_result['features']:
                    feature_counts[feature] = feature_counts.get(feature, 0) + 1
            
            sorted_features = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)
            final_features = [f for f, _ in sorted_features[:max_features]]
            
            results['final_selection'] = {
                'features': final_features,
                'feature_counts': feature_counts
            }
            
            logger.info(f"特征选择完成，从{X.shape[1]}个特征中选择了{len(final_features)}个")
            
        except Exception as e:
            logger.error(f"特征选择失败: {e}")
            results['final_selection'] = {'features': list(X.columns)}
        
        return results
    
    def optimize_ensemble_weights(self, predictions: Dict[str, np.ndarray], 
                                 y_true: np.ndarray,
                                 method: OptimizationMethod = OptimizationMethod.BAYESIAN) -> Dict[str, float]:
        """优化集成模型权重"""
        model_names = list(predictions.keys())
        n_models = len(model_names)
        
        if n_models < 2:
            return {model_names[0]: 1.0} if model_names else {}
        
        # 定义参数空间
        parameter_spaces = []
        for i, model_name in enumerate(model_names):
            parameter_spaces.append(ParameterSpace(
                name=f"weight_{model_name}",
                param_type='float',
                bounds=(0.0, 1.0)
            ))
        
        def objective_function(params: Dict[str, Any]) -> float:
            """集成目标函数"""
            try:
                # 标准化权重
                weights = np.array([params[f"weight_{name}"] for name in model_names])
                weights = weights / np.sum(weights)  # 归一化
                
                # 计算加权预测
                ensemble_pred = np.zeros_like(y_true, dtype=float)
                for i, model_name in enumerate(model_names):
                    ensemble_pred += weights[i] * predictions[model_name]
                
                # 计算评分（负MSE，因为要最大化）
                score = -mean_squared_error(y_true, ensemble_pred)
                return score
                
            except Exception as e:
                logger.error(f"集成权重评估失败: {e}")
                return float('-inf')
        
        # 执行优化
        optimizer_class = self.optimizers.get(method, RandomSearchOptimizer)
        optimizer = optimizer_class(objective_function, parameter_spaces)
        result = optimizer.optimize(max_evaluations=50)
        
        # 提取并标准化权重
        raw_weights = {}
        for model_name in model_names:
            raw_weights[model_name] = result.best_params[f"weight_{model_name}"]
        
        # 标准化权重
        total_weight = sum(raw_weights.values())
        normalized_weights = {name: weight / total_weight for name, weight in raw_weights.items()}
        
        logger.info(f"集成权重优化完成: {normalized_weights}")
        return normalized_weights
    
    def adaptive_learning_rate_schedule(self, model, X_train, y_train, X_val, y_val,
                                      initial_lr: float = 0.01, patience: int = 10) -> List[float]:
        """自适应学习率调度"""
        learning_rates = [initial_lr]
        best_score = float('-inf')
        patience_counter = 0
        
        current_lr = initial_lr
        
        for epoch in range(100):  # 最多100个epoch
            try:
                # 设置学习率（如果模型支持）
                if hasattr(model, 'set_params'):
                    model.set_params(learning_rate=current_lr)
                elif hasattr(model, 'learning_rate'):
                    model.learning_rate = current_lr
                
                # 训练一个epoch（如果模型支持）
                if hasattr(model, 'partial_fit'):
                    model.partial_fit(X_train, y_train)
                else:
                    model.fit(X_train, y_train)
                
                # 验证
                y_pred = model.predict(X_val)
                score = -mean_squared_error(y_val, y_pred)
                
                # 检查改进
                if score > best_score:
                    best_score = score
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                # 早停或调整学习率
                if patience_counter >= patience:
                    if current_lr > 1e-6:
                        current_lr *= 0.5  # 减半学习率
                        patience_counter = 0
                        logger.info(f"学习率调整为: {current_lr}")
                    else:
                        logger.info("学习率已达到最小值，停止训练")
                        break
                
                learning_rates.append(current_lr)
                
            except Exception as e:
                logger.error(f"自适应学习率调度失败: {e}")
                break
        
        return learning_rates
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """获取优化总结"""
        if not self.optimization_history:
            return {'message': '暂无优化历史'}
        
        summary = {
            'total_optimizations': len(self.optimization_history),
            'methods_used': {},
            'objectives_used': {},
            'best_results': {},
            'average_optimization_time': 0
        }
        
        total_time = 0
        
        for opt in self.optimization_history:
            # 统计方法使用
            method = opt['method']
            summary['methods_used'][method] = summary['methods_used'].get(method, 0) + 1
            
            # 统计目标使用
            objective = opt['objective']
            summary['objectives_used'][objective] = summary['objectives_used'].get(objective, 0) + 1
            
            # 记录最佳结果
            key = f"{method}_{objective}"
            if key not in summary['best_results'] or opt['result'].best_score > summary['best_results'][key]['score']:
                summary['best_results'][key] = {
                    'score': opt['result'].best_score,
                    'params': opt['result'].best_params,
                    'evaluations': opt['result'].total_evaluations
                }
            
            total_time += opt['result'].optimization_time
        
        summary['average_optimization_time'] = total_time / len(self.optimization_history)
        
        return summary


# 全局智能调优器实例
_intelligent_tuner = None

def get_intelligent_tuner() -> IntelligentTuner:
    """获取智能调优器实例（单例模式）"""
    global _intelligent_tuner
    if _intelligent_tuner is None:
        _intelligent_tuner = IntelligentTuner()
    return _intelligent_tuner


def main():
    """测试主函数"""
    # 创建智能调优器
    tuner = get_intelligent_tuner()
    
    # 生成模拟数据
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(1000, 10), columns=[f'feature_{i}' for i in range(10)])
    y = pd.Series(np.random.randn(1000))
    
    # 分割数据
    split_idx = 800
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    print("🧠 测试智能调优系统...")
    
    # 测试特征选择
    print("\n1. 自动特征选择测试...")
    try:
        feature_results = tuner.auto_feature_selection(X_train, y_train, max_features=5)
        selected_features = feature_results['final_selection']['features']
        print(f"   选择的特征: {selected_features}")
    except Exception as e:
        print(f"   特征选择测试失败: {e}")
    
    # 测试参数优化（使用简单的线性模型）
    print("\n2. 参数优化测试...")
    try:
        if SKLEARN_AVAILABLE:
            from sklearn.linear_model import Ridge
            
            # 定义参数空间
            param_spaces = [
                ParameterSpace('alpha', 'float', bounds=(0.01, 10.0), log_scale=True)
            ]
            
            # 执行优化
            result = tuner.optimize_model_parameters(
                Ridge, param_spaces, X_train, y_train, X_val, y_val,
                method=OptimizationMethod.RANDOM_SEARCH,
                max_evaluations=10,
                objective=OptimizationObjective.MSE
            )
            
            print(f"   最佳参数: {result.best_params}")
            print(f"   最佳分数: {result.best_score:.4f}")
            print(f"   优化时间: {result.optimization_time:.2f}秒")
        else:
            print("   跳过参数优化测试（sklearn不可用）")
    except Exception as e:
        print(f"   参数优化测试失败: {e}")
    
    # 测试集成权重优化
    print("\n3. 集成权重优化测试...")
    try:
        # 模拟多个模型的预测结果
        predictions = {
            'model_1': np.random.randn(len(y_val)),
            'model_2': np.random.randn(len(y_val)),
            'model_3': np.random.randn(len(y_val))
        }
        
        weights = tuner.optimize_ensemble_weights(
            predictions, y_val.values,
            method=OptimizationMethod.RANDOM_SEARCH
        )
        
        print(f"   优化的权重: {weights}")
    except Exception as e:
        print(f"   集成权重优化测试失败: {e}")
    
    # 输出优化总结
    print("\n4. 优化总结...")
    summary = tuner.get_optimization_summary()
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    
    print("\n✅ 智能调优系统测试完成！")


if __name__ == "__main__":
    main()
